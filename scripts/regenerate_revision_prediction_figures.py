#!/usr/bin/env python3
"""Regenerate prediction figures from the audited five-seed LSTM checkpoints.

The figures are derived only from the corrected complete-trip protocol. For each
vehicle, all five validation-selected checkpoints are evaluated on the fixed
held-out test trips. The time-series panel uses a representative trip selected
*without predictions*: the median-length test trip, with trip ID as a stable
tie-breaker. The displayed mean and standard-deviation band summarize variation
across training seeds and are explicitly diagnostic, not a newly selected
ensemble result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.LSTM import MultipleLayerLSTM
from src.revision_protocol import (
    ACTUATION,
    DATASETS,
    SEED,
    WINDOW_SIZE,
    build_windows,
    complete_trip_split,
    fit_feature_scaler,
    load_dataset,
    regression_metrics,
    save_json,
    window_count_by_trip,
)

DATASET_ORDER = ("ev", "qx50", "blazer", "pacifica")
TRAINING_SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805)
SPLIT_SEED = 20260801
CHECKPOINT_PATTERN = re.compile(r"(?P<dataset>[^/]+)_emissions_seed(?P<seed>\d+)_best\.pt$")
RESULT_PATTERN = re.compile(r"(?P<dataset>[^/]+)_emissions_seed(?P<seed>\d+)\.json$")

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument(
        "--output-dir", default="artifacts/revision_prediction_figures"
    )
    parser.add_argument("--full-predictions-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def discover(root: Path) -> tuple[dict[tuple[str, int], Path], dict[tuple[str, int], Path]]:
    checkpoints: dict[tuple[str, int], Path] = {}
    results: dict[tuple[str, int], Path] = {}
    for path in root.rglob("*_emissions_seed*_best.pt"):
        match = CHECKPOINT_PATTERN.search(path.name)
        if match:
            checkpoints[(match.group("dataset"), int(match.group("seed")))] = path
    for path in root.rglob("*_emissions_seed*.json"):
        match = RESULT_PATTERN.search(path.name)
        if match:
            results[(match.group("dataset"), int(match.group("seed")))] = path

    expected = {(dataset, seed) for dataset in DATASET_ORDER for seed in TRAINING_SEEDS}
    if set(checkpoints) != expected:
        raise RuntimeError(
            f"Checkpoint set mismatch: missing={sorted(expected-set(checkpoints))}, "
            f"extra={sorted(set(checkpoints)-expected)}"
        )
    if set(results) != expected:
        raise RuntimeError(
            f"Result JSON set mismatch: missing={sorted(expected-set(results))}, "
            f"extra={sorted(set(results)-expected)}"
        )
    return checkpoints, results


def predict(model: torch.nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size])
            outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs, axis=0).reshape(-1)


def representative_trip(test_trips: np.ndarray) -> tuple[str, dict[str, int]]:
    counts = {
        str(trip): int(np.sum(test_trips == trip))
        for trip in np.unique(test_trips)
    }
    ranked = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    chosen = ranked[(len(ranked) - 1) // 2][0]
    return chosen, counts


def write_trip_csv(
    path: Path,
    trip_id: str,
    observed: np.ndarray,
    predictions: np.ndarray,
    ensemble_mean: np.ndarray,
    ensemble_sd: np.ndarray,
) -> None:
    mask_length = len(observed)
    fieldnames = [
        "trip_id",
        "window_index",
        "observed_co2",
        *[f"prediction_seed_{seed}" for seed in TRAINING_SEEDS],
        "five_seed_prediction_mean",
        "five_seed_prediction_sample_sd",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(mask_length):
            row: dict[str, Any] = {
                "trip_id": trip_id,
                "window_index": index,
                "observed_co2": float(observed[index]),
                "five_seed_prediction_mean": float(ensemble_mean[index]),
                "five_seed_prediction_sample_sd": float(ensemble_sd[index]),
            }
            for seed_index, seed in enumerate(TRAINING_SEEDS):
                row[f"prediction_seed_{seed}"] = float(predictions[seed_index, index])
            writer.writerow(row)


def dataset_predictions(
    dataset: str,
    checkpoints: dict[tuple[str, int], Path],
    results: dict[tuple[str, int], Path],
    output_dir: Path,
    full_predictions_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    spec = DATASETS[dataset]
    frame = load_dataset(spec)
    split = complete_trip_split(frame, seed=SPLIT_SEED)
    scaler = fit_feature_scaler(frame, split.train_trip_ids, ACTUATION)
    x_test, y_test, test_trips = build_windows(
        frame,
        split.test_trip_ids,
        ACTUATION,
        ["CO2 Emissions"],
        scaler,
        WINDOW_SIZE,
    )
    observed = y_test.reshape(-1).astype(np.float64)

    seed_predictions: list[np.ndarray] = []
    seed_metrics: list[dict[str, Any]] = []
    histories: list[list[dict[str, Any]]] = []
    checkpoint_hashes: dict[str, str] = {}

    for training_seed in TRAINING_SEEDS:
        checkpoint_path = checkpoints[(dataset, training_seed)]
        result_path = results[(dataset, training_seed)]
        checkpoint = load_torch_payload(checkpoint_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))

        if int(checkpoint["split_seed"]) != SPLIT_SEED:
            raise RuntimeError(f"{dataset}/{training_seed}: wrong checkpoint split seed")
        if checkpoint["split_manifest"] != split.as_dict():
            raise RuntimeError(f"{dataset}/{training_seed}: checkpoint manifest differs")
        if result["split_manifest"] != split.as_dict():
            raise RuntimeError(f"{dataset}/{training_seed}: result manifest differs")
        if result["used_window_counts"] != result["full_window_counts"]:
            raise RuntimeError(f"{dataset}/{training_seed}: not a full-data result")
        if result["checkpoint_selection"]["test_set_used_for_selection"] is not False:
            raise RuntimeError(f"{dataset}/{training_seed}: test used for selection")
        if list(checkpoint["feature_columns"]) != list(ACTUATION):
            raise RuntimeError(f"{dataset}/{training_seed}: feature set differs")

        model_config = checkpoint["model"]
        model = MultipleLayerLSTM(
            input_size=len(checkpoint["feature_columns"]),
            hidden_dim=int(model_config["hidden_dim"]),
            output_size=len(checkpoint["target_columns"]),
            num_blocks=int(model_config["num_blocks"]),
        ).cpu()
        model.load_state_dict(checkpoint["model_state_dict"])
        prediction = predict(model, x_test, batch_size)
        metrics = regression_metrics(observed, prediction)

        for metric in ("mae", "rmse", "r2"):
            reference = float(result["test"][metric])
            if not np.isclose(metrics[metric], reference, rtol=2e-6, atol=2e-7):
                raise RuntimeError(
                    f"{dataset}/{training_seed}: regenerated {metric}={metrics[metric]} "
                    f"does not match artifact {reference}"
                )

        seed_predictions.append(prediction.astype(np.float64))
        seed_metrics.append(
            {
                "training_seed": training_seed,
                "best_epoch": int(result["checkpoint_selection"]["best_epoch"]),
                **metrics,
            }
        )
        histories.append(result["history"])
        checkpoint_hashes[str(training_seed)] = sha256(checkpoint_path)

    predictions = np.stack(seed_predictions, axis=0)
    ensemble_mean = np.mean(predictions, axis=0)
    ensemble_sd = np.std(predictions, axis=0, ddof=1)
    diagnostic_metrics = regression_metrics(observed, ensemble_mean)

    chosen_trip, trip_counts = representative_trip(test_trips)
    chosen_mask = test_trips == chosen_trip
    trip_csv = output_dir / f"{dataset}_median_length_test_trip_predictions.csv"
    write_trip_csv(
        trip_csv,
        chosen_trip,
        observed[chosen_mask],
        predictions[:, chosen_mask],
        ensemble_mean[chosen_mask],
        ensemble_sd[chosen_mask],
    )

    prediction_archive = full_predictions_dir / f"{dataset}_all_test_predictions.npz"
    np.savez_compressed(
        prediction_archive,
        observed=observed.astype(np.float32),
        predictions=predictions.astype(np.float32),
        five_seed_mean=ensemble_mean.astype(np.float32),
        five_seed_sample_sd=ensemble_sd.astype(np.float32),
        trip_ids=test_trips.astype(str),
        training_seeds=np.asarray(TRAINING_SEEDS, dtype=np.int64),
    )

    train_history = np.asarray(
        [[float(row["train_mse"]) for row in history] for history in histories],
        dtype=np.float64,
    )
    validation_history = np.asarray(
        [[float(row["validation_mse"]) for row in history] for history in histories],
        dtype=np.float64,
    )

    return {
        "dataset": dataset,
        "display_name": spec.display_name,
        "split_manifest": split.as_dict(),
        "window_counts_by_trip": window_count_by_trip(
            frame, split.test_trip_ids, WINDOW_SIZE
        ),
        "test_window_count": int(len(observed)),
        "observed": observed,
        "test_trips": test_trips,
        "predictions": predictions,
        "five_seed_mean": ensemble_mean,
        "five_seed_sample_sd": ensemble_sd,
        "diagnostic_five_seed_mean_metrics": diagnostic_metrics,
        "seed_metrics": seed_metrics,
        "representative_trip": {
            "selection_rule": (
                "median test-trip window count, ranked by (window_count, trip_id); "
                "selection is independent of observed targets and predictions"
            ),
            "trip_id": chosen_trip,
            "window_count": int(np.sum(chosen_mask)),
            "all_test_trip_window_counts": trip_counts,
            "csv": trip_csv.name,
        },
        "train_history": train_history,
        "validation_history": validation_history,
        "checkpoint_sha256": checkpoint_hashes,
        "full_prediction_archive": prediction_archive.name,
        "full_prediction_archive_sha256": sha256(prediction_archive),
    }


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_prediction_panels(data: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, item in zip(axes.flat, data):
        trip = item["representative_trip"]["trip_id"]
        mask = item["test_trips"] == trip
        observed = item["observed"][mask]
        mean = item["five_seed_mean"][mask]
        sd = item["five_seed_sample_sd"][mask]
        count = len(observed)
        display_count = min(count, 5000)
        indices = np.unique(np.linspace(0, count - 1, display_count).astype(int))

        axis.plot(indices, observed[indices], color="black", linewidth=0.9, label="Observed")
        axis.plot(
            indices,
            mean[indices],
            color="#1f77b4",
            linewidth=0.9,
            label="Five-seed prediction mean",
        )
        axis.fill_between(
            indices,
            mean[indices] - sd[indices],
            mean[indices] + sd[indices],
            color="#1f77b4",
            alpha=0.22,
            linewidth=0,
            label="±1 sample SD across seeds",
        )
        axis.set_title(
            f"{item['display_name']} — median-length test trip ({count:,} windows)"
        )
        axis.set_xlabel("Window index within trip")
        axis.set_ylabel("CO₂ emissions")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Leakage-free held-out-trip predictions: five training seeds",
        fontsize=12,
    )
    return save_figure(fig, output_dir / "figure_lstm_predictions_median_test_trips")


def plot_parity_panels(data: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, item in zip(axes.flat, data):
        observed = item["observed"]
        predicted = item["five_seed_mean"]
        combined_min = float(min(np.min(observed), np.min(predicted)))
        combined_max = float(max(np.max(observed), np.max(predicted)))
        margin = 0.02 * max(combined_max - combined_min, 1e-9)
        low, high = combined_min - margin, combined_max + margin
        hexbin = axis.hexbin(
            observed,
            predicted,
            gridsize=75,
            bins="log",
            mincnt=1,
            cmap="viridis",
            linewidths=0,
        )
        axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        metrics = item["diagnostic_five_seed_mean_metrics"]
        axis.text(
            0.03,
            0.97,
            f"Diagnostic mean prediction\nMAE={metrics['mae']:.4g}\nR²={metrics['r2']:.4f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(item["display_name"])
        axis.set_xlabel("Observed CO₂ emissions")
        axis.set_ylabel("Five-seed prediction mean")
        fig.colorbar(hexbin, ax=axis, label="log₁₀ bin count")
    fig.suptitle(
        "Observed versus predicted on all held-out test-trip windows\n"
        "(five-seed mean is a diagnostic visualization, not the canonical scored model)",
        fontsize=11,
    )
    return save_figure(fig, output_dir / "figure_lstm_parity_full_test_sets")


def plot_training_panels(data: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, item in zip(axes.flat, data):
        train = item["train_history"]
        validation = item["validation_history"]
        epochs = np.arange(1, train.shape[1] + 1)
        train_mean = np.mean(train, axis=0)
        train_sd = np.std(train, axis=0, ddof=1)
        validation_mean = np.mean(validation, axis=0)
        validation_sd = np.std(validation, axis=0, ddof=1)

        axis.plot(epochs, train_mean, color="#2ca02c", label="Train MSE mean")
        axis.fill_between(
            epochs,
            np.maximum(train_mean - train_sd, np.finfo(float).tiny),
            train_mean + train_sd,
            color="#2ca02c",
            alpha=0.2,
            linewidth=0,
        )
        axis.plot(epochs, validation_mean, color="#d62728", label="Validation MSE mean")
        axis.fill_between(
            epochs,
            np.maximum(validation_mean - validation_sd, np.finfo(float).tiny),
            validation_mean + validation_sd,
            color="#d62728",
            alpha=0.2,
            linewidth=0,
        )
        axis.set_yscale("log")
        axis.set_title(item["display_name"])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE (log scale)")
        axis.set_xticks(epochs[::2])
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Training and validation trajectories across five seeds", fontsize=12)
    return save_figure(fig, output_dir / "figure_lstm_training_validation_five_seeds")


def public_manifest(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key
        not in {
            "observed",
            "test_trips",
            "predictions",
            "five_seed_mean",
            "five_seed_sample_sd",
            "train_history",
            "validation_history",
        }
    }


def markdown_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Regenerated corrected-protocol prediction figures",
        "",
        "These figures use all five validation-selected LSTM checkpoints from the",
        "fixed-split study. The split is by complete trips (`split_seed=20260801`),",
        "the feature scaler is fitted only on training-trip rows, and every displayed",
        "prediction is from held-out test trips.",
        "",
        "The mean and standard-deviation band across seeds are diagnostic visualizations.",
        "They do not replace the canonical per-seed metrics reported in the manuscript.",
        "Representative trips are selected by median test-trip length, independently of",
        "observed emissions and model predictions.",
        "",
        "| Dataset | Representative trip | Windows | Diagnostic five-seed-mean MAE | R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for item in manifest["datasets"]:
        metrics = item["diagnostic_five_seed_mean_metrics"]
        representative = item["representative_trip"]
        lines.append(
            f"| {item['display_name']} | `{representative['trip_id']}` | "
            f"{representative['window_count']:,} | {metrics['mae']:.8g} | "
            f"{metrics['r2']:.8g} |"
        )
    lines.extend(
        [
            "",
            "Generated assets:",
            "",
            "- `figure_lstm_predictions_median_test_trips.{png,pdf}`",
            "- `figure_lstm_parity_full_test_sets.{png,pdf}`",
            "- `figure_lstm_training_validation_five_seeds.{png,pdf}`",
            "- one representative-trip prediction CSV per dataset",
            "- full compressed prediction arrays retained in the workflow artifact",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    root = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    full_predictions_dir = Path(args.full_predictions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_predictions_dir.mkdir(parents=True, exist_ok=True)

    checkpoints, results = discover(root)
    datasets = [
        dataset_predictions(
            dataset,
            checkpoints,
            results,
            output_dir,
            full_predictions_dir,
            args.batch_size,
        )
        for dataset in DATASET_ORDER
    ]

    generated_files: list[Path] = []
    generated_files.extend(plot_prediction_panels(datasets, output_dir))
    generated_files.extend(plot_parity_panels(datasets, output_dir))
    generated_files.extend(plot_training_panels(datasets, output_dir))
    generated_files.extend(
        sorted(output_dir.glob("*_median_length_test_trip_predictions.csv"))
    )

    manifest: dict[str, Any] = {
        "status": "completed_regenerated_prediction_figures",
        "source_lstm_run_id": 30742131505,
        "split_seed": SPLIT_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "feature_columns": list(ACTUATION),
        "target": "CO2 Emissions",
        "window_size": WINDOW_SIZE,
        "protocol_checks": {
            "complete_trip_split": True,
            "training_only_feature_scaler": True,
            "all_test_windows_used": True,
            "validation_selected_checkpoints": True,
            "test_set_used_for_model_selection": False,
            "representative_trip_selected_without_targets_or_predictions": True,
        },
        "visualization_semantics": (
            "Five-seed prediction means and sample-standard-deviation bands are "
            "diagnostic summaries only; canonical performance remains the distribution "
            "of per-seed metrics."
        ),
        "datasets": [public_manifest(item) for item in datasets],
        "generated_files": {},
    }
    for path in generated_files:
        manifest["generated_files"][path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    manifest_path = output_dir / "prediction_figure_manifest.json"
    save_json(manifest_path, manifest)
    report_path = output_dir / "README.md"
    report_path.write_text(markdown_report(manifest), encoding="utf-8")
    print(markdown_report(manifest))


if __name__ == "__main__":
    main()
