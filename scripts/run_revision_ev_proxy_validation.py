#!/usr/bin/env python3
"""Recompute EV proxy validation under the corrected complete-trip protocol.

The feature checkpoint predicts raw motor torque and throttle from shared
context. The emissions checkpoint predicts CO2 from ten-step actuation windows.
To compose the two models without measured actuation, predictions are aligned
within each held-out trip. Because both models use a ten-step lookback, the
common direct-versus-proxy evaluation begins at timestep 2*WINDOW_SIZE.

Model selection is not performed here. Both checkpoints were selected by their
validation losses in the audited full-data LSTM run, and the test trips are used
only for the final within-domain proxy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    SHARED_CONTEXT,
    WINDOW_SIZE,
    complete_trip_split,
    fit_feature_scaler,
    load_dataset,
    regression_metrics,
    save_json,
    summarize_trip_metric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emissions-checkpoint", required=True)
    parser.add_argument("--feature-checkpoint", required=True)
    parser.add_argument("--emissions-result", required=True)
    parser.add_argument("--feature-result", required=True)
    parser.add_argument("--output-dir", default="artifacts/revision_ev_proxy_validation")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_model(checkpoint: dict[str, Any]) -> MultipleLayerLSTM:
    config = checkpoint["model"]
    model = MultipleLayerLSTM(
        input_size=len(checkpoint["feature_columns"]),
        hidden_dim=int(config["hidden_dim"]),
        output_size=len(checkpoint["target_columns"]),
        num_blocks=int(config["num_blocks"]),
    ).cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def batched_predict(model: torch.nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size])
            outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def sliding_windows(x: np.ndarray, width: int) -> np.ndarray:
    if len(x) <= width:
        return np.empty((0, width, x.shape[1]), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(x, width, axis=0)
    return windows[:-1].transpose(0, 2, 1).copy().astype(np.float32)


def context_windows(x: np.ndarray, width: int) -> np.ndarray:
    """Inputs x[t-width:t] used to predict a target at original index t."""
    return sliding_windows(x, width)


def per_trip_payload(
    trip_id: str,
    observed: np.ndarray,
    direct: np.ndarray,
    proxy: np.ndarray,
) -> dict[str, Any]:
    direct_metrics = regression_metrics(observed, direct)
    proxy_metrics = regression_metrics(observed, proxy)
    return {
        "trip_id": trip_id,
        "n_aligned_windows": int(len(observed)),
        "direct": direct_metrics,
        "proxy": proxy_metrics,
        "proxy_minus_direct_mae": float(proxy_metrics["mae"] - direct_metrics["mae"]),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    emissions_checkpoint = load_checkpoint(args.emissions_checkpoint)
    feature_checkpoint = load_checkpoint(args.feature_checkpoint)
    emissions_result = json.loads(Path(args.emissions_result).read_text(encoding="utf-8"))
    feature_result = json.loads(Path(args.feature_result).read_text(encoding="utf-8"))

    if emissions_checkpoint["task"] != "emissions_model":
        raise RuntimeError("Wrong emissions checkpoint task")
    if feature_checkpoint["task"] != "ev_feature_model":
        raise RuntimeError("Wrong feature checkpoint task")
    if emissions_checkpoint["seed"] != SEED or feature_checkpoint["seed"] != SEED:
        raise RuntimeError("Expected audited seed 20260801 checkpoints")
    if emissions_checkpoint["checkpoint_selection"]["test_set_used_for_selection"]:
        raise RuntimeError("Emissions checkpoint used test selection")
    if feature_checkpoint["checkpoint_selection"]["test_set_used_for_selection"]:
        raise RuntimeError("Feature checkpoint used test selection")
    if list(emissions_checkpoint["feature_columns"]) != list(ACTUATION):
        raise RuntimeError("Unexpected emissions features")
    if list(feature_checkpoint["feature_columns"]) != list(SHARED_CONTEXT):
        raise RuntimeError("Unexpected feature-model context")
    if list(feature_checkpoint["target_columns"]) != [
        "Motor Torque [Nm]",
        "Throttle [%]",
    ]:
        raise RuntimeError("Unexpected feature-model targets")

    spec = DATASETS["ev"]
    frame = load_dataset(spec)
    split = complete_trip_split(frame, seed=SEED)
    if emissions_result["split_manifest"] != split.as_dict():
        raise RuntimeError("Emissions result manifest mismatch")
    if feature_result["split_manifest"] != split.as_dict():
        raise RuntimeError("Feature result manifest mismatch")
    if emissions_result["used_window_counts"] != emissions_result["full_window_counts"]:
        raise RuntimeError("Emissions checkpoint is not full-data")
    if feature_result["used_window_counts"] != feature_result["full_window_counts"]:
        raise RuntimeError("Feature checkpoint is not full-data")

    feature_scaler = fit_feature_scaler(frame, split.train_trip_ids, SHARED_CONTEXT)
    emissions_scaler = fit_feature_scaler(frame, split.train_trip_ids, ACTUATION)
    feature_model = make_model(feature_checkpoint)
    emissions_model = make_model(emissions_checkpoint)

    all_observed: list[np.ndarray] = []
    all_direct: list[np.ndarray] = []
    all_proxy: list[np.ndarray] = []
    all_trip_ids: list[np.ndarray] = []
    trip_results: list[dict[str, Any]] = []
    trip_series: dict[str, dict[str, np.ndarray]] = {}

    for trip_id in split.test_trip_ids:
        trip = frame.loc[frame["Trip"] == str(trip_id)].reset_index(drop=True)
        if len(trip) <= 2 * WINDOW_SIZE:
            continue

        context_raw = trip[SHARED_CONTEXT].to_numpy(dtype=np.float64)
        context_scaled = feature_scaler.transform(context_raw).astype(np.float32)
        feature_x = context_windows(context_scaled, WINDOW_SIZE)
        feature_predictions = batched_predict(feature_model, feature_x, args.batch_size)
        # feature_predictions[k] is aligned to original trip index WINDOW_SIZE+k.

        aligned_velocity = trip["Velocity [km/h]"].to_numpy(dtype=np.float64)[WINDOW_SIZE:]
        measured_throttle = trip["Throttle [%]"].to_numpy(dtype=np.float64)[WINDOW_SIZE:]
        measured_torque = trip["Motor Torque [Nm]"].to_numpy(dtype=np.float64)[WINDOW_SIZE:]

        direct_raw = np.column_stack(
            [aligned_velocity, measured_throttle, measured_torque]
        )
        proxy_raw = np.column_stack(
            [aligned_velocity, feature_predictions[:, 1], feature_predictions[:, 0]]
        )
        direct_scaled = emissions_scaler.transform(direct_raw).astype(np.float32)
        proxy_scaled = emissions_scaler.transform(proxy_raw).astype(np.float32)

        direct_x = sliding_windows(direct_scaled, WINDOW_SIZE)
        proxy_x = sliding_windows(proxy_scaled, WINDOW_SIZE)
        observed = trip["CO2 Emissions"].to_numpy(dtype=np.float32)[2 * WINDOW_SIZE :]
        if len(direct_x) != len(observed) or len(proxy_x) != len(observed):
            raise RuntimeError(f"Alignment mismatch for trip {trip_id}")

        direct_prediction = batched_predict(
            emissions_model, direct_x, args.batch_size
        ).reshape(-1)
        proxy_prediction = batched_predict(
            emissions_model, proxy_x, args.batch_size
        ).reshape(-1)

        payload = per_trip_payload(
            str(trip_id), observed, direct_prediction, proxy_prediction
        )
        trip_results.append(payload)
        trip_series[str(trip_id)] = {
            "observed": observed,
            "direct": direct_prediction,
            "proxy": proxy_prediction,
        }
        all_observed.append(observed)
        all_direct.append(direct_prediction)
        all_proxy.append(proxy_prediction)
        all_trip_ids.append(np.repeat(str(trip_id), len(observed)))

    observed = np.concatenate(all_observed)
    direct = np.concatenate(all_direct)
    proxy = np.concatenate(all_proxy)
    trip_ids = np.concatenate(all_trip_ids)

    direct_trip_mae = [float(item["direct"]["mae"]) for item in trip_results]
    proxy_trip_mae = [float(item["proxy"]["mae"]) for item in trip_results]
    difference_trip_mae = [
        float(item["proxy_minus_direct_mae"]) for item in trip_results
    ]

    counts = sorted(
        ((trip, len(series["observed"])) for trip, series in trip_series.items()),
        key=lambda item: (item[1], item[0]),
    )
    representative_trip = counts[(len(counts) - 1) // 2][0]
    representative = trip_series[representative_trip]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ev_proxy_trip_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trip_id",
                "n_aligned_windows",
                "direct_mae",
                "direct_rmse",
                "direct_r2",
                "proxy_mae",
                "proxy_rmse",
                "proxy_r2",
                "proxy_minus_direct_mae",
            ],
        )
        writer.writeheader()
        for item in trip_results:
            writer.writerow(
                {
                    "trip_id": item["trip_id"],
                    "n_aligned_windows": item["n_aligned_windows"],
                    "direct_mae": item["direct"]["mae"],
                    "direct_rmse": item["direct"]["rmse"],
                    "direct_r2": item["direct"]["r2"],
                    "proxy_mae": item["proxy"]["mae"],
                    "proxy_rmse": item["proxy"]["rmse"],
                    "proxy_r2": item["proxy"]["r2"],
                    "proxy_minus_direct_mae": item["proxy_minus_direct_mae"],
                }
            )

    fig, axis = plt.subplots(figsize=(10, 3.7), constrained_layout=True)
    length = len(representative["observed"])
    display_count = min(length, 5000)
    indices = np.unique(np.linspace(0, length - 1, display_count).astype(int))
    axis.plot(indices, representative["observed"][indices], color="black", linewidth=0.8, label="Observed")
    axis.plot(indices, representative["direct"][indices], linewidth=0.8, label="Measured-actuation input")
    axis.plot(indices, representative["proxy"][indices], linewidth=0.8, label="Predicted-actuation proxy")
    axis.set_xlabel("Aligned window index within trip")
    axis.set_ylabel("CO2 emissions (g/s)")
    axis.set_title(f"Corrected EV proxy validation - median-length test trip {representative_trip}")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    fig.savefig(output_dir / "figure_ev_proxy_validation_corrected.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure_ev_proxy_validation_corrected.pdf", bbox_inches="tight")
    plt.close(fig)

    result: dict[str, Any] = {
        "status": "completed_corrected_ev_proxy_validation",
        "dataset": "ev",
        "display_name": spec.display_name,
        "split_seed": SEED,
        "training_seed": SEED,
        "window_size": WINDOW_SIZE,
        "alignment": {
            "feature_prediction_target_offset": WINDOW_SIZE,
            "emissions_prediction_target_offset": 2 * WINDOW_SIZE,
            "reason": "Both feature and emissions models require ten-step histories; direct and proxy predictions are compared on the identical aligned subset.",
        },
        "split_manifest": split.as_dict(),
        "test_set_used_for_model_selection": False,
        "validation_selected_checkpoints": True,
        "all_eligible_aligned_test_windows_used": True,
        "global": {
            "n_windows": int(len(observed)),
            "direct": regression_metrics(observed, direct),
            "proxy": regression_metrics(observed, proxy),
            "proxy_minus_direct_mae": float(
                regression_metrics(observed, proxy)["mae"]
                - regression_metrics(observed, direct)["mae"]
            ),
        },
        "trip_level": {
            "direct_mae": summarize_trip_metric(direct_trip_mae, SEED),
            "proxy_mae": summarize_trip_metric(proxy_trip_mae, SEED),
            "proxy_minus_direct_mae": summarize_trip_metric(
                difference_trip_mae, SEED
            ),
        },
        "per_trip": trip_results,
        "representative_trip": {
            "selection_rule": "median aligned test-trip length, independent of targets and predictions",
            "trip_id": representative_trip,
            "n_aligned_windows": int(len(representative["observed"])),
        },
        "interpretation_guardrail": (
            "This is an in-domain EV component-composition check. It does not validate "
            "cross-powertrain operating-condition equivalence or a causal EV-ICEV comparison."
        ),
    }
    save_json(output_dir / "ev_proxy_validation.json", result)

    direct_summary = result["trip_level"]["direct_mae"]
    proxy_summary = result["trip_level"]["proxy_mae"]
    diff_summary = result["trip_level"]["proxy_minus_direct_mae"]
    report = f"""# Corrected EV proxy validation

- Fixed complete-trip split seed: `{SEED}`
- Validation-selected full-data checkpoints: emissions epoch {emissions_checkpoint['checkpoint_selection']['best_epoch']}, feature epoch {feature_checkpoint['checkpoint_selection']['best_epoch']}
- Common aligned held-out windows: {len(observed):,}
- Test set used for model selection: `false`

| Quantity | Mean across trips | Median | Sample SD | 95% bootstrap CI for mean |
|---|---:|---:|---:|---:|
| Direct MAE | {direct_summary['mean']:.8g} | {direct_summary['median']:.8g} | {direct_summary['sample_std']:.8g} | [{direct_summary['bootstrap_95pct_mean_ci'][0]:.8g}, {direct_summary['bootstrap_95pct_mean_ci'][1]:.8g}] |
| Proxy MAE | {proxy_summary['mean']:.8g} | {proxy_summary['median']:.8g} | {proxy_summary['sample_std']:.8g} | [{proxy_summary['bootstrap_95pct_mean_ci'][0]:.8g}, {proxy_summary['bootstrap_95pct_mean_ci'][1]:.8g}] |
| Proxy - direct MAE | {diff_summary['mean']:.8g} | {diff_summary['median']:.8g} | {diff_summary['sample_std']:.8g} | [{diff_summary['bootstrap_95pct_mean_ci'][0]:.8g}, {diff_summary['bootstrap_95pct_mean_ci'][1]:.8g}] |

This is an in-domain component-composition check and not a cross-powertrain counterfactual validation.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
