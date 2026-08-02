#!/usr/bin/env python3
"""Audit and consolidate the fixed-split five-seed LSTM study."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

DATASETS = ("ev", "qx50", "blazer", "pacifica")
TRAINING_SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805)
SPLIT_SEED = 20260801
METRICS = ("mae", "rmse", "r2")
START_MARKER = "<!-- FIVE_SEED_RESULTS_START -->"
END_MARKER = "<!-- FIVE_SEED_RESULTS_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument(
        "--baseline-csv",
        default="artifacts/revision_full_data_head_to_head_results.csv",
    )
    parser.add_argument(
        "--summary-md", default="artifacts/revision_lstm_cpu_summary.md"
    )
    parser.add_argument(
        "--output-csv", default="artifacts/revision_lstm_five_seeds_results.csv"
    )
    parser.add_argument("--pr-snippet", default="artifacts/revision_lstm_five_seeds_pr.md")
    return parser.parse_args()


def load_payloads(root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*_emissions_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_artifact_json"] = str(path)
        payloads.append(payload)
    return payloads


def audit(payloads: list[dict[str, Any]]) -> None:
    expected = {(dataset, seed) for dataset in DATASETS for seed in TRAINING_SEEDS}
    observed = {(item["dataset"], int(item["training_seed"])) for item in payloads}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"Unexpected artifact set; missing={missing}, extra={extra}")
    if len(payloads) != len(expected):
        raise RuntimeError("Duplicate seed artifacts were found")

    manifests: dict[str, dict[str, Any]] = {}
    for item in payloads:
        dataset = item["dataset"]
        training_seed = int(item["training_seed"])
        if int(item["split_seed"]) != SPLIT_SEED:
            raise RuntimeError(f"{dataset}/{training_seed}: wrong split_seed")
        if int(item["split_manifest"]["seed"]) != SPLIT_SEED:
            raise RuntimeError(f"{dataset}/{training_seed}: split manifest seed changed")
        if item["used_window_counts"] != item["full_window_counts"]:
            raise RuntimeError(f"{dataset}/{training_seed}: not all windows were used")
        selection = item["checkpoint_selection"]
        if selection["policy"] != "minimum_validation_mse":
            raise RuntimeError(f"{dataset}/{training_seed}: wrong selection policy")
        if selection["test_set_used_for_selection"] is not False:
            raise RuntimeError(f"{dataset}/{training_seed}: test set used for selection")
        if selection["restored_before_primary_test_evaluation"] is not True:
            raise RuntimeError(f"{dataset}/{training_seed}: selected checkpoint not restored")
        design = item.get("seed_design", {})
        fixed = set(design.get("fixed_components", []))
        if "training_only_feature_scaler" not in fixed:
            raise RuntimeError(f"{dataset}/{training_seed}: training-only scaler not recorded")
        if design.get("split_held_fixed") is not True:
            raise RuntimeError(f"{dataset}/{training_seed}: split not marked fixed")
        manifest = item["split_manifest"]
        if dataset not in manifests:
            manifests[dataset] = manifest
        elif manifests[dataset] != manifest:
            raise RuntimeError(f"{dataset}: trip manifest differs across seeds")


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def selected_baselines(path: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("model_family") == "non_recurrent_baseline"
                and row.get("selected_by_validation", "").lower() == "true"
            ):
                selected[row["dataset"]] = {
                    "model": row["model"],
                    "mae": float(row["test_mae"]),
                    "rmse": float(row["test_rmse"]),
                    "r2": float(row["test_r2"]),
                }
    return selected


def write_csv(
    path: Path,
    payloads: list[dict[str, Any]],
    summaries: dict[str, dict[str, dict[str, float]]],
    baselines: dict[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "row_type",
        "dataset",
        "display_name",
        "split_seed",
        "training_seed",
        "best_epoch",
        "best_validation_mse",
        "test_mae",
        "test_rmse",
        "test_r2",
        "train_windows",
        "validation_windows",
        "test_windows",
        "statistic",
        "mae",
        "rmse",
        "r2",
        "baseline_model",
        "baseline_mae",
        "baseline_rmse",
        "baseline_r2",
        "mean_mae_minus_baseline",
        "all_windows_used",
        "split_fixed",
        "training_only_scaler",
        "checkpoint_policy",
        "test_set_used_for_selection",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(payloads, key=lambda x: (x["dataset"], x["training_seed"])):
            counts = item["used_window_counts"]
            writer.writerow(
                {
                    "row_type": "seed",
                    "dataset": item["dataset"],
                    "display_name": item["display_name"],
                    "split_seed": item["split_seed"],
                    "training_seed": item["training_seed"],
                    "best_epoch": item["checkpoint_selection"]["best_epoch"],
                    "best_validation_mse": item["checkpoint_selection"]["best_validation_mse"],
                    "test_mae": item["test"]["mae"],
                    "test_rmse": item["test"]["rmse"],
                    "test_r2": item["test"]["r2"],
                    "train_windows": counts["train"],
                    "validation_windows": counts["validation"],
                    "test_windows": counts["test"],
                    "all_windows_used": True,
                    "split_fixed": True,
                    "training_only_scaler": True,
                    "checkpoint_policy": "minimum_validation_mse",
                    "test_set_used_for_selection": False,
                }
            )
        display_names = {
            item["dataset"]: item["display_name"] for item in payloads
        }
        for dataset in DATASETS:
            baseline = baselines.get(dataset)
            for statistic in ("mean", "sample_sd", "median", "min", "max"):
                row = {
                    "row_type": "summary",
                    "dataset": dataset,
                    "display_name": display_names[dataset],
                    "split_seed": SPLIT_SEED,
                    "statistic": statistic,
                    "mae": summaries[dataset]["mae"][statistic],
                    "rmse": summaries[dataset]["rmse"][statistic],
                    "r2": summaries[dataset]["r2"][statistic],
                    "all_windows_used": True,
                    "split_fixed": True,
                    "training_only_scaler": True,
                    "checkpoint_policy": "minimum_validation_mse",
                    "test_set_used_for_selection": False,
                }
                if baseline:
                    row.update(
                        {
                            "baseline_model": baseline["model"],
                            "baseline_mae": baseline["mae"],
                            "baseline_rmse": baseline["rmse"],
                            "baseline_r2": baseline["r2"],
                        }
                    )
                    if statistic == "mean":
                        row["mean_mae_minus_baseline"] = (
                            summaries[dataset]["mae"]["mean"] - baseline["mae"]
                        )
                writer.writerow(row)


def fmt(value: float) -> str:
    return f"{value:.8g}"


def build_section(
    payloads: list[dict[str, Any]],
    summaries: dict[str, dict[str, dict[str, float]]],
    baselines: dict[str, dict[str, Any]],
) -> str:
    by_dataset = {
        dataset: sorted(
            [item for item in payloads if item["dataset"] == dataset],
            key=lambda item: item["training_seed"],
        )
        for dataset in DATASETS
    }
    lines = [
        START_MARKER,
        "## Five-seed fixed-split LSTM robustness study",
        "",
        "GitHub Actions `Revision LSTM Five Seeds` run `30742131505` completed all 20 jobs successfully. The complete-trip split is fixed with `split_seed=20260801`; training seeds `20260801`–`20260805` vary model initialization and minibatch order only.",
        "",
        "Audit checks passed for every artifact: identical trip manifests within each dataset, all available windows, feature scaler fitted only on training-trip rows, checkpoint selected by minimum validation MSE, selected checkpoint restored before test evaluation, and no test-set use in model selection.",
        "",
        "| Dataset | MAE mean ± sample SD | MAE median [min, max] | RMSE mean ± sample SD | R2 mean ± sample SD | Best epochs |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        items = by_dataset[dataset]
        s = summaries[dataset]
        epochs = ", ".join(str(item["checkpoint_selection"]["best_epoch"]) for item in items)
        lines.append(
            f"| {items[0]['display_name']} | {fmt(s['mae']['mean'])} ± {fmt(s['mae']['sample_sd'])} | "
            f"{fmt(s['mae']['median'])} [{fmt(s['mae']['min'])}, {fmt(s['mae']['max'])}] | "
            f"{fmt(s['rmse']['mean'])} ± {fmt(s['rmse']['sample_sd'])} | "
            f"{fmt(s['r2']['mean'])} ± {fmt(s['r2']['sample_sd'])} | {epochs} |"
        )
    lines.extend(
        [
            "",
            "### Comparison with validation-selected non-recurrent baselines",
            "",
            "The canonical baseline CSV contains exact full-data selected baselines for BMW i3 and QX50. Blazer and Pacifica remain documented in the broader benchmark summary but are not represented as selected non-recurrent rows in that CSV.",
            "",
            "| Dataset | Five-seed LSTM mean MAE | Baseline | Baseline MAE | Mean MAE difference | Lower MAE |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for dataset in ("ev", "qx50"):
        item = by_dataset[dataset][0]
        mean_mae = summaries[dataset]["mae"]["mean"]
        baseline = baselines[dataset]
        difference = mean_mae - baseline["mae"]
        lower = "LSTM mean" if difference < 0 else baseline["model"]
        lines.append(
            f"| {item['display_name']} | {fmt(mean_mae)} | {baseline['model']} | "
            f"{fmt(baseline['mae'])} | {fmt(difference)} | {lower} |"
        )
    lines.extend(
        [
            "",
            "The five-seed analysis is the canonical robustness result. It should replace single-seed language when discussing LSTM performance variability. Comparisons remain descriptive because the five training seeds do not constitute independent test datasets and the vehicle comparison remains conditioned on observed covariates rather than causally matched operating conditions.",
            END_MARKER,
        ]
    )
    return "\n".join(lines)


def replace_section(text: str, section: str) -> str:
    if START_MARKER in text and END_MARKER in text:
        prefix = text.split(START_MARKER, 1)[0].rstrip()
        suffix = text.split(END_MARKER, 1)[1].lstrip()
        return f"{prefix}\n\n{section}\n\n{suffix}".rstrip() + "\n"
    return text.rstrip() + "\n\n" + section + "\n"


def main() -> None:
    args = parse_args()
    payloads = load_payloads(Path(args.artifacts_dir))
    audit(payloads)

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in DATASETS:
        items = [item for item in payloads if item["dataset"] == dataset]
        summaries[dataset] = {
            metric: summarize([float(item["test"][metric]) for item in items])
            for metric in METRICS
        }

    baselines = selected_baselines(Path(args.baseline_csv))
    for required in ("ev", "qx50"):
        if required not in baselines:
            raise RuntimeError(f"Missing canonical selected baseline for {required}")

    write_csv(Path(args.output_csv), payloads, summaries, baselines)
    section = build_section(payloads, summaries, baselines)

    summary_path = Path(args.summary_md)
    summary_path.write_text(
        replace_section(summary_path.read_text(encoding="utf-8"), section),
        encoding="utf-8",
    )
    Path(args.pr_snippet).write_text(section + "\n", encoding="utf-8")

    print(json.dumps({
        "artifacts_audited": len(payloads),
        "datasets": DATASETS,
        "training_seeds": TRAINING_SEEDS,
        "split_seed": SPLIT_SEED,
        "summary": summaries,
        "selected_baselines": baselines,
    }, indent=2))


if __name__ == "__main__":
    main()
