#!/usr/bin/env python3
"""Run an exact full-data recurrent-versus-non-recurrent comparison for the EV feature model.

The task maps shared observed context to motor torque and throttle. All models
use the corrected complete-trip split, a scaler fitted only on training-trip
rows, length-10 windows built independently inside each trip, and every
available train/validation/test window. The non-recurrent family is selected
by the mean of the two validation MAEs; the test set is evaluated only after
selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_revision_experiments import flatten_windows, make_model
from src.revision_protocol import (
    DATASETS,
    SEED,
    SHARED_CONTEXT,
    WINDOW_SIZE,
    build_windows,
    complete_trip_split,
    fit_feature_scaler,
    load_dataset,
    save_json,
    set_global_seed,
    summarize_trip_metric,
    window_count_by_trip,
)

TARGETS = ["Motor Torque [Nm]", "Throttle [%]"]
MODELS = ["ridge", "hist_gb", "random_forest", "mlp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="artifacts/revision_feature_model_head_to_head"
    )
    return parser.parse_args()


def per_output_mae(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    return [float(x) for x in np.mean(np.abs(y_pred - y_true), axis=0)]


def per_target_trip_summary(
    y_true: np.ndarray, y_pred: np.ndarray, trips: np.ndarray
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for column, target in enumerate(TARGETS):
        values = []
        by_trip = {}
        for trip in np.unique(trips):
            mask = trips == trip
            value = float(np.mean(np.abs(y_pred[mask, column] - y_true[mask, column])))
            by_trip[str(trip)] = value
            values.append(value)
        result[target] = {
            "per_trip_mae": by_trip,
            "summary": summarize_trip_metric(values, SEED),
        }
    return result


def evaluate_candidate(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_trips: np.ndarray,
) -> dict:
    started = time.perf_counter()
    if name == "training_mean":
        mean = np.mean(y_train, axis=0, keepdims=True)
        val_pred = np.repeat(mean, len(y_validation), axis=0)
        test_pred = np.repeat(mean, len(y_test), axis=0)
    else:
        model = make_model(name, output_dim=2, seed=SEED)
        model.fit(flatten_windows(x_train), y_train)
        val_pred = np.asarray(model.predict(flatten_windows(x_validation))).reshape(-1, 2)
        test_pred = np.asarray(model.predict(flatten_windows(x_test))).reshape(-1, 2)
    fit_seconds = time.perf_counter() - started

    val_output = per_output_mae(y_validation, val_pred)
    test_output = per_output_mae(y_test, test_pred)
    return {
        "model": name,
        "fit_seconds": fit_seconds,
        "validation_mae_by_target": dict(zip(TARGETS, val_output)),
        "validation_mean_output_mae": float(np.mean(val_output)),
        "test_window_mae_by_target": dict(zip(TARGETS, test_output)),
        "test_mean_output_mae": float(np.mean(test_output)),
        "test_trip_mae_by_target": per_target_trip_summary(
            y_test, test_pred, test_trips
        ),
    }


def markdown(payload: dict) -> str:
    selected = payload["selected_by_validation_mean_output_mae"]
    lines = [
        "# Full-data EV feature-model head-to-head benchmark",
        "",
        "All candidates use the same corrected complete-trip protocol and every",
        "available window. Selection uses only the mean of the torque and throttle",
        "validation MAEs. The test set is evaluated after the ranking is fixed.",
        "",
        f"- Windows: train={payload['window_counts']['train']:,}, validation={payload['window_counts']['validation']:,}, test={payload['window_counts']['test']:,}",
        f"- Validation-selected baseline: `{selected['model']}`",
        "",
        "| Model | Validation torque MAE | Validation throttle MAE | Validation mean | Test torque trip-mean MAE | Test throttle trip-mean MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in payload["models"]:
        torque = item["test_trip_mae_by_target"][TARGETS[0]]["summary"]["mean"]
        throttle = item["test_trip_mae_by_target"][TARGETS[1]]["summary"]["mean"]
        lines.append(
            f"| {item['model']} | {item['validation_mae_by_target'][TARGETS[0]]:.6f} | "
            f"{item['validation_mae_by_target'][TARGETS[1]]:.6f} | "
            f"{item['validation_mean_output_mae']:.6f} | {torque:.6f} | {throttle:.6f} |"
        )
    lines.extend(
        [
            "",
            "The manuscript LSTM reference values are torque MAE 4.0504 Nm and",
            "throttle MAE 3.7780 percentage points, computed as held-out trip means.",
            "The comparison is descriptive because the feature-model LSTM was not",
            "rerun across five training seeds.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)
    spec = DATASETS["ev"]
    frame = load_dataset(spec)
    split = complete_trip_split(frame, seed=SEED)
    scaler = fit_feature_scaler(frame, split.train_trip_ids, SHARED_CONTEXT)

    x_train, y_train, _ = build_windows(
        frame, split.train_trip_ids, SHARED_CONTEXT, TARGETS, scaler, WINDOW_SIZE
    )
    x_validation, y_validation, _ = build_windows(
        frame,
        split.validation_trip_ids,
        SHARED_CONTEXT,
        TARGETS,
        scaler,
        WINDOW_SIZE,
    )
    x_test, y_test, test_trips = build_windows(
        frame, split.test_trip_ids, SHARED_CONTEXT, TARGETS, scaler, WINDOW_SIZE
    )

    results = [
        evaluate_candidate(
            name,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            test_trips,
        )
        for name in ["training_mean", *MODELS]
    ]
    selected = min(results, key=lambda item: item["validation_mean_output_mae"])

    payload = {
        "status": "completed_full_data_feature_model_head_to_head",
        "dataset": "ev",
        "display_name": spec.display_name,
        "task": "shared observed context to motor torque and throttle",
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "feature_columns": SHARED_CONTEXT,
        "target_columns": TARGETS,
        "split_manifest": split.as_dict(),
        "window_counts": {
            "train": len(x_train),
            "validation": len(x_validation),
            "test": len(x_test),
        },
        "window_counts_by_trip": {
            "train": window_count_by_trip(frame, split.train_trip_ids),
            "validation": window_count_by_trip(frame, split.validation_trip_ids),
            "test": window_count_by_trip(frame, split.test_trip_ids),
        },
        "models": results,
        "selected_by_validation_mean_output_mae": selected,
        "selection_metric": "mean of torque and throttle validation MAEs",
        "test_set_used_for_selection": False,
        "lstm_reference_trip_mean_mae": {
            "Motor Torque [Nm]": 4.0504,
            "Throttle [%]": 3.7780,
        },
        "interpretation_guardrail": (
            "Descriptive single-split comparison. The feature-model LSTM reference "
            "was not rerun across five training seeds."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "ev_feature_model.json", payload)
    (output_dir / "ev_feature_model.md").write_text(markdown(payload), encoding="utf-8")

    with (output_dir / "ev_feature_model.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "validation_torque_mae",
                "validation_throttle_mae",
                "validation_mean_output_mae",
                "test_torque_window_mae",
                "test_throttle_window_mae",
                "test_torque_trip_mean_mae",
                "test_torque_ci_low",
                "test_torque_ci_high",
                "test_throttle_trip_mean_mae",
                "test_throttle_ci_low",
                "test_throttle_ci_high",
                "selected",
            ],
        )
        writer.writeheader()
        for item in results:
            torque_summary = item["test_trip_mae_by_target"][TARGETS[0]]["summary"]
            throttle_summary = item["test_trip_mae_by_target"][TARGETS[1]]["summary"]
            writer.writerow(
                {
                    "model": item["model"],
                    "validation_torque_mae": item["validation_mae_by_target"][TARGETS[0]],
                    "validation_throttle_mae": item["validation_mae_by_target"][TARGETS[1]],
                    "validation_mean_output_mae": item["validation_mean_output_mae"],
                    "test_torque_window_mae": item["test_window_mae_by_target"][TARGETS[0]],
                    "test_throttle_window_mae": item["test_window_mae_by_target"][TARGETS[1]],
                    "test_torque_trip_mean_mae": torque_summary["mean"],
                    "test_torque_ci_low": torque_summary["bootstrap_95pct_mean_ci"][0],
                    "test_torque_ci_high": torque_summary["bootstrap_95pct_mean_ci"][1],
                    "test_throttle_trip_mean_mae": throttle_summary["mean"],
                    "test_throttle_ci_low": throttle_summary["bootstrap_95pct_mean_ci"][0],
                    "test_throttle_ci_high": throttle_summary["bootstrap_95pct_mean_ci"][1],
                    "selected": item["model"] == selected["model"],
                }
            )

    print(markdown(payload))
    print(json.dumps({
        "selected_model": selected["model"],
        "validation_mean_output_mae": selected["validation_mean_output_mae"],
        "selected_test_trip_mae_by_target": {
            target: selected["test_trip_mae_by_target"][target]["summary"]
            for target in TARGETS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
