#!/usr/bin/env python3
"""Run the canonical EV context-to-actuation non-recurrent benchmark.

The benchmark uses the corrected complete-trip split, a MinMaxScaler fitted on
training trips only, length-10 windows built independently within each trip,
and all eligible windows. Torque and throttle validation MAEs stay in their
physical units. A candidate is selected only if the same fitted configuration
independently minimizes validation MAE for every output; no mixed-unit aggregate
is used. The test set is evaluated once after selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_protocol import (  # noqa: E402
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

CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "training_mean": [{}],
    "ridge": [{"alpha": alpha} for alpha in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [
        {
            "n_estimators": 50,
            "max_depth": 12,
            "min_samples_leaf": leaf,
            "max_features": max_features,
            "n_jobs": -1,
            "random_state": SEED,
        }
        for leaf in (5, 20)
        for max_features in (1.0, "sqrt")
    ],
    "hist_gradient_boosting": [
        {"max_iter": 200, "learning_rate": learning_rate, "random_state": SEED}
        for learning_rate in (0.05, 0.1)
    ],
    "mlp": [
        {
            "hidden_layer_sizes": hidden_layers,
            "max_iter": 60,
            "batch_size": 512,
            "early_stopping": True,
            "random_state": SEED,
        }
        for hidden_layers in ((64,), (128, 64))
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="artifacts/revision_feature_model_head_to_head"
    )
    return parser.parse_args()


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(len(x), -1)


def build_model(family: str, params: dict[str, Any]):
    if family == "training_mean":
        return DummyRegressor(strategy="mean")
    if family == "ridge":
        return Ridge(**params)
    if family == "random_forest":
        return RandomForestRegressor(**params)
    if family == "hist_gradient_boosting":
        return MultiOutputRegressor(HistGradientBoostingRegressor(**params))
    if family == "mlp":
        return MLPRegressor(**params)
    raise ValueError(f"Unknown family: {family}")


def output_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true)), axis=0)


def trip_summary(
    y_true: np.ndarray, y_pred: np.ndarray, trips: np.ndarray, output_index: int
) -> dict[str, Any]:
    by_trip: dict[str, float] = {}
    for trip in np.unique(trips):
        mask = trips == trip
        by_trip[str(trip)] = float(
            np.mean(np.abs(y_pred[mask, output_index] - y_true[mask, output_index]))
        )
    return {
        "per_trip_mae": by_trip,
        "summary": summarize_trip_metric(by_trip.values(), SEED),
    }


def markdown(payload: dict[str, Any]) -> str:
    selected = payload["selected_candidate"]
    test = payload["test"]
    return "\n".join(
        [
            "# Full-data EV feature-model benchmark",
            "",
            "All candidates use the same corrected 44/12/14 complete-trip manifest,",
            "training-only scaling and every eligible length-10 window. Torque and",
            "throttle validation MAEs remain separate; no mixed-unit aggregate is used.",
            "",
            f"- Windows: train={payload['window_counts']['train']:,}, validation={payload['window_counts']['validation']:,}, test={payload['window_counts']['test']:,}",
            f"- Selected family: `{selected['family']}`",
            f"- Selected parameters: `{json.dumps(selected['params'], sort_keys=True)}`",
            f"- Validation torque MAE: {selected['validation_mae_by_target'][TARGETS[0]]:.6f} Nm",
            f"- Validation throttle MAE: {selected['validation_mae_by_target'][TARGETS[1]]:.6f} percentage points",
            "",
            "## Held-out trip means",
            "",
            f"- Torque: {test[TARGETS[0]]['trip_mean_mae']:.6f} Nm, 95% CI {test[TARGETS[0]]['bootstrap_95pct_mean_ci']}",
            f"- Throttle: {test[TARGETS[1]]['trip_mean_mae']:.6f} percentage points, 95% CI {test[TARGETS[1]]['bootstrap_95pct_mean_ci']}",
            "",
            "Canonical single-run LSTM references: 4.0504 Nm torque and 3.7780",
            "percentage points throttle. The comparison is mixed by output and is",
            "descriptive; the feature-model LSTM was not rerun across five seeds.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)

    frame = load_dataset(DATASETS["ev"])
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

    flat_train = flatten_windows(x_train)
    flat_validation = flatten_windows(x_validation)
    flat_test = flatten_windows(x_test)

    candidates: list[dict[str, Any]] = []
    fitted: list[tuple[str, dict[str, Any], Any, np.ndarray]] = []

    for family, grid in CANDIDATES.items():
        for params in grid:
            started = time.perf_counter()
            model = build_model(family, params)
            model.fit(flat_train, y_train)
            validation_prediction = np.asarray(model.predict(flat_validation)).reshape(-1, 2)
            validation_mae = output_mae(y_validation, validation_prediction)
            record = {
                "family": family,
                "params": params,
                "validation_mae_by_target": dict(zip(TARGETS, map(float, validation_mae))),
                "fit_seconds": time.perf_counter() - started,
            }
            candidates.append(record)
            fitted.append((family, params, model, validation_mae))
            print(json.dumps(record), flush=True)

    validation_matrix = np.vstack([entry[3] for entry in fitted])
    minimizers = np.argmin(validation_matrix, axis=0)
    if not np.all(minimizers == minimizers[0]):
        winners = [fitted[int(index)][0] for index in minimizers]
        raise RuntimeError(
            "No single candidate independently minimizes every output-specific "
            f"validation MAE. Winners: {dict(zip(TARGETS, winners))}. "
            "Pre-specify a dimensionless or task-priority selection rule before testing."
        )

    selected_index = int(minimizers[0])
    family, params, selected_model, selected_validation_mae = fitted[selected_index]

    test_prediction = np.asarray(selected_model.predict(flat_test)).reshape(-1, 2)
    test_window_mae = output_mae(y_test, test_prediction)
    test_payload: dict[str, Any] = {}
    for index, target in enumerate(TARGETS):
        summary = trip_summary(y_test, test_prediction, test_trips, index)["summary"]
        test_payload[target] = {
            "window_mae": float(test_window_mae[index]),
            "trip_mean_mae": float(summary["mean"]),
            "bootstrap_95pct_mean_ci": summary["bootstrap_95pct_mean_ci"],
            "n_test_trips": int(len(np.unique(test_trips))),
        }

    selected_candidate = {
        "family": family,
        "params": params,
        "validation_mae_by_target": dict(zip(TARGETS, map(float, selected_validation_mae))),
        "selection_rule": (
            "unique fitted candidate with minimum validation MAE for both outputs; "
            "torque and throttle are not combined across units"
        ),
    }

    payload: dict[str, Any] = {
        "status": "completed_full_data_feature_model_head_to_head",
        "dataset": "ev",
        "display_name": DATASETS["ev"].display_name,
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
        "candidates": candidates,
        "selected_candidate": selected_candidate,
        "test": test_payload,
        "test_set_used_for_selection": False,
        "lstm_reference_trip_mean_mae": {
            TARGETS[0]: 4.0504,
            TARGETS[1]: 3.7780,
            "training_seed": SEED,
            "multi_seed_rerun": False,
        },
        "interpretation_guardrail": (
            "The result is mixed by output and does not establish general recurrent "
            "or non-recurrent superiority."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "ev_feature_model.json", payload)
    (output_dir / "ev_feature_model.md").write_text(markdown(payload), encoding="utf-8")

    with (output_dir / "ev_feature_model_selection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["family", "params", f"val_mae_{TARGETS[0]}", f"val_mae_{TARGETS[1]}"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "family": item["family"],
                    "params": json.dumps(item["params"]),
                    f"val_mae_{TARGETS[0]}": item["validation_mae_by_target"][TARGETS[0]],
                    f"val_mae_{TARGETS[1]}": item["validation_mae_by_target"][TARGETS[1]],
                }
            )

    with (output_dir / "ev_feature_model.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "target",
            "selected_family",
            "selected_params",
            "test_window_mae",
            "test_trip_mean_mae",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "n_test_trips",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for target in TARGETS:
            test_item = test_payload[target]
            writer.writerow(
                {
                    "target": target,
                    "selected_family": family,
                    "selected_params": json.dumps(params),
                    "test_window_mae": test_item["window_mae"],
                    "test_trip_mean_mae": test_item["trip_mean_mae"],
                    "bootstrap_ci_low": test_item["bootstrap_95pct_mean_ci"][0],
                    "bootstrap_ci_high": test_item["bootstrap_95pct_mean_ci"][1],
                    "n_test_trips": test_item["n_test_trips"],
                }
            )

    print(markdown(payload))


if __name__ == "__main__":
    main()
