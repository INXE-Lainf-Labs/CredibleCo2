#!/usr/bin/env python3
"""Run reviewer-requested leakage-free benchmarks and input ablations.

The experiments use complete-trip train/validation/test partitions, fit feature
scalers on training trips only, and construct sliding windows only after the
split. The main reviewer-facing ablation compares speed-only input against the
richer observed context and actuation sets. Road grade, payload, GPS, and driver
identity are not present in the repository datasets and therefore cannot be
included empirically; this absence is recorded in the output.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_protocol import (
    ACTUATION,
    ALL_OBSERVED,
    DATASETS,
    SEED,
    SHARED_CONTEXT,
    SPEED,
    WINDOW_SIZE,
    build_windows,
    complete_trip_split,
    deterministic_sample,
    fit_feature_scaler,
    load_dataset,
    per_trip_mae,
    regression_metrics,
    save_json,
    set_global_seed,
    summarize_trip_metric,
    window_count_by_trip,
)

FEATURE_SETS = {
    "speed_only": SPEED,
    "shared_observed_context": SHARED_CONTEXT,
    "actuation_inputs": ACTUATION,
    "all_observed_inputs": ALL_OBSERVED,
}

ABSENT_COVARIATES = [
    "road grade or elevation",
    "GPS latitude/longitude",
    "payload or vehicle mass variation",
    "driver identity or explicit driving-style label",
    "gear position or transmission state",
]


def make_model(name: str, output_dim: int, seed: int):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "hist_gb":
        base = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            random_state=seed,
        )
        return base if output_dim == 1 else MultiOutputRegressor(base)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1)


def evaluate_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_trips: np.ndarray,
    seed: int,
) -> dict:
    output_dim = y_train.shape[1]
    model = make_model(model_name, output_dim, seed)
    y_fit = y_train.ravel() if output_dim == 1 else y_train

    started = time.perf_counter()
    model.fit(flatten_windows(x_train), y_fit)
    fit_seconds = time.perf_counter() - started

    val_pred = np.asarray(model.predict(flatten_windows(x_validation)))
    test_pred = np.asarray(model.predict(flatten_windows(x_test)))
    if output_dim == 1:
        val_pred = val_pred.reshape(-1, 1)
        test_pred = test_pred.reshape(-1, 1)

    trip_values = per_trip_mae(y_test, test_pred, test_trips)
    return {
        "model": model_name,
        "fit_seconds": fit_seconds,
        "validation": regression_metrics(y_validation, val_pred),
        "test": regression_metrics(y_test, test_pred),
        "test_trip_mae": trip_values,
        "test_trip_mae_summary": summarize_trip_metric(trip_values.values(), seed),
    }


def mean_baseline(
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    test_trips: np.ndarray,
    seed: int,
) -> dict:
    mean = np.mean(y_train, axis=0, keepdims=True)
    val_pred = np.repeat(mean, len(y_validation), axis=0)
    test_pred = np.repeat(mean, len(y_test), axis=0)
    trip_values = per_trip_mae(y_test, test_pred, test_trips)
    return {
        "model": "training_mean",
        "fit_seconds": 0.0,
        "validation": regression_metrics(y_validation, val_pred),
        "test": regression_metrics(y_test, test_pred),
        "test_trip_mae": trip_values,
        "test_trip_mae_summary": summarize_trip_metric(trip_values.values(), seed),
    }


def prepare_task(
    frame,
    manifest,
    feature_columns,
    target_columns,
    max_train_windows,
    max_validation_windows,
    max_test_windows,
    seed,
):
    scaler = fit_feature_scaler(frame, manifest.train_trip_ids, feature_columns)
    prepared = {}
    for offset, (split, trips, maximum) in enumerate(
        [
            ("train", manifest.train_trip_ids, max_train_windows),
            ("validation", manifest.validation_trip_ids, max_validation_windows),
            ("test", manifest.test_trip_ids, max_test_windows),
        ]
    ):
        x, y, window_trips = build_windows(
            frame,
            trips,
            feature_columns,
            target_columns,
            scaler,
            WINDOW_SIZE,
        )
        x, y, window_trips = deterministic_sample(
            x, y, window_trips, maximum, seed + offset
        )
        prepared[split] = (x, y, window_trips)
    return prepared


def run_emissions_experiments(args, frame, manifest) -> dict:
    results = {}
    for feature_name, columns in FEATURE_SETS.items():
        prepared = prepare_task(
            frame,
            manifest,
            columns,
            ["CO2 Emissions"],
            args.max_train_windows,
            args.max_validation_windows,
            args.max_test_windows,
            args.seed,
        )
        x_train, y_train, _ = prepared["train"]
        x_val, y_val, _ = prepared["validation"]
        x_test, y_test, test_trips = prepared["test"]

        feature_result = {
            "features": columns,
            "window_counts_used": {
                "train": len(x_train),
                "validation": len(x_val),
                "test": len(x_test),
            },
            "models": [mean_baseline(y_train, y_val, y_test, test_trips, args.seed)],
        }
        for model_name in args.models:
            feature_result["models"].append(
                evaluate_model(
                    model_name,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    x_test,
                    y_test,
                    test_trips,
                    args.seed,
                )
            )
        results[feature_name] = feature_result
    return results


def run_ev_feature_model_benchmarks(args, frame, manifest) -> dict:
    prepared = prepare_task(
        frame,
        manifest,
        SHARED_CONTEXT,
        ["Motor Torque [Nm]", "Throttle [%]"],
        args.max_train_windows,
        args.max_validation_windows,
        args.max_test_windows,
        args.seed,
    )
    x_train, y_train, _ = prepared["train"]
    x_val, y_val, _ = prepared["validation"]
    x_test, y_test, test_trips = prepared["test"]
    models = [mean_baseline(y_train, y_val, y_test, test_trips, args.seed)]
    for model_name in args.models:
        models.append(
            evaluate_model(
                model_name,
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                test_trips,
                args.seed,
            )
        )
    return {
        "features": SHARED_CONTEXT,
        "targets": ["Motor Torque [Nm]", "Throttle [%]"],
        "window_counts_used": {
            "train": len(x_train),
            "validation": len(x_val),
            "test": len(x_test),
        },
        "models": models,
    }


def markdown_summary(payload: dict) -> str:
    lines = [
        f"# Corrected-protocol experiments: {payload['dataset']['display_name']}",
        "",
        "All results use disjoint complete trips for training, validation, and test; scalers are fitted on training trips only; windows are constructed after splitting.",
        "",
        "## Covariate scope",
        "",
        "The repository dataset does not contain road grade/elevation, GPS, payload, driver identity, or transmission state. Consequently, the input ablation evaluates whether speed alone is sufficient relative to the richer observed context; it does not claim to quantify the causal effect of unobserved grade.",
        "",
        "## Emissions benchmarks and ablations",
        "",
        "| Input set | Model | Test MAE | Test RMSE | Test R2 | Trip-level mean MAE | 95% bootstrap CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for feature_name, result in payload["emissions_experiments"].items():
        for model in result["models"]:
            summary = model["test_trip_mae_summary"]
            ci = summary.get("bootstrap_95pct_mean_ci", [float("nan"), float("nan")])
            lines.append(
                f"| {feature_name} | {model['model']} | {model['test']['mae']:.6g} | {model['test']['rmse']:.6g} | {model['test']['r2']:.5f} | {summary.get('mean', float('nan')):.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] |"
            )

    if payload.get("ev_feature_model_benchmarks"):
        lines.extend(
            [
                "",
                "## EV feature-model benchmarks",
                "",
                "| Model | Test MAE | Test RMSE | Test R2 | Trip-level mean MAE | 95% bootstrap CI |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model in payload["ev_feature_model_benchmarks"]["models"]:
            summary = model["test_trip_mae_summary"]
            ci = summary.get("bootstrap_95pct_mean_ci", [float("nan"), float("nan")])
            lines.append(
                f"| {model['model']} | {model['test']['mae']:.6g} | {model['test']['rmse']:.6g} | {model['test']['r2']:.5f} | {summary.get('mean', float('nan')):.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--models",
        default="ridge,hist_gb,random_forest,mlp",
        help="Comma-separated list: ridge,hist_gb,random_forest,mlp",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-windows", type=int, default=200000)
    parser.add_argument("--max-validation-windows", type=int, default=100000)
    parser.add_argument("--max-test-windows", type=int, default=150000)
    parser.add_argument("--output-dir", default="artifacts/revision_experiments")
    args = parser.parse_args()
    args.models = [name.strip() for name in args.models.split(",") if name.strip()]
    return args


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    spec = DATASETS[args.dataset]
    frame = load_dataset(spec)
    manifest = complete_trip_split(frame, args.seed)

    payload = {
        "protocol_version": 1,
        "dataset": {
            "key": spec.key,
            "display_name": spec.display_name,
            "kind": spec.kind,
            "path": spec.path,
            "rows_after_cleaning": len(frame),
            "trip_count_after_cleaning": int(frame["Trip"].nunique()),
        },
        "seed": args.seed,
        "window_size": WINDOW_SIZE,
        "split_manifest": manifest.as_dict(),
        "full_window_counts_by_split": {
            "train": window_count_by_trip(frame, manifest.train_trip_ids),
            "validation": window_count_by_trip(frame, manifest.validation_trip_ids),
            "test": window_count_by_trip(frame, manifest.test_trip_ids),
        },
        "absent_covariates": ABSENT_COVARIATES,
        "interpretation": "The speed-only versus richer-observed-input ablation tests sufficiency of measured covariates; it cannot identify effects of absent road grade, payload, or driver variables.",
        "emissions_experiments": run_emissions_experiments(args, frame, manifest),
    }
    if spec.kind == "EV":
        payload["ev_feature_model_benchmarks"] = run_ev_feature_model_benchmarks(
            args, frame, manifest
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{spec.key}.json"
    md_path = output_dir / f"{spec.key}.md"
    save_json(json_path, payload)
    md_path.write_text(markdown_summary(payload), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
