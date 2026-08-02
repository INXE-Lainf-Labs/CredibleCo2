#!/usr/bin/env python3
"""Run final full-data XGBoost and Extra Trees baselines.

The protocol is aligned with the corrected reviewer-facing experiments:

- complete-trip train/validation/test split fixed by ``SEED``;
- feature scaler fitted only on training-trip rows;
- windows created independently inside each trip after the split;
- all available windows used;
- hyperparameters selected exclusively by validation MAE;
- only the selected configuration in each model family is evaluated on test.

The input set matches the emissions LSTM: velocity, throttle, and motor torque.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from src.revision_protocol import (
    ACTUATION,
    DATASETS,
    SEED,
    WINDOW_SIZE,
    build_windows,
    complete_trip_split,
    fit_feature_scaler,
    load_dataset,
    per_trip_mae,
    regression_metrics,
    save_json,
    set_global_seed,
    summarize_trip_metric,
    window_count_by_trip,
)


EXTRA_TREES_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "extra_trees_sqrt_d20_leaf2",
        "n_estimators": 256,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "name": "extra_trees_half_d30_leaf2",
        "n_estimators": 256,
        "max_depth": 30,
        "min_samples_leaf": 2,
        "max_features": 0.5,
    },
    {
        "name": "extra_trees_all_d30_leaf5",
        "n_estimators": 256,
        "max_depth": 30,
        "min_samples_leaf": 5,
        "max_features": 1.0,
    },
]

XGBOOST_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "xgb_d4_mcw1_lr005",
        "max_depth": 4,
        "min_child_weight": 1,
        "learning_rate": 0.05,
    },
    {
        "name": "xgb_d6_mcw1_lr005",
        "max_depth": 6,
        "min_child_weight": 1,
        "learning_rate": 0.05,
    },
    {
        "name": "xgb_d6_mcw5_lr005",
        "max_depth": 6,
        "min_child_weight": 5,
        "learning_rate": 0.05,
    },
    {
        "name": "xgb_d8_mcw5_lr003",
        "max_depth": 8,
        "min_child_weight": 5,
        "learning_rate": 0.03,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output-dir", default="artifacts/revision_strong_tabular_baselines"
    )
    return parser.parse_args()


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return np.asarray(x.reshape(len(x), -1), dtype=np.float32)


def evaluation_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trips: np.ndarray | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = regression_metrics(y_true, y_pred)
    if trips is not None:
        trip_values = per_trip_mae(y_true, y_pred, trips)
        payload["per_trip_mae"] = trip_values
        payload["trip_level_mae_summary"] = summarize_trip_metric(
            trip_values.values(), SEED
        )
    return payload


def extra_trees_factory(config: dict[str, Any], threads: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=config["max_features"],
        bootstrap=False,
        n_jobs=threads,
        random_state=SEED,
    )


def xgboost_factory(config: dict[str, Any], threads: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_estimators=1200,
        early_stopping_rounds=60,
        max_depth=int(config["max_depth"]),
        min_child_weight=float(config["min_child_weight"]),
        learning_rate=float(config["learning_rate"]),
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        max_bin=256,
        n_jobs=threads,
        random_state=SEED,
        verbosity=1,
    )


def select_family(
    family: str,
    configs: list[dict[str, Any]],
    factory: Callable[[dict[str, Any], int], Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_trips: np.ndarray,
    threads: int,
    work_dir: Path,
) -> dict[str, Any]:
    candidate_results: list[dict[str, Any]] = []
    best_validation_mae = float("inf")
    best_config: dict[str, Any] | None = None
    best_model_path = work_dir / f"{family}_selected.joblib"

    for config in configs:
        model = factory(config, threads)
        started = time.perf_counter()
        if family == "xgboost":
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_validation, y_validation)],
                verbose=False,
            )
        else:
            model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        validation_prediction = np.asarray(model.predict(x_validation)).reshape(-1, 1)
        validation = evaluation_payload(y_validation, validation_prediction)
        candidate = {
            "family": family,
            "config": config,
            "validation": validation,
            "fit_seconds": fit_seconds,
        }
        if family == "xgboost":
            candidate["best_iteration"] = (
                int(model.best_iteration) if model.best_iteration is not None else None
            )
        candidate_results.append(candidate)
        print(json.dumps(candidate), flush=True)

        if float(validation["mae"]) < best_validation_mae:
            best_validation_mae = float(validation["mae"])
            best_config = dict(config)
            joblib.dump(model, best_model_path, compress=0)

        del validation_prediction
        del model
        gc.collect()

    if best_config is None or not best_model_path.exists():
        raise RuntimeError(f"No selected model was produced for {family}")

    selected_model = joblib.load(best_model_path)
    test_prediction = np.asarray(selected_model.predict(x_test)).reshape(-1, 1)
    test = evaluation_payload(y_test, test_prediction, test_trips)
    selected_candidate = min(
        candidate_results, key=lambda item: float(item["validation"]["mae"])
    )

    result = {
        "family": family,
        "selection_metric": "validation_mae",
        "selected_config": best_config,
        "selected_validation": selected_candidate["validation"],
        "selected_fit_seconds": selected_candidate["fit_seconds"],
        "selected_best_iteration": selected_candidate.get("best_iteration"),
        "test": test,
        "test_set_used_for_selection": False,
        "candidates": candidate_results,
    }
    del selected_model
    del test_prediction
    gc.collect()
    return result


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Final strong tabular baselines: {payload['display_name']}",
        "",
        "All candidates use the same complete-trip split and every available window.",
        "The feature scaler is fitted only on training-trip rows. Hyperparameters are",
        "selected exclusively by validation MAE, and only the selected configuration",
        "in each family is evaluated on the test set.",
        "",
        f"- Split seed: `{payload['seed']}`",
        f"- Features: {', '.join(payload['feature_columns'])}",
        f"- Windows: train={payload['window_counts']['train']:,}, validation={payload['window_counts']['validation']:,}, test={payload['window_counts']['test']:,}",
        "",
        "| Family | Selected configuration | Validation MAE | Test MAE | Test RMSE | Test R2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in payload["families"]:
        config = family["selected_config"]["name"]
        lines.append(
            f"| {family['family']} | `{config}` | "
            f"{family['selected_validation']['mae']:.8g} | "
            f"{family['test']['mae']:.8g} | "
            f"{family['test']['rmse']:.8g} | "
            f"{family['test']['r2']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Candidate validation results",
            "",
            "| Family | Configuration | Validation MAE | Validation RMSE | Validation R2 | Fit seconds |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for family in payload["families"]:
        for candidate in family["candidates"]:
            validation = candidate["validation"]
            lines.append(
                f"| {candidate['family']} | `{candidate['config']['name']}` | "
                f"{validation['mae']:.8g} | {validation['rmse']:.8g} | "
                f"{validation['r2']:.8g} | {candidate['fit_seconds']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            payload["interpretation_guardrail"],
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "dataset",
        "display_name",
        "family",
        "config",
        "selected",
        "validation_mae",
        "validation_rmse",
        "validation_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
        "fit_seconds",
        "best_iteration",
        "train_windows",
        "validation_windows",
        "test_windows",
        "split_seed",
        "test_set_used_for_selection",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family in payload["families"]:
            selected_name = family["selected_config"]["name"]
            for candidate in family["candidates"]:
                selected = candidate["config"]["name"] == selected_name
                row = {
                    "dataset": payload["dataset"],
                    "display_name": payload["display_name"],
                    "family": family["family"],
                    "config": candidate["config"]["name"],
                    "selected": selected,
                    "validation_mae": candidate["validation"]["mae"],
                    "validation_rmse": candidate["validation"]["rmse"],
                    "validation_r2": candidate["validation"]["r2"],
                    "fit_seconds": candidate["fit_seconds"],
                    "best_iteration": candidate.get("best_iteration"),
                    "train_windows": payload["window_counts"]["train"],
                    "validation_windows": payload["window_counts"]["validation"],
                    "test_windows": payload["window_counts"]["test"],
                    "split_seed": payload["seed"],
                    "test_set_used_for_selection": False,
                }
                if selected:
                    row.update(
                        {
                            "test_mae": family["test"]["mae"],
                            "test_rmse": family["test"]["rmse"],
                            "test_r2": family["test"]["r2"],
                        }
                    )
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    set_global_seed(SEED)
    spec = DATASETS[args.dataset]
    frame = load_dataset(spec)
    split = complete_trip_split(frame, seed=SEED)
    scaler = fit_feature_scaler(frame, split.train_trip_ids, ACTUATION)

    x_train, y_train, _ = build_windows(
        frame, split.train_trip_ids, ACTUATION, ["CO2 Emissions"], scaler, WINDOW_SIZE
    )
    x_validation, y_validation, _ = build_windows(
        frame,
        split.validation_trip_ids,
        ACTUATION,
        ["CO2 Emissions"],
        scaler,
        WINDOW_SIZE,
    )
    x_test, y_test, test_trips = build_windows(
        frame, split.test_trip_ids, ACTUATION, ["CO2 Emissions"], scaler, WINDOW_SIZE
    )

    x_train_flat = flatten_windows(x_train)
    x_validation_flat = flatten_windows(x_validation)
    x_test_flat = flatten_windows(x_test)
    y_train_flat = np.asarray(y_train).reshape(-1)

    with tempfile.TemporaryDirectory(prefix="strong-baselines-") as temporary:
        work_dir = Path(temporary)
        extra_trees = select_family(
            "extra_trees",
            EXTRA_TREES_CONFIGS,
            extra_trees_factory,
            x_train_flat,
            y_train_flat,
            x_validation_flat,
            y_validation,
            x_test_flat,
            y_test,
            test_trips,
            args.threads,
            work_dir,
        )
        xgboost = select_family(
            "xgboost",
            XGBOOST_CONFIGS,
            xgboost_factory,
            x_train_flat,
            y_train_flat,
            x_validation_flat,
            y_validation,
            x_test_flat,
            y_test,
            test_trips,
            args.threads,
            work_dir,
        )

    payload: dict[str, Any] = {
        "status": "completed_final_strong_tabular_baselines",
        "dataset": args.dataset,
        "display_name": spec.display_name,
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "feature_set": "actuation_inputs",
        "feature_columns": ACTUATION,
        "target_columns": ["CO2 Emissions"],
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
        "scaler_fit_scope": "training_trip_rows_only",
        "all_windows_used": True,
        "families": [extra_trees, xgboost],
        "test_set_used_for_selection": False,
        "interpretation_guardrail": (
            "Predictive comparison conditioned on observed actuation inputs. Road grade, "
            "payload, wind, driver identity, and transmission state remain unobserved, "
            "so the comparison does not establish causal operating-condition equivalence."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / f"{args.dataset}.json", payload)
    (output_dir / f"{args.dataset}.md").write_text(markdown(payload), encoding="utf-8")
    write_csv(output_dir / f"{args.dataset}.csv", payload)
    print(markdown(payload))


if __name__ == "__main__":
    main()
