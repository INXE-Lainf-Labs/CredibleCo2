#!/usr/bin/env python3
"""Run exact full-data non-recurrent benchmarks for the LSTM input set.

This focused runner is intentionally limited to EV and QX50 and to the
actuation-input feature set used by the emissions LSTM: velocity, throttle,
and motor torque. It removes the window caps used by the broader exploratory
ablation workflow, allowing an exact full-data comparison on the same trip
split and all available train/validation/test windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_revision_experiments import evaluate_model, mean_baseline
from src.revision_protocol import (
    ACTUATION,
    DATASETS,
    SEED,
    WINDOW_SIZE,
    build_windows,
    complete_trip_split,
    fit_feature_scaler,
    load_dataset,
    save_json,
    set_global_seed,
    window_count_by_trip,
)

MODELS = ["ridge", "hist_gb", "random_forest", "mlp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ev", "qx50"], required=True)
    parser.add_argument("--output-dir", default="artifacts/revision_full_data_head_to_head")
    return parser.parse_args()


def markdown(payload: dict) -> str:
    lines = [
        f"# Full-data head-to-head benchmarks: {payload['display_name']}",
        "",
        "All models use the same complete-trip split, training-only scaler,",
        "actuation inputs, and every available window in train, validation, and test.",
        "Model selection is based on validation MAE; test metrics are reported only",
        "after the validation ranking is fixed.",
        "",
        f"- Features: {', '.join(payload['feature_columns'])}",
        f"- Windows: train={payload['window_counts']['train']:,}, validation={payload['window_counts']['validation']:,}, test={payload['window_counts']['test']:,}",
        f"- Selected model: `{payload['selected_by_validation_mae']['model']}`",
        "",
        "| Model | Validation MAE | Test MAE | Test RMSE | Test R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in payload["models"]:
        lines.append(
            f"| {result['model']} | {result['validation']['mae']:.8g} | "
            f"{result['test']['mae']:.8g} | {result['test']['rmse']:.8g} | "
            f"{result['test']['r2']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "This is a predictive comparison conditioned on observed actuation inputs. "
            "It does not control absent road grade, payload, wind, driver identity, or "
            "transmission state and therefore does not establish causal operating-condition equivalence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
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

    results = [
        mean_baseline(y_train, y_validation, y_test, test_trips, SEED)
    ]
    for model_name in MODELS:
        results.append(
            evaluate_model(
                model_name,
                x_train,
                y_train,
                x_validation,
                y_validation,
                x_test,
                y_test,
                test_trips,
                SEED,
            )
        )

    selected = min(results, key=lambda item: item["validation"]["mae"])
    payload = {
        "status": "completed_full_data_head_to_head",
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
        "models": results,
        "selected_by_validation_mae": selected,
        "test_set_used_for_selection": False,
        "interpretation_guardrail": (
            "Predictive comparison conditioned on observed actuation inputs; absent "
            "road grade, payload, wind, driver, and transmission variables remain uncontrolled."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / f"{args.dataset}.json", payload)
    (output_dir / f"{args.dataset}.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({
        "dataset": args.dataset,
        "selected_model": selected["model"],
        "validation_mae": selected["validation"]["mae"],
        "test": selected["test"],
    }, indent=2))


if __name__ == "__main__":
    main()
