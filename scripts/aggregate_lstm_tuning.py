#!/usr/bin/env python3
"""Aggregate staged LSTM tuning artifacts and emit downstream matrices."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

SEEDS = [20260801, 20260802, 20260803, 20260804, 20260805]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--source-stage", choices=["screen", "refine", "full", "final"], required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--next-stage", choices=["refine", "full", "final", "summary"], required=True)
    parser.add_argument("--next-epochs", type=int, default=60)
    parser.add_argument("--next-max-train-windows", type=int, default=999_999_999)
    parser.add_argument("--next-max-validation-windows", type=int, default=999_999_999)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_records(root: Path, stage: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("record_type") == "lstm_tuning_run" and payload.get("stage") == stage:
            payload["_path"] = str(path)
            records.append(payload)
    if not records:
        raise RuntimeError(f"No {stage} tuning records found under {root}")
    return records


def group_key(record: dict[str, Any]) -> str:
    return f"{record['task']}:{record['dataset']}"


def config_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    config = record["configuration"]
    return (
        float(record["checkpoint_selection"]["best_validation_score"]),
        int(config["parameter_count"]),
        int(config["window_size"]),
        int(config["hidden_dim"]),
        int(config["num_blocks"]),
        float(config["gradient_clip_global_norm"]),
        str(config["target_mode"]),
    )


def matrix_row(record: dict[str, Any], args: argparse.Namespace, seed: int) -> dict[str, Any]:
    config = record["configuration"]
    return {
        "task": record["task"],
        "dataset": record["dataset"],
        "split_seed": int(record["split_seed"]),
        "training_seed": seed,
        "epochs": args.next_epochs,
        "hidden_dim": int(config["hidden_dim"]),
        "num_blocks": int(config["num_blocks"]),
        "window_size": int(config["window_size"]),
        "gradient_clip": float(config["gradient_clip_global_norm"]),
        "target_mode": str(config["target_mode"]),
        "max_train_windows": args.next_max_train_windows,
        "max_validation_windows": args.next_max_validation_windows,
    }


def select_and_write(records: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ranked = sorted(grouped[key], key=config_sort_key)
        selected.extend(ranked[: args.top_k])

    rows: list[dict[str, Any]] = []
    if args.next_stage == "final":
        for record in selected:
            for seed in SEEDS:
                rows.append(matrix_row(record, args, seed))
    else:
        for record in selected:
            rows.append(matrix_row(record, args, 20260801))

    matrix = {"include": rows}
    (output_dir / "selected_records.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    (output_dir / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")

    with (output_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "group",
            "rank",
            "validation_standardized_mse",
            "hidden_dim",
            "num_blocks",
            "window_size",
            "gradient_clip",
            "target_mode",
            "best_epoch",
            "parameter_count",
        ])
        for key in sorted(grouped):
            for rank, record in enumerate(sorted(grouped[key], key=config_sort_key), start=1):
                config = record["configuration"]
                writer.writerow([
                    key,
                    rank,
                    record["checkpoint_selection"]["best_validation_score"],
                    config["hidden_dim"],
                    config["num_blocks"],
                    config["window_size"],
                    config["gradient_clip_global_norm"],
                    config["target_mode"],
                    record["checkpoint_selection"]["best_epoch"],
                    config["parameter_count"],
                ])

    print(json.dumps(matrix, separators=(",", ":")))


def summarize_final(records: list[dict[str, Any]], output_dir: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)

    summary_rows: list[dict[str, Any]] = []
    markdown = [
        "# Validation-selected LSTM tuning: final five-seed test summary",
        "",
        "Configuration search used validation trips only. Test trips were evaluated only after one configuration per task/dataset had been selected.",
        "",
    ]
    for key in sorted(grouped):
        runs = sorted(grouped[key], key=lambda item: int(item["training_seed"]))
        config = runs[0]["configuration"]
        markdown.extend([
            f"## {key}",
            "",
            f"Selected configuration: window={config['window_size']}, hidden={config['hidden_dim']}, blocks={config['num_blocks']}, gradient clip={config['gradient_clip_global_norm']}, target mode={config['target_mode']}.",
            "",
            "| Output | Mean global MAE | SD | Mean trip-level MAE | SD |",
            "|---|---:|---:|---:|---:|",
        ])
        output_names = list(runs[0]["test"]["global_output_mae"].keys())
        for output in output_names:
            global_values = [float(run["test"]["global_output_mae"][output]) for run in runs]
            trip_values = [
                float(run["test"]["trip_level_output_mae"][output]["mean"])
                for run in runs
            ]
            row = {
                "group": key,
                "output": output,
                "n_seeds": len(runs),
                "global_mae_mean": mean(global_values),
                "global_mae_sample_sd": stdev(global_values) if len(global_values) > 1 else 0.0,
                "trip_mae_mean": mean(trip_values),
                "trip_mae_sample_sd": stdev(trip_values) if len(trip_values) > 1 else 0.0,
                "window_size": config["window_size"],
                "hidden_dim": config["hidden_dim"],
                "num_blocks": config["num_blocks"],
                "gradient_clip": config["gradient_clip_global_norm"],
                "target_mode": config["target_mode"],
            }
            summary_rows.append(row)
            markdown.append(
                f"| {output} | {row['global_mae_mean']:.8g} | {row['global_mae_sample_sd']:.8g} | {row['trip_mae_mean']:.8g} | {row['trip_mae_sample_sd']:.8g} |"
            )
        markdown.append("")

    with (output_dir / "final_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_dir / "final_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(Path(args.input_dir), args.source_stage)
    if args.next_stage == "summary":
        summarize_final(records, output_dir)
    else:
        select_and_write(records, args, output_dir)


if __name__ == "__main__":
    main()

# Content-only touch to retrigger the branch-filtered GitHub Actions workflow.
