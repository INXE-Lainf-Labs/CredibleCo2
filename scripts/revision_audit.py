#!/usr/bin/env python3
"""Audit dataset covariates and the historical split protocol.

This script is intentionally read-only. It documents the information needed to
answer Reviewers 3 and 4 without altering the published results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

WINDOW_SIZE = 10
OUT_DIR = Path("artifacts")

DATASETS = [
    Path("data/eletrico_ieee.csv"),
    Path("data/Dataset_QX50_trip-clean.csv"),
    Path("data/Dataset_Blazer.csv"),
    Path("data/Dataset_Pacifica.csv"),
]

GRADE_TERMS = (
    "grade",
    "slope",
    "inclination",
    "altitude",
    "elevation",
    "latitude",
    "longitude",
    "gps",
    "road angle",
)

PAYLOAD_DRIVER_TERMS = (
    "payload",
    "mass",
    "weight",
    "driver",
    "pedal",
)

POWERTRAIN_TERMS = (
    "gear",
    "transmission",
    "lock-up",
    "lockup",
    "engine speed",
    "rpm",
    "throttle",
    "torque",
)


def matching_columns(columns: list[str], terms: tuple[str, ...]) -> list[str]:
    return [c for c in columns if any(term in c.lower() for term in terms)]


def historical_split_diagnostics(path: Path) -> dict[str, Any]:
    trip_df = pd.read_csv(path, usecols=lambda name: name == "Trip", low_memory=False)
    if "Trip" not in trip_df.columns:
        return {"available": False, "reason": "No Trip column"}

    trip_order = trip_df["Trip"].drop_duplicates().tolist()
    counts = trip_df.groupby("Trip", sort=False).size()
    n_trips = len(trip_order)
    n_train_pool = int(0.8 * n_trips)
    train_trips = trip_order[:n_train_pool]
    test_trips = trip_order[n_train_pool:]

    window_counts = {
        str(trip): max(int(counts.loc[trip]) - WINDOW_SIZE, 0)
        for trip in train_trips
    }
    total_train_pool_windows = sum(window_counts.values())
    historical_train_window_count = int(0.8 * total_train_pool_windows)

    cumulative = 0
    boundary_trip = None
    boundary_offset = None
    boundary_trip_windows = None
    for trip in train_trips:
        n_windows = window_counts[str(trip)]
        if cumulative + n_windows >= historical_train_window_count:
            boundary_trip = str(trip)
            boundary_offset = historical_train_window_count - cumulative
            boundary_trip_windows = n_windows
            break
        cumulative += n_windows

    cuts_trip = bool(
        boundary_trip is not None
        and boundary_offset is not None
        and boundary_trip_windows is not None
        and 0 < boundary_offset < boundary_trip_windows
    )

    return {
        "available": True,
        "rows": int(len(trip_df)),
        "trip_count": n_trips,
        "historical_outer_split": {
            "method": "first 80% of unique trips for training pool; remaining trips for test",
            "training_pool_trip_count": len(train_trips),
            "test_trip_count": len(test_trips),
            "trip_ids_disjoint": set(train_trips).isdisjoint(test_trips),
            "randomized": False,
        },
        "historical_inner_split": {
            "method": "first 80% of concatenated training-pool windows for training; remaining windows for validation",
            "training_pool_window_count": total_train_pool_windows,
            "training_window_count": historical_train_window_count,
            "validation_window_count": total_train_pool_windows - historical_train_window_count,
            "boundary_trip": boundary_trip,
            "boundary_offset_within_trip_windows": boundary_offset,
            "boundary_trip_window_count": boundary_trip_windows,
            "boundary_cuts_trip": cuts_trip,
            "overlapping_window_leakage_risk": cuts_trip,
            "maximum_adjacent_window_overlap_timesteps": WINDOW_SIZE - 1 if cuts_trip else 0,
        },
    }


def audit_dataset(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    result.update(
        {
            "columns": columns,
            "grade_or_geolocation_candidates": matching_columns(columns, GRADE_TERMS),
            "payload_or_driver_candidates": matching_columns(columns, PAYLOAD_DRIVER_TERMS),
            "powertrain_state_candidates": matching_columns(columns, POWERTRAIN_TERMS),
            "split_diagnostics": historical_split_diagnostics(path),
        }
    )
    return result


def markdown_report(audits: list[dict[str, Any]]) -> str:
    lines = [
        "# Second-revision reproducibility and covariate audit",
        "",
        "This report is generated directly from the repository datasets and the historical split logic in `src/helper.py`.",
        "",
        "## Executive findings",
        "",
        "1. The historical outer train/test split is trip-wise, so test trips are disjoint from the training-pool trips.",
        "2. Sliding windows are created separately inside each trip, so no window crosses a trip boundary.",
        "3. The historical inner train/validation split is performed after all training-pool windows are concatenated. If the 80% boundary falls inside a trip, adjacent training and validation windows share up to nine of ten raw timesteps.",
        "4. `normalize` in `src/helper.py` fits `MinMaxScaler` before the trip split, using the full dataset. This leaks test-distribution minima and maxima into preprocessing even though labels and test windows are not used for fitting the neural network.",
        "5. No claim about grade availability is made until the dataset headers below are inspected.",
        "",
        "## Dataset headers and split diagnostics",
        "",
    ]

    for audit in audits:
        lines.extend([f"### `{audit['path']}`", ""])
        if not audit.get("exists"):
            lines.extend(["File not found.", ""])
            continue

        lines.append("**Columns:** " + ", ".join(f"`{c}`" for c in audit["columns"]))
        lines.append("")
        lines.append(
            "**Grade/geolocation candidates:** "
            + (", ".join(f"`{c}`" for c in audit["grade_or_geolocation_candidates"]) or "none")
        )
        lines.append("")
        lines.append(
            "**Payload/driver candidates:** "
            + (", ".join(f"`{c}`" for c in audit["payload_or_driver_candidates"]) or "none")
        )
        lines.append("")
        lines.append(
            "**Powertrain-state candidates:** "
            + (", ".join(f"`{c}`" for c in audit["powertrain_state_candidates"]) or "none")
        )
        lines.append("")

        split = audit["split_diagnostics"]
        if split.get("available"):
            outer = split["historical_outer_split"]
            inner = split["historical_inner_split"]
            lines.extend(
                [
                    f"- Rows: {split['rows']:,}",
                    f"- Trips: {split['trip_count']}",
                    f"- Historical outer split: {outer['training_pool_trip_count']} training-pool trips / {outer['test_trip_count']} test trips; trip IDs disjoint = {outer['trip_ids_disjoint']}; randomized = {outer['randomized']}.",
                    f"- Historical inner split: {inner['training_window_count']:,} training windows / {inner['validation_window_count']:,} validation windows.",
                    f"- Validation boundary trip: `{inner['boundary_trip']}`; boundary cuts the trip = {inner['boundary_cuts_trip']}.",
                    f"- Overlapping-window leakage risk at the training/validation boundary = {inner['overlapping_window_leakage_risk']} (up to {inner['maximum_adjacent_window_overlap_timesteps']} shared timesteps).",
                    "",
                ]
            )

    lines.extend(
        [
            "## Required corrective protocol for any new experiments",
            "",
            "- Assign complete trips to train, validation, and test sets before constructing sliding windows.",
            "- Fit every scaler on training trips only; apply the fitted transformation unchanged to validation and test trips.",
            "- Record the exact ordered trip IDs in each split and use a fixed random seed when shuffling trip IDs.",
            "- Construct windows independently within each split and trip.",
            "- Report split ratios at the trip level and the resulting sample/window counts.",
            "- Preserve the historical results as historical results; do not silently relabel them as leakage-free reruns.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    audits = [audit_dataset(path) for path in DATASETS]
    payload = {
        "window_size": WINDOW_SIZE,
        "datasets": audits,
        "code_level_findings": {
            "scaler_fit_before_split": True,
            "historical_validation_split_after_window_concatenation": True,
        },
    }
    (OUT_DIR / "revision_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = markdown_report(audits)
    (OUT_DIR / "revision_audit.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
