#!/usr/bin/env python3
"""Recompute EV electricity-factor sensitivity on the corrected held-out trips.

The calculation integrates the measured instantaneous CO2 rate over the
recorded trip time using the trapezoidal rule.  The held-out trip IDs are
obtained from the same complete-trip split used by the corrected experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.revision_protocol import CANONICAL_RENAME, DATASETS, SEED, complete_trip_split, load_dataset

BASELINE_PHI = 38.5
SCENARIOS = (("-10%", 0.90), ("Nominal", 1.00), ("+10%", 1.10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/revision_ev_electricity_sensitivity")
    return parser.parse_args()


def find_column(columns: list[str], preferred: str, alternatives: tuple[str, ...]) -> str:
    if preferred in columns:
        return preferred
    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in columns}
    for candidate in alternatives:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Could not identify {preferred!r}; available columns: {columns}")


def integrate_trip(group: pd.DataFrame, time_col: str, emission_col: str) -> float:
    data = group[[time_col, emission_col]].copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[emission_col] = pd.to_numeric(data[emission_col], errors="coerce")
    data = data.dropna().sort_values(time_col, kind="stable")
    if len(data) < 2:
        return 0.0
    time = data[time_col].to_numpy(dtype=np.float64)
    rate = data[emission_col].to_numpy(dtype=np.float64)
    keep = np.concatenate(([True], np.diff(time) > 0))
    time = time[keep]
    rate = rate[keep]
    if len(time) < 2:
        return 0.0
    return float(np.trapezoid(rate, time))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned = load_dataset(DATASETS["ev"])
    split = complete_trip_split(cleaned, seed=SEED)
    test_ids = [str(x) for x in split.test_trip_ids]

    raw = pd.read_csv(DATASETS["ev"].path, low_memory=False).rename(columns=CANONICAL_RENAME)
    columns = list(raw.columns)
    trip_col = find_column(columns, "Trip", ("trip", "trip_id", "tripid"))
    time_col = find_column(columns, "Time [s]", ("Time[s]", "time", "timestamp", "time_s"))
    emission_col = find_column(columns, "CO2 Emissions", ("CO2 Emissions[g/s]", "co2emissions", "co2_gps"))
    raw[trip_col] = raw[trip_col].astype(str)

    all_trip_totals = {
        str(trip): integrate_trip(group, time_col, emission_col)
        for trip, group in raw.groupby(trip_col, sort=False)
    }
    missing = [trip for trip in test_ids if trip not in all_trip_totals]
    if missing:
        raise RuntimeError(f"Corrected test trips missing from raw EV data: {missing}")

    nominal = np.asarray([all_trip_totals[trip] for trip in test_ids], dtype=np.float64)
    if len(nominal) != 14:
        raise RuntimeError(f"Expected 14 held-out EV trips, found {len(nominal)}")
    if not np.all(np.isfinite(nominal)) or np.any(nominal <= 0):
        raise RuntimeError(f"Invalid held-out trip totals: {nominal}")

    per_trip_rows = []
    for trip, total in zip(test_ids, nominal):
        row = {"trip_id": trip, "nominal_total_gco2": float(total)}
        for label, multiplier in SCENARIOS:
            row[f"total_{label.replace('%','pct').replace('+','plus').replace('-','minus').lower()}_gco2"] = float(total * multiplier)
        per_trip_rows.append(row)
    pd.DataFrame(per_trip_rows).to_csv(output_dir / "ev_test_trip_totals.csv", index=False)

    scenario_rows = []
    for label, multiplier in SCENARIOS:
        values = nominal * multiplier
        scenario_rows.append(
            {
                "scenario": label,
                "phi_gco2_per_kwh": BASELINE_PHI * multiplier,
                "multiplier": multiplier,
                "n_test_trips": len(values),
                "aggregate_total_gco2": float(values.sum()),
                "mean_trip_total_gco2": float(values.mean()),
                "median_trip_total_gco2": float(np.median(values)),
                "q1_trip_total_gco2": float(np.quantile(values, 0.25)),
                "q3_trip_total_gco2": float(np.quantile(values, 0.75)),
            }
        )
    pd.DataFrame(scenario_rows).to_csv(output_dir / "electricity_factor_sensitivity_test_trips.csv", index=False)

    full_total = float(sum(all_trip_totals.values()))
    manifest = {
        "status": "completed_ev_test_trip_sensitivity",
        "split_seed": SEED,
        "test_trip_ids": test_ids,
        "n_test_trips": len(test_ids),
        "integration_method": "trapezoidal integration of measured instantaneous CO2 rate over recorded Time [s] within each complete trip",
        "source_columns": {"trip": trip_col, "time": time_col, "emission_rate": emission_col},
        "baseline_phi_gco2_per_kwh": BASELINE_PHI,
        "scenario_multipliers": {label: multiplier for label, multiplier in SCENARIOS},
        "full_dataset_trip_count": len(all_trip_totals),
        "full_dataset_nominal_total_gco2_recomputed": full_total,
        "previous_revision_full_dataset_total_gco2": 13166.22,
        "previous_total_difference_gco2": full_total - 13166.22,
        "test_trip_nominal_aggregate_total_gco2": float(nominal.sum()),
        "test_trip_nominal_mean_total_gco2": float(nominal.mean()),
        "scenario_summary": scenario_rows,
        "per_trip_nominal_totals_gco2": {trip: float(total) for trip, total in zip(test_ids, nominal)},
    }
    (output_dir / "sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
