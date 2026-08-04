#!/usr/bin/env python3
"""Aggregate paired five-seed tuned EV feature-to-emissions compositions."""
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

EXPECTED_SEEDS = [20260801, 20260802, 20260803, 20260804, 20260805]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/tuned_ev_proxy_composition_five_seed",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def seed_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "n_seeds": int(len(array)),
        "mean": float(array.mean()),
        "sample_std": sample_std(values),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def trip_arrays(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = result["per_trip"]
    direct = np.asarray([row["direct"]["mae"] for row in rows], dtype=float)
    proxy = np.asarray([row["proxy"]["mae"] for row in rows], dtype=float)
    delta = np.asarray([row["proxy_minus_direct_mae"] for row in rows], dtype=float)
    return direct, proxy, delta


def hierarchical_bootstrap(
    results: list[dict[str, Any]],
    n_resamples: int,
    seed: int = 20260806,
) -> dict[str, Any]:
    """Paired resampling of seeds and trips within each selected seed."""
    arrays = [trip_arrays(result) for result in results]
    n_seeds = len(arrays)
    rng = np.random.default_rng(seed)
    direct_boot = np.empty(n_resamples, dtype=float)
    proxy_boot = np.empty(n_resamples, dtype=float)
    delta_boot = np.empty(n_resamples, dtype=float)

    for iteration in range(n_resamples):
        selected_seeds = rng.integers(0, n_seeds, size=n_seeds)
        direct_means: list[float] = []
        proxy_means: list[float] = []
        delta_means: list[float] = []
        for seed_index in selected_seeds:
            direct, proxy, delta = arrays[int(seed_index)]
            selected_trips = rng.integers(0, len(direct), size=len(direct))
            direct_means.append(float(direct[selected_trips].mean()))
            proxy_means.append(float(proxy[selected_trips].mean()))
            delta_means.append(float(delta[selected_trips].mean()))
        direct_boot[iteration] = np.mean(direct_means)
        proxy_boot[iteration] = np.mean(proxy_means)
        delta_boot[iteration] = np.mean(delta_means)

    def describe(values: np.ndarray) -> dict[str, Any]:
        return {
            "resamples": int(n_resamples),
            "mean_of_bootstrap_distribution": float(values.mean()),
            "bootstrap_95pct_ci": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
        }

    return {
        "method": "paired hierarchical bootstrap: resample five training-seed pairs, then resample fixed test trips within each selected pair",
        "direct_trip_mean_mae": describe(direct_boot),
        "proxy_trip_mean_mae": describe(proxy_boot),
        "proxy_minus_direct_trip_mean_mae": describe(delta_boot),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.input_dir).rglob("tuned_ev_proxy_composition.json"))
    if not paths:
        raise RuntimeError("No tuned composition JSON files found")

    by_seed: dict[int, dict[str, Any]] = {}
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        seed = int(result["training_seed"])
        if seed in by_seed:
            raise RuntimeError(f"Duplicate composition result for seed {seed}")
        by_seed[seed] = result

    observed_seeds = sorted(by_seed)
    if observed_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"Expected seeds {EXPECTED_SEEDS}, observed {observed_seeds}"
        )

    results = [by_seed[seed] for seed in EXPECTED_SEEDS]
    reference = results[0]
    for result in results[1:]:
        if result["dataset_sha256"] != reference["dataset_sha256"]:
            raise RuntimeError("Dataset hashes differ across seeds")
        if result["split_manifest"] != reference["split_manifest"]:
            raise RuntimeError("Split manifests differ across seeds")
        if result["global_window_weighted"]["n_windows"] != reference["global_window_weighted"]["n_windows"]:
            raise RuntimeError("Aligned window counts differ across seeds")
        if [row["trip_id"] for row in result["per_trip"]] != [row["trip_id"] for row in reference["per_trip"]]:
            raise RuntimeError("Test trip ordering differs across seeds")

    per_seed: list[dict[str, Any]] = []
    for seed, result in zip(EXPECTED_SEEDS, results):
        direct_trip = float(result["trip_level"]["direct_mae"]["mean"])
        proxy_trip = float(result["trip_level"]["proxy_mae"]["mean"])
        delta_trip = float(result["trip_level"]["proxy_minus_direct_mae"]["mean"])
        global_result = result["global_window_weighted"]
        per_seed.append(
            {
                "training_seed": seed,
                "direct_trip_mean_mae": direct_trip,
                "proxy_trip_mean_mae": proxy_trip,
                "proxy_minus_direct_trip_mean_mae": delta_trip,
                "direct_global_mae": float(global_result["direct"]["mae"]),
                "proxy_global_mae": float(global_result["proxy"]["mae"]),
                "proxy_minus_direct_global_mae": float(global_result["proxy_minus_direct_mae"]),
                "emissions_best_epoch": int(result["checkpoints"]["emissions"]["best_epoch"]),
                "feature_best_epoch": int(result["checkpoints"]["feature"]["best_epoch"]),
            }
        )

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in per_seed]

    aggregate = {
        "status": "completed_tuned_ev_proxy_composition_five_seed",
        "dataset": reference["dataset"],
        "dataset_sha256": reference["dataset_sha256"],
        "split_seed": reference["split_seed"],
        "training_seeds": EXPECTED_SEEDS,
        "n_test_trips": len(reference["per_trip"]),
        "n_aligned_windows_per_seed": int(reference["global_window_weighted"]["n_windows"]),
        "test_set_used_for_model_selection": False,
        "pairing_rule": "emissions and feature checkpoints with the same training seed are composed as one paired replicate",
        "trip_mean_mae_across_training_seeds": {
            "direct": seed_summary(values("direct_trip_mean_mae")),
            "proxy": seed_summary(values("proxy_trip_mean_mae")),
            "proxy_minus_direct": seed_summary(values("proxy_minus_direct_trip_mean_mae")),
        },
        "global_window_weighted_mae_across_training_seeds": {
            "direct": seed_summary(values("direct_global_mae")),
            "proxy": seed_summary(values("proxy_global_mae")),
            "proxy_minus_direct": seed_summary(values("proxy_minus_direct_global_mae")),
        },
        "hierarchical_bootstrap": hierarchical_bootstrap(
            results, args.bootstrap_resamples
        ),
        "per_seed": per_seed,
        "interpretation_guardrail": reference["interpretation_guardrail"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "tuned_ev_proxy_composition_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)

    all_trip_rows: list[dict[str, Any]] = []
    for seed, result in zip(EXPECTED_SEEDS, results):
        for row in result["per_trip"]:
            all_trip_rows.append(
                {
                    "training_seed": seed,
                    "trip_id": row["trip_id"],
                    "n_aligned_windows": row["n_aligned_windows"],
                    "direct_mae": row["direct"]["mae"],
                    "proxy_mae": row["proxy"]["mae"],
                    "proxy_minus_direct_mae": row["proxy_minus_direct_mae"],
                }
            )
    with (output_dir / "tuned_ev_proxy_composition_seed_trip_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_trip_rows[0]))
        writer.writeheader()
        writer.writerows(all_trip_rows)

    (output_dir / "tuned_ev_proxy_composition_five_seed.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    x = np.arange(len(EXPECTED_SEEDS))
    direct_values = np.asarray(values("direct_trip_mean_mae"))
    proxy_values = np.asarray(values("proxy_trip_mean_mae"))
    figure, axis = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    axis.plot(x, direct_values, marker="o", label="Measured-actuation input")
    axis.plot(x, proxy_values, marker="o", label="Predicted-actuation proxy")
    axis.set_xticks(x, [str(seed) for seed in EXPECTED_SEEDS], rotation=20)
    axis.set_xlabel("Paired training seed")
    axis.set_ylabel("Mean trip-level CO2 MAE (g/s)")
    axis.set_title("Tuned EV feature-to-emissions composition across five seeds")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(
        output_dir / "figure_tuned_ev_proxy_composition_five_seed.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "figure_tuned_ev_proxy_composition_five_seed.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    trip_summary = aggregate["trip_mean_mae_across_training_seeds"]
    bootstrap = aggregate["hierarchical_bootstrap"]
    direct = trip_summary["direct"]
    proxy = trip_summary["proxy"]
    delta = trip_summary["proxy_minus_direct"]
    delta_ci = bootstrap["proxy_minus_direct_trip_mean_mae"]["bootstrap_95pct_ci"]
    global_summary = aggregate["global_window_weighted_mae_across_training_seeds"]

    report = (
        "# Tuned EV feature-to-emissions composition: five paired seeds\n\n"
        f"Five paired training seeds were evaluated on the same {aggregate['n_test_trips']} fixed test trips. "
        "Each replicate paired emissions and feature checkpoints carrying the same training seed.\n\n"
        f"Mean trip-level MAE across seeds was {direct['mean']:.8f} ± {direct['sample_std']:.8f} g/s "
        f"with measured actuation and {proxy['mean']:.8f} ± {proxy['sample_std']:.8f} g/s "
        f"with tuned predicted actuation. The paired increase was {delta['mean']:.8f} ± "
        f"{delta['sample_std']:.8f} g/s across seeds.\n\n"
        f"The paired hierarchical 95% bootstrap interval for the increase was "
        f"[{delta_ci[0]:.8f}, {delta_ci[1]:.8f}] g/s.\n\n"
        f"Window-weighted global MAE across seeds was {global_summary['direct']['mean']:.8f} ± "
        f"{global_summary['direct']['sample_std']:.8f} g/s with measured actuation and "
        f"{global_summary['proxy']['mean']:.8f} ± {global_summary['proxy']['sample_std']:.8f} g/s "
        f"with predicted actuation; increase {global_summary['proxy_minus_direct']['mean']:.8f} ± "
        f"{global_summary['proxy_minus_direct']['sample_std']:.8f} g/s.\n\n"
        "This is an in-domain BMW i3 component-composition check. It is not evidence of causal "
        "or cross-powertrain operating-condition equivalence.\n"
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
