#!/usr/bin/env python3
"""Run one full-data LSTM training seed with a fixed complete-trip split.

This wrapper reuses the audited corrected-protocol LSTM implementation while
separating two sources of randomness:

- ``split_seed`` fixes the train/validation/test trip manifest;
- ``training_seed`` controls model initialization and minibatch shuffling.

The intended five-seed study fixes ``split_seed=20260801`` and runs training
seeds 20260801 through 20260805. The test set is never used for checkpoint
selection; the minimum-validation-MSE checkpoint remains canonical.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

import scripts.run_revision_lstm_cpu as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(base.DATASETS), required=True)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", default="artifacts/revision_lstm_multiseed")
    return parser.parse_args()


def fixed_split_wrapper(split_seed: int):
    original_split = base.complete_trip_split

    def fixed_split(
        frame,
        seed: int | None = None,
        test_fraction: float = 0.20,
        validation_fraction_of_remaining: float = 0.20,
    ):
        del seed
        return original_split(
            frame,
            seed=split_seed,
            test_fraction=test_fraction,
            validation_fraction_of_remaining=validation_fraction_of_remaining,
        )

    return fixed_split


def checkpoint_payload(
    result: dict[str, Any], best_state_dict: dict[str, torch.Tensor]
) -> dict[str, Any]:
    return {
        "model_state_dict": best_state_dict,
        "dataset": result["dataset"],
        "display_name": result["display_name"],
        "task": result["task"],
        "seed": result["training_seed"],
        "training_seed": result["training_seed"],
        "split_seed": result["split_seed"],
        "feature_columns": result["feature_columns"],
        "target_columns": result["target_columns"],
        "model": result["model"],
        "checkpoint_selection": result["checkpoint_selection"],
        "split_manifest": result["split_manifest"],
    }


def main() -> None:
    args = parse_args()

    # The audited base runner uses its module-level SEED for initialization,
    # minibatch shuffling, and deterministic operations. Rebind it to the
    # requested training seed, while overriding only the trip split to remain
    # fixed across all runs.
    base.SEED = args.training_seed
    base.complete_trip_split = fixed_split_wrapper(args.split_seed)

    run_args = Namespace(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
        warmup_epochs=args.warmup_epochs,
        max_train_windows=999_999_999,
        max_validation_windows=999_999_999,
        max_test_windows=999_999_999,
        feature_model=False,
        threads=args.threads,
        output_dir=args.output_dir,
    )

    result, best_state_dict = base.run(run_args)
    result["seed"] = args.training_seed
    result["training_seed"] = args.training_seed
    result["split_seed"] = args.split_seed
    result["seed_design"] = {
        "split_held_fixed": True,
        "split_seed": args.split_seed,
        "training_seed": args.training_seed,
        "varied_components": [
            "model_parameter_initialization",
            "training_minibatch_order",
        ],
        "fixed_components": [
            "complete_trip_split",
            "training_only_feature_scaler",
            "architecture",
            "optimizer_and_learning_rate_schedule",
            "epochs",
            "validation_checkpoint_policy",
            "all_train_validation_test_windows",
        ],
    }

    if result["split_manifest"]["seed"] != args.split_seed:
        raise RuntimeError("Trip split seed was not held fixed")
    if result["used_window_counts"] != result["full_window_counts"]:
        raise RuntimeError("Multiseed study must use every available window")
    if result["checkpoint_selection"]["test_set_used_for_selection"]:
        raise RuntimeError("Test set must not be used for checkpoint selection")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_emissions_seed{args.training_seed}"
    checkpoint_name = f"{stem}_best.pt"
    result["checkpoint_artifact"] = checkpoint_name

    torch.save(
        checkpoint_payload(result, best_state_dict),
        output_dir / checkpoint_name,
    )
    base.save_json(output_dir / f"{stem}.json", result)

    seed_header = (
        f"# Five-seed LSTM run: {result['display_name']}\n\n"
        f"- Fixed split seed: `{args.split_seed}`\n"
        f"- Training seed: `{args.training_seed}`\n"
        "- Varied: model initialization and minibatch order\n"
        "- Fixed: trip split, scaler protocol, architecture, hyperparameters, "
        "all windows, and validation-only checkpoint selection\n\n"
    )
    report = seed_header + base.markdown(result)
    (output_dir / f"{stem}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
