#!/usr/bin/env python3
"""Validation-only LSTM tuning and final five-seed evaluation.

Candidate stages never construct or evaluate test windows. The fixed test split is
used only by the final stage after a configuration has been selected exclusively
from validation trips.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from src.models.LSTM import MultipleLayerLSTM
from src.revision_protocol import (
    ACTUATION,
    DATASETS,
    SHARED_CONTEXT,
    build_windows,
    complete_trip_split,
    deterministic_sample,
    fit_feature_scaler,
    load_dataset,
    save_json,
    set_global_seed,
)

TARGET_NAMES = ["Motor Torque [Nm]", "Throttle [%]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "refine", "full", "final"], required=True)
    parser.add_argument("--task", choices=["emissions", "feature"], required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--training-seed", type=int, default=20260801)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--target-mode", choices=["raw", "zscore"], default="raw")
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--max-train-windows", type=int, default=999_999_999)
    parser.add_argument("--max-validation-windows", type=int, default=999_999_999)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--output-dir", default="artifacts/lstm_validation_tuning")
    return parser.parse_args()


def scheduler_for(
    optimizer: AdamW,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    max_lr: float,
    final_lr: float,
) -> LambdaLR:
    warmup_epochs = min(max(warmup_epochs, 0), max(total_epochs - 1, 0))

    def factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            alpha = epoch / warmup_epochs
            return (base_lr + alpha * (max_lr - base_lr)) / base_lr
        denominator = max(total_epochs - warmup_epochs - 1, 1)
        progress = min(max((epoch - warmup_epochs) / denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = final_lr + (max_lr - final_lr) * cosine
        return lr / base_lr

    return LambdaLR(optimizer, factor)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def transform_targets(y: np.ndarray, mode: str, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if mode == "zscore":
        return ((y - mean) / std).astype(np.float32)
    return y.astype(np.float32)


def inverse_targets(y: np.ndarray, mode: str, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if mode == "zscore":
        return y * std + mean
    return y


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    target_mode: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    model.eval()
    true_raw: list[np.ndarray] = []
    pred_raw: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, yb_model in loader:
            pred_model = model(xb).cpu().numpy()
            y_model = yb_model.cpu().numpy()
            true_raw.append(inverse_targets(y_model, target_mode, target_mean, target_std))
            pred_raw.append(inverse_targets(pred_model, target_mode, target_mean, target_std))
    y_true = np.concatenate(true_raw).astype(np.float64)
    y_pred = np.concatenate(pred_raw).astype(np.float64)
    standardized_error = (y_pred - y_true) / target_std
    score = float(np.mean(standardized_error**2))
    output_mae = {
        str(index): float(np.mean(np.abs(y_pred[:, index] - y_true[:, index])))
        for index in range(y_true.shape[1])
    }
    return y_true, y_pred, score, output_mae


def per_trip_output_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trip_ids: np.ndarray,
    target_names: list[str],
) -> dict[str, Any]:
    per_output: dict[str, dict[str, float]] = {name: {} for name in target_names}
    for trip in np.unique(trip_ids):
        mask = trip_ids == trip
        for index, name in enumerate(target_names):
            value = float(np.mean(np.abs(y_pred[mask, index] - y_true[mask, index])))
            per_output[name][str(trip)] = value
    summary: dict[str, Any] = {}
    for name, values in per_output.items():
        array = np.asarray(list(values.values()), dtype=np.float64)
        summary[name] = {
            "per_trip": values,
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }
    return summary


def config_slug(args: argparse.Namespace) -> str:
    clip = "off" if args.gradient_clip <= 0 else str(args.gradient_clip).replace(".", "p")
    return (
        f"{args.stage}_{args.task}_{args.dataset}_w{args.window_size}_h{args.hidden_dim}"
        f"_b{args.num_blocks}_clip{clip}_{args.target_mode}_seed{args.training_seed}"
    )


def main() -> None:
    args = parse_args()
    if args.task == "feature" and args.dataset != "ev":
        raise ValueError("The feature task is defined only for the EV dataset")
    if args.task == "emissions" and args.target_mode != "raw":
        raise ValueError("Emissions tuning uses raw targets; z-score mode is reserved for the two-output feature task")
    if args.gradient_clip < 0:
        raise ValueError("gradient clip must be non-negative")
    if args.stage != "final" and args.training_seed != 20260801:
        raise ValueError("Candidate stages must use the fixed screening seed 20260801")

    set_global_seed(args.training_seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    spec = DATASETS[args.dataset]
    frame = load_dataset(spec)
    split = complete_trip_split(frame, seed=args.split_seed)

    if args.task == "feature":
        feature_columns = SHARED_CONTEXT
        target_columns = TARGET_NAMES
    else:
        feature_columns = ACTUATION
        target_columns = ["CO2 Emissions"]

    feature_scaler = fit_feature_scaler(frame, split.train_trip_ids, feature_columns)
    x_train, y_train_raw, train_trips = build_windows(
        frame,
        split.train_trip_ids,
        feature_columns,
        target_columns,
        feature_scaler,
        window_size=args.window_size,
    )
    x_validation, y_validation_raw, validation_trips = build_windows(
        frame,
        split.validation_trip_ids,
        feature_columns,
        target_columns,
        feature_scaler,
        window_size=args.window_size,
    )

    full_counts = {"train": int(len(x_train)), "validation": int(len(x_validation))}
    x_train, y_train_raw, train_trips = deterministic_sample(
        x_train,
        y_train_raw,
        train_trips,
        args.max_train_windows,
        args.split_seed + 100,
    )
    x_validation, y_validation_raw, validation_trips = deterministic_sample(
        x_validation,
        y_validation_raw,
        validation_trips,
        args.max_validation_windows,
        args.split_seed + 101,
    )
    used_counts = {"train": int(len(x_train)), "validation": int(len(x_validation))}

    target_mean = y_train_raw.astype(np.float64).mean(axis=0)
    target_std = y_train_raw.astype(np.float64).std(axis=0)
    target_std = np.where(target_std > 1e-12, target_std, 1.0)
    y_train = transform_targets(y_train_raw, args.target_mode, target_mean, target_std)
    y_validation = transform_targets(y_validation_raw, args.target_mode, target_mean, target_std)

    train_loader = make_loader(x_train, y_train, args.batch_size, True, args.training_seed)
    validation_loader = make_loader(
        x_validation, y_validation, args.batch_size, False, args.training_seed
    )

    model = MultipleLayerLSTM(
        input_size=len(feature_columns),
        hidden_dim=args.hidden_dim,
        output_size=len(target_columns),
        num_blocks=args.num_blocks,
    ).cpu()
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.base_lr)
    scheduler = scheduler_for(
        optimizer,
        args.warmup_epochs,
        args.epochs,
        args.base_lr,
        args.max_lr,
        args.final_lr,
    )

    best_epoch = 0
    best_validation_score = math.inf
    best_validation_output_mae: dict[str, float] = {}
    best_state_dict: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        max_observed_grad_norm = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if args.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.gradient_clip
                )
                max_observed_grad_norm = max(max_observed_grad_norm, float(grad_norm))
            optimizer.step()
            loss_sum += float(loss.item()) * len(xb)
            seen += len(xb)

        _, _, validation_score, validation_output_mae = evaluate(
            model,
            validation_loader,
            args.target_mode,
            target_mean,
            target_std,
        )
        if validation_score < best_validation_score:
            best_validation_score = validation_score
            best_validation_output_mae = validation_output_mae
            best_epoch = epoch + 1
            best_state_dict = clone_state_dict(model)

        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(seen, 1),
            "validation_standardized_mse": validation_score,
            "validation_output_mae": validation_output_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "max_preclip_gradient_norm": max_observed_grad_norm,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        scheduler.step()

    if best_state_dict is None:
        raise RuntimeError("No validation checkpoint was selected")
    runtime_seconds = time.perf_counter() - start

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    payload: dict[str, Any] = {
        "record_type": "lstm_tuning_run",
        "stage": args.stage,
        "task": args.task,
        "dataset": args.dataset,
        "display_name": spec.display_name,
        "split_seed": args.split_seed,
        "training_seed": args.training_seed,
        "test_set_used_for_selection": False,
        "test_evaluated": args.stage == "final",
        "split_manifest": split.as_dict(),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "target_preprocessing": {
            "mode": args.target_mode,
            "fit_partition": "training windows only",
            "mean": target_mean.tolist(),
            "std": target_std.tolist(),
        },
        "selection_metric": "mean squared error after division of each output error by its training-only target standard deviation",
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "window_size": args.window_size,
            "gradient_clip_global_norm": args.gradient_clip,
            "target_mode": args.target_mode,
            "optimizer": "AdamW",
            "base_lr": args.base_lr,
            "max_lr": args.max_lr,
            "final_lr": args.final_lr,
            "warmup_epochs": args.warmup_epochs,
            "parameter_count": parameter_count,
        },
        "full_window_counts": full_counts,
        "used_window_counts": used_counts,
        "checkpoint_selection": {
            "policy": "minimum_validation_standardized_mse",
            "best_epoch": best_epoch,
            "best_validation_score": best_validation_score,
            "best_validation_output_mae": best_validation_output_mae,
            "restored_before_test": args.stage == "final",
        },
        "history": history,
        "runtime_seconds": runtime_seconds,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = config_slug(args)

    if args.stage == "final":
        model.load_state_dict(best_state_dict)
        x_test, y_test_raw, test_trips = build_windows(
            frame,
            split.test_trip_ids,
            feature_columns,
            target_columns,
            feature_scaler,
            window_size=args.window_size,
        )
        y_test = transform_targets(y_test_raw, args.target_mode, target_mean, target_std)
        test_loader = make_loader(x_test, y_test, args.batch_size, False, args.training_seed)
        y_true, y_pred, test_score, test_output_mae = evaluate(
            model,
            test_loader,
            args.target_mode,
            target_mean,
            target_std,
        )
        output_names = target_columns
        payload["test"] = {
            "n_windows": int(len(x_test)),
            "standardized_mse": test_score,
            "global_output_mae": {
                output_names[index]: value for index, value in enumerate(test_output_mae.values())
            },
            "trip_level_output_mae": per_trip_output_mae(
                y_true, y_pred, test_trips, output_names
            ),
        }
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "payload": {key: value for key, value in payload.items() if key != "history"},
                "feature_scaler_min": feature_scaler.min_,
                "feature_scaler_scale": feature_scaler.scale_,
                "target_mean": target_mean,
                "target_std": target_std,
            },
            output_dir / f"{slug}.pt",
        )

    save_json(output_dir / f"{slug}.json", payload)
    print(json.dumps({
        "slug": slug,
        "best_epoch": best_epoch,
        "best_validation_score": best_validation_score,
        "test_evaluated": args.stage == "final",
    }, indent=2))


if __name__ == "__main__":
    main()
