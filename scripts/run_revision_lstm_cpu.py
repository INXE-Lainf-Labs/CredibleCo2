#!/usr/bin/env python3
"""CPU feasibility rerun of the LSTM under the corrected trip-wise protocol.

The default configuration intentionally uses a deterministic subset and five
training epochs. It is a feasibility run, not a silent replacement for the
historical 20-epoch results. Once runtime is known, the same script can be run
with larger limits or all windows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

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
    SEED,
    SHARED_CONTEXT,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="qx50")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--max-train-windows", type=int, default=100000)
    parser.add_argument("--max-validation-windows", type=int, default=50000)
    parser.add_argument("--max-test-windows", type=int, default=100000)
    parser.add_argument("--feature-model", action="store_true")
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--output-dir", default="artifacts/revision_lstm_cpu")
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
        progress = (epoch - warmup_epochs) / denominator
        progress = min(max(progress, 0.0), 1.0)
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_sum = 0.0
    element_count = 0
    with torch.inference_mode():
        for xb, yb in loader:
            pred = model(xb)
            loss_sum += float(criterion(pred, yb).item())
            element_count += int(yb.numel())
            predictions.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    return (
        np.concatenate(targets),
        np.concatenate(predictions),
        loss_sum / max(element_count, 1),
    )


def run(args: argparse.Namespace) -> dict:
    set_global_seed(SEED)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    spec = DATASETS[args.dataset]
    frame = load_dataset(spec)
    split = complete_trip_split(frame, seed=SEED)

    if args.feature_model:
        if args.dataset != "ev":
            raise ValueError("The feature-model rerun is defined only for the EV dataset")
        feature_columns = SHARED_CONTEXT
        target_columns = ["Motor Torque [Nm]", "Throttle [%]"]
        task = "ev_feature_model"
    else:
        feature_columns = ACTUATION
        target_columns = ["CO2 Emissions"]
        task = "emissions_model"

    scaler = fit_feature_scaler(frame, split.train_trip_ids, feature_columns)

    x_train, y_train, train_trips = build_windows(
        frame, split.train_trip_ids, feature_columns, target_columns, scaler
    )
    x_validation, y_validation, validation_trips = build_windows(
        frame, split.validation_trip_ids, feature_columns, target_columns, scaler
    )
    x_test, y_test, test_trips = build_windows(
        frame, split.test_trip_ids, feature_columns, target_columns, scaler
    )

    full_counts = {
        "train": len(x_train),
        "validation": len(x_validation),
        "test": len(x_test),
    }

    x_train, y_train, train_trips = deterministic_sample(
        x_train, y_train, train_trips, args.max_train_windows, SEED
    )
    x_validation, y_validation, validation_trips = deterministic_sample(
        x_validation,
        y_validation,
        validation_trips,
        args.max_validation_windows,
        SEED + 1,
    )
    x_test, y_test, test_trips = deterministic_sample(
        x_test, y_test, test_trips, args.max_test_windows, SEED + 2
    )

    train_loader = make_loader(x_train, y_train, args.batch_size, True, SEED)
    validation_loader = make_loader(
        x_validation, y_validation, args.batch_size, False, SEED
    )
    test_loader = make_loader(x_test, y_test, args.batch_size, False, SEED)

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

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(xb)
            seen += len(xb)

        _, _, validation_mse = evaluate(model, validation_loader)
        record = {
            "epoch": epoch + 1,
            "train_mse": total / max(seen, 1),
            "validation_mse": validation_mse,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        scheduler.step()

    training_seconds = time.perf_counter() - start
    y_true, y_pred, test_mse = evaluate(model, test_loader)
    per_trip = per_trip_mae(y_true, y_pred, test_trips)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "status": "completed_cpu_feasibility_run",
        "dataset": args.dataset,
        "display_name": spec.display_name,
        "task": task,
        "device": "cpu",
        "threads": args.threads,
        "seed": SEED,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "split_manifest": split.as_dict(),
        "window_counts_by_trip": {
            "train": window_count_by_trip(frame, split.train_trip_ids),
            "validation": window_count_by_trip(frame, split.validation_trip_ids),
            "test": window_count_by_trip(frame, split.test_trip_ids),
        },
        "full_window_counts": full_counts,
        "used_window_counts": {
            "train": len(x_train),
            "validation": len(x_validation),
            "test": len(x_test),
        },
        "model": {
            "architecture": "MultipleLayerLSTM",
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "parameter_count": parameter_count,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": "AdamW",
            "base_lr": args.base_lr,
            "max_lr": args.max_lr,
            "final_lr": args.final_lr,
            "warmup_epochs": args.warmup_epochs,
        },
        "history": history,
        "test": {
            "mse": test_mse,
            **regression_metrics(y_true, y_pred),
            "per_trip_mae": per_trip,
            "trip_level_mae_summary": summarize_trip_metric(per_trip.values(), SEED),
        },
        "runtime_seconds": training_seconds,
        "interpretation_guardrail": (
            "This CPU run uses a deterministic subset when a maximum-window limit is set. "
            "It validates the corrected implementation and estimates runtime; it must not be "
            "reported as a full-data replacement without an explicit sensitivity comparison."
        ),
    }
    return result


def markdown(result: dict) -> str:
    test = result["test"]
    counts = result["used_window_counts"]
    model = result["model"]
    lines = [
        f"# CPU LSTM corrected-protocol rerun: {result['display_name']}",
        "",
        f"- Task: `{result['task']}`",
        f"- Features: {', '.join(result['feature_columns'])}",
        f"- Targets: {', '.join(result['target_columns'])}",
        f"- Complete trips: {result['split_manifest']['trip_counts']}",
        f"- Used windows: train={counts['train']:,}, validation={counts['validation']:,}, test={counts['test']:,}",
        f"- Architecture: {model['num_blocks']} residual LSTM blocks, hidden dimension {model['hidden_dim']}, {model['parameter_count']:,} parameters",
        f"- Training: {model['epochs']} epochs, batch size {model['batch_size']}, CPU runtime {result['runtime_seconds']:.1f} s",
        "",
        "## Test metrics",
        "",
        f"- MSE: {test['mse']:.8g}",
        f"- MAE: {test['mae']:.8g}",
        f"- RMSE: {test['rmse']:.8g}",
        f"- R2: {test['r2']:.8g}",
        f"- Trip-level MAE summary: `{json.dumps(test['trip_level_mae_summary'])}`",
        "",
        "## Guardrail",
        "",
        result["interpretation_guardrail"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = run(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "feature" if args.feature_model else "emissions"
    stem = f"{args.dataset}_{suffix}"
    save_json(out_dir / f"{stem}.json", result)
    (out_dir / f"{stem}.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
