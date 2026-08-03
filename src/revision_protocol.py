from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SEED = 20260801
WINDOW_SIZE = 10


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: str
    kind: str
    display_name: str


DATASETS: dict[str, DatasetSpec] = {
    "ev": DatasetSpec("ev", "data/eletrico_ieee.csv", "EV", "BMW i3"),
    "qx50": DatasetSpec("qx50", "data/Dataset_QX50_trip-clean.csv", "ICEV", "Infiniti QX50"),
    "blazer": DatasetSpec("blazer", "data/Dataset_Blazer.csv", "ICEV", "Chevrolet Blazer"),
    "pacifica": DatasetSpec("pacifica", "data/Dataset_Pacifica.csv", "ICEV", "Chrysler Pacifica"),
}

CANONICAL_RENAME = {
    "Time[s]": "Time [s]",
    "Velocity[km/h]": "Velocity [km/h]",
    "Throttle[%]": "Throttle [%]",
    "Motor Torque[Nm]": "Motor Torque [Nm]",
    "CO2 Emissions[g/s]": "CO2 Emissions",
    "Ambient Temperature[°C]": "Ambient Temperature [°C]",
    "Cabin Temperature Sensor[°C]": "Cabin Temperature Sensor [°C]",
    "Heat Exchanger Temperature[°C]": "Heat Exchanger Temperature [°C]",
    "Longitudinal Acceleration[m/s^2]": "Longitudinal Acceleration [m/s^2]",
}

SPEED = ["Velocity [km/h]"]
SHARED_CONTEXT = [
    "Velocity [km/h]",
    "Ambient Temperature [°C]",
    "Cabin Temperature Sensor [°C]",
    "Longitudinal Acceleration [m/s^2]",
]
ACTUATION = ["Velocity [km/h]", "Throttle [%]", "Motor Torque [Nm]"]
ALL_OBSERVED = [
    "Velocity [km/h]",
    "Ambient Temperature [°C]",
    "Cabin Temperature Sensor [°C]",
    "Longitudinal Acceleration [m/s^2]",
    "Throttle [%]",
    "Motor Torque [Nm]",
]


@dataclass
class SplitManifest:
    seed: int
    train_trip_ids: list[str]
    validation_trip_ids: list[str]
    test_trip_ids: list[str]

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "split_rule": "shuffle complete trip IDs, then allocate approximately 64/16/20 percent to train/validation/test",
            "train_trip_ids": self.train_trip_ids,
            "validation_trip_ids": self.validation_trip_ids,
            "test_trip_ids": self.test_trip_ids,
            "trip_counts": {
                "train": len(self.train_trip_ids),
                "validation": len(self.validation_trip_ids),
                "test": len(self.test_trip_ids),
            },
        }


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.path, low_memory=False).rename(columns=CANONICAL_RENAME)
    required = sorted(set(ALL_OBSERVED + ["CO2 Emissions", "Trip"]))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{spec.key}: missing required columns: {missing}")

    frame = frame[required].copy()
    frame = frame.dropna(subset=required).reset_index(drop=True)

    # Preserve the data treatment used by the historical notebook: two Blazer
    # sessions were explicitly discarded as noisy before model fitting.
    if spec.key == "blazer":
        trip_order = frame["Trip"].drop_duplicates().tolist()
        noisy = [trip_order[i] for i in (0, 8) if i < len(trip_order)]
        frame = frame.loc[~frame["Trip"].isin(noisy)].reset_index(drop=True)

    frame["Trip"] = frame["Trip"].astype(str)
    for column in required:
        if column != "Trip":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).reset_index(drop=True)
    return frame


def complete_trip_split(
    frame: pd.DataFrame,
    seed: int = SEED,
    test_fraction: float = 0.20,
    validation_fraction_of_remaining: float = 0.20,
) -> SplitManifest:
    trips = frame["Trip"].drop_duplicates().astype(str).to_numpy(copy=True)
    if len(trips) < 3:
        raise ValueError("At least three trips are required for train/validation/test splitting")

    rng = np.random.default_rng(seed)
    rng.shuffle(trips)

    n_test = max(1, int(np.ceil(test_fraction * len(trips))))
    n_remaining = len(trips) - n_test
    n_validation = max(1, int(np.ceil(validation_fraction_of_remaining * n_remaining)))
    n_train = len(trips) - n_validation - n_test
    if n_train < 1:
        raise ValueError("Split leaves no training trips")

    train = trips[:n_train].tolist()
    validation = trips[n_train : n_train + n_validation].tolist()
    test = trips[n_train + n_validation :].tolist()
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    return SplitManifest(seed, train, validation, test)


def subset_by_trips(frame: pd.DataFrame, trip_ids: Sequence[str]) -> pd.DataFrame:
    return frame.loc[frame["Trip"].isin(set(map(str, trip_ids)))].copy()


def fit_feature_scaler(
    frame: pd.DataFrame, train_trip_ids: Sequence[str], feature_columns: Sequence[str]
) -> MinMaxScaler:
    train_rows = subset_by_trips(frame, train_trip_ids)
    scaler = MinMaxScaler()
    scaler.fit(train_rows[list(feature_columns)].to_numpy(dtype=np.float64))
    return scaler


def transform_features(
    frame: pd.DataFrame, scaler: MinMaxScaler, feature_columns: Sequence[str]
) -> pd.DataFrame:
    result = frame[["Trip"]].copy()
    transformed = scaler.transform(frame[list(feature_columns)].to_numpy(dtype=np.float64))
    for index, column in enumerate(feature_columns):
        result[column] = transformed[:, index].astype(np.float32)
    return result


def _trip_windows(
    x: np.ndarray, y: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= window_size:
        return (
            np.empty((0, window_size, x.shape[1]), dtype=np.float32),
            np.empty((0,) + y.shape[1:], dtype=np.float32),
        )
    windows = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=0)
    windows = windows[:-1].transpose(0, 2, 1).copy().astype(np.float32)
    targets = y[window_size:].copy().astype(np.float32)
    return windows, targets


def build_windows(
    frame: pd.DataFrame,
    trip_ids: Sequence[str],
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    scaler: MinMaxScaler,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    window_trips: list[np.ndarray] = []

    for trip in trip_ids:
        trip_frame = frame.loc[frame["Trip"] == str(trip)]
        if trip_frame.empty:
            continue
        x = scaler.transform(
            trip_frame[list(feature_columns)].to_numpy(dtype=np.float64)
        ).astype(np.float32)
        y = trip_frame[list(target_columns)].to_numpy(dtype=np.float32)
        x_windows, y_targets = _trip_windows(x, y, window_size)
        if len(x_windows) == 0:
            continue
        frames.append(x_windows)
        targets.append(y_targets)
        window_trips.append(np.repeat(str(trip), len(x_windows)))

    if not frames:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty((0, len(target_columns)), dtype=np.float32),
            np.empty((0,), dtype=str),
        )
    return np.concatenate(frames), np.concatenate(targets), np.concatenate(window_trips)


def deterministic_sample(
    x: np.ndarray,
    y: np.ndarray,
    trips: np.ndarray,
    maximum: int | None,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if maximum is None or len(x) <= maximum:
        return x, y, trips
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(x), size=maximum, replace=False))
    return x[selected], y[selected], trips[selected]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def per_trip_mae(
    y_true: np.ndarray, y_pred: np.ndarray, trip_ids: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred).reshape(len(y_pred), -1)
    result: dict[str, float] = {}
    for trip in pd.unique(trip_ids):
        mask = trip_ids == trip
        result[str(trip)] = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
    return result


def summarize_trip_metric(values: Iterable[float], seed: int = SEED) -> dict[str, object]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {}
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return {
        "n_trips": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "iqr": [float(np.quantile(array, 0.25)), float(np.quantile(array, 0.75))],
        "bootstrap_95pct_mean_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def window_count_by_trip(frame: pd.DataFrame, trip_ids: Sequence[str], window_size: int = WINDOW_SIZE) -> dict[str, int]:
    counts = frame.groupby("Trip", sort=False).size()
    return {str(t): max(int(counts.get(str(t), 0)) - window_size, 0) for t in trip_ids}


def save_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
