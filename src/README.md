# Source-code status

## Canonical revised protocol

Use [`revision_protocol.py`](revision_protocol.py) together with the runners in [`../scripts/`](../scripts/) to reproduce the revised manuscript.

The corrected order of operations is:

1. split complete trip IDs into train, validation, and test sets;
2. fit `MinMaxScaler` only on training-trip rows;
3. transform validation and test rows using the already fitted scaler;
4. create length-10 windows independently within each trip and partition;
5. select models and checkpoints using validation trips only;
6. evaluate once on the held-out test trips.

The relevant functions are:

- `complete_trip_split`;
- `fit_feature_scaler`;
- `build_windows`.

## Legacy files retained for provenance

The following files implement the original, pre-revision workflow and are retained only to document project history:

- [`helper.py`](helper.py);
- [`LSTM_EV.ipynb`](LSTM_EV.ipynb);
- [`LSTM_ICEV.ipynb`](LSTM_ICEV.ipynb).

Do **not** use their historical `normalize(...) -> time_series_dataset_split(...) -> train_model(...)` sequence to reproduce the revised manuscript. In that workflow, the scaler can be fitted before the trip split, and the inner train/validation split can be made after overlapping windows have been concatenated. The revised results do not use that pipeline.

## Min-max scaling in the revised experiments

For each task, `fit_feature_scaler(frame, train_trip_ids, feature_columns)` extracts only training-trip rows and calls `MinMaxScaler.fit(...)` on those rows. `build_windows(...)` subsequently calls only `scaler.transform(...)` for train, validation, and test trips. Held-out values may consequently fall below 0 or above 1; they are not clipped or used to refit the scaler.
