# Regenerated corrected-protocol prediction figures

These figures use all five validation-selected LSTM checkpoints from the
fixed-split study. The split is by complete trips (`split_seed=20260801`),
the feature scaler is fitted only on training-trip rows, and every displayed
prediction is from held-out test trips.

The mean and standard-deviation band across seeds are diagnostic visualizations.
They do not replace the canonical per-seed metrics reported in the manuscript.
Representative trips are selected by median test-trip length, independently of
observed emissions and model predictions.

| Dataset | Representative trip | Windows | Diagnostic five-seed-mean MAE | R2 |
|---|---|---:|---:|---:|
| BMW i3 | `TripB21` | 10,387 | 0.016854203 | 0.96112613 |
| Infiniti QX50 | `61907052` | 8,114 | 0.15077623 | 0.96726144 |
| Chevrolet Blazer | `61908018.0` | 13,709 | 0.13491952 | 0.7857091 |
| Chrysler Pacifica | `62001035` | 8,565 | 0.88713211 | 0.72913741 |

Generated assets:

- `figure_lstm_predictions_median_test_trips.{png,pdf}`
- `figure_lstm_parity_full_test_sets.{png,pdf}`
- `figure_lstm_training_validation_five_seeds.{png,pdf}`
- one representative-trip prediction CSV per dataset
- full compressed prediction arrays retained in the workflow artifact
