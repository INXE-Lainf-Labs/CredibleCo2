# Validation-only LSTM tuning protocol

## Purpose

The original manuscript comparison uses a **fixed LSTM reference configuration**. It does not constitute an exhaustive comparison of the full LSTM model family. This addendum defines a leakage-free sensitivity experiment to determine whether the reference LSTM was materially under-tuned.

## Non-negotiable safeguards

1. Keep the existing complete-trip train/validation/test manifest fixed with split seed `20260801`.
2. Fit all input and target scalers using training trips only.
3. Construct windows independently within each trip and partition.
4. Use validation trips only for checkpoint, architecture, lookback, clipping and target-scaling selection.
5. Do not evaluate test trips during the search stages.
6. After one configuration has been selected for each task, retrain it with training seeds `20260801`–`20260805` and evaluate the unchanged test trips once per seed.
7. Report torque and throttle separately; never average their raw-unit errors.

## Search stages

### Stage A — training stabilization

Starting from the reference configuration (`hidden_dim=32`, four residual blocks, window 10):

- extend training from 20 to 60 epochs;
- restore the checkpoint minimizing the validation criterion over all 60 epochs;
- compare gradient clipping disabled versus maximum global norm `1.0`;
- for the two-output EV feature model, compare raw targets with training-only z-score standardization of torque and throttle.

For the feature model, select using the mean of the two output MSEs after dividing each residual by the corresponding training-target standard deviation. This criterion is dimensionless and does not average incompatible physical units.

### Stage B — recurrent architecture

Holding the Stage A choices fixed, evaluate:

- residual LSTM blocks: `1`, `2`, `4`;
- hidden dimension: `16`, `32`, `64`.

This gives nine architecture candidates per task. Ties are resolved in favor of fewer trainable parameters.

### Stage C — temporal lookback

Holding the selected Stage B architecture fixed, evaluate windows of:

- `10` samples;
- `30` samples;
- `60` samples.

The best validation configuration becomes the final tuned candidate for that task.

### Stage D — five-seed confirmation

Retrain only the validation-selected configuration with five optimization seeds while preserving the same trip split. Report mean and sample standard deviation of test MAE, RMSE and R². For the feature task, report torque and throttle separately.

## Tasks

The protocol is applied independently to:

- BMW i3 emissions;
- Infiniti QX50 emissions;
- Chevrolet Blazer emissions;
- Chrysler Pacifica emissions;
- BMW i3 context-to-actuation feature prediction.

A different LSTM configuration may be selected for each dataset/task, exactly as a different non-recurrent baseline family was selected by validation for each dataset.

## Interpretation

The experiment can establish whether a validation-selected LSTM improves on the originally reported reference configuration. It cannot establish that LSTMs are universally superior on unseen vehicle datasets, because the same finite collection of vehicle datasets and a fixed trip split are used.

## Manuscript wording before tuned results are available

> The recurrent results correspond to a fixed LSTM reference configuration rather than an exhaustive optimization of the LSTM family. The comparison therefore tests whether that pre-specified temporal reference outperforms validation-selected non-recurrent baselines under the same trip-wise protocol.

## Manuscript wording after completion

> A separate validation-only sensitivity analysis evaluated training duration, gradient clipping, recurrent depth, hidden dimension, lookback length and, for the two-output feature model, training-only target standardization. The held-out test trips were not evaluated during configuration search. Only the validation-selected configuration was retrained with five optimization seeds and evaluated on the fixed test split.
