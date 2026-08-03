# Model card — corrected second-revision experiments

## Intended use

The models support in-domain prediction of instantaneous CO2-equivalent emissions and, for the BMW i3 branch, prediction of motor torque and throttle from observed context. The study is a methodological readiness assessment, not a causal EV-versus-ICEV comparison.

## Data and conditioning scope

Observed shared variables include speed, acceleration and thermal context. Grade/elevation, payload, wind, driver identity, gear position and transmission state are absent from at least one dataset. Predictions and comparisons are therefore conditioned only on the observed shared covariates.

## Architecture and training

The recurrent implementation uses an input LSTM with hidden dimension 32, four residual LSTM blocks with layer normalization and dropout, and a linear output head. Window length is 10. Training uses AdamW, MSE, batch size 512, 20 epochs, one warm-up epoch and cosine learning-rate decay. The checkpoint minimizing validation MSE is restored before testing.

## Validation protocol

Complete trips are disjoint across train, validation and test. The two-stage split reserves 20% for test and then 20% of the remainder for validation, with fractional holdout counts rounded up; the BMW i3 manifest contains 44/12/14 trips. Scaling uses training trips only, and each length-10 window predicts the following timestep.

Five fixed-split training seeds, 20260801–20260805, quantify optimization variation for the four emissions models. The EV feature-model LSTM is one pre-specified validation-selected run at training seed 20260801. Trip-level uncertainty uses 10,000 complete-trip bootstrap resamples. Non-recurrent candidates include the training-set mean, Ridge, histogram gradient boosting, random forest and MLP.

## Performance summary

Five-seed mean emissions MAE is 0.01790 g/s for BMW i3, 0.15449 g/s for QX50, 0.13757 g/s for Blazer and 0.91359 g/s for Pacifica. The selected non-recurrent emissions baselines have lower mean MAE than the corresponding five-seed LSTM means for all four datasets; no general LSTM-superiority claim is supported.

For the EV context-to-actuation task, histogram gradient boosting is selected without combining torque and throttle units because the same configuration independently minimizes both validation MAEs: 3.8308 Nm for torque and 3.6779 percentage points for throttle. Across the 14 fixed test trips, its trip-mean MAE is 4.2330 Nm for torque and 3.6218 percentage points for throttle, compared with the canonical single-run LSTM values of 4.0504 Nm and 3.7780 percentage points. The result is mixed by output and does not establish a general advantage for recurrent or non-recurrent estimators.

Replacing measured EV actuation with feature-model predictions increases mean trip-level emissions MAE from 0.0175 to 0.0273 g/s. The mean proxy penalty is 0.0098 g/s, with a 95% complete-trip bootstrap interval of [0.0071, 0.0127].

## Limitations and prohibited interpretation

Do not interpret equal measured speed and temperature as equivalent operating conditions. Do not interpret these results as a causal powertrain effect or as a completed cross-domain counterfactual comparison. Route-matched deployment requires grade, payload, wind, driver protocol and richer transmission-state measurements, together with extrapolation checks and fuller uncertainty propagation.

## Canonical reproduction

- Leakage-free protocol: [`../../src/revision_protocol.py`](../../src/revision_protocol.py)
- LSTM runner: [`../../scripts/run_revision_lstm_cpu.py`](../../scripts/run_revision_lstm_cpu.py)
- EV feature-model benchmark: [`../../scripts/run_revision_feature_model_head_to_head.py`](../../scripts/run_revision_feature_model_head_to_head.py)
- Final feature-model outputs: [`../../artifacts/revision_feature_model_head_to_head/`](../../artifacts/revision_feature_model_head_to_head/)
