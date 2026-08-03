# Model card — corrected second-revision experiments

## Intended use

The models support in-domain prediction of instantaneous CO2-equivalent emissions and, for the BMW i3 branch, prediction of torque and throttle from observed context. The study is a methodological readiness assessment, not a causal EV-versus-ICEV comparison.

## Conditioning scope

Observed shared variables include speed, acceleration and thermal context. Grade/elevation, payload, wind, driver identity, gear position and transmission state are absent from at least one dataset. Comparisons are conditioned only on the observed shared covariates.

## Architecture and training

The recurrent implementation uses a single-layer LSTM, a ReLU fully connected layer and a linear output head. Window length is 10. Training uses Adam, MSE, batch size 512, 20 epochs and cosine learning-rate scheduling. The checkpoint minimizing validation MSE is restored before testing.

## Validation

Complete trips are disjoint across train, validation and test. Scaling uses training trips only. Five fixed-split training seeds quantify optimization variation for the emissions models. Trip-level uncertainty uses 10,000 bootstrap resamples. Simpler validation-selected benchmarks include Ridge, histogram gradient boosting, random forest and MLP for both the emissions tasks and the EV feature model.

## Performance summary

Five-seed emissions-model mean MAE: BMW i3 `0.01790465 g/s`; QX50 `0.15449141 g/s`; Blazer `0.13757026 g/s`; Pacifica `0.91359455 g/s`. Selected non-recurrent baselines have lower MAE than the five-seed LSTM mean for all four datasets. No general LSTM-superiority claim is supported.

For the two-output EV feature model, a full-data comparison uses 691,930 training, 216,951 validation and 185,212 test windows. The validation-selected random forest gives held-out trip-mean MAEs of `4.0585 Nm` for torque and `3.6717 percentage points` for throttle, compared with `4.0504 Nm` and `3.7780 percentage points` for the LSTM. The LSTM is therefore essentially tied on torque and slightly worse on throttle. This is a descriptive fixed-split comparison; matched multi-seed feature-model evaluation was not performed.

## Limitations

Equal measured speed and temperature must not be interpreted as equivalent operating conditions. Results must not be interpreted as a causal powertrain effect or completed cross-domain counterfactual. Route-matched deployment requires grade, payload, wind, driver protocol and richer transmission-state measurements, together with extrapolation checks and uncertainty propagation.
