# Model card — corrected second-revision experiments

## Intended use

The models support in-domain prediction of instantaneous CO2-equivalent emissions and, for the BMW i3 branch, prediction of torque and throttle from observed context. The study is a methodological readiness assessment, not a causal EV-versus-ICEV comparison.

## Conditioning scope

Observed shared variables include speed, acceleration and thermal context. Grade/elevation, payload, wind, driver identity, gear position and transmission state are absent from at least one dataset. Comparisons are conditioned only on the observed shared covariates.

## Architecture and training

The recurrent implementation uses a single-layer LSTM, a ReLU fully connected layer and a linear output head. Window length is 10. Training uses Adam, MSE, batch size 512, 20 epochs and cosine learning-rate scheduling. The checkpoint minimizing validation MSE is restored before testing.

## Validation

Complete trips are disjoint across train, validation and test. Scaling uses training trips only. Five fixed-split training seeds quantify optimization variation. Trip-level uncertainty uses 10,000 bootstrap resamples. Simpler validation-selected benchmarks include Ridge, histogram gradient boosting, random forest and MLP.

## Performance summary

Five-seed mean MAE: BMW i3 `0.01790465 g/s`; QX50 `0.15449141 g/s`; Blazer `0.13757026 g/s`; Pacifica `0.91359455 g/s`. Selected non-recurrent baselines have lower MAE than the five-seed LSTM mean for all four datasets. No general LSTM-superiority claim is supported.

## Limitations

Equal measured speed and temperature must not be interpreted as equivalent operating conditions. Results must not be interpreted as a causal powertrain effect or completed cross-domain counterfactual. Route-matched deployment requires grade, payload, wind, driver protocol and richer transmission-state measurements, together with extrapolation checks and uncertainty propagation.
