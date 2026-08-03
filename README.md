# CredibleCO2

CredibleCO2 is a machine-learning framework for vehicle-powertrain emissions assessment conditioned on the shared operating covariates that are actually observed. The final second revision does **not** treat equal speed as proof of identical operation and does not claim that a causal route-matched EV–ICEV comparison has already been executed.

## Canonical reproducibility path

The revised manuscript results use the leakage-free complete-trip protocol implemented in [`src/revision_protocol.py`](src/revision_protocol.py) and the runners under [`scripts/`](scripts/):

1. split complete trip IDs into train, validation and test before preprocessing;
2. fit each `MinMaxScaler` only on training-trip rows;
3. transform validation and test trips without refitting or clipping;
4. construct length-10 windows independently inside each trip and partition;
5. select checkpoints, model families and hyperparameters using validation trips only;
6. evaluate the selected model once on held-out test trips.

The exact split seed is `20260801`. Emissions-model robustness uses training seeds `20260801`–`20260805` with a fixed trip manifest. The EV feature-model LSTM is one pre-specified validation-selected run at seed `20260801`.

## Primary implementation and evidence

- [`src/revision_protocol.py`](src/revision_protocol.py): deterministic complete-trip splitting, training-only scaling, trip-local windows, metrics and bootstrap utilities.
- [`scripts/run_revision_lstm_cpu.py`](scripts/run_revision_lstm_cpu.py): corrected LSTM training with restoration of the minimum-validation-MSE checkpoint.
- [`scripts/run_revision_experiments.py`](scripts/run_revision_experiments.py): emissions benchmarks and input analyses.
- [`scripts/run_revision_feature_model_head_to_head.py`](scripts/run_revision_feature_model_head_to_head.py): output-specific EV feature-model benchmark without averaging incompatible torque and throttle units.
- [`artifacts/revision_feature_model_head_to_head/`](artifacts/revision_feature_model_head_to_head/): final candidate grid, selected HGB result and provenance.
- [`submission/r2/`](submission/r2/): final model card, reproducibility record, correction note and point-by-point response.

The final manuscript package contains 20 pages, 13 figures and 10 tables. The public repository stores computational and textual reproducibility assets; submission PDF, DOCX and TIFF binaries are maintained outside Git.

## Important legacy-code notice

The original notebooks [`src/LSTM_EV.ipynb`](src/LSTM_EV.ipynb), [`src/LSTM_ICEV.ipynb`](src/LSTM_ICEV.ipynb), and historical functions in [`src/helper.py`](src/helper.py) are retained for provenance only. They must not be used to reproduce the revised manuscript. Calls to the leakage-prone historical entry points now emit visible `FutureWarning` messages.

The historical workflow fitted `MinMaxScaler` before the trip split and created the inner train/validation division after concatenating overlapping windows. See [`src/README.md`](src/README.md) for the canonical/legacy distinction.

## Main corrected findings

- Five-seed emissions results do not support universal LSTM superiority over non-recurrent baselines.
- The EV feature-model benchmark selects histogram gradient boosting because one configuration independently minimizes both validation MAEs; the held-out result is mixed by output relative to the LSTM.
- Replacing measured torque and throttle with predicted actuation increases mean trip-level emissions MAE by 0.0098 g/s, with 95% bootstrap interval [0.0071, 0.0127].
- Final Stage 3 remains a readiness target requiring paired-route data and richer measurement of grade, payload, wind, driver behavior and transmission state.

## Dependencies

The project uses Python, PyTorch, NumPy, pandas and scikit-learn. General environment specifications are provided in [`environment.yaml`](environment.yaml) and [`osx_environment.yaml`](osx_environment.yaml). The final EV feature baseline was audited with NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.0.

## Contribution policy

Contributions are encouraged through pull requests and are assessed according to technical relevance and reproducibility impact.
