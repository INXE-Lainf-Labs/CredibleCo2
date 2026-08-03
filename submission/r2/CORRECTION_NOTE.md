# Correction note

A full second-round reproducibility audit changed the reported experimental results.

1. The train/validation/test protocol is now complete-trip, with all windows created after splitting and the scaler fitted only to training-trip rows.
2. Validation-selected checkpoints are restored before test evaluation.
3. Five training seeds replace single-run robustness language for the emissions models.
4. Validation-selected non-recurrent benchmarks show dataset-dependent performance and do not support general LSTM superiority.
5. The prior EV proxy interpretation was incorrect. Under the corrected trip-aligned evaluation, proxy actuation increases mean trip-level CO2 MAE by `0.0098 g/s`, with 95% CI `[0.0071, 0.0127]`. All “negligible degradation” and “denoising” language was removed.
6. Figures 3–11 were regenerated under the corrected protocol; Figures 12–13 add all-test-window parity and five-seed stability diagnostics.
7. The public EV feature dataset and benchmark were executed against the canonical two-stage split, reproducing the 44/12/14 trip manifest, next-timestep target alignment and exact window counts.
8. Feature-model candidate selection is reported output-wise. Histogram gradient boosting independently minimizes both validation MAEs; no mixed-unit aggregate is used.
9. Five-seed robustness applies to emissions models only; the EV feature-model LSTM reference is the single seed-20260801 run.
10. EV battery power is defined as `max(VI, 0)`, making the exclusion of regenerative phases explicit and internally consistent.

The revised manuscript replaces “identical operating conditions” with “comparison conditioned on the observed shared covariates” and explicitly documents missing grade, payload, wind, driver and transmission variables.
