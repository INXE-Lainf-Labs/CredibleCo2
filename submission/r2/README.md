# ITS-2025-10-0427.R1 final second-revision record

This directory records the final editorial and computational integration for **Credible CO2 Comparisons: A Machine Learning Approach to Vehicle Powertrain Assessment**.

## Status

- Corrected computational results: complete.
- Leakage-free protocol and canonical runners: merged into `main`.
- Historical helper warnings: merged into `main`.
- Clean and marked manuscripts: compiled and visually audited.
- Point-by-point response: synchronized with the final 20-page manuscript.
- Model card, reproducibility record, correction note and machine-readable feature baseline: synchronized.
- Final manuscript structure: 20 pages, 13 figures, 10 tables and 35 references.
- Funding statement: Research Development Foundation (FUNDEP), MOVER Program, grant `29271.01.01/2023.03-0`.

The complete submission archive is maintained outside Git because it contains PDF, DOCX and TIFF binaries. This directory contains the corresponding textual and reproducibility record.

## Canonical protocol

All reported experiments use:

- complete-trip train/validation/test splitting before window construction;
- two-stage allocation: 20% test, then 20% of the remainder for validation, with fractional holdouts rounded up;
- split seed `20260801`;
- emissions training seeds `20260801`–`20260805` with a fixed manifest;
- `MinMaxScaler` fitted only on training-trip rows;
- length-10 windows predicting the following timestep and built independently within each trip;
- validation-selected checkpoints or validation-selected non-recurrent models;
- no test-set use in model or hyperparameter selection;
- 10,000 complete-trip bootstrap resamples.

## Central corrected results

The trip-aligned EV proxy evaluation gives:

- direct mean trip MAE: `0.0175 g/s`;
- proxy mean trip MAE: `0.0273 g/s`;
- mean proxy penalty: `0.0098 g/s`;
- 95% trip-bootstrap interval: `[0.0071, 0.0127]`.

The EV feature-model benchmark uses 691,930 training, 216,951 validation and 185,212 test windows. Torque and throttle validation errors are kept separate. Histogram gradient boosting is selected because one configuration independently minimizes both validation MAEs: `3.8308 Nm` and `3.6779 percentage points`. On the fixed 14-trip test set it gives `4.2330 Nm` and `3.6218 percentage points`, compared with the canonical single-run LSTM values `4.0504 Nm` and `3.7780 percentage points`. The result is mixed by output and does not establish general estimator superiority.

## Interpretation boundary

The paper no longer claims identical operating conditions or causal isolation of a powertrain effect. The estimand is a comparison conditioned on observed shared covariates. Grade/elevation, payload, wind, driver identity, gear position and transmission state are unavailable in at least one source dataset and remain requirements for future route-matched Stage 3 deployment.
