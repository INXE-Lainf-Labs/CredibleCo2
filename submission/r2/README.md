# ITS-2025-10-0427.R1 second-revision submission

This directory records the editorial integration of the corrected experiments for **Credible CO2 Comparisons: A Machine Learning Approach to Vehicle Powertrain Assessment**.

## Status

- Corrected computational results: complete.
- Full 13-figure structure: restored and regenerated under the corrected protocol where applicable.
- Clean manuscript: compiled and visually audited.
- Marked manuscript: compiled and visually audited; second-round additions/corrections are blue.
- Point-by-point response: complete with final manuscript page numbers.
- Title page and CRediT statement: complete.
- Model card, reproducibility record, correction note, machine-readable tables and SHA-256 manifest: complete.
- Full-data recurrent-versus-non-recurrent EV feature-model benchmark: complete and reported in standalone Table 5.

The pull request remains **draft** and must not be merged until the authors approve the final submission package.

## Reporting decisions

The revised paper no longer claims identical operating conditions or causal isolation of a powertrain effect. The estimand is a **comparison conditioned on the observed shared covariates**. Road grade/elevation, payload, wind, driver identity, gear position and transmission state are unavailable in at least one source dataset and are disclosed as limitations.

All results use:

- a two-stage complete-trip train/validation/test split before window construction;
- approximately 64/16/20 percent train/validation/test allocation;
- split seed `20260801`;
- training seeds `20260801`-`20260805` for the emissions-model robustness analysis;
- scaler fitted only on training-trip rows;
- length-10 next-timestep windows built independently within each trip;
- validation-selected checkpoints or validation-selected baseline families;
- no test-set use in model selection;
- 10,000 complete-trip bootstrap resamples.

## Central numerical correction

The corrected EV two-stage proxy does **not** show negligible degradation or denoising:

- direct mean trip MAE: `0.0175 g/s`;
- proxy mean trip MAE: `0.0273 g/s`;
- proxy-minus-direct mean: `0.0098 g/s`;
- 95 percent bootstrap CI: `[0.0071, 0.0127]`.

All earlier negligible-degradation and denoising interpretations were removed.

## EV feature-model benchmark

The completed full-data comparison uses 691,930 training, 216,951 validation and 185,212 test windows. Standalone Table 5 reports all evaluated estimators separately. The validation-selected random forest gives held-out trip-mean MAEs of `4.0585 Nm` for torque and `3.6717 percentage points` for throttle, compared with `4.0504 Nm` and `3.7780 percentage points` for the LSTM. The LSTM is essentially tied on torque and slightly worse on throttle. Because this is a single fixed-split comparison rather than a matched multi-seed study, it is reported descriptively and does not establish superiority of either estimator family.

## Manuscript contents

The final rendering contains 19 pages, 13 figures, 9 tables and 35 references. It includes the Stage 3 readiness matrix, exact leakage-resistant split and training protocol, five-seed emissions-model robustness analysis, validation-selected non-recurrent emissions benchmarks, standalone Table 5 for the completed EV feature-model benchmark, separate torque/throttle diagnostics, the corrected six-row bootstrap table, the grade sensitivity envelope, electricity-accounting caveats and the reviewer-suggested Fischer and Alberti references.

The complete submission archive is maintained outside the Git repository because it contains DOCX, PDF and TIFF submission binaries. Repository computational assets remain under `artifacts/`, and the textual response and reproducibility documents are stored in this directory.
