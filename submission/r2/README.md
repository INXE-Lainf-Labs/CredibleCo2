# ITS-2025-10-0427.R1 second-revision submission

This directory records the editorial integration of the corrected experiments for **Credible CO2 Comparisons: A Machine Learning Approach to Vehicle Powertrain Assessment**.

## Status

- Corrected computational results: complete.
- Full 11-figure structure: restored; Figures 3-11 regenerated under the corrected protocol.
- Clean manuscript: compiled and visually audited.
- Marked manuscript: compiled and visually audited; second-round additions/corrections are blue.
- Point-by-point response: complete with final manuscript page numbers.
- Title page and CRediT statement: complete.
- Model card, reproducibility record, correction note, machine-readable tables and SHA-256 manifest: complete.

The pull request remains **draft** and must not be merged until the authors approve the final submission package.

## Reporting decisions

The revised paper no longer claims identical operating conditions or causal isolation of a powertrain effect. The estimand is a **comparison conditioned on the observed shared covariates**. Road grade/elevation, payload, wind, driver identity, gear position and transmission state are unavailable in at least one source dataset and are disclosed as limitations.

All results use:

- complete-trip train/validation/test splitting before window construction;
- approximately 64/16/20 percent train/validation/test allocation;
- split seed `20260801`;
- training seeds `20260801`-`20260805`;
- scaler fitted only on training-trip rows;
- length-10 windows built independently within each trip;
- validation-selected checkpoints restored before testing;
- no test-set use in model selection;
- 10,000 complete-trip bootstrap resamples.

## Central numerical correction

The corrected EV two-stage proxy does **not** show negligible degradation or denoising:

- direct mean trip MAE: `0.0175 g/s`;
- proxy mean trip MAE: `0.0273 g/s`;
- proxy-minus-direct mean: `0.0098 g/s`;
- 95 percent bootstrap CI: `[0.0071, 0.0127]`.

All earlier negligible-degradation and denoising interpretations were removed.

## Manuscript contents

The final compact rendering contains 10 pages, 11 figures and 8 tables. It includes the Stage 3 readiness matrix, exact leakage-resistant split and training protocol, five-seed robustness analysis, validation-selected non-recurrent benchmarks, separate torque/throttle diagnostics, the corrected six-row bootstrap table, electricity-accounting caveats and the reviewer-suggested Fischer and Alberti references.

The complete submission archive is maintained outside the Git repository because it contains DOCX, PDF and TIFF submission binaries. Repository computational assets remain under `artifacts/`, and the textual response and reproducibility documents are stored in this directory.