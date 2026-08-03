# Response to decision letter — second revision

**Manuscript:** Credible CO2 Comparisons: A Machine Learning Approach to Vehicle Powertrain Assessment  
**Manuscript ID:** ITS-2025-10-0427.R1  
**Revision round:** Second revision

We thank the Editor, Associate Editor and Reviewers for the careful assessment. The comments exposed a mismatch between the wording of the previous revision and the variables actually observed in the data. We therefore revised both the estimand and the experiments. The paper no longer claims identical operating conditions or a causal technology effect.

A reproducibility audit also identified two leakage paths in the historical pipeline: the `MinMaxScaler` was fitted before the trip split, and the inner training/validation split was made after overlapping windows had been concatenated. Every reported numerical result and every experimental result figure was regenerated under a leakage-resistant complete-trip protocol.

Page references below correspond to the final clean manuscript: 20 pages, 13 figures and 10 tables.

## Summary of major revisions

- Replaced “identical operating conditions” and causal language with “comparison conditioned on the observed shared covariates”.
- Explicitly disclosed missing grade/elevation, payload, wind, driver identity, gear position and transmission state.
- Regenerated all experiments with complete-trip splitting before windowing, training-only scaling, split seed `20260801`, next-timestep targets and validation-selected checkpoints.
- Used five training seeds, `20260801`–`20260805`, for the four emissions models; the EV feature-model LSTM remains one pre-specified validation-selected run at seed `20260801`.
- Added validation-selected training-mean, Ridge, histogram-gradient-boosting, random-forest and MLP benchmarks and removed all general LSTM-superiority claims.
- Added a Stage 3 readiness matrix separating validated components from missing data, models and external validation.
- Corrected the EV proxy interpretation: mean trip-level MAE rises from `0.0175` to `0.0273 g/s`; the mean increase is `0.0098 g/s`, with 95% complete-trip bootstrap interval `[0.0071, 0.0127]`.
- Added a quantitative grade sensitivity envelope, updated electricity-accounting caveats, CRediT contributions, Conflict of Interest and Funding Information.
- Added a dedicated EV feature-model benchmark and an explicit baseline-definition/hyperparameter table.

## Reviewer 3

### Road grade

We agree that grade is important. The revised manuscript now distinguishes two cases. The ICEV records are chassis-dynamometer tests, so grade is absent by construction under level-road road-load coefficients. The EV records are real-world driving data, so grade is physically present but unrecorded. Its omission is therefore most direct for the context-to-actuation model and any future paired-route comparison.

A new first-order grade sensitivity envelope quantifies the scale of the omitted term using added tractive power `m g v sin(theta)`. Table 7 reports the grade whose tractive contribution alone would shift the instantaneous CO2-equivalent rate by an amount equal to each platform’s held-out MAE at 50 km/h. Under the stated assumptions, the BMW i3 threshold is about `0.87%`; the corresponding Pacifica scale reference is about `1.17%`. These values are sensitivity scales, not attributions of observed residuals and not guaranteed lower bounds.

The central-assumption conversion ratio between ICEV and EV tractive energy is approximately 23, with the reported efficiency range giving about 18–31. This supports the reviewer’s concern that unmatched grade can generate systematic route-dependent differences rather than symmetric measurement noise. Grade instrumentation is now listed as a precondition for final Stage 3 deployment.

**Where addressed:** Introduction, page 2; Section 3.1, pages 5–7; Section 4.5 and Table 7, pages 15–17; readiness matrix, Table 10, page 19; Conclusion, page 18.

### Equal speed does not imply equivalent operating conditions

We accept this criticism and changed the estimand rather than defending the previous assumption. The study is now framed as prediction and component validation conditioned on observed shared covariates. Equal speed and measured temperature are not treated as evidence of equivalent operation, and no causal powertrain effect is claimed.

Final Stage 3 is explicitly unexecuted. A route-matched comparison requires measured or controlled grade, payload, wind and driver protocol, plus richer ICEV transmission and powertrain state, overlap/extrapolation checks and external paired-route validation.

**Where addressed:** Abstract and Introduction, pages 1–2; System Model, pages 3–5; Experimental Settings, pages 7–9; Section 4.7 and Table 10, pages 18–19; Conclusion, page 18.

## Reviewer 4

### Stage 3 readiness

The revised readiness matrix separates: domain-specific emissions models; the EV context-to-actuation model; in-domain EV proxy composition; the missing ICEV feature model; missing route covariates; overlap/extrapolation checks; and route-matched external validation. The final cross-powertrain comparison is not claimed as completed or validated.

### Reproducibility and leakage control

All results use a two-stage complete-trip split performed before window construction: 20% test, then 20% of the remainder for validation, with fractional holdouts rounded up. For BMW i3 this gives 44/12/14 trips and 691,930/216,951/185,212 train/validation/test windows. The scaler is fitted only on training-trip rows. Each length-10 window predicts the following timestep. The test set is excluded from checkpoint, model-family and hyperparameter selection.

The four emissions models use five fixed-split training seeds. At every seed, the minimum-validation-MSE checkpoint is restored before the primary test evaluation. Trip-level uncertainty uses 10,000 complete-trip bootstrap resamples.

**Where addressed:** Section 3.3, page 7; Section 4.3 and Figure 13, pages 13–15; Table 1, page 10; Data Availability, page 19.

### Benchmark models and LSTM justification

The non-recurrent search comprises the training-set mean, Ridge, random forest, histogram gradient boosting and MLP. All candidates use the same corrected trip manifests, training-fitted scaler, observed inputs and eligible windows. Selection uses validation data only.

For the EV feature task, each length-10 × 4 context window is flattened to 40 inputs and predicts next-timestep motor torque and throttle. Torque and throttle validation MAEs remain in their physical units and are not averaged into a mixed-unit selection score. Histogram gradient boosting is selected because one configuration independently minimizes both validation MAEs: `3.8308 Nm` for torque and `3.6779 percentage points` for throttle.

On the 14 fixed test trips, histogram gradient boosting gives trip-mean MAE `4.2330 Nm` for torque and `3.6218 percentage points` for throttle. The canonical single-run LSTM reference gives `4.0504 Nm` and `3.7780 percentage points`. The LSTM is lower for torque, while HGB is lower for throttle. The result is mixed by output and does not establish a general recurrence advantage.

For emissions, selected non-recurrent baselines also demonstrate dataset-dependent performance; no universal LSTM-superiority claim remains. The LSTM is retained as the pre-specified reference architecture and as a reasonable temporal model, not as an estimator presumed optimal for every dataset, task or output.

**Where addressed:** Section 3.3, page 7; Section 4.4, pages 13–16; Tables 4–6, page 16; Conclusion, page 18.

### Missing ICEV feature model

The manuscript now explains that gear position, engine speed, direct pedal/throttle position, transmission state and torque-converter lock-up would make the ICEV context-to-actuation inverse problem substantially more identifiable. With these variables, a symmetric ICEV feature model could be trained and validated using the same complete-trip protocol.

**Where addressed:** Section 3.4, pages 7–9; Section 4.7 and Table 10, pages 18–19; Conclusion, page 18.

### Emission-accounting assumptions

The `38.5 gCO2/kWh` electricity factor is identified as an attributional annual-average baseline. Regional, time-varying or marginal factors can change absolute EV values and potentially cross-powertrain rankings.

Battery-discharge power is implemented as `max(VI, 0)`. Negative-current regenerative phases receive zero rather than a negative credit or absolute-value conversion. Excluding recovered-energy credit is therefore conservative relative to net-energy accounting. The electricity-factor sensitivity was recalculated on the 14 corrected held-out BMW i3 trips.

**Where addressed:** Section 2, pages 4–5; Section 3.2, pages 5–7; Section 4.6 and Tables 8–9, pages 17–18; Conclusion, page 18.

### Recent related work

The suggested Fischer et al. (2025) and Alberti et al. (2024) studies were added and discussed in the Introduction to support the role of route-specific operation and powertrain/transmission configuration.

**Where addressed:** Introduction, page 2; References, pages 19–20.

## Additional numerical and editorial correction

The previous revision’s “negligible degradation” and “mild denoising” interpretation of the EV proxy was not reproduced under the corrected trip-aligned pipeline. The revised result is a positive, resolved penalty: direct mean trip MAE `0.0175 g/s`, proxy mean `0.0273 g/s`, and mean difference `0.0098 g/s` with 95% bootstrap interval `[0.0071, 0.0127]`. The proxy step is therefore reported as a source of intermediate error, not a denoising mechanism.

Figures 1–2 retain the methodological pipeline diagrams. Figures 3–11 replace the reviewer-seen experimental analyses with their corrected-protocol recomputations. Figures 12–13 add all-test-window parity and five-seed stability diagnostics. No result is duplicated under two protocols.

The three previous per-vehicle ICEV trip tables were consolidated into Table 2. Table 5 presents the EV feature-model comparison, Table 6 defines the baseline families and released candidate grid, and Tables 7–10 report grade sensitivity, trip-level uncertainty, electricity sensitivity and readiness. The final manuscript contains 20 pages, 13 figures, 10 tables and 35 references.

## Funding Information

This work was supported by the Research Development Foundation (FUNDEP) through the MOVER Program, grant `29271.01.01/2023.03-0`.

**Sincerely,**  
**The authors**
