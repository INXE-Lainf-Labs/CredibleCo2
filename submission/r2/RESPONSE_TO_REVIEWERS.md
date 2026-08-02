# Response to decision letter — second revision

**Manuscript:** Credible CO2 Comparisons: A Machine Learning Approach to Vehicle Powertrain Assessment  
**Manuscript ID:** ITS-2025-10-0427.R1

The complete point-by-point DOCX/PDF response is included in the final submission package. The principal responses are recorded here for provenance.

## Reviewer 3

### Road grade

The source-data audit confirmed that road grade/elevation and GPS are unavailable in the released files used for the corrected experiments. A grade ablation cannot therefore be performed without inventing an unverified variable. The manuscript now states this limitation and specifies calibrated GNSS/elevation or road-map measurements for the future paired-route experiment.

The new grade calculation is explicitly described as a **first-order scale comparison**, not an attribution of observed residuals and not a guaranteed lower bound. At 50 km/h, under the stated efficiency assumptions, a sustained grade of approximately 0.87 percent would change the BMW i3 operational CO2 rate by an amount equal to its held-out MAE. This shows that ordinary route-grade differences can matter for the context-to-actuation model and a future paired-route comparison; it does not establish that grade caused the observed EV residuals, because measured torque and throttle may already encode part of the load response in the direct emissions pathway.

For the Chrysler Pacifica, the equivalent threshold is approximately 1.17 percent. Because its data were generated on a chassis dynamometer with level-road road-load coefficients, this number is presented only as a cross-domain scale reference and not as an explanation of its prediction error.

The manuscript now states that modest sustained grade differences can produce shifts comparable to or larger than reported prediction errors depending on platform, speed, duration, actual mass, instantaneous efficiency and route profile. Unmatched grade therefore creates a systematic route-dependent component rather than symmetric measurement noise.

Reference masses are disclosed as non-homogeneous source conventions: ANL curb weights for Blazer and Pacifica, EPA equivalent test weight for QX50 and BMW DIN unladen mass for the i3. The reproducibility material reports mass plus or minus 10 percent and efficiency sensitivity ranges. EPA, DOE/AFDC and BMW official sources were added for the gasoline factor, lower heating value, efficiency assumptions and i3 mass.

### Equivalent operating conditions

The paper no longer treats equal speed and measured temperature as equivalent operation and no longer claims to isolate a causal technology effect. The estimand is now a **comparison conditioned on the observed shared covariates**. Grade, payload, wind, driver actions, gear position and transmission state are explicitly listed as unobserved. Final cross-domain Stage 3 comparison is not executed or claimed as validated.

## Reviewer 4

### Stage 3 readiness

A readiness matrix separates validated emissions models and EV feature/proxy components from the missing ICEV feature model, unmeasured covariates, route-matched external validation and causal-identification requirements.

### Reproducibility

All results were regenerated using complete-trip splitting before windowing, approximately 64/16/20 percent train/validation/test allocation, split seed `20260801`, five training seeds, training-only scaling, length-10 within-trip windows and validation-selected checkpoints. The test set is not used for model selection.

### Benchmark models and LSTM justification

Ridge, histogram gradient boosting, random forest and MLP candidates were compared using the same corrected trip manifests and validation-only model selection. The evidence is dataset dependent and does not support general LSTM superiority.

The LSTM is now justified only as the pre-specified reference implementation used consistently across tasks and because the context-to-actuation mapping is temporally structured. A common implementation facilitates training and audit, but using the same estimator family is neither required nor presumed optimal. Estimator selection should be performed separately for each dataset and task.

A direct recurrent-versus-non-recurrent comparison for the two-output EV feature model was not completed in the submitted manuscript and remains an open empirical question. A full-data runner and GitHub Actions workflow have been added to execute that comparison under the same corrected protocol.

### Missing ICEV feature model

The manuscript explains that gear position, engine speed, direct pedal/throttle, transmission state and torque-converter lock-up would make the ICEV context-to-actuation inverse problem more identifiable and permit symmetric feature-model validation.

### Emission accounting

The 38.5 gCO2/kWh factor is identified as an attributional annual-average baseline. Time-varying, regional and marginal factors may change absolute values and rankings. Excluding regenerative-braking credits can conservatively overstate net grid-attributed emissions.

### Literature

The suggested Fischer et al. (2025) and Alberti et al. (2024) studies were added to support the role of route-specific operation and powertrain/transmission configuration.

## Numerical correction

The corrected trip-aligned EV calculation gives mean direct MAE `0.0175 g/s`, mean proxy MAE `0.0273 g/s`, and proxy-minus-direct `0.0098 g/s` with 95 percent bootstrap CI `[0.0071, 0.0127]`. All negligible-degradation and denoising language was removed. The current manuscript contains 19 pages, 13 figures, 8 tables and 35 references. Section 4.5 begins on page 15, Table 5 is on page 16, Section 4.7 begins on page 16, the Conclusion spans pages 16–17 and Table 8 is on page 18.
