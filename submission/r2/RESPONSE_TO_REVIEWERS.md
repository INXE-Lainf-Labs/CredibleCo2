# Response to decision letter — second revision

**Manuscript:** Credible CO2 Comparisons: A Machine Learning Approach to Vehicle Powertrain Assessment  
**Manuscript ID:** ITS-2025-10-0427.R1

The complete point-by-point DOCX/PDF response is included in the final submission package. The principal responses are recorded here for provenance.

## Reviewer 3

### Road grade

The source-data audit confirmed that road grade/elevation and GPS are unavailable in the released files used for the corrected experiments. A grade ablation cannot therefore be performed without inventing an unverified variable. The manuscript now states this limitation and specifies calibrated GNSS/elevation or road-map measurements for the future paired-route experiment.

### Equivalent operating conditions

The paper no longer treats equal speed and measured temperature as equivalent operation and no longer claims to isolate a causal technology effect. The estimand is now a **comparison conditioned on the observed shared covariates**. Grade, payload, wind, driver actions, gear position and transmission state are explicitly listed as unobserved. Final cross-domain Stage 3 comparison is not executed or claimed as validated.

## Reviewer 4

### Stage 3 readiness

A readiness matrix separates validated emissions models and EV feature/proxy components from the missing ICEV feature model, unmeasured covariates, route-matched external validation and causal-identification requirements.

### Reproducibility

All results were regenerated using complete-trip splitting before windowing, approximately 64/16/20 percent train/validation/test allocation, split seed `20260801`, five training seeds, training-only scaling, length-10 within-trip windows and validation-selected checkpoints. The test set is not used for model selection.

### Benchmark models

Ridge, histogram gradient boosting, random forest and MLP candidates were compared using the same corrected trip manifests and validation-only model selection. The evidence is dataset dependent and does not support general LSTM superiority.

### Missing ICEV feature model

The manuscript explains that gear position, engine speed, direct pedal/throttle, transmission state and torque-converter lock-up would make the ICEV context-to-actuation inverse problem more identifiable and permit symmetric feature-model validation.

### Emission accounting

The 38.5 gCO2/kWh factor is identified as an attributional annual-average baseline. Time-varying, regional and marginal factors may change absolute values and rankings. Excluding regenerative-braking credits can conservatively overstate net grid-attributed emissions.

### Literature

The suggested Fischer et al. (2025) and Alberti et al. (2024) studies were added to support the role of route-specific operation and powertrain/transmission configuration.

## Numerical correction

The corrected trip-aligned EV calculation gives mean direct MAE `0.0175 g/s`, mean proxy MAE `0.0273 g/s`, and proxy-minus-direct `0.0098 g/s` with 95 percent bootstrap CI `[0.0071, 0.0127]`. All negligible-degradation and denoising language was removed. Figures 3–11 and the complete tables were regenerated under the corrected protocol.
