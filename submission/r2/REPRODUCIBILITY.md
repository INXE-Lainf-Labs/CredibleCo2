# Reproducibility record

## Canonical protocol

- Complete-trip split performed before window construction.
- Split seed: `20260801`.
- Allocation: 20% test, then 20% of the remaining trips for validation, with fractional holdouts rounded up.
- BMW i3 split: 44 train / 12 validation / 14 test trips.
- Window length: 10.
- Target: the timestep immediately following each length-10 context window.
- Scaler: `MinMaxScaler` fitted only on training-trip rows.
- Emissions-model LSTM training seeds: `20260801`–`20260805`.
- EV feature-model LSTM reference: one pre-specified validation-selected run at seed `20260801`; no five-seed feature-model rerun was performed.
- Sequence-model training: 20 epochs, batch size 512, AdamW, MSE, one warm-up epoch and cosine learning-rate decay.
- Model selection: minimum validation MSE; selected checkpoint restored before test evaluation.
- Bootstrap: 10,000 complete-trip resamples, percentile interval for the mean.
- Test set used for selection: no.

## Window counts — train / validation / test

- BMW i3: 691,930 / 216,951 / 185,212
- Infiniti QX50: 227,341 / 74,676 / 74,790
- Chevrolet Blazer: 36,573 / 8,441 / 30,997
- Chrysler Pacifica: 103,882 / 42,619 / 37,303

## Architecture

The canonical sequence model is `MultipleLayerLSTM`: input LSTM with hidden dimension 32, four residual LSTM blocks with layer normalization and dropout, and a linear output head. The output dimension is one for emissions and two for EV torque/throttle.

## Corrected audit findings

The historical inner split was performed after overlapping-window concatenation, allowing adjacent training and validation windows to share timesteps. The historical scaler was also fitted before the trip split. All second-revision results were regenerated after removing both leakage paths.

## Feature-model benchmark

The non-recurrent EV feature benchmark keeps torque and throttle validation errors separate. Histogram gradient boosting independently minimizes both validation MAEs and is evaluated once on the 14 fixed test trips. The machine-readable outputs are stored under `artifacts/revision_feature_model_head_to_head/`.

## Figure selection

- ICEV best/worst panels: minimum/maximum mean held-out-trip MAE across the five fixed-split emissions training seeds.
- EV diagnostic trip: median eligible window count with trip-ID tie break, selected independently of targets and predictions.

## Public source-data provenance for the EV feature baseline

- Repository: `https://github.com/INXE-Lainf-Labs/descarbonize.ai-inmetro`
- Commit: `15569a347d4b53eadf7b36869d793da540dc2268`
- Path: `7_veiculos-propulsao-alternativa/7.2_desenvolvimento-validacao-ml/data/eletrico_ieee.csv`
- SHA-256: `8ccb1489d4d041e5688e1aa808921c1f7694fb6968521f77b619bee8cafb8262`
- Rows: 1,094,793; complete trips: 70.

## Final manuscript record

- Pages: 20
- Figures: 13
- Tables: 10
- References: 35
- Funding: Research Development Foundation (FUNDEP), MOVER Program, grant `29271.01.01/2023.03-0`.
