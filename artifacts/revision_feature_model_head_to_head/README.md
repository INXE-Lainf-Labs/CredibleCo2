# EV feature-model non-recurrent baseline provenance

## Public input

- Repository: `https://github.com/INXE-Lainf-Labs/descarbonize.ai-inmetro`
- Commit: `15569a347d4b53eadf7b36869d793da540dc2268`
- File: `7_veiculos-propulsao-alternativa/7.2_desenvolvimento-validacao-ml/data/eletrico_ieee.csv`
- SHA-256: `8ccb1489d4d041e5688e1aa808921c1f7694fb6968521f77b619bee8cafb8262`
- Content used: 1,094,793 rows and 70 complete trips, with no missing values in the four context inputs, two actuation targets or trip ID.

## Canonical protocol

The benchmark uses the two-stage complete-trip split: 20% test and then 20% of the remainder for validation, with fractional holdouts rounded up. It reproduces exactly:

- 44 training, 12 validation and 14 test trips;
- 691,930 training, 216,951 validation and 185,212 test windows;
- length-10 context windows predicting the following timestep;
- a `MinMaxScaler` fitted only on training-trip rows;
- split and bootstrap seed `20260801`;
- 10,000 complete-trip bootstrap resamples.

## Candidate grid and resource guard

All eligible windows are retained. The random-forest grid uses 50 trees, maximum depth 12, `min_samples_leaf` in `{5, 20}`, and `max_features` in `{1.0, sqrt}`. This bounds tree complexity without changing the split or sample count. Ridge, histogram gradient boosting, MLP and the training-set mean are also evaluated.

## Selection and held-out result

Torque and throttle validation errors remain in their physical units and are not combined. Histogram gradient boosting with `max_iter=200`, `learning_rate=0.1` and `random_state=20260801` is selected because one fitted configuration independently obtains the lowest validation MAE for both torque (`3.8308 Nm`) and throttle (`3.6779 percentage points`).

On the 14 fixed test trips:

| Output | Baseline trip-mean MAE | 95% trip-bootstrap interval | LSTM reference | Lower MAE |
|---|---:|---:|---:|---|
| Motor torque | 4.23298 Nm | [3.95388, 4.55261] | 4.05040 Nm | LSTM |
| Throttle | 3.62185 percentage points | [2.86583, 4.90192] | 3.77800 percentage points | HGB |

The LSTM reference is one pre-specified validation-selected run at training seed `20260801`; it is not a five-seed feature-model average. The result is mixed by output and does not establish general recurrent or non-recurrent superiority.

## Environment

- NumPy 2.4.6
- pandas 3.0.5
- scikit-learn 1.9.0

Machine-readable outputs are `ev_feature_model.csv`, `ev_feature_model_selection.csv` and `ev_feature_model.json`.
