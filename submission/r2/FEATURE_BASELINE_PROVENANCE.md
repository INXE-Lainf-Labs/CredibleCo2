# EV feature-model non-recurrent baseline provenance

## Public input

- Repository: `https://github.com/INXE-Lainf-Labs/descarbonize.ai-inmetro`
- Commit: `15569a347d4b53eadf7b36869d793da540dc2268`
- File: `7_veiculos-propulsao-alternativa/7.2_desenvolvimento-validacao-ml/data/eletrico_ieee.csv`
- SHA-256: `8ccb1489d4d041e5688e1aa808921c1f7694fb6968521f77b619bee8cafb8262`
- Content used: 1,094,793 rows and 70 complete trips.

## Protocol audit

The canonical experiment uses a two-stage complete-trip split: 20% test and then 20% of the remainder for validation, with fractional holdouts rounded up. Each length-10 context window predicts the following timestep. The final manifest and counts are:

- 44 training, 12 validation and 14 test trips;
- 691,930 training, 216,951 validation and 185,212 test windows;
- `MinMaxScaler` fitted only on training-trip rows;
- split and bootstrap seed `20260801`;
- 10,000 complete-trip bootstrap resamples.

## Candidate grid

The benchmark evaluates the training-set mean, Ridge with `alpha` in `{0.01, 0.1, 1, 10}`, random forest with 50 trees, depth 12, minimum leaf size in `{5, 20}` and `max_features` in `{1.0, sqrt}`, histogram gradient boosting with 200 iterations and learning rate in `{0.05, 0.1}`, and MLP hidden layers `(64)` or `(128, 64)` with at most 60 iterations, batch size 512 and early stopping.

## Selection and result

Torque and throttle validation errors remain in their physical units and are not combined. Histogram gradient boosting with learning rate `0.1` is selected because one configuration independently obtains the lowest validation MAE for both torque (`3.8308 Nm`) and throttle (`3.6779 percentage points`).

On the 14 fixed test trips, it gives trip-mean MAE `4.23298 Nm` for torque and `3.62185 percentage points` for throttle, with bootstrap intervals `[3.95388, 4.55261]` and `[2.86583, 4.90192]`. The canonical single-run LSTM reference is lower for torque (`4.0504 Nm`) and higher for throttle (`3.7780 percentage points`). The result is mixed by output and does not establish general estimator superiority.

The audited environment used NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.0.
