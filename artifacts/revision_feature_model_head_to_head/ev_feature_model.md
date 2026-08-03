# Full-data EV feature-model benchmark

All candidates use the corrected 44/12/14 complete-trip manifest, training-only scaling and every eligible length-10 window. Torque and throttle validation MAEs remain separate; no mixed-unit aggregate is used.

- Windows: train=691,930, validation=216,951, test=185,212
- Selected family: `hist_gradient_boosting`
- Selected parameters: `{"max_iter": 200, "learning_rate": 0.1, "random_state": 20260801}`
- Validation torque MAE: 3.830848 Nm
- Validation throttle MAE: 3.677871 percentage points

## Held-out trip means

- Torque: 4.232977 Nm, 95% CI [3.953879, 4.552607]
- Throttle: 3.621846 percentage points, 95% CI [2.865828, 4.901915]

Canonical single-run LSTM references are 4.0504 Nm for torque and 3.7780 percentage points for throttle. The comparison is mixed by output and descriptive; the feature-model LSTM was not rerun across five training seeds.
