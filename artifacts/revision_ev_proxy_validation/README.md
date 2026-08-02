# Corrected EV proxy validation

- Fixed complete-trip split seed: `20260801`
- Validation-selected full-data checkpoints: emissions epoch 14, feature epoch 19
- Common aligned held-out windows: 185,072
- Test set used for model selection: `false`

| Quantity | Mean across trips | Median | Sample SD | 95% bootstrap CI for mean |
|---|---:|---:|---:|---:|
| Direct MAE | 0.017526676 | 0.017198269 | 0.0046323305 | [0.015224182, 0.01986516] |
| Proxy MAE | 0.027316321 | 0.028186929 | 0.0056250213 | [0.024460825, 0.030128651] |
| Proxy - direct MAE | 0.0097896451 | 0.0076390676 | 0.0055401228 | [0.0071151926, 0.012652217] |

This is an in-domain component-composition check and not a cross-powertrain counterfactual validation.
