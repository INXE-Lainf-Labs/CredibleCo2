# Corrected-protocol results: full-data LSTMs and exact head-to-head benchmarks

Canonical LSTM run: GitHub Actions `Revision LSTM CPU` #10 (`30725904091`), commit `14cb004b1f8870f59075103ab33bb6b365bbbc67`.

Exact EV/QX50 benchmark run: GitHub Actions `Revision full-data head-to-head` #2 (`30727389646`), commit `05ff4f11a68122edf6efb8ce7c774047a312ac1c`.

## Audit status

- All four LSTM dataset jobs completed successfully, including the EV torque/throttle feature model.
- Every LSTM ran for 20 epochs with batch size 512 and used all available windows.
- Complete trip IDs are disjoint across training, validation, and test.
- Feature scalers are fitted only on training-trip rows; windows are created independently inside each trip after the split.
- The primary LSTM test result restores the checkpoint with minimum validation MSE. The test set is not used for checkpoint selection.
- A valid PyTorch checkpoint and SHA-256 digest were verified for each LSTM.
- The exact EV and QX50 non-recurrent benchmarks use the identical trip manifests and complete train/validation/test window sets used by the corresponding LSTMs.
- Non-recurrent model choice is made by minimum validation MAE. The artifacts explicitly record `test_set_used_for_selection: false`.

## Canonical LSTM results

| Dataset / task | Windows train / val / test | Best epoch | Best val MSE | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|---:|---:|---:|
| BMW i3 — emissions | 691,930 / 216,951 / 185,212 | 14 | 0.0011289948 | 0.016999224 | 0.030245076 | 0.955440 |
| Infiniti QX50 — emissions | 227,341 / 74,676 / 74,790 | 19 | 0.053031886 | 0.15130749 | 0.25643459 | 0.967146 |
| Chevrolet Blazer — emissions | 36,573 / 8,441 / 30,997 | 17 | 0.059771687 | 0.13710105 | 0.23035044 | 0.784379 |
| Chrysler Pacifica — emissions | 103,882 / 42,619 / 37,303 | 2 | 0.96187795 | 0.92454867 | 1.4117892 | 0.717932 |
| BMW i3 — torque/throttle | 691,930 / 216,951 / 185,212 | 19 | 44.260652 | 4.1758483 | 7.6642369 | 0.928868 |

## Effect of restoring the validation-selected checkpoint

| Dataset / task | Selected MAE | Last-epoch MAE | Selected RMSE | Last-epoch RMSE | Selected R2 | Last-epoch R2 |
|---|---:|---:|---:|---:|---:|---:|
| BMW i3 — emissions | 0.016999224 | 0.018080286 | 0.030245076 | 0.035057248 | 0.955440 | 0.940132 |
| Infiniti QX50 — emissions | 0.15130749 | 0.15236630 | 0.25643459 | 0.25733075 | 0.967146 | 0.966916 |
| Chevrolet Blazer — emissions | 0.13710105 | 0.14013464 | 0.23035044 | 0.23336242 | 0.784379 | 0.778703 |
| Chrysler Pacifica — emissions | 0.92454867 | 1.4726813 | 1.4117892 | 2.4717857 | 0.717932 | 0.135360 |
| BMW i3 — torque/throttle | 4.1758483 | 4.1454332 | 7.6642369 | 7.6313938 | 0.928868 | 0.929476 |

The Pacifica result confirms the need for checkpoint restoration: the validation-selected epoch-2 model gives R2 0.717932, whereas the epoch-20 model gives R2 0.135360. For the EV feature model, epoch 19 remains canonical because it minimizes validation MSE, even though epoch 20 is marginally better on the test set.

## Exact full-data EV and QX50 head-to-head

Both exact benchmarks use velocity, throttle, and motor torque, matching the emissions LSTM inputs. They use every available window and exactly the same complete-trip split as the LSTM.

### BMW i3

Windows: 691,930 train / 216,951 validation / 185,212 test.

| Model | Validation MAE | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|---:|
| Training mean | 0.094612068 | 0.097688090 | 0.14331541 | -0.000520 |
| Ridge | 0.071863793 | 0.072637266 | 0.10748436 | 0.437230 |
| Histogram gradient boosting | 0.021400455 | 0.018177079 | 0.035018648 | 0.940264 |
| Random forest | 0.020807186 | 0.017436869 | 0.034952413 | 0.940489 |
| **MLP — selected by validation MAE** | **0.020659138** | **0.017710154** | **0.033168917** | **0.946408** |
| LSTM validation-selected checkpoint | — | 0.016999224 | 0.030245076 | 0.955440 |

The LSTM is numerically better than the validation-selected MLP on all three aggregate test metrics. Its MAE is approximately 4.0% lower. The trip-level bootstrap intervals overlap substantially: LSTM mean MAE 0.017579 [0.015274, 0.019918] and MLP mean MAE 0.018420 [0.015714, 0.021045]. This is a descriptive difference, not evidence of statistically established superiority.

### Infiniti QX50

Windows: 227,341 train / 74,676 validation / 74,790 test.

| Model | Validation MAE | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|---:|
| Training mean | 1.4270941 | 1.3112796 | 1.5630509 | -0.220615 |
| Ridge | 0.36434833 | 0.26663110 | 0.38700717 | 0.925171 |
| Histogram gradient boosting | 0.12639428 | 0.14144617 | 0.23527117 | 0.972345 |
| **Random forest — selected by validation MAE** | **0.11398306** | **0.12850532** | **0.22372522** | **0.974993** |
| MLP | 0.12722143 | 0.14925553 | 0.25100869 | 0.968522 |
| LSTM validation-selected checkpoint | — | 0.15130749 | 0.25643459 | 0.967146 |

The validation-selected random forest is numerically better than the LSTM on all three aggregate test metrics. Its MAE is approximately 15.1% lower. The trip-level bootstrap intervals again overlap substantially: random-forest mean MAE 0.164045 [0.109123, 0.231796] and LSTM mean MAE 0.179810 [0.128539, 0.234275]. This does not establish a statistically resolved difference.

## Exact actuation-input comparison across all four vehicles

For Blazer and Pacifica, the broader benchmark run already used every available window because their datasets were below the configured caps. EV and QX50 are replaced here by the exact uncapped results.

| Dataset | LSTM MAE | Validation-selected benchmark | Benchmark MAE | Numerically lower MAE |
|---|---:|---|---:|---|
| BMW i3 | 0.016999224 | MLP | 0.017710154 | LSTM |
| Infiniti QX50 | 0.15130749 | Random forest | 0.12850532 | Random forest |
| Chevrolet Blazer | 0.13710105 | Random forest | 0.11400737 | Random forest |
| Chrysler Pacifica | 0.92454867 | Ridge | 0.56747335 | Ridge |

The corrected evidence does not support a general claim that the LSTM is superior to conventional regressors. It supports reporting the LSTM as one predictive model under the corrected protocol, with dataset-dependent relative performance.

## Broader input-ablation context

The broader ablation compares speed only, shared observed context, actuation inputs, and all observed inputs. For EV and QX50, those exploratory ablation runs used window caps; the exact uncapped results above resolve the actuation-input head-to-head required for a fair LSTM comparison. Blazer and Pacifica remained below the caps.

The ablation supports the narrower conclusion that speed alone is not generally sufficient relative to actuation inputs. It does not establish equivalence of operating conditions, because road grade, payload, wind, driver identity, and transmission state are unavailable. More observed variables also do not always improve held-out-trip performance, which is consistent with route-level distribution shift and supports cautious, non-causal wording.

## Reporting decision

- Use the validation-selected checkpoint metrics as the canonical LSTM results.
- Keep last-epoch metrics only as an audit/sensitivity result.
- Use the uncapped EV and QX50 benchmark results for the exact actuation-input comparison; use the existing full-window Blazer and Pacifica results.
- State explicitly that non-recurrent model selection used validation MAE and that the test set was not used for selection.
- Do not claim general LSTM superiority: the LSTM is numerically best only for the BMW i3 in this comparison.
- Do not interpret overlapping trip-bootstrap intervals as proof of equivalence or significance.
- Do not describe the vehicle comparison as causal or as identical operating conditions. Use “comparison conditioned on the observed shared covariates.”

<!-- FIVE_SEED_RESULTS_START -->
## Five-seed fixed-split LSTM robustness study

GitHub Actions `Revision LSTM Five Seeds` run `30742131505` completed all 20 jobs successfully. The complete-trip split is fixed with `split_seed=20260801`; training seeds `20260801`–`20260805` vary model initialization and minibatch order only.

Audit checks passed for every artifact: identical trip manifests within each dataset, all available windows, feature scaler fitted only on training-trip rows, checkpoint selected by minimum validation MSE, selected checkpoint restored before test evaluation, and no test-set use in model selection.

| Dataset | MAE mean ± sample SD | MAE median [min, max] | RMSE mean ± sample SD | R2 mean ± sample SD | Best epochs |
|---|---:|---:|---:|---:|---|
| BMW i3 | 0.01790465 ± 0.001838746 | 0.017722719 [0.015595206, 0.020468385] | 0.029667757 ± 0.0029987745 | 0.95677402 ± 0.0086621145 | 9, 9, 13, 14, 14 |
| Infiniti QX50 | 0.15449141 ± 0.0036553399 | 0.15606217 [0.1500752, 0.15876343] | 0.25975494 ± 0.0047864319 | 0.96628073 ± 0.001241985 | 19, 18, 19, 20, 19 |
| Chevrolet Blazer | 0.13757026 ± 0.0034010254 | 0.13710105 [0.13330638, 0.14184047] | 0.23398074 ± 0.0070429564 | 0.77736788 ± 0.013483629 | 17, 19, 18, 12, 18 |
| Chrysler Pacifica | 0.91359455 ± 0.099626502 | 0.87870148 [0.8249572, 1.0798166] | 1.4454613 ± 0.1360514 | 0.70222133 ± 0.057228902 | 2, 2, 4, 2, 1 |

### Comparison with validation-selected non-recurrent baselines

The canonical baseline CSV contains exact full-data selected baselines for BMW i3 and QX50. Blazer and Pacifica remain documented in the broader benchmark summary but are not represented as selected non-recurrent rows in that CSV.

| Dataset | Five-seed LSTM mean MAE | Baseline | Baseline MAE | Mean MAE difference | Lower MAE |
|---|---:|---|---:|---:|---|
| BMW i3 | 0.01790465 | mlp | 0.017710154 | 0.00019449556 | mlp |
| Infiniti QX50 | 0.15449141 | random_forest | 0.12850532 | 0.025986096 | random_forest |

The five-seed analysis is the canonical robustness result. It should replace single-seed language when discussing LSTM performance variability. Comparisons remain descriptive because the five training seeds do not constitute independent test datasets and the vehicle comparison remains conditioned on observed covariates rather than causally matched operating conditions.
<!-- FIVE_SEED_RESULTS_END -->
