# Corrected-protocol LSTM results: full-data, validation-selected checkpoints

GitHub Actions run: `Revision LSTM CPU` #10 (`30725904091`), commit `14cb004b1f8870f59075103ab33bb6b365bbbc67`.

## Audit status

- All four dataset jobs completed successfully, including the EV torque/throttle feature model.
- Every model ran for 20 epochs with batch size 512 and used all available windows.
- Complete trip IDs are disjoint across training, validation, and test.
- The feature scaler is fitted only on training-trip rows; windows are created independently inside each trip after the split.
- The primary test result restores the checkpoint with minimum validation MSE. The test set is not used for checkpoint selection.
- A valid PyTorch checkpoint and SHA-256 digest were verified for each model.

## Primary test results

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
| Infiniti QX50 — emissions | 0.15130749 | 0.1523663 | 0.25643459 | 0.25733075 | 0.967146 | 0.966916 |
| Chevrolet Blazer — emissions | 0.13710105 | 0.14013464 | 0.23035044 | 0.23336242 | 0.784379 | 0.778703 |
| Chrysler Pacifica — emissions | 0.92454867 | 1.4726813 | 1.4117892 | 2.4717857 | 0.717932 | 0.135360 |
| BMW i3 — torque/throttle | 4.1758483 | 4.1454332 | 7.6642369 | 7.6313938 | 0.928868 | 0.929476 |

The Pacifica result confirms the need for checkpoint restoration: epoch 2 gives MAE 0.92455 and R2 0.71793, whereas the epoch-20 model gives MAE 1.47268 and R2 0.13536. For the EV feature model, epoch 19 is retained because it minimizes validation MSE, although epoch 20 is marginally better on the test set; using epoch 20 would be test-driven selection.

## Trip-level robustness

| Dataset / task | Test trips | Mean MAE | Median | Sample SD | IQR | 95% bootstrap CI for mean |
|---|---:|---:|---:|---:|---:|---:|
| BMW i3 — emissions | 14 | 0.017579487 | 0.017268661 | 0.0046336457 | [0.016214721, 0.020541181] | [0.015274022, 0.019917923] |
| Infiniti QX50 — emissions | 7 | 0.1798102 | 0.13350457 | 0.077811262 | [0.12893391, 0.2354837] | [0.12853905, 0.2342749] |
| Chevrolet Blazer — emissions | 2 | 0.13702057 | 0.13702057 | 0.00098580425 | [0.13667203, 0.1373691] | [0.1363235, 0.13771763] |
| Chrysler Pacifica — emissions | 4 | 0.9882495 | 1.0338004 | 0.24324873 | [0.83386122, 1.1881886] | [0.78787896, 1.18862] |
| BMW i3 — torque/throttle | 14 | 3.9141846 | 3.4077668 | 1.4469537 | [3.2326418, 3.6501129] | [3.3425206, 4.7621337] |

## Preliminary benchmark and input-ablation context

The benchmark artifacts use the same trip manifests and training-only scaling. However, the benchmark workflow capped training at 200,000 windows, validation at 100,000, and test at 150,000. Blazer and Pacifica were below these caps; QX50 used all validation/test windows but a capped training set; EV used capped training, validation, and test sets. Therefore, EV and QX50 comparisons below are informative but should not be presented as an exact full-data head-to-head table without regenerating the benchmark evaluation on all test windows.

### Best benchmark per input set, selected by validation MAE

| Dataset | Input set | Selected model | Test MAE | Test RMSE | Test R2 |
|---|---|---|---:|---:|---:|
| BMW i3 | `speed_only` | mlp | 0.046487144 | 0.0675317 | 0.777745 |
| BMW i3 | `shared_observed_context` | hist_gb | 0.020660795 | 0.031608312 | 0.951310 |
| BMW i3 | `actuation_inputs` | random_forest | 0.017680974 | 0.035201102 | 0.939612 |
| BMW i3 | `all_observed_inputs` | random_forest | 0.0094860713 | 0.01620024 | 0.987210 |
| Infiniti QX50 | `speed_only` | mlp | 0.38730825 | 0.61057726 | 0.813742 |
| Infiniti QX50 | `shared_observed_context` | random_forest | 0.346011 | 0.53171348 | 0.858750 |
| Infiniti QX50 | `actuation_inputs` | random_forest | 0.12999971 | 0.226166 | 0.974444 |
| Infiniti QX50 | `all_observed_inputs` | random_forest | 0.1067211 | 0.21233848 | 0.977474 |
| Chevrolet Blazer | `speed_only` | random_forest | 0.31596656 | 0.46423555 | 0.124231 |
| Chevrolet Blazer | `shared_observed_context` | hist_gb | 0.61602738 | 0.84267322 | -1.885568 |
| Chevrolet Blazer | `actuation_inputs` | random_forest | 0.11400737 | 0.19996334 | 0.837515 |
| Chevrolet Blazer | `all_observed_inputs` | random_forest | 0.12052499 | 0.20989784 | 0.820969 |
| Chrysler Pacifica | `speed_only` | mlp | 1.2054086 | 1.8082881 | 0.537247 |
| Chrysler Pacifica | `shared_observed_context` | mlp | 1.4249646 | 1.955422 | 0.458879 |
| Chrysler Pacifica | `actuation_inputs` | ridge | 0.56747335 | 0.98569287 | 0.862502 |
| Chrysler Pacifica | `all_observed_inputs` | ridge | 0.58183732 | 0.99989105 | 0.858512 |

### LSTM versus the best actuation-input benchmark

Both models use velocity, throttle, and motor torque. Model choice among the non-recurrent benchmarks is based on validation MAE.

| Dataset | LSTM MAE | Best benchmark | Benchmark MAE | Lower MAE |
|---|---:|---|---:|---|
| BMW i3 | 0.016999224 | random_forest | 0.017680974 | LSTM |
| Infiniti QX50 | 0.15130749 | random_forest | 0.12999971 | random_forest |
| Chevrolet Blazer | 0.13710105 | random_forest | 0.11400737 | random_forest |
| Chrysler Pacifica | 0.92454867 | ridge | 0.56747335 | ridge |

The ablation consistently shows that speed alone is insufficient relative to actuation inputs. It does not establish equivalence of operating conditions, because road grade, payload, wind, driver identity, and transmission state are unavailable. More observed variables also do not always improve held-out-trip performance, which is consistent with route-level distribution shift and supports cautious, non-causal wording.

## Checkpoint integrity

| Model | Checkpoint | SHA-256 |
|---|---|---|
| `ev-emissions` | `ev_emissions_best.pt` | `055d8f143a6ed090d9905e77fb4ddc5bd5509ffab87edf861a84b6d91320716a` |
| `qx50-emissions` | `qx50_emissions_best.pt` | `82e1c9934e078b39c2f4ca79b85d37e2eb6f2bd0293ac11fabd4049d63720774` |
| `blazer-emissions` | `blazer_emissions_best.pt` | `795da5f17f4a0accc63dd7d894f47a09ee7ecc9d065d4ae9563f362ed8be76a9` |
| `pacifica-emissions` | `pacifica_emissions_best.pt` | `4df95d487b8e2e7d91f557cc90df55e673f083339fadd077da5f8c7cc545a2c9` |
| `ev-feature` | `ev_feature_best.pt` | `0a110c7be32c35895e559ea1fcc07cbaea739fc3cf334695b119619a2ed96272` |

## Reporting decision

- Use the validation-selected checkpoint metrics as the canonical LSTM results.
- Keep last-epoch metrics only as an audit/sensitivity result.
- Treat the existing benchmark/ablation comparison as preliminary for EV and QX50 because of window caps; Blazer and Pacifica are full-window comparisons.
- Do not describe the comparison as causal or as identical operating conditions. Use “comparison conditioned on the observed shared covariates.”
