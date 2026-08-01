# Corrected-protocol CPU LSTM feasibility results

Run: GitHub Actions `Revision LSTM CPU`, run 30723705348

Configuration shared by all runs:

- complete-trip train/validation/test split;
- feature scaler fitted only on training trips;
- windows created after splitting and independently within each trip;
- deterministic seed 20260801;
- 4 residual LSTM blocks, hidden dimension 32;
- batch size 512;
- 5 epochs;
- deterministic cap of 100,000 training windows, 50,000 validation windows, and 100,000 test windows.

These are feasibility runs. They validate the corrected implementation and establish CPU runtime, but they are not yet full-data, 20-epoch replacements for the historical results.

## Emissions models

| Dataset | Train/Val/Test trips | Used train/val/test windows | MAE | RMSE | R2 | CPU training time |
|---|---:|---:|---:|---:|---:|---:|
| BMW i3 EV | 44 / 12 / 14 | 100,000 / 50,000 / 100,000 | 0.042787 | 0.060791 | 0.820163 | 42.2 s |
| Infiniti QX50 | 21 / 6 / 7 | 100,000 / 50,000 / 74,790 | 0.187315 | 0.293151 | 0.957065 | 41.7 s |
| Chevrolet Blazer | 4 / 1 / 2 | 36,573 / 8,441 / 30,997 | 0.154211 | 0.264377 | 0.715973 | 15.8 s |
| Chrysler Pacifica | 12 / 3 / 4 | 100,000 / 42,619 / 37,303 | 0.940980 | 1.419351 | 0.714903 | 44.9 s |

Trip-level MAE summaries:

- BMW i3 EV: mean 0.044492; 95% bootstrap interval for the mean [0.039911, 0.049234], 14 test trips.
- Infiniti QX50: mean 0.207031; 95% bootstrap interval [0.167594, 0.252375], 7 test trips.
- Chevrolet Blazer: mean 0.154882; 95% bootstrap interval [0.149076, 0.160687], 2 test trips.
- Chrysler Pacifica: mean 0.990001; 95% bootstrap interval [0.799275, 1.180727], 4 test trips.

## EV torque/throttle feature model

Input context: velocity, ambient temperature, cabin temperature, and longitudinal acceleration.

| Dataset | Train/Val/Test trips | Used train/val/test windows | MAE | RMSE | R2 | CPU training time |
|---|---:|---:|---:|---:|---:|---:|
| BMW i3 EV | 44 / 12 / 14 | 100,000 / 50,000 / 100,000 | 6.122744 | 11.232784 | 0.846819 | 41.6 s |

Trip-level MAE mean: 6.085300; 95% bootstrap interval [5.479764, 6.865763], 14 test trips.

## Interpretation

The corrected LSTM implementation is computationally feasible on standard GitHub-hosted CPUs. Five epochs over 100,000 training windows required about 42 seconds for the EV and QX50 emissions models and about 45 seconds for Pacifica; Blazer used all available training windows and required about 16 seconds. A full 20-epoch run is therefore operationally feasible, although full-data runtime will scale with the number of available windows.

The large differences in absolute MAE between vehicle datasets should not be interpreted as cross-dataset model rankings without checking target units and target scales. The main valid conclusion from this feasibility round is that corrected trip-wise LSTM reruns are practical and produce meaningful held-out-trip predictive performance.
