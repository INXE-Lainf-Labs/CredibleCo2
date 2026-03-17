# Model Card — Credible Co2

Decarbonizing road transport requires consistent and transparent methods for comparing CO2 emissions across vehicle technologies. This work proposes a machine learning–based framework for like-for-like operational assessment of internal combustion engine vehicles (ICEVs) and electric vehicles (EVs) under identical, real-world driving conditions.
## 1. General Information:
Here we provide a list of the LSTM models and datasets they were trained to predict Carbon Dioxide (Co2) emissions (or features related to it).

|    Dataset |   Entries |   Type    |
|    :------:    |   :------:    |   :------:    | 
|    Infiniti QX50   |   `377149`    |   ICEV    |
|    Chevrolet Blazer   |   `108678`    |   ICEV    |
|    Chrysler Pacifica   |   `183996`    |   ICEV    |
|    BMW i3 (ieee)   |   `1094794`    |   EV    |


LSTMs:
- Electrical Vehicle (EV): Between 1-4 layers with 32 hidden units per layer + layer norm and residual connections.
- Internal Combustion Engine Vehicle (ICEV): Between 1-4 layers with 64 hidden units per layer + layer norm and residual connections.
## 2. Intended uses
The overall methodology (models, training scritps, data processing routines, etc) is intended to researchers or enthusiasts who may feel inspired to build upon this project to carry out work involving Co2 prediction with vehicle (time-series) data or related.

## 3. Evaluation scenarios
- 3.1 - Domain Specific Training:
     - Emission model: Trained to predict Co2 emissions.
     - Feature model: Trained to predict domain-specific actuation variables, [torque, throttle], using only contextual variables common to both domains, namely: velocity, ambient temperature, cabin temperature, and longitudinal acceleration.
- 3.2 - Proxy Validation:
     - This stage acts as a pseudo-counterfactual analysis, in which we assess the capability of the feature models by evaluating how emission models behave when using features predicted by the feature models instead of original (dataset) features.
- 3.3 - Proposed Test-Time Conterfactual Analysis:
     - In this final stage, we propose to treat the EV as a counterfactual system under identical operating conditions as an ICEV’s trajectory. The ICEV context (velocity, temperatures, longitudinal acceleration) would be fed to the pre trained EV Feature model to infer the torque and throttle that an EV would likely produce. These inferred signals, together with the velocity profile, would then be passed to the EV Emissions model to generate the counterfactual EV emissions series
## 4. Results
|    ICEV Dataset |  Metric (Emission Model) |   Split   |   Result    |
|    :------:    |   :------:    |   :------:    |   :------:    |
|    QX50    |    mse  |  train   |   0.09683 ± 0.10378  |
|    QX50    |    mse  |  val  |  0.07491 ± 0.0143    |
|    QX50    |    mae  |  val  |  0.07491 |
|    :------:    |   :------:    |   :------:    |   :------:    |
|    Blazer    |    mse  |  train   |   0.0726 ± 0.07577  |
|    Blazer    |    mse  |  val  |  2.96825 ± 0.41827    |
|    Blazer    |    mae  |  val  |  2.96825 |
|    :------:    |   :------:    |   :------:    |   :------:    |
|    Pacifica    |    mse  |  train   |   0.83086 ± 0.2706  |
|    Pacifica    |    mse  |  val  |  0.46186 ± 0.12619    |
|    Pacifica    |    mae  |  val  |  0.46186 |
|    :------:    |   :------:    |   :------:    |   :------:    |
