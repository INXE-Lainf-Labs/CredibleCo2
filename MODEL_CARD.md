# Model Card — Credible Co2

Decarbonizing road transport requires consistent and transparent methods for comparing CO2 emissions across vehicle technologies. This work proposes a machine learning–based framework for like-for-like operational assessment of internal combustion engine vehicles (ICEVs) and electric vehicles (EVs) under identical, real-world driving conditions.
## 1. General Information:
Here we provide a list of the LSTM models and datasets they were trained to predict Carbon Dioxide (Co2) emissions (or features related to it).

|    Dataset |   Entries |   Type    |
|    :------:    |   :------:    |   :------:    | 
|    Infiniti QX50   |   `377149`    |   ICEV    |
|    Chevrolet Blazer   |   `108678`    |   ICEV    |
|    Chrysler Pacifica   |   `183996`    |   ICEV    |
|    BMW i3/Ieee   |   `1094794`    |   EV    |


LSTMs:
- Electrical Vehicle (EV): Between 1-4 layers with 32 hidden units per layer + layer norm and residual connections.
- Internal Combustion Engine Vehicle (ICEV): Between 1-4 layers with 64 hidden units per layer + layer norm and residual connections.
## 2. Intended uses
The overall methodology (models, training scritps, data processing routines, etc) is intended to researchers or enthusiasts who may feel inspired to build upon this project to carry out work involving Co2 prediction with vehicle (time-series) data or related.
