# Model Card — Credible Co2

Project description: lorem ipsum
## 1. General Information:
Here we provide a list of the LSTM models and datasets they were trained to predict Carbon Dioxide (Co2) emissions.

|    Dataset |   Entries |   Type    |
|    :------:    |   :------:    |   :------:    | 
|    Infiniti QX50   |   `377149`    |   ICEV    |
|    Chevrolet Blazer   |   `108678`    |   ICEV    |
|    Chrysler Pacifica   |   `183996`    |   ICEV    |
|    BMW i3/Ieee   |   `1094794`    |   EV    |


LSTMs:
- Electrical Vehicle (EV): Between 1-4 layers with 32 hidden units per layer + layer norm with residual connections.
- Internal Combustion Engine Vehicle (ICEV): Between 1-4 layers with 64 hidden units per layer + layer norm with residual connections
## 2. Intended uses
