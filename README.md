# CredibleCo2
Decarbonizing road transport requires consistent and transparent methods for comparing CO2 emissions across vehicle technologies. This work proposes a machine learning–based framework for like-for-like operational assessment of internal combustion engine vehicles (ICEVs) and electric vehicles (EVs) under identical, real-world driving conditions. 

# Repository structure
- [src](https://github.com/INXE-Lainf-Labs/CredibleCo2/tree/main/src): Main source code (jupyter notebooks for experiments, data preparation, helper functions, model architecture etc).
- [results](https://github.com/INXE-Lainf-Labs/CredibleCo2/tree/main/results): JSON files containing experiment results presented in the [MODEL_CARD](https://github.com/INXE-Lainf-Labs/CredibleCo2/blob/main/MODEL_CARD.md). 
- [data](https://github.com/INXE-Lainf-Labs/CredibleCo2/tree/main/data): Directory covering the datasets we acquired. 

# Reproducibility 
In order to promote better reproducibility, the code required to run the experiments were written in jupyter-notebook format with commented sections. 

Detailed structure:
- The file [src/LSTM_EV.ipynb](https://github.com/INXE-Lainf-Labs/CredibleCo2/tree/main/src/LSTM_EV.ipynb) presents the code required to train the LSTM EV model, which is the model that enables predicting Co2 emission for Electric Vehicles. In addition to that, the user can use this notebook to carry the proxy analysis, which can be achieved by further training the Torque/Throttle model and assessment through the EV model. 
- Analogously, the file [src/LSTM_ICEV.ipynb](https://github.com/INXE-Lainf-Labs/CredibleCo2/tree/main/src/LSTM_ICEV.ipynb) contains the steps required to reproduce experiments regarding the Co2 emission model for ICEVs. 
- The file [src/prepare_data.ipynb](https://github.com/INXE-Lainf-Labs/CredibleCo2/tree/main/src/prepare_data.ipynb) contains auxiliary scripts for processing TMQD files.

## Co2 Emissions Model

Reproducing any of the experiments listed above mainly involve the steps listed below:

1. Loading the dataframe, (optionally filtering required columns) for either ICEV or EV data.
1. Data normalization, through the `normalize` function. 
1. Train-test split, through `time_series_dataset_split` function. 
1. Train model, through the `train_model` function.

These steps enable to obtain the models required for Co2 emission prediction, whereas test-time analyzes can be further carried by selecting test-trip index. 

## Proxy Scenario

Carrying out the proxy analysis step is very straightforward. Once we have obtained the corresponding Emission Model following the steps above, we then need to reproduce the same pipeline (carried to obtain the Co2 Model) to obtain a model that predict torque and throttle features. 

After training the torque/throttle model, we can then carry out the same test-time analysis described (by selecting a testset trip index) and executing the corresponding notebook cell. 

In this step we use the torque/throttle model to obtain these features and then we substitute the predicted torque/throttle features from the original dataset torque and throttle. After that we can input this tensor into the original Co2 model to assess the predictions.  

# Dependencies
Our whole project was built using Python, mainly PyTorch. The [requirement.yaml](https://github.com/INXE-Lainf-Labs/CredibleCo2/blob/main/environment.yaml) and [osx_environment.yaml](https://github.com/INXE-Lainf-Labs/CredibleCo2/blob/main/osx_environment.yaml) files contains a list of dependencies and can be used for creating an environment with the required packages (e.g., via micromamba, conda) for either OSX or Linux platforms. 

# Contribution policy
Contributions are encouraged via pull requests, and will be assessed according to the significance of the changes proposed. 

