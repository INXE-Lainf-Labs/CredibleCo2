"""Historical helper functions from the pre-revision workflow.

These functions are retained for provenance and compatibility only. They must
not be used to reproduce the revised manuscript results. Use
``src.revision_protocol`` and the canonical runners under ``scripts/`` for
complete-trip splitting, training-only scaling, trip-local window creation,
validation-based model selection, and held-out testing.
"""

import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import math
import torch
from torch.optim.lr_scheduler import LambdaLR

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.LSTM import LSTMModel, MultipleLayerLSTM


def _warn_legacy_api(function_name: str, detail: str) -> None:
    warnings.warn(
        f"src.helper.{function_name} belongs to the historical pre-revision "
        f"workflow and must not be used to reproduce the revised manuscript. "
        f"{detail} Use src.revision_protocol and the canonical scripts instead.",
        FutureWarning,
        stacklevel=2,
    )


def train_model(X_seq, y_seq, title=None, hidden_dim=32, train_size=0.8, epochs=5, num_blocks=4, batch_size=128, warmup_epochs=5, base_lr=1e-5, max_lr=1e-3, final_lr=1e-6, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Trains an LSTM model for multi-output regression.

    Historical API retained for provenance only.
    """
    _warn_legacy_api(
        "train_model",
        "It slices training and validation after overlapping windows have already been concatenated, which can place adjacent overlapping windows in different partitions.",
    )

    # Create dataset and split
    if y_seq.size(-1) > 2:
        y_seq = y_seq.unsqueeze(-1)

    X_seq, y_seq = X_seq.to(device), y_seq.to(device)

    dataset = TensorDataset(X_seq, y_seq)
    train_len = int(len(dataset) * train_size)

    train_set = TensorDataset(X_seq[:train_len], y_seq[:train_len])
    val_set = TensorDataset(X_seq[train_len:], y_seq[train_len:])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Model dimensions
    input_size = X_seq.size(-1)
    output_size = y_seq.size(-1) if y_seq.size(-1) == 2 else 1
    if num_blocks > 1:
        model = MultipleLayerLSTM(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, num_blocks=num_blocks).to(device)
    else:
        model = LSTMModel(input_size, output_size).to(device)
    print(f'Model config:{model}')

    # Learning rate scheduler setup
    total_epochs = epochs

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, base_lr, max_lr, final_lr)

    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')

        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step()
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    if title is not None:
        plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Final evaluation using MAE
    model.eval()
    mae_loss = nn.L1Loss()
    mae = 0.0
    mae_list = []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            mae += mae_loss(pred, yb).item() * xb.size(0)
    mae /= len(val_loader.dataset)
    mae_list.append(mae)

    print(f"\nFinal MAE: {mae:.8f}, Final MSE: {val_losses[-1]:.8f}")
    print(f'Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')

    return model, train_losses, val_losses, mae_list


def time_series_dataset_split(dataset, labels, id_column, train_size=0.8, WINDOW_SIZE=10):
    _warn_legacy_api(
        "time_series_dataset_split",
        "This function is part of the historical API and does not create a separate validation-trip partition; its output was historically passed to train_model, which then split concatenated overlapping windows.",
    )

    x_train_seq = []
    y_train_seq = []

    x_test_seq = []
    y_test_seq = []

    trips = dataset[id_column].unique()
    num_trips = len(trips)

    training_size = int(train_size * num_trips)
    training_trips = trips[:training_size]
    test_trips = trips[training_size:]
    assert len(training_trips) + len(test_trips) == num_trips, 'Data split indexing error'

    for trip in training_trips:
        slice = dataset[dataset[id_column] == trip]
        x_data = slice.drop(columns=labels + [id_column])
        y_data = slice[labels]
        x_train_tensor = torch.from_numpy(x_data.to_numpy()).float()
        y_train_tensor = torch.from_numpy(y_data.to_numpy()).float()

        X_seq, y_seq = [], []
        t = x_train_tensor.size(0)
        for i in range(WINDOW_SIZE, t):
            X_seq.append(x_train_tensor[i - WINDOW_SIZE:i])
            y_seq.append(y_train_tensor[i])
        x_train_seq.append(torch.stack(X_seq))
        y_train_seq.append(torch.stack(y_seq))

    for trip in test_trips:
        slice = dataset[dataset[id_column] == trip]
        x_data = slice.drop(columns=labels + [id_column])
        y_data = slice[labels]
        x_test_tensor = torch.from_numpy(x_data.to_numpy()).float()
        y_test_tensor = torch.from_numpy(y_data.to_numpy()).float()

        X_seq, y_seq = [], []
        t = x_test_tensor.size(0)
        for i in range(WINDOW_SIZE, t):
            X_seq.append(x_test_tensor[i - WINDOW_SIZE:i])
            y_seq.append(y_test_tensor[i])
        x_test_seq.append(torch.stack(X_seq))
        y_test_seq.append(torch.stack(y_seq))

    return torch.cat(x_train_seq), torch.cat(y_train_seq), x_test_seq, y_test_seq


def time_series_dataset(dataset, labels, id_column, train_size=0.8):
    '''
        Split data with a trip-wise stratification.
        x_train_seq: list of trip tensors (data cube).  (N_i, WINDOW_SIZE, num_features) for each trip `i`.
            - Where N_i = (train_size * len(trip_i)) / WINDOW_SIZE, correspond to the number of windows for trip `i`.
        y_train_seq: correspondent target values for each entry in x_train_seq list of tensors in the shape (N_i,)

        x_val_tensor: same as x_train seq, but has lenght = ((1 - train_size) * len(trip_i)) / WINDOW_SIZE).
        y_val_tensor: corresponding target values for x_val_tensor.

        For each trip in the dataset we split the data (x and y) such as the train set has length=train_size*len(trip_i), and
        the val set has the remaining samples, hence producing x_train, x_val, y_train and y_val.

        We then apply a sliding windows over each of these subsets to obtain the corresponding sequences. For the x data we
        always retrieve the last `WINDOW_SIZE` timesteps, whereas the target values (y set) is filled with the last time steps
        only, e.g., (y[WINDOW_i]) for each window in the corresponding set of trip entries.
    '''
    x_train_seq = []
    y_train_seq = []

    x_test = []
    y_test = []

    WINDOW_SIZE = 10
    for trip in dataset[id_column].unique():
        slice = dataset[dataset[id_column] == trip]
        size = int(len(slice) * train_size)
        x_data = slice.drop(columns=[labels, id_column])
        y_data = slice[labels]

        x_train_tensor = torch.from_numpy(x_data[:size].to_numpy()).float()
        y_train_tensor = torch.from_numpy(y_data[:size].to_numpy()).float()

        x_val_tensor = torch.from_numpy(x_data[size:].to_numpy()).float()
        x_test.append(x_val_tensor)

        y_val_tensor = torch.from_numpy(y_data[size:].to_numpy()).float()
        y_test.append(y_val_tensor)

        X_seq, y_seq = [], []
        t = x_train_tensor.size(0)
        for i in range(WINDOW_SIZE, t):
            X_seq.append(x_train_tensor[i - WINDOW_SIZE:i])
            y_seq.append(y_train_tensor[i])
        x_train_seq.append(torch.stack(X_seq))
        y_train_seq.append(torch.stack(y_seq))

    return torch.cat(x_train_seq), torch.cat(y_train_seq), x_test, y_test


def time_series_train_test_split_multiple(dataset, labels, id_column, train_size=0.8):
    """
    Realiza a divisão de treino e teste de N séries temporais com múltiplas colunas de rótulo.
    """
    X_trains, X_tests, y_trains, y_tests = [], [], [], []

    for trip in dataset[id_column].unique():
        data_slice = dataset[dataset[id_column] == trip]
        size = int(len(data_slice) * train_size)

        X_data = data_slice.drop(columns=labels)
        y_data = data_slice[labels]

        X_trains.append(X_data.iloc[:size])
        X_tests.append(X_data.iloc[size:])
        y_trains.append(y_data.iloc[:size])
        y_tests.append(y_data.iloc[size:])

    return (
        pd.concat(X_trains).reset_index(drop=True),
        pd.concat(X_tests).reset_index(drop=True),
        pd.concat(y_trains).reset_index(drop=True),
        pd.concat(y_tests).reset_index(drop=True)
    )


def normalize(data, X_columns, y_column):
    """
    Realiza a normalização dos dados.

    Historical API retained for provenance only.
    """
    _warn_legacy_api(
        "normalize",
        "It fits MinMaxScaler on every row passed to the function, so calling it before a trip split leaks validation/test minima and maxima into preprocessing.",
    )

    X = data[X_columns]
    y = data[y_column].values

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X.values)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )
    X_scaled['Trip'] = data['Trip']
    scaled_dataset = X_scaled
    scaled_dataset[y_column] = y

    return scaled_dataset, scaler_X


def time_series_train_test_split(dataset, label, id_column, train_size=0.8):
    """
    Realiza a divisão de treino e teste de N séries temporais.
    """
    X_trains, X_tests, y_trains, y_tests = [], [], [], []

    for trip in dataset[id_column].unique():
        data_slice = dataset[dataset[id_column] == trip]
        size = int(len(data_slice) * train_size)

        X_trains.append(data_slice.drop(columns=[label]).iloc[:size, :])
        X_tests.append(data_slice.drop(columns=[label]).iloc[size:, :])

        y_trains.append(data_slice[label].iloc[:size])
        y_tests.append(data_slice[label].iloc[size:])

    return pd.concat(X_trains), pd.concat(X_tests), pd.concat(y_trains), pd.concat(y_tests)


def create_seq_fast_multiple(X, y, id_column, window_size=10):
    """
    Cria sequências temporais para modelos baseados em janela deslizante, suportando múltiplos alvos.
    """
    X_seq, y_seq = [], []

    data = pd.concat([X.copy(), y.copy()], axis=1)

    for _, group in data.groupby(id_column):
        group = group.reset_index(drop=True)

        if len(group) < window_size:
            continue

        for i in range(window_size, len(group)):
            window = group.iloc[i - window_size:i]
            target = group.iloc[i][y.columns]

            X_seq.append(window.drop(columns=[id_column] + list(y.columns)).values)
            y_seq.append(target.values if isinstance(target, pd.Series) else target)

    return np.array(X_seq), np.array(y_seq)


def create_seq_fast(X, y, id_column, window_size=10):
    X_seq, y_seq = [], []

    data = X.copy()
    data['__y__'] = y

    for _, group in data.groupby(id_column):
        group = group.reset_index(drop=True)

        if len(group) < window_size:
            continue

        for i in range(window_size, len(group)):
            window = group.iloc[i - window_size:i]
            target = group.iloc[i]['__y__']

            X_seq.append(window.drop(columns=[id_column, '__y__']).values)
            y_seq.append(target)

    return np.array(X_seq), np.array(y_seq)


def create_seq(X, y, id_column, window_size=10):
    """
    O LSTM precisa de um cubo de dados adicionando a dimensão tamanho da janela.
    """
    X_seq, y_seq = [], []

    for i in range(window_size, len(X)):
        data_slice = X.iloc[i-window_size:i, :]

        if len(data_slice[id_column].unique()) == 1:
            X_seq.append(data_slice.drop(columns=[id_column]).values)
            y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


def plot_error_histogram(y_pred, y_test, bins=50, color='blue'):
    """
    Plota um histograma do erro do modelo.
    """
    error = y_pred - y_test

    mean_error = np.mean(error)
    std_error = np.std(error)
    min_error = np.min(error)
    max_error = np.max(error)

    plt.hist(error, bins=bins, color=color, alpha=0.7, edgecolor='black')

    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f"Média: {mean_error:.4f}")
    plt.axvline(mean_error + 2*std_error, color='green', linestyle='dashed', linewidth=2, label=f'+2σ: {(mean_error + 2*std_error):.4f}')
    plt.axvline(mean_error - 2*std_error, color='green', linestyle='dashed', linewidth=2, label=f'-2σ: {(mean_error - 2*std_error):.4f}')
    plt.axvline(min_error, color='purple', linestyle='dashed', linewidth=2, label=f'Mínimo: {min_error:.4f}')
    plt.axvline(max_error, color='orange', linestyle='dashed', linewidth=2, label=f'Máximo: {max_error:.4f}')

    plt.xlabel("Erro (y_pred - y_real)")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(True)

    plt.show()


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, base_lr, max_lr, final_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return (base_lr + (max_lr - base_lr) * (current_epoch / warmup_epochs)) / base_lr
        else:
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = final_lr + (max_lr - final_lr) * cosine_decay
            return lr / base_lr

    return LambdaLR(optimizer, lr_lambda)
