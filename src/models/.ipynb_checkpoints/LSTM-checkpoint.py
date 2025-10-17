import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of outputs (e.g., 2 for ['Motor Torque', 'Throttle']).
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)  # Output layer adapted

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]             # Use output from last time step
        out = self.relu(self.fc1(out)) # Apply hidden layer
        out = self.fc2(out)            # Final output (multiple values)
        return out