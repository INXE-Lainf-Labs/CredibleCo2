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
        self.norm = nn.LayerNorm(32)    
        self.fc2 = nn.Linear(32, output_size)  # Output layer adapted
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, full_sequence=False):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        if not full_sequence:
            out = out[:, -1, :]        # Use output from last time step
        out = self.fc1(out)
        out = self.relu(self.norm(out)) # Apply hidden layer
        out = self.fc2(out)            # Final output (multiple values)
        return out


class LSTM_Block(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_Block, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x,_ = self.lstm(x)
        return self.norm(x)    

class MultipleLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_blocks=4):
        super(MultipleLayerLSTM, self).__init__()
        self.num_blocks = num_blocks
        self.hidden_size = hidden_dim

        self.input_layer = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, batch_first=True)
        self.lstm_blocks = nn.ModuleList([
            LSTM_Block(input_size=self.hidden_size, hidden_size=self.hidden_size)
            for _ in range(self.num_blocks)])
        
        self.bottleneck = nn.Linear(self.hidden_size, 32)
        
        self.tanh = nn.Tanh()
        self.act = nn.GELU()
        
        self.norm_1 = nn.LayerNorm(self.hidden_size)    
        self.norm_2 = nn.LayerNorm(self.hidden_size)

        self.fc2 = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x, full_sequence=False):
        out, _ = self.input_layer(x)
        out = self.norm_1(out)

        for blk in self.lstm_blocks:
            out = out + self.dropout(blk(out)) # run through lstm block, normalize, activate, apply dropout then add residual

        if not full_sequence:
            out = out[:, -1, :]        # Use output from last time step
                
        return self.fc2(out)


