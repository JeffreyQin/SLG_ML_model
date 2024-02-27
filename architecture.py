import torch
from torch import nn, utils



"""
down-sampler to reduce input sequence length by summarizing frames of adjacent timestamps

input shape: [batch, # timestamp, # feature = 4]
output shape: [batch, # timestamp (downsampled), # feature = 8]

"""
class DownSampler(nn.Module):
    
    def __init__(self, input_channels=4):
        super(DownSampler, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=8, stride=2),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        
        self.norm_layer = nn.BatchNorm1d(num_features=16)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.norm_layer(x)
        return x
        
""" 
bidirectional lstm 

ensure:
    - self.lstm_input_dim == output dim from downsampler
"""
class LSTMModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(LSTMModel, self).__init__()

        self.lstm_input_dim = 8
        self.lstm_hidden_dim = 128
        self.num_lstm_layers = 3
        self.bidir_lstm = True

        self.downsampler = DownSampler(input_channels=input_size)
        
        self.lstm_layer = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=self.bidir_lstm
        )

        self.linear_layer = nn.Linear(
            in_features=(self.lstm_hidden_dim * 2 if self.bidir_lstm else self.lstm_hidden_dim),
            out_features=output_size
        )

    def forward(self, x):
        x = self.downsampler(x)
        output, state = self.lstm_layer(x)
        logits = self.linear_layer(output)
        return logits



