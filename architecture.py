import torch
from torch import nn, utils



"""
Downsampler to reduce input sequence length by summarizing features in adjacent timestamps
    - inspired by Deep Speech 2
"""
class DownSampler(nn.Module):
    
    def __init__(self, input_channels=8):
        super(DownSampler, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=8, stride=2),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        
        self.norm_layer = nn.BatchNorm1d(num_features=32)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.norm_layer(x)
        return x



"""
LSTM Encoder

Ensure
    - self.enc_input_dim == output from self.downsampler
"""
class LSTMEncoder(nn.Module):
    def __init__(self, input_size):
        super(LSTMEncoder, self).__init__()

        self.enc_input_dim = 32
        self.enc_hidden_dim = 128
        self.num_lstm_layers = 3
        self.bidir_lstm = True

        self.downsampler = DownSampler(input_channels=input_size)

        self.lstm_layer = nn.LSTM(
            input_size=self.enc_input_dim,
            hidden_size=self.enc_hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=self.bidir_lstm
        )
    
    """
    downsampler layer
        - input shape: [batch, # channels = 8, # timestamps]
        - output shape: [batch, # channels = 16, # timestamps (downsampled)]

    lstm (encoder) layer (batch_first=True)
        - input shape: [batch, # timestamps, # channels]
        - output shape: [batch, # timestamps, hidden dim (x3 for bidor)]
    """
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.downsampler(x)

        x = x.permute(0, 2, 1)
        outputs, (hidden_states, cell_states) = self.lstm_layer(x)
        return outputs, hidden_states, cell_states


""" 
bidirectional lstm 

ensure:
    - self.lstm_input_dim == output dim from downsampler
"""

class LSTMModel(nn.Module):

    def __init__(self, input_size, vocab_size):
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
            out_features=vocab_size
        )
        
    """
    downsample layer
        - expected input shape: [batch, # channels = 8, # timestamps]
        - output shape: [batch, # channels = 16, # timestamps (downsampled)]

    lstm layer (batch_first=True)  
        - expected input shape: [batch, # timestamps, # channels]
        - output shape: [batch, # timestamps, hidden dim (x2 for bidir)]

    linear layer 
        - output shape: [batch, # timestamps, # features = vocab_size] (passed to softmax)

    """

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.downsampler(x)
    
        x = x.permute(0,2,1)
        output, state = self.lstm_layer(x)
        
        output = self.linear_layer(output)

        # softmax probabilities
        output_probs = nn.functional.softmax(output, dim=2)
        output_probs = output_probs.permute(0,2,1)
        
        return output_probs



