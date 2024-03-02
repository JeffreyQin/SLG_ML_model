import torch
from torch import nn, utils

import math



"""
DOWN-SAMPLER to reduce input sequence length by summarizing features in adjacent timestamps
    - inspired by Deep Speech 2

input shape: [batch size, num features = 8, seq length]
output shape: [batch size, num features = 32, seq length (downsampled)]
"""
class DownSampler(nn.Module):
    
    def __init__(self, input_size=8):
        super(DownSampler, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=8, stride=2),
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
SCALED DOT PRODUCT ATTENTION mechanism

"""
class SelfAttention(nn.Module):

    def __init__(self, input_size=256):
        super(SelfAttention, self).__init__()

        self.num_features = input_size
        # initialize key, query, value matrices
        self.key_matrix = nn.Linear(in_features=self.num_features, out_features=self.num_features)
        self.query_matrix = nn.Linear(in_features=self.num_features, out_features=self.num_features)
        self.value_matrix = nn.Linear(in_features=self.num_features, out_features=self.num_features)

    def forward(self, x):

        # calculate key, query, value vectors for each timestamp
        key_vecs = self.key_matrix(x)
        query_vecs = self.query_matrix(x)
        value_vecs = self.value_matrix(x)

        # calculate attention score matrix
        attn_scores = torch.matmul(query_vecs, key_vecs.permute(0, 2, 1))
        attn_scores /= math.sqrt(float(self.num_features))

        # weighted attention
        attn_weights = nn.functional.softmax(attn_scores, dim=-1) 

        outputs = torch.matmul(attn_weights, value_vecs)

        return outputs, attn_weights


""" 
LSTM Model
"""

class LSTMModel(nn.Module):
    
    def __init__(self, output_size, input_size=8):
        super(LSTMModel, self).__init__()

        self.lstm_input_dim = 32
        self.lstm_hidden_dim = 128
        self.num_lstm_layers = 2
        self.bidir_lstm = True

        self.downsampler = DownSampler(input_size=input_size)

        self.lstm_layer = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=self.bidir_lstm
        )

        self. lstm_output_dim = self.lstm_hidden_dim * 2 if self.bidir_lstm else self.lstm_hidden_dim

        self.attention = SelfAttention(input_size=self.lstm_output_dim)

        self.linear_layer = nn.Linear(
            in_features=self.lstm_output_dim,
            out_features=output_size # vocab size
        )
    
    """
    downsample layer
        - input shape: [batch size, feature num, seq length]
        - output shape: [batch size, feature num = 16, seq length (downsampled)]

    lstm layer (batch_first=True)  
        - expected input shape: [batch, # timestamps, # channels]
        - output shape: [batch, # timestamps, hidden dim (x2 for bidir)]

    linear layer 
        - output shape: [batch, # timestamps, # features = vocab_size] (passed to softmax)

    """
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.downsampler(x)
        x = x.permute(0, 2, 1)

        outputs, (final_hidden, final_cell) = self.lstm_layer(x)

        outputs, attention_weights = self.attention(outputs)

        outputs = self.linear_layer(outputs)

        # softmax probabilities
        output_probs = nn.functional.softmax(outputs, dim=2)
        return output_probs
        


