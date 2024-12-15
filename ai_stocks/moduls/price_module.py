import torch
import torch.nn as nn

class PriceModule(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, output_size=1, dropout=0.2, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=input_size, 
                           hidden_size=self.hidden_size, 
                           num_layers=self.num_layers, 
                           batch_first=self.batch_first, 
                           dropout=(0 if num_layers == 1 else self.dropout))
        
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        return out
