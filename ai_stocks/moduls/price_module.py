import torch.nn as nn

class PriceModule(nn.Module):

    def __init__(self, input_size=8, hidden_size=64, num_layers=1, output_size=1, dropout=0, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first

        self.rnn = nn.LSTM(input_size=self.input_size, 
                           hidden_size=self.hidden_size, 
                           num_layers=self.num_layers, 
                           batch_first=self.batch_first, 
                           dropout=(0 if num_layers == 1 else self.dropout))
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # self.relu = nn.ReLU()
        
    def forward(self, x):
        # out shape: (batch, seq_len, hidden_size) if batch_first=True
        # print('input shape: ', x.shape)
        out, (hidden, cell) = self.rnn(x)
        # print('rnn output shape: ', out.shape, 'others: ', hidden.shape, cell.shape)
        # Get the last timestep's output
        out = out[:, -1, :]  # shape: (batch, hidden_size)
        # print('before linear: ', out.shape)
        out = self.linear(out)  # shape: (batch, output_size)
        # print('linear ouput: ', out.shape)
        # out = self.relu(out)
        # print('relu ouput: ', out.shape)
        return out
