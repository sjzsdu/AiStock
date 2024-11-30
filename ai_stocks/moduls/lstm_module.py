import torch.nn as nn

class LSTMModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 32, num_layers = 2, dropout=0.2):
        super(LSTMModule, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_lstm_out = lstm_out[:, -1, :]
        out = self.fc(last_lstm_out)
        return out

