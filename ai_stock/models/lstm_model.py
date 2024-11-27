import torch.nn as nn

class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModule, self).__init__()
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout)
        
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM层的前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取LSTM的最后一个时间步的输出
        last_lstm_out = lstm_out[:, -1, :]
        
        # 全连接层的前向传播
        out = self.fc(last_lstm_out)
        
        return out

