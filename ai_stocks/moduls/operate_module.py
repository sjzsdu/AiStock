import torch
import torch.nn as nn
import torch.nn.functional as F

class OperateModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 32, num_layers=1):
        super().__init__()
        
        # 定义一个LSTM网络
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 定义一个全连接层，将LSTM输出映射到类别
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM的输出结果和隐藏状态
        out, (hn, cn) = self.lstm(x)
        
        # 选取LSTM最后一个时间步的输出
        out = out[:, -1, :]
        
        # 输入通过全连接层
        out = self.fc(out)
        return out