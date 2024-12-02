import torch.nn as nn

class LSTMModule(nn.Module):
    def __init__(self, input_size, output_size, conv_channels=128, lstm_hidden_size=128, lstm_num_layers=2, dropout=0.1, kernel_size=3):
        super().__init__()

        # 一维卷积层用于特征提取
        self.conv1d = nn.Conv1d(in_channels=input_size, 
                                out_channels=conv_channels, 
                                kernel_size=kernel_size, 
                                padding=kernel_size // 2)

        # 批量归一化可以稳定训练过程
        self.batch_norm = nn.BatchNorm1d(conv_channels)
        
        # LSTM层，用于时间依赖性学习
        self.lstm = nn.LSTM(input_size=conv_channels, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True, 
                            dropout=dropout, 
                            bidirectional=True)

        # 全连接层产生最终的输出
        self.fc = nn.Linear(lstm_hidden_size * 2, output_size)

    def forward(self, x):
        # 输入形状转换以适应卷积层
        x = x.permute(0, 2, 1)

        # 卷积 + 批量归一化 + 激活函数
        conv_out = nn.functional.relu(self.batch_norm(self.conv1d(x)))

        # 转换回LSTM层需要的输入形状
        conv_out = conv_out.permute(0, 2, 1)

        # LSTM 前向传播
        lstm_out, _ = self.lstm(conv_out)
        last_lstm_out = lstm_out[:, -1, :]

        # 全连接层生成输出
        out = self.fc(last_lstm_out)
        return out
