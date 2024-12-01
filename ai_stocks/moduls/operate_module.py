import torch
import torch.nn as nn

class OperateModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=3, dropout=0.2, cnn_filters=64, cnn_kernel_size=3):
        super(OperateModule, self).__init__()

        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=cnn_kernel_size, padding=1)
        
        # 定义单向LSTM
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers, batch_first=True, bidirectional=False)
        
        # Attention机制或者Pooling层
        self.attn = nn.Linear(hidden_size, 1)

        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层和Softmax，用于分类
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 输入数据通过卷积层
        # 需要将输入 reshape，以适应 conv1d: (batch_size, input_size, seq_length)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = torch.relu(x)
        
        # 卷积输出调整回LSTM需要的维度: (batch_size, seq_length, cnn_filters)
        x = x.permute(0, 2, 1)
        
        # LSTM的输出
        out, (hn, cn) = self.lstm(x)
        
        # 聚合最终的输出，使用attention或mean/max pooling
        out = self.attention_pooling(out)
        
        # Dropout regularization
        out = self.dropout(out)
        
        # 将输出通过全连接层
        out = self.fc(out)
        
        # 使用Softmax进行分类
        return self.softmax(out)

    def attention_pooling(self, lstm_output):
        # 简单的 attention pooling 示例
        weights = torch.tanh(self.attn(lstm_output))
        weights = torch.softmax(weights, dim=1)
        weighted_output = torch.sum(weights * lstm_output, dim=1)
        return weighted_output
