import torch
import torch.nn as nn
import torch.nn.functional as F

class KlineModule(nn.Module):
    def __init__(self, num_classes=5):
        super(KlineModule, self).__init__()
        
        # 第一部分卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        # 第二部分卷积层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        # 第三部分卷积层
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)
        
        # 全连接层
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.global_avg_pool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)
        return x

