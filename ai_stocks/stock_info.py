from .stock_data_handler import StockDataHandler
from .models.lstm_model import LSTMModule
from torch.autograd import Variable
import torch.nn as nn
import torch
import os

STOCK_PATH = 'data/stock'
class StockInfo:
    def __init__(self, 
        symbol, 
        epochs=100, 
        device='cpu', 
        gpu=0, 
        learning_rate=0.001, 
        input_size=21,  # 输入特征数量
        hidden_size=64, 
        output_size=1,  # 输出特征数量，假设只预测收盘价
        layers=2
    ):
        self.symbol = symbol
        self.epochs = epochs
        self.device = device
        self.gpu = gpu
        self.learning_rate = learning_rate
        self.stock_data_handler = StockDataHandler(symbol)
        self.model = LSTMModule(input_size=input_size, hidden_size=hidden_size, num_layers=layers, output_size=output_size)
        self.model_path = f'{STOCK_PATH}/{symbol}.pth'
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        
        self.trained = False
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        try:
            train_loader = self.stock_data_handler.get_train_loader()
            for i in range(self.epochs):
                total_loss = 0
                for idx, (data, label) in enumerate(train_loader):
                    if self.device == 'gpu':
                        data1 = data.squeeze(1).cuda(self.gpu)
                        label = label[:, 0].cuda(self.gpu)
                    else:
                        data1 = data.squeeze(1)
                        label = label[:, 0]

                    pred = self.model(Variable(data1))
                    pred = pred[:, 0]

                    loss = self.criterion(pred, label)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {i}, Total Loss: {total_loss}")
                if i % 10 == 0:
                    # 确保目录存在
                    if not os.path.exists(STOCK_PATH):
                        os.makedirs(STOCK_PATH)
                    torch.save({'state_dict': self.model.state_dict()}, self.model_path)
                    print(f'Epoch {i}, model saved.')

            self.trained = True
            torch.save({'state_dict': self.model.state_dict()}, self.model_path)
            print(f'model saved.')
        except Exception as e:
            print(f"Error during model training: {e}")



    def evaluate(self):
        self.model.to(self.device)
        preds = []
        labels = []
        test_loader = self.stock_data_handler.get_test_loader()
        for idx, (x, label) in enumerate(test_loader):
            print(label.shape)
            if self.device == 'gpu':
                x = x.squeeze(1).cuda(self.gpu)
            else:
                x = x.squeeze(1)
            pred = self.model(x)
            pred_list = pred.data.squeeze(1).tolist()  # 使用不同的变量名
            preds.append(pred_list[-1])  # 使用 append 而不是 extend
            labels.extend(label[:, 0].tolist())  # 假设我们只预测收盘价
        for i in range(len(preds)):
            print('预测值是%.2f,真实值是%.2f' % (
                preds[i] * self.stock_data_handler.price_max, labels[i] * self.stock_data_handler.price_max))


    def predict_next_day(self):
        self.model.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 在预测时不需要计算梯度
            # 获取最近的历史数据
            stock_data = self.stock_data_handler.get_recent_data()
            
            # 将 recent_data 转换为 PyTorch Tensor
            recent_data = torch.tensor(stock_data, dtype=torch.float32)

            # 确保数据形状符合模型输入要求
            if self.device == 'gpu':
                recent_data = recent_data.unsqueeze(0).cuda(self.gpu)
            else:
                recent_data = recent_data.unsqueeze(0)

            # 预测下一个时间步
            prediction = self.model(recent_data)
            predicted_price = prediction.item() * self.stock_data_handler.price_max  # 反归一化

            print(f"Predicted price for the next day is: {predicted_price:.2f}")
            return predicted_price