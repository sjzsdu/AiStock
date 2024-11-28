from china_stock_data import StockData
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from matplotlib import pyplot as plt
import math

MAX_PERCENT= 1.1
MIN_PERCENT= 0.9

class StockDataHandler:
    def __init__(self, symbol, sequence_length=30, batch_size=32, test_ratio=0.2):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.stock_data = StockData(symbol, days=365 * 100)
        self.data = None
        self.cached_dataset = None  # 缓存数据集
        self.price_cols = ['开盘', '收盘', '最高', '最低', '涨跌额', '平均', '加权平均', '平均成本', '90成本-低', '90成本-高', '70成本-低', '70成本-高']
        self.other_cols = ['日期', '成交量', '成交额']
        self.handle_data()
        
    def unnormalize_price(self, val):
        return val * (self.max_price - self.min_price) + self.min_price
    
    def normalize_price(self, val):
        return (val - self.min_price) / (self.max_price - self.min_price)
    
    def min_max_normalize(self, series):
        return (series - series.min()) / (series.max() - series.min())

    def handle_data(self):
        kline = self.stock_data.kline
        chip = self.stock_data.chip
        data = pd.merge(kline, chip, on='日期', how='left').fillna(0)
        data.drop(['股票代码'], axis=1, inplace=True)
        data['日期'] = pd.to_datetime(data['日期']).dt.strftime('%Y%m%d').astype(int)
        self.max_price = math.ceil(data['最高'].max() * MAX_PERCENT)
        self.min_price = math.floor(data['最低'].min() * MIN_PERCENT)

        # Min-max normalization
        data[self.price_cols] = data[self.price_cols].apply(
            lambda x: self.normalize_price(x)
        )
        
        data[self.other_cols] = data[self.other_cols].apply(self.min_max_normalize)
        
        self.data = data
        return self

    def create_sequences(self):
        X, Y = [], []
        for i in range(len(self.data) - self.sequence_length):
            X.append(self.data.iloc[i:i + self.sequence_length].values)
            Y.append(self.data.iloc[i + self.sequence_length][['收盘']].values)
        
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def get_dataset(self):
        if self.cached_dataset is None:
            X, Y = self.create_sequences()
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_ratio, shuffle=False)
            self.cached_dataset = (X_train, X_test, Y_train, Y_test)
        return self.cached_dataset

    def get_train_loader(self):
        X_train, _, Y_train, _ = self.get_dataset()
        train_dataset = StockDataset(X_train, Y_train)
        return DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_loader(self):
        _, X_test, _, Y_test = self.get_dataset()
        test_dataset = StockDataset(X_test, Y_test)
        return DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_recent_data(self):
        # 获取最近的 sequence_length 天的数据
        recent_data = self.data.iloc[-self.sequence_length:].values
        return np.array(recent_data, dtype=np.float32)
    
    def draw(self):
        # Get the training and testing datasets
        X_train, X_test, Y_train, Y_test = self.get_dataset()

        # Plotting the training and testing datasets
        plt.figure(figsize=(12, 6))

        # Plot training data
        plt.subplot(1, 2, 1)
        for i in range(min(len(X_train), 5)):  # Plot first 5 sequences for example
            plt.plot(range(self.sequence_length), X_train[i, :, -1], label=f'Sequence {i+1}')
        plt.title('Training Data Sequences')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Price')
        plt.legend()

        # Plot testing data
        plt.subplot(1, 2, 2)
        for i in range(min(len(X_test), 5)):  # Plot first 5 sequences for example
            plt.plot(range(self.sequence_length), X_test[i, :, -1], label=f'Sequence {i+1}')
        plt.title('Testing Data Sequences')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Price')
        plt.legend()

        plt.tight_layout()
        plt.show()


class StockDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

