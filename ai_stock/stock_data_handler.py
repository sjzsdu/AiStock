from china_stock_data import StockData
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

class StockDataHandler:
    def __init__(self, symbol, sequence_length=30, batch_size=32):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.stock_data = StockData(symbol, days=365 * 100)
        self.data = None
        self.cached_dataset = None  # 缓存数据集
        self.price_cols = ['开盘', '收盘', '最高', '最低', '涨跌额', '平均', '加权平均', '平均成本', '90成本-低', '90成本-高', '70成本-低', '70成本-高']
        self.price_max = 1000
        self.volumn_cols = ['成交量', '成交额']
        self.volumn_max = 10000000000
        self.handle_data()

    def handle_data(self):
        kline = self.stock_data.kline
        chip = self.stock_data.chip
        data = pd.merge(kline, chip, on='日期', how='left').fillna(0)
        data.drop(['股票代码'], axis=1, inplace=True)
        data['日期'] = pd.to_datetime(data['日期']).dt.strftime('%Y%m%d').astype(int)

        # Min-max normalization
        data[self.price_cols] = data[self.price_cols] / self.price_max
        data[self.volumn_cols] = data[self.volumn_cols] / self.volumn_max
        
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
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
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

