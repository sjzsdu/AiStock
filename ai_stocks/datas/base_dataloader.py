from sklearn.model_selection import train_test_split
import numpy as np
from pandas import DataFrame
from typing import List
from .stock_dataset import StockDataset
from torch.utils.data import DataLoader
from ai_stocks.utils import generate_short_md5
import pandas as pd

class BaseDataloader:
    """
    该类是一个数据加载器的基类，用于加载时间序列数据并生成训练集和测试集。

    参数:
    - data (DataFrame): 包含特征和标签的数据框。
    - feature_cols (List[str]): 特征列的列表。
    - label_cols (List[str]): 标签列的列表。
    - sequence_length (int): 序列长度，默认为 30。
    - batch_size (int): 批量大小，默认为 32。
    - test_ratio (float): 测试集比例，默认为 0.2。

    方法:
    - create_sequences(): 从数据中创建序列。
    - get_dataset(): 获取数据集，如果数据集不存在，则创建它。
    - get_train_loader(): 获取训练集数据加载器。
    - get_test_loader(): 获取测试集数据加载器。
    """
    def __init__(self, data: DataFrame, feature_cols: List[str], label_cols: List[str], sequence_length=30, predict_length=1, batch_size=32, test_ratio=0.1):
        """
        初始化 BaseDataloader 类的实例。

        参数:
        - data (DataFrame): 包含特征和标签的数据框。
        - feature_cols (List[str]): 特征列的列表。
        - label_cols (List[str]): 标签列的列表。
        - sequence_length (int): 序列长度，默认为 30。
        - batch_size (int): 批量大小，默认为 32。
        - test_ratio (float): 测试集比例，默认为 0.2。
        """
        self.data = data
        self.sequence_length = sequence_length
        self.predict_length = predict_length
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.label_cols = label_cols
        self.feature_cols = feature_cols
        self.cached_dataset = None  # 缓存数据集

    def create_sequences(self, days = None, is_predict = False):
        """
        从数据中创建序列。

        返回:
        - X (numpy.ndarray): 特征序列。
        - Y (numpy.ndarray): 标签序列。
        """
        data = self.data
        if (is_predict):
            return self._create_sequences(data, is_predict = is_predict)
        if days is not None:
            data = self.data.copy()
            last_row = data.iloc[-1]
            expanded_rows = pd.DataFrame([last_row] * days, columns=data.columns)
            print('expanded rows', expanded_rows)
            data = pd.concat([data, expanded_rows], ignore_index=True)
            return self._create_sequences(data)

        if not hasattr(self, 'cached_X') or not hasattr(self, 'cached_Y'):
            X, Y = self._create_sequences(data)
            self.cached_X = X
            self.cached_Y = Y

        return self.cached_X, self.cached_Y
    
    def _create_sequences(self, data, is_predict = False):
        labels = data[self.label_cols]
        features = data[self.feature_cols]
        X, Y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(features.iloc[i:i + self.sequence_length].values)
            if i < len(labels) - self.sequence_length - 1:
                Y.append(labels.iloc[i + self.sequence_length + 1].values)
            else:
                Y.append(labels.iloc[i + self.sequence_length].values)

        arr_x = np.array(X, dtype=np.float32)
        arr_y = np.array(Y, dtype=np.float32)

        if len(arr_x.shape) == 1 or (arr_y.shape[-1] == 1):
           arr_y = arr_y.flatten()
        return (arr_x, arr_y)

    def get_dataset(self):
        """
        获取数据集，如果数据集不存在，则创建它。

        返回:
        - X_train (numpy.ndarray): 训练集特征。
        - X_test (numpy.ndarray): 测试集特征。
        - Y_train (numpy.ndarray): 训练集标签。
        - Y_test (numpy.ndarray): 测试集标签。
        """
        if self.cached_dataset is None:
            X, Y = self.create_sequences()
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_ratio, shuffle=False)
            self.cached_dataset = (X_train, X_test, Y_train, Y_test)
        return self.cached_dataset

    def get_train_loader(self):
        """
        获取训练集数据加载器。

        返回:
        - DataLoader: 训练集数据加载器。
        """
        X_train, _, Y_train, _ = self.get_dataset()
        train_dataset = StockDataset(X_train, Y_train)
        return DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 4)

    def get_test_loader(self):
        """
        获取测试集数据加载器。

        返回:
        - DataLoader: 测试集数据加载器。
        """
        _, X_test, _, Y_test = self.get_dataset()
        test_dataset = StockDataset(X_test, Y_test)
        return DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 4)
    
    def get_recent_loader(self, batchs = None, days = None):
        batch_size = self.batch_size
        if batchs is not None:
            batch_size = batchs * self.batch_size
        
        X, Y = self.create_sequences(days = days, is_predict=True)
        train_dataset = StockDataset(X[-batch_size:], Y[-batch_size:])
        return DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)
    
    
    def get_recent_data(self, days = 0):
        # 获取最近的 sequence_length 天的数据
        start_idx = -self.sequence_length - days if days != 0 else -self.sequence_length
        end_idx = -days if days != 0 else None

        # 打印调试信息（可选）
        print('recentData', start_idx, end_idx)

        # 获取所需的数据片段
        recent_data = self.data.iloc[start_idx:end_idx].values

        # 返回以 np.float32 类型的 numpy 数组
        return np.array(recent_data, dtype=np.float32)
        
    def feature_length(self):
        return len(self.feature_cols)
    
    def label_length(self):
        return len(self.label_cols)
    
    def get_key(self):
        data = ','.join(self.cols)
        return generate_short_md5(data)
