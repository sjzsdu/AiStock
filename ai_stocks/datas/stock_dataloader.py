import numpy as np
import pandas as pd
import math
from .base_dataloader import BaseDataloader
from ai_stocks import StockHandler

class StockDataloader(BaseDataloader):
    def __init__(self, symbol: str, **kwargs):
        self.symbol = symbol
        self.data_handler = StockHandler(symbol)
        feature_cols = ['开盘', '收盘', '最高', '最低', '成交量']
        label_cols = ['收盘']
        dataset = self.data_handler.data[feature_cols]
        
        test_ratio = kwargs.pop('test_ratio', 0.2)
        
        self.price_cols = ['开盘', '收盘', '最高', '最低']
        self.other_cols = ['成交量']
        
        if 'test_ratio' in kwargs:
            test_ratio = kwargs.pop('test_ratio')
        else:
            test_ratio = 0.2
        if 'sequence_length' not in kwargs:
            kwargs['sequence_length'] = 90
        if 'predict_length' not in kwargs:
            kwargs['predict_length'] = 5
        
        super().__init__(dataset, feature_cols, label_cols, test_ratio=test_ratio, **kwargs)
        
    def _create_sequences(self, data, is_predict = False):
        labels = data[self.label_cols]
        features = data[self.feature_cols]
        X, Y = [], []
        predict_length = self.predict_length if not is_predict else 0

        for i in range(len(data) - self.sequence_length - predict_length):
            feature_start = i
            feature_end = i + self.sequence_length
            label_start = feature_end
            label_end = label_start + predict_length
            
            x_seq, y_seq = self.format_data(features.iloc[feature_start:feature_end].copy(),
                                            labels.iloc[label_start:label_end].copy(), is_predict)
            X.append(x_seq)
            Y.append(y_seq)
                
        arr_x = np.array(X, dtype=np.float32)
        arr_y = np.array(Y, dtype=np.float32)
        return arr_x, arr_y
    
    def normalize_price(self, max_price, min_price, val):
        return (val - min_price) / (max_price - min_price)
    
    def unnormalize_price(self, val):
        return val * (self.max_price - self.min_price) + self.min_price
    
    def min_max_normalize(self, series):
        return (series - series.min()) / (series.max() - series.min())
    
    def format_data(self, features, labels, is_predict = False):
        max_price = math.ceil(features['最高'].max())
        min_price = math.floor(features['最低'].min())
        if (is_predict):
            self.max_price = max_price
            self.min_price = min_price
        
        # Normalize prices
        features.loc[:, self.price_cols] = features[self.price_cols].apply(
            lambda x: self.normalize_price(max_price, min_price, x)
        )
        # Normalize other features
        features.loc[:, self.other_cols] = features[self.other_cols].apply(self.min_max_normalize)
        # Normalize labels
        labels.loc[:, self.label_cols] = labels[self.label_cols].apply(
            lambda x: self.normalize_price(max_price, min_price, x)
        )
        
        return features.values, labels.values
    
    
