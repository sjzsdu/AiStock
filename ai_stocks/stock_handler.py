from china_stock_data import StockData, StockMarket
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib.dates as mdates
from scipy.signal import find_peaks
from ai_stocks.config import DATE_FORMAT

class StockHandler:
    def __init__(self, symbol):
        self.symbol = symbol
        self.init()
        
    def init(self):
        self.stock_data = StockData(self.symbol, days=365 * 10)
        self.market_data = StockMarket()

        self.price_cols = ['开盘', '收盘', '最高', '最低']
        self.other_cols = ['成交量', '成交额']
        self.handle_data()
        self.format_data()
        
    def handle_data(self):
        kline = self.stock_data.kline
        origin = kline.copy()
        self.origin = origin
        return origin
        
    def unnormalize_price(self, val):
        return val * (self.max_price - self.min_price) + self.min_price
    
    def normalize_price(self, val):
        return (val - self.min_price) / (self.max_price - self.min_price)
    
    def min_max_normalize(self, series):
        return (series - series.min()) / (series.max() - series.min())
        
    def format_data(self):
        data = self.origin.copy()
        data.drop(['股票代码'], axis=1, inplace=True)
        self.max_price = math.ceil(data['最高'].max())
        self.min_price = math.floor(data['最低'].min())
        data['成交量'] = data['成交量'].astype(float)
        self.data = data
        return self
