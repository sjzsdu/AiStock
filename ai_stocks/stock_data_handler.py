from china_stock_data import StockData, StockMarket
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib.dates as mdates
from scipy.signal import find_peaks
from ai_stocks.config import DATE_FORMAT

class StockDataHandler:
    def __init__(self, symbol):
        self.symbol = symbol
        self.stock_data = StockData(symbol, days=365 * 10)
        self.market_data = StockMarket()

        self.price_cols = ['开盘', '收盘', '最高', '最低', '涨跌额', '平均', '加权平均', '平均成本', '90成本-低', '90成本-高', '70成本-低', '70成本-高']
        self.other_cols = ['成交量', '成交额', 'us_volume', 'us_price', '沪深300指数']
        self.handle_data()
        self.format_data()
        
    def handle_data(self):
        kline = self.stock_data.kline
        chip = self.stock_data.chip
        origin = pd.merge(kline, chip, on='日期', how='left').fillna(0)
        
        us_index = self.market_data.us_index
        us_index = us_index.rename(columns={'date': '日期'})
        us_index = us_index.rename(columns={'close': 'us_price'})
        us_index = us_index.rename(columns={'volume': 'us_volume'})
        market_motion = self.market_data.market_motion
        origin = pd.merge(origin, market_motion, on='日期', how='left').fillna(0)
        origin = pd.merge(origin, us_index, on='日期', how='left').fillna(0)
        origin.fillna(0, inplace=True)

        close_prices = np.array(origin['收盘'].tolist())
        peaks, troughs = self.find_peaks_and_troughs(close_prices)
        origin['操盘'] = origin.apply(lambda row: self.determine_action(row.name, close_prices, peaks, troughs), axis=1)
        self.origin = origin
        return origin
        
    def unnormalize_price(self, val):
        return val * (self.max_price - self.min_price) + self.min_price
    
    def normalize_price(self, val):
        return (val - self.min_price) / (self.max_price - self.min_price)
    
    def min_max_normalize(self, series):
        return (series - series.min()) / (series.max() - series.min())
    
    def find_peaks_and_troughs(self, prices, distance=5, prominence=1):
        peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
        troughs, _ = find_peaks(-prices, distance=distance, prominence=prominence)
        return peaks, troughs

    def determine_action(self, row_index, prices, peaks, troughs):
        for trough in troughs:
            if abs(row_index - trough) <= 5 and prices[row_index] <= prices[trough] * 1.05:
                return "buy"
        for peak in peaks:
            if abs(row_index - peak) <= 3 and prices[row_index] >= prices[peak] * 0.95:
                return "sell"

        return "inaction"
        
    def format_data(self):
        data = self.origin.copy()
        data.drop(['股票代码'], axis=1, inplace=True)
        self.max_price = math.ceil(data['最高'].max())
        self.min_price = math.floor(data['最低'].min())

        data[self.price_cols] = data[self.price_cols].apply(
            lambda x: self.normalize_price(x)
        )
        data[self.other_cols] = data[self.other_cols].apply(self.min_max_normalize)
        data['操盘'] = data['操盘'].apply(lambda x: 0 if x == 'buy' else (1 if x == 'sell' else 2))

        data['年'] = pd.to_datetime(data['日期'], format=DATE_FORMAT).dt.year / 2000
        data['月'] = pd.to_datetime(data['日期'], format=DATE_FORMAT).dt.month / 12
        data['日'] = pd.to_datetime(data['日期'], format=DATE_FORMAT).dt.day / 30
        data['星期'] = pd.to_datetime(data['日期'], format=DATE_FORMAT).dt.dayofweek / 7
        self.data = data
        return self

    
    def plot_signals(self, days=None):
        if self.origin is None:
            print("No data to plot. Please run handle_data() first.")
            return
        
        self.origin['日期'] = pd.to_datetime(self.origin['日期'])
        
        if days is not None:
            data_to_plot = self.origin.tail(days)
        else:
            data_to_plot = self.origin
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        plt.plot(data_to_plot['日期'], data_to_plot['收盘'], label='Close Price', color='blue', linewidth=1.5)
        
        buy_signals = data_to_plot[data_to_plot['操盘'] == 'buy']
        sell_signals = data_to_plot[data_to_plot['操盘'] == 'sell']
        
        plt.scatter(buy_signals['日期'], buy_signals['收盘'], label='Buy Signal', marker='^', color='green', s=100)
        plt.scatter(sell_signals['日期'], sell_signals['收盘'], label='Sell Signal', marker='v', color='red', s=100)

        ax.set_title('Stock Price with Buy and Sell Signals')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))  # 控制最大的日期标签数量

        ax.grid(False)
        fig.autofmt_xdate()

        plt.show()
