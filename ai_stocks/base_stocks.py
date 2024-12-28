from abc import ABC, abstractmethod
from china_stock_data import StockMarket
from ai_stocks.utils import is_a_share

class BaseStocks(ABC):
    """
    BaseStocks 类是一个抽象基类，用于管理股票数据和预测实例。

    该类提供了两种初始化方式：
    1. 通过传递股票代码列表（symbols），为每个代码创建一个预测实例。
    2. 通过传递股票市场指数（index），获取该指数下的所有股票代码，并为这些代码创建预测实例。
    """
    def __init__(self, symbols=None, index=None, start=None, limit=None, **kwargs):
        """
        初始化 BaseStocks 实例。

        参数:
            symbols (list): 股票代码列表。
            index (str): 股票市场指数。
            start (int): 从索引开始获取股票代码的位置。
            limit (int): 获取股票代码的数量限制。
            **kwargs: 传递给预测实例的额外参数。
        """
        self.stocks = {}
        self.kwargs = kwargs
        if symbols is not None:
            self.symbols = symbols
            for symbol in symbols:
                self.stocks[symbol] = self.create_prediction_instance(symbol, **kwargs)
        if index:
            self.stock_market = StockMarket(index)
            codes = self.stock_market['index_codes']
            
            if start is None:
                start = 0
            if limit is None:
                limit = len(codes)

            actual_limit = min(start + limit, len(codes))

            for i in range(start, actual_limit):
                symbol = codes[i]
                if symbol not in self.stocks:
                    if is_a_share(symbol):
                        self.stocks[symbol] = self.create_prediction_instance(symbol, **kwargs)

    @abstractmethod
    def create_prediction_instance(self, symbol, **kwargs):
        """创建预测实例，子类需要实现这个方法"""
        pass
    
    def train(self, **kwargs):
        for symbol, stock in self.stocks.items():
            stock.train(**kwargs)
            
    def train_test(self, **kwargs):
        for symbol, stock in self.stocks.items():
            stock.train_test(**kwargs)
         
    def evaluate(self, **kwargs):
        percents = []
        res = {}
        for symbol, stock in self.stocks.items():
            percent = stock.evaluate(**kwargs) 
            res[symbol] = percent
            percents.append(percent)
        return res, sum(percents) / len(percents)
            
    def predict(self, **kwargs):
        res = {}
        for symbol, stock in self.stocks.items():
            out = stock.predict(**kwargs)
            res[symbol] = out
        return res