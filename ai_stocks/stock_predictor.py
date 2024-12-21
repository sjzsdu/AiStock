from ai_stocks.predictions import StockPrediction
from china_stock_data import StockMarket
from ai_stocks.utils import is_a_share

class StockPredictor:
    def __init__(self, symbols = None, index = None, start = None, limit = None, **kwargs):
        self.stocks: dict[str, StockPrediction] = {}
        self.kwargs = kwargs
        self.stocks_by_growth = None
        if symbols is not None:
            self.symbols = symbols
            for symbol in symbols:
                self.stocks[symbol] = StockPrediction(symbol, **kwargs)
        if index:
            self.stock_market = StockMarket(index)
            codes = self.stock_market['index_codes']

            # 当 start 或 limit 为 None 时的处理
            if start is None:
                start = 0
            if limit is None:
                limit = len(codes)

            # 确保 limit 不超过列表的长度
            actual_limit = min(start + limit, len(codes))

            for i in range(start, actual_limit):
                symbol = codes[i]
                if symbol not in self.stocks:
                    if is_a_share(symbol):
                        self.stocks[symbol] = StockPrediction(symbol=symbol, **kwargs)
                        
                        
    def train(self):
        for symbol, stock in self.stocks.items():
            stock.train()
            print(f'{symbol} trained')
            
            
    def predict(self, symbols: list[str] = None):
        for symbol in symbols:
            sp = StockPrediction(symbol=symbol, **self.kwargs)
            sp.predict()
            
    def predict_all(self):
        # 创建一个字典来保存每个增长百分比对应的股票列表
        if (self.stocks_by_growth is not None):
            return self
        self.stocks_by_growth = {i: [] for i in range(1, 6)}  # 5到1的字典

        for symbol, stock in self.stocks.items():
            st = stock.predict()
            per = st.grow_percent()
            
            # 将股票添加到对应的增长百分比列表中
            for i in range(5, 1, -1):  # 从5到1
                if per > i:
                    self.stocks_by_growth[i].append(stock)
                    break
                    
    def show_predicts(self, percent = 5):
        stocks = self.stocks_by_growth[percent]
        for stock in stocks:
            per = stock.grow_percent()
            print(f'{stock.loader.symbol} will grow {per}%')
            stock.show()
            
    
    
    
       