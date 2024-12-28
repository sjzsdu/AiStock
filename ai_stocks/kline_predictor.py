from ai_stocks.predictions import KlinePrediction
from china_stock_data import StockMarket
from ai_stocks.utils import is_a_share
from .base_stocks import BaseStocks

class KlinePredictor(BaseStocks):
    def __init__(self, symbols = None, index = None, start = None, limit = None, **kwargs):
        super().__init__(symbols, index, start, limit, **kwargs)
               
    def create_prediction_instance(self, symbol, **kwargs):
        return KlinePrediction(symbol, **kwargs)         
                        
    
            
    
    
    
       