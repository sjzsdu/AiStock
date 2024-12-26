from .base_prediction import BasePrediction
from .operate_prediction import OperatePrediction
from .price_prediction import PricePrediction
from .stock_prediction import StockPrediction
from .kline_prediction import KlinePrediction

predictions_list = [OperatePrediction, PricePrediction, StockPrediction, KlinePrediction]