from .base_dataloader import BaseDataloader
from ai_stocks import StockDataHandler

class PriceDataloader(BaseDataloader):
    def __init__(self, symbol: str, **kwargs):
        self.data_handler = StockDataHandler(symbol)
        feature_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '涨跌额',
       '平均', '加权平均', '获利比例', '平均成本', '90成本-低', '90成本-高', '90集中度', '70成本-低',
       '70成本-高', '70集中度', '年', '月', '日', '星期']
        dataset = self.data_handler.data[feature_cols]
        self.cols = feature_cols + ['收盘']
        if 'test_ratio' in kwargs:
            test_ratio = kwargs.pop('test_ratio')
        else:
            test_ratio = 0.2
        super().__init__(dataset, ['收盘'], test_ratio = test_ratio, **kwargs)
        
