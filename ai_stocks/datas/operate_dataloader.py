from .base_dataloader import BaseDataloader
from ai_stocks import StockDataHandler

class OperateDataloader(BaseDataloader):
    def __init__(self, symbol: str, **kwargs):
        self.data_handler = StockDataHandler(symbol)
        feature_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率',
       '平均', '加权平均', '获利比例', '平均成本', '90成本-低', '90成本-高', '90集中度', '70成本-低',
       '70成本-高', '70集中度', '操盘']
        dataset = self.data_handler.data[feature_cols]
        self.cols = feature_cols + ['操盘']
        super().__init__(dataset, ['操盘'],  **kwargs)
