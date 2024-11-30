from .base_predictiion import BasePrediction
from ai_stocks.datas import OperateDataloader
from ai_stocks.moduls import LSTMModule, OperateModule
import os
import torch.nn as nn
import torch

class OperatePrediction(BasePrediction):
    name = 'operate'
    def __init__(self, stock_info, loader_kwargs = {}, model_kwargs = {}, opt_kwargs = {}, **kwargs):
        self.stock_info = stock_info
        loader = OperateDataloader(stock_info.symbol, **loader_kwargs)
        path = f'data/{stock_info.symbol}'
        file = f'{path}/operate.pth'
        if not os.path.exists(path):
            os.makedirs(path)

        input_size = loader.feature_length()
        output_size = loader.label_length()
        model = LSTMModule(input_size=input_size, output_size=output_size, **model_kwargs)
        
        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=1, **opt_kwargs)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        super().__init__(loader = loader, model= model, criterion = criterion, optimizer = optimizer, file = file, **kwargs)
        
    
