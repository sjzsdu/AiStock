from .base_prediction import BasePrediction
from astock_loaders import StockKlineLoader
from ai_stocks.moduls import KlineModule
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ai_stocks.utils import generate_short_md5
import numpy as np

class KlinePrediction(BasePrediction):
    name = 'kline'
    
    def __init__(self, symbol, loader_kwargs={}, model_kwargs={}, opt_kwargs={}, sch_kwargs={}, **kwargs):
        self.symbol = symbol
        _loader_kwargs = {
            'show_volume': True,
            'figsize': (3, 3), 
            'sequence_length': 60, 
            'predict_length': 5,
            'categtory_nums': (5,2,-2,-5)
        } | loader_kwargs
        loader = StockKlineLoader(symbol, **_loader_kwargs)
        
        path = 'data/kline/'
        
        
        _model_kwargs = {
            'num_classes': len(_loader_kwargs['categtory_nums']) + 1
        } | model_kwargs
        
        model = KlineModule(**_model_kwargs)
        
        criterion = nn.CrossEntropyLoss()
        _opt_kwargs = {
            'lr': 0.001,
            'weight_decay': 1e-5
        } | opt_kwargs
        optimizer = torch.optim.Adam(model.parameters(), **_opt_kwargs)
        
        _sch_kwargs = {
            'patience': 5,
            'factor': 0.2
        } | sch_kwargs
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', **_sch_kwargs)
        
        key = generate_short_md5(f'{str(_loader_kwargs)}-{str(_model_kwargs)}')
        file = f'{path}/kline-{key}.pth'
        if not os.path.exists(path):
            os.makedirs(path)

        super().__init__(
            loader = loader, 
            model= model, 
            criterion = criterion, 
            optimizer = optimizer, 
            scheduler = scheduler, 
            file = file, 
            **kwargs
        )
        
    def format_input(self, input, label):
        input = input.unsqueeze(1)
        return input, label
    
    def format_output(self, output, label):
        output = output.squeeze(1)
        return (output, label) 
    
    def evaluate_ouput(self, output, label):
        output = torch.argmax(output, dim=1)
        label = torch.argmax(label, dim=1)
        correct_count = (output == label).sum().item()
        incorrect_count = (output != label).sum().item()
        return correct_count, incorrect_count
    
    def get_recent_data(self):
        data = self.loader.get_recent_data()
        return np.array([data])
    
    def show(self): 
        raise NotImplementedError()
    
