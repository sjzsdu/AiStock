from .base_prediction import BasePrediction
from astock_loaders import StockKlineLoader
from ai_stocks.moduls import KlineModule
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ai_stocks.utils import generate_short_md5

class KlinePrediction(BasePrediction):
    name = 'kline'
    
    def __init__(self, stock_info, loader_kwargs={}, model_kwargs={}, opt_kwargs={}, sch_kwargs={}, **kwargs):
        self.stock_info = stock_info
        _loader_kwargs = {
            'show_volume': True,
            'figsize': (4, 4), 
            'sequence_length': 60, 
            'predict_length': 5,
            'categtory_nums': (5,2,-2,-5)
        } | loader_kwargs
        loader = StockKlineLoader(stock_info.symbol, **_loader_kwargs)
        
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
        
        key = generate_short_md5(f'{str(_loader_kwargs)}-{str(_model_kwargs)}-{str(_opt_kwargs)}-{str(_sch_kwargs)}')
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
        
    def format_input(self, input):
        input = input.unsqueeze(1)
        return input
    
    def format_output(self, output, label):
        print('format_output', output.shape)
        output = output.squeeze(1)
        return (output, label)    
    
    
    def evaluate_recent(self, **kwargs):
        self.model.eval()
        preds = []
        labels = []
        test_loader = self.loader.get_recent_loader(**kwargs)
        for idx, (data, label) in enumerate(test_loader):
            data, label = data.to(self.device), label.to(self.device)
            pred = self.model(data)
            pred, label = self.format_output(pred, label)
            preds.append(pred.tolist())
            labels.append(label.tolist())
        self.create_dataframe(preds, labels)
        return self
    
