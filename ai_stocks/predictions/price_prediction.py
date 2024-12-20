from .base_prediction import BasePrediction
from ai_stocks.datas import PriceDataloader
from ai_stocks.moduls import PriceModule
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ai_stocks.utils import generate_short_md5

class PricePrediction(BasePrediction):
    name = 'price'
    
    def __init__(self, stock_info, loader_kwargs={}, model_kwargs={}, opt_kwargs={}, sch_kwargs={}, **kwargs):
        self.stock_info = stock_info
        loader = PriceDataloader(stock_info.symbol, **loader_kwargs)
        path = f'data/{stock_info.symbol}'
        input_size = loader.feature_length()
        output_size = loader.label_length()
        
        _model_kwargs = {
            'input_size': input_size,
            'output_size': output_size
        } | model_kwargs
        
        model = PriceModule(**_model_kwargs)
        
        criterion = nn.MSELoss()
        _opt_kwargs = {
            'lr': 0.01,
            'weight_decay': 1e-5
        } | opt_kwargs
        optimizer = torch.optim.Adam(model.parameters(), **_opt_kwargs)
        
        _sch_kwargs = {
            'step_size': 10,
            'gamma': 0.1
        } | sch_kwargs
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, **_sch_kwargs)
        
        key = generate_short_md5(f'{loader.get_key()}-{str(_model_kwargs)}-{str(_opt_kwargs)}-{str(_sch_kwargs)}')
        file = f'{path}/price-{key}.pth'
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
        
    def format_output(self, output, label):
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
    
