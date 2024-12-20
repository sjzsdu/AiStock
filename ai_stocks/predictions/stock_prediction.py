from .base_prediction import BasePrediction
from ai_stocks.datas import StockDataloader
from ai_stocks.moduls import PriceModule
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ai_stocks.utils import generate_short_md5
import matplotlib.pyplot as plt
import numpy as np

class StockPrediction(BasePrediction):
    name = 'stock'
    
    def __init__(self, symbol, loader_kwargs={}, model_kwargs={}, opt_kwargs={}, sch_kwargs={}, **kwargs):
        loader = StockDataloader(symbol, **loader_kwargs)
        
        input_size = loader.feature_length()
        output_size = 5
        
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
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2)
        key = generate_short_md5(f'{str(_model_kwargs)}-{str(_opt_kwargs)}-{str(_sch_kwargs)}')
        path = f'data/{key}'
        file = f'{path}/common.pth'
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
        if label is not None:
            label = label[:,:,0]
        return (output, label)    
    
    
    def predict(self):
        self.model.eval()
        with torch.no_grad():
            recent_loader = self.loader.get_recent_loader()
            first_item = next(iter(recent_loader))
            inputs, labels = first_item
            inputs = inputs[-1:]
            prices = inputs[-1][:,1]
            recent_data = inputs.detach().requires_grad_(True).to(self.device)
            output = self.model(recent_data)
            self.vals = prices[-30:]
            self.predicts = output[0]
        return self
    
    def grow_percent(self):
        # 计算涨幅
        growth = (self.predicts[0] - self.vals[-1]) * 100 / self.vals[-1]
        return round(growth.item(), 2)   
    
    def show(self):
        vals_np = self.vals.cpu().numpy()
        predicts_np = self.predicts.cpu().numpy()
        
        vals_np = [self.loader.unnormalize_price(val) for val in vals_np]
        predicts_np = [self.loader.unnormalize_price(val) for val in predicts_np]

        vals_index_list = list(range(len(vals_np)))
        predicts_index_list = list(range(len(vals_np), len(vals_np) + len(predicts_np)))

        plt.figure(figsize=(10, 6))
        plt.plot(vals_index_list, vals_np, label='Actual Values', color='green', marker='x')
        plt.plot(predicts_index_list, predicts_np, label='Predictions', color='red', marker='o')

        plt.title('Predictions vs Actual Values' + self.loader.symbol)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        
