from .base_predictiion import BasePrediction
from ai_stocks.datas import PriceDataloader
from ai_stocks.moduls import PriceModule
import os
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
from ai_stocks.utils import generate_short_md5

class PricePrediction(BasePrediction):
    name = 'price'
    def __init__(self, stock_info, loader_kwargs = {}, model_kwargs = {}, opt_kwargs = {}, **kwargs):
        self.stock_info = stock_info
        loader = PriceDataloader(stock_info.symbol, **loader_kwargs)
        path = f'data/{stock_info.symbol}'
        key = generate_short_md5(f'{loader.get_key()}-{str(model_kwargs)}-{str(opt_kwargs)}')
        file = f'{path}/price-{key}.pth'
        if not os.path.exists(path):
            os.makedirs(path)

        input_size = loader.feature_length()
        output_size = loader.label_length()
        model = PriceModule(input_size=input_size, output_size=output_size, **model_kwargs)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, **opt_kwargs)
        # self.scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        super().__init__(loader = loader, model= model, criterion = criterion, optimizer = optimizer, file = file, **kwargs)
        
    def do_train(self, data, label):
        data = data.to(self.device)
        label = label.to(self.device)
        output = self.model(data)
        output = output.squeeze(1)
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        return loss
        
    def do_evaluate(self, data, label, preds, labels):
        data = data.to(self.device)
        pred = self.model(data)
        pred = pred.squeeze(1)
        preds.append(pred.tolist())
        labels.append(label.tolist())
        
    def predict(self, days = 0):
        with torch.no_grad():
            stock_data = self.loader.get_recent_data(days)
            recent_data = torch.tensor(stock_data, dtype=torch.float32)
            if self.device == 'gpu':
                recent_data = recent_data.unsqueeze(1).cuda(self.gpu)
            else:
                recent_data = recent_data.unsqueeze(1)

            pred = self.model(recent_data)
            pred = pred[0]
            return self.loader.data_handler.unnormalize_price(pred[-1])      
    