from .base_prediction import BasePrediction
from ai_stocks.datas import OperateDataloader
from ai_stocks.moduls import LSTMModule, OperateModule
import os
import torch.nn as nn
import torch
import numpy as np
class OperatePrediction(BasePrediction):
    name = 'operate'
    def __init__(self, stock_info, loader_kwargs = {}, model_kwargs = {}, opt_kwargs = {}, **kwargs):
        self.stock_info = stock_info
        loader = OperateDataloader(stock_info.symbol, **loader_kwargs)
        path = f'data/{stock_info.symbol}'
        file = f'{path}/operate-{loader.get_key()}.pth'
        if not os.path.exists(path):
            os.makedirs(path)

        input_size = loader.feature_length()
        model = OperateModule(input_size=input_size, output_size=3, **model_kwargs)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, **opt_kwargs)

        super().__init__(loader = loader, model= model, criterion = criterion, optimizer = optimizer, file = file, **kwargs)
        
    def do_train(self, data, label):
        data = data.to(self.device)
        label = label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, label.long())
        loss.backward()
        self.optimizer.step()
        return loss
        
    def do_evaluate(self, data, label, preds, labels):
        data = data.to(self.device)
        pred = self.model(data)
        prediction_values = np.argmax(pred.tolist(), axis=1)
        preds.append(prediction_values)
        labels.append(label.tolist())
        
    
        
    
