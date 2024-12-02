from ai_stocks.datas import BaseDataloader
from torch.autograd import Variable
import torch.nn as nn
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class BasePrediction:
    def __init__(self, loader: BaseDataloader, model: nn.Module , criterion: nn.Module, optimizer: torch.optim.Optimizer, file: str, epochs = 100, device='cpu', **kwargs):
        self.epochs = epochs
        self.device = device
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.model.to(self.device)
        self.file = file
        if os.path.exists(file):
            try:
                checkpoint = torch.load(file, weights_only=True)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.trained = True
            except Exception as e:
                print(f"An error occurred: {e}")
                os.remove(file)
                self.trained = False
        else:
            self.trained = False
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def train(self):
        try:
            train_loader = self.loader.get_train_loader()
            for i in range(self.epochs):
                total_loss = 0
                for idx, (data, label) in enumerate(train_loader):
                    loss = self.do_train(data, label)
                total_loss += loss.item()
                if i % 10 == 0:
                    torch.save({'state_dict': self.model.state_dict()}, self.file)
                print(f'Epoch [{i+1}/{self.epochs}], Loss: {loss.item():.4f}')

            self.trained = True
            torch.save({'state_dict': self.model.state_dict()}, self.file)
            print(f'Training finished.')
        except Exception as e:
            print(f"Error during model training: {e}")
            
    def do_train(self, data, label):
        pass

    def evaluate(self):
        preds = []
        labels = []
        test_loader = self.loader.get_test_loader()
        for idx, (data, label) in enumerate(test_loader):
            self.do_evaluate(data, label, preds, labels)
        
        self.create_dataframe(preds, labels)
        return self
    
    def do_evaluate(self, data, label, preds, labels):
       pass
    
    def predict(self):
        self.model.eval()
        with torch.no_grad():
            stock_data = self.loader.get_recent_data()
            recent_data = torch.tensor(stock_data, dtype=torch.float32)
            if self.device == 'gpu':
                recent_data = recent_data.unsqueeze(1).cuda(self.gpu)
            else:
                recent_data = recent_data.unsqueeze(1)

            prediction = self.model(recent_data)

            return prediction
    
    def create_dataframe(self, preds, labels):
        res_pred = []
        res_label = []
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                pred_price = self.loader.data_handler.unnormalize_price(preds[i][j])
                label_price = self.loader.data_handler.unnormalize_price(labels[i][j])
                res_pred.append(pred_price)
                res_label.append(label_price)
        self.df = pd.DataFrame({'Predictions': res_pred, 'Actual Values': res_label})
        return self.df
    
    def show(self):
        row = self.df
        index_list = list(range(len(row['Predictions'])))
        plt.figure(figsize=(10, 6))
        plt.plot(index_list, row['Predictions'], label='Predictions', color='blue', marker='o')
        plt.plot(index_list, row['Actual Values'], label='Actual Values', color='red', marker='x')

        plt.title('Predictions vs Actual Values')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        
        