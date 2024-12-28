from ai_stocks.datas import BaseDataloader
import torch.nn as nn
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BasePrediction(ABC):
    def __init__(self, loader: BaseDataloader, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, file: str, epochs=10, device=None, scheduler=None, **kwargs):
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2)
        self.model.to(self.device)
        self.file = file
        if os.path.exists(file):
            try:
                checkpoint = torch.load(file, weights_only=True)  # 使用 weights_only=True
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
        
    def train(self, show_loss = False):
        train_loader = self.loader.get_train_loader()
        return self.do_train(train_loader, show_loss)

    def show_loss_history(self, loss_history):
        # Plot loss history
        plt.figure(figsize=(10, 5))
        plt.plot(range(self.epochs), loss_history)
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        
    def do_train(self, loader, show_loss = False):
        try:
            self.model.train()
            loss_history = []
            for i in range(self.epochs):
                total_loss = self.single_train(loader, i)
                loss_history.append(total_loss)

            self.trained = True
            torch.save({'state_dict': self.model.state_dict()}, self.file)
            if show_loss:
                self.show_loss_history(loss_history)
            return loss_history
        except Exception as e:
            print(f"Error during model training: {e}")
            
    def single_train(self, loader, i):
        total_loss = 0
        for idx, (data, label) in enumerate(loader):
            self.optimizer.zero_grad()
            output, label = self._model_func(data, label)
            loss = self.criterion(output, label)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        if i % 10 == 0:
            torch.save({'state_dict': self.model.state_dict()}, self.file)
            print(f'Epoch [{i+1}/{self.epochs}], Loss: {total_loss:.4f}, Learning Rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.scheduler:
            self.scheduler.step(total_loss)
        return total_loss
    
    def _model_func(self, data, label):
        if self.format_input:
            data, label = self.format_input(data, label)
        data, label = data.to(self.device), label.to(self.device)  # 确保数据在正确的设备上
        output = self.model(data)
        output, label = self.format_output(output, label)
        return output, label

    def train_test(self, show_loss = False):
        test_loader = self.loader.get_test_loader()
        return self.do_train(test_loader, show_loss)
            
    def format_output(self, output, label):
        return (output, label)
    
    def format_input(self, data, label):
        return (data, label)

    def evaluate(self):
        self.model.eval()
        test_loader = self.loader.get_test_loader()
        total_correct = 0
        total_wrong = 0
        for idx, (data, label) in enumerate(test_loader):
            output, label = self._model_func(data, label)
            correct, wrong = self.evaluate_ouput(output, label)
            total_correct += correct
            total_wrong += wrong
        return total_correct / (total_correct + total_wrong)
    
    @abstractmethod
    def evaluate_ouput(self, output, label):
        """创建预测实例，子类需要实现这个方法"""
        pass
   
    def get_recent_data(self):
        return self.loader.get_recent_data()

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            stock_data = self.get_recent_data()
            stock_data = torch.tensor(stock_data, dtype=torch.float32)
            if self.format_input:
                stock_data, _ = self.format_input(stock_data, None)
            stock_data = stock_data.to(self.device)
            prediction = self.model(stock_data)
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
