from ai_stocks.datas import BaseDataloader
import torch.nn as nn
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

class BasePrediction:
    def __init__(self, loader: BaseDataloader, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, file: str, epochs=100, device='cpu', scheduler=None, **kwargs):
        self.epochs = epochs
        self.device = device
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
        
    def train(self):
        try:
            train_loader = self.loader.get_train_loader()
            self.model.train()
            loss_history = []
            for i in range(self.epochs):
                total_loss = 0
                for idx, (data, label) in enumerate(train_loader):
                    data, label = data.to(self.device), label.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    output = self.format_output(output)
                    loss = self.criterion(output, label)
                    loss.backward()
                    total_loss += loss.item()
                    self.optimizer.step()
                loss_history.append(total_loss)
                if i % 10 == 0:
                    torch.save({'state_dict': self.model.state_dict()}, self.file)
                    print(f'Epoch [{i+1}/{self.epochs}], Loss: {total_loss:.4f}, Learning Rate: {self.optimizer.param_groups[0]["lr"]}')
                if self.scheduler:
                    self.scheduler.step(total_loss)

            self.trained = True
            torch.save({'state_dict': self.model.state_dict()}, self.file)
            print(f'Training finished.')
            
            # Plot loss history
            plt.figure(figsize=(10, 5))
            plt.plot(range(self.epochs), loss_history)
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        except Exception as e:
            print(f"Error during model training: {e}")

    def train_test(self):
        try:
            test_loader = self.loader.get_test_loader()
            self.model.train()
            for i in range(self.epochs):
                total_loss = 0
                for idx, (data, label) in enumerate(test_loader):
                    data, label = data.to(self.device), label.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    output = self.format_output(output)
                    loss = self.criterion(output, label)
                    loss.backward()
                    total_loss += loss.item()
                    self.optimizer.step()
                if i % 10 == 0:
                    torch.save({'state_dict': self.model.state_dict()}, self.file)
                print(f'Epoch [{i+1}/{self.epochs}], Loss: {total_loss:.4f}')
                if self.scheduler:
                    self.scheduler.step(total_loss)

            self.trained = True
            torch.save({'state_dict': self.model.state_dict()}, self.file)
            print(f'Training finished.')
        except Exception as e:
            print(f"Error during model training: {e}")
            
    def format_output(self, output):
        return output

    def evaluate(self):
        self.model.eval()
        preds = []
        labels = []
        test_loader = self.loader.get_test_loader()
        for idx, (data, label) in enumerate(test_loader):
            data, label = data.to(self.device), label.to(self.device)
            pred = self.model(data)
            pred = self.format_output(pred)
            preds.append(pred.tolist())
            labels.append(label.tolist())
        
        self.create_dataframe(preds, labels)
        return self
   
    def evaluate_recent(self, **kwargs):
        self.model.eval()
        preds = []
        labels = []
        test_loader = self.loader.get_recent_loader(**kwargs)
        for idx, (data, label) in enumerate(test_loader):
            data, label = data.to(self.device), label.to(self.device)
            pred = self.model(data)
            pred = self.format_output(pred)
            preds.append(pred.tolist())
            labels.append(label.tolist())
        
        self.create_dataframe(preds, labels)
        return self
    
    def predict(self):
        self.model.eval()
        with torch.no_grad():
            stock_data = self.loader.get_recent_data()
            recent_data = torch.tensor(stock_data, dtype=torch.float32).to(self.device)

            prediction = self.model(recent_data.unsqueeze(1))
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
