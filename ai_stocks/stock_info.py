from ai_stocks.predictions import predictions_list


class StockInfo:
    def __init__(self, 
        symbol, 
        predicts = [],
        **kwargs
    ):
        self.symbol = symbol
        self.predicts = {}
        for item in predicts:
            if isinstance(item, str):
                predict = self.get_predict(item)
                if (predict is not None):
                    self.predicts[item] = predict(self)
            else:
                self.predicts[item.name] = item(self)
                
    def get_predict(self, name: str):
        for predict in predictions_list:
            if predict.name == name:
                return predict
        return None
        
    def __getattr__(self, key: str):
        if (key in self.predicts):
            return self.predicts[key]
        raise AttributeError(f"'StockInfo' object has no attribute '{key}'")