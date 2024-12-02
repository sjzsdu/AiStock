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
                predict_kwargs = kwargs[item] if item in kwargs else {}
                if (predict is not None):
                    self.predicts[item] = predict(self, **predict_kwargs)
            else:
                predict_kwargs = kwargs[item.name] if item.name in kwargs else {}
                self.predicts[item.name] = item(self, **kwargs[item.name])
                
    def get_predict(self, name: str):
        for predict in predictions_list:
            if predict.name == name:
                return predict
        return None
        
    def __getattr__(self, key: str):
        if (key in self.predicts):
            return self.predicts[key]
        raise AttributeError(f"'StockInfo' object has no attribute '{key}'")