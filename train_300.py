from ai_stocks import KlinePredictor

def train_model():
    kp = KlinePredictor(index='000300', device='mps')
    print("开始训练模型...")
    kp.train()
    print("结束训练")

if __name__ == "__main__":
    train_model()