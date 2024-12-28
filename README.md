# K线预测后市涨跌模型
K线图结合交易量，是其他所有的指标分析的最原始的数据，因此我相信K线中隐藏着某种规律，通过深度学习模型，学习到这种规律并将其应用到未来的预测中，是一个非常有价值的研究方向。K线预测后市涨跌模型是一种基于深度学习的股票价格预测模型，旨在通过分析股票的历史K线数据来预测未来的涨跌趋势。该模型结合了先进的深度学习技术和金融市场的专业知识，为投资者提供了一种全新的、数据驱动的投资决策工具。

```python
from ai_stocks import KlinePredictor
# 以沪深300指数个股为样本，使用MPS加速训练， 可以用GPU, CPU
kp = KlinePredictor(index = '000300', device = 'mps')
# 训练模型
kp.train()
# 评估模型
kp.evaluate()
# 预测个股的后期表现
kp.predict()
```


# AI Stock Predictor
AI Stock Predictor 是一个简单易用的人工智能工具，用于预测股票的下一交易日开盘价。通过几行代码，即可轻松实现预测。

特点
简单易用：只需几行代码，即可预测下一交易日的开盘价。
快速训练：内置高效的训练算法，快速获得预测结果。
灵活扩展：可以根据需要扩展功能，适用于多种股票分析场景。
快速开始
以下是如何使用 AI Stock Predictor 进行股票预测的简单示例：

```python
from ai_stocks import StockInfo

# 初始化股票信息
s = StockInfo('601688')

# 训练模型
s.train()

# 预测下一交易日开盘价
predicted_price = s.predict_next_day()
print(f"Predicted opening price for next trading day: {predicted_price}")
```

安装
```bash
pip install ai_stocks
```

## 贡献
欢迎对本项目的改进提出建议或提交PR！请先 Fork 该项目，然后在本地进行修改，最后提交相关 PR。

## 许可证
本项目采用 MIT 许可证开源。有关更多详细信息，请参阅 LICENSE 文件。

有问题请联系: 122828837@qq.com