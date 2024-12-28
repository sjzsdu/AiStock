from .stock_handler import StockHandler
from .stock_data_handler import StockDataHandler
from .stock_info import StockInfo
from .config import *
from .stock_predictor import StockPredictor
from .kline_predictor import KlinePredictor
from .utils import gpu_info, is_a_share, generate_short_md5


import toml
import os
pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
if not os.path.exists(pyproject_path):
    pyproject_path = os.path.join(os.path.dirname(__file__), 'pyproject.toml')

try:
    pyproject_data = toml.load(pyproject_path)
    # 获取版本号
    __version__ = pyproject_data['tool']['poetry']['version']
except FileNotFoundError:
    print(f"错误: 找不到文件 '{pyproject_path}'。")
    __version__ = None
except toml.TomlDecodeError:
    print(f"错误: 文件 '{pyproject_path}' 的格式无效。")
    __version__ = None
except Exception as e:
    print(f"发生意外错误: {e}")
    __version__ = None