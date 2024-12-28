import hashlib

def generate_short_md5(input_string, len = 8):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()[:len]

def is_a_share(stock_code):
    """
    判断给定的股票代码是否是A股代码且不含字母。

    Args:
    stock_code (str): 股票代码

    Returns:
    bool: 如果是A股代码且不含字母则返回True，否则返回False
    """
    # 检查股票代码长度是否为6位，且全部由数字组成
    if len(stock_code) == 6 and stock_code.isdigit():
        # 检查股票代码是否以0, 3, 6开头
        if stock_code.startswith(("0", "3", "6")):
            return True
    return False


def gpu_info():
    import torch
    # 获取可用的 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    # 列出每个 GPU 的名称
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
