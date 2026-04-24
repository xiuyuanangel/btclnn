"""液态神经网络(LNN)项目配置文件"""

import os

# ==================== 交易对配置 ====================
SYMBOL = "BTC-USDT"
CONTRACT_TYPE = "swap"  # 永续合约

# ==================== 数据配置 ====================
TARGET_PERIOD = "10min"     # 目标预测周期(由5分钟聚合)
TRAIN_RATIO = 0.7           # 训练集比例
VAL_RATIO = 0.15            # 验证集比例

# ==================== 多周期配置 ====================
# 每个周期独立编码, 覆盖微观到宏观的不同时间尺度
# seq_length: 该周期输入序列长度
# lookback_days: 该周期获取的历史天数(需满足滚动特征窗口需求)
TIMEFRAMES = {
    '1min':  {'seq_length': 60,  'lookback_days': 15},     # 60根×1min = 1小时(仅近期, 避免数据量过大)
    '5min':  {'seq_length': 72,  'lookback_days': 16},   # 72根×5min = 6小时短期趋势
    '15min': {'seq_length': 48,  'lookback_days': 18},   # 48根×15min = 12小时中期
    '60min': {'seq_length': 48,  'lookback_days': 20},   # 48根×60min = 2天长期
    '4hour': {'seq_length': 42,  'lookback_days': 25},   # 42根×4hour = 7天宏观
    '1day':  {'seq_length': 60,  'lookback_days': 100},   # 60根×1day = 60天趋势(需200天预热)
}

# ==================== 模型配置 ====================
HIDDEN_SIZE = 60            # 隐藏层大小
NUM_LAYERS = 4              # 隐藏层数量
DROPOUT = 0.3               # 丢弃率
LEARNING_RATE = 0.001       # 学习率
BATCH_SIZE = 1024             # 批处理大小
EPOCHS = 90                 # 训练轮数
PATIENCE = 90              # 早停耐心值

# ==================== 数据存储路径 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cache")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "lnn_best.pth")
MODEL_PATH_FINAL = os.path.join(CHECKPOINT_DIR, "lnn_final.pth")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==================== 火币API配置 ====================
HUOBI_BASE_URLS = [
    "https://api.hbdm.vn",       # 主节点
    "https://api.hbdm.com",      # 备用节点1
    "https://api.huobi.de",      # 备用节点2 (国际)
]
HUOBI_BASE_URL = HUOBI_BASE_URLS[0]  # 默认主节点
HUOBI_KLINE_ENDPOINT = "/linear-swap-ex/market/history/kline"
API_REQUEST_INTERVAL = 0.5  # API请求间隔(秒)，避免触发限频

# ==================== 通知推送配置 ====================
# MeoW消息推送配置 (https://www.chuckfang.com/MeoW/api_doc.html)
MEOW_NICKNAME = "修远啊"  # 设置你的MeoW昵称，为空字符串时不发送通知
MEOW_BASE_URL = "https://api.chuckfang.com"  # API基础地址
