"""液态神经网络(LNN)项目配置文件"""

import os

# ==================== 交易对配置 ====================
SYMBOL = "BTC-USDT"
CONTRACT_TYPE = "swap"  # 永续合约

# ==================== 数据配置 ====================
LOOKBACK_DAYS = 60          # 历史数据天数
KLINE_PERIOD = "5min"       # API请求的K线周期
TARGET_PERIOD = "10min"     # 目标K线周期(由5分钟聚合)
SEQ_LENGTH = 144            # 序列长度(144个10分钟=24小时)
TRAIN_RATIO = 0.7           # 训练集比例
VAL_RATIO = 0.15            # 验证集比例

# ==================== 模型配置 ====================
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10               # 早停耐心值

# ==================== 数据存储路径 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cache")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "lnn_best.pth")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==================== 火币API配置 ====================
HUOBI_BASE_URL = "https://api.hbdm.com"
HUOBI_KLINE_ENDPOINT = "/linear-swap-ex/market/history/kline"
API_REQUEST_INTERVAL = 0.5  # API请求间隔(秒)，避免触发限频
