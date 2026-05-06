"""液态神经网络(LNN)项目配置文件"""

import os

# ==================== 交易对配置 ====================
SYMBOL = "BTC-USDT"
CONTRACT_TYPE = "swap"  # 永续合约

# ==================== 数据配置 ====================
TRAIN_RATIO = 0.7           # 训练集比例(非CV模式使用)
VAL_RATIO = 0.15            # 验证集比例(非CV模式使用)

# ==================== 滚动窗口交叉验证 ====================
# 使用Expanding Window CV替代单一验证集切分:
#   Fold 1: Train[0:65%], Val[65%:75%]
#   Fold 2: Train[0:75%], Val[75%:85%]
#   Fold 3: Train[0:85%], Val[85%:95%]
#   Test:   [95%:100%] (始终独立保留)
# 每折训练独立模型, 选val_loss最低的那折模型做最终测试评估。
USE_ROLLING_CV = True        # 启用滚动窗口交叉验证
CV_N_FOLDS = 3               # 折数(受时间限制, CPU环境推荐3折)
CV_VAL_RATIO = 0.10          # 每折验证集占总数比例
CV_TEST_RATIO = 0.05         # 独立测试集(始终保留, 不参与CV)

# ==================== 多预测周期配置 ====================
# 同时预测多个时间窗口后的涨跌(分钟)
# 10min: 短期趋势, 30min: 中期趋势
PREDICTION_HORIZONS = [10, 30]
# 标签来源: 用5min粒度的close计算各horizon后的价格变化
LABEL_SOURCE_PERIOD = "5min"  # 标签基于该周期的K线数据计算
LABEL_SMOOTH_WINDOW = 3       # 未来价格平滑窗口(单位: bar)
LABEL_MIN_RETURN = 0.0008     # 5min 标签最小收益门限(0.08%)
LABEL_DROP_NEUTRAL = True     # 是否丢弃噪声中性样本

# ==================== 多周期配置 ====================
# 每个周期独立编码, 覆盖微观到宏观的不同时间尺度
# seq_length: 该周期输入序列长度
# lookback_days: 该周期获取的历史天数(需满足滚动特征窗口需求)
#    火币API最多可获取连续2年数据, 5min按730天取满,
#    长周期(4hour/1day)取更久以支持上下文特征窗口
TIMEFRAMES = {
    '5min':  {'seq_length': 72,  'lookback_days': 1460},   # 72根×5min = 6小时短期趋势(标签来源, 决定时间范围)
    '15min': {'seq_length': 48,  'lookback_days': 1460},   # 48根×15min = 12小时中期
    '60min': {'seq_length': 48,  'lookback_days': 1460},   # 48根×60min = 2天长期
    '4hour': {'seq_length': 42,  'lookback_days': 1520},   # 42根×4hour = 7天宏观(context特征来源)
    '1day':  {'seq_length': 60,  'lookback_days': 1800},   # 60根×1day = 60天趋势
}

# ==================== 模型配置 ====================
HIDDEN_SIZE = 128            # 隐藏层大小(从96降至64, 配合精简特征降低过拟合风险)
NUM_LAYERS = 2              # 隐藏层数量(减少深度防止梯度消失)
DROPOUT = 0.5               # 丢弃率(从0.3提升至0.5, 增强正则化)
LEARNING_RATE = 5e-4        # 学习率(从1e-3降低至5e-4, 更稳定收敛)
WEIGHT_DECAY = 1e-4         # L2正则化(新增, 进一步抑制过拟合)
BATCH_SIZE = 512            # 批处理大小
EPOCHS = 99999               # 训练轮数(上限, 实际由 MAX_TRAIN_SECONDS 控制)
MAX_TRAIN_SECONDS = 17640   # 最大训练时长(秒), 默认4.8小时(预留余量给测试+上传)
PATIENCE = 140               # 早停耐心值(更快截断过拟合)

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
