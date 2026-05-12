"""液态神经网络(LNN)项目配置文件"""

import os

# ==================== 交易对配置 ====================
SYMBOLS = ["BTC-USDT", "ETH-USDT"]  # 支持多币种训练，增加样本数量
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
USE_ROLLING_CV = False         # 启用滚动窗口交叉验证, 使用Expanding Window更贴近真实时序分布
CV_N_FOLDS = 3               # 折数(受时间限制, CPU环境推荐3折)
CV_VAL_RATIO = 0.10          # 每折验证集占总数比例
CV_TEST_RATIO = 0.05         # 独立测试集(始终保留, 不参与CV)

# ==================== 多预测周期配置 ====================
# 同时预测多个时间窗口后的涨跌(分钟)
# 10min: 短期趋势, 30min: 中期趋势, 60min: 长期趋势
PREDICTION_HORIZONS = [10, 30, 60]
# 标签来源: 用5min粒度的close计算各horizon后的价格变化
LABEL_SOURCE_PERIOD = "5min"  # 标签基于该周期的K线数据计算
LABEL_SMOOTH_WINDOW = 3       # 未来价格平滑窗口(单位: bar)
LABEL_MIN_RETURN = 0.001     # 5min 标签最小收益门限(0.08%)
LABEL_DROP_NEUTRAL = True     # 是否丢弃噪声中性样本

# ==================== 多周期配置 ====================
# 每个周期独立编码, 覆盖微观到宏观的不同时间尺度
# seq_length: 该周期输入序列长度
# lookback_days: 该周期获取的历史天数(需满足滚动特征窗口需求)
#    长周期(4hour/1day)取更久以支持上下文特征窗口
TIMEFRAMES = {
    '5min':  {'seq_length': 72,  'lookback_days': 1460},   # 72根×5min = 6小时短期趋势(标签来源, 决定时间范围)
    '15min': {'seq_length': 48,  'lookback_days': 1520},   # 48根×15min = 12小时中期
    '60min': {'seq_length': 48,  'lookback_days': 1520},   # 48根×60min = 2天长期
    '4hour': {'seq_length': 42,  'lookback_days': 1520},   # 42根×4hour = 7天宏观(context特征来源)
    '1day':  {'seq_length': 60,  'lookback_days': 1800},   # 60根×1day = 60天趋势
}

# ==================== 模型配置 ====================
HIDDEN_SIZE = 64            # 隐藏层大小(从96降至64, 配合精简特征降低过拟合风险)
NUM_LAYERS = 2              # 隐藏层数量(减少深度防止梯度消失)
DROPOUT = 0.5               # 丢弃率(从0.3提升至0.5, 增强正则化)
LEARNING_RATE = 5e-4        # 学习率(从1e-3降低至5e-4, 更稳定收敛)
WEIGHT_DECAY = 3e-4         # L2正则化(新增, 进一步抑制过拟合)
BATCH_SIZE = 512            # 批处理大小(USE_AUTO_BATCH_SIZE=False时使用)
USE_AUTO_BATCH_SIZE = False  # 自动调整BATCH_SIZE根据可用内存/显存
GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积步数，等效 batch_size = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS
EPOCHS = 200                # 训练轮数上限
MAX_TRAIN_SECONDS = 18000   # 最大训练时长(秒), 默认5小时(预留余量给测试+上传)
TRAIN_STOP_MODE = 'time_only'    # 训练停止模式: 'epochs_only', 'time_only', 'both', 'infinite'
                            # - 'epochs_only': 仅由EPOCHS控制
                            # - 'time_only': 仅由MAX_TRAIN_SECONDS控制
                            # - 'both': 两者任一达到即停止
                            # - 'infinite': 无限训练(仅靠早停PATIENCE停止)
PATIENCE = 140              # 早停耐心值(更快截断过拟合)

# ==================== Focal Loss配置 ====================
FOCAL_ALPHA = 1.0           # Focal Loss alpha参数(正负样本权重平衡)
FOCAL_GAMMA = 0.5           # Focal Loss gamma参数(难例聚焦程度，先设小一点)

# ==================== OneCycleLR配置 ====================
ONECYCLE_MAX_LR_SCALE = 2.0  # OneCycleLR最大LR相对于scaled_lr的倍数
ONECYCLE_PCT_START = 0.2    # OneCycleLR预热阶段占总步数比例
ONECYCLE_ANNEAL_STRATEGY = 'cos'  # OneCycleLR退火策略('cos'或'linear')
ONECYCLE_FINAL_DIV_FACTOR = 1000.0  # OneCycleLR最终LR衰减因子

# ==================== Transformer增强配置 ====================
USE_TRANSFORMER = True       # 是否启用Transformer增强(融合LTC+Transformer)
TRANSFORMER_HEADS = 4        # Transformer注意力头数(需整除HIDDEN_SIZE)
CROSS_ATTN_HEADS = 4         # 跨周期注意力头数

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

# ==================== 待验证预测存储 ====================
PENDING_VERIFICATIONS_PATH = os.path.join(BASE_DIR, "data", "pending_verifications.json")
VERIFICATION_STATS_PATH = os.path.join(BASE_DIR, "data", "verification_stats.json")

# ==================== 双标准化配置 ====================
# 启用后, 序列特征维度翻倍 (feature_size × 2):
#   - 通道1 (前7维): 滚动窗口标准化 (局部K线形态)
#   - 通道2 (后7维): 全局标准化 (绝对价格位置)
# 让模型同时学会识别K线形态和理解当前价格相对于历史水平的相对位置
USE_DUAL_NORMALIZATION = True

# ==================== 调试配置 ====================
DEBUG_EXPORT_CSV = False     # 是否导出数据集样本到CSV供人工核验

# ==================== Batch Size 适配 ====================
BASE_BATCH_SIZE = 512  # LR 调优时的基准 batch size

def get_scaled_learning_rate(target_batch_size=None):
    """Linear Scaling Rule: LR_new = LR * (target_batch / base_batch)

    batch size 放大 k 倍时, LR 也应放大 k 倍以保持梯度更新量级一致。
    参考: Goyal et al. "Accurate, Large Minibatch SGD" (2017)
    """
    if target_batch_size is None:
        target_batch_size = BATCH_SIZE
    scale = target_batch_size / BASE_BATCH_SIZE
    scaled_lr = LEARNING_RATE * scale
    return scaled_lr
