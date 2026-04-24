"""特征工程模块

将原始K线数据转换为模型输入特征:
- 多周期序列特征: 每个时间周期独立的序列特征(收益率、技术指标等)
- 上下文特征: 60天窗口的统计摘要(来自4hour数据)
- 标签: 10分钟后涨跌
"""

import numpy as np
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_returns(df, windows=(1, 6, 36, 144)):
    """计算多周期对数收益率"""
    for w in windows:
        col_name = f'return_{w}'
        df[col_name] = np.log(df['close'] / df['close'].shift(w))
    return df


def compute_volume_features(df):
    """计算成交量特征"""
    df['vol_ma_144'] = df['vol'].rolling(window=144, min_periods=1).mean()
    df['volume_ratio'] = df['vol'] / (df['vol_ma_144'] + 1e-8)
    return df


def compute_price_features(df):
    """计算价格形态特征"""
    eps = 1e-8
    df['range_ratio'] = (df['high'] - df['low']) / (df['close'] + eps)
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + eps)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + eps)
    return df


def compute_rolling_stats(df, windows=(6, 36, 144)):
    """计算滚动均值和标准差"""
    for w in windows:
        df[f'ma_{w}'] = df['close'].rolling(window=w, min_periods=1).mean()
        df[f'std_{w}'] = df['close'].rolling(window=w, min_periods=1).std()
        df[f'deviation_{w}'] = (df['close'] - df[f'ma_{w}']) / (df[f'std_{w}'] + 1e-8)
    return df


def compute_rsi(df, windows=(6, 36, 144)):
    """计算RSI(相对强弱指标)"""
    for w in windows:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=w, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=w, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df[f'rsi_{w}'] = 100 - (100 / (1 + rs))
    return df


def compute_labels(df):
    """计算标签: 下一根10分钟K线是否上涨"""
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df


# 序列特征列名(每个时间步13个特征, 所有周期共享)
SEQ_FEATURE_COLS = [
    'return_1', 'return_6', 'return_36', 'return_144',
    'volume_ratio',
    'range_ratio', 'body_ratio', 'close_position',
    'rsi_6', 'rsi_36',
    'deviation_6', 'deviation_144',
]

# 上下文特征列名(60天窗口的6个统计特征)
CONTEXT_FEATURE_COLS = [
    'trend_60d',
    'volatility_60d',
    'volume_trend_60d',
    'max_return_60d',
    'max_drawdown_60d',
    'mean_return_60d',
]


def compute_all_features(df):
    """对DataFrame计算全部序列特征(原地修改)"""
    df = compute_returns(df)
    df = compute_volume_features(df)
    df = compute_price_features(df)
    df = compute_rolling_stats(df)
    df = compute_rsi(df)
    return df


def compute_context_features(df):
    """计算60天窗口的上下文特征(标量摘要)"""
    n = len(df)
    context = pd.DataFrame(index=df.index)

    x = np.arange(n, dtype=np.float64)
    y = df['close'].values.astype(np.float64)
    if n > 2 and np.std(y) > 1e-8:
        slope, _ = np.polyfit(x, y, 1)
        context['trend_60d'] = slope / (np.mean(y) + 1e-8)
    else:
        context['trend_60d'] = 0.0

    returns = df['close'].pct_change().dropna()
    context['volatility_60d'] = returns.std() if len(returns) > 0 else 0.0

    vol_ma = df['vol'].rolling(window=144, min_periods=1).mean()
    if n > 2 and vol_ma.std() > 1e-8:
        vol_slope, _ = np.polyfit(x, vol_ma.values, 1)
        context['volume_trend_60d'] = vol_slope / (df['vol'].mean() + 1e-8)
    else:
        context['volume_trend_60d'] = 0.0

    context['max_return_60d'] = returns.max() if len(returns) > 0 else 0.0

    cummax = df['close'].cummax()
    drawdown = (df['close'] - cummax) / (cummax + 1e-8)
    context['max_drawdown_60d'] = drawdown.min()

    context['mean_return_60d'] = returns.mean() if len(returns) > 0 else 0.0

    for col in CONTEXT_FEATURE_COLS:
        if col in context.columns:
            context[col] = context[col].iloc[-1]

    return context


def align_tf_sequences(tf_timestamps, tf_features, target_timestamps, seq_length):
    """将单周期的特征按目标时间戳对齐, 提取滑动窗口序列(向量化)

    当某周期数据不足时, 返回全零序列。
    """
    feature_size = tf_features.shape[1]
    n_target = len(target_timestamps)

    # 诊断日志
    # if tf_timestamps.size > 0:
    #     logger.info(f"    TF范围: [{tf_timestamps[0]}, {tf_timestamps[-1]}]")
    logger.info(f"    目标范围: [{target_timestamps[0]}, {target_timestamps[-1]}]")

    # 向量化: 一次性计算所有target的searchsorted
    indices = np.searchsorted(tf_timestamps, target_timestamps, side='right') - 1

    # 构建全零结果
    sequences = np.zeros((n_target, seq_length, feature_size), dtype=np.float32)

    # 批量判断有效位置: idx >= seq_length-1
    valid = indices >= seq_length - 1
    valid_indices = np.where(valid)[0]

    # 批量提取有效序列
    for i in valid_indices:
        idx = indices[i]
        seq = tf_features[idx - seq_length + 1:idx + 1]
        if np.isnan(seq).any():
            continue
        sequences[i] = seq

    logger.info(f"    有效对齐: {valid.sum()}/{n_target} "
                f"({valid.sum()/n_target*100:.1f}%)")
    return sequences, valid


class MultiTimeframeDataset(Dataset):
    """多周期融合数据集"""

    def __init__(self, X_dict, X_ctx, y, periods):
        """
        Args:
            X_dict: dict of {period: np.array (N, seq_length, feature_size)}
            X_ctx: np.array (N, context_size)
            y: np.array (N,)
            periods: list of period names (固定key顺序)
        """
        self.X_dict = X_dict
        self.X_ctx = X_ctx
        self.y = y
        self.periods = periods

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        tf_seqs = {p: torch.FloatTensor(self.X_dict[p][idx]) for p in self.periods}
        ctx = torch.FloatTensor(self.X_ctx[idx])
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return tf_seqs, ctx, label


def build_multi_tf_dataset(tf_dfs, target_df):
    """构建多周期融合数据集

    对每个目标预测时间点(10min粒度), 从各周期提取对应时间窗口的序列特征。
    上下文特征来自最宏观的周期(4hour)。

    Args:
        tf_dfs: dict of {period: DataFrame} 各周期的原始K线DataFrame
        target_df: DataFrame 10min目标K线数据(含DatetimeIndex和OHLCV)

    Returns:
        X_dict: dict of {period: np.array (N, seq_len, feature_size)}
        X_ctx: np.array (N, context_size)
        y: np.array (N,)
    """
    periods = list(config.TIMEFRAMES.keys())

    # 1. 各周期计算特征
    tf_featured = {}
    for period in periods:
        df = tf_dfs[period].copy()
        df = compute_all_features(df)
        df = df.dropna(subset=SEQ_FEATURE_COLS)
        tf_featured[period] = df
        logger.info(f"  {period} 特征: {len(df)} 条有效")

    # 2. 上下文特征(来自4hour, 最宏观视角)
    context = compute_context_features(tf_featured['4hour'])

    # 3. 目标标签(来自10min数据)
    target_df = target_df.copy()
    target_df = compute_labels(target_df)
    # 上下文特征直接赋值(已广播, 避免pd.concat跨index报错)
    for col in CONTEXT_FEATURE_COLS:
        if col in context.columns:
            target_df[col] = context[col].iloc[0]
    target_df = target_df.dropna(subset=CONTEXT_FEATURE_COLS + ['label'])

    # 直接使用'id'列作为时间戳(秒级整数), 绕开DatetimeIndex dtype差异
    target_timestamps = target_df['id'].values.astype(np.int64)
    label_data = target_df['label'].values
    ctx_data = target_df[CONTEXT_FEATURE_COLS].values

    # 4. 各周期对齐提取序列 + 收集有效掩码
    all_sequences = {}
    all_valid_masks = {}
    for period in periods:
        seq_length = config.TIMEFRAMES[period]['seq_length']
        df = tf_featured[period]
        tf_ts = df['id'].values.astype(np.int64)
        tf_feat = df[SEQ_FEATURE_COLS].values
        sequences, valid_mask = align_tf_sequences(
            tf_ts, tf_feat, target_timestamps, seq_length
        )
        all_sequences[period] = sequences
        all_valid_masks[period] = valid_mask

    # 5. 取所有周期有效对齐的交集(只保留所有周期都有真实数据的样本)
    global_valid = np.ones(len(target_timestamps), dtype=bool)
    for period, mask in all_valid_masks.items():
        global_valid &= mask
    n_filtered_out = len(target_timestamps) - global_valid.sum()
    logger.info(f"全局有效对齐: {global_valid.sum()}/{len(target_timestamps)} "
                f"({global_valid.sum()/len(target_timestamps)*100:.1f}%) "
                f"(过滤掉{n_filtered_out}个部分缺失样本)")

    # 6. 构建最终数组(仅包含全周期有效的样本)
    X_dict = {}
    for period in periods:
        X_dict[period] = all_sequences[period][global_valid]
        logger.info(f"  {period} 序列形状: {X_dict[period].shape}")

    X_ctx = ctx_data.astype(np.float32)[global_valid]
    y = label_data.astype(np.float32)[global_valid]

    if len(y) > 0:
        up_ratio = y.mean()
        logger.info(f"标签分布: 涨={y.sum():.0f} ({up_ratio:.2%}), "
                     f"跌={len(y)-y.sum():.0f} ({1-up_ratio:.2%})")
    logger.info(f"多周期数据集: {len(y)} 个有效样本")

    return X_dict, X_ctx, y


def split_multi_tf_dataset(X_dict, X_ctx, y, train_ratio=None, val_ratio=None):
    """按时间顺序划分多周期训练集/验证集/测试集"""
    train_ratio = train_ratio or config.TRAIN_RATIO
    val_ratio = val_ratio or config.VAL_RATIO

    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        (
            {p: X_dict[p][:train_end] for p in X_dict},
            X_ctx[:train_end], y[:train_end],
        ),
        (
            {p: X_dict[p][train_end:val_end] for p in X_dict},
            X_ctx[train_end:val_end], y[train_end:val_end],
        ),
        (
            {p: X_dict[p][val_end:] for p in X_dict},
            X_ctx[val_end:], y[val_end:],
        ),
    )


# ==================== 兼容旧接口 ====================
def build_dataset(df, seq_length=None):
    """单周期数据集构建(兼容旧接口)"""
    seq_length = seq_length or 144

    df = compute_returns(df)
    df = compute_volume_features(df)
    df = compute_price_features(df)
    df = compute_rolling_stats(df)
    df = compute_rsi(df)
    context = compute_context_features(df)
    df = compute_labels(df)

    df = pd.concat([df, context], axis=1)
    df = df.dropna(subset=SEQ_FEATURE_COLS + CONTEXT_FEATURE_COLS + ['label'])

    feature_data = df[SEQ_FEATURE_COLS].values
    context_data = df[CONTEXT_FEATURE_COLS].values
    label_data = df['label'].values
    n = len(df)

    X_seq, X_ctx, y = [], [], []
    for i in range(seq_length, n - 1):
        seq = feature_data[i - seq_length:i]
        ctx = context_data[i]
        label = label_data[i]
        if np.isnan(seq).any() or np.isnan(ctx).any():
            continue
        X_seq.append(seq)
        X_ctx.append(ctx)
        y.append(label)

    return np.array(X_seq, dtype=np.float32), np.array(X_ctx, dtype=np.float32), np.array(y, dtype=np.float32)


def split_dataset(X_seq, X_ctx, y, train_ratio=None, val_ratio=None):
    """按时间顺序划分训练集/验证集/测试集(兼容旧接口)"""
    train_ratio = train_ratio or config.TRAIN_RATIO
    val_ratio = val_ratio or config.VAL_RATIO
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        (X_seq[:train_end], X_ctx[:train_end], y[:train_end]),
        (X_seq[train_end:val_end], X_ctx[train_end:val_end], y[train_end:val_end]),
        (X_seq[val_end:], X_ctx[val_end:], y[val_end:]),
    )
