"""特征工程模块

将原始K线数据转换为模型输入特征:
- 序列特征: 每个时间步的特征(收益率、技术指标等)
- 上下文特征: 60天窗口的统计摘要
- 标签: 10分钟后涨跌
"""

import numpy as np
import pandas as pd
import logging

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
    # 成交量移动平均
    df['vol_ma_144'] = df['vol'].rolling(window=144, min_periods=1).mean()
    # 当前成交量与均值的比率
    df['volume_ratio'] = df['vol'] / (df['vol_ma_144'] + 1e-8)
    return df


def compute_price_features(df):
    """计算价格形态特征"""
    eps = 1e-8
    # 振幅比率: (最高-最低)/收盘价
    df['range_ratio'] = (df['high'] - df['low']) / (df['close'] + eps)
    # 实体比率: |收盘-开盘|/(最高-最低)
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + eps)
    # 收盘价在区间中的位置: (收盘-最低)/(最高-最低)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + eps)
    return df


def compute_rolling_stats(df, windows=(6, 36, 144)):
    """计算滚动均值和标准差"""
    for w in windows:
        df[f'ma_{w}'] = df['close'].rolling(window=w, min_periods=1).mean()
        df[f'std_{w}'] = df['close'].rolling(window=w, min_periods=1).std()
        # 价格偏离均线的标准化程度
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
    """计算标签: 下一根10分钟K线是否上涨

    label=1 表示下一根K线收盘价高于当前收盘价(涨)
    label=0 表示跌
    """
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df


# 序列特征列名(每个时间步13个特征)
SEQ_FEATURE_COLS = [
    'return_1', 'return_6', 'return_36', 'return_144',      # 多周期收益率
    'volume_ratio',                                          # 成交量比率
    'range_ratio', 'body_ratio', 'close_position',          # 价格形态
    'rsi_6', 'rsi_36',                                       # RSI指标
    'deviation_6', 'deviation_144',                          # 均线偏离度
]

# 上下文特征列名(60天窗口的6个统计特征)
CONTEXT_FEATURE_COLS = [
    'trend_60d',          # 60天价格趋势
    'volatility_60d',     # 60天波动率
    'volume_trend_60d',   # 60天成交量趋势
    'max_return_60d',     # 60天最大单期收益
    'max_drawdown_60d',   # 60天最大回撤
    'mean_return_60d',    # 60天平均收益
]


def compute_context_features(df):
    """计算60天窗口的上下文特征(标量摘要)

    这些特征描述了整个60天窗口的市场状态，
    会在每个时间步注入到模型中。
    """
    n = len(df)
    context = pd.DataFrame(index=df.index)

    # 1. 60天价格趋势(线性回归斜率，标准化)
    x = np.arange(n, dtype=np.float64)
    y = df['close'].values.astype(np.float64)
    if n > 2 and np.std(y) > 1e-8:
        slope, _ = np.polyfit(x, y, 1)
        context['trend_60d'] = slope / (np.mean(y) + 1e-8)
    else:
        context['trend_60d'] = 0.0

    # 2. 60天波动率(收益率标准差)
    returns = df['close'].pct_change().dropna()
    context['volatility_60d'] = returns.std() if len(returns) > 0 else 0.0

    # 3. 60天成交量趋势
    vol_ma = df['vol'].rolling(window=144, min_periods=1).mean()
    if n > 2 and vol_ma.std() > 1e-8:
        vol_slope, _ = np.polyfit(x, vol_ma.values, 1)
        context['volume_trend_60d'] = vol_slope / (df['vol'].mean() + 1e-8)
    else:
        context['volume_trend_60d'] = 0.0

    # 4. 60天最大单期收益
    context['max_return_60d'] = returns.max() if len(returns) > 0 else 0.0

    # 5. 60天最大回撤
    cummax = df['close'].cummax()
    drawdown = (df['close'] - cummax) / (cummax + 1e-8)
    context['max_drawdown_60d'] = drawdown.min()

    # 6. 60天平均收益
    context['mean_return_60d'] = returns.mean() if len(returns) > 0 else 0.0

    # 广播到所有行(整个窗口共享同一个上下文)
    for col in CONTEXT_FEATURE_COLS:
        if col in context.columns:
            context[col] = context[col].iloc[-1]

    return context


def build_dataset(df, seq_length=None):
    """构建滑动窗口数据集

    对每个时间点t:
    - 序列输入: [t-seq_length : t] 的K线特征序列
    - 上下文输入: 截至 t 的60天统计特征
    - 标签: t+1时刻是否上涨

    Args:
        df: 原始K线DataFrame
        seq_length: 序列窗口长度

    Returns:
        X_seq: 序列特征数组 (N, seq_length, 13)
        X_ctx: 上下文特征数组 (N, 6)
        y: 标签数组 (N,)
    """
    seq_length = seq_length or config.SEQ_LENGTH

    # 计算所有特征
    df = compute_returns(df)
    df = compute_volume_features(df)
    df = compute_price_features(df)
    df = compute_rolling_stats(df)
    df = compute_rsi(df)
    context = compute_context_features(df)
    df = compute_labels(df)

    # 合并上下文
    df = pd.concat([df, context], axis=1)

    # 去除NaN行
    df = df.dropna(subset=SEQ_FEATURE_COLS + CONTEXT_FEATURE_COLS + ['label'])

    # 构建滑动窗口(提前提取numpy数组，避免循环中反复df.loc造成OOM)
    feature_data = df[SEQ_FEATURE_COLS].values   # (n, 13)
    context_data = df[CONTEXT_FEATURE_COLS].values  # (n, 6)
    label_data = df['label'].values               # (n,)
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

    X_seq = np.array(X_seq, dtype=np.float32)
    X_ctx = np.array(X_ctx, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # 标签分布统计
    if len(y) > 0:
        up_ratio = y.mean()
        logger.info(f"标签分布: 涨={y.sum():.0f} ({up_ratio:.2%}), "
                     f"跌={len(y)-y.sum():.0f} ({1-up_ratio:.2%})")

    logger.info(f"数据集: {len(X_seq)} 个样本, "
                f"序列形状 {X_seq.shape}, 上下文形状 {X_ctx.shape}")
    return X_seq, X_ctx, y


def split_dataset(X_seq, X_ctx, y, train_ratio=None, val_ratio=None):
    """按时间顺序划分训练集/验证集/测试集"""
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


if __name__ == "__main__":
    from data_fetcher import HuobiDataFetcher

    fetcher = HuobiDataFetcher()
    data = fetcher.get_10min_data()
    df = fetcher.get_dataframe(data)

    if not df.empty:
        X_seq, X_ctx, y = build_dataset(df)
        if len(y) > 0:
            train, val, test = split_dataset(X_seq, X_ctx, y)
            print(f"训练集: {len(train[2])} 样本")
            print(f"验证集: {len(val[2])} 样本")
            print(f"测试集: {len(test[2])} 样本")
