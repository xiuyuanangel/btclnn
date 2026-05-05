"""特征工程模块

将原始K线数据转换为模型输入特征:
- 多周期序列特征: 每个时间周期独立的序列特征(收益率、技术指标等)
- 上下文特征: 60天窗口的统计摘要(来自4hour数据)
- 标签: 多周期涨跌预测(10分钟/30分钟后的涨跌)
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


def compute_labels(df, horizons=None, source_period_minutes=5):
    """计算多周期标签: 各预测时间窗口后是否上涨

    Args:
        df: K线DataFrame(需包含close列)
        horizons: 预测时间窗口列表(分钟), 默认从config读取
        source_period_minutes: 数据源周期(分钟), 用于计算shift步数
    """
    if horizons is None:
        horizons = config.PREDICTION_HORIZONS

    for h in horizons:
        shift_steps = h // source_period_minutes
        col_name = f'label_{h}m'
        df[col_name] = (df['close'].shift(-shift_steps) > df['close']).astype(int)
    return df


# 序列特征列名(混合方案: 原始OHLCV + 单期收益率, 每个时间步5个特征, 所有周期共享)
SEQ_FEATURE_COLS = [
    'close',      # 核心价格
    'vol',        # 成交量
    'return_1',   # 单期收益率（最直接的变化量）
    'high',       # 最高价
    'low',        # 最低价
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
    """对DataFrame计算序列特征(原地修改)
    
    混合方案: 仅计算 return_1, 其余(close/vol/high/low)为原始OHLCV字段
    """
    df = compute_returns(df, windows=(1,))
    return df


def compute_context_features(df, window_bars=360):
    """计算滚动窗口上下文特征: 每个样本基于前window_bars根K线独立计算统计量

    Args:
        df: K线DataFrame(需包含close/vol列), 来自4hour周期
        window_bars: 回溯窗口大小(4hour K线数, 默认360≈60天)

    Returns:
        context: DataFrame(shape同df), 每行对应该时刻的上下文特征
                 (不再广播为全局常量!)
    """
    eps = 1e-8
    close = df['close'].astype(np.float64)
    vol = df['vol'].astype(np.float64)
    n = len(df)
    context = pd.DataFrame(index=df.index)

    # --- 趋势: 滚动窗口线性回归斜率 ---
    # 使用pandas rolling + apply 计算每点的局部趋势斜率
    def _rolling_slope(series, win):
        """滚动窗口内的线性回归斜率(归一化)"""
        x = np.arange(win, dtype=np.float64)
        mean_x = x.mean()
        ss_xx = ((x - mean_x) ** 2).sum()
        if ss_xx < eps:
            return pd.Series(0.0, index=series.index)
        # 滚动均值
        roll_mean = series.rolling(window=win, min_periods=win // 2).mean()
        # (x*y)的滚动均值 - mean_x * mean_y = cov(x,y)
        xy_roll = (series * pd.Series(
            np.arange(len(series)), index=series.index
        )).rolling(window=win, min_periods=win // 2).mean()
        slope = (xy_roll - mean_x * roll_mean) / ss_xx
        return slope

    context['trend_60d'] = _rolling_slope(close, window_bars)
    # 归一化: 斜率/价格均值, 使趋势无量纲
    price_mean = close.rolling(window=window_bars, min_periods=window_bars // 2).mean()
    context['trend_60d'] = context['trend_60d'] / (price_mean + eps)

    # --- 波动率: 滚动窗口收益率标准差 ---
    rets = close.pct_change()
    context['volatility_60d'] = rets.rolling(window=window_bars, min_periods=window_bars // 2).std()

    # --- 成交量趋势: 滚动窗口成交量MA的斜率 ---
    vol_ma = vol.rolling(window=144, min_periods=1).mean()
    context['volume_trend_60d'] = _rolling_slope(vol_ma, window_bars)
    vol_mean_global = vol.mean()
    context['volume_trend_60d'] = context['volume_trend_60d'] / (vol_mean_global + eps)

    # --- 最大单期收益率: 滚动窗口最大值 ---
    context['max_return_60d'] = rets.rolling(window=window_bars, min_periods=window_bars // 2).max()

    # --- 最大回撤: 滚动窗口 ---
    cummax_roll = close.rolling(window=window_bars, min_periods=window_bars // 2).max()
    context['max_drawdown_60d'] = (close - cummax_roll) / (cummax_roll + eps)

    # --- 平均收益率: 滚动窗口均值 ---
    context['mean_return_60d'] = rets.rolling(window=window_bars, min_periods=window_bars // 2).mean()

    # 填充NaN为0(窗口不足的前几行)
    context = context.fillna(0.0)

    # 保留原始'id'列用于后续按时间戳对齐
    if 'id' in df.columns:
        context['id'] = df['id']

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
    """多周期融合数据集(接收numpy数组, __getitem__逐样本转tensor) — 多标签版"""

    def __init__(self, X_dict, X_ctx, y, periods):
        """
        Args:
            X_dict: dict of {period: np.array (N, seq_length, feature_size)}
            X_ctx: np.array (N, context_size)
            y: np.array (N,) 或 (N, num_horizons) 标签
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
        # 支持多标签: y可能是1D或2D
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return tf_seqs, ctx, label


class PreConvertedTensorDataset(Dataset):
    """GPU优化数据集: 数据已预转为Tensor并搬入设备, __getitem__仅做索引(零开销) — 多标签版"""

    def __init__(self, X_dict_tensors, X_ctx_tensor, y_tensor, periods):
        """
        Args:
            X_dict_tensors: dict of {period: torch.Tensor (N, seq_length, feature_size), 已在目标device上}
            X_ctx_tensor: torch.Tensor (N, context_size), 已在目标device上
            y_tensor: torch.Tensor (N,) 或 (N, num_horizons), 已在目标device上
            periods: list of period names
        """
        self.X_dict = X_dict_tensors
        self.X_ctx = X_ctx_tensor
        self.y = y_tensor
        self.periods = periods

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        tf_seqs = {p: self.X_dict[p][idx] for p in self.periods}
        ctx = self.X_ctx[idx]
        label = self.y[idx]
        return tf_seqs, ctx, label


def build_multi_tf_dataset(tf_dfs, target_df, label_source_df=None):
    """构建多周期融合数据集(多标签版)

    对每个目标预测时间点(10min粒度), 从各周期提取对应时间窗口的序列特征。
    上下文特征来自最宏观的周期(4hour)。
    标签支持多个预测时间窗口(10min/30min等), 基于细粒度数据计算。

    Args:
        tf_dfs: dict of {period: DataFrame} 各周期的原始K线DataFrame
        target_df: DataFrame 目标K线数据(含DatetimeIndex和OHLCV, 用于确定样本时间轴)
        label_source_df: DataFrame 标签来源的细粒度K线数据(默认用target_df)

    Returns:
        X_dict: dict of {period: np.array (N, seq_len, feature_size)}
        X_ctx: np.array (N, context_size)
        y: np.array (N, num_horizons) 多周期标签
    """
    periods = list(config.TIMEFRAMES.keys())
    horizons = config.PREDICTION_HORIZONS
    source_period_minutes = 5 if config.LABEL_SOURCE_PERIOD == "5min" else 10
    # 标签列名列表
    label_cols = [f'label_{h}m' for h in horizons]

    # 1. 各周期计算特征
    tf_featured = {}
    for period in periods:
        df = tf_dfs[period].copy()
        df = compute_all_features(df)
        df = df.dropna(subset=SEQ_FEATURE_COLS)
        tf_featured[period] = df
        logger.info(f"  {period} 特征: {len(df)} 条有效")

    # 2. 上下文特征(来自4hour, 最宏观视角) — 滚动窗口逐样本计算
    context_df_4h = tf_featured['4hour']
    context = compute_context_features(context_df_4h)

    # 3. 多周期目标标签(基于细粒度5min数据计算, 再对齐到target_df的时间戳)
    _label_df = (label_source_df if label_source_df is not None else target_df).copy()
    _label_df = compute_labels(_label_df, horizons=horizons, source_period_minutes=source_period_minutes)

    # 将多标签对齐到target_df的时间轴(通过最近的时间戳匹配)
    if label_source_df is not None and _label_df.index.equals(target_df.index) is False:
        # 非同一DataFrame时, 通过id列对齐标签
        if 'id' in _label_df.columns and 'id' in target_df.columns:
            label_ts = _label_df['id'].values.astype(np.int64)
            target_ts = target_df['id'].values.astype(np.int64)
            aligned_labels = pd.DataFrame(index=target_df.index)
            for col in label_cols:
                if col in _label_df.columns:
                    # 对每个目标时间戳, 找label_df中最近的标签值
                    idx = np.searchsorted(label_ts, target_ts, side='right') - 1
                    valid = idx >= 0
                    vals = np.full(len(target_ts), np.nan)
                    if valid.any():
                        vals[valid] = _label_df[col].values[idx[valid]]
                    aligned_labels[col] = vals
            _label_df = aligned_labels

    target_df = target_df.copy()

    # 上下文特征: 按时间戳对齐(每样本有独立的滑动窗口统计量)
    if 'id' in context.columns and 'id' in target_df.columns:
        ctx_ts = context['id'].values.astype(np.int64)
        tgt_ts = target_df['id'].values.astype(np.int64)
        for col in CONTEXT_FEATURE_COLS:
            if col in context.columns:
                # 对每个目标时间戳, 找4hour数据中最近的上下文特征值
                idx = np.searchsorted(ctx_ts, tgt_ts, side='right') - 1
                valid = idx >= 0
                vals = np.full(len(tgt_ts), np.nan)
                if valid.any():
                    vals[valid] = context[col].values[idx[valid]]
                target_df[col] = vals
    else:
        # fallback: 按index对齐
        for col in CONTEXT_FEATURE_COLS:
            if col in context.columns:
                target_df[col] = context[col]
    # 添加所有标签列
    for col in label_cols:
        if col in _label_df.columns:
            target_df[col] = _label_df[col].values

    target_df = target_df.dropna(subset=CONTEXT_FEATURE_COLS + label_cols)

    # 直接使用'id'列作为时间戳(秒级整数), 绕开DatetimeIndex dtype差异
    target_timestamps = target_df['id'].values.astype(np.int64)
    label_data = target_df[label_cols].values  # (N, num_horizons)
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

    # 6. 构建最终数组(仅包含全周期有效的样本, 返回原始未标准化数据)
    X_dict = {}
    for period in periods:
        X_dict[period] = all_sequences[period][global_valid]
        logger.info(f"  {period} 序列形状: {X_dict[period].shape}")

    X_ctx = ctx_data.astype(np.float32)[global_valid]
    y = label_data.astype(np.float32)[global_valid]

    # 注意: 标准化移至 split_multi_tf_dataset 之后进行(避免数据泄露)

    if len(y) > 0:
        for i, h in enumerate(horizons):
            up_ratio = y[:, i].mean()
            logger.info(f"标签[{h}m]: 涨={y[:, i].sum():.0f} ({up_ratio:.2%}), "
                         f"跌={(len(y)-y[:, i].sum()):.0f} ({1-up_ratio:.2%})")
    logger.info(f"多周期数据集: {len(y)} 个有效样本, {len(horizons)} 个预测窗口 (原始数据)")

    return X_dict, X_ctx, y


def normalize_datasets(train_data, val_data, test_data):
    """基于训练集统计量对三个数据划分执行Z-Score标准化

    在 split 之后调用, 确保验证集/测试集不参与统计量计算, 避免数据泄露。

    Args:
        train_data: (X_dict_train, X_ctx_train, y_train)
        val_data:   (X_dict_val,   X_ctx_val,   y_val)
        test_data:  (X_dict_test,  X_ctx_test,  y_test)

    Returns:
        标准化后的三个元组 (同输入格式)
    """
    import pickle, os

    X_tr, ctx_tr, y_tr = train_data
    periods = list(X_tr.keys())

    logger.info("执行Z-Score标准化 (基于训练集统计量)...")
    _norm_stats = {}

    # 各周期序列特征: 仅从训练集拟合mean/std
    for period in periods:
        arr = X_tr[period]
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        _mean = flat.mean(axis=0)
        _std = flat.std(axis=0) + 1e-8
        _norm_stats[period] = {'mean': _mean, 'std': _std}

        # 训练集
        X_tr[period] = ((flat - _mean) / _std).reshape(shape)
        logger.info(f"  {period} train 标准化后: mean={X_tr[period].mean():.4f}, std={X_tr[period].std():.4f}")

        # 验证集 & 测试集: 使用训练集的统计量
        for data_tuple in [val_data, test_data]:
            d_arr = data_tuple[0][period]
            d_shape = d_arr.shape
            d_flat = d_arr.reshape(-1, d_shape[-1])
            data_tuple[0][period] = ((d_flat - _mean) / _std).reshape(d_shape)

    # 上下文特征: 仅从训练集拟合
    _ctx_mean = ctx_tr.mean(axis=0)
    _ctx_std = ctx_tr.std(axis=0) + 1e-8
    _norm_stats['context'] = {'mean': _ctx_mean, 'std': _ctx_std}

    # 训练集上下文
    train_data[1] = (ctx_tr - _ctx_mean) / _ctx_std
    logger.info(f"  context train 标准化后: mean={train_data[1].mean():.4f}, std={train_data[1].std():.4f}")

    # 验证集 & 测试集上下文
    for data_tuple in [val_data, test_data]:
        data_tuple[1] = (data_tuple[1] - _ctx_mean) / _ctx_std

    # 保存标准化参数(供预测时复用)
    _norm_path = os.path.join(config.CHECKPOINT_DIR, 'feature_norm_stats.pkl')
    with open(_norm_path, 'wb') as f:
        pickle.dump({'periods': periods, 'stats': _norm_stats}, f)
    logger.info(f"标准化统计量已保存 -> {_norm_path}")

    return train_data, val_data, test_data


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
    """单周期数据集构建(兼容旧接口, 多标签版)"""
    seq_length = seq_length or 144

    df = compute_returns(df)
    df = compute_volume_features(df)
    df = compute_price_features(df)
    df = compute_rolling_stats(df)
    df = compute_rsi(df)
    context = compute_context_features(df)
    horizons = config.PREDICTION_HORIZONS
    df = compute_labels(df, horizons=horizons)

    df = pd.concat([df, context], axis=1)
    label_cols = [f'label_{h}m' for h in horizons]
    df = df.dropna(subset=SEQ_FEATURE_COLS + CONTEXT_FEATURE_COLS + label_cols)

    feature_data = df[SEQ_FEATURE_COLS].values
    context_data = df[CONTEXT_FEATURE_COLS].values
    label_data = df[label_cols].values
    n = len(df)

    X_seq, X_ctx, y = [], [], []
    for i in range(seq_length, n - max(horizons) // 5):
        seq = feature_data[i - seq_length:i]
        ctx = context_data[i]
        label = label_data[i]
        if np.isnan(seq).any() or np.isnan(ctx).any() or np.isnan(label).any():
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
