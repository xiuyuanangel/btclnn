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


def compute_bollinger_bands(df, windows=(20,), k=2.0):
    """计算布林带特征

    布林带反映价格相对位置和波动率扩张/收缩:
      - bb_width: 带宽(上轨-下轨)/中轨, 表示波动率大小(可用于识别布林带收缩)
      - bb_pct: %B位置 (close-下轨)/(上轨-下轨), 0~1表示价格在带内位置

    Args:
        df: K线DataFrame
        windows: 滚动窗口大小列表
        k: 标准差倍数(默认2)
    """
    eps = 1e-8
    for w in windows:
        ma = df['close'].rolling(window=w, min_periods=1).mean()
        std = df['close'].rolling(window=w, min_periods=1).std()
        upper = ma + k * std
        lower = ma - k * std
        # 带宽: (上轨-下轨)/中轨, 归一化波动率
        df[f'bb_width_{w}'] = (upper - lower) / (ma + eps)
        # %B: (close-下轨)/(上轨-下轨), 价格在带内的相对位置
        df[f'bb_pct_{w}'] = (df['close'] - lower) / (upper - lower + eps)
    return df


def compute_labels(df, horizons=None, source_period_minutes=5):
    """计算多周期标签: 各预测时间窗口后是否上涨

    使用未来平滑价格和最小收益门限降低噪声，并可丢弃中性样本。

    Args:
        df: K线DataFrame(需包含close列)
        horizons: 预测时间窗口列表(分钟), 默认从config读取
        source_period_minutes: 数据源周期(分钟), 用于计算shift步数
    """
    if horizons is None:
        horizons = config.PREDICTION_HORIZONS

    smooth_window = getattr(config, 'LABEL_SMOOTH_WINDOW', 1)
    min_return = getattr(config, 'LABEL_MIN_RETURN', 0.0)
    drop_neutral = getattr(config, 'LABEL_DROP_NEUTRAL', False)

    for h in horizons:
        shift_steps = h // source_period_minutes
        future_close = df['close'].shift(-shift_steps)
        if smooth_window > 1:
            # 对未来价格做平滑，减少单根K线噪声
            future_close = future_close.rolling(window=smooth_window, min_periods=1).mean()
            future_close = future_close.shift(-(smooth_window - 1))

        future_return = future_close / df['close'] - 1.0
        col_name = f'label_{h}m'
        label = pd.Series(np.nan, index=df.index)
        label[future_return > min_return] = 1
        label[future_return < -min_return] = 0

        if drop_neutral:
            df[col_name] = label
        else:
            # 保留中性样本为0/1二值标签，仍可训练但噪声较大
            df[col_name] = label.fillna(0).astype(int)
    return df


def normalize_sequence_samplewise(tf_seqs):
    """对每个样本序列进行内部归一化，强调相对结构而非绝对量级。"""
    normalized = {}
    for period, seqs in tf_seqs.items():
        mean = seqs.mean(axis=1, keepdims=True)
        std = seqs.std(axis=1, keepdims=True)
        normalized[period] = (seqs - mean) / (std + 1e-8)
    return normalized


# 序列特征列名(9维: 原始OHLCV + 收益率 + 布林带 + 均线比率)
# close_ratio/vol_ratio 使用价格/成交量与144-bar均线的比率,
# 使得BTC=60k和ETH=3k在同一尺度下可比(ratio=1.05含义完全相同)。
SEQ_FEATURE_COLS = [
    'close',        # 核心价格
    'vol',          # 成交量
    'return_1',     # 单期收益率（最直接的变化量）
    'high',         # 最高价
    'low',          # 最低价
    'bb_width_20',  # 布林带带宽(波动率扩张/收缩)
    'bb_pct_20',    # 布林带%B价格位置
    'close_ratio',  # 价格/均线比率, 跨币种可比的相对价格位置
    'vol_ratio',    # 成交量/均线比率, 跨币种可比的相对成交量水平
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




def compute_position_features(df, window=144):
    """计算价格/成交量的长期位置比率特征（跨币种可比）

    使用 close/ma(close) 和 vol/ma(vol) 代替原始绝对值，
    使得 BTC=60k 和 ETH=3k 在同一个尺度下可比：
      ratio=1.05 意味着"价格高于长期均线5%"。

    Args:
        df: K线DataFrame(需含close/vol列)
        window: 滚动窗口大小(默认144), 各周期代表不同时间长度但语义一致

    Returns:
        原地修改的DataFrame, 添加'close_ratio'和'vol_ratio'列
    """
    eps = 1e-8
    # 价格比率: close / 长期均线,  >1 = 高于均值, <1 = 低于均值
    close_ma = df['close'].rolling(window=window, min_periods=1).mean()
    df['close_ratio'] = df['close'] / (close_ma + eps)

    # 成交量比率: vol / 长期均量,  >1 = 放量, <1 = 缩量
    vol_ma = df['vol'].rolling(window=window, min_periods=1).mean()
    df['vol_ratio'] = df['vol'] / (vol_ma + eps)

    return df


def compute_all_features(df):
    """对DataFrame计算序列特征(原地修改)

    混合方案: 收益率 + 布林带 + 位置比率, 其余(close/vol/high/low)为原始OHLCV字段
    """
    df = compute_returns(df, windows=(1,))
    df = compute_bollinger_bands(df, windows=(20,))
    df = compute_position_features(df)
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


def build_multi_tf_dataset(tf_dfs, target_df, label_source_df=None,
                          export_debug_csv=False):
    """构建多周期融合数据集(多标签版)

    对每个目标预测时间点(10min粒度), 从各周期提取对应时间窗口的序列特征。
    上下文特征来自最宏观的周期(4hour)。
    标签支持多个预测时间窗口(10min/30min等), 基于细粒度数据计算。

    Args:
        tf_dfs: dict of {period: DataFrame} 各周期的原始K线DataFrame
        target_df: DataFrame 目标K线数据(含DatetimeIndex和OHLCV, 用于确定样本时间轴)
        label_source_df: DataFrame 标签来源的细粒度K线数据(默认用target_df)
        export_debug_csv: 是否导出样本CSV供人工核验

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

    # 4.5 每个样本序列级别归一化，强调同一序列内部结构关系
    all_sequences = normalize_sequence_samplewise(all_sequences)

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

    # 导出调试CSV(仅需少量样本, 不影响训练性能)
    if export_debug_csv:
        _csv_path = os.path.join(config.BASE_DIR, 'data', 'dataset_debug.csv')
        export_dataset_debug_csv(
            X_dict, X_ctx, y,
            target_df_with_id=target_df,
            tf_dfs_raw=tf_dfs,
            global_valid_mask=global_valid,
            output_path=_csv_path,
            num_samples=30,
        )

    return X_dict, X_ctx, y


def build_multi_symbol_dataset(all_symbols_tf_data, fetcher, export_debug_csv=False):
    """构建多币种融合数据集，合并所有币种的样本，增加样本数量

    Args:
        all_symbols_tf_data: dict {symbol: {period: list_of_kline_dicts}}
        fetcher: HuobiDataFetcher实例，用于转换为DataFrame
        export_debug_csv: 是否导出调试CSV

    Returns:
        X_dict_merged: dict of {period: np.array (N, seq_len, feature_size)}
        X_ctx_merged: np.array (N, context_size)
        y_merged: np.array (N, num_horizons)
    """
    all_X_dicts = []
    all_X_ctx = []
    all_y = []

    for symbol, symbol_tf_data in all_symbols_tf_data.items():
        logger.info(f"{'='*60}")
        logger.info(f"处理币种: {symbol}")
        logger.info(f"{'='*60}")

        # 转换为DataFrames
        tf_dfs = {}
        for period, data in symbol_tf_data.items():
            if period == '10min':
                continue
            if data:
                tf_dfs[period] = fetcher.get_dataframe(data)

        # 检查必需周期
        required_periods = list(config.TIMEFRAMES.keys())
        missing_periods = [p for p in required_periods if p not in tf_dfs or len(tf_dfs[p]) == 0]
        if missing_periods:
            logger.warning(f"{symbol} 缺少必需周期数据: {missing_periods}，跳过该币种")
            continue

        # 获取目标数据
        target_df = fetcher.get_dataframe(symbol_tf_data['10min'])
        label_source_df = fetcher.get_dataframe(symbol_tf_data['5min'])

        if target_df.empty or label_source_df.empty:
            logger.warning(f"{symbol} 目标数据为空，跳过该币种")
            continue

        # 构建该币种的数据集
        try:
            X_dict, X_ctx, y = build_multi_tf_dataset(
                tf_dfs, target_df, label_source_df=label_source_df,
                export_debug_csv=export_debug_csv
            )
            if len(y) > 0:
                all_X_dicts.append(X_dict)
                all_X_ctx.append(X_ctx)
                all_y.append(y)
                logger.info(f"{symbol}: 成功获取 {len(y)} 个样本")
            else:
                logger.warning(f"{symbol}: 样本数为0，跳过")
        except Exception as e:
            logger.error(f"{symbol}: 构建数据集失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # 合并所有币种的数据
    if not all_X_dicts:
        logger.error("没有有效币种数据，无法构建数据集")
        return {}, np.array([]), np.array([])

    # 合并各数组
    merged_X_dict = {}
    for period in all_X_dicts[0].keys():
        merged_X_dict[period] = np.concatenate([d[period] for d in all_X_dicts], axis=0)

    merged_X_ctx = np.concatenate(all_X_ctx, axis=0)
    merged_y = np.concatenate(all_y, axis=0)

    logger.info(f"{'='*60}")
    logger.info(f"多币种数据集构建完成: {len(all_symbols_tf_data)} 个币种，共 {len(merged_y)} 个样本")
    logger.info(f"{'='*60}")
    for period, arr in merged_X_dict.items():
        logger.info(f"  {period}: {arr.shape}")
    logger.info(f"  上下文特征: {merged_X_ctx.shape}")
    logger.info(f"  标签: {merged_y.shape}")

    # 统计合并后各标签窗口的涨跌比例
    horizons = config.PREDICTION_HORIZONS
    for i, h in enumerate(horizons):
        up_ratio = merged_y[:, i].mean()
        logger.info(f"  标签[{h}m]: 涨={merged_y[:, i].sum():.0f} ({up_ratio:.2%}), "
                     f"跌={(len(merged_y)-merged_y[:, i].sum()):.0f} ({1-up_ratio:.2%})")

    return merged_X_dict, merged_X_ctx, merged_y


def export_dataset_debug_csv(X_dict, X_ctx, y, target_df_with_id, tf_dfs_raw,
                             global_valid_mask, output_path, num_samples=30):
    """导出少量数据集样本到CSV，供人工核验多周期对齐是否正确

    每行对应一个样本，包含：
      - target_datetime: 样本时间点(10min粒度)
      - 各周期的窗口时间范围
      - 各周期的窗口两端收盘价(原始价格)
      - 上下文特征
      - 标签值

    Args:
        X_dict: build_multi_tf_dataset 的输出
        X_ctx: 同上
        y: 同上
        target_df_with_id: 包含'id'列的target DataFrame
        tf_dfs_raw: {period: DataFrame} 原始K线数据(含id/close列)
        global_valid_mask: np.array(bool) 有效样本掩码
        output_path: csv保存路径
        num_samples: 导出的样本数
    """
    periods = list(config.TIMEFRAMES.keys())
    seq_lengths = {p: cfg['seq_length'] for p, cfg in config.TIMEFRAMES.items()}
    horizons = config.PREDICTION_HORIZONS
    target_ts = target_df_with_id['id'].values.astype(np.int64)
    target_dt = pd.to_datetime(target_ts, unit='s')

    # 有效样本对应的目标时间戳和原始价格
    valid_ts = target_ts[global_valid_mask]

    n = min(num_samples, len(y))
    logger.info(f"导出 {n} 条数据样本到 {output_path}")

    rows = []
    for i in range(n):
        row = {
            'sample_idx': i,
            'target_datetime': pd.to_datetime(valid_ts[i], unit='s'),
            'target_timestamp': valid_ts[i],
        }

        # 各周期的窗口信息
        for period in periods:
            seq_len = seq_lengths[period]
            # 查找target时刻在该周期中最接近的历史bar索引
            tf_ts_raw = tf_dfs_raw[period]['id'].values.astype(np.int64)
            idx = np.searchsorted(tf_ts_raw, valid_ts[i], side='right') - 1
            if idx >= seq_len - 1:
                win_start_ts = tf_ts_raw[idx - seq_len + 1]
                win_end_ts = tf_ts_raw[idx]
                win_start_close = tf_dfs_raw[period].iloc[idx - seq_len + 1]['close']
                win_end_close = tf_dfs_raw[period].iloc[idx]['close']
                row[f'{period}_window_start'] = pd.to_datetime(win_start_ts, unit='s')
                row[f'{period}_window_end'] = pd.to_datetime(win_end_ts, unit='s')
                row[f'{period}_bars'] = seq_len
                row[f'{period}_close_first'] = round(win_start_close, 1)
                row[f'{period}_close_last'] = round(win_end_close, 1)
            else:
                row[f'{period}_window_start'] = 'INSUFFICIENT_DATA'
                row[f'{period}_window_end'] = ''
                row[f'{period}_bars'] = 0
                row[f'{period}_close_first'] = ''
                row[f'{period}_close_last'] = ''

        # 上下文特征
        for j, col in enumerate(CONTEXT_FEATURE_COLS):
            row[f'ctx_{col}'] = round(X_ctx[i, j], 6)

        # 标签
        for j, h in enumerate(horizons):
            row[f'label_{h}m'] = int(y[i, j])

        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    logger.info(f"CSV已保存: {output_path} ({n}行)")


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

    # 标准化上下文特征
    ctx_tr = (ctx_tr - _ctx_mean) / _ctx_std
    ctx_val = (val_data[1] - _ctx_mean) / _ctx_std
    ctx_test = (test_data[1] - _ctx_mean) / _ctx_std
    logger.info(f"  context train 标准化后: mean={ctx_tr.mean():.4f}, std={ctx_tr.std():.4f}")

    # 保存标准化参数(供预测时复用)
    _norm_path = os.path.join(config.CHECKPOINT_DIR, 'feature_norm_stats.pkl')
    with open(_norm_path, 'wb') as f:
        pickle.dump({'periods': periods, 'stats': _norm_stats}, f)
    logger.info(f"标准化统计量已保存 -> {_norm_path}")

    return (
        (X_tr, ctx_tr, y_tr),
        (val_data[0], ctx_val, val_data[2]),
        (test_data[0], ctx_test, test_data[2]),
    )


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


def rolling_cv_split(X_dict, X_ctx, y, n_folds=None, val_ratio=None, test_ratio=None):
    """滚动时间窗口交叉验证切分 (Expanding Window)

    每折使用从历史起点到折分割点的数据训练, 在紧接的验证窗口上评估。
    始终保留最后一段作为独立测试集(不参与折内训练)。

    折布局示例 (n_folds=3, val_ratio=0.10, test_ratio=0.05):
        Fold 1: Train [0:65%]  | Val [65%:75%]   |  (未使用)        | (test)
        Fold 2: Train [0:75%]  | Val [75%:85%]   |  (未使用)        | (test)
        Fold 3: Train [0:85%]  | Val [85%:95%]   |  (未使用)        | (test)
        Test:   (所有折均未使用的最后 5%) ➜ [95%:100%]

    Args:
        X_dict: dict of {period: np.array (N, seq_len, feat_size)}
        X_ctx: np.array (N, ctx_size)
        y: np.array (N, num_horizons)
        n_folds: 折数
        val_ratio: 每折验证集占总数比例
        test_ratio: 保留的独立测试集比例

    Returns:
        folds: list of [fold_idx, ...], 每项为:
            (train_data, val_data) 元组, 格式同 split_multi_tf_dataset
        test_data: 独立测试集元组 (X_dict_test, X_ctx_test, y_test)
    """
    n_folds = n_folds or config.CV_N_FOLDS
    val_ratio = val_ratio or config.CV_VAL_RATIO
    test_ratio = test_ratio or config.CV_TEST_RATIO

    n = len(y)
    test_start = int(n * (1.0 - test_ratio))

    # 独立测试集 (最后 test_ratio 部分)
    test_data = (
        {p: X_dict[p][test_start:] for p in X_dict},
        X_ctx[test_start:],
        y[test_start:],
    )

    # 剩余数据用于 CV 各折
    cv_n = test_start
    fold_val_size = int(cv_n * val_ratio)

    folds = []
    for fold_idx in range(n_folds):
        # 第 k 折: train_end 逐渐右移, val 窗口紧随其后
        val_end_fraction = 1.0 - (n_folds - fold_idx) * val_ratio
        val_start_fraction = val_end_fraction - val_ratio
        train_end = int(cv_n * val_start_fraction)
        val_start = train_end
        val_end = int(cv_n * val_end_fraction)

        train_data = (
            {p: X_dict[p][:train_end] for p in X_dict},
            X_ctx[:train_end],
            y[:train_end],
        )
        val_data_part = (
            {p: X_dict[p][val_start:val_end] for p in X_dict},
            X_ctx[val_start:val_end],
            y[val_start:val_end],
        )
        folds.append((train_data, val_data_part))

        logger.info(f"  CV Fold {fold_idx+1}/{n_folds}: "
                     f"Train [0:{train_end}] ({train_end}条), "
                     f"Val [{val_start}:{val_end}] ({val_end-val_start}条)")

    logger.info(f"  CV 测试集: [{test_start}:{n}] ({n-test_start}条, 独立保留)")
    return folds, test_data


# ==================== 兼容旧接口 ====================
def build_dataset(df, seq_length=None):
    """单周期数据集构建(兼容旧接口, 多标签版)"""
    seq_length = seq_length or 144

    df = compute_returns(df)
    df = compute_volume_features(df)
    df = compute_price_features(df)
    df = compute_rolling_stats(df)
    df = compute_rsi(df)
    df = compute_bollinger_bands(df, windows=(20,))
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
