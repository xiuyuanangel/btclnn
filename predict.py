"""液态神经网络预测脚本 — 多周期融合版

使用训练好的多周期融合模型，基于最新K线数据预测10分钟后涨跌。
"""

import os
import time
import logging

import numpy as np
import pandas as pd
import torch

import config
from data_fetcher import HuobiDataFetcher
from notifier import MeoWNotifier
from features import (
    compute_all_features, compute_context_features,
    SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS,
)
from lnn_model import MultiTimeframeLNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_norm_stats():
    """加载训练时保存的特征标准化统计量"""
    import pickle
    path = os.path.join(config.CHECKPOINT_DIR, 'feature_norm_stats.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"标准化参数文件不存在: {path}\n"
            f"请先运行 train.py 训练模型以生成该文件"
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def normalize_with_stats(tf_seqs_raw, ctx_raw, norm_data):
    """使用训练时的统计量对原始特征进行Z-Score标准化

    Args:
        tf_seqs_raw: dict of {period: np.array (1, seq_len, feat_size)} 原始特征
        ctx_raw: np.array (1, ctx_size) 原始上下文
        norm_data: dict {'periods': list, 'stats': dict} 标准化参数

    Returns:
        标准化后的 tf_seqs, ctx (同shape)
    """
    stats = norm_data['stats']
    periods = norm_data['periods']

    tf_seqs = {}
    for p in periods:
        s = stats[p]
        tf_seqs[p] = (tf_seqs_raw[p] - s['mean']) / s['std']

    cs = stats['context']
    ctx = (ctx_raw - cs['mean']) / cs['std']

    return tf_seqs, ctx


def load_model(device):
    """加载训练好的多周期融合模型"""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"模型文件不存在: {config.MODEL_PATH}\n"
            f"请先运行 train.py 训练模型"
        )

    checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model_cfg = checkpoint['config']

    model = MultiTimeframeLNN(
        timeframe_configs=model_cfg['timeframe_configs'],
        context_feature_size=model_cfg['context_feature_size'],
        hidden_size=model_cfg.get('hidden_size', config.HIDDEN_SIZE),
        num_layers=model_cfg.get('num_layers', config.NUM_LAYERS),
        dropout=model_cfg.get('dropout', config.DROPOUT),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(
        f"模型加载成功 (Epoch {checkpoint['epoch']}, "
        f"Val Loss: {checkpoint['val_loss']:.4f})"
    )
    return model


def prepare_multi_tf_features(timeframe_data):
    """为最新时刻准备多周期模型输入特征

    Args:
        timeframe_data: dict of {period: list_of_kline_dicts}

    Returns:
        tf_seqs: dict of {period: np.array (1, seq_length, feature_size)}
        ctx: np.array (1, context_size)
        target_df: DataFrame (用于获取当前价格和时间)
    """
    fetcher = HuobiDataFetcher()
    periods = list(config.TIMEFRAMES.keys())

    # 各周期计算特征
    tf_dfs = {}
    for period in periods:
        df = fetcher.get_dataframe(timeframe_data[period])
        df = compute_all_features(df)
        df = df.dropna(subset=SEQ_FEATURE_COLS)
        tf_dfs[period] = df

    # 上下文特征(来自4hour)
    ctx_df = compute_context_features(tf_dfs['4hour'])
    ctx = ctx_df[CONTEXT_FEATURE_COLS].values[-1:].astype(np.float32)

    # 目标时间点: 当前时刻
    now_ts = int(pd.Timestamp.now().timestamp())
    target_ts = np.array([now_ts])

    # 各周期提取最新序列
    tf_seqs = {}
    for period in periods:
        seq_length = config.TIMEFRAMES[period]['seq_length']
        df = tf_dfs[period]
        tf_ts = df['id'].values.astype(np.int64)
        tf_feat = df[SEQ_FEATURE_COLS].values

        idx = np.searchsorted(tf_ts, now_ts, side='right') - 1
        if idx < seq_length - 1:
            raise ValueError(f"{period} 数据不足: 需要 {seq_length} 条, 可用 {idx + 1} 条")

        seq = tf_feat[idx - seq_length + 1:idx + 1]
        if np.isnan(seq).any():
            raise ValueError(f"{period} 序列包含 NaN")

        tf_seqs[period] = seq[np.newaxis, :, :].astype(np.float32)

    # 用5min数据获取当前价格(最新)
    target_df = tf_dfs['5min']

    return tf_seqs, ctx, target_df


def predict():
    """执行预测: 基于多周期数据判断10分钟后涨跌"""
    # 检测CUDA兼容性(同train.py)
    _use_cuda = False
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 7:
                _use_cuda = True
        except Exception:
            pass
    device = torch.device("cuda" if _use_cuda else "cpu")
    logger.info(f"使用设备: {device}")

    # 1. 加载模型
    model = load_model(device)

    # 2. 加载标准化参数(必须与训练时一致)
    norm_data = load_norm_stats()

    # 3. 获取多周期数据
    logger.info("正在获取多周期K线数据...")
    fetcher = HuobiDataFetcher()
    timeframe_data = fetcher.fetch_multi_timeframe()

    # 4. 特征准备(原始)
    try:
        tf_seqs_raw, ctx_raw, df_featured = prepare_multi_tf_features(timeframe_data)
    except ValueError as e:
        logger.error(str(e))
        return None

    # 5. 应用与训练一致的Z-Score标准化
    tf_seqs_norm, ctx_norm = normalize_with_stats(tf_seqs_raw, ctx_raw, norm_data)

    # 6. 模型推理
    tf_seqs_tensor = {p: torch.from_numpy(v.copy()).float().to(device) for p, v in tf_seqs_norm.items()}
    ctx_tensor = torch.from_numpy(ctx_norm.copy()).float().to(device)

    with torch.no_grad():
        probability = model(tf_seqs_tensor, ctx_tensor).item()

    # 5. 输出结果
    direction = "涨 (UP)" if probability > 0.5 else "跌 (DOWN)"
    confidence = abs(probability - 0.5) * 2

    current_price = df_featured['close'].iloc[-1]
    latest_time = df_featured.index[-1]

    # 发送通知推送
    if config.MEOW_NICKNAME:
        try:
            notifier = MeoWNotifier(config.MEOW_NICKNAME)
            notifier.send_prediction(
                time=str(latest_time),
                price=float(current_price),
                direction=direction,
                probability=probability,
                confidence=confidence
            )
        except Exception as e:
            logger.warning(f"通知推送失败: {e}")

    print()
    print("=" * 50)
    print(f"  多周期融合 LNN 预测结果")
    print("=" * 50)
    print(f"  融合周期:   {', '.join(config.TIMEFRAMES.keys())}")
    print(f"  当前时间:   {latest_time}")
    print(f"  当前价格:   {current_price:.2f} USDT")
    print(f"  预测方向:   {direction}")
    print(f"  上涨概率:   {probability:.4f} ({probability * 100:.2f}%)")
    print(f"  置信度:     {confidence:.4f} ({confidence * 100:.2f}%)")
    print(f"  预测窗口:   未来10分钟")
    print("=" * 50)
    print()

    # 6. 等待10分钟验证预测结果
    logger.info("等待10分钟获取最新价格验证预测...")
    wait_seconds = 600
    # 对齐到下一个整10分钟边界(减少等待时间)
    now = pd.Timestamp.now()
    next_10min = now.floor('10min') + pd.Timedelta(minutes=10)
    wait = max((next_10min - now).total_seconds(), 60)
    if wait < wait_seconds:
        logger.info(f"对齐到 {next_10min}, 等待 {wait:.0f}s")
        time.sleep(wait)
    else:
        time.sleep(wait_seconds)

    # 获取验证数据(只需5min即可对比价格)
    try:
        verify_data = fetcher.fetch_multi_timeframe()
        verify_df = fetcher.get_dataframe(verify_data['5min'])
        verify_price = verify_df['close'].iloc[-1]
        verify_time = verify_df.index[-1]

        actual_direction = "涨 (UP)" if verify_price > current_price else "跌 (DOWN)"
        is_correct = (probability > 0.5) == (verify_price > current_price)
        price_change = (verify_price - float(current_price)) / float(current_price) * 100

        result_mark = "✅ 正确" if is_correct else "❌ 错误"
        print("-" * 50)
        print(f"  📊 预测验证结果 [{result_mark}]")
        print("-" * 50)
        print(f"  预测时间:   {latest_time} | 价格: {current_price:.2f} USDT")
        print(f"  验证时间:   {verify_time} | 价格: {verify_price:.2f} USDT")
        print(f"  预测方向:   {direction}")
        print(f"  实际方向:   {actual_direction}")
        print(f"  价格变化:   {price_change:+.2f}%")
        print("=" * 50)
        print()

        logger.info(
            f"预测验证: {'正确' if is_correct else '错误'}, "
            f"预测{direction}, 实际{actual_direction}, 变化{price_change:+.2f}%"
        )

        # 推送验证结果通知
        if config.MEOW_NICKNAME:
            try:
                notifier = MeoWNotifier(config.MEOW_NICKNAME)
                notifier.send_prediction_verify(
                    direction=direction,
                    actual_direction=actual_direction,
                    is_correct=is_correct,
                    current_price=float(current_price),
                    verify_price=float(verify_price),
                    price_change_pct=price_change,
                )
            except Exception as e:
                logger.warning(f"验证通知推送失败: {e}")

        return {
            'time': str(latest_time),
            'price': float(current_price),
            'direction': direction,
            'probability': probability,
            'confidence': confidence,
            'verified': True,
            'is_correct': is_correct,
            'verify_time': str(verify_time),
            'verify_price': float(verify_price),
            'price_change_pct': price_change,
        }
    except Exception as e:
        logger.warning(f"验证阶段获取数据失败: {e}")
        print("⚠️  无法获取验证数据，跳过预测验证")
        return {
            'time': str(latest_time),
            'price': float(current_price),
            'direction': direction,
            'probability': probability,
            'confidence': confidence,
            'verified': False,
        }


if __name__ == "__main__":
    predict()
