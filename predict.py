"""液态神经网络预测脚本 — 多周期融合版

使用训练好的多周期融合模型，基于最新K线数据预测10分钟后涨跌。
"""

import os
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
        tf_ts = df.index.values.astype('datetime64[s]').astype(np.int64)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 1. 加载模型
    model = load_model(device)

    # 2. 获取多周期数据
    logger.info("正在获取多周期K线数据...")
    fetcher = HuobiDataFetcher()
    timeframe_data = fetcher.fetch_multi_timeframe()

    # 3. 特征准备
    try:
        tf_seqs, ctx, df_featured = prepare_multi_tf_features(timeframe_data)
    except ValueError as e:
        logger.error(str(e))
        return None

    # 4. 模型推理
    tf_seqs_tensor = {p: torch.FloatTensor(v).to(device) for p, v in tf_seqs.items()}
    ctx_tensor = torch.FloatTensor(ctx).to(device)

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

    return {
        'time': str(latest_time),
        'price': float(current_price),
        'direction': direction,
        'probability': probability,
        'confidence': confidence,
    }


if __name__ == "__main__":
    predict()
