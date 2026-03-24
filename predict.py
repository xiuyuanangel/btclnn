"""液态神经网络预测脚本

使用训练好的模型，基于最新K线数据预测10分钟后涨跌。
"""

import os
import logging

import numpy as np
import pandas as pd
import torch

import config
from data_fetcher import HuobiDataFetcher
from features import (
    SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS,
    compute_returns, compute_volume_features, compute_price_features,
    compute_rolling_stats, compute_rsi, compute_context_features,
)
from lnn_model import LiquidNeuralNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(device):
    """加载训练好的模型"""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"模型文件不存在: {config.MODEL_PATH}\n"
            f"请先运行 train.py 训练模型"
        )

    checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model_cfg = checkpoint['config']

    model = LiquidNeuralNetwork(
        seq_feature_size=model_cfg['seq_feature_size'],
        context_feature_size=model_cfg['context_feature_size'],
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(
        f"模型加载成功 (Epoch {checkpoint['epoch']}, "
        f"Val Loss: {checkpoint['val_loss']:.4f}, "
        f"Val Acc: {checkpoint.get('val_acc', 'N/A')})"
    )
    return model


def prepare_latest_features(df):
    """为最新时刻准备模型输入特征"""
    # 计算特征
    df = compute_returns(df)
    df = compute_volume_features(df)
    df = compute_price_features(df)
    df = compute_rolling_stats(df)
    df = compute_rsi(df)
    context = compute_context_features(df)

    # 去除 NaN
    df = pd.concat([df, context], axis=1)
    df = df.dropna(subset=SEQ_FEATURE_COLS + CONTEXT_FEATURE_COLS)

    if len(df) < config.SEQ_LENGTH + 1:
        raise ValueError(f"有效数据不足: 需要 {config.SEQ_LENGTH + 1}, 实际 {len(df)}")

    # 提取最新窗口
    seq = df[SEQ_FEATURE_COLS].iloc[-config.SEQ_LENGTH:].values.astype(np.float32)
    ctx = df[CONTEXT_FEATURE_COLS].iloc[-1].values.astype(np.float32)

    return seq, ctx, df


def predict():
    """执行预测: 基于最新数据判断10分钟后涨跌"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 1. 加载模型
    model = load_model(device)

    # 2. 获取最新数据
    logger.info("正在获取最新K线数据...")
    fetcher = HuobiDataFetcher()
    data = fetcher.get_10min_data(config.LOOKBACK_DAYS)

    if len(data) < config.SEQ_LENGTH + 1:
        logger.error(f"数据不足: 需要至少 {config.SEQ_LENGTH + 1} 条")
        return None

    df = fetcher.get_dataframe(data)

    # 3. 特征准备
    try:
        seq, ctx, df_featured = prepare_latest_features(df)
    except ValueError as e:
        logger.error(str(e))
        return None

    if np.isnan(seq).any() or np.isnan(ctx).any():
        logger.error("特征数据包含 NaN")
        return None

    # 4. 模型推理
    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
    ctx_tensor = torch.FloatTensor(ctx).unsqueeze(0).to(device)

    with torch.no_grad():
        probability = model(seq_tensor, ctx_tensor).item()

    # 5. 输出结果
    direction = "涨 (UP)" if probability > 0.5 else "跌 (DOWN)"
    confidence = abs(probability - 0.5) * 2  # 映射到 [0, 1]

    current_price = df_featured['close'].iloc[-1]
    latest_time = df_featured.index[-1]

    print()
    print("=" * 50)
    print(f"  液态神经网络 (LNN) 预测结果")
    print("=" * 50)
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
