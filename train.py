"""液态神经网络训练脚本"""

import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data_fetcher import HuobiDataFetcher
from features import (
    build_dataset, split_dataset,
    SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS,
)
from lnn_model import LiquidNeuralNetwork, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def train_model():
    """完整的训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ==================== 1. 获取数据 ====================
    logger.info("=" * 60)
    logger.info("步骤 1: 获取K线数据")
    logger.info("=" * 60)

    fetcher = HuobiDataFetcher()
    data = fetcher.get_10min_data(config.LOOKBACK_DAYS)

    min_required = config.SEQ_LENGTH * 2
    if len(data) < min_required:
        logger.error(f"数据不足: 需要至少 {min_required} 条, 实际 {len(data)} 条")
        return None

    df = fetcher.get_dataframe(data)
    if df.empty or len(df) < min_required:
        logger.error("有效数据不足")
        return None

    # ==================== 2. 构建数据集 ====================
    logger.info("=" * 60)
    logger.info("步骤 2: 特征工程与数据集构建")
    logger.info("=" * 60)

    X_seq, X_ctx, y = build_dataset(df, config.SEQ_LENGTH)

    if len(y) < 100:
        logger.error(f"有效样本不足: {len(y)} 个, 需要至少 100 个")
        return None

    train_data, val_data, test_data = split_dataset(X_seq, X_ctx, y)
    logger.info(f"数据划分 -> 训练: {len(train_data[2])}, "
                f"验证: {len(val_data[2])}, 测试: {len(test_data[2])}")

    # 创建 DataLoader
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(train_data[0]),
            torch.FloatTensor(train_data[1]),
            torch.FloatTensor(train_data[2]),
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(val_data[0]),
            torch.FloatTensor(val_data[1]),
            torch.FloatTensor(val_data[2]),
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(test_data[0]),
            torch.FloatTensor(test_data[1]),
            torch.FloatTensor(test_data[2]),
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # ==================== 3. 创建模型 ====================
    logger.info("=" * 60)
    logger.info("步骤 3: 构建液态神经网络")
    logger.info("=" * 60)

    seq_feat_size = len(SEQ_FEATURE_COLS)
    ctx_feat_size = len(CONTEXT_FEATURE_COLS)

    model = LiquidNeuralNetwork(
        seq_feature_size=seq_feat_size,
        context_feature_size=ctx_feat_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # ==================== 4. 训练配置 ====================
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    # ==================== 5. 训练循环 ====================
    logger.info("=" * 60)
    logger.info(f"步骤 4: 开始训练 ({config.EPOCHS} epochs)")
    logger.info("=" * 60)

    for epoch in range(config.EPOCHS):
        t0 = time.time()

        # --- 训练阶段 ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_s, X_c, labels in train_loader:
            X_s, X_c, labels = X_s.to(device), X_c.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(X_s, X_c)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- 验证阶段 ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for X_s, X_c, labels in val_loader:
                X_s, X_c, labels = X_s.to(device), X_c.to(device), labels.to(device)
                outputs = model(X_s, X_c)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': {
                    'seq_feature_size': seq_feat_size,
                    'context_feature_size': ctx_feat_size,
                    'hidden_size': config.HIDDEN_SIZE,
                    'num_layers': config.NUM_LAYERS,
                    'seq_length': config.SEQ_LENGTH,
                },
            }, config.MODEL_PATH)
            logger.info(f"  -> 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"早停: 连续 {config.PATIENCE} 轮验证损失未改善")
                break

    # ==================== 6. 测试评估 ====================
    logger.info("=" * 60)
    logger.info("步骤 5: 测试集评估")
    logger.info("=" * 60)

    checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss, test_correct, test_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_s, X_c, labels in test_loader:
            X_s, X_c, labels = X_s.to(device), X_c.to(device), labels.to(device)
            outputs = model(X_s, X_c)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)
            preds = (outputs > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= test_total
    test_acc = test_correct / test_total

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = int(((all_preds > 0.5) & (all_labels == 1)).sum())
    fp = int(((all_preds > 0.5) & (all_labels == 0)).sum())
    tn = int(((all_preds <= 0.5) & (all_labels == 0)).sum())
    fn = int(((all_preds <= 0.5) & (all_labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"测试集 Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"混淆矩阵: TP={tp} FP={fp} TN={tn} FN={fn}")
    logger.info(f"最佳模型来自 Epoch {checkpoint['epoch']}")

    return model


if __name__ == "__main__":
    train_model()
