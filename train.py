"""液态神经网络训练脚本 — 多周期融合版"""

import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data_fetcher import HuobiDataFetcher
from notifier import MeoWNotifier
from features import (
    build_multi_tf_dataset, split_multi_tf_dataset,
    MultiTimeframeDataset, SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS,
)
from lnn_model import MultiTimeframeLNN, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def train_model():
    """完整的训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    periods = list(config.TIMEFRAMES.keys())
    logger.info(f"多周期融合: {periods}")

    # 初始化通知器
    notifier = None
    if config.MEOW_NICKNAME:
        notifier = MeoWNotifier(config.MEOW_NICKNAME)
        notifier.send_training_start(config.EPOCHS)

    # ==================== 1. 获取数据 ====================
    logger.info("=" * 60)
    logger.info("步骤 1: 获取多周期K线数据")
    logger.info("=" * 60)

    fetcher = HuobiDataFetcher()
    timeframe_data = fetcher.fetch_multi_timeframe()

    # 构建10min目标时间线(用于标签对齐)
    data_10min = fetcher.resample_to_10min(timeframe_data['5min'])
    target_df = fetcher.get_dataframe(data_10min)

    if target_df.empty or len(target_df) < 100:
        logger.error("目标时间线数据不足")
        return None

    # 各周期转DataFrame
    tf_dfs = {}
    for period, data in timeframe_data.items():
        tf_dfs[period] = fetcher.get_dataframe(data)

    # ==================== 2. 构建数据集 ====================
    logger.info("=" * 60)
    logger.info("步骤 2: 多周期特征工程与数据集构建")
    logger.info("=" * 60)

    X_dict, X_ctx, y = build_multi_tf_dataset(tf_dfs, target_df)

    if len(y) < 100:
        logger.error(f"有效样本不足: {len(y)} 个, 需要至少 100 个")
        if notifier:
            notifier.send_training_error(f"有效样本不足: {len(y)} 个, 需要至少 100 个")
        return None

    train_data, val_data, test_data = split_multi_tf_dataset(X_dict, X_ctx, y)
    logger.info(f"数据划分 -> 训练: {len(train_data[2])}, "
                f"验证: {len(val_data[2])}, 测试: {len(test_data[2])}")

    # 创建 DataLoader
    train_dataset = MultiTimeframeDataset(train_data[0], train_data[1], train_data[2], periods)
    val_dataset = MultiTimeframeDataset(val_data[0], val_data[1], val_data[2], periods)
    test_dataset = MultiTimeframeDataset(test_data[0], test_data[1], test_data[2], periods)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # ==================== 3. 创建模型 ====================
    logger.info("=" * 60)
    logger.info("步骤 3: 构建多周期液态神经网络")
    logger.info("=" * 60)

    feat_size = len(SEQ_FEATURE_COLS)
    ctx_size = len(CONTEXT_FEATURE_COLS)

    tf_configs = {
        p: {'seq_length': cfg['seq_length'], 'feature_size': feat_size}
        for p, cfg in config.TIMEFRAMES.items()
    }

    model = MultiTimeframeLNN(
        timeframe_configs=tf_configs,
        context_feature_size=ctx_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # ==================== 4. 训练配置 ====================
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = nn.BCELoss()

    # 优先从GitHub Release下载最新模型(仅CI环境)
    gh_token = os.environ.get('GH_TOKEN')
    if gh_token:
        try:
            import subprocess, json as _json
            result = subprocess.run(
                ['gh', 'release', 'view', '--json', 'tagName'],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                tag = _json.loads(result.stdout).get('tagName')
                logger.info(f"检测到最新Release: {tag}, 正在下载模型...")
                dl = subprocess.run(
                    ['gh', 'release', 'download', tag,
                     '--pattern', '*.pth', '--dir', 'checkpoints/',
                     '--clobber'],
                    capture_output=True, text=True, timeout=120,
                )
                if dl.returncode == 0 and os.path.exists(config.MODEL_PATH):
                    logger.info("Release模型下载成功")
                else:
                    logger.warning(f"Release模型下载失败: {dl.stderr.strip()}")
            else:
                logger.info("未找到已有Release")
        except Exception as e:
            logger.warning(f"获取Release信息失败: {e}")
    else:
        logger.info("未找到已有GITHUB_TOKEN, 忽略Release下载")

    # 加载已有模型权重(作为初始化, 每次固定训练25轮)
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if os.path.exists(config.MODEL_PATH):
        try:
            resume_checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
            ckpt_config = resume_checkpoint.get('config', {})
            if 'timeframe_configs' in ckpt_config:
                model.load_state_dict(resume_checkpoint['model_state_dict'])
                logger.info("加载已有模型权重作为初始化")
            else:
                logger.info("检测到旧架构checkpoint，从头训练新模型")
        except Exception as e:
            logger.warning(f"加载checkpoint失败，从头训练: {e}")

    # ==================== 5. 训练循环 ====================
    logger.info("=" * 60)
    logger.info(f"步骤 4: 开始训练 ({start_epoch+1}~{config.EPOCHS} epochs)")
    logger.info("=" * 60)

    for epoch in range(start_epoch, config.EPOCHS):
        t0 = time.time()

        # --- 训练阶段 ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for tf_seqs, ctx, labels in train_loader:
            tf_seqs = {p: v.to(device) for p, v in tf_seqs.items()}
            ctx = ctx.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(tf_seqs, ctx)
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
            for tf_seqs, ctx, labels in val_loader:
                tf_seqs = {p: v.to(device) for p, v in tf_seqs.items()}
                ctx = ctx.to(device)
                labels = labels.to(device)
                outputs = model(tf_seqs, ctx)
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
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': {
                    'timeframe_configs': tf_configs,
                    'context_feature_size': ctx_size,
                    'hidden_size': config.HIDDEN_SIZE,
                    'num_layers': config.NUM_LAYERS,
                    'dropout': config.DROPOUT,
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
        for tf_seqs, ctx, labels in test_loader:
            tf_seqs = {p: v.to(device) for p, v in tf_seqs.items()}
            ctx = ctx.to(device)
            labels = labels.to(device)
            outputs = model(tf_seqs, ctx)
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

    # 发送训练完成通知
    if notifier:
        notifier.send_training_complete(
            epoch=checkpoint['epoch'],
            val_loss=checkpoint['val_loss'],
            val_acc=checkpoint['val_acc'],
            test_acc=test_acc,
            precision=precision,
            recall=recall,
            f1=f1
        )

    return model


if __name__ == "__main__":
    import sys
    try:
        model = train_model()
        if model is None:
            logger.error("训练失败，未生成模型")
            if config.MEOW_NICKNAME:
                notifier = MeoWNotifier(config.MEOW_NICKNAME)
                notifier.send_training_error("训练失败，未生成模型")
            sys.exit(1)
    except Exception as e:
        logger.error(f"训练过程中发生异常: {e}")
        if config.MEOW_NICKNAME:
            notifier = MeoWNotifier(config.MEOW_NICKNAME)
            notifier.send_training_error(str(e))
        raise
