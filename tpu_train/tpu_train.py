#!/usr/bin/env python3
"""TPU 训练版本 — 液态神经网络多周期融合

基于 torch_xla 适配 Google Cloud TPU (v3/v4/v5e)，支持:
    - 自动检测 TPU/CPU 回退
    - bfloat16 混合精度
    - MpDeviceLoader 高效数据搬运
    - 多 TPU 核心自动并行
    - OneCycleLR + FocalLoss 保持不变

用法:
    export PJRT_DEVICE=TPU        # 在 TPU VM 上设置
    python tpu_train/tpu_train.py

    无 TPU 时将自动回退 CPU 模式 (仅用于调试)。
"""

import os
import sys
import time
import logging
import importlib
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 确保项目根目录在路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config as project_config
from data_fetcher import HuobiDataFetcher
from notifier import MeoWNotifier
from features import (
    build_multi_tf_dataset, build_multi_symbol_dataset,
    split_multi_tf_dataset, normalize_datasets,
    MultiTimeframeDataset,
    SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS,
)
from lnn_model import MultiTimeframeLNN, count_parameters

import tpu_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# =============================================================================
# TPU 设备检测
# =============================================================================

def _detect_device():
    """检测并返回 TPU 设备；无 TPU 时回退 CPU

    Returns:
        (device, is_tpu, xm, xmp, pl):
            - device: torch.device
            - is_tpu: bool
            - xm/xmp/pl: torch_xla 模块或 None
    """
    try:
        import torch_xla.core.xla_model as _xm
        import torch_xla.distributed.xla_multiprocessing as _xmp
        import torch_xla.distributed.parallel_loader as _pl

        device = _xm.xla_device()
        logger.info(f"使用 TPU 设备: {device}")
        logger.info(f"TPU 核心数量: {_xm.xrt_world_size()}")
        return device, True, _xm, _xmp, _pl

    except (ImportError, RuntimeError, AttributeError) as e:
        logger.warning(f"torch_xla 不可用 ({e})，回退到 CPU 训练")
        device = torch.device("cpu")
        return device, False, None, None, None


# =============================================================================
# 损失函数 (模块级定义，确保 pickling 兼容)
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss — 多标签版本

    缓解类别不平衡: 对置信样本降低权重，聚焦难例。
    """

    def __init__(self, alpha=1.0, gamma=0.5, per_horizon_weights=None,
                 num_horizons=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if per_horizon_weights is not None:
            self.register_buffer(
                'weights',
                torch.tensor(per_horizon_weights, dtype=torch.float32),
            )
        else:
            self.register_buffer('weights', torch.ones(num_horizons))

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        weight_vec = self.weights.to(target.device).unsqueeze(0)
        sample_weights = torch.where(target >= 0.5, weight_vec,
                                     torch.ones_like(target))
        return (bce * sample_weights * self.alpha).mean()


# =============================================================================
# Checkpoint 前缀适配 (TPU 不用 DataParallel，但可能加载 GPU 训练的 checkpoint)
# =============================================================================

def _strip_module_prefix(state_dict):
    """移除 state_dict 中因 DataParallel 产生的 'module.' 前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len('module.'):] if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def _safe_load_state_dict(model, state_dict, device):
    """安全加载 state_dict，自动处理 DataParallel 前缀"""
    has_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if has_prefix:
        state_dict = _strip_module_prefix(state_dict)
        logger.info(f"移除 checkpoint 的 module. 前缀")
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.warning(f"严格加载失败 ({e})，尝试宽松加载...")
        model.load_state_dict(state_dict, strict=False)


def _load_fallback(model, device):
    """从 checkpoints 加载权重作为初始化 (同原版 _load_best_fallback)"""
    import glob

    final_path = project_config.MODEL_PATH_FINAL
    best_path = project_config.MODEL_PATH

    for path in [final_path, best_path]:
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            ckpt_cfg = ckpt.get('config', {})
            if 'timeframe_configs' in ckpt_cfg:
                _safe_load_state_dict(model, ckpt['model_state_dict'], device)
                logger.info(f"从 {path} 加载权重作为初始化")
                return
        except (FileNotFoundError, Exception) as e:
            logger.info(f"尝试加载 {path} 失败: {e}")

    # 尝试折模型
    fold_pattern = final_path.replace('.pth', '_fold*.pth')
    for fold_path in sorted(glob.glob(fold_pattern)):
        try:
            ckpt = torch.load(fold_path, map_location='cpu', weights_only=False)
            ckpt_cfg = ckpt.get('config', {})
            if 'timeframe_configs' in ckpt_cfg:
                _safe_load_state_dict(model, ckpt['model_state_dict'], device)
                logger.info(f"从折模型 {fold_path} 加载权重")
                return
        except Exception:
            continue

    logger.info("未找到可加载的 checkpoint，从头训练")


# =============================================================================
# 计算类别平衡权重
# =============================================================================

def _compute_pos_weights(train_labels, num_horizons):
    """计算每个预测窗口的正样本权重 (防止类别不平衡)

    Returns:
        list[float]: 每个 horizon 的 positive weight
    """
    labels = train_labels
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    weights = []
    for h_idx in range(num_horizons):
        pos = (labels[:, h_idx] == 1).sum()
        neg = (labels[:, h_idx] == 0).sum()
        if pos > 0:
            pw = float(neg) / float(pos)
            pw = min(pw, 5.0)
            weights.append(pw)
        else:
            weights.append(1.0)
    return weights


# =============================================================================
# TPU 训练主流程 — 单进程版 (供 xmp.spawn 启动)
# =============================================================================

def _train_worker(index=None):
    """TPU 训练工作进程

    当通过 xmp.spawn 启动时，每个 TPU core 运行一个实例；
    直接调用 (index=None) 时以单进程 CPU/TPU 模式运行。
    """
    # --- 设备检测 ---
    device, is_tpu, xm, xmp_mod, pl = _detect_device()

    # --- 合并配置 ---
    cfg = tpu_config.get_training_config()
    batch_size = cfg['BATCH_SIZE']
    accum_steps = cfg['GRADIENT_ACCUMULATION_STEPS']
    use_bf16 = is_tpu and cfg.get('USE_BF16', False)
    max_epochs = project_config.EPOCHS
    max_seconds = cfg['MAX_TRAIN_SECONDS']
    stop_mode = cfg['TRAIN_STOP_MODE']
    patience = cfg['PATIENCE']
    _ = cfg  # keep reference

    _use_epoch_limit = stop_mode in ('epochs_only', 'both')
    _use_time_limit = stop_mode in ('time_only', 'both')

    periods = list(project_config.TIMEFRAMES.keys())
    _num_horizons = len(project_config.PREDICTION_HORIZONS)

    # --- 主进程日志 (仅 rank 0 打印) ---
    def master_log(msg):
        if is_tpu:
            if xm.is_master_ordinal():
                logger.info(msg)
        else:
            logger.info(msg)

    master_log("=" * 60)
    master_log(f"TPU 训练启动 | batch={batch_size} | accum={accum_steps} | "
               f"bf16={'ON' if use_bf16 else 'OFF'}")
    master_log(f"周期: {periods} | 窗口: {project_config.PREDICTION_HORIZONS}")
    master_log("=" * 60)

    # --- 通知器 (仅 rank 0) ---
    notifier = None
    if not is_tpu or xm.is_master_ordinal():
        if project_config.MEOW_NICKNAME:
            notifier = MeoWNotifier(project_config.MEOW_NICKNAME)
            notifier.send_training_start(max_epochs)

    # ==================== 1. 获取数据 (CPU 侧) ====================
    master_log("步骤 1: 获取多币种多周期 K 线数据...")
    fetcher = HuobiDataFetcher()
    all_symbols_data = fetcher.fetch_all_symbols_data()

    # ==================== 2. 构建数据集 ====================
    master_log("步骤 2: 多币种多周期特征工程与数据集构建...")
    X_dict, X_ctx, y = build_multi_symbol_dataset(
        all_symbols_data, fetcher,
        export_debug_csv=project_config.DEBUG_EXPORT_CSV,
    )

    if len(y) < 100:
        msg = f"有效样本不足: {len(y)} 个"
        master_log(f"错误: {msg}")
        if notifier:
            notifier.send_training_error(msg)
        return None

    # ==================== 3. 切分 + 标准化 ====================
    train_data, val_data, test_data = split_multi_tf_dataset(
        X_dict, X_ctx, y)
    master_log(f"数据划分 -> 训练: {len(train_data[2])}, "
               f"验证: {len(val_data[2])}, 测试: {len(test_data[2])}")

    train_data, val_data, test_data = normalize_datasets(
        train_data, val_data, test_data)

    # ==================== 4. 创建 Dataset + DataLoader ====================
    # TPU 无法使用 PreConvertedTensorDataset，统一使用 MultiTimeframeDataset
    train_dataset = MultiTimeframeDataset(
        train_data[0], train_data[1], train_data[2], periods)
    val_dataset = MultiTimeframeDataset(
        val_data[0], val_data[1], val_data[2], periods)
    test_dataset = MultiTimeframeDataset(
        test_data[0], test_data[1], test_data[2], periods)

    # DataLoader 基础配置
    _dl_kw = {'num_workers': 4, 'pin_memory': True} if not is_tpu else {}
    train_loader_raw = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        **_dl_kw)
    val_loader_raw = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **_dl_kw)
    test_loader_raw = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **_dl_kw)

    # TPU 模式包装 MpDeviceLoader
    if is_tpu and pl is not None:
        train_loader = pl.MpDeviceLoader(train_loader_raw, device)
        val_loader = pl.MpDeviceLoader(val_loader_raw, device)
        test_loader = pl.MpDeviceLoader(test_loader_raw, device)
        master_log("使用 MpDeviceLoader 包装 DataLoader")
    else:
        train_loader = train_loader_raw
        val_loader = val_loader_raw
        test_loader = test_loader_raw

    # ==================== 5. 创建模型 ====================
    feat_size = len(SEQ_FEATURE_COLS)
    ctx_size = len(CONTEXT_FEATURE_COLS)
    tf_configs = {
        p: {'seq_length': project_config.TIMEFRAMES[p]['seq_length'],
            'feature_size': feat_size}
        for p in periods
    }

    model = MultiTimeframeLNN(
        timeframe_configs=tf_configs,
        context_feature_size=ctx_size,
        hidden_size=project_config.HIDDEN_SIZE,
        num_layers=project_config.NUM_LAYERS,
        dropout=project_config.DROPOUT,
        output_size=_num_horizons,
        use_transformer=project_config.USE_TRANSFORMER,
        transformer_heads=project_config.TRANSFORMER_HEADS,
        cross_attn_heads=project_config.CROSS_ATTN_HEADS,
    ).to(device)

    master_log(f"模型参数: 总计 {count_parameters(model)[0]:,}")
    master_log(f"序列特征维度: {feat_size} | 上下文: {ctx_size}")
    master_log(f"Transformer: {'启用' if project_config.USE_TRANSFORMER else '禁用'}")

    # 尝试加载已有权重
    from predict import download_release_model as _dl_release
    _dl_release()
    _load_fallback(model, device)

    # ==================== 6. 优化器 + 调度器 + 损失 ====================
    scaled_lr = project_config.get_scaled_learning_rate(batch_size)
    master_log(f"LR 缩放: base_batch={project_config.BASE_BATCH_SIZE}, "
               f"target={batch_size}, LR={project_config.LEARNING_RATE:.2e} → {scaled_lr:.2e}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=project_config.WEIGHT_DECAY,
    )

    steps_per_epoch = max(1, len(train_loader) // accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=scaled_lr * project_config.ONECYCLE_MAX_LR_SCALE,
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=project_config.ONECYCLE_PCT_START,
        anneal_strategy=project_config.ONECYCLE_ANNEAL_STRATEGY,
        final_div_factor=project_config.ONECYCLE_FINAL_DIV_FACTOR,
    )

    pos_weights = _compute_pos_weights(train_data[2], _num_horizons)
    master_log(f"正样本权重: {pos_weights}")

    criterion = FocalLoss(
        alpha=project_config.FOCAL_ALPHA,
        gamma=project_config.FOCAL_GAMMA,
        per_horizon_weights=pos_weights,
        num_horizons=_num_horizons,
    )

    # ==================== 7. 训练循环 ====================
    master_log("=" * 60)
    master_log(f"开始训练 (mode={stop_mode}, max_epochs={max_epochs})")
    master_log("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0
    train_start_time = time.time()
    global_step = 0

    # AMP autocast (TPU bf16)
    amp_ctx = None
    if use_bf16 and xm is not None:
        try:
            from torch_xla.amp import autocast
            amp_ctx = autocast
            master_log("启用 bfloat16 混合精度")
        except ImportError:
            amp_ctx = None
            master_log("torch_xla.amp 不可用，禁用混合精度")

    for epoch in range(1, max_epochs + 1):
        # --- 停止条件 ---
        if _use_time_limit:
            elapsed = time.time() - train_start_time
            if elapsed >= max_seconds:
                master_log(f"达到最大训练时长 ({max_seconds/3600:.1f}h)，停止")
                break

        # ========== 训练阶段 ==========
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0
        train_h_correct = [0] * _num_horizons
        train_h_total = [0] * _num_horizons

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if is_tpu:
                # MpDeviceLoader 返回的 batch 已在 TPU 上
                tf_seqs, ctx, labels = batch
            else:
                # CPU 模式: 手动搬运
                tf_seqs = {p: v.to(device) for p, v in batch[0].items()}
                ctx = batch[1].to(device)
                labels = batch[2].to(device)

            # 前向 (bf16 autocast)
            if amp_ctx is not None:
                with amp_ctx():
                    outputs = model(tf_seqs, ctx)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(tf_seqs, ctx)
                loss = criterion(outputs, labels)

            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            # 统计 (使用 detach 避免图追踪)
            bs = labels.size(0)
            train_loss_sum += loss.detach() * bs
            preds = (outputs.detach() > 0).float()
            train_correct += (preds == labels).sum()
            train_total += labels.numel()
            for h in range(_num_horizons):
                train_h_correct[h] += (preds[:, h] == labels[:, h]).sum()
                train_h_total[h] += labels[:, h].size(0)

            # 梯度累积步
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if is_tpu and xm is not None:
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # epoch 统计归约 (TPU: 跨核心聚合)
        if is_tpu and xm is not None:
            train_loss_sum = xm.all_reduce(xm.REDUCE_SUM, train_loss_sum)
            train_correct = xm.all_reduce(xm.REDUCE_SUM, train_correct)
            train_total = xm.all_reduce(xm.REDUCE_SUM,
                                        torch.tensor(train_total, device=device)).item()
            for h in range(_num_horizons):
                train_h_correct[h] = xm.all_reduce(xm.REDUCE_SUM, train_h_correct[h])
                train_h_total[h] = xm.all_reduce(xm.REDUCE_SUM,
                                                torch.tensor(train_h_total[h], device=device)).item()

        train_loss_avg = train_loss_sum.item() / max(train_total, 1)
        train_acc = train_correct.item() / max(train_total, 1)

        # ========== 验证阶段 ==========
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_h_correct = [0] * _num_horizons
        val_h_total = [0] * _num_horizons

        with torch.no_grad():
            for batch in val_loader:
                if is_tpu:
                    tf_seqs, ctx, labels = batch
                else:
                    tf_seqs = {p: v.to(device) for p, v in batch[0].items()}
                    ctx = batch[1].to(device)
                    labels = batch[2].to(device)

                if amp_ctx is not None:
                    with amp_ctx():
                        outputs = model(tf_seqs, ctx)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(tf_seqs, ctx)
                    loss = criterion(outputs, labels)

                bs = labels.size(0)
                val_loss_sum += loss.detach() * bs
                preds = (outputs.detach() > 0).float()
                val_correct += (preds == labels).sum()
                val_total += labels.numel()
                for h in range(_num_horizons):
                    val_h_correct[h] += (preds[:, h] == labels[:, h]).sum()
                    val_h_total[h] += labels[:, h].size(0)

        if is_tpu and xm is not None:
            val_loss_sum = xm.all_reduce(xm.REDUCE_SUM, val_loss_sum)
            val_correct = xm.all_reduce(xm.REDUCE_SUM, val_correct)
            val_total = xm.all_reduce(xm.REDUCE_SUM,
                                      torch.tensor(val_total, device=device)).item()
            for h in range(_num_horizons):
                val_h_correct[h] = xm.all_reduce(xm.REDUCE_SUM, val_h_correct[h])
                val_h_total[h] = xm.all_reduce(xm.REDUCE_SUM,
                                              torch.tensor(val_h_total[h], device=device)).item()

        val_loss_avg = val_loss_sum.item() / max(val_total, 1)
        val_acc = val_correct.item() / max(val_total, 1)

        # ========== 日志 (仅 rank 0) ==========
        if not is_tpu or xm.is_master_ordinal():
            epoch_time = time.time() - train_start_time
            current_lr = optimizer.param_groups[0]['lr']
            h_train_str = " | ".join([
                f"{project_config.PREDICTION_HORIZONS[h]}m:"
                f"{train_h_correct[h].item()/max(train_h_total[h], 1):.3f}"
                for h in range(_num_horizons)
            ])
            h_val_str = " | ".join([
                f"{project_config.PREDICTION_HORIZONS[h]}m:"
                f"{val_h_correct[h].item()/max(val_h_total[h], 1):.3f}"
                for h in range(_num_horizons)
            ])

            logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss_avg:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss_avg:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | {epoch_time:.0f}s"
            )
            logger.info(f"  TrainAcc -> {h_train_str}")
            logger.info(f"  Val   Acc -> {h_val_str}")

        # ========== Checkpoint (仅 rank 0) ==========
        if (not is_tpu or xm.is_master_ordinal()) and val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss_avg,
                'val_acc': val_acc,
                'best_val_loss': best_val_loss,
                'config': {
                    'timeframe_configs': tf_configs,
                    'context_feature_size': ctx_size,
                    'hidden_size': project_config.HIDDEN_SIZE,
                    'num_layers': project_config.NUM_LAYERS,
                    'dropout': project_config.DROPOUT,
                    'output_size': _num_horizons,
                    'horizons': project_config.PREDICTION_HORIZONS,
                    'use_transformer': project_config.USE_TRANSFORMER,
                    'transformer_heads': project_config.TRANSFORMER_HEADS,
                    'cross_attn_heads': project_config.CROSS_ATTN_HEADS,
                },
            }

            save_path = project_config.MODEL_PATH
            if is_tpu and xm is not None:
                xm.save(ckpt, save_path)
            else:
                torch.save(ckpt, save_path)

            logger.info(f"  -> 保存最佳模型 (val_loss={val_loss_avg:.4f})")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                if not is_tpu or xm.is_master_ordinal():
                    logger.info(f"早停: {patience} 轮未改善")
                break

    # 同步所有核心
    if is_tpu and xm is not None:
        xm.rendezvous('training_complete')

    # ==================== 8. 测试评估 (仅 rank 0) ====================
    if not is_tpu or xm.is_master_ordinal():
        logger.info("=" * 60)
        logger.info("加载最佳模型进行测试评估...")
        logger.info("=" * 60)

        try:
            best_ckpt = torch.load(project_config.MODEL_PATH, map_location='cpu',
                                   weights_only=False)
            _safe_load_state_dict(model, best_ckpt['model_state_dict'], device)
            model.eval()
        except Exception as e:
            logger.warning(f"加载最佳模型失败 ({e})，使用当前模型")

        # 测试
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0
        test_h_correct = [0] * _num_horizons
        test_h_total = [0] * _num_horizons
        all_preds = []
        all_labels_list = []

        with torch.no_grad():
            for batch in test_loader:
                if is_tpu:
                    tf_seqs, ctx, labels = batch
                else:
                    tf_seqs = {p: v.to(device) for p, v in batch[0].items()}
                    ctx = batch[1].to(device)
                    labels = batch[2].to(device)

                if amp_ctx is not None:
                    with amp_ctx():
                        outputs = model(tf_seqs, ctx)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(tf_seqs, ctx)
                    loss = criterion(outputs, labels)

                bs = labels.size(0)
                test_loss_sum += loss.detach() * bs
                preds = (outputs.detach() > 0).float()
                test_correct += (preds == labels).sum()
                test_total += labels.numel()
                for h in range(_num_horizons):
                    test_h_correct[h] += (preds[:, h] == labels[:, h]).sum()
                    test_h_total[h] += labels[:, h].size(0)
                all_preds.append(outputs.cpu().numpy())
                all_labels_list.append(labels.cpu().numpy())

        if is_tpu and xm is not None:
            test_loss_sum = xm.all_reduce(xm.REDUCE_SUM, test_loss_sum)
            # 只有 rank 0 执行下面的日志

        if not is_tpu or xm.is_master_ordinal():
            test_loss_avg = test_loss_sum.item() / max(test_total, 1)
            test_acc = test_correct.item() / max(test_total, 1)

            all_preds_np = np.concatenate(all_preds, axis=0)
            all_labels_np = np.concatenate(all_labels_list, axis=0)

            # 整体指标
            tp = int(((all_preds_np > 0.5) & (all_labels_np == 1)).sum())
            fp = int(((all_preds_np > 0.5) & (all_labels_np == 0)).sum())
            tn = int(((all_preds_np <= 0.5) & (all_labels_np == 0)).sum())
            fn = int(((all_preds_np <= 0.5) & (all_labels_np == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            logger.info(f"测试集 Loss: {test_loss_avg:.4f} Acc: {test_acc:.4f}")
            logger.info(f"Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
            logger.info(f"混淆矩阵: TP={tp} FP={fp} TN={tn} FN={fn}")

            for h_idx, h_name in enumerate(project_config.PREDICTION_HORIZONS):
                h_acc = test_h_correct[h_idx].item() / max(test_h_total[h_idx], 1)
                h_pred = all_preds_np[:, h_idx]
                h_label = all_labels_np[:, h_idx]
                h_tp = int(((h_pred > 0.5) & (h_label == 1)).sum())
                h_fp = int(((h_pred > 0.5) & (h_label == 0)).sum())
                h_tn = int(((h_pred <= 0.5) & (h_label == 0)).sum())
                h_fn = int(((h_pred <= 0.5) & (h_label == 1)).sum())
                h_prec = h_tp / (h_tp + h_fp) if (h_tp + h_fp) > 0 else 0
                h_rec = h_tp / (h_tp + h_fn) if (h_tp + h_fn) > 0 else 0
                logger.info(
                    f"  [{h_name}min] Acc:{h_acc:.4f} Prec:{h_prec:.4f} "
                    f"Rec:{h_rec:.4f} TP={h_tp} FP={h_fp} TN={h_tn} FN={h_fn}"
                )

            # 保存最终模型
            final_ckpt = {
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'test_acc': test_acc,
                'test_f1': f1,
                'config': {
                    'timeframe_configs': tf_configs,
                    'context_feature_size': ctx_size,
                    'hidden_size': project_config.HIDDEN_SIZE,
                    'num_layers': project_config.NUM_LAYERS,
                    'dropout': project_config.DROPOUT,
                    'output_size': _num_horizons,
                    'horizons': project_config.PREDICTION_HORIZONS,
                    'use_transformer': project_config.USE_TRANSFORMER,
                    'transformer_heads': project_config.TRANSFORMER_HEADS,
                    'cross_attn_heads': project_config.CROSS_ATTN_HEADS,
                },
            }
            torch.save(final_ckpt, project_config.MODEL_PATH_FINAL)
            logger.info(f"最终模型已保存: {project_config.MODEL_PATH_FINAL}")

            # 通知
            if notifier:
                horizon_results = {}
                for h_idx, h_name in enumerate(project_config.PREDICTION_HORIZONS):
                    h_acc = test_h_correct[h_idx].item() / max(test_h_total[h_idx], 1)
                    horizon_results[f"{h_name}m"] = h_acc
                notifier.send_training_complete(
                    epoch=epoch, val_loss=best_val_loss, val_acc=best_val_loss,
                    test_acc=test_acc, precision=precision, recall=recall,
                    f1=f1, horizon_results=horizon_results,
                )

    return model


# =============================================================================
# 入口: 自动选择 多 TPU core 启动 或 单进程模式
# =============================================================================

def train():
    """TPU 训练入口函数"""
    # 检查是否在 TPU 环境中
    try:
        import torch_xla.core.xla_model as _xm
        has_tpu = True
    except ImportError:
        has_tpu = False

    if has_tpu:
        try:
            import torch_xla.distributed.xla_multiprocessing as _xmp
            # 用 xmp.spawn 启动，自动覆盖所有 TPU core
            logger.info("使用 xmp.spawn 启动多核心 TPU 训练...")
            _xmp.spawn(_train_worker, args=(), nprocs=None, start_method='fork')
        except Exception as e:
            logger.warning(f"xmp.spawn 失败 ({e})，回退到单进程模式")
            _train_worker()
    else:
        logger.info("单进程模式 (CPU/单卡)")
        _train_worker()


if __name__ == "__main__":
    train()
