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
# TPU 兼容层 — 平滑处理 torch_xla 新旧 API 差异
#
# torch_xla 2.9+ 移除了 xla_device() / xrt_world_size():
#   旧: xm.xla_device()          → 新: torch_xla.device()
#   旧: xm.xrt_world_size()      → 新: torch_xla.runtime.world_size()
#   旧: xm.is_master_ordinal()   → 新: torch_xla.runtime.global_ordinal() == 0
#   旧: xm.get_ordinal()         → 新: torch_xla.runtime.global_ordinal()
#
# xm.optimizer_step / xm.all_reduce / xm.save / xm.rendezvous 等 2.9 仍兼容。
# =============================================================================

class _TPUCompat:
    """统一 TPU 操作接口，处理新旧 API 差异"""

    def __init__(self):
        self.available = False
        self.device = torch.device("cpu")
        self.world_size = 1
        self.is_master = True

        try:
            import torch_xla
            import torch_xla.core.xla_model as _xm
            import torch_xla.distributed.parallel_loader as _pl

            # --- 获取设备 (新API优先) ---
            if hasattr(torch_xla, 'device'):
                self.device = torch_xla.device()
            else:
                self.device = _xm.xla_device()

            # --- 获取核心数 (新API优先) ---
            if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'world_size'):
                self.world_size = torch_xla.runtime.world_size()
            elif hasattr(_xm, 'xrt_world_size'):
                self.world_size = _xm.xrt_world_size()
            else:
                self.world_size = 1

            # --- Master 判断 (新API优先) ---
            if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'global_ordinal'):
                self.is_master = torch_xla.runtime.global_ordinal() == 0
            elif hasattr(_xm, 'is_master_ordinal'):
                self.is_master = _xm.is_master_ordinal()

            self._xm = _xm
            self._pl = _pl
            self.available = True

            logger.info(f"使用 TPU 设备: {self.device}")
            logger.info(f"TPU 核心数: {self.world_size}")

        except (ImportError, RuntimeError, AttributeError) as e:
            err_msg = str(e)
            if 'vfio' in err_msg or 'busy' in err_msg.lower():
                logger.warning(
                    f"TPU 设备忙 ({err_msg.split(':')[0].strip()})。\n"
                    f"  >>> 修复方法: 运行时 → 恢复出厂设置 (Factory reset runtime)，然后重新运行 <<<")
            else:
                logger.warning(f"torch_xla 不可用 ({e})，回退到 CPU")
            self.available = False
            self.device = torch.device("cpu")
            self._xm = None
            self._pl = None

    # ---------- 以下方法映射到 xm.*，在 torch_xla 2.9 中仍然兼容 ----------

    def optimizer_step(self, optimizer, barrier=True):
        if self._xm is not None:
            self._xm.optimizer_step(optimizer, barrier=barrier)

    def all_reduce(self, reduce_type, tensor, scale=1.0):
        if self._xm is not None:
            return self._xm.all_reduce(reduce_type, tensor, scale=scale)
        return tensor

    def save(self, obj, path):
        if self._xm is not None:
            self._xm.save(obj, path)
        else:
            torch.save(obj, path)

    def rendezvous(self, tag):
        if self._xm is not None:
            self._xm.rendezvous(tag)

    def mark_step(self):
        if self._xm is not None:
            self._xm.mark_step()

    def MpDeviceLoader(self, loader):
        """包装 DataLoader 为 TPU 并行加载器"""
        if self._pl is not None and self.available:
            return self._pl.MpDeviceLoader(loader, self.device)
        return loader

    @property
    def REDUCE_SUM(self):
        if self._xm is not None:
            return self._xm.REDUCE_SUM
        return None

    # ---------- AMP (bfloat16) ----------

    def get_amp_context(self):
        """返回 bfloat16 autocast context manager，不支持的版本返回 dummy"""
        try:
            from torch_xla.amp import autocast
            return autocast
        except ImportError:
            return None


# 全局兼容层实例
_tpu = _TPUCompat()


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
# TPU 训练主流程 — 单进程版 (使用 _tpu 兼容层)
# =============================================================================

def _train_worker():
    """TPU 训练工作进程 (单进程模式，PJRT 自动管理多 core)

    使用全局 _tpu 兼容层，自动适配 torch_xla 2.9+ / 旧版 / CPU 回退。
    """
    device = _tpu.device
    cfg = tpu_config.get_training_config()

    batch_size = cfg['BATCH_SIZE']
    accum_steps = cfg['GRADIENT_ACCUMULATION_STEPS']
    use_bf16 = _tpu.available and cfg.get('USE_BF16', False)
    max_epochs = project_config.EPOCHS
    max_seconds = cfg['MAX_TRAIN_SECONDS']
    stop_mode = cfg['TRAIN_STOP_MODE']
    patience = cfg['PATIENCE']

    _use_time_limit = stop_mode in ('time_only', 'both')
    periods = list(project_config.TIMEFRAMES.keys())
    _num_horizons = len(project_config.PREDICTION_HORIZONS)

    # --- 主进程日志 (仅 rank 0) ---
    def master_log(msg):
        if _tpu.is_master:
            logger.info(msg)

    master_log("=" * 60)
    master_log(f"TPU 训练启动 | batch={batch_size} | accum={accum_steps} | "
               f"bf16={'ON' if use_bf16 else 'OFF'} | "
               f"core={_tpu.world_size}")
    master_log(f"周期: {periods} | 窗口: {project_config.PREDICTION_HORIZONS}")
    master_log("=" * 60)

    # --- 通知器 ---
    notifier = None
    if _tpu.is_master and project_config.MEOW_NICKNAME:
        notifier = MeoWNotifier(project_config.MEOW_NICKNAME)
        notifier.send_training_start(max_epochs)

    # ==================== 1. 获取数据 ====================
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
        master_log(f"错误: 有效样本不足 ({len(y)})")
        if notifier:
            notifier.send_training_error(f"有效样本不足: {len(y)}")
        return None

    # ==================== 3. 切分 + 标准化 ====================
    train_data, val_data, test_data = split_multi_tf_dataset(X_dict, X_ctx, y)
    master_log(f"数据划分 -> 训练: {len(train_data[2])}, "
               f"验证: {len(val_data[2])}, 测试: {len(test_data[2])}")
    train_data, val_data, test_data = normalize_datasets(train_data, val_data, test_data)

    # ==================== 4. Dataset + DataLoader ====================
    _make_dataset = lambda d: MultiTimeframeDataset(d[0], d[1], d[2], periods)
    train_dataset = _make_dataset(train_data)
    val_dataset = _make_dataset(val_data)
    test_dataset = _make_dataset(test_data)

    # TPU 上 MpDeviceLoader 需要 num_workers=0；CPU 可用多 worker
    _dl_kw = {} if _tpu.available else {'num_workers': 4, 'pin_memory': True}
    train_loader = _tpu.MpDeviceLoader(DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **_dl_kw))
    val_loader = _tpu.MpDeviceLoader(DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **_dl_kw))
    test_loader = _tpu.MpDeviceLoader(DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **_dl_kw))
    if _tpu.available:
        master_log("使用 MpDeviceLoader 包装 DataLoader")

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
        context_feature_size=ctx_size, hidden_size=project_config.HIDDEN_SIZE,
        num_layers=project_config.NUM_LAYERS, dropout=project_config.DROPOUT,
        output_size=_num_horizons,
        use_transformer=project_config.USE_TRANSFORMER,
        transformer_heads=project_config.TRANSFORMER_HEADS,
        cross_attn_heads=project_config.CROSS_ATTN_HEADS,
    ).to(device)
    master_log(f"模型参数: {count_parameters(model)[0]:,} | "
               f"特征: {feat_size}seq+{ctx_size}ctx | "
               f"Transformer: {'ON' if project_config.USE_TRANSFORMER else 'OFF'}")

    from predict import download_release_model as _dl_release
    _dl_release()
    _load_fallback(model, device)

    # ==================== 6. 优化器 + 调度器 + 损失 ====================
    scaled_lr = project_config.get_scaled_learning_rate(batch_size)
    master_log(f"LR 缩放: base={project_config.BASE_BATCH_SIZE}→target={batch_size}, "
               f"LR={project_config.LEARNING_RATE:.2e}→{scaled_lr:.2e}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=scaled_lr, weight_decay=project_config.WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader) // accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=scaled_lr * project_config.ONECYCLE_MAX_LR_SCALE,
        epochs=max_epochs, steps_per_epoch=steps_per_epoch,
        pct_start=project_config.ONECYCLE_PCT_START,
        anneal_strategy=project_config.ONECYCLE_ANNEAL_STRATEGY,
        final_div_factor=project_config.ONECYCLE_FINAL_DIV_FACTOR)
    pos_weights = _compute_pos_weights(train_data[2], _num_horizons)
    criterion = FocalLoss(alpha=project_config.FOCAL_ALPHA,
                          gamma=project_config.FOCAL_GAMMA,
                          per_horizon_weights=pos_weights,
                          num_horizons=_num_horizons)

    # ==================== 7. 训练循环 ====================
    master_log("=" * 60)
    master_log(f"开始训练 (mode={stop_mode}, max_epochs={max_epochs})")
    master_log("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0
    train_start_time = time.time()

    amp_ctx = _tpu.get_amp_context() if use_bf16 else None
    if use_bf16:
        master_log("启用 bfloat16 混合精度" if amp_ctx else "torch_xla.amp 不可用")

    def _batch_to_device(batch):
        """TPU: batch 已在设备上；CPU: 手动搬运"""
        if _tpu.available:
            return batch
        return ({p: v.to(device) for p, v in batch[0].items()},
                batch[1].to(device), batch[2].to(device))

    for epoch in range(1, max_epochs + 1):
        if _use_time_limit and (time.time() - train_start_time) >= max_seconds:
            master_log(f"达到最大训练时长 ({max_seconds/3600:.1f}h)，停止")
            break

        # --- 训练 ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        t_h_correct = [0] * _num_horizons
        t_h_total = [0] * _num_horizons
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            tf_seqs, ctx, labels = _batch_to_device(batch)
            if amp_ctx:
                with amp_ctx():
                    outputs = model(tf_seqs, ctx)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(tf_seqs, ctx)
                loss = criterion(outputs, labels)

            (loss / accum_steps).backward()
            bs = labels.size(0)
            t_loss += loss.detach() * bs
            preds = (outputs.detach() > 0).float()
            t_correct += (preds == labels).sum()
            t_total += labels.numel()
            for h in range(_num_horizons):
                t_h_correct[h] += (preds[:, h] == labels[:, h]).sum()
                t_h_total[h] += labels[:, h].size(0)

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                _tpu.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()

        # 跨核心归约
        if _tpu.available:
            t_loss = _tpu.all_reduce(_tpu.REDUCE_SUM, t_loss)
            t_correct = _tpu.all_reduce(_tpu.REDUCE_SUM, t_correct)
            t_total = _tpu.all_reduce(_tpu.REDUCE_SUM,
                                       torch.tensor(t_total, device=device)).item()
            for h in range(_num_horizons):
                t_h_correct[h] = _tpu.all_reduce(_tpu.REDUCE_SUM, t_h_correct[h])
                t_h_total[h] = _tpu.all_reduce(_tpu.REDUCE_SUM,
                                               torch.tensor(t_h_total[h], device=device)).item()

        train_loss = t_loss.item() / max(t_total, 1)
        train_acc = t_correct.item() / max(t_total, 1)

        # --- 验证 ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        v_h_correct = [0] * _num_horizons
        v_h_total = [0] * _num_horizons
        with torch.no_grad():
            for batch in val_loader:
                tf_seqs, ctx, labels = _batch_to_device(batch)
                if amp_ctx:
                    with amp_ctx():
                        outputs = model(tf_seqs, ctx)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(tf_seqs, ctx)
                    loss = criterion(outputs, labels)
                bs = labels.size(0)
                v_loss += loss.detach() * bs
                preds = (outputs.detach() > 0).float()
                v_correct += (preds == labels).sum()
                v_total += labels.numel()
                for h in range(_num_horizons):
                    v_h_correct[h] += (preds[:, h] == labels[:, h]).sum()
                    v_h_total[h] += labels[:, h].size(0)

        if _tpu.available:
            v_loss = _tpu.all_reduce(_tpu.REDUCE_SUM, v_loss)
            v_correct = _tpu.all_reduce(_tpu.REDUCE_SUM, v_correct)
            v_total = _tpu.all_reduce(_tpu.REDUCE_SUM,
                                      torch.tensor(v_total, device=device)).item()
            for h in range(_num_horizons):
                v_h_correct[h] = _tpu.all_reduce(_tpu.REDUCE_SUM, v_h_correct[h])
                v_h_total[h] = _tpu.all_reduce(_tpu.REDUCE_SUM,
                                               torch.tensor(v_h_total[h], device=device)).item()

        val_loss = v_loss.item() / max(v_total, 1)
        val_acc = v_correct.item() / max(v_total, 1)

        # --- 日志 ---
        if _tpu.is_master:
            et = time.time() - train_start_time
            lr = optimizer.param_groups[0]['lr']
            hts = " | ".join(f"{project_config.PREDICTION_HORIZONS[h]}m:"
                             f"{t_h_correct[h].item()/max(t_h_total[h],1):.3f}"
                             for h in range(_num_horizons))
            hvs = " | ".join(f"{project_config.PREDICTION_HORIZONS[h]}m:"
                             f"{v_h_correct[h].item()/max(v_h_total[h],1):.3f}"
                             for h in range(_num_horizons))
            logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {lr:.6f} | {et:.0f}s")
            logger.info(f"  TrainAcc -> {hts}")
            logger.info(f"  Val   Acc -> {hvs}")

        # --- Checkpoint ---
        if _tpu.is_master and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss, 'val_acc': val_acc,
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
            _tpu.save(ckpt, project_config.MODEL_PATH)
            logger.info(f"  -> 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if _tpu.is_master:
                    logger.info(f"早停: {patience} 轮未改善")
                break

    _tpu.rendezvous('training_complete')

    # ==================== 8. 测试评估 ====================
    if _tpu.is_master:
        logger.info("=" * 60)
        logger.info("加载最佳模型进行测试评估...")
        logger.info("=" * 60)
        try:
            ckpt = torch.load(project_config.MODEL_PATH, map_location='cpu',
                              weights_only=False)
            _safe_load_state_dict(model, ckpt['model_state_dict'], device)
            model.eval()
        except Exception as e:
            logger.warning(f"加载最佳模型失败 ({e})，使用当前模型")

        te_loss, te_correct, te_total = 0.0, 0, 0
        te_h_correct = [0] * _num_horizons
        te_h_total = [0] * _num_horizons
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                tf_seqs, ctx, labels = _batch_to_device(batch)
                if amp_ctx:
                    with amp_ctx():
                        outputs = model(tf_seqs, ctx)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(tf_seqs, ctx)
                    loss = criterion(outputs, labels)
                bs = labels.size(0)
                te_loss += loss.detach() * bs
                preds = (outputs.detach() > 0).float()
                te_correct += (preds == labels).sum()
                te_total += labels.numel()
                for h in range(_num_horizons):
                    te_h_correct[h] += (preds[:, h] == labels[:, h]).sum()
                    te_h_total[h] += labels[:, h].size(0)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if _tpu.available:
            te_loss = _tpu.all_reduce(_tpu.REDUCE_SUM, te_loss)
            te_correct = _tpu.all_reduce(_tpu.REDUCE_SUM, te_correct)
            for h in range(_num_horizons):
                te_h_correct[h] = _tpu.all_reduce(_tpu.REDUCE_SUM, te_h_correct[h])

        test_loss = te_loss.item() / max(te_total, 1)
        test_acc = te_correct.item() / max(te_total, 1)

        p_arr = np.concatenate(all_preds)
        l_arr = np.concatenate(all_labels)
        tp = int(((p_arr > 0.5) & (l_arr == 1)).sum())
        fp = int(((p_arr > 0.5) & (l_arr == 0)).sum())
        tn = int(((p_arr <= 0.5) & (l_arr == 0)).sum())
        fn = int(((p_arr <= 0.5) & (l_arr == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        logger.info(f"测试集 Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
        logger.info(f"Precision: {prec:.4f} Recall: {rec:.4f} F1: {f1:.4f}")
        logger.info(f"混淆矩阵: TP={tp} FP={fp} TN={tn} FN={fn}")

        for h_idx, h_name in enumerate(project_config.PREDICTION_HORIZONS):
            h_acc = te_h_correct[h_idx].item() / max(te_h_total[h_idx], 1)
            h_p = p_arr[:, h_idx]
            h_l = l_arr[:, h_idx]
            h_tp = int(((h_p > 0.5) & (h_l == 1)).sum())
            h_fp = int(((h_p > 0.5) & (h_l == 0)).sum())
            h_tn = int(((h_p <= 0.5) & (h_l == 0)).sum())
            h_fn = int(((h_p <= 0.5) & (h_l == 1)).sum())
            logger.info(
                f"  [{h_name}min] Acc:{h_acc:.4f} Prec:{h_tp/(h_tp+h_fp) if (h_tp+h_fp)>0 else 0:.4f} "
                f"Rec:{h_tp/(h_tp+h_fn) if (h_tp+h_fn)>0 else 0:.4f} "
                f"TP={h_tp} FP={h_fp} TN={h_tn} FN={h_fn}")

        _tpu.save({
            'model_state_dict': model.state_dict(),
            'val_loss': best_val_loss, 'test_acc': test_acc, 'test_f1': f1,
            'config': ckpt['config'],
        }, project_config.MODEL_PATH_FINAL)
        logger.info(f"最终模型已保存: {project_config.MODEL_PATH_FINAL}")

        if notifier:
            hr = {f"{h}m": te_h_correct[h_idx].item() / max(te_h_total[h_idx], 1)
                  for h_idx, h in enumerate(project_config.PREDICTION_HORIZONS)}
            notifier.send_training_complete(
                epoch=epoch, val_loss=best_val_loss, val_acc=val_loss,
                test_acc=test_acc, precision=prec, recall=rec, f1=f1,
                horizon_results=hr)

    return model


# =============================================================================
# 入口: PJRT/Colab 直接调用；旧 XRT 模式不提供支持（已由 _tpu 兼容层替代）
# =============================================================================

def train():
    """TPU 训练入口"""
    if _tpu.available:
        logger.info(f"TPU 模式启动 ({_tpu.world_size} core) — 使用 _tpu 兼容层")
    else:
        logger.info("CPU 回退模式启动")
    _train_worker()


if __name__ == "__main__":
    train()
