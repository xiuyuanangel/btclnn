"""液态神经网络训练脚本 - 多周期融合版"""

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
    MultiTimeframeDataset, PreConvertedTensorDataset,
    SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS,
)
from lnn_model import MultiTimeframeLNN, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def _load_best_fallback(model, device):
    """从best模型加载权重作为初始化的降级方案"""
    try:
        best_ckpt = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
        ckpt_config = best_ckpt.get('config', {})
        if 'timeframe_configs' in ckpt_config:
            model.load_state_dict(best_ckpt['model_state_dict'])
            logger.info("从最佳模型加载权重作为初始化")
        else:
            logger.info("检测到旧架构checkpoint，从头训练新模型")
    except Exception as e:
        logger.warning(f"加载best checkpoint也失败，从头训练: {e}")


def train_model():
    """完整的训练流程"""
    # 检测CUDA兼容性: PyTorch>=2.4仅支持sm_70+, P100(sm_60)/V100(sm_70)需验证
    _use_cuda = False
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability()
            major, minor = cap[0], cap[1]
            if major < 7:
                logger.warning(
                    f"GPU计算能力为sm_{major}{minor}，当前PyTorch要求>=sm_70，自动降级到CPU训练"
                )
            else:
                _use_cuda = True
        except Exception:
            logger.warning("检测GPU能力失败，自动降级到CPU")
    else:
        logger.info("未检测到可用GPU")

    device = torch.device("cuda" if _use_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    periods = list(config.TIMEFRAMES.keys())
    logger.info(f"多周期融合 {periods}")

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

    # 构建10min目标时间帧(用于标签对齐)
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
        logger.error(f"有效样本不足: {len(y)} 个, 需要至少100 个")
        if notifier:
            notifier.send_training_error(f"有效样本不足: {len(y)} 个, 需要至少100 个")
        return None

    train_data, val_data, test_data = split_multi_tf_dataset(X_dict, X_ctx, y)
    logger.info(f"数据划分 -> 训练: {len(train_data[2])}, "
                f"验证: {len(val_data[2])}, 测试: {len(test_data[2])}")

    # 创建 Dataset: GPU平台预转Tensor并搬入GPU, CPU平台使用原始numpy
    _use_preconverted = False
    if _use_cuda:
        t_pre = time.time()
        # 一次性将全部numpy转为Tensor并搬入GPU(消除训练循环中的逐样本创建逐batch搬运)
        def _to_gpu_tensor_dict(data_tuple):
            x_d, x_c, y_arr = data_tuple
            return (
                {p: torch.tensor(x_d[p], dtype=torch.float32, device=device) for p in periods},
                torch.tensor(x_c, dtype=torch.float32, device=device),
                torch.tensor(y_arr, dtype=torch.float32, device=device),
            )

        train_data = _to_gpu_tensor_dict(train_data)
        val_data = _to_gpu_tensor_dict(val_data)
        test_data = _to_gpu_tensor_dict(test_data)

        train_dataset = PreConvertedTensorDataset(train_data[0], train_data[1], train_data[2], periods)
        val_dataset = PreConvertedTensorDataset(val_data[0], val_data[1], val_data[2], periods)
        test_dataset = PreConvertedTensorDataset(test_data[0], test_data[1], test_data[2], periods)
        _use_preconverted = True
        logger.info(f"GPU数据预转完成 ({time.time()-t_pre:.2f}s), 数据已常驻显存")
    else:
        train_dataset = MultiTimeframeDataset(train_data[0], train_data[1], train_data[2], periods)
        val_dataset = MultiTimeframeDataset(val_data[0], val_data[1], val_data[2], periods)
        test_dataset = MultiTimeframeDataset(test_data[0], test_data[1], test_data[2], periods)

    # DataLoader配置: GPU数据已在显存, 无需多进程页锁, CPU平台启用加锁
    if _use_cuda:
        _dl_kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        _dl_kwargs = {'num_workers': 0, 'pin_memory': False}
    logger.info(f"DataLoader参数: {_dl_kwargs} | 预转GPU: {_use_preconverted}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False, **_dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, **_dl_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, **_dl_kwargs)

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
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练{trainable_params:,}")

    # ==================== 4. 训练配置 ====================
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6,
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
                if dl.returncode == 0:
                    logger.info("Release模型下载成功")
                else:
                    logger.warning(f"Release模型下载失败: {dl.stderr.strip()}")
            else:
                logger.info("未找到已有Release")
        except Exception as e:
            logger.warning(f"获取Release信息失败: {e}")
    else:
        logger.info("未找到已有GITHUB_TOKEN, 忽略Release下载")

    # 加载最终模型权重作为初始化(每次固定训练EPOCHS)
    best_val_loss = float('inf')
    patience_counter = 0
    epoch = 0

    if os.path.exists(config.MODEL_PATH_FINAL):
        try:
            resume_checkpoint = torch.load(config.MODEL_PATH_FINAL, map_location=device, weights_only=False)
            ckpt_config = resume_checkpoint.get('config', {})
            if 'timeframe_configs' in ckpt_config:
                model.load_state_dict(resume_checkpoint['model_state_dict'])
                logger.info(f"从最终模型加载权重初始化 (上次已训练{resume_checkpoint.get('epoch', 0)} 轮)")
                epoch = resume_checkpoint.get('epoch', 0)
            else:
                logger.info("检测到旧架构checkpoint，从头训练新模型")
        except Exception as e:
            logger.warning(f"加载final checkpoint失败，尝试从best模型加载: {e}")
            _load_best_fallback(model, device)
    elif os.path.exists(config.MODEL_PATH):
        _load_best_fallback(model, device)
    else:
        logger.info("未找到已有模型，从头训练")

    # ==================== 5. 训练循环 ====================
    _max_epochs = config.EPOCHS
    _max_seconds = getattr(config, 'MAX_TRAIN_SECONDS', None)
    logger.info("=" * 60)
    if _max_seconds:
        logger.info(f"步骤 4: 开始训练(时间限制: {_max_seconds/3600:.1f}小时, 上限{_max_epochs} epochs)")
    else:
        logger.info(f"步骤 4: 开始训练(1~{_max_epochs} epochs)")
    logger.info("=" * 60)

    train_start_time = time.time()
    

    # ====== 诊断: 首次前向传播信号追踪 ======
    model.eval()
    _diag_iter = iter(train_loader)
    _diag_tf, _diag_ctx, _diag_lbl = next(_diag_iter)
    if not _use_preconverted:
        _diag_tf = {p: v.to(device) for p, v in _diag_tf.items()}
        _diag_ctx = _diag_ctx.to(device)

    with torch.no_grad():
        # 1) 输入统计
        for p in periods:
            d = _diag_tf[p]
            logger.info(f"[诊断] 输入 {p}: mean={d.mean():.6f}, std={d.std():.6f}, "
                       f"min={d.min():.4f}, max={d.max():.4f}")
        logger.info(f"[诊断] ctx: mean={_diag_ctx.mean():.6f}, std={_diag_ctx.std():.6f}")

        # 2) 各编码器输出(兼容DataParallel)
        _base_model = model.module if hasattr(model, 'module') else model
        _enc_outs = []
        for p in periods:
            h = _base_model.encoders[p](_diag_tf[p])
            _enc_outs.append(h)
            logger.info(f"[诊断] {p} 编码器输出 mean={h.mean():.8f}, std={h.std():.8f}, "
                       f"abs_mean={h.abs().mean():.8f}")

        # 3) 注意力后(如果有)
        if hasattr(_base_model, 'cross_attn'):
            _attn_out = _base_model.cross_attn(_enc_outs)
            for i, h in enumerate(_attn_out):
                logger.info(f"[诊断] 注意力后 {periods[i]}: mean={h.mean():.8f}, std={h.std():.8f}")
            _use_attn = _attn_out
        else:
            _use_attn = _enc_outs

        # 4) 融合层输入输出
        _fused_in = torch.cat(_use_attn + [_diag_ctx], dim=-1)
        logger.info(f"[诊断] 融合输入: mean={_fused_in.mean():.8f}, std={_fused_in.std():.8f}")

        _fused = _base_model.fusion(_fused_in)
        logger.info(f"[诊断] 融合输出: mean={_fused.mean():.8f}, std={_fused.std():.8f}")

        # 5) 最终预测
        _out = _base_model.classifier(_fused).squeeze(-1)
        logger.info(f"[诊断] 最终输出 mean={_out.mean():.6f}, min={_out.min():.6f}, max={_out.max():.6f}")

        # 6) 反向梯度诊断
    model.train()
    optimizer.zero_grad()
    _diag_out = model(_diag_tf, _diag_ctx)
    _diag_loss = criterion(_diag_out, _diag_lbl[:len(_diag_out)] if _use_preconverted else _diag_lbl)
    _diag_loss.backward()
    total_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**00.5
    logger.info(f"[诊断] 初始loss={_diag_loss.item():.6f}, 梯度L2范数={total_norm:.8f}")
    # for name, param in model.named_parameters():
    #     if param.grad is not None and param.grad.norm().item() > 0:
    #         logger.info(f"  grad | {name}: norm={param.grad.norm():.8e}")
    optimizer.zero_grad()
    # ====== 诊断结束 ======

    while epoch < _max_epochs:
        # 时间限制检查
        if _max_seconds and (time.time() - train_start_time) >= _max_seconds:
            logger.info(f"达到最大训练时长({_max_seconds/3600:.1f}小时), 停止训练")
            break

        t0 = time.time()
        epoch += 1

        # --- 训练阶段 ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for tf_seqs, ctx, labels in train_loader:
            if not _use_preconverted:
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
                if not _use_preconverted:
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
            f"Epoch {epoch:3d}/{_max_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s | "
            f"已用{(time.time()-train_start_time)/3600:.1f}h"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
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
            logger.info(f"  -> 保存最佳模型(val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"早停: 连续 {config.PATIENCE} 轮验证损失未改善")
                break

    # ==================== 5.5 保存最终模型用于断点续训) ====================
    _total_time = time.time() - train_start_time
    logger.info(f"训练结束: 共{epoch}轮, 总耗时{_total_time/3600:.1f}小时")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
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
    }, config.MODEL_PATH_FINAL)
    logger.info(f"保存最终模型(epoch={epoch}, val_loss={val_loss:.4f}) -> {config.MODEL_PATH_FINAL}")

    # 上传到GitHub Release(仅CI环境)
    if gh_token:
        try:
            import subprocess, json as _json
            from datetime import datetime
            tag_name = f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            # 查找是否已存在release
            result = subprocess.run(
                ['gh', 'release', 'view', '--json', 'tagName', '--jq', '.tagName'],
                capture_output=True, text=True, timeout=30,
            )
            existing_tag = result.stdout.strip() if result.returncode == 0 else None

            notes = (
                f"## 多周期融合LNN模型\n\n"
                f"- **最佳模型**: `lnn_best.pth` (val_loss={best_val_loss:.4f})\n"
                f"- **最终模型**: `lnn_final.pth` (训练{epoch}轮后的完整状态 用于断点续训)\n"
                f"\n包含模型权重、优化器状态、学习率调度器状态，可直接加载继续训练\n"
            )

            if existing_tag:
                # 已有Release: 删除旧版并创建新tag(更新notes和文件)
                subprocess.run(
                    ['gh', 'release', 'delete', existing_tag, '--yes'],
                    capture_output=True, text=True, timeout=30,
                )

            release_title = f"LNN Model {tag_name}"
            ul = subprocess.run(
                ['gh', 'release', 'create', tag_name,
                 '--title', release_title, '--notes', notes,
                 config.MODEL_PATH, config.MODEL_PATH_FINAL],
                capture_output=True, text=True, timeout=180,
            )
            if ul.returncode == 0:
                action = "更新" if existing_tag else "创建"
                logger.info(f"{action}Release成功: {tag_name} (best + final)")
            else:
                logger.warning(f"Release上传失败: {ul.stderr.strip()}")
        except Exception as e:
            logger.warning(f"上传模型到Release失败: {e}")

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
            if not _use_preconverted:
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

    logger.info(f"测试集Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"混淆矩阵: TP={tp} FP={fp} TN={tn} FN={fn}")
    logger.info(f"最佳模型来自Epoch {checkpoint['epoch']}")

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
        logger.error(f"训练过程中发生异常 {e}")
        if config.MEOW_NICKNAME:
            notifier = MeoWNotifier(config.MEOW_NICKNAME)
            notifier.send_training_error(str(e))
        raise
