"""液态神经网络训练脚本 - 多周期融合版"""

import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data_fetcher import HuobiDataFetcher
from notifier import MeoWNotifier
from features import (
    build_multi_tf_dataset, split_multi_tf_dataset, rolling_cv_split,
    normalize_datasets,
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

    # 构建10min目标时间帧(用于样本时间轴对齐)
    data_10min = fetcher.resample_to_10min(timeframe_data['5min'])
    target_df = fetcher.get_dataframe(data_10min)

    if target_df.empty or len(target_df) < 100:
        logger.error("目标时间线数据不足")
        return None

    # 5min数据作为标签来源(细粒度计算多周期标签)
    label_source_df = fetcher.get_dataframe(timeframe_data['5min'])

    # 各周期转DataFrame
    tf_dfs = {}
    for period, data in timeframe_data.items():
        tf_dfs[period] = fetcher.get_dataframe(data)

    # ==================== 2. 构建数据集 ====================
    logger.info("=" * 60)
    logger.info("步骤 2: 多周期特征工程与数据集构建")
    logger.info("=" * 60)

    X_dict, X_ctx, y = build_multi_tf_dataset(
        tf_dfs, target_df, label_source_df=label_source_df,
        export_debug_csv=getattr(config, 'DEBUG_EXPORT_CSV', False),
    )

    # 多标签维度
    _num_horizons = len(config.PREDICTION_HORIZONS)
    logger.info(f"多标签训练: {_num_horizons} 个预测窗口 -> {config.PREDICTION_HORIZONS}")

    if len(y) < 100:
        logger.error(f"有效样本不足: {len(y)} 个, 需要至少100 个")
        if notifier:
            notifier.send_training_error(f"有效样本不足: {len(y)} 个, 需要至少100 个")
        return None

    # ==================== 数据切分: CV 或 单次切分 ====================
    _use_cv = getattr(config, 'USE_ROLLING_CV', False)
    if _use_cv:
        logger.info(f"使用滚动时间窗口交叉验证 ({config.CV_N_FOLDS}折)")
        cv_folds, cv_test_data = rolling_cv_split(X_dict, X_ctx, y)
        # 将各折的train/val分别标准化(防止数据泄露)
        cv_folds_normalized = []
        for fold_idx, (train_data, val_data) in enumerate(cv_folds):
            logger.info(f"CV Fold {fold_idx+1}: 标准化中...")
            (norm_train, norm_val, _) = normalize_datasets(
                train_data, val_data, val_data  # 占位test, 实际不使用
            )
            cv_folds_normalized.append((norm_train, norm_val))
        # 测试集：使用所有折训练数据合并的统计量标准化
        _all_train_X = {p: np.concatenate([cv_folds[i][0][0][p] for i in range(len(cv_folds))])
                        for p in X_dict}
        _all_train_ctx = np.concatenate([cv_folds[i][0][1] for i in range(len(cv_folds))])
        _all_train_y = np.concatenate([cv_folds[i][0][2] for i in range(len(cv_folds))])
        _, _, cv_test_data = normalize_datasets(
            (_all_train_X, _all_train_ctx, _all_train_y),
            cv_test_data, cv_test_data,
        )
        logger.info(f"CV数据准备完成: {len(cv_folds)}折 + 独立测试集({len(cv_test_data[2])}条)")
    else:
        train_data, val_data, test_data = split_multi_tf_dataset(X_dict, X_ctx, y)
        logger.info(f"数据划分 -> 训练: {len(train_data[2])}, "
                    f"验证: {len(val_data[2])}, 测试: {len(test_data[2])}")
        train_data, val_data, test_data = normalize_datasets(train_data, val_data, test_data)
        # 包装成单折格式以便统一后续处理
        cv_folds_normalized = [(train_data, val_data)]
        cv_test_data = test_data

    # ==================== 开始训练(支持多折CV) ====================
    _max_seconds = getattr(config, 'MAX_TRAIN_SECONDS', None)
    _n_folds = len(cv_folds_normalized)
    _fold_time_budget = None
    if _max_seconds and _use_cv:
        _fold_time_budget = _max_seconds / _n_folds
        logger.info(f"每折时间预算: {_fold_time_budget/3600:.1f}h")

    _best_across_folds = {
        'val_loss': float('inf'),
        'fold_idx': -1,
        'model_path': None,
        'epoch': 0,
        'state_dict': None,
    }

    for fold_idx, (train_data, val_data) in enumerate(cv_folds_normalized):
        if _use_cv:
            logger.info(f"\n{'='*60}")
            logger.info(f"CV Fold {fold_idx+1}/{_n_folds}")
            logger.info(f"{'='*60}")
            logger.info(f"训练集: {len(train_data[2])} 条, 验证集: {len(val_data[2])} 条")

        # ---- 创建 DataLoader (复用原有逻辑) ----
        _use_preconverted = False
        if _use_cuda:
            t_pre = time.time()
            def _to_gpu_tensor_dict(data_tuple):
                x_d, x_c, y_arr = data_tuple
                return (
                    {p: torch.tensor(x_d[p], dtype=torch.float32, device=device) for p in periods},
                    torch.tensor(x_c, dtype=torch.float32, device=device),
                    torch.tensor(y_arr, dtype=torch.float32, device=device),
                )
            _train_gpu = _to_gpu_tensor_dict(train_data)
            _val_gpu = _to_gpu_tensor_dict(val_data)
            _test_gpu = _to_gpu_tensor_dict(cv_test_data)
            train_dataset = PreConvertedTensorDataset(_train_gpu[0], _train_gpu[1], _train_gpu[2], periods)
            val_dataset = PreConvertedTensorDataset(_val_gpu[0], _val_gpu[1], _val_gpu[2], periods)
            test_dataset = PreConvertedTensorDataset(_test_gpu[0], _test_gpu[1], _test_gpu[2], periods)
            _use_preconverted = True
            logger.info(f"GPU数据预转完成, 数据已常驻显存")
        else:
            train_dataset = MultiTimeframeDataset(train_data[0], train_data[1], train_data[2], periods)
            val_dataset = MultiTimeframeDataset(val_data[0], val_data[1], val_data[2], periods)
            test_dataset = MultiTimeframeDataset(cv_test_data[0], cv_test_data[1], cv_test_data[2], periods)

        _dl_kwargs = {'num_workers': 0, 'pin_memory': False}

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False, **_dl_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, **_dl_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, **_dl_kwargs)

        # ---- 创建模型(每折独立, 防止跨折泄露) ----
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
            output_size=_num_horizons,
        ).to(device)

        # 从GitHub Release下载最新模型作为初始化(仅第0折/非CV模式)
        if fold_idx == 0 or not _use_cv:
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
                            # 尝试加载到模型
                            _release_path = config.MODEL_PATH
                            if os.path.exists(_release_path):
                                try:
                                    _rel_ckpt = torch.load(_release_path, map_location=device, weights_only=False)
                                    _rel_cfg = _rel_ckpt.get('config', {})
                                    if _rel_cfg.get('timeframe_configs'):
                                        model.load_state_dict(_rel_ckpt['model_state_dict'])
                                        logger.info(f"Release模型权重已加载 (val_loss={_rel_ckpt.get('val_loss', 'N/A')})")
                                    else:
                                        logger.info("Release模型架构不匹配，从头训练")
                                except Exception as e:
                                    logger.warning(f"加载Release模型失败: {e}")
                        else:
                            logger.warning(f"Release模型下载失败: {dl.stderr.strip()}")
                    else:
                        logger.info("未找到已有Release")
                except Exception as e:
                    logger.warning(f"获取Release信息失败: {e}")
            else:
                logger.info("未找到GH_TOKEN, 跳过Release下载")

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        total_params, trainable_params = count_parameters(model)
        if fold_idx == 0:
            logger.info(f"模型参数: 总计 {total_params:,}, 可训练{trainable_params:,}")

        # ---- 优化器/学习率调度/损失函数 ----
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6,
        )

        # 类别平衡权重
        _train_labels = train_data[2]
        if _train_labels.ndim == 1:
            _train_labels = _train_labels.reshape(-1, 1)
        _pos_weights = []
        for h_idx in range(_num_horizons):
            _pos = (_train_labels[:, h_idx] == 1).sum()
            _neg = (_train_labels[:, h_idx] == 0).sum()
            if _pos > 0:
                pw = float(_neg) / float(_pos)
                pw = min(pw, 5.0)
                _pos_weights.append(pw)
            else:
                _pos_weights.append(1.0)

        class WeightedFocalLoss(nn.Module):
            """加权 Focal Loss

            在 WeightedBCE 基础上引入 Focal Loss 机制:
              FL(p_t) = -(1-p_t)^γ * log(p_t)
            - gamma=0 → 退化为标准 BCE
            - gamma=2.0 → 关注难分类样本(靠近决策边界的)
            - 保持 per_horizon pos_weight 做类别平衡

            金融涨跌分类中大量样本模糊不清,
            Focal Loss 让模型专注于有价值"难例"而非已分类正确的样本。
            """
            def __init__(self, per_horizon_weights, gamma=2.0):
                super().__init__()
                self.gamma = gamma
                self.register_buffer(
                    'weights',
                    torch.tensor(per_horizon_weights, dtype=torch.float32),
                )

            def forward(self, pred, target):
                # BCE 基础项
                bce = F.binary_cross_entropy(pred, target, reduction='none')
                # 预测置信度: p_t = p if y=1 else 1-p
                pt = torch.where(target >= 0.5, pred, 1 - pred)
                # Focal 调制因子: (1-p_t)^γ, 正确高置信样本loss被压制
                focal = (1 - pt) ** self.gamma
                # 类别平衡权重
                weight_vec = self.weights.to(target.device).unsqueeze(0)
                sample_weights = torch.where(target >= 0.5, weight_vec,
                                              torch.ones_like(target))
                return (bce * focal * sample_weights).mean()

        criterion = WeightedFocalLoss(_pos_weights, gamma=0.5)
        if fold_idx == 0:
            logger.info(f"使用 WeightedFocalLoss(gamma=0.5), 各窗口pos_weight={_pos_weights}")

        # ---- CV 各折训练循环 ----
        _max_epochs = config.EPOCHS
        best_val_loss = float('inf')
        patience_counter = 0
        epoch = 0
        fold_start_time = time.time()

        if fold_idx == 0:
            logger.info("=" * 60)
            logger.info(f"步骤 4: 开始训练(上限{_max_epochs} epochs)")
            logger.info("=" * 60)

        while epoch < _max_epochs:
            # 单折时间预算检查
            if _fold_time_budget and (time.time() - fold_start_time) >= _fold_time_budget:
                if _use_cv:
                    logger.info(f"Fold {fold_idx+1} 达到时间预算({_fold_time_budget/3600:.1f}h)")
                else:
                    logger.info(f"达到最大训练时长({_max_seconds/3600:.1f}小时), 停止训练")
                break

            t0 = time.time()
            epoch += 1

            # --- 训练 ---
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            _train_horizon_correct = [0] * _num_horizons
            _train_horizon_total = [0] * _num_horizons

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

                batch_size = labels.size(0)
                train_loss += loss.item() * batch_size
                preds = (outputs > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.numel()
                for h in range(_num_horizons):
                    _train_horizon_correct[h] += (preds[:, h] == labels[:, h]).sum().item()
                    _train_horizon_total[h] += labels[:, h].size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total
            _train_acc_per_h = [_train_horizon_correct[h] / max(_train_horizon_total[h], 1) for h in range(_num_horizons)]

            # --- 验证 ---
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            _val_horizon_correct = [0] * _num_horizons
            _val_horizon_total = [0] * _num_horizons

            with torch.no_grad():
                for tf_seqs, ctx, labels in val_loader:
                    if not _use_preconverted:
                        tf_seqs = {p: v.to(device) for p, v in tf_seqs.items()}
                        ctx = ctx.to(device)
                        labels = labels.to(device)
                    outputs = model(tf_seqs, ctx)
                    loss = criterion(outputs, labels)

                    batch_size = labels.size(0)
                    val_loss += loss.item() * batch_size
                    preds = (outputs > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.numel()
                    for h in range(_num_horizons):
                        _val_horizon_correct[h] += (preds[:, h] == labels[:, h]).sum().item()
                        _val_horizon_total[h] += labels[:, h].size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total
            _val_acc_per_h = [_val_horizon_correct[h] / max(_val_horizon_total[h], 1) for h in range(_num_horizons)]
            scheduler.step(val_loss)

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]['lr']
            fold_tag = f"[Fold {fold_idx+1}] " if _use_cv else ""

            logger.info(
                f"{fold_tag}Epoch {epoch:3d}/{_max_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | {elapsed:.1f}s"
            )
            _h_train_str = " | ".join([f"{config.PREDICTION_HORIZONS[h]}m:{_train_acc_per_h[h]:.3f}" for h in range(_num_horizons)])
            _h_val_str = " | ".join([f"{config.PREDICTION_HORIZONS[h]}m:{_val_acc_per_h[h]:.3f}" for h in range(_num_horizons)])
            logger.info(f"  {fold_tag}TrainAcc -> {_h_train_str}")
            logger.info(f"  {fold_tag}Val   Acc -> {_h_val_str}")

            # 保存本折最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                _fold_model_path = config.MODEL_PATH.replace('.pth', f'_fold{fold_idx}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'fold_idx': fold_idx,
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
                        'output_size': _num_horizons,
                        'horizons': config.PREDICTION_HORIZONS,
                    },
                }, _fold_model_path)
                logger.info(f"  {fold_tag}-> 保存最佳模型(val_loss={val_loss:.4f})")

                # 全局最佳追踪
                if val_loss < _best_across_folds['val_loss']:
                    _best_across_folds['val_loss'] = val_loss
                    _best_across_folds['fold_idx'] = fold_idx
                    _best_across_folds['model_path'] = _fold_model_path
                    _best_across_folds['epoch'] = epoch + 1
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    logger.info(f"{fold_tag}早停: 连续 {config.PATIENCE} 轮验证损失未改善")
                    break

    # ==================== CV 汇总: 选择最佳折模型 ====================
    if _use_cv:
        logger.info(f"\n{'='*60}")
        logger.info(f"CV 完成: 最佳模型来自 Fold {_best_across_folds['fold_idx']+1} "
                     f"(val_loss={_best_across_folds['val_loss']:.4f}, "
                     f"epoch={_best_across_folds['epoch']})")
        logger.info(f"{'='*60}")
        # 加载最佳折的模型用于测试
        _best_ckpt = torch.load(_best_across_folds['model_path'], map_location=device, weights_only=False)
    else:
        _best_ckpt = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(_best_ckpt['model_state_dict'])
    model.eval()

    test_loss, test_correct, test_total = 0.0, 0, 0
    all_preds, all_labels = [], []
    # 各窗口分别统计
    _test_h_correct = [0] * _num_horizons
    _test_h_total = [0] * _num_horizons

    with torch.no_grad():
        for tf_seqs, ctx, labels in test_loader:
            if not _use_preconverted:
                tf_seqs = {p: v.to(device) for p, v in tf_seqs.items()}
                ctx = ctx.to(device)
                labels = labels.to(device)
            outputs = model(tf_seqs, ctx)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            test_loss += loss.item() * batch_size
            preds = (outputs > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.numel()
            # 各窗口统计
            for h in range(_num_horizons):
                _test_h_correct[h] += (preds[:, h] == labels[:, h]).sum().item()
                _test_h_total[h] += labels[:, h].size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= test_total
    test_acc = test_correct / test_total

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 整体指标(所有窗口合并)
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
    # 各窗口独立指标
    for h_idx, h_name in enumerate(config.PREDICTION_HORIZONS):
        _h_acc = _test_h_correct[h_idx] / max(_test_h_total[h_idx], 1)
        _h_pred = all_preds[:, h_idx]
        _h_label = all_labels[:, h_idx]
        _h_tp = int(((_h_pred > 0.5) & (_h_label == 1)).sum())
        _h_fp = int(((_h_pred > 0.5) & (_h_label == 0)).sum())
        _h_tn = int(((_h_pred <= 0.5) & (_h_label == 0)).sum())
        _h_fn = int(((_h_pred <= 0.5) & (_h_label == 1)).sum())
        _h_prec = _h_tp / (_h_tp + _h_fp) if (_h_tp + _h_fp) > 0 else 0
        _h_rec = _h_tp / (_h_tp + _h_fn) if (_h_tp + _h_fn) > 0 else 0
        logger.info(f"  [{h_name}min窗口] Acc:{_h_acc:.4f} Prec:{_h_prec:.4f} Rec:{_h_rec:.4f} "
                     f"TP={_h_tp} FP={_h_fp} TN={_h_tn} FN={_h_fn}")
    logger.info(f"最佳模型来自Epoch {_best_ckpt['epoch']} (Fold {_best_ckpt.get('fold_idx', 0)+1})")

    # ==================== 保存最佳模型用于断点续训/Release ====================
    torch.save({
        'epoch': _best_ckpt['epoch'],
        'fold_idx': _best_ckpt.get('fold_idx', 0),
        'model_state_dict': _best_ckpt['model_state_dict'],
        'val_loss': _best_ckpt['val_loss'],
        'val_acc': _best_ckpt['val_acc'],
        'test_acc': test_acc,
        'test_f1': f1,
        'config': _best_ckpt.get('config', {}),
    }, config.MODEL_PATH)
    torch.save({
        'epoch': _best_ckpt['epoch'],
        'model_state_dict': model.state_dict(),
        'val_loss': _best_ckpt['val_loss'],
        'config': _best_ckpt.get('config', {}),
    }, config.MODEL_PATH_FINAL)
    logger.info(f"最佳模型已保存: {config.MODEL_PATH}")
    logger.info(f"最终模型已保存: {config.MODEL_PATH_FINAL}")

    # 上传到GitHub Release(仅CI环境)
    gh_token = os.environ.get('GH_TOKEN')
    if gh_token:
        try:
            import subprocess, json as _json
            from datetime import datetime
            tag_name = f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            result = subprocess.run(
                ['gh', 'release', 'view', '--json', 'tagName', '--jq', '.tagName'],
                capture_output=True, text=True, timeout=30,
            )
            existing_tag = result.stdout.strip() if result.returncode == 0 else None

            cv_note = f"CV {config.CV_N_FOLDS}折" if _use_cv else "单次切分"
            notes = (
                f"## 多周期融合LNN模型\n\n"
                f"- **数据划分**: {cv_note}\n"
                f"- **最佳模型**: `lnn_best.pth` (val_loss={_best_ckpt['val_loss']:.4f}, "
                f"test_acc={test_acc:.4f})\n"
                f"- **最终模型**: `lnn_final.pth` (完整状态 用于断点续训)\n"
            )

            if existing_tag:
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

    # 发送训练完成通知
    if notifier:
        _horizon_results = {}
        for h_idx, h_name in enumerate(config.PREDICTION_HORIZONS):
            _h_acc = _test_h_correct[h_idx] / max(_test_h_total[h_idx], 1)
            _horizon_results[f"{h_name}m"] = _h_acc
        notifier.send_training_complete(
            epoch=_best_ckpt['epoch'],
            val_loss=_best_ckpt['val_loss'],
            val_acc=_best_ckpt['val_acc'],
            test_acc=test_acc,
            precision=precision,
            recall=recall,
            f1=f1,
            horizon_results=_horizon_results,
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
