"""液态神经网络训练脚本 - 多周期融合版"""

import os
import time
import logging
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data_fetcher import HuobiDataFetcher
from notifier import MeoWNotifier
from features import (
    build_multi_tf_dataset, build_multi_symbol_dataset, split_multi_tf_dataset, rolling_cv_split,
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


def _strip_module_prefix(state_dict):
    """移除state_dict中因DataParallel产生的 'module.' 前缀

    仅在单GPU/CPU加载时需要。双GPU环境模型本身就是DataParallel包装的，
    保存时已经带了module.前缀，加载时如果再次strip会导致key不匹配错误。
    """
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len('module.'):] if k.startswith('module.') else k
        new_state_dict[new_key] = v
    logger.info(f"检测到DataParallel前缀(module.), 已自动移除 ({len(state_dict)} keys)")
    return new_state_dict


def _safe_load_state_dict(model, state_dict, device):
    """安全加载state_dict，自动处理DataParallel前缀问题

    根据模型和checkpoint的key前缀情况，自动决定是否需要strip/add前缀。
    """
    model_key_prefix = ''
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_key_prefix = 'module.'

    ckpt_keys = list(state_dict.keys())
    has_ckpt_prefix = any(k.startswith('module.') for k in ckpt_keys)

    model_keys = set(model.state_dict().keys())
    ckpt_keys_set = set(state_dict.keys())

    if model_key_prefix and not has_ckpt_prefix:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = f'module.{k}'
            new_state_dict[new_key] = v
        state_dict = new_state_dict
        logger.info(f"为checkpoint添加module.前缀以匹配DataParallel模型")
    elif has_ckpt_prefix and not model_key_prefix:
        state_dict = _strip_module_prefix(state_dict)
        logger.info(f"移除checkpoint的module.前缀以匹配非DataParallel模型")

    model.load_state_dict(state_dict)


def _load_best_fallback(model, device):
    """从checkpoints加载权重作为初始化的降级方案

    加载优先级:
    1. 最终模型 lnn_final.pth (最后训练保存的完整状态)
    2. 最佳模型 lnn_best.pth (val_loss最低)
    3. 各折模型 lnn_best_fold{idx}.pth
    """
    import glob

    # 尝试1: 加载最终模型 (最后保存, 带完整状态)
    try:
        final_ckpt = torch.load(config.MODEL_PATH_FINAL, map_location=device, weights_only=False)
        ckpt_config = final_ckpt.get('config', {})
        if 'timeframe_configs' in ckpt_config:
            _safe_load_state_dict(model, final_ckpt['model_state_dict'], device)
            logger.info(f"从最终模型 {config.MODEL_PATH_FINAL} 加载权重作为初始化")
            return
        else:
            logger.info("最终模型架构不匹配")
    except FileNotFoundError:
        logger.info(f"未找到 {config.MODEL_PATH_FINAL}")
    except Exception as e:
        logger.warning(f"加载最终模型失败: {e}")

    # 尝试2: 加载最佳模型 (val_loss最低)
    try:
        best_ckpt = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
        ckpt_config = best_ckpt.get('config', {})
        if 'timeframe_configs' in ckpt_config:
            _safe_load_state_dict(model, best_ckpt['model_state_dict'], device)
            logger.info("从最佳模型加载权重作为初始化")
            return
        else:
            logger.info("检测到旧架构checkpoint，尝试其他模型...")
    except FileNotFoundError:
        logger.info(f"未找到 {config.MODEL_PATH}，尝试加载折模型...")
    except Exception as e:
        logger.warning(f"加载best checkpoint失败: {e}，尝试其他模型...")

    # 尝试3: 加载各折模型
    fold_pattern = config.MODEL_PATH.replace('.pth', '_fold*.pth')
    fold_files = sorted(glob.glob(fold_pattern))

    if fold_files:
        fold_path = fold_files[-1]  # 最后一个折模型（通常是性能最好的）
        try:
            fold_ckpt = torch.load(fold_path, map_location=device, weights_only=False)
            ckpt_config = fold_ckpt.get('config', {})
            if 'timeframe_configs' in ckpt_config:
                _safe_load_state_dict(model, fold_ckpt['model_state_dict'], device)
                logger.info(f"从折模型 {fold_path} 加载权重作为初始化")
                return
            else:
                logger.info(f"折模型 {fold_path} 架构不匹配")
        except Exception as e:
            logger.warning(f"加载折模型 {fold_path} 失败: {e}")

    logger.info("未找到可加载的模型，从头训练新模型")


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

    # ==================== 多GPU环境检测 ====================
    _n_gpu = torch.cuda.device_count() if _use_cuda else 0
    if _n_gpu > 1:
        logger.info(f"检测到 {_n_gpu} 块GPU, 将使用 DataParallel 自动并行")
        for i in range(_n_gpu):
            props = torch.cuda.get_device_properties(i)
            used = torch.cuda.memory_allocated(i) / (1024**3)
            free = props.total_memory / (1024**3) - used
            logger.info(
                f"  GPU {i}: {props.name}, "
                f"总显存{props.total_memory/(1024**3):.1f}GB, "
                f"已用{used:.1f}GB, 可用{free:.1f}GB, "
                f"计算能力sm_{props.major}{props.minor}"
            )
    elif _n_gpu == 1:
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name}, "
                    f"总显存{props.total_memory/(1024**3):.1f}GB, "
                    f"计算能力sm_{props.major}{props.minor}")

    def _get_available_memory_gb():
        """获取可用内存/显存(GB).
        
        GPU: 使用 torch.cuda.mem_get_info() 获取驱动层真实剩余显存,
             替代 memory_allocated (PyTorch内部计数器, 会漏算预留缓存).
             取所有卡的最小值(瓶颈).
        CPU: 依次尝试 psutil → /proc/meminfo.
        """
        if torch.cuda.is_available() and _use_cuda:
            min_free = float('inf')
            for i in range(max(1, _n_gpu)):
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                free_gb = free_bytes / (1024**3)
                if free_gb < min_free:
                    min_free = free_gb
            return min_free

        # CPU: 先试 psutil
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            pass

        # CPU: 再试 /proc/meminfo (Linux, 含GitHub Actions)
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemFree:'):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        except (FileNotFoundError, IOError, ValueError):
            pass

        # CPU: 均失败, 返回None由调用方决定回退策略
        logger.warning("无法检测系统可用内存, 将使用默认BATCH_SIZE")
        return None

    def _auto_batch_size(device):
        """根据可用显存/内存自动估算全局BATCH_SIZE

        调用时机: 模型+数据已在GPU上, free_gb 即真正可用于batch的显存.
        多GPU (DataParallel) 时全局 batch = per_gpu_batch * n_gpu,
        因为每卡只分摊全局batch的 1/n_gpu 样本.
        """
        gpu_count = max(1, _n_gpu)

        if device.type == 'cuda':
            # 清空PyTorch缓存后再测量, 得到更准确的驱动层剩余显存
            torch.cuda.empty_cache()
            for i in range(gpu_count):
                torch.cuda.synchronize(i)
            free_gb = _get_available_memory_gb()

            # 安全裕度: 只用剩余显存的70%, 留30%给激活值/梯度/临时变量
            SAFETY_FACTOR = 0.7
            eff_free_gb = free_gb * SAFETY_FACTOR

            # 以下为**每块GPU**能安全运行的batch大小
            if eff_free_gb < 1:
                per_gpu_batch = 32
            elif eff_free_gb < 2:
                per_gpu_batch = 64
            elif eff_free_gb < 3:
                per_gpu_batch = 128
            elif eff_free_gb < 5:
                per_gpu_batch = 256
            elif eff_free_gb < 8:
                per_gpu_batch = 512
            elif eff_free_gb < 12:
                per_gpu_batch = 1024
            else:
                per_gpu_batch = 1536

            # 全局 batch = 每卡batch * 卡数
            global_batch = per_gpu_batch * gpu_count

            logger.info(
                f"显存: 驱动层剩余{free_gb:.1f}GB(有效{eff_free_gb:.1f}GB×{SAFETY_FACTOR:.0%}), "
                f"{gpu_count}卡 → 每卡batch={per_gpu_batch}, "
                f"全局batch={global_batch}"
            )
            return global_batch
        else:
            free_gb = _get_available_memory_gb()
            if free_gb is None:
                # 完全无法检测, 用配置默认值
                logger.info(f"无法检测内存, 使用默认BATCH_SIZE={config.BATCH_SIZE}")
                return config.BATCH_SIZE
            if free_gb < 4:
                logger.warning(f"系统可用内存{free_gb:.1f}GB，降低BATCH_SIZE至128")
                return 128
            elif free_gb < 8:
                logger.info(f"系统可用内存{free_gb:.1f}GB，降低BATCH_SIZE至256")
                return 256
            elif free_gb < 16:
                logger.info(f"系统可用内存{free_gb:.1f}GB，降低BATCH_SIZE至384")
                return 384
            elif free_gb < 24:
                logger.info(f"系统可用内存{free_gb:.1f}GB，使用BATCH_SIZE=512")
                return 512
            elif free_gb < 32:
                logger.info(f"系统可用内存{free_gb:.1f}GB，使用BATCH_SIZE=768")
                return 768
            else:
                logger.info(f"系统可用内存充足({free_gb:.1f}GB)，使用BATCH_SIZE=1024")
                return 1024

    periods = list(config.TIMEFRAMES.keys())
    logger.info(f"多周期融合 {periods}")

    # 初始化通知器
    notifier = None
    if config.MEOW_NICKNAME:
        notifier = MeoWNotifier(config.MEOW_NICKNAME)
        notifier.send_training_start(config.EPOCHS)

    # ==================== 1. 获取数据 ====================
    logger.info("=" * 60)
    logger.info("步骤 1: 获取多币种多周期K线数据")
    logger.info("=" * 60)

    fetcher = HuobiDataFetcher()
    all_symbols_data = fetcher.fetch_all_symbols_data()

    # ==================== 2. 构建数据集 ====================
    logger.info("=" * 60)
    logger.info("步骤 2: 多币种多周期特征工程与数据集构建")
    logger.info("=" * 60)

    X_dict, X_ctx, y = build_multi_symbol_dataset(
        all_symbols_data, fetcher,
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
    _stop_mode = getattr(config, 'TRAIN_STOP_MODE', 'both')
    _max_seconds = getattr(config, 'MAX_TRAIN_SECONDS', None)
    _max_epochs = config.EPOCHS

    _use_epoch_limit = _stop_mode in ('epochs_only', 'both')
    _use_time_limit = _stop_mode in ('time_only', 'both')

    if _stop_mode == 'infinite':
        logger.info("训练模式: 无限训练(仅靠早停停止)")
    elif _stop_mode == 'epochs_only':
        logger.info(f"训练模式: 仅EPOCHS限制 ({_max_epochs} epochs)")
    elif _stop_mode == 'time_only':
        logger.info(f"训练模式: 仅时间限制 ({_max_seconds/3600:.1f}h)")
    else:
        logger.info(f"训练模式: 双重限制 (epochs={_max_epochs}, time={_max_seconds/3600:.1f}h)")

    _n_folds = len(cv_folds_normalized)
    _fold_time_budget = None
    if _use_time_limit and _max_seconds and _use_cv:
        _fold_time_budget = _max_seconds / _n_folds
        logger.info(f"每折时间预算: {_fold_time_budget/3600:.1f}h")
    elif _use_time_limit and _max_seconds:
        logger.info(f"训练时间限制: {_max_seconds/3600:.1f}h")

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

        # ---- 创建数据集(预转GPU) ----
        _use_preconverted = False
        if _use_cuda:
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

        # ---- 创建模型(每折独立, 防止跨折泄露) ----
        feat_size = len(SEQ_FEATURE_COLS)
        _dual_norm = getattr(config, 'USE_DUAL_NORMALIZATION', False)
        if _dual_norm:
            feat_size *= 2  # 双标准化: [局部K线形态 || 绝对价格位置]
        ctx_size = len(CONTEXT_FEATURE_COLS)
        tf_configs = {
            p: {'seq_length': cfg['seq_length'], 'feature_size': feat_size}
            for p, cfg in config.TIMEFRAMES.items()
        }

        # 获取Transformer配置
        _use_transformer = getattr(config, 'USE_TRANSFORMER', False)
        _transformer_heads = getattr(config, 'TRANSFORMER_HEADS', 4)
        _cross_attn_heads = getattr(config, 'CROSS_ATTN_HEADS', 4)

        if fold_idx == 0:
            _norm_desc = f" (双标准化) ×2={feat_size}" if _dual_norm else ""
            logger.info(f"序列特征维度: {len(SEQ_FEATURE_COLS)}{_norm_desc}")
            logger.info(f"上下文特征维度: {ctx_size}")

        model = MultiTimeframeLNN(
            timeframe_configs=tf_configs,
            context_feature_size=ctx_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            output_size=_num_horizons,
            use_transformer=_use_transformer,
            transformer_heads=_transformer_heads,
            cross_attn_heads=_cross_attn_heads,
        ).to(device)

        if fold_idx == 0:
            logger.info(f"Transformer增强: {'启用' if _use_transformer else '禁用'}")
            if _use_transformer:
                logger.info(f"  Transformer头数: {_transformer_heads}")
                logger.info(f"  跨周期注意力头数: {_cross_attn_heads}")

        # 从GitHub Release下载最新模型作为初始化(仅第0折/非CV模式)
        if fold_idx == 0 or not _use_cv:
            from predict import download_release_model as _dl_release
            if _dl_release():
                logger.info("尝试加载Release下载的模型...")
            else:
                logger.info("尝试从checkpoints加载模型...")
            _load_best_fallback(model, device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        total_params, trainable_params = count_parameters(model)
        if fold_idx == 0:
            logger.info(f"模型参数: 总计 {total_params:,}, 可训练{trainable_params:,}")

        # ---- **模型+数据已在GPU上**, 在此测量剩余显存计算batch ----
        # 重新加载config, 确保运行中修改config.py能即时生效(多折CV后续折)
        importlib.reload(config)
        _auto_mode = config.USE_AUTO_BATCH_SIZE
        if _auto_mode:
            _effective_batch_size = _auto_batch_size(device)
            logger.info(f"BATCH_SIZE: auto={_auto_mode} → {_effective_batch_size}")
        else:
            _effective_batch_size = config.BATCH_SIZE
            logger.info(f"BATCH_SIZE: auto={_auto_mode} → 配置值={config.BATCH_SIZE}")
        _accum_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        _effective_batch_size_accum = _effective_batch_size * _accum_steps
        if fold_idx == 0 and _accum_steps > 1:
            logger.info(
                f"梯度累积: {_accum_steps} 步, "
                f"等效batch_size={_effective_batch_size} × {_accum_steps} = {_effective_batch_size_accum}"
            )
        _dl_kwargs = {'num_workers': 0, 'pin_memory': False}
        train_loader = DataLoader(train_dataset, batch_size=_effective_batch_size, shuffle=True, drop_last=False, **_dl_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=_effective_batch_size, shuffle=False, **_dl_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=_effective_batch_size, shuffle=False, **_dl_kwargs)

        # ---- 优化器/学习率调度/损失函数 ----
        # 根据 Linear Scaling Rule 缩放 LR: 大 batch 配大 LR
        _scaled_lr = config.get_scaled_learning_rate(_effective_batch_size)
        if fold_idx == 0:
            logger.info(
                f"LR 线性缩放: base_batch={config.BASE_BATCH_SIZE}, "
                f"target_batch={_effective_batch_size}, "
                f"LR={config.LEARNING_RATE:.2e} → {_scaled_lr:.2e}"
            )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=_scaled_lr,
            weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-4),
        )
        # steps_per_epoch 按 optimizer 步数计（非 micro-batch）
        steps_per_epoch = max(1, len(train_loader) // _accum_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=_scaled_lr * config.ONECYCLE_MAX_LR_SCALE,
            epochs=config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.ONECYCLE_PCT_START,
            anneal_strategy=config.ONECYCLE_ANNEAL_STRATEGY,
            final_div_factor=config.ONECYCLE_FINAL_DIV_FACTOR,
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

        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=0.5, per_horizon_weights=None):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                if per_horizon_weights is not None:
                    self.register_buffer(
                        'weights',
                        torch.tensor(per_horizon_weights, dtype=torch.float32),
                    )
                else:
                    self.register_buffer('weights', torch.ones(_num_horizons))

            def forward(self, logits, target):
                bce = F.binary_cross_entropy_with_logits(
                    logits, target, reduction='none'
                )
                pt = torch.exp(-bce)
                focal_weight = (1 - pt) ** self.gamma
                weight_vec = self.weights.to(target.device).unsqueeze(0)
                sample_weights = torch.where(target >= 0.5, weight_vec,
                                             torch.ones_like(target))
                return (bce * sample_weights * self.alpha).mean()

        criterion = FocalLoss(
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
            per_horizon_weights=_pos_weights
        )
        if fold_idx == 0:
            logger.info(f"使用 FocalLoss(alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA}), 各窗口pos_weight={_pos_weights}")

        # ---- CV 各折训练循环 ----
        best_val_loss = float('inf')
        patience_counter = 0
        epoch = 0
        fold_start_time = time.time()

        if fold_idx == 0:
            _log_epochs = f"epochs={_max_epochs}" if _use_epoch_limit else "∞"
            _log_time = f"time={_max_seconds/3600:.1f}h" if (_use_time_limit and _max_seconds) else "∞"
            logger.info("=" * 60)
            logger.info(f"步骤 4: 开始训练 ({_log_epochs}, {_log_time})")
            logger.info("=" * 60)

        while True:
            # 停止条件检查
            if _use_epoch_limit and epoch >= _max_epochs:
                if fold_idx == 0:
                    logger.info(f"达到最大epoch数 ({_max_epochs}), 停止训练")
                break

            if _use_time_limit and _max_seconds:
                _time_budget = _fold_time_budget or _max_seconds
                _time_elapsed = time.time() - fold_start_time
                if _time_elapsed >= _time_budget:
                    if _use_cv:
                        logger.info(f"Fold {fold_idx+1} 达到时间预算({_time_budget/3600:.1f}h)")
                    else:
                        logger.info(f"达到最大训练时长({_time_budget/3600:.1f}h), 停止训练")
                    break

            t0 = time.time()
            epoch += 1

            # --- 训练 ---
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            _train_horizon_correct = [0] * _num_horizons
            _train_horizon_total = [0] * _num_horizons

            optimizer.zero_grad()
            for batch_idx, (tf_seqs, ctx, labels) in enumerate(train_loader):
                if not _use_preconverted:
                    tf_seqs = {p: v.to(device) for p, v in tf_seqs.items()}
                    ctx = ctx.to(device)
                    labels = labels.to(device)

                outputs = model(tf_seqs, ctx)
                loss = criterion(outputs, labels)
                # 梯度累积: loss 除以累积步数，使等效 batch 的梯度量级一致
                scaled_loss = loss / _accum_steps
                scaled_loss.backward()

                batch_size = labels.size(0)
                # 记录原始（未缩放）的 loss 用于日志
                train_loss += loss.item() * batch_size
                preds = (outputs > 0).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.numel()
                for h in range(_num_horizons):
                    _train_horizon_correct[h] += (preds[:, h] == labels[:, h]).sum().item()
                    _train_horizon_total[h] += labels[:, h].size(0)

                # 每 _accum_steps 个 micro-batch 执行一次 optimizer 步
                if (batch_idx + 1) % _accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            train_loss /= train_total
            train_acc = train_correct / train_total
            _train_acc_per_h = [_train_horizon_correct[h] / max(_train_horizon_total[h], 1) for h in range(_num_horizons)]

            # 梯度诊断: 每 epoch 末尾采样一次总梯度范数
            _total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    _total_norm += p.grad.norm().item() ** 2
            _total_norm = _total_norm ** 0.5
            _grad_info = f"grad_norm={_total_norm:.4e}"

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
                    preds = (outputs > 0).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.numel()
                    for h in range(_num_horizons):
                        _val_horizon_correct[h] += (preds[:, h] == labels[:, h]).sum().item()
                        _val_horizon_total[h] += labels[:, h].size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total
            _val_acc_per_h = [_val_horizon_correct[h] / max(_val_horizon_total[h], 1) for h in range(_num_horizons)]

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]['lr']
            fold_tag = f"[Fold {fold_idx+1}] " if _use_cv else ""

            logger.info(
                f"{fold_tag}Epoch {epoch:3d}/{_max_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | {_grad_info} | {elapsed:.1f}s"
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
                        'use_transformer': _use_transformer,
                        'transformer_heads': _transformer_heads,
                        'cross_attn_heads': _cross_attn_heads,
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
        _best_ckpt = torch.load(_best_across_folds['model_path'], map_location=device, weights_only=False)
    _safe_load_state_dict(model, _best_ckpt['model_state_dict'], device)
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
            preds = (outputs > 0).float()
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
