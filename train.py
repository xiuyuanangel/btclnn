"""液态神经网络训练脚本 - 多周期融合版"""

import os
import time
import random
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

# ==================== 全局随机种子(可复现性) ====================
SEED = getattr(config, 'RANDOM_SEED', 42)


def set_global_seed(seed=SEED):
    """设置全局随机种子, 保证多次运行结果可复现

    覆盖 python hash / random / numpy / torch(CPU+CUDA)。
    CUDA 相关设置带异常保护, CPU-only 环境(如GitHub Actions)不会报错。

    Args:
        seed: 随机种子, 默认取 config.RANDOM_SEED 或 42
    """
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _cuda_note = ''
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            _cuda_note = ', CUDA确定性模式已开启(cudnn.deterministic=True)'
        except Exception as e:
            # CPU-only / 无cudnn 环境不应因此中断训练
            logger.warning(f"设置CUDA确定性模式失败(忽略): {e}")
    logger.info(f"全局随机种子已设置: seed={seed}{_cuda_note}")


def _seed_worker(worker_id):
    """DataLoader worker 种子初始化(num_workers>0 时生效)

    Args:
        worker_id: DataLoader 分配的 worker 序号
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# 在构建模型/数据之前立即固定种子
set_global_seed(SEED)


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

    Returns:
        dict | None: 成功加载权重的那个checkpoint字典(供后续恢复
        optimizer/scheduler/训练进度使用); 未加载到任何模型时返回 None。
    """
    import glob

    # 尝试1: 加载最终模型 (最后保存, 带完整状态)
    try:
        final_ckpt = torch.load(config.MODEL_PATH_FINAL, map_location=device, weights_only=False)
        ckpt_config = final_ckpt.get('config', {})
        if 'timeframe_configs' in ckpt_config:
            _safe_load_state_dict(model, final_ckpt['model_state_dict'], device)
            logger.info(f"从最终模型 {config.MODEL_PATH_FINAL} 加载权重作为初始化")
            return final_ckpt
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
            return best_ckpt
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
                return fold_ckpt
            else:
                logger.info(f"折模型 {fold_path} 架构不匹配")
        except Exception as e:
            logger.warning(f"加载折模型 {fold_path} 失败: {e}")

    logger.info("未找到可加载的模型，从头训练新模型")
    return None


def _check_optimizer_compatible(optimizer, opt_state):
    """校验checkpoint中的optimizer状态与当前optimizer是否形状兼容

    只做只读校验, 不修改任何状态。不兼容时抛出 ValueError, 由调用方降级处理。

    Args:
        optimizer: 当前新建的优化器
        opt_state: checkpoint 中的 optimizer_state_dict

    Raises:
        ValueError: param_group 数量、参数个数或动量张量形状不一致
    """
    cur_state = optimizer.state_dict()
    cur_groups = cur_state.get('param_groups', [])
    ckpt_groups = opt_state.get('param_groups', [])
    if len(cur_groups) != len(ckpt_groups):
        raise ValueError(
            f"param_groups数量不一致(当前{len(cur_groups)} vs checkpoint{len(ckpt_groups)})"
        )
    for gi, (cur_g, ckpt_g) in enumerate(zip(cur_groups, ckpt_groups)):
        n_cur = len(cur_g.get('params', []))
        n_ckpt = len(ckpt_g.get('params', []))
        if n_cur != n_ckpt:
            raise ValueError(
                f"param_group[{gi}]参数个数不一致(当前{n_cur} vs checkpoint{n_ckpt})"
            )

    # 逐参数比对动量张量形状(Adam: exp_avg / exp_avg_sq)
    cur_params = [p for g in optimizer.param_groups for p in g['params']]
    saved_state = opt_state.get('state', {}) or {}
    for idx, param in enumerate(cur_params):
        entry = saved_state.get(idx)
        if not entry:
            continue
        for k, v in entry.items():
            if torch.is_tensor(v) and v.dim() > 0 and tuple(v.shape) != tuple(param.shape):
                raise ValueError(
                    f"参数[{idx}]的'{k}'形状不匹配(当前{tuple(param.shape)} vs checkpoint{tuple(v.shape)})"
                )


def _restore_train_state(ckpt, optimizer, scheduler, expected_total_steps=None):
    """断点续训: 恢复 optimizer / scheduler 状态与训练进度

    向后兼容: checkpoint 缺少相关key或形状不兼容时, 自动降级为
    "仅加载权重 + 全新optimizer"(即修复前的行为), 只打印warning, 不抛异常。

    Args:
        ckpt: _load_best_fallback 返回的checkpoint字典, 可能为 None
        optimizer: 当前新建的优化器
        scheduler: 当前新建的 OneCycleLR 调度器
        expected_total_steps: 当前配置下的计划总步数(仅用于一致性告警)

    Returns:
        tuple[int, int]: (start_epoch, start_step); 全新训练时为 (0, 0)
    """
    if not isinstance(ckpt, dict):
        logger.info("Starting fresh training | 全新训练(无可续训的checkpoint)")
        return 0, 0

    # ---- 1. 恢复优化器状态(动量/二阶矩/LR) ----
    opt_state = ckpt.get('optimizer_state_dict')
    opt_restored = False
    if opt_state is None:
        logger.warning(
            "checkpoint中缺少 optimizer_state_dict(旧版模型), 降级为新建优化器, "
            "本次训练的动量与LR调度将从头开始"
        )
    else:
        try:
            _check_optimizer_compatible(optimizer, opt_state)
            optimizer.load_state_dict(opt_state)
            opt_restored = True
            logger.info("已恢复 optimizer 状态(动量/自适应二阶矩)")
        except Exception as e:
            logger.warning(f"optimizer状态与当前模型不兼容, 降级为新建优化器: {e}")

    # ---- 2. 恢复LR调度器状态(OneCycleLR 的 last_epoch/total_steps) ----
    sch_state = ckpt.get('scheduler_state_dict')
    sch_restored = False
    if sch_state is None:
        logger.warning("checkpoint中缺少 scheduler_state_dict(旧版模型), LR调度将从头开始warmup")
    elif not opt_restored:
        logger.warning("optimizer未恢复, 为避免LR轨迹与权重错配, 跳过scheduler状态恢复")
    else:
        try:
            saved_total = sch_state.get('total_steps')
            scheduler.load_state_dict(sch_state)
            sch_restored = True
            cur_total = int(getattr(scheduler, 'total_steps', 0) or 0)
            cur_last = int(getattr(scheduler, 'last_epoch', 0) or 0)
            logger.info(
                f"已恢复 scheduler 状态: last_step={cur_last}/{cur_total} "
                f"(LR轨迹从上次断点继续, 不重启warmup)"
            )
            if expected_total_steps and saved_total and int(saved_total) != int(expected_total_steps):
                logger.warning(
                    f"scheduler总步数与本次配置不一致(checkpoint={saved_total}, "
                    f"当前计划={expected_total_steps}), 沿用checkpoint轨迹以保持LR连续"
                )
            if cur_total and cur_last >= cur_total - 1:
                logger.warning(
                    f"OneCycleLR 已走完全部 {cur_total} 步, 本次续训将保持末端LR"
                    f"(不会重启warmup, 也不会因步数越界报错)"
                )
        except Exception as e:
            logger.warning(f"scheduler状态恢复失败, LR调度将从头开始: {e}")

    # ---- 3. 训练进度(epoch / global_step) ----
    # 仅在 optimizer+scheduler 均成功恢复时才继承进度计数,
    # 否则(旧checkpoint)保持修复前行为: 从 0 重新计数, 避免"LR重启warmup但预算被砍"。
    if not (opt_restored and sch_restored):
        logger.info("Starting fresh training | 仅继承权重, epoch/global_step 从0开始计数")
        return 0, 0

    start_epoch = ckpt.get('completed_epochs')
    if start_epoch is None:
        # 旧checkpoint只有 'epoch'(保存时为 epoch+1), 容忍1轮误差
        start_epoch = ckpt.get('epoch', 0)
    try:
        start_epoch = max(0, int(start_epoch))
    except (TypeError, ValueError):
        start_epoch = 0

    start_step = ckpt.get('global_step')
    if start_step is None:
        start_step = int(getattr(scheduler, 'last_epoch', 0) or 0)
    try:
        start_step = max(0, int(start_step))
    except (TypeError, ValueError):
        start_step = 0

    logger.info(f"Resuming training from epoch={start_epoch}, global_step={start_step}")
    return start_epoch, start_step


def compute_class_distribution(train_labels, num_horizons, num_classes):
    """统计训练集各窗口的类别分布

    Args:
        train_labels: np.array (N, num_horizons) 训练集标签
        num_horizons: 预测窗口数
        num_classes: 类别数(3=跌/平/涨)

    Returns:
        list of list: [horizon][class] = 样本数
    """
    counts = []
    for h_idx in range(num_horizons):
        col = train_labels[:, h_idx]
        counts.append([int((col == c).sum()) for c in range(num_classes)])
    return counts


def compute_class_weights(class_counts, num_classes):
    """由类别分布计算类权重(受 config 开关控制)

    方案:
      - 'inv_count': w_c = total / (C * count_c)  → 各类梯度贡献均等(默认)
      - 'inv_sqrt' : w_c = 1/sqrt(count_c) 归一化到均值1 → 更温和
    中性类(c=1)再乘 config.NEUTRAL_WEIGHT_SCALE。

    重要: NEUTRAL_WEIGHT_SCALE 历史硬编码为 0.5, 而实测「平」类占比高达
    24.9%~46.6%(并非少数类), 该降权是模型从不输出「平」(中性Recall=0.0000)
    的直接原因。现默认 1.0, 不再人为压制中性类。

    Args:
        class_counts: list of list, [horizon][class] = 样本数
        num_classes: 类别数

    Returns:
        list of list: [horizon][class] = 权重值 (float)
    """
    use_weights = getattr(config, 'USE_CLASS_WEIGHTS', True)
    scheme = getattr(config, 'CLASS_WEIGHT_SCHEME', 'inv_count')
    neutral_scale = float(getattr(config, 'NEUTRAL_WEIGHT_SCALE', 1.0))

    all_weights = []
    for counts in class_counts:
        if not use_weights:
            all_weights.append([1.0] * num_classes)
            continue

        total = sum(counts)
        weights = []
        for c in range(num_classes):
            n_c = counts[c]
            if n_c <= 0 or total <= 0:
                weights.append(1.0)
                continue
            if scheme == 'inv_sqrt':
                w = 1.0 / float(np.sqrt(n_c))
            else:  # 'inv_count'
                w = total / float(num_classes * n_c)
            weights.append(w)

        if scheme == 'inv_sqrt':
            # 归一化到均值1, 使量级与 inv_count 方案可比
            _mean = sum(weights) / max(len(weights), 1)
            if _mean > 0:
                weights = [w / _mean for w in weights]

        # 中性类额外缩放(默认1.0=不干预)
        if num_classes >= 3 and neutral_scale != 1.0:
            weights[1] *= neutral_scale

        all_weights.append(weights)
    return all_weights


def log_class_distribution(class_counts, class_weights, horizons, num_classes):
    """打印各窗口的类别分布与实际生效的类权重(便于QA确认生效)"""
    use_weights = getattr(config, 'USE_CLASS_WEIGHTS', True)
    scheme = getattr(config, 'CLASS_WEIGHT_SCHEME', 'inv_count')
    neutral_scale = float(getattr(config, 'NEUTRAL_WEIGHT_SCALE', 1.0))

    logger.info(
        f"类权重: {'启用' if use_weights else '禁用'} "
        f"(方案={scheme}, 中性缩放={neutral_scale})"
    )
    _names = ["跌", "平", "涨"] if num_classes >= 3 else ["跌", "涨"]
    for h_idx, counts in enumerate(class_counts):
        total = max(sum(counts), 1)
        h_name = horizons[h_idx] if h_idx < len(horizons) else h_idx
        _dist = ", ".join(
            f"{_names[c] if c < len(_names) else c}={counts[c]}"
            f"({counts[c]/total*100:.1f}%)"
            for c in range(num_classes)
        )
        _w = ", ".join(f"{w:.4f}" for w in class_weights[h_idx])
        logger.info(f"  [{h_name}m] 各类样本: {_dist} | class_weight=[{_w}]")


def build_class_balanced_sampler(train_labels, class_counts, generator=None):
    """构建按类反比加权的 WeightedRandomSampler (仅在配置开启时使用)

    以最短窗口(索引0)的标签作为重采样依据, 每个样本的采样权重 = 1/该类样本数。
    与已有的 DataLoader 种子修复兼容: 复用传入的 generator 保证可复现。

    Args:
        train_labels: np.array (N, num_horizons) 训练集标签
        class_counts: list of list, [horizon][class] = 样本数
        generator: torch.Generator, 用于可复现采样

    Returns:
        torch.utils.data.WeightedRandomSampler
    """
    from torch.utils.data import WeightedRandomSampler

    base_counts = class_counts[0]
    per_class_w = [1.0 / max(n, 1) for n in base_counts]
    labels_h0 = train_labels[:, 0].astype(int)
    sample_weights = np.array(
        [per_class_w[c] if 0 <= c < len(per_class_w) else 0.0 for c in labels_h0],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    logger.info(
        f"启用类别均衡重采样 WeightedRandomSampler: "
        f"基准窗口各类样本={base_counts}, 采样数={len(sample_weights)}"
    )
    return sampler


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

    # 多标签维度 & 分类类别数
    _num_horizons = len(config.PREDICTION_HORIZONS)
    _num_classes = getattr(config, 'LABEL_NUM_CLASSES', 2)
    logger.info(f"多标签训练: {_num_horizons} 个预测窗口 -> {config.PREDICTION_HORIZONS}, "
                 f"{'三' if _num_classes >= 3 else '二'}分类模式")

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
        feat_size = len(SEQ_FEATURE_COLS)  # 9维: close,vol,return_1,high,low,bb_width,bb_pct,close_ratio,vol_ratio
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
            logger.info(f"序列特征维度: {feat_size} ({len(SEQ_FEATURE_COLS)}个原始特征)")
            logger.info(f"上下文特征维度: {ctx_size}")

        model = MultiTimeframeLNN(
            timeframe_configs=tf_configs,
            context_feature_size=ctx_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            output_size=_num_horizons,
            num_classes_per_head=_num_classes,
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
        # _resume_ckpt: 续训用的完整checkpoint(含optimizer/scheduler状态), 无则为None
        # Release下载与本地checkpoints两条路径最终都走 _load_best_fallback, 续训逻辑统一生效
        _resume_ckpt = None
        if fold_idx == 0 or not _use_cv:
            from predict import download_release_model as _dl_release
            if _dl_release():
                logger.info("尝试加载Release下载的模型...")
            else:
                logger.info("尝试从checkpoints加载模型...")
            _resume_ckpt = _load_best_fallback(model, device)

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
        _dl_kwargs = {'num_workers': 0, 'pin_memory': False, 'worker_init_fn': _seed_worker}
        # 固定 shuffle 采样顺序(当前 num_workers=0, 全局种子已足够;
        # 显式generator保证每折的打乱顺序与运行历史无关, 可复现)
        _dl_generator = torch.Generator()
        _dl_generator.manual_seed(SEED + fold_idx)

        # ---- 训练集类别分布 + 类权重(损失与可选重采样共用同一份统计) ----
        _train_labels = train_data[2]
        if _train_labels.ndim == 1:
            _train_labels = _train_labels.reshape(-1, 1)
        _class_counts = compute_class_distribution(_train_labels, _num_horizons, _num_classes)
        _per_horizon_weights = compute_class_weights(_class_counts, _num_classes)
        log_class_distribution(_class_counts, _per_horizon_weights,
                               config.PREDICTION_HORIZONS, _num_classes)

        # 可选: 类别均衡重采样(默认关闭, 实测分布不极端时无需开启)
        _train_sampler = None
        if getattr(config, 'USE_CLASS_BALANCED_SAMPLER', False) and _num_classes >= 3:
            _train_sampler = build_class_balanced_sampler(
                _train_labels, _class_counts, generator=_dl_generator
            )

        if _train_sampler is not None:
            # 使用 sampler 时不可同时指定 shuffle
            train_loader = DataLoader(train_dataset, batch_size=_effective_batch_size,
                                      sampler=_train_sampler, drop_last=False, **_dl_kwargs)
        else:
            train_loader = DataLoader(train_dataset, batch_size=_effective_batch_size, shuffle=True, drop_last=False, generator=_dl_generator, **_dl_kwargs)
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

        # ---- 断点续训: 恢复optimizer/scheduler状态与训练进度 ----
        # 必须在 optimizer/scheduler 创建之后执行(权重已在上面加载完毕)
        _start_epoch, _start_step = _restore_train_state(
            _resume_ckpt, optimizer, scheduler,
            expected_total_steps=steps_per_epoch * config.EPOCHS,
        )
        if _use_epoch_limit and _start_epoch >= _max_epochs:
            logger.warning(
                f"续训起点 epoch={_start_epoch} 已达 EPOCHS上限({_max_epochs}), "
                f"为避免本次run空转, 将从 epoch={max(0, _max_epochs - 1)} 继续; "
                f"如需继续长训请调大 config.EPOCHS"
            )
            _start_epoch = max(0, _max_epochs - 1)

        # 损失函数: 二分类用 FocalLoss(BCEWithLogits), 三分类用 FocalCrossEntropy(含类权重)
        # 注: _train_labels / _class_counts / _per_horizon_weights 已在 DataLoader 构建前算好
        if _num_classes >= 3:
            # 三分类模式: CrossEntropyLoss with ignore_index=1 (中性)
            # 每个horizon独立计算class weights
            class FocalCrossEntropyLoss(nn.Module):
                def __init__(self, gamma=0.5, per_horizon_weights=None):
                    super().__init__()
                    self.gamma = gamma
                    if per_horizon_weights is not None:
                        # per_horizon_weights: list of shape (num_classes,) per horizon
                        self.register_buffer(
                            'weights',
                            torch.as_tensor(per_horizon_weights, dtype=torch.float32),
                        )
                    else:
                        self.register_buffer(
                            'weights',
                            torch.ones(_num_horizons, _num_classes)
                        )

                def forward(self, logits, target):
                    # logits: (batch, num_horizons * num_classes) -> reshape
                    batch_size = logits.size(0)
                    logits_3d = logits.view(batch_size, _num_horizons, _num_classes)
                    target_long = target.long()  # (batch, num_horizons)

                    total_loss = 0.0
                    for h in range(_num_horizons):
                        logits_h = logits_3d[:, h, :]  # (batch, C)
                        target_h = target_long[:, h]    # (batch,)

                        # 三分类全量计算(含涨/跌/平), 权重由 per_horizon_weights 控制
                        ce = F.cross_entropy(
                            logits_h, target_h,
                            weight=self.weights[h].to(target.device),
                            reduction='none'
                        )
                        pt = torch.exp(-ce)
                        focal_weight = (1 - pt) ** self.gamma
                        total_loss += (focal_weight * ce).mean()
                    return total_loss / max(_num_horizons, 1)

            # 类权重已在 DataLoader 构建前由 compute_class_weights 算出并打印,
            # 此处直接接入损失; 张量统一 .to(device) 避免跨设备拷贝开销
            criterion = FocalCrossEntropyLoss(
                gamma=config.FOCAL_GAMMA,
                per_horizon_weights=_per_horizon_weights,
            ).to(device)
            if fold_idx == 0:
                logger.info(f"使用 FocalCrossEntropyLoss(全三分类涨/跌/平, gamma={config.FOCAL_GAMMA})")
        else:
            # 二分类模式(旧逻辑): FocalLoss + BCEWithLogits
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
                logger.info(f"使用 FocalLoss(alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA}), "
                             f"各窗口pos_weight={_pos_weights}")

        # ---- CV 各折训练循环 ----
        best_val_loss = float('inf')
        patience_counter = 0
        # 续训时从上次断点的 epoch/global_step 继续累计(全新训练为0)
        epoch = _start_epoch
        global_step = _start_step
        _sched_exhausted = False  # OneCycleLR 步数用尽标记, 用尽后保持末端LR
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
            _last_grad_norm = 0.0  # 兜底: 无batch时取0
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

                if _num_classes >= 3:
                    # 三分类: outputs (batch, H*C) → reshape → argmax
                    logits_3d = outputs.view(batch_size, _num_horizons, _num_classes)
                    preds = logits_3d.argmax(dim=-1)  # (batch, H)
                    labels_long = labels.long()
                    for h in range(_num_horizons):
                        # 全口径三分类准确率 (含涨/跌/平)
                        n_correct = (preds[:, h] == labels_long[:, h]).sum().item()
                        n_total = labels_long[:, h].size(0)
                        _train_horizon_correct[h] += n_correct
                        _train_horizon_total[h] += n_total
                    train_correct = sum(_train_horizon_correct[h] for h in range(_num_horizons))
                    train_total = sum(_train_horizon_total[h] for h in range(_num_horizons))
                else:
                    # 二分类: 原始逻辑
                    preds = (outputs > 0).float()
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.numel()
                    for h in range(_num_horizons):
                        _train_horizon_correct[h] += (preds[:, h] == labels[:, h]).sum().item()
                        _train_horizon_total[h] += labels[:, h].size(0)

                # 每 _accum_steps 个 micro-batch 执行一次 optimizer 步
                if (batch_idx + 1) % _accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # 在 zero_grad 之前采样梯度范数, 取最后一步的 grad_norm
                    _last_grad_norm = 0.0
                    for _p in model.parameters():
                        if _p.grad is not None:
                            _last_grad_norm += _p.grad.norm().item() ** 2
                    _last_grad_norm = _last_grad_norm ** 0.5
                    optimizer.step()
                    global_step += 1
                    # OneCycleLR 走完 total_steps 后 step() 会抛 ValueError,
                    # 续训场景需保持末端LR继续训练而非崩溃
                    if not _sched_exhausted:
                        try:
                            scheduler.step()
                        except ValueError as _sched_err:
                            _sched_exhausted = True
                            logger.warning(
                                f"LR调度已走完全部步数(total_steps="
                                f"{getattr(scheduler, 'total_steps', '?')}), 后续保持当前LR: {_sched_err}"
                            )
                    optimizer.zero_grad()

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)
            _train_acc_per_h = [_train_horizon_correct[h] / max(_train_horizon_total[h], 1) for h in range(_num_horizons)]

            # 梯度诊断: 使用最后一次 optimizer step 前的 grad_norm
            _grad_info = f"grad_norm={_last_grad_norm:.4e}"

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

                    if _num_classes >= 3:
                        logits_3d = outputs.view(batch_size, _num_horizons, _num_classes)
                        preds = logits_3d.argmax(dim=-1)
                        labels_long = labels.long()
                        _batch_correct = 0
                        _batch_total = 0
                        for h in range(_num_horizons):
                            # 全口径三分类准确率 (含涨/跌/平)
                            n_correct = (preds[:, h] == labels_long[:, h]).sum().item()
                            n_total = labels_long[:, h].size(0)
                            _val_horizon_correct[h] += n_correct
                            _val_horizon_total[h] += n_total
                            _batch_correct += n_correct
                            _batch_total += n_total
                        val_correct += _batch_correct
                        val_total += _batch_total
                    else:
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
                    'completed_epochs': epoch,   # 已完成的epoch数(续训起点, 无off-by-one)
                    'global_step': global_step,  # 已完成的optimizer步数(与scheduler对齐)
                    'fold_idx': fold_idx,
                    'use_cv': _use_cv,
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
                        'num_classes_per_head': _num_classes,
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
    all_logits, all_labels = [], []
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

            if _num_classes >= 3:
                logits_3d = outputs.view(batch_size, _num_horizons, _num_classes)
                preds = logits_3d.argmax(dim=-1)
                labels_long = labels.long()
                for h in range(_num_horizons):
                    # 全口径三分类准确率 (含涨/跌/平)
                    n_correct = (preds[:, h] == labels_long[:, h]).sum().item()
                    n_total = labels_long[:, h].size(0)
                    _test_h_correct[h] += n_correct
                    _test_h_total[h] += n_total
                    test_correct += n_correct
                    test_total += n_total
            else:
                preds = (outputs > 0).float()
                test_correct += (preds == labels).sum().item()
                test_total += labels.numel()
                for h in range(_num_horizons):
                    _test_h_correct[h] += (preds[:, h] == labels[:, h]).sum().item()
                    _test_h_total[h] += labels[:, h].size(0)
            all_logits.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= test_total
    test_acc = test_correct / test_total

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)

    # 整体指标
    if _num_classes >= 3:
        # 三分类: 全量统计 (涨/平/跌)
        all_preds_flat = all_logits.reshape(-1, _num_horizons, _num_classes).argmax(axis=-1)
        all_labels_flat = all_labels.astype(int)

        # 涨跌方向性统计 (非中性样本)
        non_neutral = (all_labels_flat != 1)
        tp = int(((all_preds_flat == 2) & (all_labels_flat == 2)).sum())
        fp = int(((all_preds_flat == 2) & (all_labels_flat == 0)).sum())
        tn = int(((all_preds_flat == 0) & (all_labels_flat == 0)).sum())
        fn = int(((all_preds_flat == 0) & (all_labels_flat == 2)).sum())
        _n_directional_correct = tp + tn
        _n_directional_total = int(non_neutral.sum())

        # 中性样本统计
        _n_neutral_true = int((all_labels_flat == 1).sum())
        _n_neutral_correct = int(((all_preds_flat == 1) & (all_labels_flat == 1)).sum())
        _n_neutral_pred = int((all_preds_flat == 1).sum())
        _neutral_recall = _n_neutral_correct / _n_neutral_true if _n_neutral_true > 0 else 0
        _directional_acc = _n_directional_correct / _n_directional_total if _n_directional_total > 0 else 0

        logger.info(f"中性样本: 真实={_n_neutral_true} 预测={_n_neutral_pred} "
                     f"命中={_n_neutral_correct} Recall={_neutral_recall:.4f}")
    else:
        # 二分类: 原始逻辑
        tp = int(((all_logits > 0.5) & (all_labels == 1)).sum())
        fp = int(((all_logits > 0.5) & (all_labels == 0)).sum())
        tn = int(((all_logits <= 0.5) & (all_labels == 0)).sum())
        fn = int(((all_logits <= 0.5) & (all_labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"测试集Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    if _num_classes >= 3:
        logger.info(f"  其中 方向Acc(涨/跌): {_directional_acc:.4f} | 中性Recall: {_neutral_recall:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"混淆矩阵: TP={tp} FP={fp} TN={tn} FN={fn}")
    # 各窗口独立指标
    for h_idx, h_name in enumerate(config.PREDICTION_HORIZONS):
        _h_acc = _test_h_correct[h_idx] / max(_test_h_total[h_idx], 1)
        if _num_classes >= 3:
            # 三分类: 从重塑后的logits获取预测
            all_preds_flat = all_logits.reshape(-1, _num_horizons, _num_classes).argmax(axis=-1)
            _h_pred = all_preds_flat[:, h_idx]
            _h_label = all_labels[:, h_idx].astype(int)
            _h_tp = int(((_h_pred == 2) & (_h_label == 2)).sum())
            _h_fp = int(((_h_pred == 2) & (_h_label == 0)).sum())
            _h_tn = int(((_h_pred == 0) & (_h_label == 0)).sum())
            _h_fn = int(((_h_pred == 0) & (_h_label == 2)).sum())
            _h_neutral_pred = int((_h_pred == 1).sum())
            _h_neutral_true = int((_h_label == 1).sum())
            _h_neutral_correct = int(((_h_pred == 1) & (_h_label == 1)).sum())
            _h_neutral_recall = _h_neutral_correct / _h_neutral_true if _h_neutral_true > 0 else 0
            _h_directional_correct = _h_tp + _h_tn
            _h_directional_total = int((_h_label != 1).sum())
            _h_directional_acc = _h_directional_correct / _h_directional_total if _h_directional_total > 0 else 0
            _neutral_info = (f" 方向Acc:{_h_directional_acc:.4f} "
                             f"中性Recall:{_h_neutral_recall:.4f} "
                             f"中性(真/预):{_h_neutral_true}/{_h_neutral_pred}")
        else:
            _h_pred = all_logits[:, h_idx]
            _h_label = all_labels[:, h_idx]
            _h_tp = int(((_h_pred > 0.5) & (_h_label == 1)).sum())
            _h_fp = int(((_h_pred > 0.5) & (_h_label == 0)).sum())
            _h_tn = int(((_h_pred <= 0.5) & (_h_label == 0)).sum())
            _h_fn = int(((_h_pred <= 0.5) & (_h_label == 1)).sum())
            _neutral_info = ""
        _h_prec = _h_tp / (_h_tp + _h_fp) if (_h_tp + _h_fp) > 0 else 0
        _h_rec = _h_tp / (_h_tp + _h_fn) if (_h_tp + _h_fn) > 0 else 0
        logger.info(f"  [{h_name}min窗口] Acc:{_h_acc:.4f} Prec:{_h_prec:.4f} Rec:{_h_rec:.4f} "
                     f"TP={_h_tp} FP={_h_fp} TN={_h_tn} FN={_h_fn}{_neutral_info}")
    logger.info(f"最佳模型来自Epoch {_best_ckpt['epoch']} (Fold {_best_ckpt.get('fold_idx', 0)+1})")

    # ==================== 保存最佳模型用于断点续训/Release ====================
    # 关键: 必须连同 optimizer/scheduler/进度 一起落盘, 否则下一轮run会
    # 新建optimizer并重启OneCycleLR warmup, 跨run训练无法接成连续轨迹。
    # 这里取 _best_ckpt 的优化器/调度器状态(与所保存权重来自同一时刻),
    # 保证权重与LR轨迹严格对应; 缺失时回退到当前实时状态。
    _resume_opt_state = _best_ckpt.get('optimizer_state_dict') or optimizer.state_dict()
    _resume_sch_state = _best_ckpt.get('scheduler_state_dict') or scheduler.state_dict()
    _resume_completed_epochs = _best_ckpt.get('completed_epochs')
    if _resume_completed_epochs is None:
        _resume_completed_epochs = max(0, int(_best_ckpt.get('epoch', 1)) - 1)
    _resume_global_step = _best_ckpt.get('global_step')
    if _resume_global_step is None:
        # 旧checkpoint无step计数: 用 scheduler 的已走步数兜底(等价进度信息)
        _resume_global_step = int(getattr(scheduler, 'last_epoch', 0) or 0)

    _resume_meta = {
        'epoch': _best_ckpt['epoch'],
        'completed_epochs': int(_resume_completed_epochs),
        'global_step': int(_resume_global_step),
        'fold_idx': _best_ckpt.get('fold_idx', 0),
        'use_cv': _use_cv,
        'n_folds': _n_folds,
        'optimizer_state_dict': _resume_opt_state,
        'scheduler_state_dict': _resume_sch_state,
        'last_epoch_in_run': epoch,          # 本次run最后一个epoch编号(诊断用)
        'last_global_step_in_run': global_step,
    }

    torch.save({
        **_resume_meta,
        'model_state_dict': _best_ckpt['model_state_dict'],
        'val_loss': _best_ckpt['val_loss'],
        'val_acc': _best_ckpt['val_acc'],
        'test_acc': test_acc,
        'test_f1': f1,
        'config': _best_ckpt.get('config', {}),
    }, config.MODEL_PATH)
    torch.save({
        **_resume_meta,
        'model_state_dict': model.state_dict(),
        'val_loss': _best_ckpt['val_loss'],
        'val_acc': _best_ckpt['val_acc'],
        'config': _best_ckpt.get('config', {}),
    }, config.MODEL_PATH_FINAL)
    logger.info(f"最佳模型已保存: {config.MODEL_PATH}")
    logger.info(f"最终模型已保存: {config.MODEL_PATH_FINAL}")
    logger.info(
        f"续训状态已写入(下一轮run可接续): completed_epochs={_resume_meta['completed_epochs']}, "
        f"global_step={_resume_meta['global_step']}, use_cv={_use_cv}"
    )

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
