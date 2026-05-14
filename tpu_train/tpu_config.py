"""TPU 训练专用配置 — 继承项目 config 并覆盖 TPU 特有参数

使用方法:
    export PJRT_DEVICE=TPU  # 在 TPU VM 上设置
    python tpu_train/tpu_train.py

    # Colab 上运行时，环境变量已自动设置

TPU 训练与 GPU 训练的关键差异:
    1. Batch size 必须是固定编译时常量 (禁用自动检测)
    2. 启用 bfloat16 混合精度 (TPU 原生加速)
    3. Batch size 建议为 128 的倍数 (per TPU core)
    4. 数据加载使用 MpDeviceLoader 替代标准 DataLoader
"""

import os
import sys

# 将项目根目录加入 sys.path，以便导入父级模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config as base_config

# =============================================================================
# TPU 核心覆盖参数
# =============================================================================

# --- Batch Size ---
# TPU 要求 batch size 在编译时固定，不能用 _auto_batch_size() 动态计算。
#
# 当前 Colab 提供 TPU v5e-1 (单核心, 16GB HBM)
#   单核心下 batch 直接设为核心 batch 大小 (MpDeviceLoader 不会分片)。
#
# TPU 类型参考:
#   v2-8:  8 core × 8GB  → 全局 512~1024 (每 core 64~128)
#   v3-8:  8 core × 16GB → 全局 1024~2048 (每 core 128~256)
#   v5e-1: 1 core × 16GB → 全局 128~256  (单核)
#   v5e-4: 4 core × 16GB → 全局 512~1024 (每 core 128~256)
#   v5e-8: 8 core × 16GB → 全局 1024~2048 (每 core 128~256)
#
# Colab 环境变量 'COLAB_TPU_ACCELERATOR' 包含 TPU 类型，如 'v5e-1'
_DETECTED_ACCEL = os.environ.get('COLAB_TPU_ACCELERATOR', '').lower()
if 'v5e' in _DETECTED_ACCEL:
    # TPU v5e 系列: 16GB per core，以 128 per core 为基准，解析核心数
    try:
        _n_cores = int(_DETECTED_ACCEL.split('-')[-1])
    except (ValueError, IndexError):
        _n_cores = 1
    BATCH_SIZE = 128 * _n_cores
elif 'v2' in _DETECTED_ACCEL:
    BATCH_SIZE = 512       # v2-8: 8GB/core
elif 'v3' in _DETECTED_ACCEL:
    BATCH_SIZE = 1024      # v3-8: 16GB/core
else:
    # 回退检测
    _hw = os.environ.get('TPU_VISIBLE_DEVICES', '')
    if 'v5e' in _hw:
        BATCH_SIZE = 128
    elif 'v3' in _hw:
        BATCH_SIZE = 1024
    elif 'v2' in _hw:
        BATCH_SIZE = 512
    else:
        BATCH_SIZE = 128   # 默认安全值 (适配 v5e-1 单核 16GB)
USE_AUTO_BATCH_SIZE = False  # 必须关闭

# --- 混合精度 ---
# TPU 原生使用 bfloat16 计算效率最高。启用后，前向/反向自动使用 bf16。
# 权重更新仍以 float32 精度执行（由 xm.amp.autocast 自动管理）
USE_BF16 = True

# --- 梯度累积 ---
# 由于 TPU batch size 可能受限于显存，可用梯度累积模拟更大的 batch。
# 等效 batch = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS
# v5e-1 BATCH_SIZE=128 可用 accum=2 等效 256
# v3-8  BATCH_SIZE=1024 可直接设 1
GRADIENT_ACCUMULATION_STEPS = 2  # v5e-1: 128×2=256 等效 batch

# --- 学习率 ---
# bfloat16 精度低于 float32，LR 通常需要略微降低以避免溢出
LEARNING_RATE = 3e-4  # 原 5e-4，TPU 下适度降低

# --- 训练停止条件 ---
# TPU 按时间计费，建议用 time_only 控制成本
TRAIN_STOP_MODE = 'time_only'
MAX_TRAIN_SECONDS = 10800  # 3 小时 (原 18000=5h)

# --- 早停 ---
PATIENCE = 80  # TPU 训练更快收敛，减少 patience

# =============================================================================
# 导出接口：生成最终的训练参数字典（合并 base + override）
# =============================================================================

def get_training_config():
    """生成 TPU 训练参数，合并 base_config 的原始值 + 本文件的覆盖值

    Returns:
        dict: 所有训练参数的字典，可用作 **kwargs 传递给 MultiTimeframeLNN
    """
    config = {}

    # 从 base_config 复制标准参数
    standard_params = [
        'HIDDEN_SIZE', 'NUM_LAYERS', 'DROPOUT', 'TIMEFRAMES',
        'PREDICTION_HORIZONS', 'FOCAL_ALPHA', 'FOCAL_GAMMA',
        'SYMBOLS', 'CONTRACT_TYPE',
        'USE_TRANSFORMER', 'TRANSFORMER_HEADS', 'CROSS_ATTN_HEADS',
        'WEIGHT_DECAY', 'BASE_BATCH_SIZE',
        'ONECYCLE_MAX_LR_SCALE', 'ONECYCLE_PCT_START',
        'ONECYCLE_ANNEAL_STRATEGY', 'ONECYCLE_FINAL_DIV_FACTOR',
        'LABEL_SOURCE_PERIOD', 'LABEL_SMOOTH_WINDOW',
        'LABEL_MIN_RETURN', 'LABEL_DROP_NEUTRAL',
        'MEOW_NICKNAME', 'DEBUG_EXPORT_CSV',
        'TRAIN_RATIO', 'VAL_RATIO',
        'CHECKPOINT_DIR', 'DATA_DIR', 'MODEL_PATH', 'MODEL_PATH_FINAL',
    ]
    for attr in standard_params:
        if hasattr(base_config, attr):
            config[attr] = getattr(base_config, attr)

    # TPU 覆盖值 (优先级高于 base_config)
    overrides = {
        'BATCH_SIZE': BATCH_SIZE,
        'USE_AUTO_BATCH_SIZE': USE_AUTO_BATCH_SIZE,
        'USE_BF16': USE_BF16,
        'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
        'TRAIN_STOP_MODE': TRAIN_STOP_MODE,
        'MAX_TRAIN_SECONDS': MAX_TRAIN_SECONDS,
        'PATIENCE': PATIENCE,
        'LEARNING_RATE': LEARNING_RATE,
    }
    config.update(overrides)

    return config
