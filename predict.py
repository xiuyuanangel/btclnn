"""液态神经网络预测脚本 — 多周期多窗口融合版

使用训练好的多周期融合模型，基于最新K线数据预测多个时间窗口后的涨跌。
默认同时输出10分钟、30分钟和60分钟后的涨跌概率。
"""

import os
import time
import logging

import numpy as np
import pandas as pd
import torch

import config
from data_fetcher import HuobiDataFetcher
from notifier import MeoWNotifier
from features import (
    compute_all_features, compute_context_features,
    SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS,
)
from lnn_model import MultiTimeframeLNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_norm_stats():
    """加载训练时保存的特征标准化统计量"""
    import pickle
    path = os.path.join(config.CHECKPOINT_DIR, 'feature_norm_stats.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"标准化参数文件不存在: {path}\n"
            f"请先运行 train.py 训练模型以生成该文件"
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def normalize_with_stats(tf_seqs_raw, ctx_raw, norm_data):
    """使用训练时的统计量对原始特征进行Z-Score标准化

    Args:
        tf_seqs_raw: dict of {period: np.array (1, seq_len, feat_size)} 原始特征
        ctx_raw: np.array (1, ctx_size) 原始上下文
        norm_data: dict {'periods': list, 'stats': dict} 标准化参数

    Returns:
        标准化后的 tf_seqs, ctx (同shape)
    """
    stats = norm_data['stats']
    periods = norm_data['periods']

    tf_seqs = {}
    for p in periods:
        s = stats[p]
        tf_seqs[p] = (tf_seqs_raw[p] - s['mean']) / s['std']

    cs = stats['context']
    ctx = (ctx_raw - cs['mean']) / cs['std']

    return tf_seqs, ctx


def _rename_state_dict_keys(state_dict):
    """重命名checkpoint中的旧版key以兼容新版模型架构

    模型架构变更历史:
    - v1: cells, layer_norms (无前缀)
    - v2: ltc_cells, ltc_layer_norms (增加ltc_前缀区分)
    """
    key_mapping = {}
    for old_key in list(state_dict.keys()):
        new_key = old_key
        if '.cells.' in old_key and '.ltc_cells.' not in old_key:
            new_key = old_key.replace('.cells.', '.ltc_cells.')
        if '.layer_norms.' in old_key and '.ltc_layer_norms.' not in old_key:
            new_key = old_key.replace('.layer_norms.', '.ltc_layer_norms.')
        if new_key != old_key:
            key_mapping[old_key] = new_key

    if key_mapping:
        logger.info(f"检测到旧版模型key，转换 {len(key_mapping)} 个key")
        for old_key, new_key in key_mapping.items():
            state_dict[new_key] = state_dict.pop(old_key)

    return state_dict


def load_model(device):
    """加载训练好的多周期融合模型(多标签版)"""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"模型文件不存在: {config.MODEL_PATH}\n"
            f"请先运行 train.py 训练模型"
        )

    checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model_cfg = checkpoint['config']

    # 从checkpoint或config获取输出维度
    _output_size = model_cfg.get('output_size', len(getattr(config, 'PREDICTION_HORIZONS', [10])))
    _horizons = model_cfg.get('horizons', getattr(config, 'PREDICTION_HORIZONS', [10]))
    
    # 从checkpoint获取Transformer配置
    _use_transformer = model_cfg.get('use_transformer', False)
    _transformer_heads = model_cfg.get('transformer_heads', 4)
    _cross_attn_heads = model_cfg.get('cross_attn_heads', 4)

    model = MultiTimeframeLNN(
        timeframe_configs=model_cfg['timeframe_configs'],
        context_feature_size=model_cfg['context_feature_size'],
        hidden_size=model_cfg.get('hidden_size', config.HIDDEN_SIZE),
        num_layers=model_cfg.get('num_layers', config.NUM_LAYERS),
        dropout=model_cfg.get('dropout', config.DROPOUT),
        output_size=_output_size,
        use_transformer=_use_transformer,
        transformer_heads=_transformer_heads,
        cross_attn_heads=_cross_attn_heads,
    ).to(device)

    # 处理模型权重key不兼容问题(旧版checkpoint)
    state_dict = checkpoint['model_state_dict']
    state_dict = _rename_state_dict_keys(state_dict)
    
    # 处理DataParallel前缀问题
    model_key_prefix = ''
    if isinstance(model, torch.nn.DataParallel):
        model_key_prefix = 'module.'
    elif hasattr(model, 'module'):
        model_key_prefix = 'module.'

    ckpt_keys = list(state_dict.keys())
    has_ckpt_prefix = any(k.startswith('module.') for k in ckpt_keys)

    if model_key_prefix and not has_ckpt_prefix:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = f'module.{k}'
            new_state_dict[new_key] = v
        state_dict = new_state_dict
        logger.info(f"为checkpoint添加module.前缀以匹配DataParallel模型")
    elif has_ckpt_prefix and not model_key_prefix:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[len('module.'):] if k.startswith('module.') else k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
        logger.info(f"移除checkpoint的module.前缀以匹配非DataParallel模型")
    
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(
        f"模型加载成功 (Epoch {checkpoint['epoch']}, "
        f"Val Loss: {checkpoint['val_loss']:.4f}, "
        f"输出维度: {_output_size} 窗口={_horizons}, "
        f"Transformer: {'启用' if _use_transformer else '禁用'})"
    )
    return model, _horizons


def prepare_multi_tf_features(timeframe_data):
    """为最新时刻准备多周期模型输入特征

    Args:
        timeframe_data: dict of {period: list_of_kline_dicts}

    Returns:
        tf_seqs: dict of {period: np.array (1, seq_length, feature_size)}
        ctx: np.array (1, context_size)
        target_df: DataFrame (用于获取当前价格和时间)
    """
    fetcher = HuobiDataFetcher()
    periods = list(config.TIMEFRAMES.keys())

    # 各周期计算特征
    tf_dfs = {}
    for period in periods:
        df = fetcher.get_dataframe(timeframe_data[period])
        df = compute_all_features(df)
        df = df.dropna(subset=SEQ_FEATURE_COLS)
        tf_dfs[period] = df

    # 上下文特征(来自4hour)
    ctx_df = compute_context_features(tf_dfs['4hour'])
    ctx = ctx_df[CONTEXT_FEATURE_COLS].values[-1:].astype(np.float32)

    # 目标时间点: 当前时刻
    now_ts = int(pd.Timestamp.now().timestamp())
    target_ts = np.array([now_ts])

    # 各周期提取最新序列
    tf_seqs = {}
    for period in periods:
        seq_length = config.TIMEFRAMES[period]['seq_length']
        df = tf_dfs[period]
        tf_ts = df['id'].values.astype(np.int64)
        tf_feat = df[SEQ_FEATURE_COLS].values

        idx = np.searchsorted(tf_ts, now_ts, side='right') - 1
        if idx < seq_length - 1:
            raise ValueError(f"{period} 数据不足: 需要 {seq_length} 条, 可用 {idx + 1} 条")

        seq = tf_feat[idx - seq_length + 1:idx + 1]
        if np.isnan(seq).any():
            raise ValueError(f"{period} 序列包含 NaN")

        tf_seqs[period] = seq[np.newaxis, :, :].astype(np.float32)

    # 用5min数据获取当前价格(最新)
    target_df = tf_dfs['5min']

    return tf_seqs, ctx, target_df


def predict():
    """执行预测: 基于多周期数据判断多个时间窗口后的涨跌"""
    # 检测CUDA兼容性(同train.py)
    _use_cuda = False
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 7:
                _use_cuda = True
        except Exception:
            pass
    device = torch.device("cuda" if _use_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    if _use_cuda and torch.cuda.device_count() > 1:
        logger.info(f"检测到 {torch.cuda.device_count()} 块GPU")

    # 1. 加载模型(返回模型和窗口列表)
    model, horizons = load_model(device)
    num_horizons = len(horizons)

    # 2. 加载标准化参数(必须与训练时一致)
    norm_data = load_norm_stats()

    # 3. 获取多周期数据
    logger.info("正在获取多周期K线数据...")
    fetcher = HuobiDataFetcher()
    timeframe_data = fetcher.fetch_multi_timeframe()

    # 4. 特征准备(原始)
    try:
        tf_seqs_raw, ctx_raw, df_featured = prepare_multi_tf_features(timeframe_data)
    except ValueError as e:
        logger.error(str(e))
        return None

    # 5. 应用序列级别归一化 + 与训练一致的Z-Score标准化
    from features import normalize_sequence_samplewise
    tf_seqs_raw = normalize_sequence_samplewise(tf_seqs_raw)
    tf_seqs_norm, ctx_norm = normalize_with_stats(tf_seqs_raw, ctx_raw, norm_data)

    # 6. 模型推理
    tf_seqs_tensor = {p: torch.from_numpy(v.copy()).float().to(device) for p, v in tf_seqs_norm.items()}
    ctx_tensor = torch.from_numpy(ctx_norm.copy()).float().to(device)

    with torch.no_grad():
        logits = model(tf_seqs_tensor, ctx_tensor)  # (1, num_horizons), 输出为logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]  # (num_horizons,)

    # 7. 输出结果
    current_price = df_featured['close'].iloc[-1]
    prediction_time = pd.Timestamp.now()
    latest_kline_time = df_featured.index[-1]

    # 构建各窗口预测结果
    results = []
    print()
    print("=" * 60)
    print(f"  多周期融合 LNN 预测结果 (多窗口)")
    print("=" * 60)
    print(f"  融合周期:   {', '.join(config.TIMEFRAMES.keys())}")
    print(f"  预测时间:   {prediction_time}")
    print(f"  当前价格:   {current_price:.2f} USDT")
    print("-" * 60)

    for i, h in enumerate(horizons):
        prob = float(probabilities[i])
        direction = "涨 (UP)" if prob > 0.5 else "跌 (DOWN)"
        confidence = abs(prob - 0.5) * 2

        result = {
            'horizon': h,
            'direction': direction,
            'probability': prob,
            'confidence': confidence,
        }
        results.append(result)

        print(f"  [{h:>3}分钟] 方向: {direction:<12} "
              f"上涨概率: {prob:.4f} ({prob*100:.2f}%)  "
              f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")

    print("=" * 60)
    print()

    # 发送通知推送
    if config.MEOW_NICKNAME:
        try:
            notifier = MeoWNotifier(config.MEOW_NICKNAME)
            notifier.send_multi_horizon_prediction(
                time=str(prediction_time),
                price=float(current_price),
                horizons_results=results,
            )
        except Exception as e:
            logger.warning(f"通知推送失败: {e}")

    # 8. 分阶段验证各窗口预测
    verify_results = {}
    sorted_horizons = sorted(horizons)  # 按时间从小到大排序

    for h in sorted_horizons:
        # 找到对应的result
        h_result = next(r for r in results if r['horizon'] == h)
        h_prob = h_result['probability']
        h_direction = h_result['direction']

        wait_seconds = h * 60  # 等待h分钟

        logger.info(f"等待{h}分钟验证{h}min窗口的预测...")
        now = pd.Timestamp.now()

        # 对齐到下一个整h分钟边界
        # 例如h=10对齐到下一个10min边界, h=30对齐到下一个30min边界
        if h <= 30:
            align_floor = f'{h}min'
        elif h == 60:
            align_floor = '1h'
        else:
            align_floor = '1h'
        # 计算对齐后的等待边界
        next_boundary = now.floor(align_floor) + pd.Timedelta(minutes=h)
        wait = max((next_boundary - now).total_seconds(), 30)
        # 如果对齐后等待时间仍不足h分钟, 跳过不足的边界, 再等一个完整周期
        if wait < wait_seconds:
            next_boundary += pd.Timedelta(minutes=h)
            wait = max((next_boundary - now).total_seconds(), 30)
        logger.info(f"对齐到 {next_boundary}, 等待 {wait:.0f}s")
        time.sleep(wait)

        try:
            # 强制刷新缓存获取最新K线数据
            verify_data = fetcher.fetch_multi_timeframe(force_refresh=True)

            # 用5min聚合为对应粒度的数据进行验证
            if h % 5 == 0:
                shift_bars = h // 5
                verify_df = fetcher.get_dataframe(verify_data['5min'])
            elif h % 10 == 0:
                verify_10min = fetcher.resample_to_10min(verify_data['5min'])
                verify_df = fetcher.get_dataframe(verify_10min)
                shift_bars = h // 10
            else:
                verify_5min = fetcher.get_dataframe(verify_data['5min'])
                verify_df = verify_5min
                shift_bars = h // 5

            if verify_df.empty or len(verify_df) < shift_bars + 1:
                logger.warning(f"验证阶段({h}m): 数据不足, 跳过")
                verify_results[h] = {'verified': False, 'reason': 'data_insufficient'}
                continue

            # 用shift后的价格作为验证价格
            verify_price_idx = min(shift_bars, len(verify_df) - 1)
            verify_price = verify_df['close'].iloc[-1]
            # 取shift_bars前的当前价格进行对比
            base_price_for_verify = verify_df['close'].iloc[-(shift_bars + 1)] if len(verify_df) > shift_bars else current_price

            verify_time = pd.Timestamp.now()

            actual_direction = "涨 (UP)" if verify_price > base_price_for_verify else "跌 (DOWN)"
            is_correct = (h_prob > 0.5) == (verify_price > base_price_for_verify)
            price_change = (verify_price - float(base_price_for_verify)) / float(base_price_for_verify) * 100

            result_mark = "✅ 正确" if is_correct else "❌ 错误"
            print("-" * 50)
            print(f"  📊 [{h}分钟窗口] 预测验证结果 [{result_mark}]")
            print("-" * 50)
            print(f"  预测时间:   {prediction_time} | 价格: {current_price:.2f}")
            print(f"  验证时间:   {verify_time} | 价格: {verify_price:.2f}")
            print(f"  预测方向:   {h_direction}")
            print(f"  实际方向:   {actual_direction}")
            print(f"  价格变化:   {price_change:+.2f}%")
            print("=" * 50)
            print()

            logger.info(
                f"[{h}m验证] {'正确' if is_correct else '错误'}, "
                f"预测{h_direction}, 实际{actual_direction}, 变化{price_change:+.2f}%"
            )

            verify_results[h] = {
                'verified': True,
                'is_correct': is_correct,
                'verify_time': str(verify_time),
                'verify_price': float(verify_price),
                'base_price': float(base_price_for_verify),
                'price_change_pct': price_change,
                'actual_direction': actual_direction,
            }

            # 推送验证结果通知
            if config.MEOW_NICKNAME:
                try:
                    notifier = MeoWNotifier(config.MEOW_NICKNAME)
                    notifier.send_prediction_verify(
                        direction=h_direction,
                        actual_direction=actual_direction,
                        is_correct=is_correct,
                        current_price=float(current_price),
                        verify_price=float(verify_price),
                        price_change_pct=price_change,
                        horizon=h,
                    )
                except Exception as e:
                    logger.warning(f"{h}m验证通知推送失败: {e}")

        except Exception as e:
            logger.warning(f"验证阶段({h}m)获取数据失败: {e}")
            print(f"⚠️  [{h}分钟窗口] 无法获取验证数据，跳过预测验证")
            verify_results[h] = {'verified': False, 'reason': str(e)}

    # 9. 汇总返回所有结果
    final_result = {
        'time': str(prediction_time),
        'price': float(current_price),
        'horizons': [r['horizon'] for r in results],
        'predictions': results,
        'verifications': verify_results,
    }
    return final_result


if __name__ == "__main__":
    predict()
