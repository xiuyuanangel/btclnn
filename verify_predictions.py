"""待验证预测处理脚本 — 在 Actions 运行时独立执行

功能:
  加载 predict.py 保存的待验证预测记录,
  对已到验证时间的预测执行验证, 发送验证通知,
  并清理已验证/已过期的记录。
  累积真实验证统计, 随通知发送历史正确率。

与 predict.py 配合使用:
  1. predict.py 仅做预测并保存到 pending_verifications.json
  2. 在下次 Actions 运行时, 先执行 verify_predictions.py,
     处理所有到期的验证, 再执行 predict.py 做新预测
"""

import json
import os
import logging

import pandas as pd

import config
from data_fetcher import HuobiDataFetcher
from notifier import MeoWNotifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ==============================
# 待验证记录 IO
# ==============================

def load_pending():
    """加载待验证的预测记录列表"""
    if not os.path.exists(config.PENDING_VERIFICATIONS_PATH):
        logger.info("没有待验证的预测记录")
        return []
    try:
        with open(config.PENDING_VERIFICATIONS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 去重: 同一预测时间+同一窗口只保留第一条
        seen = set()
        deduped = []
        for rec in data:
            key = (rec.get('prediction_ts', 0), rec['horizon'])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)

        if len(deduped) < len(data):
            logger.warning(f"检测到 {len(data) - len(deduped)} 条重复记录, 已自动去重")
            # 去重后直接写回文件, 避免下次再次处理重复
            with open(config.PENDING_VERIFICATIONS_PATH, 'w', encoding='utf-8') as f:
                json.dump(deduped, f, ensure_ascii=False, indent=2)

        logger.info(f"已加载 {len(deduped)} 条待验证预测记录")
        return deduped
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"加载待验证记录失败: {e}")
        return []


def save_pending(pending_list):
    """保存待验证记录(覆盖写入)"""
    if pending_list:
        os.makedirs(os.path.dirname(config.PENDING_VERIFICATIONS_PATH), exist_ok=True)
        with open(config.PENDING_VERIFICATIONS_PATH, 'w', encoding='utf-8') as f:
            json.dump(pending_list, f, ensure_ascii=False, indent=2)
        logger.info(f"保存 {len(pending_list)} 条待验证记录")
    else:
        # 无待验证记录, 删除文件
        if os.path.exists(config.PENDING_VERIFICATIONS_PATH):
            os.remove(config.PENDING_VERIFICATIONS_PATH)
            logger.info("所有待验证记录已处理, 删除文件")


# ==============================
# 验证统计 IO
# ==============================

_DEFAULT_STATS = {
    'overall_accuracy': 0.0,
    'total_verified': 0,
    'by_horizon': {},
    'last_updated': '',
}


def load_stats():
    """加载历史验证统计

    Returns:
        dict: 包含 overall_accuracy, total_verified, by_horizon, last_updated
    """
    if not os.path.exists(config.VERIFICATION_STATS_PATH):
        logger.info("没有历史验证统计, 从零开始")
        return dict(_DEFAULT_STATS)
    try:
        with open(config.VERIFICATION_STATS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 补齐缺失字段
        for k, v in _DEFAULT_STATS.items():
            data.setdefault(k, v)
        logger.info(f"已加载验证统计: {data['total_verified']} 次验证, "
                     f"正确率 {data['overall_accuracy']*100:.1f}%")
        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"加载验证统计失败: {e}")
        return dict(_DEFAULT_STATS)


def save_stats(stats):
    """保存验证统计到文件"""
    if stats['total_verified'] > 0:
        os.makedirs(os.path.dirname(config.VERIFICATION_STATS_PATH), exist_ok=True)
        # 只保留小数精度, 避免无限浮点
        stats['overall_accuracy'] = round(stats['overall_accuracy'], 6)
        for h_key in stats['by_horizon']:
            s = stats['by_horizon'][h_key]
            s['accuracy'] = round(s['accuracy'], 6)
        with open(config.VERIFICATION_STATS_PATH, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"验证统计已保存: {stats['total_verified']} 次, "
                     f"正确率 {stats['overall_accuracy']*100:.1f}%")
    else:
        # 无数据时删除文件
        if os.path.exists(config.VERIFICATION_STATS_PATH):
            os.remove(config.VERIFICATION_STATS_PATH)
            logger.info("验证统计清零, 删除统计文件")


def update_stats(stats, horizon, is_correct):
    """用新的验证结果更新累积统计

    Args:
        stats: 当前统计 dict (原地修改)
        horizon: 预测窗口(分钟), int
        is_correct: 本次验证是否正确

    Returns:
        dict: 更新后的 stats (与入参同一对象)
    """
    h_key = str(horizon)
    if h_key not in stats['by_horizon']:
        stats['by_horizon'][h_key] = {'correct': 0, 'total': 0, 'accuracy': 0.0}

    h_stat = stats['by_horizon'][h_key]
    h_stat['total'] += 1
    if is_correct:
        h_stat['correct'] += 1
    h_stat['accuracy'] = h_stat['correct'] / max(h_stat['total'], 1)

    stats['total_verified'] += 1
    # 整体正确率 = 所有窗口正确数之和 / 所有窗口总次数之和
    total_correct = sum(s['correct'] for s in stats['by_horizon'].values())
    total_all = sum(s['total'] for s in stats['by_horizon'].values())
    stats['overall_accuracy'] = total_correct / max(total_all, 1)
    stats['last_updated'] = str(pd.Timestamp.now())

    return stats


# ==============================
# 单条验证
# ==============================

def _fetch_verify_price(fetcher, verify_target_ts, pred_direction):
    """高精度验证价格获取 — 模拟真实交易执行的不确定性

    核心优化:
    1. 使用1min K线替代5min, 时间精度从±2.5min提升至±30s
    2. 执行窗口内取「不利方向最差价格」, 模拟真实成交无法精准择时
       - 预测涨→交易者需买入, 窗口内最高价是最差买入价(买贵了)
       - 预测跌→交易者需卖出, 窗口内最低价是最差卖出价(卖便宜了)
    3. 可选的额外滑点惩罚(VERIFY_SLIPPAGE_BPS)

    Returns:
        (verify_price, verify_time, actual_price_before_slippage)
    """
    use_1min = getattr(config, 'VERIFY_USE_1MIN', True)
    exec_window = max(1, getattr(config, 'VERIFY_EXECUTION_WINDOW', 2))
    slippage_bps = getattr(config, 'VERIFY_SLIPPAGE_BPS', 0.0)

    period = '1min' if use_1min else '5min'
    minutes_per = 1 if use_1min else 5

    # 获取验证需要的K线: 目标时间前后各exec_window根 + buffer
    lookback_minutes = (exec_window + 3) * minutes_per
    verify_data = fetcher.fetch_history(period, days=max(1, lookback_minutes // (24 * 60) + 1),
                                         force_refresh=True)
    verify_df = fetcher.get_dataframe(verify_data)

    if verify_df.empty or len(verify_df) < 2:
        raise ValueError(f"验证数据不足 ({period})")

    verify_target_time = pd.to_datetime(verify_target_ts, unit='s')

    # 找到覆盖目标时间的那根K线及其前后exec_window根
    # K线index是开盘时间, close是该周期结束时的价格
    candle_positions = verify_df.index  # 所有K线开盘时间
    # 找到 ≤ target_time 的最近K线索引
    idx = candle_positions.searchsorted(verify_target_time, side='right') - 1
    idx = max(0, idx)

    # 取执行窗口内的K线: [idx - exec_window, idx + exec_window]
    win_start = max(0, idx - exec_window)
    win_end = min(len(verify_df) - 1, idx + exec_window)
    window_df = verify_df.iloc[win_start:win_end + 1]

    if window_df.empty:
        # 回退到单根K线
        verify_candle = verify_df.iloc[idx]
        raw_price = float(verify_candle['close'])
        verify_time = verify_target_time
    else:
        # 根据预测方向取窗口内最不利价格 (模拟真实成交)
        if pred_direction == "涨 (UP)":
            # 预测涨: 交易者买入, 窗口内最高价是最差买入价
            worst_idx = window_df['high'].idxmax()
            raw_price = float(window_df.loc[worst_idx, 'high'])
        elif pred_direction == "跌 (DOWN)":
            # 预测跌: 交易者卖出, 窗口内最低价是最差卖出价
            worst_idx = window_df['low'].idxmin()
            raw_price = float(window_df.loc[worst_idx, 'low'])
        else:
            # 平: 使用目标K线收盘价
            raw_price = float(verify_df.iloc[idx]['close'])

        # 验证时间用窗口中心K线时间
        verify_time = verify_df.index[idx]

    # 应用额外滑点惩罚
    if slippage_bps > 0:
        slippage_factor = slippage_bps / 10000.0  # bps → 小数
        if pred_direction == "涨 (UP)":
            verify_price = raw_price * (1.0 + slippage_factor)  # 买入更贵
        elif pred_direction == "跌 (DOWN)":
            verify_price = raw_price * (1.0 - slippage_factor)  # 卖出更便宜
        else:
            verify_price = raw_price
    else:
        verify_price = raw_price

    return verify_price, verify_time, raw_price


def verify_single(pred):
    """验证单条预测记录 — 高精度执行窗口版

    Args:
        pred: dict, 包含 prediction_time, price, horizon, direction, probability 等字段

    Returns:
        dict 或 None: 验证结果, None 表示跳过(数据不足等)
    """
    horizon = pred['horizon']
    pred_price = pred['price']
    pred_direction = pred['direction']
    h = horizon

    logger.info(f"开始验证 [{h}min] 预测: 时间={pred['prediction_time']}, 价格={pred_price}")

    fetcher = HuobiDataFetcher()

    try:
        # 计算目标验证时间 = 预测时间 + 预测窗口(分钟)
        prediction_ts = pred.get('prediction_ts', 0)
        verify_target_ts = prediction_ts + h * 60

        # 高精度获取验证价格 (1min K线 + 执行窗口)
        verify_price, verify_time, raw_price = _fetch_verify_price(
            fetcher, verify_target_ts, pred_direction
        )

        # 价格变化
        price_change = (verify_price - float(pred_price)) / float(pred_price) * 100

        # 使用与标签生成一致的中性门限 (百分比单位)
        neutral_threshold = config.LABEL_MIN_RETURN * 100

        if verify_price > pred_price + neutral_threshold:
            actual_direction = "涨 (UP)"
        elif verify_price < pred_price - neutral_threshold:
            actual_direction = "跌 (DOWN)"
        else:
            actual_direction = "平 (NEUTRAL)"

        # 基于方向字段验证 (兼容二分类和三分类预测)
        is_correct = (pred_direction == actual_direction)

        result_mark = "正确" if is_correct else "错误"
        use_1min = getattr(config, 'VERIFY_USE_1MIN', True)
        exec_win = getattr(config, 'VERIFY_EXECUTION_WINDOW', 2)
        print("-" * 50)
        print(f"  [{h}分钟窗口] 预测验证结果 [{result_mark}]")
        print("-" * 50)
        print(f"  验证精度:   {'1min K线' if use_1min else '5min K线'} | "
              f"执行窗口: ±{exec_win}根")
        print(f"  预测时间:   {pred['prediction_time']} | 价格: {pred_price:.2f}")
        print(f"  验证时间:   {verify_time} | 成交价: {verify_price:.2f}")
        if raw_price != verify_price:
            print(f"  (原始价: {raw_price:.2f}, 滑点扣减后: {verify_price:.2f})")
        print(f"  预测方向:   {pred_direction}")
        print(f"  实际方向:   {actual_direction}")
        print(f"  价格变化:   {price_change:+.2f}%")
        print("=" * 50)
        print()

        logger.info(
            f"[{h}m验证] {'正确' if is_correct else '错误'}, "
            f"预测{pred_direction}, 实际{actual_direction}, 变化{price_change:+.2f}%"
        )

        return {
            'verified': True,
            'is_correct': is_correct,
            'verify_time': str(verify_time),
            'verify_price': float(verify_price),
            'base_price': float(pred_price),
            'price_change_pct': price_change,
            'actual_direction': actual_direction,
        }

    except Exception as e:
        logger.warning(f"验证阶段({h}m)获取数据失败: {e}")
        print(f"  [{h}分钟窗口] 无法获取验证数据，跳过预测验证")
        return None


# ==============================
# 主入口
# ==============================

def verify_all(reverify_all=False):
    """处理所有到期的待验证预测

    Args:
        reverify_all: 是否强制重新验证所有记录(忽略 verify_after_ts 检查)

    返回:
        dict: {'verified': int, 'skipped': int, 'pending': int}
    """
    # 1. 加载待验证记录
    pending = load_pending()
    if not pending:
        return {'verified': 0, 'skipped': 0, 'pending': 0}

    # 2. 加载累积统计
    stats = load_stats()

    now_ts = int(pd.Timestamp.now().timestamp())

    still_pending = []
    verified_count = 0
    skipped_count = 0

    # 创建通知器(在循环外复用)
    notifier = MeoWNotifier(config.MEOW_NICKNAME) if config.MEOW_NICKNAME else None

    for pred in pending:
        verify_after_ts = pred.get('verify_after_ts', pred.get('prediction_ts', 0) + pred['horizon'] * 60)

        if not reverify_all and now_ts < verify_after_ts:
            # 还没到验证时间，保留
            still_pending.append(pred)
            continue

        # 到验证时间了，执行验证
        result = verify_single(pred)
        if result is not None:
            verified_count += 1
            # 先更新累积统计(后续通知将包含本次结果)
            update_stats(stats, pred['horizon'], result['is_correct'])

            if notifier:
                try:
                    notifier.send_prediction_verify(
                        direction=pred['direction'],
                        actual_direction=result['actual_direction'],
                        is_correct=result['is_correct'],
                        current_price=float(pred['price']),
                        verify_price=result['verify_price'],
                        price_change_pct=result['price_change_pct'],
                        horizon=pred['horizon'],
                        stats=stats,  # 已包含本次验证结果
                    )
                except Exception as e:
                    logger.warning(f"{pred['horizon']}m验证通知推送失败: {e}")
        else:
            skipped_count += 1
        # 无论验证成功与否，都从待验证队列移除(单次尝试)

    # 3. 保存剩余的待验证记录
    save_pending(still_pending)

    # 4. 保存更新后的统计
    save_stats(stats)

    stats_summary = {
        'verified': verified_count,
        'skipped': skipped_count,
        'pending': len(still_pending),
    }

    logger.info(
        f"验证完成: 已验证 {verified_count} 条, "
        f"跳过 {skipped_count} 条, "
        f"待验证 {len(still_pending)} 条, "
        f"累积正确率 {stats['overall_accuracy']*100:.1f}%"
    )
    return stats_summary


if __name__ == "__main__":
    import sys
    reverify = '--reverify' in sys.argv
    if reverify:
        logger.info("=" * 50)
        logger.info("强制重新验证模式: 将处理所有待验证记录")
        logger.info("=" * 50)
    verify_all(reverify_all=reverify)
