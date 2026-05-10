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

def verify_single(pred, stats=None):
    """验证单条预测记录

    Args:
        pred: dict, 包含 prediction_time, price, horizon, direction, probability 等字段
        stats: dict 或 None, 当前累积统计(用于通知), 不传入则不发送统计

    Returns:
        dict 或 None: 验证结果, None 表示跳过(数据不足等)
    """
    horizon = pred['horizon']
    pred_price = pred['price']
    pred_direction = pred['direction']
    pred_prob = pred['probability']
    h = horizon

    logger.info(f"开始验证 [{h}min] 预测: 时间={pred['prediction_time']}, 价格={pred_price}")

    fetcher = HuobiDataFetcher()

    try:
        # 获取最新的5min K线数据
        verify_data = fetcher.fetch_multi_timeframe(force_refresh=True)
        verify_df = fetcher.get_dataframe(verify_data['5min'])

        if verify_df.empty or len(verify_df) < 2:
            logger.warning(f"验证阶段({h}m): 数据不足, 跳过")
            return None

        # 计算验证价格: 预测时的价格 vs 当前最新收盘价
        verify_price = verify_df['close'].iloc[-1]
        verify_time = pd.Timestamp.now()

        actual_direction = "涨 (UP)" if verify_price > pred_price else "跌 (DOWN)"
        is_correct = (pred_prob > 0.5) == (verify_price > pred_price)
        price_change = (verify_price - float(pred_price)) / float(pred_price) * 100

        result_mark = "正确" if is_correct else "错误"
        print("-" * 50)
        print(f"  [{h}分钟窗口] 预测验证结果 [{result_mark}]")
        print("-" * 50)
        print(f"  预测时间:   {pred['prediction_time']} | 价格: {pred_price:.2f}")
        print(f"  验证时间:   {verify_time} | 价格: {verify_price:.2f}")
        print(f"  预测方向:   {pred_direction}")
        print(f"  实际方向:   {actual_direction}")
        print(f"  价格变化:   {price_change:+.2f}%")
        print("=" * 50)
        print()

        logger.info(
            f"[{h}m验证] {'正确' if is_correct else '错误'}, "
            f"预测{pred_direction}, 实际{actual_direction}, 变化{price_change:+.2f}%"
        )

        # 发送验证通知（带累积统计）
        if config.MEOW_NICKNAME:
            try:
                notifier = MeoWNotifier(config.MEOW_NICKNAME)
                notifier.send_prediction_verify(
                    direction=pred_direction,
                    actual_direction=actual_direction,
                    is_correct=is_correct,
                    current_price=float(pred_price),
                    verify_price=float(verify_price),
                    price_change_pct=price_change,
                    horizon=h,
                    stats=stats,  # 传入累积统计
                )
            except Exception as e:
                logger.warning(f"{h}m验证通知推送失败: {e}")

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

def verify_all():
    """处理所有到期的待验证预测

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

    for pred in pending:
        verify_after_ts = pred.get('verify_after_ts', pred.get('prediction_ts', 0) + pred['horizon'] * 60)

        if now_ts < verify_after_ts:
            # 还没到验证时间，保留
            still_pending.append(pred)
            continue

        # 到验证时间了，执行验证（传入 stats 用于通知显示）
        result = verify_single(pred, stats=stats)
        if result is not None:
            verified_count += 1
            # 更新累积统计
            update_stats(stats, pred['horizon'], result['is_correct'])
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
    verify_all()
