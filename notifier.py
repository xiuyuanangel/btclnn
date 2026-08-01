"""MeoW消息推送通知模块

基于 https://www.chuckfang.com/MeoW/api_doc.html 文档实现
接口地址: http://api.chuckfang.com/ 或 https://api.chuckfang.com/
"""

import json
import logging
import urllib.parse
import urllib.request
from typing import Optional, Sequence, Union

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================
# 统一预测信号 (predict / notifier / verify 三处共用口径)
# ==============================
# 背景: 历史实现中 predict.py 用 argmax 定方向、notifier.py 用「上涨概率阈值」定文案,
#       两套口径在低置信度时必然打架(例: 涨概率0.19是三类最大 → 屏幕打印「涨」,
#       同时 0.19 < 0.3 → 推送喊「强烈看跌」)。
# 现统一为: 方向 = argmax(概率最大的类); 强度 = 置信度(最大类概率)。

# 模型三分类输出的类索引顺序: 0=跌, 1=平, 2=涨 (与 features.compute_labels 一致)
CLASS_IDX_DOWN = 0
CLASS_IDX_FLAT = 1
CLASS_IDX_UP = 2

# 类索引 → 内部键名
CLASS_KEYS = {CLASS_IDX_DOWN: 'down', CLASS_IDX_FLAT: 'flat', CLASS_IDX_UP: 'up'}
# 内部键名 → 类索引
CLASS_INDICES = {v: k for k, v in CLASS_KEYS.items()}
# 内部键名 → 中文展示串 (与 verify_predictions.py 的 actual_direction 字面量保持一致)
DIRECTION_LABELS = {
    'up': "涨 (UP)",
    'down': "跌 (DOWN)",
    'flat': "平 (NEUTRAL)",
}


def build_prediction_signal(probs: Union[Sequence[float], dict],
                            conf_thresh: Optional[float] = None) -> dict:
    """构建统一预测信号 — 全项目唯一的「方向 + 强度」判定入口

    方向一律取 argmax, 强度一律取置信度(最大类概率), 两者来自同一份概率分布,
    因此 predict.py 的打印与 notifier.py 的推送文案在方向上永远一致。

    Args:
        probs: 概率分布, 支持两种形式:
            - 序列 (down, flat, up), 即模型三分类输出顺序
            - 字典 {'down': x, 'flat': y, 'up': z}
        conf_thresh: 「信号弱」判定阈值, 默认取 config.CONF_THRESH

    Returns:
        dict: 统一信号, 字段:
            - pred_class:     'up' | 'down' | 'flat'  (argmax 结果)
            - pred_class_idx: 0=跌 / 1=平 / 2=涨
            - direction:      中文方向串, 如 '涨 (UP)'
            - probs:          {'up': float, 'down': float, 'flat': float} 完整分布
            - confidence:     最大类概率
            - margin:         |P(涨) - P(跌)|, 方向性强弱的辅助指标
            - is_weak:        置信度是否低于阈值(低置信 → 文案标注中性观望)
            - conf_thresh:    实际生效的阈值
    """
    if conf_thresh is None:
        conf_thresh = getattr(config, 'CONF_THRESH', 0.4)

    if isinstance(probs, dict):
        p_down = float(probs.get('down', 0.0))
        p_flat = float(probs.get('flat', 0.0))
        p_up = float(probs.get('up', 0.0))
    else:
        seq = [float(x) for x in probs]
        if len(seq) == 3:
            p_down, p_flat, p_up = seq
        elif len(seq) == 2:
            # 二分类 (down, up), 无中性类
            p_down, p_up = seq
            p_flat = 0.0
        else:
            raise ValueError(f"probs 长度必须为 2 或 3, 收到 {len(seq)}")

    prob_map = {'down': p_down, 'flat': p_flat, 'up': p_up}
    ordered = [p_down, p_flat, p_up]
    pred_class_idx = int(max(range(3), key=lambda i: ordered[i]))
    pred_class = CLASS_KEYS[pred_class_idx]
    confidence = float(ordered[pred_class_idx])

    return {
        'pred_class': pred_class,
        'pred_class_idx': pred_class_idx,
        'direction': DIRECTION_LABELS[pred_class],
        'probs': prob_map,
        'confidence': confidence,
        'margin': abs(p_up - p_down),
        'is_weak': bool(confidence < conf_thresh),
        'conf_thresh': float(conf_thresh),
    }


def build_signal_from_prob_up(prob_up: float,
                              conf_thresh: Optional[float] = None) -> dict:
    """二分类兼容入口: 由「上涨概率」构建统一信号

    用于仅有 prob_up 的旧数据/旧调用方(二分类模型、历史待验证记录)。
    此时不存在中性类, P(跌) = 1 - P(涨), 置信度 = max(P涨, P跌)。
    """
    p_up = float(prob_up)
    return build_prediction_signal({'up': p_up, 'down': 1.0 - p_up, 'flat': 0.0},
                                   conf_thresh=conf_thresh)


def signal_to_display(signal: dict) -> tuple:
    """由统一信号推导展示用的 (emoji, 趋势文案)

    方向来自 signal['pred_class'] (argmax), 强度来自 signal['confidence']。
    低置信度(< conf_thresh)时不再喊「强烈看涨/看跌」, 而是显式标注信号弱。

    Returns:
        (emoji: str, trend: str)
    """
    pred_class = signal.get('pred_class', 'flat')
    confidence = float(signal.get('confidence', 0.0))
    strong = getattr(config, 'CONF_STRONG', 0.7)
    moderate = getattr(config, 'CONF_MODERATE', 0.55)

    # 低置信度: 无论 argmax 指向哪边, 一律标注信号弱, 避免误导
    if signal.get('is_weak', False):
        arrow = {'up': '偏涨', 'down': '偏跌', 'flat': '横盘'}.get(pred_class, '横盘')
        return "⚖️", f"信号弱 · 中性观望({arrow})"

    if pred_class == 'flat':
        return "➖", "横盘整理"

    if pred_class == 'up':
        if confidence >= strong:
            return "🚀", "强烈看涨"
        if confidence >= moderate:
            return "📈", "看涨"
        return "↗️", "轻微看涨"

    # pred_class == 'down'
    if confidence >= strong:
        return "🔻", "强烈看跌"
    if confidence >= moderate:
        return "📉", "看跌"
    return "↘️", "轻微看跌"


def _ensure_signal(item: dict) -> dict:
    """从预测结果 dict 中取出/重建统一信号 (向后兼容旧格式)

    优先级:
      1. item['signal'] 已是统一信号 → 直接用
      2. item['probs'] 存在完整分布 → 由分布构建
      3. 仅有 item['probability'] (上涨概率) → 退化为二分类口径构建

    Args:
        item: 单窗口预测结果 dict

    Returns:
        dict: 统一信号
    """
    if isinstance(item.get('signal'), dict) and 'pred_class' in item['signal']:
        return item['signal']
    if isinstance(item.get('probs'), dict) and item['probs']:
        return build_prediction_signal(item['probs'])
    return build_signal_from_prob_up(item.get('probability', 0.5))


class MeoWNotifier:
    """MeoW消息推送通知器"""

    def __init__(self, nickname: str, base_url: str = "https://api.chuckfang.com"):
        """
        初始化通知器

        Args:
            nickname: 用户昵称（不允许包含斜杠）
            base_url: API基础地址，默认使用HTTPS
        """
        self.nickname = nickname
        self.base_url = base_url.rstrip('/')
        self.timeout = 10  # 请求超时时间（秒）

    def _build_get_url(self, title: str, msg: str, url: Optional[str] = None,
                       msg_type: str = "text", html_height: Optional[int] = None) -> str:
        """构建GET请求URL"""
        # 对路径参数进行URL编码
        encoded_nickname = urllib.parse.quote(self.nickname, safe='')
        encoded_title = urllib.parse.quote(title, safe='')
        encoded_msg = urllib.parse.quote(msg, safe='')

        # 构建基础URL
        path = f"/{encoded_nickname}/{encoded_title}/{encoded_msg}"

        # 构建查询参数
        query_params = []
        if url:
            query_params.append(f"url={urllib.parse.quote(url, safe='')}")
        if msg_type != "text":
            query_params.append(f"msgType={msg_type}")
        if html_height is not None and msg_type == "html":
            query_params.append(f"htmlHeight={html_height}")

        if query_params:
            path += "?" + "&".join(query_params)

        return f"{self.base_url}{path}"

    def _send_request(self, url: str, method: str = "GET", data: Optional[bytes] = None,
                      headers: Optional[dict] = None) -> dict:
        """发送HTTP请求"""
        try:
            req = urllib.request.Request(url, method=method)
            if headers:
                for key, value in headers.items():
                    req.add_header(key, value)
            if data:
                req.data = data

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP错误: {e.code} - {e.reason}")
            try:
                error_body = e.read().decode('utf-8')
                return json.loads(error_body)
            except:
                return {"status": e.code, "msg": str(e.reason)}
        except Exception as e:
            logger.error(f"请求失败: {e}")
            return {"status": 500, "msg": str(e)}

    def send_get(self, title: str, msg: str, url: Optional[str] = None,
                 msg_type: str = "text", html_height: Optional[int] = None) -> dict:
        """
        使用GET方式发送消息

        Args:
            title: 消息标题
            msg: 消息内容
            url: 跳转链接（需URL编码）
            msg_type: 消息显示类型: 'text'(默认，纯文本显示), 'html'(在App中渲染HTML格式)
            html_height: 仅在msgType=html时生效，App中显示HTML内容的高度（单位：像素），默认200

        Returns:
            dict: API响应结果 {"status": 200, "msg": "推送成功"}
        """
        request_url = self._build_get_url(title, msg, url, msg_type, html_height)
        logger.debug(f"GET请求URL: {request_url}")
        return self._send_request(request_url, method="GET")

    def send_post_json(self, title: str, msg: str, url: Optional[str] = None,
                       msg_type: str = "text", html_height: Optional[int] = None) -> dict:
        """
        使用POST JSON方式发送消息

        Args:
            title: 消息标题
            msg: 消息内容
            url: 跳转链接
            msg_type: 消息显示类型
            html_height: HTML内容高度

        Returns:
            dict: API响应结果
        """
        # 构建请求URL（包含查询参数）
        query_params = []
        if msg_type != "text":
            query_params.append(f"msgType={msg_type}")
        if html_height is not None and msg_type == "html":
            query_params.append(f"htmlHeight={html_height}")

        encoded_nickname = urllib.parse.quote(self.nickname, safe='')
        path = f"/{encoded_nickname}"
        if query_params:
            path += "?" + "&".join(query_params)

        request_url = f"{self.base_url}{path}"

        # 构建JSON数据
        payload = {
            "title": title,
            "msg": msg
        }
        if url:
            payload["url"] = url

        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        headers = {
            "Content-Type": "application/json"
        }

        logger.debug(f"POST请求URL: {request_url}")
        logger.debug(f"POST请求数据: {payload}")

        return self._send_request(request_url, method="POST", data=data, headers=headers)

    def send(self, title: str, msg: str, url: Optional[str] = None,
             msg_type: str = "text", html_height: Optional[int] = None,
             use_post: bool = True) -> bool:
        """
        发送消息（推荐使用的方法）

        Args:
            title: 消息标题
            msg: 消息内容
            url: 跳转链接
            msg_type: 消息显示类型
            html_height: HTML内容高度
            use_post: 是否使用POST方式（默认True，更稳定）

        Returns:
            bool: 是否发送成功
        """
        if use_post:
            result = self.send_post_json(title, msg, url, msg_type, html_height)
        else:
            result = self.send_get(title, msg, url, msg_type, html_height)

        if result.get("status") == 200:
            logger.info(f"消息推送成功: {title}")
            return True
        else:
            logger.error(f"消息推送失败: {result}")
            return False

    def send_prediction(self, time: str, price: float, direction: str = None,
                        probability: float = None, confidence: float = None,
                        pred_class: str = None, probs: dict = None) -> bool:
        """
        发送预测结果通知 (统一信号口径)

        方向取 argmax(pred_class), 强度取置信度; 低置信度显式标注「信号弱」,
        与 predict.py 打印的方向严格一致。

        Args:
            time: 当前时间
            price: 当前价格
            direction: 预测方向中文串(可选, 缺省由信号推导)
            probability: 上涨概率(二分类兼容入口, 有 probs 时可省略)
            confidence: 置信度(可选, 缺省由信号推导)
            pred_class: 预测类 'up'/'down'/'flat' (可选, 缺省由 probs argmax 得出)
            probs: 完整概率分布 {'up','down','flat'} (推荐传入)

        Returns:
            bool: 是否发送成功
        """
        signal = _ensure_signal({
            'probs': probs,
            'probability': probability if probability is not None else 0.5,
        })
        # 调用方显式给了 pred_class 时以其为准(保持与 predict.py 打印完全一致)
        if pred_class in CLASS_INDICES:
            signal = dict(signal)
            signal['pred_class'] = pred_class
            signal['pred_class_idx'] = CLASS_INDICES[pred_class]
            signal['direction'] = DIRECTION_LABELS[pred_class]
            signal['confidence'] = float(signal['probs'].get(pred_class, signal['confidence']))
            signal['is_weak'] = signal['confidence'] < signal['conf_thresh']

        emoji, trend = signal_to_display(signal)
        _direction = direction or signal['direction']
        _confidence = confidence if confidence is not None else signal['confidence']
        _p = signal['probs']

        title = f"{emoji} BTC预测 - {trend}"

        msg = (f'<div style="font-size:40px;line-height:1.8">'
               f'<b>时间:</b> {time}<br>'
               f'<b>当前价格:</b> {price:.2f} USDT<br>'
               f'<b>预测方向:</b> {_direction}<br>'
               f'<b>概率分布:</b> 涨{_p["up"]*100:.1f}% / '
               f'平{_p["flat"]*100:.1f}% / 跌{_p["down"]*100:.1f}%<br>'
               f'<b>置信度:</b> {_confidence*100:.2f}%'
               f'{" ⚠️信号弱" if signal["is_weak"] else ""}<br>'
               f'<b>预测窗口:</b> 未来10分钟</div>')

        return self.send(title, msg, msg_type="html", html_height=220)

    def send_training_start(self, epochs: int) -> bool:
        """发送训练开始通知"""
        title = "🔄 模型训练开始"
        msg = f"LNN模型开始训练，计划训练 {epochs} 个epoch"
        return self.send(title, msg)

    def send_training_complete(self, epoch: int, val_loss: float, val_acc: float,
                               test_acc: float = None, precision: float = None,
                               recall: float = None, f1: float = None,
                               horizon_results: dict = None) -> bool:
        """
        发送训练完成通知

        Args:
            epoch: 最终epoch数
            val_loss: 验证损失
            val_acc: 验证准确率
            test_acc: 测试准确率（可选）
            precision: 精确率（可选）
            recall: 召回率（可选）
            f1: F1分数（可选）
            horizon_results: 各预测窗口的准确率字典 (可选), 如 {"10m": 0.65, "30m": 0.58}
        """
        title = "✅ 模型训练完成"

        msg_lines = [
            f"<b>最终Epoch:</b> {epoch}",
            f"<b>验证损失:</b> {val_loss:.4f}",
            f"<b>验证准确率:</b> {val_acc*100:.2f}%"
        ]

        if test_acc is not None:
            msg_lines.append(f"<b>测试准确率:</b> {test_acc*100:.2f}%")
        if precision is not None:
            msg_lines.append(f"<b>精确率:</b> {precision:.4f}")
        if recall is not None:
            msg_lines.append(f"<b>召回率:</b> {recall:.4f}")
        if f1 is not None:
            msg_lines.append(f"<b>F1分数:</b> {f1:.4f}")

        # 各窗口准确率
        if horizon_results:
            h_str = " | ".join([f"{k}:{v*100:.1f}%" for k, v in horizon_results.items()])
            msg_lines.append(f"<b>各窗口准确率:</b> {h_str}")

        _height = 280 if horizon_results else 250
        msg = '<div style="font-size:40px;line-height:1.8">' + "<br>".join(msg_lines) + '</div>'
        return self.send(title, msg, msg_type="html", html_height=_height)

    def send_training_error(self, error_msg: str) -> bool:
        """发送训练错误通知"""
        title = "❌ 训练出错"
        msg = f"模型训练过程中发生错误:\n{error_msg}"
        return self.send(title, msg)

    def send_data_fetch_error(self, error_msg: str) -> bool:
        """发送数据获取错误通知"""
        title = "⚠️ 数据获取失败"
        msg = f"获取K线数据时发生错误:\n{error_msg}"
        return self.send(title, msg)

    def send_prediction_verify(self, direction: str, actual_direction: str,
                                is_correct: bool, current_price: float,
                                verify_price: float, price_change_pct: float,
                                horizon: int = None, stats: dict = None) -> bool:
        """
        发送预测验证结果通知（含累积统计信息）

        Args:
            direction: 预测方向
            actual_direction: 实际方向
            is_correct: 预测是否正确
            current_price: 预测时价格
            verify_price: 验证时价格
            price_change_pct: 价格变化百分比
            horizon: 预测窗口(分钟), 可选
            stats: 累积统计字典, 可选, 包含:
                - overall_accuracy: 总正确率
                - total_verified: 总验证次数
                - by_horizon: 各窗口统计 dict[horizon] = {'correct': int, 'total': int, 'accuracy': float}
                - last_updated: 最后更新时间

        Returns:
            bool: 是否发送成功
        """
        mark = "✅" if is_correct else "❌"
        h_label = f"[{horizon}min]" if horizon else ""
        title = f"{mark} BTC预测验证{h_label} - {'正确' if is_correct else '错误'}"
        stats_html_height = 0

        lines = [
            f'<b>预测窗口:</b> {horizon or 10}分钟',
            f'<b>预测方向:</b> {direction}',
            f'<b>实际方向:</b> {actual_direction}',
            f'<b>结果:</b> {"正确 ✅" if is_correct else "错误 ❌"}',
            f'<br>',
            f'<b>预测价格:</b> {current_price:.2f} USDT',
            f'<b>验证价格:</b> {verify_price:.2f} USDT',
            f'<b>价格变化:</b> {price_change_pct:+.2f}%',
        ]

        # 追加累积统计信息
        if stats:
            lines.append('<br>')
            lines.append(f'<b>━━━ 累积统计 ━━━</b>')
            lines.append(f'<b>总正确率:</b> {stats.get("overall_accuracy", 0)*100:.1f}% '
                         f'({stats.get("total_verified", 0)}次)')

            by_horizon = stats.get('by_horizon', {})
            for h_key in sorted(by_horizon.keys(), key=int):
                h_stat = by_horizon[h_key]
                lines.append(
                    f'<b>[{h_key}min]</b> {h_stat["accuracy"]*100:.1f}% '
                    f'({h_stat["correct"]}/{h_stat["total"]})'
                )
            stats_html_height = 40 + len(by_horizon) * 30

        msg = '<div style="font-size:40px;line-height:1.8">' + "<br>".join(lines) + '</div>'
        html_h = 220 + stats_html_height
        return self.send(title, msg, msg_type="html", html_height=html_h)

    def send_multi_horizon_prediction(self, time: str, price: float,
                                       horizons_results: list) -> bool:
        """
        发送多窗口预测结果通知

        Args:
            time: 当前时间
            price: 当前价格
            horizons_results: 各窗口预测结果列表, 每项包含:
                - horizon:    窗口(分钟)
                - pred_class: 'up'/'down'/'flat' (argmax 结果, 推荐传入)
                - probs:      完整概率分布 {'up','down','flat'} (推荐传入)
                - direction:  方向中文串(可选, 缺省由信号推导)
                - confidence: 置信度(可选, 缺省由信号推导)
                - probability: 上涨概率(旧格式兼容; 无 probs 时退化为二分类口径)

        Returns:
            bool: 是否发送成功
        """
        # 为每个窗口构建统一信号(方向=argmax, 强度=置信度)
        signals = []
        for r in horizons_results:
            sig = _ensure_signal(r)
            _pc = r.get('pred_class')
            if _pc in CLASS_INDICES:
                # 以调用方(predict.py)给出的 argmax 结果为准, 保证方向口径完全一致
                sig = dict(sig)
                sig['pred_class'] = _pc
                sig['pred_class_idx'] = CLASS_INDICES[_pc]
                sig['direction'] = DIRECTION_LABELS[_pc]
                sig['confidence'] = float(sig['probs'].get(_pc, sig['confidence']))
                sig['is_weak'] = sig['confidence'] < sig['conf_thresh']
            signals.append(sig)

        # 整体表情/趋势取最短窗口(主窗口)的统一信号
        if signals:
            emoji, trend = signal_to_display(signals[0])
        else:
            emoji, trend = "➖", "无数据"

        title = f"{emoji} BTC多窗口预测 - {trend}"

        lines = [
            f'<b>时间:</b> {time}',
            f'<b>当前价格:</b> {price:.2f} USDT',
            '<br>',
        ]
        for r, sig in zip(horizons_results, signals):
            h = r['horizon']
            _p = sig['probs']
            _, _h_trend = signal_to_display(sig)
            lines.append(
                f"<b>[{h:>3}min]</b> {sig['direction']} · {_h_trend}"
            )
            lines.append(
                f"　　涨{_p['up']*100:.1f}% / 平{_p['flat']*100:.1f}% / "
                f"跌{_p['down']*100:.1f}%　置信:{sig['confidence']*100:.1f}%"
            )

        # 全部窗口都是弱信号时, 追加一句整体提示
        if signals and all(s['is_weak'] for s in signals):
            lines.append('<br>')
            lines.append(
                f'<b>⚠️ 所有窗口置信度均低于 '
                f'{signals[0]["conf_thresh"]*100:.0f}%, 建议观望</b>'
            )

        msg = '<div style="font-size:40px;line-height:1.8">' + "<br>".join(lines) + '</div>'
        # 每个窗口两行, 约100px高度 + 头部信息
        html_h = max(220, len(horizons_results) * 105 + 120)
        return self.send(title, msg, msg_type="html", html_height=html_h)


# 便捷函数，用于快速发送通知
def notify_prediction(nickname: str, time: str, price: float, direction: str,
                      probability: float, confidence: float) -> bool:
    """快速发送预测结果通知"""
    notifier = MeoWNotifier(nickname)
    return notifier.send_prediction(time, price, direction, probability, confidence)


def notify_training_start(nickname: str, epochs: int) -> bool:
    """快速发送训练开始通知"""
    notifier = MeoWNotifier(nickname)
    return notifier.send_training_start(epochs)


def notify_training_complete(nickname: str, epoch: int, val_loss: float,
                             val_acc: float, **kwargs) -> bool:
    """快速发送训练完成通知"""
    notifier = MeoWNotifier(nickname)
    return notifier.send_training_complete(epoch, val_loss, val_acc, **kwargs)


def notify_error(nickname: str, title: str, error_msg: str) -> bool:
    """快速发送错误通知"""
    notifier = MeoWNotifier(nickname)
    return notifier.send(title, f"发生错误:\n{error_msg}")


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) < 3:
        print("用法: python notifier.py <昵称> <测试消息>")
        print("示例: python notifier.py myname 测试消息")
        sys.exit(1)

    test_nickname = sys.argv[1]
    test_msg = sys.argv[2]

    notifier = MeoWNotifier(test_nickname)

    # 测试纯文本消息
    # print("测试发送纯文本消息...")
    # result = notifier.send("测试标题", test_msg)
    # print(f"结果: {'成功' if result else '失败'}")

    # 测试HTML消息
    # print("\n测试发送HTML消息...")
    # html_msg = "<b>粗体文本</b> 和 <i>斜体文本</i>"
    # result = notifier.send("HTML测试", html_msg, msg_type="html", html_height=150)
    # print(f"结果: {'成功' if result else '失败'}")

    # 测试预测通知
    print("\n测试发送预测通知...")
    result = notifier.send_prediction(
        time="2024-01-01 12:00:00",
        price=65000.50,
        direction="涨 (UP)",
        probability=0.75,
        confidence=0.50
    )
    print(f"结果: {'成功' if result else '失败'}")
