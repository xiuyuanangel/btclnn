"""MeoW消息推送通知模块

基于 https://www.chuckfang.com/MeoW/api_doc.html 文档实现
接口地址: http://api.chuckfang.com/ 或 https://api.chuckfang.com/
"""

import json
import logging
import urllib.parse
import urllib.request
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    def send_prediction(self, time: str, price: float, direction: str,
                        probability: float, confidence: float) -> bool:
        """
        发送预测结果通知

        Args:
            time: 当前时间
            price: 当前价格
            direction: 预测方向
            probability: 上涨概率
            confidence: 置信度

        Returns:
            bool: 是否发送成功
        """
        # 根据概率确定表情和颜色
        if probability > 0.7:
            emoji = "🚀"
            trend = "强烈看涨"
        elif probability > 0.6:
            emoji = "📈"
            trend = "看涨"
        elif probability > 0.5:
            emoji = "↗️"
            trend = "轻微看涨"
        elif probability > 0.4:
            emoji = "↘️"
            trend = "轻微看跌"
        elif probability > 0.3:
            emoji = "📉"
            trend = "看跌"
        else:
            emoji = "🔻"
            trend = "强烈看跌"

        title = f"{emoji} BTC预测 - {trend}"

        msg = (f'<div style="font-size:40px;line-height:1.8">'
               f'<b>时间:</b> {time}<br>'
               f'<b>当前价格:</b> {price:.2f} USDT<br>'
               f'<b>预测方向:</b> {direction}<br>'
               f'<b>上涨概率:</b> {probability*100:.2f}%<br>'
               f'<b>置信度:</b> {confidence*100:.2f}%<br>'
               f'<b>预测窗口:</b> 未来10分钟</div>')

        return self.send(title, msg, msg_type="html", html_height=200)

    def send_training_start(self, epochs: int) -> bool:
        """发送训练开始通知"""
        title = "🔄 模型训练开始"
        msg = f"LNN模型开始训练，计划训练 {epochs} 个epoch"
        return self.send(title, msg)

    def send_training_complete(self, epoch: int, val_loss: float, val_acc: float,
                               test_acc: float = None, precision: float = None,
                               recall: float = None, f1: float = None) -> bool:
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

        msg = '<div style="font-size:40px;line-height:1.8">' + "<br>".join(msg_lines) + '</div>'
        return self.send(title, msg, msg_type="html", html_height=250)

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
                                verify_price: float, price_change_pct: float) -> bool:
        """
        发送预测验证结果通知(10分钟后对比实际价格)

        Args:
            direction: 预测方向
            actual_direction: 实际方向
            is_correct: 预测是否正确
            current_price: 预测时价格
            verify_price: 验证时价格
            price_change_pct: 价格变化百分比

        Returns:
            bool: 是否发送成功
        """
        mark = "✅" if is_correct else "❌"
        title = f"{mark} BTC预测验证 - {'正确' if is_correct else '错误'}"

        msg = (
            f'<div style="font-size:40px;line-height:1.8">'
            f'<b>预测方向:</b> {direction}<br>'
            f'<b>实际方向:</b> {actual_direction}<br>'
            f'<b>结果:</b> {"正确 ✅" if is_correct else "错误 ❌"}<br>'
            f'<br>'
            f'<b>预测价格:</b> {current_price:.2f} USDT<br>'
            f'<b>验证价格:</b> {verify_price:.2f} USDT<br>'
            f'<b>价格变化:</b> {price_change_pct:+.2f}%</div>'
        )
        return self.send(title, msg, msg_type="html", html_height=220)


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
