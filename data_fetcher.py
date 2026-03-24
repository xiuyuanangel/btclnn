"""火币永续合约K线数据获取器，支持本地缓存和数据去重"""

import json
import os
import time
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HuobiDataFetcher:
    """火币永续合约数据获取器

    功能:
    - 从火币API获取永续合约K线数据
    - 本地JSON缓存，减少API请求
    - 自动数据去重
    - 5分钟K线聚合为10分钟K线
    """

    def __init__(self, symbol=None, base_url=None):
        self.symbol = symbol or config.SYMBOL
        self.base_url = base_url or config.HUOBI_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "LNN-Bot/1.0"
        })

    def get_cache_path(self, period):
        """获取缓存文件路径"""
        safe_symbol = self.symbol.replace("-", "_")
        return os.path.join(config.DATA_DIR, f"{safe_symbol}_{period}.json")

    def load_cache(self, period):
        """加载本地缓存数据"""
        cache_path = self.get_cache_path(period)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"从缓存加载 {len(data)} 条 {period} 数据")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"缓存加载失败: {e}")
        return []

    def save_cache(self, data, period):
        """保存数据到本地缓存"""
        cache_path = self.get_cache_path(period)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"已缓存 {len(data)} 条 {period} 数据到 {cache_path}")

    def fetch_kline_batch(self, period, size=None, end_time=None):
        """从API获取单批K线数据

        Args:
            period: K线周期 (1min, 5min, 15min, 30min, 60min, 4hour, 1day)
            size: 请求数量
            end_time: 结束时间戳(秒)

        Returns:
            list: K线数据列表
        """
        url = f"{self.base_url}{config.HUOBI_KLINE_ENDPOINT}"
        params = {
            "contract_code": self.symbol,
            "period": period,
            "size": size or config.API_BATCH_SIZE,
        }
        if end_time:
            params["to"] = int(end_time)

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("status") != "ok":
                logger.error(f"API返回错误: {result}")
                return []

            data = result.get("data", [])
            if not data:
                return []

            # 兼容数组和字典两种返回格式
            normalized = []
            for item in data:
                if isinstance(item, list):
                    normalized.append({
                        "id": item[0],
                        "open": float(item[1]),
                        "close": float(item[2]),
                        "high": float(item[3]),
                        "low": float(item[4]),
                        "vol": float(item[5]),
                        "amount": float(item[6]) if len(item) > 6 else 0,
                    })
                elif isinstance(item, dict):
                    normalized.append({
                        "id": item["id"],
                        "open": float(item.get("open", 0)),
                        "close": float(item.get("close", 0)),
                        "high": float(item.get("high", 0)),
                        "low": float(item.get("low", 0)),
                        "vol": float(item.get("vol", 0)),
                        "amount": float(item.get("amount", 0)),
                    })
            return normalized

        except requests.RequestException as e:
            logger.error(f"API请求失败: {e}")
            return []
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"数据解析失败: {e}")
            return []

    def deduplicate(self, data):
        """基于时间戳去重，保留最后出现的记录(最新数据优先)"""
        seen = {}
        for item in data:
            ts = item.get("id")
            if ts is not None:
                seen[ts] = item

        unique_data = sorted(seen.values(), key=lambda x: x.get("id", 0))
        removed = len(data) - len(unique_data)
        if removed > 0:
            logger.info(f"去重: {len(data)} -> {len(unique_data)} 条数据 (移除 {removed} 条重复)")
        return unique_data

    def sort_data(self, data):
        """按时间升序排列"""
        return sorted(data, key=lambda x: x.get("id", 0))

    def fetch_history(self, period, days=None):
        """获取历史K线数据，自动分页和缓存

        策略:
        1. 先加载本地缓存
        2. 从API获取缓存之后的新数据
        3. 如果总量不足，向前补充更早的数据
        4. 合并去重后保存缓存

        Args:
            period: K线周期
            days: 需要的历史天数

        Returns:
            list: K线数据列表(按时间升序)
        """
        days = days or config.LOOKBACK_DAYS
        minutes_per_candle = self._period_to_minutes(period)
        target_count = days * 24 * 60 // minutes_per_candle

        # 1. 加载缓存
        cached_data = self.load_cache(period)
        cached_data = self.deduplicate(cached_data)
        cached_data = self.sort_data(cached_data)

        # 检查缓存是否充足且足够新(最后一根K线在10分钟内)
        if cached_data:
            last_ts = cached_data[-1].get("id", 0)
            now_ts = int(time.time())
            if len(cached_data) >= target_count and (now_ts - last_ts) < 600:
                logger.info(f"缓存数据充足且新鲜: {len(cached_data)} 条")
                return cached_data

        # 2. 获取新数据(从最新缓存到当前时间)
        all_new_data = []
        oldest_cached_time = cached_data[0].get("id") if cached_data else None

        # 向前获取数据
        now_ts = int(time.time())
        end_ts = now_ts

        max_fetches = 30  # 安全限制，防止无限循环
        fetch_count = 0

        while fetch_count < max_fetches:
            batch = self.fetch_kline_batch(period, end_time=end_ts)
            if not batch:
                break

            batch = self.deduplicate(batch)
            if not batch:
                break

            # 检查获取到的数据是否与缓存重叠
            if cached_data:
                cached_ids = {item["id"] for item in cached_data}
                new_items = [item for item in batch if item["id"] not in cached_ids]
            else:
                new_items = batch

            all_new_data.extend(new_items)

            # 获取到的最早时间
            oldest_new = batch[0].get("id", 0)
            start_ts = now_ts - days * 24 * 3600

            if oldest_new <= start_ts or len(new_items) == 0:
                break

            end_ts = oldest_new - 1
            time.sleep(config.API_REQUEST_INTERVAL)
            fetch_count += 1

            # 如果已经有足够数据，停止向前获取
            total = len(cached_data) + len(all_new_data)
            if total >= target_count:
                break

        # 3. 合并去重排序
        combined = cached_data + all_new_data
        combined = self.deduplicate(combined)
        combined = self.sort_data(combined)

        # 4. 裁剪到目标天数范围
        now_ts = int(time.time())
        start_ts = now_ts - days * 24 * 3600
        combined = [item for item in combined if item.get("id", 0) >= start_ts]

        # 5. 保存缓存
        if combined:
            self.save_cache(combined, period)

        logger.info(f"获取完成: 共 {len(combined)} 条 {period} K线数据")
        return combined

    def _period_to_minutes(self, period):
        """将周期字符串转换为分钟数"""
        mapping = {
            "1min": 1, "5min": 5, "15min": 15, "30min": 30,
            "60min": 60, "4hour": 240, "1day": 1440,
            "1week": 10080, "1mon": 43200
        }
        return mapping.get(period, 5)

    def resample_to_10min(self, data_5min):
        """将5分钟K线聚合为10分钟K线

        聚合规则:
        - open: 第一根K线的开盘价
        - high: 所有K线的最高价
        - low: 所有K线的最低价
        - close: 最后一根K线的收盘价
        - vol: 成交量求和
        - amount: 成交额求和
        """
        if not data_5min:
            return []

        df = pd.DataFrame(data_5min)
        df['datetime'] = pd.to_datetime(df['id'], unit='s')

        # 按10分钟对齐
        df['period'] = df['datetime'].dt.floor('10min')

        # 聚合
        agg_rules = {
            'id': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum',
            'amount': 'sum',
        }
        agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

        df_10min = df.groupby('period').agg(agg_rules).reset_index()

        # 用对齐后的时间作为id
        df_10min['id'] = df_10min['period'].astype(int) // 10**9
        df_10min = df_10min.drop(columns=['period'])

        result = df_10min.to_dict('records')
        logger.info(f"5分钟 -> 10分钟聚合: {len(data_5min)} -> {len(result)} 条")
        return result

    def get_10min_data(self, days=None):
        """获取10分钟K线数据(从5分钟聚合而来)

        Args:
            days: 历史天数

        Returns:
            list: 10分钟K线数据列表
        """
        days = days or config.LOOKBACK_DAYS

        # 获取5分钟数据(多获取2天用于边缘处理)
        data_5min = self.fetch_history("5min", days + 2)

        if not data_5min:
            logger.error("未能获取到5分钟K线数据")
            return []

        # 聚合为10分钟
        data_10min = self.resample_to_10min(data_5min)

        # 缓存10分钟数据
        self.save_cache(data_10min, "10min")

        return data_10min

    def get_dataframe(self, data):
        """将字典列表转换为DataFrame"""
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if 'id' in df.columns:
            df['timestamp'] = pd.to_datetime(df['id'], unit='s')
            df = df.set_index('timestamp')

        for col in ['open', 'high', 'low', 'close', 'vol', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df


if __name__ == "__main__":
    fetcher = HuobiDataFetcher()
    data_10min = fetcher.get_10min_data()
    print(f"获取到 {len(data_10min)} 条10分钟K线数据")

    if data_10min:
        df = fetcher.get_dataframe(data_10min)
        print(df[['open', 'high', 'low', 'close', 'vol']].tail(10))
