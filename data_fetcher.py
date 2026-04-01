"""火币永续合约K线数据获取器，支持本地缓存和数据去重

API文档参考: https://www.htx.com/zh-cn/opend/newApiPages/?id=8cb73746-77b5-11ed-9966-0242ac110003
端点: /linear-swap-ex/market/history/kline
分页规则:
  - size与from&to必填其一，若全不填则返回空数据
  - 如果填写from，也要填写to。最多可获取连续两年的数据
  - 如果size、from、to均填写，会忽略from、to参数(只用size)
  => 分页必须使用 from&to，不能同时传size
"""

import json
import os
import time
import logging

import requests
import pandas as pd
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HuobiDataFetcher:
    """火币永续合约数据获取器"""

    def __init__(self, symbol=None, base_url=None):
        self.symbol = symbol or config.SYMBOL
        self.base_url = base_url or config.HUOBI_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "LNN-Bot/1.0"
        })

    def get_cache_path(self, period):
        safe_symbol = self.symbol.replace("-", "_")
        return os.path.join(config.DATA_DIR, f"{safe_symbol}_{period}.json")

    def load_cache(self, period):
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
        cache_path = self.get_cache_path(period)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"已缓存 {len(data)} 条 {period} 数据到 {cache_path}")

    def _normalize_kline(self, data):
        """标准化K线数据为统一格式"""
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

    def _fetch_kline_range(self, period, from_ts, to_ts):
        """使用 from&to 获取指定时间范围的K线数据(不传size)

        Args:
            period: K线周期 (1min, 5min, 15min, 30min, 60min, 4hour, 1day)
            from_ts: 开始时间戳(秒，10位)
            to_ts: 结束时间戳(秒，10位)

        Returns:
            list: K线数据列表
        """
        url = f"{self.base_url}{config.HUOBI_KLINE_ENDPOINT}"
        params = {
            "contract_code": self.symbol,
            "period": period,
            "from": int(from_ts),
            "to": int(to_ts),
        }

        # print(params)

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

            return self._normalize_kline(data)

        except requests.RequestException as e:
            logger.error(f"API请求失败: {e}")
            # 发送数据获取错误通知
            if config.MEOW_NICKNAME:
                try:
                    notifier = MeoWNotifier(config.MEOW_NICKNAME)
                    notifier.send_data_fetch_error(f"API请求失败: {e}")
                except:
                    pass
            return []
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"数据解析失败: {e}")
            # 发送数据获取错误通知
            if config.MEOW_NICKNAME:
                try:
                    notifier = MeoWNotifier(config.MEOW_NICKNAME)
                    notifier.send_data_fetch_error(f"数据解析失败: {e}")
                except:
                    pass
            return []

    def deduplicate(self, data):
        """基于时间戳去重"""
        seen = {}
        for item in data:
            ts = item.get("id")
            if ts is not None:
                seen[ts] = item
        unique_data = sorted(seen.values(), key=lambda x: x.get("id", 0))
        removed = len(data) - len(unique_data)
        if removed > 0:
            logger.info(f"去重: {len(data)} -> {len(unique_data)} 条 (移除 {removed} 条重复)")
        return unique_data

    def sort_data(self, data):
        return sorted(data, key=lambda x: x.get("id", 0))

    def fetch_history(self, period, days=None):
        """获取历史K线数据，按7天分块使用from&to分页

        API规则: size与from&to必填其一；三者都填时忽略from/to
        因此分页只用from&to(不传size)

        Args:
            period: K线周期
            days: 需要的历史天数

        Returns:
            list: K线数据列表(按时间升序)
        """
        days = days or config.LOOKBACK_DAYS
        minutes_per_candle = self._period_to_minutes(period)
        target_count = days * 24 * 60 // minutes_per_candle

        now_ts = int(time.time())
        start_ts = now_ts - (days + 2) * 24 * 3600  # 多取2天buffer

        # 1. 加载缓存
        cached_data = self.load_cache(period)
        cached_data = self.deduplicate(cached_data)
        cached_data = self.sort_data(cached_data)

        # 缓存检查: 数据量充足且最新K线在10分钟内
        if cached_data:
            last_ts = cached_data[-1].get("id", 0)
            if len(cached_data) >= target_count and (now_ts - last_ts) < 600:
                logger.info(f"缓存数据充足且新鲜: {len(cached_data)} 条")
                return cached_data

        # 2. 计算需要获取的时间范围
        if cached_data:
            # 需要获取: 比缓存最早的更早数据 + 比缓存最晚的更新的数据
            fetch_start = min(start_ts, cached_data[0].get("id", now_ts)) - 3600
            fetch_end = now_ts
        else:
            fetch_start = start_ts
            fetch_end = now_ts

        # 3. 按周期动态分块(确保每块不超过API的2000条限制)
        # 5min周期: 2000 * 300s ≈ 6.9天(7天=2016根会超限)
        max_candles_per_chunk = 1500  # 留buffer
        chunk_seconds = max_candles_per_chunk * minutes_per_candle * 60
        all_new_data = []
        chunk_start = fetch_start

        while chunk_start < fetch_end:
            chunk_end = min(chunk_start + chunk_seconds, fetch_end)
            batch = self._fetch_kline_range(period, chunk_start, chunk_end)

            if batch:
                # 排除缓存中已有的数据
                if cached_data:
                    cached_ids = {item["id"] for item in cached_data}
                    batch = [item for item in batch if item["id"] not in cached_ids]
                all_new_data.extend(batch)

            chunk_start = chunk_end
            time.sleep(config.API_REQUEST_INTERVAL)

        # 4. 合并去重排序
        combined = cached_data + all_new_data
        combined = self.deduplicate(combined)
        combined = self.sort_data(combined)

        # 5. 裁剪到目标范围
        combined = [item for item in combined if item.get("id", 0) >= start_ts]

        # 6. 保存缓存
        if combined:
            self.save_cache(combined, period)

        logger.info(f"获取完成: 共 {len(combined)} 条 {period} K线数据")
        return combined

    def _period_to_minutes(self, period):
        mapping = {
            "1min": 1, "5min": 5, "15min": 15, "30min": 30,
            "60min": 60, "4hour": 240, "1day": 1440,
            "1week": 10080, "1mon": 43200
        }
        return mapping.get(period, 5)

    def resample_to_10min(self, data_5min):
        """将5分钟K线聚合为10分钟K线"""
        if not data_5min:
            return []

        df = pd.DataFrame(data_5min)
        df['datetime'] = pd.to_datetime(df['id'], unit='s')
        df['period'] = df['datetime'].dt.floor('10min')

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
        df_10min['id'] = df_10min['period'].astype('datetime64[s]').astype(np.int64)
        df_10min = df_10min.drop(columns=['period'])

        result = df_10min.to_dict('records')
        logger.info(f"5分钟 -> 10分钟聚合: {len(data_5min)} -> {len(result)} 条")
        return result

    def get_10min_data(self, days=None):
        """获取10分钟K线数据(从5分钟聚合)"""
        days = days or config.LOOKBACK_DAYS

        data_5min = self.fetch_history("5min", days + 2)
        if not data_5min:
            logger.error("未能获取到5分钟K线数据")
            return []

        data_10min = self.resample_to_10min(data_5min)
        self.save_cache(data_10min, "10min")
        return data_10min

    def fetch_multi_timeframe(self):
        """获取所有配置周期的K线数据

        Returns:
            dict: {period: list_of_kline_dicts} 每个周期的原始K线数据
        """
        timeframe_data = {}
        for period, cfg in config.TIMEFRAMES.items():
            lookback = cfg['lookback_days']
            logger.info(f"获取 {period} 数据 (lookback={lookback}天)...")
            data = self.fetch_history(period, lookback)
            timeframe_data[period] = data
            logger.info(f"  {period}: {len(data)} 条K线")
        return timeframe_data

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
