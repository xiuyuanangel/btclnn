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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
import urllib3

import config

# 禁用SSL不安全警告(火币备用节点证书有问题)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局API并发信号量: 限制所有线程的总请求数，避免触发API限流
_api_semaphore = threading.Semaphore(4)


class HuobiDataFetcher:
    """火币永续合约数据获取器，支持多节点备用切换"""

    def __init__(self, symbol=None, base_url=None):
        self.symbol = symbol or getattr(config, 'SYMBOL', config.SYMBOLS[0] if hasattr(config, 'SYMBOLS') and config.SYMBOLS else 'BTC-USDT')
        self.base_urls = config.HUOBI_BASE_URLS
        self.base_url = base_url or config.HUOBI_BASE_URL
        self.current_url_index = 0
        if self.base_url in self.base_urls:
            self.current_url_index = self.base_urls.index(self.base_url)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "LNN-Bot/1.0"
        })
        # 禁用SSL验证(火币备用节点证书问题) + 增强连接池
        self.session.mount('https://', requests.adapters.HTTPAdapter(
            pool_maxsize=10,
            max_retries=0,  # 我们自己控制重试逻辑
        ))
        # 节点切换锁(多线程安全)
        self._url_lock = threading.Lock()

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

    def _switch_to_next_url(self):
        """切换到下一个备用API节点(线程安全)"""
        with self._url_lock:
            next_index = (self.current_url_index + 1) % len(self.base_urls)
            old_url = self.base_url
            self.base_url = self.base_urls[next_index]
            self.current_url_index = next_index
        logger.warning(f"API节点切换: {old_url} -> {self.base_url}")
        return self.base_url

    def _fetch_kline_range(self, period, from_ts, to_ts):
        """使用 from&to 获取指定时间范围的K线数据(不传size)
        支持多节点自动切换 + 限流自动退避

        Args:
            period: K线周期 (1min, 5min, 15min, 30min, 60min, 4hour, 1day)
            from_ts: 开始时间戳(秒，10位)
            to_ts: 结束时间戳(秒，10位)

        Returns:
            list: K线数据列表
        """
        # 获取全局信号量(限制总并发请求数, 带超时防死锁)
        if not _api_semaphore.acquire(timeout=120):
            logger.error(f"获取API信号量超时 ({period}, {from_ts}-{to_ts})")
            return []

        max_node_retries = len(self.base_urls)  # 每个节点尝试一次
        rate_limit_backoff = 1.0  # 限流退避初始秒数

        try:
            for attempt in range(max_node_retries):
                url = f"{self.base_url}{config.HUOBI_KLINE_ENDPOINT}"
                params = {
                    "contract_code": self.symbol,
                    "period": period,
                    "from": int(from_ts),
                    "to": int(to_ts),
                }

                try:
                    response = self.session.get(
                        url, params=params, timeout=30,
                        verify=False,  # 跳过SSL证书验证
                    )
                    response.raise_for_status()
                    result = response.json()

                    if result.get("status") != "ok":
                        err_code = result.get("err-code", "")
                        err_msg = result.get("err_msg", result)

                        # 限流错误: 等待后重试(不切换节点)
                        if "limit" in str(err_code).lower() or "limit" in str(err_msg).lower():
                            logger.warning(
                                f"[{self.base_url}] API限流 ({period}), "
                                f"等待 {rate_limit_backoff:.1f}s 后重试..."
                            )
                            time.sleep(rate_limit_backoff)
                            rate_limit_backoff = min(rate_limit_backoff * 2, 10.0)
                            continue

                        # 其他API错误: 切换到下一个节点
                        logger.error(f"[{self.base_url}] API返回错误: {err_msg}")
                        if attempt < max_node_retries - 1:
                            self._switch_to_next_url()
                            continue
                        return []

                    data = result.get("data", [])
                    if not data:
                        return []

                    return self._normalize_kline(data)

                except requests.exceptions.SSLError as e:
                    logger.error(f"[{self.base_url}] SSL错误: {e}")
                    if attempt < max_node_retries - 1:
                        self._switch_to_next_url()
                        continue
                    return []

                except requests.exceptions.ConnectionError as e:
                    logger.error(f"[{self.base_url}] 连接错误: {e}")
                    if attempt < max_node_retries - 1:
                        self._switch_to_next_url()
                        time.sleep(0.5)
                        continue
                    return []

                except requests.RequestException as e:
                    logger.error(f"[{self.base_url}] API请求失败: {e}")
                    if attempt < max_node_retries - 1:
                        self._switch_to_next_url()
                        continue
                    return []

                except (IndexError, KeyError, ValueError) as e:
                    logger.error(f"数据解析失败: {e}")
                    return []
            return []
        finally:
            _api_semaphore.release()

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

    def fetch_history(self, period, days=None, force_refresh=False):
        """获取历史K线数据，按时间块分块并发使用from&to分页

        Args:
            period: K线周期
            days: 需要的历史天数
            force_refresh: 是否强制刷新缓存(跳过新鲜度检查直接拉取最新数据)
        """
        days = days or 30  # 默认30天(兼容 get_10min_data 等无参调用)
        minutes_per_candle = self._period_to_minutes(period)
        target_count = days * 24 * 60 // minutes_per_candle

        now_ts = int(time.time())
        start_ts = now_ts - (days + 2) * 24 * 3600  # 多取2天buffer

        # 1. 加载缓存
        cached_data = self.load_cache(period)
        cached_data = self.deduplicate(cached_data)
        cached_data = self.sort_data(cached_data)

        # 缓存检查: 数据量充足且最新K线在10分钟内 (force_refresh时跳过)
        if cached_data and not force_refresh:
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
        
        # 生成时间块列表
        chunks = []
        chunk_start = fetch_start
        while chunk_start < fetch_end:
            chunk_end = min(chunk_start + chunk_seconds, fetch_end)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
        
        logger.info(f"{period} 数据分块: {len(chunks)} 个时间块，每块约 {minutes_per_candle * max_candles_per_chunk} 根K线")
        
        # 4. 并发获取所有时间块数据(全局信号量已限制总并发数)
        all_new_data = []
        lock = threading.Lock()  # 用于线程安全的batch聚合

        def _fetch_chunk(args):
            cs, ce = args
            batch = self._fetch_kline_range(period, cs, ce)
            if batch:
                # 排除缓存中已有的数据(使用预构建的ID集合)
                if cached_ids:
                    batch = [item for item in batch if item["id"] not in cached_ids]
                if batch:
                    with lock:
                        all_new_data.extend(batch)
            return len(batch)

        logger.info(f"并发获取 {period} 数据: {len(chunks)} 个块")

        # 预构建缓存ID集合(避免每个线程内重复构建)
        cached_ids = {item["id"] for item in cached_data} if cached_data else None

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            # 提交所有任务
            futures = [executor.submit(_fetch_chunk, chunk) for chunk in chunks]
            
            # 等待所有任务完成并统计
            completed = 0
            total_fetched = 0
            for future in as_completed(futures):
                try:
                    count = future.result()
                    total_fetched += count
                    completed += 1
                    if completed % 5 == 0 or completed == len(chunks):
                        logger.info(f"{period} 进度: {completed}/{len(chunks)} 块完成，已获取 {total_fetched} 条")
                except Exception as e:
                    logger.error(f"获取时间块数据异常: {e}")
        
        # 5. 合并去重排序
        combined = cached_data + all_new_data
        combined = self.deduplicate(combined)
        combined = self.sort_data(combined)

        # 6. 裁剪到目标范围
        combined = [item for item in combined if item.get("id", 0) >= start_ts]

        # 7. 保存缓存
        if combined:
            self.save_cache(combined, period)

        logger.info(f"获取完成: 共 {len(combined)} 条 {period} K线数据 (新增 {len(all_new_data)} 条)")
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
        days = days or 30  # 默认30天(兼容 get_10min_data 等无参调用)

        data_5min = self.fetch_history("5min", days + 2)
        if not data_5min:
            logger.error("未能获取到5分钟K线数据")
            return []

        data_10min = self.resample_to_10min(data_5min)
        self.save_cache(data_10min, "10min")
        return data_10min

    def fetch_multi_timeframe(self, force_refresh=False):
        """获取所有配置周期的K线数据(多线程并发)

        Args:
            force_refresh: 是否强制刷新所有周期的缓存

        Returns:
            dict: {period: list_of_kline_dicts} 每个周期的原始K线数据
        """
        periods = list(config.TIMEFRAMES.items())
        timeframe_data = {}

        def _fetch_one(period_cfg):
            period, cfg = period_cfg
            lookback = cfg['lookback_days']
            logger.info(f"[并发] 开始获取 {period} 数据 (lookback={lookback}天)...")
            data = self.fetch_history(period, lookback, force_refresh=force_refresh)
            logger.info(f"[并发] {period}: {len(data)} 条K线")
            return period, data

        max_workers = min(len(periods), 6)
        logger.info(f"多线程并发获取 {len(periods)} 个周期数据 (workers={max_workers})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, pc): pc[0] for pc in periods}
            for future in as_completed(futures):
                period = futures[future]
                try:
                    p, data = future.result()
                    timeframe_data[p] = data
                except Exception as e:
                    logger.error(f"获取 {period} 数据异常: {e}")
                    timeframe_data[period] = []

        # 按原始顺序输出汇总日志
        for period in config.TIMEFRAMES:
            if period in timeframe_data:
                logger.info(f"  {period}: {len(timeframe_data[period])} 条K线")
        return timeframe_data

    def fetch_all_symbols_data(self, force_refresh=False):
        """获取所有配置币种的多周期数据

        Returns:
            dict: {symbol: {period: list_of_kline_dicts}}
        """
        all_symbols_data = {}
        symbols = getattr(config, 'SYMBOLS', [config.SYMBOL]) if hasattr(config, 'SYMBOLS') else [config.SYMBOL]

        for symbol in symbols:
            logger.info(f"{'='*60}")
            logger.info(f"开始获取 {symbol} 数据")
            logger.info(f"{'='*60}")

            symbol_fetcher = HuobiDataFetcher(symbol=symbol)
            symbol_tf_data = symbol_fetcher.fetch_multi_timeframe(force_refresh=force_refresh)

            # 获取10分钟数据作为目标数据
            if '5min' in symbol_tf_data and symbol_tf_data['5min']:
                symbol_10min_data = symbol_fetcher.resample_to_10min(symbol_tf_data['5min'])
                symbol_tf_data['10min'] = symbol_10min_data

            all_symbols_data[symbol] = symbol_tf_data

        logger.info(f"{'='*60}")
        logger.info(f"多币种数据获取完成: {len(all_symbols_data)} 个币种")
        logger.info(f"{'='*60}")
        for symbol, data in all_symbols_data.items():
            logger.info(f"  {symbol}: {len(data)} 个周期")
        return all_symbols_data

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
