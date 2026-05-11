"""强化学习交易智能体 — 基于LNN模型的多周期二元期权交易

使用预训练的 MultiTimeframeLNN 模型作为状态提取器, 通过 Dueling DQN 算法
学习最优交易策略: 在哪个周期交易、方向(做多/做空)和投入本金(5~100 USDT)。

交易规则:
  - 10分钟周期: 预测正确获得 80% 收益, 错误亏损全部本金
  - 30分钟周期: 预测正确获得 85% 收益, 错误亏损全部本金
  - 60分钟周期: 预测正确获得 85% 收益, 错误亏损全部本金

用法:
  训练模式:  python rl_trader.py --mode train --episodes 500
  评估模式:  python rl_trader.py --mode eval
  实时交易:  python rl_trader.py --mode trade
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
from data_fetcher import HuobiDataFetcher
from features import (
    compute_all_features, compute_context_features,
    build_multi_tf_dataset, build_multi_symbol_dataset,
    split_multi_tf_dataset, normalize_datasets,
    normalize_sequence_samplewise,
    SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS,
)
from lnn_model import MultiTimeframeLNN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# 配置
# =============================================================================

@dataclass
class RLConfig:
    """强化学习配置"""
    # 资金管理
    initial_balance: float = 1000.0      # 初始本金 (USDT)
    min_bet: float = 5.0                 # 最小投入本金
    max_bet: float = 100.0               # 最大投入本金

    # 收益率配置 (二元期权: 正确获得收益, 错误亏损本金)
    profit_rates: Dict[int, float] = field(default_factory=lambda: {
        10: 0.80,   # 10分钟: 80%
        30: 0.85,   # 30分钟: 85%
        60: 0.85,   # 60分钟: 85%
    })

    # 可用投入本金离散化 (11档)
    bet_sizes: List[float] = field(default_factory=lambda:
        [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100])

    # DQN 超参数
    learning_rate: float = 3e-4
    gamma: float = 0.99                  # 折扣因子
    epsilon_start: float = 1.0           # 初始探索率
    epsilon_end: float = 0.01            # 最小探索率
    epsilon_decay: float = 0.995         # 探索率衰减 (每步)
    buffer_capacity: int = 100_000       # 经验回放容量
    batch_size: int = 256                # 训练批大小
    target_update: int = 200             # 目标网络更新间隔 (步数)
    hidden_dim: int = 256                # Q网络隐藏层维度

    # 训练配置
    episodes: int = 300                  # 训练回合数
    eval_interval: int = 20              # 评估间隔 (回合)
    max_steps_per_episode: int = 1000    # 每回合最大步数
    save_dir: str = os.path.join(config.CHECKPOINT_DIR, "rl")

    # 状态维度
    state_returns_window: int = 5        # 状态中包含的最近收益率数量

    # 预热步数
    warmup_steps: int = 1000

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)


# =============================================================================
# 环境: TradingEnv
# =============================================================================

class TradingEnv:
    """RL交易环境

    基于历史数据模拟逐时间步交易, 每个时间步:
      1. LNN模型给出三个周期的涨跌概率
      2. 智能体选择动作 (skip / 指定周期+方向+本金)
      3. 根据历史真实涨跌计算盈亏
    """

    def __init__(
        self,
        dataset_dict: dict,
        predictions: np.ndarray,   # (N, 3) 模型预测的上涨概率
        close_prices: np.ndarray,  # (N,) 每个时间步的收盘价
        rl_cfg: RLConfig,
        seed: Optional[int] = None,
    ):
        self.cfg = rl_cfg
        self.labels = dataset_dict['labels']           # (N, 3)
        self.predictions = predictions                  # (N, 3) 上涨概率
        self.close_prices = close_prices                 # (N,)

        self.num_steps = len(self.labels)

        # 构建动作空间: 0=skip, 1~N= (horizon, direction, bet_size)
        self._build_action_space()

        # 当前状态
        self.initial_balance = rl_cfg.initial_balance
        self.balance = self.initial_balance
        self.current_idx = 0
        self.last_returns = np.zeros(rl_cfg.state_returns_window, dtype=np.float32)
        self.episode_returns = []   # 记录每步盈亏
        self.episode_actions = []   # 记录每步动作
        self.episode_idx = 0

        # 预计算收益率
        self._precompute_returns()

        if seed is not None:
            np.random.seed(seed)

        # 状态维度
        self.state_dim = (
            3 +                   # 三个周期的上涨概率
            3 +                   # 三个周期的置信度
            1 +                   # 余额比例
            rl_cfg.state_returns_window +  # 最近收益率
            1 +                   # 波动率
            1                     # 已用步数比例
        )

    def _precompute_returns(self):
        """预计算每个时间步的收益率 (10min粒度)"""
        prices = self.close_prices
        rets = np.diff(np.log(prices))
        # 在前面补0, 保持长度一致
        self.returns = np.concatenate([[0.0], rets])

    def _build_action_space(self):
        """构建动作空间映射"""
        self.action_list = []   # list of (horizon, direction, bet_size)
        for h in [10, 30, 60]:
            for d in [0, 1]:  # 0=做多, 1=做空
                for b in self.cfg.bet_sizes:
                    self.action_list.append((h, d, b))
        self.n_actions = 1 + len(self.action_list)  # +1 for skip

    def decode_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        """解码动作索引 -> (horizon_minutes, direction, bet_amount)"""
        if action_idx == 0:
            return None, None, 0.0
        h, d, b = self.action_list[action_idx - 1]
        return h, d, b

    def _compute_state(self, idx: int) -> np.ndarray:
        """计算当前状态向量"""
        probs = self.predictions[idx]                 # (3,)
        confs = np.abs(probs - 0.5) * 2.0             # (3,)
        balance_ratio = self.balance / self.initial_balance

        # 最近收益率
        start = max(0, idx - self.cfg.state_returns_window + 1)
        recent_rets = self.returns[start:idx + 1]
        if len(recent_rets) < self.cfg.state_returns_window:
            recent_rets = np.pad(recent_rets,
                                 (self.cfg.state_returns_window - len(recent_rets), 0),
                                 'constant')

        # 波动率: 最近20步收益率标准差
        vol_window = min(20, idx + 1)
        vol = np.std(self.returns[idx + 1 - vol_window:idx + 1]) if vol_window > 1 else 0.0

        # 进度
        progress = idx / max(self.num_steps - 1, 1)

        return np.concatenate([
            probs,                # 3
            confs,                # 3
            [balance_ratio],      # 1
            recent_rets,          # state_returns_window
            [vol],                # 1
            [progress],           # 1
        ]).astype(np.float32)

    def reset(self) -> np.ndarray:
        """重置环境到随机起始位置"""
        # 从数据的前 60%~90% 中随机选择起始点 (留下足够验证数据)
        max_start = int(self.num_steps * 0.6)
        min_start = max(0, self.cfg.state_returns_window + 5)
        self.current_idx = np.random.randint(min_start, max_start)
        self.balance = self.initial_balance
        self.episode_returns = []
        self.episode_actions = []
        self.episode_idx += 1
        return self._compute_state(self.current_idx)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步交易"""
        horizon, direction, bet = self.decode_action(action_idx)
        idx = self.current_idx

        # 确保本金不超过当前余额
        if horizon is not None:
            bet = min(bet, max(0, self.balance))

        if horizon is None or bet <= 0:
            # skip 或无资金
            reward = 0.0
            info = {'action': 'skip', 'bet': 0, 'pnl': 0.0, 'balance': self.balance}
        else:
            h_idx = {10: 0, 30: 1, 60: 2}[horizon]
            actual_up = self.labels[idx, h_idx]  # 1=涨, 0=跌

            # 根据方向和实际涨跌计算盈亏
            if (direction == 0 and actual_up == 1) or (direction == 1 and actual_up == 0):
                # 预测正确: 获得收益
                reward = bet * self.cfg.profit_rates[horizon]
                result = 'win'
            else:
                # 预测错误: 亏损本金
                reward = -bet
                result = 'lose'

            self.balance += reward
            info = {
                'action': f'{horizon}min_{"long" if direction == 0 else "short"}',
                'bet': bet,
                'pnl': reward,
                'balance': self.balance,
                'result': result,
                'prob': float(self.predictions[idx, h_idx]),
                'prob_dir': 'up' if self.predictions[idx, h_idx] > 0.5 else 'down',
                'actual': 'up' if actual_up == 1 else 'down',
            }

        self.episode_returns.append(reward)
        self.episode_actions.append(action_idx)
        self.current_idx += 1

        # 检查是否结束
        done = (
            self.current_idx >= self.num_steps - 1 or
            self.balance < 1.0 or
            self.balance > self.initial_balance * 20
        )

        if done:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            next_state = self._compute_state(self.current_idx)

        return next_state, reward, done, info

    def get_episode_stats(self) -> dict:
        """返回当前回合的统计信息"""
        returns = np.array(self.episode_returns)
        win_mask = returns > 0
        loss_mask = returns < 0
        return {
            'total_return': self.balance - self.initial_balance,
            'final_balance': self.balance,
            'return_rate': (self.balance - self.initial_balance) / self.initial_balance,
            'win_rate': win_mask.sum() / max(len(returns), 1),
            'total_trades': (returns != 0).sum(),
            'avg_win': returns[win_mask].mean() if win_mask.any() else 0.0,
            'avg_loss': returns[loss_mask].mean() if loss_mask.any() else 0.0,
            'max_drawdown': self._compute_drawdown(),
            'sharpe': self._compute_sharpe(returns),
        }

    def _compute_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.episode_returns:
            return 0.0
        cumulative = self.initial_balance + np.cumsum(self.episode_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return float(drawdown.max()) if len(drawdown) > 0 else 0.0

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
        """计算夏普比率 (年化, 假设10min粒度=52560个交易期/年)"""
        if len(returns) < 2 or np.std(returns) < 1e-8:
            return 0.0
        periods_per_year = 365 * 24 * 6  # 10分钟
        excess = returns.mean() - rf / periods_per_year
        return excess / returns.std() * np.sqrt(periods_per_year)


# =============================================================================
# 模型: Dueling DQN
# =============================================================================

class DuelingQNetwork(nn.Module):
    """Dueling DQN 网络: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# =============================================================================
# 经验回放缓冲区
# =============================================================================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# DQN 智能体
# =============================================================================

class DQNAgent:
    """Dueling DQN 智能体"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: RLConfig,
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim
        self.step_counter = 0

        # 网络
        self.q_network = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.95)
        self.memory = ReplayBuffer(cfg.buffer_capacity)

        # 探索率
        self.epsilon = cfg.epsilon_start

        # 训练统计
        self.loss_history = []

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """选择动作 (epsilon-贪心)"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax().item())

    def update(self) -> Optional[float]:
        """从经验回放中采样并更新网络"""
        if len(self.memory) < self.cfg.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前 Q 值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN: 用当前网络选择动作, 目标网络评估
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.cfg.gamma * next_q * (1 - dones)

        # Huber Loss
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 5.0)
        self.optimizer.step()
        self.scheduler.step()

        self.step_counter += 1
        self.loss_history.append(loss.item())

        # 更新目标网络
        if self.step_counter % self.cfg.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 衰减探索率
        if self.epsilon > self.cfg.epsilon_end:
            self.epsilon *= self.cfg.epsilon_decay

        return loss.item()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_counter': self.step_counter,
            'loss_history': self.loss_history[-1000:],
        }, path)
        logger.info(f"RL智能体已保存 -> {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.cfg.epsilon_end)
        self.step_counter = checkpoint.get('step_counter', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        logger.info(f"RL智能体已加载 <- {path} "
                     f"(epsilon={self.epsilon:.4f}, step={self.step_counter})")


# =============================================================================
# 数据准备
# =============================================================================

def _detect_device():
    """检测可用设备"""
    use_cuda = False
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 7:
                use_cuda = True
        except Exception:
            pass
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    return device


def _load_model_and_horizons(device):
    """加载预训练LNN模型 (参数同 predict.py)"""
    from predict import load_model as _pmodel
    return _pmodel(device)


def _load_norm_stats():
    """加载标准化参数"""
    import pickle
    path = os.path.join(config.CHECKPOINT_DIR, 'feature_norm_stats.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"标准化参数文件不存在: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def _normalize_with_stats(tf_seqs_raw, ctx_raw, norm_data):
    """使用训练时的统计量进行 Z-Score 标准化"""
    stats = norm_data['stats']
    periods = norm_data['periods']
    tf_seqs = {}
    for p in periods:
        s = stats[p]
        tf_seqs[p] = (tf_seqs_raw[p] - s['mean']) / s['std']
    cs = stats['context']
    ctx = (ctx_raw - cs['mean']) / cs['std']
    return tf_seqs, ctx


def prepare_rl_dataset() -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """获取数据并构建RL训练数据集 (仅BTC-USDT, 保证价格序列对齐)

    Returns:
        train_data: (X_dict, X_ctx, y) 训练集
        val_data:   (X_dict, X_ctx, y) 验证集
        test_data:  (X_dict, X_ctx, y) 测试集
        close_prices: np.array 所有样本对应的10min收盘价
    """
    logger.info("=" * 60)
    logger.info("开始获取BTC-USDT多周期数据并构建RL数据集...")
    logger.info("=" * 60)

    fetcher = HuobiDataFetcher()

    # 仅使用BTC-USDT进行RL训练 (保证价格序列与样本一一对应)
    symbol = config.SYMBOLS[0]
    logger.info(f"使用币种: {symbol}")

    # 获取多周期数据
    timeframe_data = fetcher.fetch_multi_timeframe()

    # 转换为DataFrames (所有周期纳入tf_dfs, target_df固定用5min)
    tf_dfs = {}
    target_df = None
    for period, data in timeframe_data.items():
        df = fetcher.get_dataframe(data)
        tf_dfs[period] = df
        if period == '5min':
            target_df = df

    label_source_df = fetcher.get_dataframe(timeframe_data['5min'])

    # 构建数据集
    X_dict, X_ctx, y = build_multi_tf_dataset(
        tf_dfs, target_df, label_source_df=label_source_df)

    if len(y) == 0:
        raise RuntimeError("数据集为空!")

    # 收盘价序列 (与数据集样本一一对应: 数据从开头开始, 有效样本紧接前段)
    close_prices = target_df['close'].values.astype(np.float32)
    close_prices = close_prices[:len(y)]

    # 切分
    train_data, val_data, test_data = split_multi_tf_dataset(
        X_dict, X_ctx, y, train_ratio=0.6, val_ratio=0.2)

    # 标准化
    train_data, val_data, test_data = normalize_datasets(train_data, val_data, test_data)

    logger.info(f"数据集构建完成: 训练={len(train_data[2])}, "
                 f"验证={len(val_data[2])}, 测试={len(test_data[2])}")

    return train_data, val_data, test_data, close_prices



# =============================================================================
# 预计算 LNN 预测
# =============================================================================

def precompute_predictions(
    model: MultiTimeframeLNN,
    dataset_tuple: tuple,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """在数据集上批量计算模型预测的上涨概率

    Args:
        model: 预训练LNN模型 (eval模式)
        dataset_tuple: (X_dict, X_ctx, y)

    Returns:
        predictions: (N, 3) 上涨概率 [p_10m, p_30m, p_60m]
    """
    X_dict, X_ctx, _ = dataset_tuple
    model.eval()

    n = X_ctx.shape[0]
    all_probs = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch_seqs = {
                p: torch.FloatTensor(
                    normalize_sequence_samplewise({p: X_dict[p][i:end]})[p]
                ).to(device)
                for p in X_dict.keys()
            }
            batch_ctx = torch.FloatTensor(X_ctx[i:end]).to(device)

            logits = model(batch_seqs, batch_ctx)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# =============================================================================
# 训练
# =============================================================================

def train_rl(rl_cfg: RLConfig):
    """强化学习训练主流程"""
    logger.info("=" * 60)
    logger.info("RL 交易智能体训练开始")
    logger.info(f"  回合数: {rl_cfg.episodes}")
    logger.info(f"  动作空间: {1 + 3 * 2 * len(rl_cfg.bet_sizes)} 个动作")
    logger.info(f"  投入档位: {rl_cfg.bet_sizes}")
    logger.info("=" * 60)

    device = _detect_device()

    # 1. 加载LNN模型
    logger.info("加载预训练LNN模型...")
    model, horizons = _load_model_and_horizons(device)
    model.eval()
    logger.info(f"LNN模型加载成功, 预测窗口: {horizons}")

    # 2. 准备数据集
    train_data, val_data, test_data, close_prices = prepare_rl_dataset()
    n_train = len(train_data[2])
    n_val = len(val_data[2])
    n_test = len(test_data[2])
    logger.info(f"数据集: 训练={n_train}, 验证={n_val}, 测试={n_test}")

    # 3. 预计算LNN预测
    logger.info("预计算LNN模型预测 (训练集)...")
    train_preds = precompute_predictions(model, train_data, device)
    logger.info(f"训练集预测完成: {train_preds.shape}")

    logger.info("预计算LNN模型预测 (验证集)...")
    val_preds = precompute_predictions(model, val_data, device)

    # 4. 创建环境和智能体
    train_env = TradingEnv(
        dataset_dict={'labels': train_data[2], 'tf_sequences': train_data[0], 'context': train_data[1]},
        predictions=train_preds,
        close_prices=close_prices[:n_train],
        rl_cfg=rl_cfg,
    )

    val_env = TradingEnv(
        dataset_dict={'labels': val_data[2], 'tf_sequences': val_data[0], 'context': val_data[1]},
        predictions=val_preds,
        close_prices=close_prices[n_train:n_train + n_val] if len(close_prices) >= n_train + n_val else close_prices[n_train:],
        rl_cfg=rl_cfg,
    )

    agent = DQNAgent(train_env.state_dim, train_env.n_actions, rl_cfg, device)

    # 5. 加载已有checkpoint (增量训练)
    checkpoint_path = os.path.join(rl_cfg.save_dir, "rl_latest.pth")
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        logger.info("加载已有RL checkpoint, 继续增量训练")

    # 6. 训练循环
    logger.info("开始RL训练...")
    best_val_return = -float('inf')
    best_agent_path = os.path.join(rl_cfg.save_dir, "rl_best.pth")
    total_steps = 0

    metrics_history = []

    for episode in range(1, rl_cfg.episodes + 1):
        state = train_env.reset()
        episode_reward = 0.0
        episode_losses = []

        for step in range(rl_cfg.max_steps_per_episode):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = train_env.step(action)

            # 存入经验回放
            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # 训练
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            if done:
                break

        # 回合统计
        stats = train_env.get_episode_stats()
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        stats['avg_loss'] = avg_loss
        stats['episode'] = episode
        stats['steps'] = total_steps
        stats['epsilon'] = agent.epsilon

        metrics_history.append(stats)

        # 定期评估
        if episode % rl_cfg.eval_interval == 0 or episode == 1:
            val_stats = evaluate(agent, val_env, num_episodes=5)
            stats['val_return'] = val_stats['avg_return_rate']
            stats['val_win_rate'] = val_stats['avg_win_rate']

            logger.info(
                f"回合 {episode:>4d}/{rl_cfg.episodes} | "
                f"收益: {stats['total_return']:>+8.1f} ({stats['return_rate']*100:+5.1f}%) | "
                f"胜率: {stats['win_rate']:>5.1%} | "
                f"交易数: {stats['total_trades']:>4d} | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"验证收益率: {val_stats['avg_return_rate']*100:+5.1f}% | "
                f"验证胜率: {val_stats['avg_win_rate']:>5.1%}"
            )

            # 保存最佳模型
            if val_stats['avg_return_rate'] > best_val_return:
                best_val_return = val_stats['avg_return_rate']
                agent.save(best_agent_path)
                logger.info(f"!!! 新最佳模型: 验证收益率={val_stats['avg_return_rate']*100:+5.1f}%")

        else:
            logger.info(
                f"回合 {episode:>4d}/{rl_cfg.episodes} | "
                f"收益: {stats['total_return']:>+8.1f} ({stats['return_rate']*100:+5.1f}%) | "
                f"胜率: {stats['win_rate']:>5.1%} | "
                f"交易数: {stats['total_trades']:>4d} | "
                f"夏普: {stats['sharpe']:>5.1f} | "
                f"最大回撤: {stats['max_drawdown']:>5.1%}"
            )

        # 定期保存 (每50回合)
        if episode % 50 == 0:
            agent.save(os.path.join(rl_cfg.save_dir, f"rl_checkpoint_ep{episode}.pth"))

    # 保存最终模型
    agent.save(os.path.join(rl_cfg.save_dir, "rl_final.pth"))

    # 保存训练指标
    metrics_df = pd.DataFrame(metrics_history)
    metrics_path = os.path.join(rl_cfg.save_dir, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"训练指标已保存 -> {metrics_path}")

    # 加载最佳模型进行测试
    if os.path.exists(best_agent_path):
        agent.load(best_agent_path)

    logger.info("=" * 60)
    logger.info("在测试集上评估最佳模型...")
    logger.info("=" * 60)

    # 预计算测试集预测
    logger.info("预计算LNN模型预测 (测试集)...")
    test_preds = precompute_predictions(model, test_data, device)

    test_env = TradingEnv(
        dataset_dict={'labels': test_data[2], 'tf_sequences': test_data[0], 'context': test_data[1]},
        predictions=test_preds,
        close_prices=close_prices[n_train + n_val:] if len(close_prices) >= n_train + n_val else close_prices[-len(test_data[2]):],
        rl_cfg=rl_cfg,
    )

    test_results = evaluate(agent, test_env, num_episodes=20, render=True)

    logger.info("=" * 60)
    logger.info("测试集评估结果:")
    logger.info(f"  平均收益率: {test_results['avg_return_rate']*100:+5.2f}%")
    logger.info(f"  平均胜率:   {test_results['avg_win_rate']:.2%}")
    logger.info(f"  平均交易数: {test_results['avg_trades']:.0f}")
    logger.info(f"  平均夏普:   {test_results['avg_sharpe']:.2f}")
    logger.info(f"  平均最大回撤: {test_results['avg_drawdown']:.2%}")
    logger.info("=" * 60)

    return agent, test_results


# =============================================================================
# 评估
# =============================================================================

def evaluate(
    agent: DQNAgent,
    env: TradingEnv,
    num_episodes: int = 10,
    render: bool = False,
) -> dict:
    """在环境中评估智能体性能"""
    agent.q_network.eval()
    results = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            state = next_state

        stats = env.get_episode_stats()
        results.append(stats)

    agent.q_network.train()

    # 汇总
    avg_stats = {}
    keys = ['total_return', 'return_rate', 'win_rate', 'total_trades',
            'avg_win', 'avg_loss', 'max_drawdown', 'sharpe']
    for k in keys:
        vals = [r[k] for r in results]
        avg_stats[f'avg_{k}'] = float(np.mean(vals))
        avg_stats[f'std_{k}'] = float(np.std(vals))

    avg_stats['num_episodes'] = num_episodes

    if render:
        logger.info(f"\n评估结果 ({num_episodes} 回合):")
        for i, r in enumerate(results):
            logger.info(
                f"  回合 {i+1}: 收益率={r['return_rate']*100:+6.2f}% | "
                f"交易数={r['total_trades']:>4d} | "
                f"胜率={r['win_rate']:.1%} | "
                f"夏普={r['sharpe']:.1f} | "
                f"最大回撤={r['max_drawdown']:.1%}"
            )

    return avg_stats


# =============================================================================
# 实时交易
# =============================================================================

def live_trade(rl_cfg: RLConfig):
    """使用训练好的RL智能体进行实时交易"""
    device = _detect_device()

    # 加载LNN模型
    logger.info("加载预训练LNN模型...")
    model, horizons = _load_model_and_horizons(device)
    model.eval()

    # 加载标准化参数
    norm_data = _load_norm_stats()

    # 创建RL环境(只需状态维度)
    # 先构建一个最小的环境来获取state_dim
    dummy_env = TradingEnv(
        dataset_dict={'labels': np.zeros((10, 3)), 'tf_sequences': {}, 'context': np.zeros((10, 6))},
        predictions=np.zeros((10, 3)),
        close_prices=np.ones(10) * 50000,
        rl_cfg=rl_cfg,
    )

    # 创建并加载RL智能体
    agent = DQNAgent(dummy_env.state_dim, dummy_env.n_actions, rl_cfg, device)
    best_path = os.path.join(rl_cfg.save_dir, "rl_best.pth")
    latest_path = os.path.join(rl_cfg.save_dir, "rl_latest.pth")
    load_path = best_path if os.path.exists(best_path) else latest_path

    if os.path.exists(load_path):
        agent.load(load_path)
        agent.q_network.eval()
    else:
        logger.warning(f"未找到RL模型 ({load_path}), 使用随机策略")
        agent.epsilon = 0.0

    # 交易循环
    logger.info("开始实时交易...")
    logger.info(f"  初始资金: {rl_cfg.initial_balance} USDT")
    logger.info(f"  投入范围: {rl_cfg.min_bet} ~ {rl_cfg.max_bet} USDT")
    logger.info(f"  收益率: {rl_cfg.profit_rates}")
    logger.info("=" * 60)

    fetcher = HuobiDataFetcher()
    balance = rl_cfg.initial_balance
    trade_history = []

    try:
        while True:
            try:
                # 1. 获取最新多周期数据
                logger.info("获取最新K线数据...")
                timeframe_data = fetcher.fetch_multi_timeframe()

                # 2. 计算特征和模型预测
                tf_seqs_raw, ctx_raw, target_df = _prepare_live_features(
                    timeframe_data, fetcher)
                tf_seqs_raw = normalize_sequence_samplewise(tf_seqs_raw)
                tf_seqs_norm, ctx_norm = _normalize_with_stats(tf_seqs_raw, ctx_raw, norm_data)

                # 3. LNN模型推理
                tf_tensors = {
                    p: torch.from_numpy(v.copy()).float().to(device)
                    for p, v in tf_seqs_norm.items()
                }
                ctx_tensor = torch.from_numpy(ctx_norm.copy()).float().to(device)

                with torch.no_grad():
                    logits = model(tf_tensors, ctx_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]  # (3,)

                current_price = float(target_df['close'].iloc[-1])
                current_time = pd.Timestamp.now()

                logger.info(f"当前时间: {current_time}")
                logger.info(f"当前价格: {current_price:.2f} USDT")
                logger.info(f"余额: {balance:.2f} USDT")

                # 4. 构建状态
                confs = np.abs(probs - 0.5) * 2.0
                # 简化: 用最近5个10min return (无历史时用0填充)
                fake_rets = np.zeros(rl_cfg.state_returns_window, dtype=np.float32)
                state = np.concatenate([
                    probs, confs,
                    [balance / rl_cfg.initial_balance],
                    fake_rets, [0.0], [0.0],
                ]).astype(np.float32)

                # 5. RL智能体决策
                action = agent.select_action(state, eval_mode=True)
                horizon, direction, bet = dummy_env.decode_action(action)

                # 6. 输出决策
                print()
                print("=" * 60)
                print(f"  LNN预测:")
                for i, h in enumerate([10, 30, 60]):
                    d = "涨" if probs[i] > 0.5 else "跌"
                    c = confs[i]
                    print(f"    [{h:>2}分钟] {d} 概率={probs[i]:.4f} 置信度={c:.4f}")

                if horizon is None:
                    print(f"  RL决策: 不交易 (skip)")
                else:
                    dir_str = "做多" if direction == 0 else "做空"
                    risk_pct = bet / balance * 100 if balance > 0 else 0
                    potential_profit = bet * rl_cfg.profit_rates[horizon]
                    print(f"  RL决策: {horizon}分钟 {dir_str}")
                    print(f"  投入本金: {bet:.1f} USDT ({risk_pct:.1f}% 余额)")
                    print(f"  潜在收益: +{potential_profit:.1f} USDT | 潜在亏损: -{bet:.1f} USDT")

                print("=" * 60)
                print()

                # 7. 记录交易
                trade_record = {
                    'time': str(current_time),
                    'price': current_price,
                    'balance': balance,
                    'probs_10m': float(probs[0]),
                    'probs_30m': float(probs[1]),
                    'probs_60m': float(probs[2]),
                    'action': horizon,
                    'direction': direction,
                    'bet': bet,
                }
                trade_history.append(trade_record)

                # 记录到文件
                history_path = os.path.join(rl_cfg.save_dir, "trade_history.json")
                with open(history_path, 'w') as f:
                    json.dump(trade_history[-500:], f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"交易循环出错: {e}", exc_info=True)

            # 等待下一次预测
            logger.info("等待10分钟...")
            time.sleep(600)  # 10分钟

    except KeyboardInterrupt:
        logger.info("实时交易已停止")
        # 保存交易历史
        history_path = os.path.join(rl_cfg.save_dir, "trade_history_final.json")
        with open(history_path, 'w') as f:
            json.dump(trade_history, f, ensure_ascii=False, indent=2)
        logger.info(f"交易历史已保存 -> {history_path}")


def _prepare_live_features(timeframe_data, fetcher):
    """为实时交易准备LNN模型输入特征 (同 predict.py)"""
    periods = list(config.TIMEFRAMES.keys())

    tf_dfs = {}
    for period in periods:
        df = fetcher.get_dataframe(timeframe_data[period])
        df = compute_all_features(df)
        df = df.dropna(subset=SEQ_FEATURE_COLS)
        tf_dfs[period] = df

    # 上下文特征 (4hour)
    ctx_df = compute_context_features(tf_dfs['4hour'])
    ctx = ctx_df[CONTEXT_FEATURE_COLS].values[-1:].astype(np.float32)

    now_ts = int(pd.Timestamp.now().timestamp())
    target_ts = np.array([now_ts])

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

    target_df = tf_dfs['5min']
    return tf_seqs, ctx, target_df


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LNN 强化学习交易智能体")
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'eval', 'trade'],
        help='运行模式: train=训练, eval=评估, trade=实时交易')
    parser.add_argument(
        '--episodes', type=int, default=300,
        help='训练回合数')
    parser.add_argument(
        '--balance', type=float, default=1000.0,
        help='初始本金 (USDT)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子')
    parser.add_argument(
        '--eval_episodes', type=int, default=20,
        help='评估回合数')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rl_cfg = RLConfig(
        initial_balance=args.balance,
        episodes=args.episodes,
    )

    if args.mode == 'train':
        train_rl(rl_cfg)

    elif args.mode == 'eval':
        device = _detect_device()
        model, _ = _load_model_and_horizons(device)
        _, _, test_data, close_prices = prepare_rl_dataset()
        test_preds = precompute_predictions(model, test_data, device)

        env = TradingEnv(
            dataset_dict={'labels': test_data[2],
                          'tf_sequences': test_data[0],
                          'context': test_data[1]},
            predictions=test_preds,
            close_prices=close_prices[-len(test_data[2]):],
            rl_cfg=rl_cfg,
        )
        agent = DQNAgent(env.state_dim, env.n_actions, rl_cfg, device)
        best_path = os.path.join(rl_cfg.save_dir, "rl_best.pth")
        if os.path.exists(best_path):
            agent.load(best_path)

        evaluate(agent, env, num_episodes=args.eval_episodes, render=True)

    elif args.mode == 'trade':
        live_trade(rl_cfg)


if __name__ == "__main__":
    main()
