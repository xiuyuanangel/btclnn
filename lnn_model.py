"""液态神经网络(LNN)模型实现

基于 Liquid Time-Constant Networks (LTC) 论文的核心思想:
- 连续时间动态: dh/dt = f(h, u, t)
- 可学习的时间常数(每个神经元独立的时间尺度)
- 输入依赖的动态调制(网络的"液态"特性 — 动态随输入变化而改变)

参考论文:
  Hasani et al., "Liquid Time-Constant Networks", 2021
"""

import torch
import torch.nn as nn

import config


class LTCCell(nn.Module):
    """Liquid Time-Constant (液态时间常数) 循环单元

    核心微分方程:
        dh/dt = -h / tau + W_in * u + gate(h, u) * W_rec * h

    其中:
        tau  : 可学习的时间常数(每个神经元独立，通过softplus确保正值)
        gate : 输入依赖的门控函数 — 这是"液态"的关键:
               网络的动态特性会根据输入数据实时调整，
               类似液态物质在不同容器中呈现不同形态

    离散化: 使用 Euler 方法, dh_new = h + dh/dt * dt (dt=1)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 可学习的时间常数 (初始化后通过softplus + 1保证为正)
        self.tau = nn.Parameter(torch.randn(hidden_size) * 0.1)

        # 输入权重
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)

        # 循环权重
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)

        # 输入依赖的门控网络 (液态特性的核心)
        # 输入: 拼接 [当前输入 x, 隐藏状态 h]
        # 输出: 0~1 之间的门控值
        self.gate_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
        nn.init.orthogonal_(self.W_rec.weight, gain=0.5)
        for module in self.gate_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, h):
        """
        Args:
            x: 输入 (batch, input_size)
            h: 隐藏状态 (batch, hidden_size)

        Returns:
            h_new: 更新后的隐藏状态 (batch, hidden_size)
        """
        # 时间常数: softplus(tau) + 1, 保证 tau > 1
        tau = torch.nn.functional.softplus(self.tau) + 1.0

        # 液态门控: 基于当前输入和隐藏状态动态调整
        gate = torch.sigmoid(self.gate_net(torch.cat([x, h], dim=-1)))

        # 连续时间动态
        # dh/dt = -h/tau + W_in*x + gate * W_rec*h
        dhdt = (
            -h / tau +
            self.W_in(x) +
            gate * self.W_rec(h)
        )

        # Euler 离散化
        h_new = h + dhdt

        return torch.tanh(h_new)


class LiquidNeuralNetwork(nn.Module):
    """多层液态神经网络

    架构:
        [输入序列 + 上下文特征] -> LTC Layer 1 -> LTC Layer 2 -> ... -> 分类头 -> 涨/跌概率

    上下文特征(60天统计摘要)在每个时间步与序列特征拼接，
    使模型能够同时感知短期模式和长期市场环境。
    """

    def __init__(
        self,
        seq_feature_size,
        context_feature_size,
        hidden_size=None,
        num_layers=None,
        dropout=None,
        output_size=1,
    ):
        super().__init__()
        self.hidden_size = hidden_size or config.HIDDEN_SIZE
        self.num_layers = num_layers or config.NUM_LAYERS
        self.dropout_rate = dropout or config.DROPOUT

        total_input_size = seq_feature_size + context_feature_size

        # 多层 LTC 单元
        self.cells = nn.ModuleList([
            LTCCell(
                total_input_size if i == 0 else self.hidden_size,
                self.hidden_size,
            )
            for i in range(self.num_layers)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, output_size),
            nn.Sigmoid(),
        )
        self._init_classifier()

    def _init_classifier(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, seq_features, context_features):
        """
        Args:
            seq_features:      (batch, seq_len, seq_feature_size)
            context_features:  (batch, context_feature_size)

        Returns:
            output: (batch,) 预测涨的概率
        """
        batch_size, seq_len = seq_features.size(0), seq_features.size(1)
        device = seq_features.device

        # 将上下文特征广播到每个时间步
        ctx = context_features.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([seq_features, ctx], dim=-1)  # (batch, seq_len, total_input)

        # 初始化隐藏状态
        hidden = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

        # 逐时间步处理序列
        for t in range(seq_len):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(inp, hidden[i])
                inp = hidden[i]

        # 使用最终隐藏状态分类
        return self.classifier(hidden[-1]).squeeze(-1)


def count_parameters(model):
    """统计模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    from features import SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS

    seq_size = len(SEQ_FEATURE_COLS)
    ctx_size = len(CONTEXT_FEATURE_COLS)

    model = LiquidNeuralNetwork(seq_size, ctx_size)
    total, trainable = count_parameters(model)
    print(f"模型参数: 总计 {total:,}, 可训练 {trainable:,}")

    # 前向传播测试
    batch_size = 4
    seq_feat = torch.randn(batch_size, config.SEQ_LENGTH, seq_size)
    ctx_feat = torch.randn(batch_size, ctx_size)

    output = model(seq_feat, ctx_feat)
    print(f"输入: seq {seq_feat.shape}, ctx {ctx_feat.shape}")
    print(f"输出: {output.shape}, 值: {output.detach().numpy()}")
