"""液态神经网络(LNN)模型实现

基于 Liquid Time-Constant Networks (LTC) 论文的核心思想:
|- 连续时间动态: dh/dt = f(h, u, t)
|- 可学习的时间常数(每个神经元独立的时间尺度)
|- 输入依赖的动态调制(网络的"液态"特性)

多周期融合架构:
  每个时间周期(5min/15min/60min/4hour/1day)拥有独立的LTC编码器,
  通过跨周期注意力机制(Cross-TF Attention)让各周期相互影响,
  各编码器输出经融合层产生最终预测。

参考论文:
  Hasani et al., "Liquid Time-Constant Networks", 2021
"""

import torch
import torch.nn as nn
import math

import config


class LTCCell(nn.Module):
    """Liquid Time-Constant (液态时间常数) 循环单元

    核心微分方程:
        dh/dt = -h / tau + W_in * u + gate(h, u) * W_rec * h

    离散化: Euler方法, dh_new = h + dh/dt * dt (dt=1)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.tau = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_in.weight, gain=1.0)
        nn.init.orthogonal_(self.W_rec.weight, gain=1.0)
        for module in self.gate_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, h):
        tau = torch.nn.functional.softplus(self.tau) + 1.0
        gate = torch.sigmoid(self.gate_net(torch.cat([x, h], dim=-1)))
        dhdt = -h / tau + self.W_in(x) + gate * self.W_rec(h)
        h_new = h + dhdt
        return torch.tanh(h_new)


class TimeframeEncoder(nn.Module):
    """单周期LTC编码器

    将一个时间周期的序列编码为固定维度的向量表示。
    每个周期有独立的多层LTC单元, 处理原生分辨率的数据。
    层间加入LayerNorm稳定梯度流动。
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([
            LTCCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
            )
            for i in range(num_layers)
        ])
        # 层间LayerNorm: 防止深层梯度消失/爆炸
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        self._init_weights()

    def _init_weights(self):
        for cell in self.cells:
            cell._init_weights()

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            h: (batch, hidden_size) 最终隐藏状态
        """
        batch_size, seq_len = x.size(0), x.size(1)
        device = x.device

        hidden = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(len(self.cells))
        ]

        for t in range(seq_len):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(inp, hidden[i])
                # 层间归一化: 稳定后续层的输入分布
                hidden[i] = self.layer_norms[i](hidden[i])
                inp = hidden[i]

        return hidden[-1]


class CrossTimeframeAttention(nn.Module):
    """跨周期注意力机制(Cross-Timeframe Attention)

    让每个时间周期能够主动关注其他周期的信息,
    学习不同时间尺度之间的相互影响关系。

    例如: 5min微观结构可以参考1day宏观趋势来调整判断,
          宏观趋势也可以从微观结构的异常信号中获得预警。

    架构:
      各周期编码输出 → Stack为序列 → Multi-Head Self-Attention → 残差连接+LayerNorm → Unstack
    """

    def __init__(self, hidden_size, num_timeframes, num_heads=4, dropout=0.1):
        """
        Args:
            hidden_size: 每个周期的隐藏维度
            num_timeframes: 周期数量
            num_heads: 注意力头数(需整除hidden_size)
            dropout: 注意力dropout率
        """
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size({hidden_size})必须能被num_heads({num_heads})整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q/K/V投影(所有周期共享, 让注意力模式可迁移)
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # 输出投影
        self.W_o = nn.Linear(hidden_size, hidden_size)

        # 层归一化与残差
        self.norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for W in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(W.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, encoded_list):
        """
        Args:
            encoded_list: list of (batch, hidden_size), 各周期编码器的输出,
                         列表顺序与config.TIMEFRAMES.keys()一致

        Returns:
            enhanced_list: list of (batch, hidden_size), 融合了跨周期信息的增强表示
        """
        # (batch, num_tf, hidden_size)
        x = torch.stack(encoded_list, dim=1)
        batch_size, num_tf, _ = x.shape

        # 投影Q/K/V
        Q = self.W_q(x)  # (B, N, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # 多头reshape: (B, N, H, D/H) -> (B, H, N, D/H)
        Q = Q.view(batch_size, num_tf, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_tf, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_tf, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: (B, H, N, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 加权求和: (B, H, N, D/H) -> (B, N, H, D/H) -> (B, N, D)
        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_tf, self.hidden_size)

        # 输出投影 + 残差 + LayerNorm
        output = self.W_o(context)
        output = self.norm(x + output)

        # Unstack回列表
        return [output[:, i, :] for i in range(num_tf)]


class MultiTimeframeLNN(nn.Module):
    """多周期融合液态神经网络

    架构:
        各周期序列 → 独立LTC编码器 → 跨周期注意力(Cross-TF Attention)
        → 拼接(+上下文特征) → 融合层 → 分类头 → 涨/跌概率

    每个周期(5min/15min/60min/4hour/1day)有独立的编码器处理原生分辨率数据,
    通过Cross-TF Attention学习不同时间尺度之间的相互影响。
    """

    def __init__(
        self,
        timeframe_configs,
        context_feature_size,
        hidden_size=None,
        num_layers=None,
        dropout=None,
        output_size=1,
        use_cross_attention=True,
        cross_attn_heads=4,
    ):
        """
        Args:
            timeframe_configs: dict of {period: {'seq_length': int, 'feature_size': int}}
            context_feature_size: 上下文特征维度
            hidden_size: 隐藏层大小
            num_layers: 每个编码器的LTC层数
            dropout: Dropout比率
            output_size: 输出维度(默认1=二元分类)
            use_cross_attention: 是否启用跨周期注意力(默认True)
            cross_attn_heads: 跨周期注意力的头数
        """
        super().__init__()
        self.hidden_size = hidden_size or config.HIDDEN_SIZE
        self.num_layers = num_layers or config.NUM_LAYERS
        self.dropout_rate = dropout or config.DROPOUT
        self.use_cross_attention = use_cross_attention

        # 周期名称列表(保持确定顺序)
        self.period_names = list(timeframe_configs.keys())
        num_tf = len(timeframe_configs)

        # 每个周期独立的LTC编码器
        self.encoders = nn.ModuleDict({
            period: TimeframeEncoder(
                input_size=tf_cfg['feature_size'],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
            for period, tf_cfg in timeframe_configs.items()
        })

        # 跨周期注意力模块(可选)
        if self.use_cross_attention and num_tf > 1:
            self.cross_attn = CrossTimeframeAttention(
                hidden_size=self.hidden_size,
                num_timeframes=num_tf,
                num_heads=cross_attn_heads,
                dropout=self.dropout_rate,
            )

        fusion_input_size = num_tf * self.hidden_size + context_feature_size

        # 融合层: 将所有编码器输出和上下文特征融合
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, output_size),
            nn.Sigmoid(),
        )
        self._init_fusion_and_classifier()

    def _init_fusion_and_classifier(self):
        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tf_sequences, context_features):
        """
        Args:
            tf_sequences: dict of {period: (batch, seq_len, feature_size)}
            context_features: (batch, context_size)

        Returns:
            output: (batch,) 预测涨的概率
        """
        # 1. 各周期独立编码(保持固定顺序)
        encoded = []
        for period in self.period_names:
            h = self.encoders[period](tf_sequences[period])
            encoded.append(h)

        # 2. 跨周期注意力交互(可选)
        if self.use_cross_attention and hasattr(self, 'cross_attn'):
            encoded = self.cross_attn(encoded)

        # 3. 拼接所有编码器输出 + 上下文特征
        fused_input = torch.cat(encoded + [context_features], dim=-1)
        fused = self.fusion(fused_input)
        return self.classifier(fused).squeeze(-1)


class LiquidNeuralNetwork(nn.Module):
    """单周期液态神经网络(兼容旧接口)"""

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
        self.cells = nn.ModuleList([
            LTCCell(
                total_input_size if i == 0 else self.hidden_size,
                self.hidden_size,
            )
            for i in range(self.num_layers)
        ])
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
        batch_size, seq_len = seq_features.size(0), seq_features.size(1)
        device = seq_features.device
        ctx = context_features.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([seq_features, ctx], dim=-1)
        hidden = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]
        for t in range(seq_len):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(inp, hidden[i])
                inp = hidden[i]
        return self.classifier(hidden[-1]).squeeze(-1)


def count_parameters(model):
    """统计模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    from features import SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS, TIMEFRAMES

    feat_size = len(SEQ_FEATURE_COLS)
    ctx_size = len(CONTEXT_FEATURE_COLS)

    # 多周期模型测试
    tf_configs = {
        p: {'seq_length': cfg['seq_length'], 'feature_size': feat_size}
        for p, cfg in TIMEFRAMES.items()
    }
    model = MultiTimeframeLNN(tf_configs, ctx_size)
    total, trainable = count_parameters(model)
    print(f"多周期模型参数: 总计 {total:,}, 可训练 {trainable:,}")

    batch_size = 4
    tf_seqs = {
        p: torch.randn(batch_size, cfg['seq_length'], feat_size)
        for p, cfg in TIMEFRAMES.items()
    }
    ctx_feat = torch.randn(batch_size, ctx_size)
    output = model(tf_seqs, ctx_feat)
    print(f"输出: {output.shape}, 值: {output.detach().numpy()}")
