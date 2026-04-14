"""液态神经网络(LNN)模型实现

基于 Liquid Time-Constant Networks (LTC) 论文的核心思想:
- 连续时间动态: dh/dt = f(h, u, t)
- 可学习的时间常数(每个神经元独立的时间尺度)
- 输入依赖的动态调制(网络的"液态"特性)

多周期融合架构:
  每个时间周期(1min/5min/15min/60min/4hour)拥有独立的LTC编码器,
  各编码器输出拼接后经融合层产生最终预测。

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

    离散化: 4阶Runge-Kutta方法 (dt=1), 比Euler更稳定精确
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
        self.layer_norm = nn.LayerNorm(hidden_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
        nn.init.orthogonal_(self.W_rec.weight, gain=0.5)
        for module in self.gate_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _deriv(self, x, h):
        """计算 dh/dt (ODE 右端项)"""
        tau = torch.nn.functional.softplus(self.tau) + 1.0
        gate = torch.sigmoid(self.gate_net(torch.cat([x, h], dim=-1)))
        return -h / tau + self.W_in(x) + gate * self.W_rec(h)

    def forward(self, x, h):
        # 4阶 Runge-Kutta 离散化 (dt=1), 比Euler更稳定精确
        dt = torch.tensor(1.0, device=h.device)
        k1 = self._deriv(x, h)
        k2 = self._deriv(x, h + 0.5 * dt * k1)
        k3 = self._deriv(x, h + 0.5 * dt * k2)
        k4 = self._deriv(x, h + dt * k3)
        h_new = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.tanh(self.layer_norm(h_new))


class TimeframeEncoder(nn.Module):
    """单周期LTC编码器

    将一个时间周期的序列编码为固定维度的向量表示。
    每个周期有独立的多层LTC单元, 处理原生分辨率的数据。
    输出: 最后隐藏状态 + 所有层均值池化 (2 * hidden_size)
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
        self._init_weights()

    def _init_weights(self):
        for cell in self.cells:
            cell._init_weights()

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            h: (batch, 2 * hidden_size) 最后隐藏状态 + 层均值
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
                inp = hidden[i]

        # 多尺度输出: 最后状态 + 各层均值(捕获全局趋势)
        last_h = hidden[-1]
        mean_h = torch.stack(hidden, dim=0).mean(dim=0)
        return torch.cat([last_h, mean_h], dim=-1)


class CrossTimeframeAttention(nn.Module):
    """跨周期注意力机制

    让不同时间尺度的编码器输出相互交互，学习周期间的依赖关系。
    使用多头注意力机制捕获多尺度特征间的复杂关联。
    """

    def __init__(self, hidden_size, num_timeframes, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_timeframes = num_timeframes
        self.num_heads = num_heads
        self.head_dim = hidden_size * 2 // num_heads  # encoder输出是2*hidden_size

        assert (hidden_size * 2) % num_heads == 0, "hidden_size*2必须能被num_heads整除"

        # 多头注意力投影
        self.q_proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.k_proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.v_proj = nn.Linear(hidden_size * 2, hidden_size * 2)

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 可学习的周期位置编码
        self.timeframe_embedding = nn.Parameter(torch.randn(num_timeframes, hidden_size * 2) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)
        for module in self.out_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, encoded_list):
        """
        Args:
            encoded_list: list of (batch, 2*hidden_size)，各周期编码器输出

        Returns:
            output: (batch, num_timeframes, 2*hidden_size) 交互后的特征
            attn_weights: (batch, num_heads, num_timeframes, num_timeframes) 注意力权重
        """
        batch_size = encoded_list[0].size(0)
        num_tf = len(encoded_list)

        # 堆叠成 (batch, num_timeframes, 2*hidden_size)
        x = torch.stack(encoded_list, dim=1)  # (batch, num_tf, 2*hidden)

        # 添加周期位置编码
        x = x + self.timeframe_embedding[:num_tf].unsqueeze(0)

        # 多头投影
        Q = self.q_proj(x).view(batch_size, num_tf, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, num_tf, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, num_tf, self.num_heads, self.head_dim).transpose(1, 2)
        # Q,K,V: (batch, num_heads, num_tf, head_dim)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, num_tf, num_tf)

        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, num_tf, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_tf, -1)

        # 输出投影
        output = self.out_proj(attn_output)

        return output, attn_weights


class CrossTimeframeGating(nn.Module):
    """跨周期门控机制

    学习如何根据其他周期的信息来调制当前周期的表示。
    实现类似LSTM的门控，但跨时间尺度进行。
    """

    def __init__(self, hidden_size, num_timeframes):
        super().__init__()
        self.hidden_size = hidden_size * 2
        self.num_timeframes = num_timeframes

        # 全局信息聚合
        self.global_pool = nn.Sequential(
            nn.Linear(self.hidden_size * num_timeframes, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
        )

        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.global_pool:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.gate_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, encoded_list):
        """
        Args:
            encoded_list: list of (batch, 2*hidden_size)

        Returns:
            gated_list: list of (batch, 2*hidden_size) 门控后的特征
        """
        batch_size = encoded_list[0].size(0)

        # 全局信息: 拼接所有周期
        global_info = torch.cat(encoded_list, dim=-1)  # (batch, num_tf * 2*hidden)
        global_vec = self.global_pool(global_info)  # (batch, 2*hidden)

        # 对每个周期应用门控
        gated_list = []
        for enc in encoded_list:
            # 拼接局部和全局信息
            combined = torch.cat([enc, global_vec], dim=-1)  # (batch, 4*hidden)
            gate = self.gate_net(combined)  # (batch, 2*hidden)
            # 门控融合: 保留部分原始信息，融合部分全局信息
            gated = enc * gate + global_vec * (1 - gate)
            gated_list.append(gated)

        return gated_list


class MultiTimeframeLNN(nn.Module):
    """多周期融合液态神经网络 (增强版，含跨周期交互)

    架构:
        各周期序列 → 独立LTC编码器 → 跨周期注意力/门控 → 自适应融合 → 分类头

    新增跨周期交互机制:
        1. CrossTimeframeAttention: 多头注意力让周期间直接交互
        2. CrossTimeframeGating: 门控机制选择性融合全局信息
        3. 分层融合: 先交互再加权，捕获复杂的多尺度依赖

    每个周期(1min/5min/15min/60min/4hour/1day)有独立的编码器处理原生分辨率数据,
    通过跨周期交互捕获从微观结构到宏观趋势的层次化市场动态。
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
        use_cross_gating=True,
        cross_attention_heads=4,
    ):
        """
        Args:
            timeframe_configs: dict of {period: {'seq_length': int, 'feature_size': int}}
            context_feature_size: 上下文特征维度
            hidden_size: 隐藏层大小
            num_layers: 每个编码器的LTC层数
            dropout: Dropout比率
            output_size: 输出维度(默认1=二元分类)
            use_cross_attention: 是否使用跨周期注意力
            use_cross_gating: 是否使用跨周期门控
            cross_attention_heads: 跨周期注意力头数
        """
        super().__init__()
        self.hidden_size = hidden_size or config.HIDDEN_SIZE
        self.num_layers = num_layers or config.NUM_LAYERS
        self.dropout_rate = dropout or config.DROPOUT
        self.use_cross_attention = use_cross_attention
        self.use_cross_gating = use_cross_gating

        # 每个周期独立的LTC编码器
        self.encoders = nn.ModuleDict({
            period: TimeframeEncoder(
                input_size=tf_cfg['feature_size'],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
            for period, tf_cfg in timeframe_configs.items()
        })

        self.num_timeframes = len(self.encoders)
        encoder_output_size = self.hidden_size * 2  # 每个编码器输出维度

        # 跨周期交互模块
        if self.use_cross_attention:
            self.cross_attention = CrossTimeframeAttention(
                hidden_size=self.hidden_size,
                num_timeframes=self.num_timeframes,
                num_heads=cross_attention_heads,
                dropout=self.dropout_rate,
            )

        if self.use_cross_gating:
            self.cross_gating = CrossTimeframeGating(
                hidden_size=self.hidden_size,
                num_timeframes=self.num_timeframes,
            )

        # 周期注意力机制: 自适应学习各时间帧的重要性
        self.tf_attention = nn.Sequential(
            nn.Linear(encoder_output_size, encoder_output_size // 4),
            nn.Tanh(),
            nn.Linear(encoder_output_size // 4, 1),
        )

        # 加权融合: 各编码器注意力加权求和 + 上下文特征
        fusion_input_size = encoder_output_size + context_feature_size

        # 融合层: 将加权编码器输出和上下文特征融合
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        # 分类头 (输出logits, 不含Sigmoid, 配合BCEWithLogitsLoss使用)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, output_size),
        )
        self._init_fusion_and_classifier()

    def _init_fusion_and_classifier(self):
        for module in self.tf_attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
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
            attn_weights: dict 可选，包含注意力权重用于可视化
        """
        # 各周期编码
        encoded = []
        for period, encoder in self.encoders.items():
            h = encoder(tf_sequences[period])
            encoded.append(h)

        # 跨周期交互
        attn_weights_dict = {}

        if self.use_cross_attention and hasattr(self, 'cross_attention'):
            # 多头注意力交互: (batch, num_tf, 2*hidden)
            cross_attn_out, attn_weights = self.cross_attention(encoded)
            attn_weights_dict['cross_attention'] = attn_weights
            # 将交互后的特征与原始特征残差连接
            encoded = [cross_attn_out[:, i, :] + enc for i, enc in enumerate(encoded)]

        if self.use_cross_gating and hasattr(self, 'cross_gating'):
            # 门控融合全局信息
            encoded = self.cross_gating(encoded)

        # 注意力加权融合: 自适应学习各周期重要性
        scores_list = [self.tf_attention(enc) for enc in encoded]  # 各 (batch, 1)
        scores = torch.cat(scores_list, dim=-1)                     # (batch, num_tf)
        attn_weights = torch.softmax(scores, dim=-1)               # 归一化权重
        attn_weights_dict['timeframe_attention'] = attn_weights

        # 加权求和: (batch, 2*hidden_size)
        weighted_sum = torch.zeros_like(encoded[0])
        for i, enc in enumerate(encoded):
            weighted_sum = weighted_sum + attn_weights[:, i:i+1] * enc

        # 拼接加权结果 + 上下文特征
        fused_input = torch.cat([weighted_sum, context_features], dim=-1)
        fused = self.fusion(fused_input)
        output = self.classifier(fused).squeeze(-1)

        return output, attn_weights_dict


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
        # 分类头 (输出logits, 不含Sigmoid, 配合BCEWithLogitsLoss使用)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, output_size),
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
                hidden[i] = torch.clamp(hidden[i], -10, 10)
                inp = hidden[i]
        return self.classifier(hidden[-1]).squeeze(-1)


def count_parameters(model):
    """统计模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    from features import SEQ_FEATURE_COLS, CONTEXT_FEATURE_COLS
    from config import TIMEFRAMES

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
    output, attn_weights = model(tf_seqs, ctx_feat)
    print(f"输出: {output.shape}, 值: {output.detach().numpy()}")

    # 打印注意力权重信息
    if 'cross_attention' in attn_weights:
        print(f"跨周期注意力权重: {attn_weights['cross_attention'].shape}")
        print(f"  平均注意力分布: {attn_weights['cross_attention'].mean(dim=(0,1))}")
    if 'timeframe_attention' in attn_weights:
        print(f"周期注意力权重: {attn_weights['timeframe_attention'].shape}")
        print(f"  平均权重: {attn_weights['timeframe_attention'].mean(dim=0)}")
