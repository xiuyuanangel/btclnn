# Loss NaN 问题修复说明

## 问题诊断

训练过程中出现大量 `loss=nan` 和 `output range=[nan, nan]` 的警告，导致模型无法正常训练。

### 根本原因

1. **LTCCell 数值不稳定**
   - `tau` 参数初始化为 `torch.randn(hidden_size) * 0.1`，可能产生负值或过小的值
   - RK4 积分过程中没有数值裁剪，导致梯度爆炸
   - 隐藏状态可能累积到极端值

2. **注意力机制数值溢出**
   - CrossTimeframeAttention 中的 scores 范围过大，可能导致 softmax 溢出
   - 缺少数值稳定性保护（如减去最大值）

3. **权重初始化不当**
   - 多处使用 `gain=0.5` 的 xavier 初始化，可能导致梯度消失
   - 影响模型的训练稳定性

4. **学习率过高**
   - 初始学习率 0.001 对于复杂的 LTC 网络可能过大
   - 容易导致参数更新过大，产生数值不稳定

5. **特征标准化不够鲁棒**
   - 对异常值处理不足
   - epsilon 值过小（1e-8）

## 修复方案

### 1. LTCCell 数值稳定性增强

**文件**: `lnn_model.py`

#### 改进 tau 参数初始化
```python
# 修复前
self.tau = nn.Parameter(torch.randn(hidden_size) * 0.1)

# 修复后
self.tau = nn.Parameter(torch.ones(hidden_size) * 2.0 + torch.rand(hidden_size) * 0.5)
```
- 使用正值初始化（2.0-2.5 范围）
- 避免负值和过小值导致的除零问题

#### 改进权重初始化
```python
# 修复前
nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
nn.init.orthogonal_(self.W_rec.weight, gain=0.5)

# 修复后
nn.init.xavier_uniform_(self.W_in.weight, gain=1.0)
nn.init.orthogonal_(self.W_rec.weight, gain=0.8)
```
- 增大 gain 值，避免梯度消失

#### 增强 _deriv 方法的数值稳定性
```python
def _deriv(self, x, h):
    # 确保tau为正且有下界
    tau = torch.nn.functional.softplus(self.tau) + 0.5
    # 裁剪h避免数值溢出
    h_clamped = torch.clamp(h, -100.0, 100.0)
    gate_input = torch.cat([x, h_clamped], dim=-1)
    gate = torch.sigmoid(self.gate_net(gate_input))
    dh = -h_clamped / tau + self.W_in(x) + gate * self.W_rec(h_clamped)
    # 裁剪导数防止梯度爆炸
    return torch.clamp(dh, -50.0, 50.0)
```

#### 增强 forward 方法的数值保护
```python
def forward(self, x, h):
    dt = 1.0
    # 裁剪输入h
    h = torch.clamp(h, -100.0, 100.0)
    
    k1 = self._deriv(x, h)
    k2 = self._deriv(x, h + 0.5 * dt * k1)
    k3 = self._deriv(x, h + 0.5 * dt * k2)
    k4 = self._deriv(x, h + dt * k3)

    h_new = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # 在layer_norm前裁剪
    h_new = torch.clamp(h_new, -100.0, 100.0)
    # 检测NaN并替换为零
    h_new = torch.where(torch.isnan(h_new), torch.zeros_like(h_new), h_new)
    return torch.tanh(self.layer_norm(h_new))
```

### 2. CrossTimeframeAttention 数值稳定性

**文件**: `lnn_model.py`

```python
# 修复前
scores = torch.clamp(scores, min=-50.0, max=50.0)
attn_weights = torch.softmax(scores, dim=-1)

# 修复后
scores = torch.clamp(scores, min=-20.0, max=20.0)
# 数值稳定性: 减去最大值
scores = scores - scores.max(dim=-1, keepdim=True)[0]
attn_weights = torch.softmax(scores, dim=-1)
# 确保权重不是NaN
attn_weights = torch.where(torch.isnan(attn_weights), 
                            torch.ones_like(attn_weights) / num_tf, 
                            attn_weights)
```

### 3. 训练过程增强

**文件**: `train.py`

#### 更严格的梯度裁剪和检查
```python
# 检测 NaN/Inf 并跳过异常batch
if torch.isnan(loss) or torch.isinf(loss):
    logger.warning(f"检测到异常loss={loss.item():.6f}, 跳过此batch")
    continue

loss.backward()

# 更严格的梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# 检查梯度是否正常
grad_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        grad_norm += param_norm.item() ** 2
grad_norm = grad_norm ** 0.5

if torch.isnan(torch.tensor(grad_norm)) or grad_norm > 100.0:
    logger.warning(f"梯度异常 grad_norm={grad_norm:.4f}, 跳过参数更新")
    optimizer.zero_grad()
    continue

optimizer.step()
```

### 4. 降低学习率

**文件**: `config.py`

```python
# 修复前
LEARNING_RATE = 0.001

# 修复后
LEARNING_RATE = 0.0005  # 降低以提高数值稳定性
```

### 5. 增强特征标准化

**文件**: `features.py`

#### 改进 fit 方法
```python
def fit(self, X_dict_train, X_ctx_train):
    for period, X in X_dict_train.items():
        flat = X.reshape(-1, X.shape[-1])
        # 移除异常值后计算统计量
        flat_clipped = np.clip(flat, np.percentile(flat, 1, axis=0), 
                               np.percentile(flat, 99, axis=0))
        self.seq_means[period] = flat_clipped.mean(axis=0)
        self.seq_stds[period] = flat_clipped.std(axis=0) + 1e-6  # 增加epsilon

    X_ctx_clipped = np.clip(X_ctx_train, 
                            np.percentile(X_ctx_train, 1, axis=0),
                            np.percentile(X_ctx_train, 99, axis=0))
    self.ctx_mean = X_ctx_clipped.mean(axis=0)
    self.ctx_std = X_ctx_clipped.std(axis=0) + 1e-6
    return self
```

#### 改进 transform 方法
```python
def transform(self, X_dict, X_ctx):
    X_dict_norm = {}
    for period, X in X_dict.items():
        normalized = (X - self.seq_means[period]) / self.seq_stds[period]
        # 替换 Inf/NaN 并裁剪到合理范围
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = np.clip(normalized, -10.0, 10.0)
        X_dict_norm[period] = normalized
    
    X_ctx_norm = (X_ctx - self.ctx_mean) / self.ctx_std
    X_ctx_norm = np.nan_to_num(X_ctx_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_ctx_norm = np.clip(X_ctx_norm, -10.0, 10.0)
    return X_dict_norm, X_ctx_norm.astype(np.float32)
```

## 修复效果预期

1. **消除 NaN loss**：通过多层数值保护，彻底避免 NaN 的产生和传播
2. **提高训练稳定性**：降低学习率和更严格的梯度裁剪，使训练过程更平滑
3. **改善收敛性**：更合理的权重初始化，有助于模型更快收敛
4. **增强鲁棒性**：特征标准化的改进，使模型对异常数据更鲁棒

## 验证方法

1. 运行训练脚本，观察是否还有 NaN 警告
2. 检查训练日志中的 loss 曲线是否平滑下降
3. 验证集 loss 应该能够正常计算（不再是 nan）
4. 模型应该能够完成完整的训练周期

## 注意事项

1. 由于修改了模型初始化逻辑，建议删除旧的 checkpoint 重新训练
2. 如果仍有问题，可以进一步降低学习率到 0.0001
3. 可以考虑减小 BATCH_SIZE 以降低梯度方差
4. 建议监控训练过程中的梯度范数，确保在合理范围内（< 10.0）

## 修复文件清单

- ✅ `lnn_model.py` - 核心模型数值稳定性修复
- ✅ `train.py` - 训练过程增强
- ✅ `config.py` - 学习率调整
- ✅ `features.py` - 特征标准化改进
