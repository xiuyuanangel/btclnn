# GPU训练优化说明

## 已完成的优化

### 1. DataLoader多线程优化
```python
# 自动根据CPU核心数设置工作进程
num_workers = max(2, cpu_count // 2)

# 启用pin_memory加速CPU到GPU传输
pin_memory = torch.cuda.is_available()

# 启用persistent_workers保持工作进程持久化
persistent_workers = True
```

### 2. cuDNN自动优化
```python
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
```
- 自动寻找最优卷积/循环层算法
- 针对固定输入尺寸优化

### 3. 异步数据传输
```python
# 使用non_blocking=True实现异步数据传输
tf_seqs = {p: v.to(device, non_blocking=True) for p, v in tf_seqs.items()}
ctx = ctx.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)
```

### 4. 自动混合精度训练(AMP)
```python
# 仅在GPU可用时启用
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler()

# 前向传播使用自动混合精度
with torch.cuda.amp.autocast():
    outputs, _ = model(tf_seqs, ctx)
    loss = criterion(outputs, labels)

# 缩放梯度并更新
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 5. 训练监控增强
- 显示训练速度 (samples/sec)
- 显示GPU内存使用情况
- 显示每轮耗时

## 预期效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| GPU利用率 | ~30% | 80-95% |
| GPU内存占用 | ~300MB | 1-3GB |
| 训练速度 | 慢 | 提升2-3倍 |
| CPU瓶颈 | 严重 | 缓解 |

## 使用建议

### 如果GPU内存不足
1. 减小batch_size（config.py中修改）
2. 关闭混合精度（将use_amp设为False）
3. 减少num_workers数量

### 如果仍然GPU利用率低
1. 检查数据预处理是否在GPU上进行
2. 考虑增大batch_size以充分利用GPU并行能力
3. 检查是否有其他进程占用GPU

### 多GPU训练
如需使用多GPU训练，可以添加：
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 注意事项

1. **Windows系统**: `num_workers > 0` 时需要在`if __name__ == '__main__':`中运行
2. **内存充足**: 确保系统内存足够（建议16GB+）
3. **SSD硬盘**: 使用SSD存储数据可进一步提升加载速度
