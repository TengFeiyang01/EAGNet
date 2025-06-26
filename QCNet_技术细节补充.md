# QCNet 技术细节补充文档

## 数学公式详解

### 1. 注意力机制公式

#### 1.1 基础多头注意力
```latex
\begin{align}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{align}
```

#### 1.2 位置增强注意力
```latex
\begin{align}
K_{enhanced} &= K + \text{Pos\_Emb}(r) \cdot W_r^K \\
V_{enhanced} &= V + \text{Pos\_Emb}(r) \cdot W_r^V \\
r &= [d_{rel}, \theta_{rel}, \Delta t, \Delta\phi]
\end{align}
```

其中：
- $d_{rel}$: 相对距离
- $\theta_{rel}$: 相对角度  
- $\Delta t$: 时间差
- $\Delta\phi$: 航向差

### 2. 傅里叶嵌入公式

#### 2.1 频率编码
```latex
\begin{align}
\text{FourierEmb}(x) &= \text{MLP}\left(\left[\begin{array}{c}
\cos(2\pi f_1 x) \\
\sin(2\pi f_1 x) \\
\vdots \\
\cos(2\pi f_B x) \\
\sin(2\pi f_B x) \\
x
\end{array}\right]\right)
\end{align}
```

#### 2.2 可学习频率
```latex
f_i = \text{Embedding}(\text{feature\_dim})_i, \quad i = 1, \ldots, B
```

### 3. 损失函数详解

#### 3.1 拉普拉斯分布NLL损失
```latex
\begin{align}
\mathcal{L}_{\text{Laplace}} &= -\log p(y|\mu, b) \\
&= \log(2b) + \frac{|y - \mu|}{b}
\end{align}
```

#### 3.2 冯·米塞斯分布NLL损失 (角度)
```latex
\begin{align}
\mathcal{L}_{\text{VonMises}} &= -\log p(\theta|\mu, \kappa) \\
&= -\kappa \cos(\theta - \mu) + \log(2\pi I_0(\kappa))
\end{align}
```

#### 3.3 混合分布损失
```latex
\begin{align}
\mathcal{L}_{\text{Mixture}} &= -\log \sum_{k=1}^K \pi_k p_k(y|\theta_k) \\
\text{其中} \quad &\sum_{k=1}^K \pi_k = 1, \quad \pi_k \geq 0
\end{align}
```

### 4. 评估指标公式

#### 4.1 最小平均位移误差 (minADE)
```latex
\text{minADE} = \min_{k \in \{1,\ldots,K\}} \frac{1}{T} \sum_{t=1}^T \|\hat{y}_t^{(k)} - y_t\|_2
```

#### 4.2 最小最终位移误差 (minFDE)
```latex
\text{minFDE} = \min_{k \in \{1,\ldots,K\}} \|\hat{y}_T^{(k)} - y_T\|_2
```

#### 4.3 错失率 (Miss Rate)
```latex
\text{MR} = \mathbb{P}\left(\min_{k \in \{1,\ldots,K\}} \|\hat{y}_T^{(k)} - y_T\|_2 > \tau\right)
```

其中 $\tau$ 通常设为2米。

## 实现细节

### 1. 数据预处理流程

#### 1.1 坐标系转换
```python
def transform_to_agent_frame(positions, reference_pos, reference_heading):
    """转换到智能体局部坐标系"""
    # 平移
    relative_pos = positions - reference_pos
    
    # 旋转
    cos_h, sin_h = np.cos(reference_heading), np.sin(reference_heading)
    rotation_matrix = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
    
    return np.dot(relative_pos, rotation_matrix.T)
```

#### 1.2 轨迹特征提取
```python
def extract_motion_features(trajectory, heading):
    """提取运动特征"""
    # 位移向量
    displacement = np.diff(trajectory, axis=0)
    
    # 速度大小
    speed = np.linalg.norm(displacement, axis=1)
    
    # 运动方向与航向的夹角
    motion_angle = np.arctan2(displacement[:, 1], displacement[:, 0])
    relative_angle = motion_angle - heading[1:]
    
    return speed, relative_angle
```

### 2. 图构建策略

#### 2.1 半径图构建
```python
def build_radius_graph(positions, radius, batch=None):
    """基于半径构建图"""
    from torch_cluster import radius_graph
    
    edge_index = radius_graph(
        x=positions,
        r=radius,
        batch=batch,
        loop=False,  # 不包含自环
        max_num_neighbors=300
    )
    return edge_index
```

#### 2.2 时序图构建
```python
def build_temporal_graph(valid_mask, time_span):
    """构建时序连接图"""
    num_agents, num_steps = valid_mask.shape
    edge_list = []
    
    for i in range(num_agents):
        for t1 in range(num_steps):
            if not valid_mask[i, t1]:
                continue
            for t2 in range(t1 + 1, min(t1 + time_span + 1, num_steps)):
                if not valid_mask[i, t2]:
                    continue
                # 添加时序边
                edge_list.append([i * num_steps + t1, i * num_steps + t2])
    
    return torch.tensor(edge_list).T
```

### 3. 训练技巧

#### 3.1 学习率调度
```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """余弦退火学习率调度"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
```

#### 3.2 梯度裁剪
```python
def clip_gradients(model, max_norm=1.0):
    """梯度裁剪防止梯度爆炸"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

#### 3.3 权重初始化
```python
def weight_init(module):
    """权重初始化策略"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
```

### 4. 内存优化技巧

#### 4.1 梯度检查点
```python
def forward_with_checkpointing(self, x):
    """使用梯度检查点节省内存"""
    return torch.utils.checkpoint.checkpoint(self.expensive_function, x)
```

#### 4.2 稀疏注意力
```python
def sparse_attention(query, key, value, edge_index):
    """稀疏注意力实现"""
    # 只计算有边连接的节点对之间的注意力
    row, col = edge_index
    
    # 计算注意力分数
    scores = (query[col] * key[row]).sum(dim=-1)
    scores = scores / math.sqrt(query.size(-1))
    
    # 稀疏softmax
    attn_weights = softmax(scores, col)
    
    # 聚合
    output = scatter_add(attn_weights.unsqueeze(-1) * value[row], col, dim=0)
    return output
```

### 5. 调试与可视化

#### 5.1 注意力权重可视化
```python
def visualize_attention_weights(attention_weights, positions, save_path=None):
    """可视化注意力权重"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制节点
    ax.scatter(positions[:, 0], positions[:, 1], s=100, c='blue', alpha=0.7)
    
    # 绘制注意力连接
    for i, weights in enumerate(attention_weights):
        for j, weight in enumerate(weights):
            if weight > 0.1:  # 只显示强连接
                ax.plot([positions[i, 0], positions[j, 0]], 
                       [positions[i, 1], positions[j, 1]], 
                       'r-', alpha=weight, linewidth=weight*5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Attention Weights Visualization')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

#### 5.2 轨迹预测可视化
```python
def plot_trajectory_prediction(history, prediction, ground_truth, map_info=None):
    """绘制轨迹预测结果"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制地图（如果有）
    if map_info is not None:
        for polygon in map_info['polygons']:
            ax.plot(polygon[:, 0], polygon[:, 1], 'k-', alpha=0.3)
    
    # 绘制历史轨迹
    ax.plot(history[:, 0], history[:, 1], 'b-', linewidth=3, label='History')
    ax.scatter(history[-1, 0], history[-1, 1], c='blue', s=100, marker='o')
    
    # 绘制真实未来轨迹
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', linewidth=3, label='Ground Truth')
    ax.scatter(ground_truth[-1, 0], ground_truth[-1, 1], c='green', s=100, marker='s')
    
    # 绘制多模态预测
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, pred_mode in enumerate(prediction):
        ax.plot(pred_mode[:, 0], pred_mode[:, 1], '--', 
               color=colors[i % len(colors)], linewidth=2, 
               alpha=0.8, label=f'Prediction {i+1}')
        ax.scatter(pred_mode[-1, 0], pred_mode[-1, 1], 
                  c=colors[i % len(colors)], s=80, marker='^')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Multi-Modal Trajectory Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()
```

### 6. 性能分析工具

#### 6.1 模型复杂度分析
```python
def analyze_model_complexity(model, input_sample):
    """分析模型复杂度"""
    from thop import profile
    
    flops, params = profile(model, inputs=(input_sample,))
    print(f"FLOPs: {flops / 1e9:.2f}G")
    print(f"Parameters: {params / 1e6:.2f}M")
    
    # 内存使用分析
    def get_memory_usage():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    
    memory_before = get_memory_usage()
    output = model(input_sample)
    memory_after = get_memory_usage()
    
    print(f"Memory usage: {memory_after - memory_before:.2f}MB")
```

#### 6.2 推理速度测试
```python
def benchmark_inference_speed(model, input_sample, num_runs=100):
    """测试推理速度"""
    import time
    
    model.eval()
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_sample)
    
    # 计时
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_sample)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"FPS: {fps:.2f}")
```

## 常见问题与解决方案

### 1. 训练不稳定

**问题**: 损失函数震荡，难以收敛
**解决方案**:
- 降低学习率
- 增加warmup步数
- 使用梯度裁剪
- 检查数据预处理是否正确

### 2. 内存不足

**问题**: GPU内存溢出
**解决方案**:
- 减少batch size
- 使用梯度累积
- 启用混合精度训练
- 使用梯度检查点

### 3. 预测结果不合理

**问题**: 预测轨迹偏离道路或不符合物理约束
**解决方案**:
- 检查坐标系转换
- 增强地图信息编码
- 添加物理约束损失
- 调整模型架构

### 4. 推理速度慢

**问题**: 实时应用中推理速度不够快
**解决方案**:
- 模型量化
- 知识蒸馏
- 优化图结构
- 使用TensorRT等推理引擎

这份技术细节文档提供了QCNet实现的深层技术细节，帮助您更好地理解和优化模型性能。 