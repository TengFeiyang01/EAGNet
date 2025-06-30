# QCNet 多智能体轨迹预测框架学习文档

## 目录
1. [项目概述](#项目概述)
2. [核心架构](#核心架构)
3. [数据流分析](#数据流分析)
4. [模块详解](#模块详解)
5. [训练与评估](#训练与评估)
6. [从零开始学习指南](#从零开始学习指南)
7. [代码实践](#代码实践)

---

## 1. 项目概述

### 1.1 什么是QCNet？

QCNet (Query-Centric Network) 是一个用于**多智能体轨迹预测**的深度学习框架，专门设计用于预测自动驾驶场景中车辆、行人等智能体的未来运动轨迹。

### 1.2 核心特点

- **查询中心设计**：使用DETR-like的查询机制进行轨迹预测
- **多模态预测**：为每个智能体预测多种可能的未来轨迹
- **场景理解**：同时考虑地图信息和智能体交互
- **时空不变性**：具有旋转-平移不变性和时间平移不变性

### 1.3 应用场景

- 自动驾驶车辆路径规划
- 交通流量预测
- 机器人导航
- 智能交通系统

---

## 2. 核心架构

### 2.1 整体架构图

```
输入数据 (HeteroData)
├── 智能体历史轨迹 (agent)
├── 地图多边形 (map_polygon) 
└── 地图点 (map_point)
        ↓
┌─────────────────────────────────┐
│          QCNet 主网络            │
├─────────────────────────────────┤
│  ┌─────────────────────────────┐ │
│  │        编码器模块            │ │
│  │  ┌─────────────────────────┐ │ │
│  │  │     地图编码器          │ │ │
│  │  │  • 点到多边形注意力    │ │ │
│  │  │  • 多边形间注意力      │ │ │
│  │  └─────────────────────────┘ │ │
│  │  ┌─────────────────────────┐ │ │
│  │  │     智能体编码器        │ │ │
│  │  │  • 时序注意力          │ │ │
│  │  │  • 地图-智能体注意力   │ │ │
│  │  │  • 智能体间注意力      │ │ │
│  │  └─────────────────────────┘ │ │
│  └─────────────────────────────┘ │
│           ↓                     │
│  ┌─────────────────────────────┐ │
│  │        解码器模块            │ │
│  │  • 模式嵌入                │ │
│  │  • 查询-场景交互            │ │
│  │  • 两阶段预测              │ │
│  │    - 提议阶段 (Propose)    │ │
│  │    - 精化阶段 (Refine)     │ │
│  └─────────────────────────────┘ │
└─────────────────────────────────┘
        ↓
输出预测
├── 多模态轨迹 (K=6种可能)
├── 概率分布 (位置 + 航向)
└── 模式概率 (π)
```

### 2.2 关键设计原理

#### 2.2.1 查询中心机制
```latex
\text{Query} = \text{Mode Embedding} + \text{Agent Context}
```

每个智能体的每种预测模式都对应一个查询向量，通过注意力机制与场景特征交互。

#### 2.2.2 两阶段预测
```latex
\begin{align}
\text{Stage 1 (Propose):} \quad &\hat{Y}^{prop} = f_{propose}(Q, X_{scene}) \\
\text{Stage 2 (Refine):} \quad &\hat{Y}^{ref} = f_{refine}(Q, X_{scene}, \hat{Y}^{prop})
\end{align}
```

---

## 3. 数据流分析

### 3.1 输入数据结构

```python
HeteroData {
    'agent': {
        'position': [N_agents, T_hist, 3],      # 历史位置 (x,y,z)
        'heading': [N_agents, T_hist],          # 历史航向角
        'velocity': [N_agents, T_hist, 3],      # 历史速度
        'valid_mask': [N_agents, T_hist],       # 有效性掩码
        'predict_mask': [N_agents, T_total],    # 预测掩码
        'type': [N_agents],                     # 智能体类型
        'target': [N_agents, T_future, 3]      # 真实未来轨迹
    },
    'map_polygon': {
        'position': [N_poly, 3],                # 多边形中心位置
        'orientation': [N_poly],                # 多边形方向
        'type': [N_poly],                       # 多边形类型 (车道等)
        'is_intersection': [N_poly]             # 是否为交叉口
    },
    'map_point': {
        'position': [N_points, 3],              # 地图点位置
        'type': [N_points],                     # 点类型 (车道线等)
        'side': [N_points]                      # 左/右侧标识
    }
}
```

### 3.2 数据流转换

```
原始数据 → 特征嵌入 → 注意力交互 → 查询预测 → 输出分布
```

---

## 4. 模块详解

### 4.1 地图编码器 (QCNetMapEncoder)

#### 4.1.1 功能
- 编码静态地图信息
- 建立地图元素间的空间关系

#### 4.1.2 核心组件

```python
class QCNetMapEncoder(nn.Module):
    def __init__(self):
        # 特征嵌入
        self.x_pt_emb = FourierEmbedding(...)    # 点特征嵌入
        self.x_pl_emb = FourierEmbedding(...)    # 多边形特征嵌入
        
        # 关系嵌入  
        self.r_pt2pl_emb = FourierEmbedding(...) # 点到多边形关系
        self.r_pl2pl_emb = FourierEmbedding(...) # 多边形间关系
        
        # 注意力层
        self.pt2pl_layers = nn.ModuleList([...]) # 点到多边形注意力
        self.pl2pl_layers = nn.ModuleList([...]) # 多边形间注意力
```

#### 4.1.3 处理流程

```
地图点特征 ──┐
            ├─→ 点到多边形注意力 ──→ 多边形特征更新
多边形特征 ──┘                   ↓
                              多边形间注意力 ──→ 最终地图表示
```

### 4.2 智能体编码器 (QCNetAgentEncoder)

#### 4.2.1 功能
- 编码智能体历史轨迹
- 建立智能体与地图、智能体间的交互关系

#### 4.2.2 特征提取

```python
# 运动特征计算
motion_vector = position[t] - position[t-1]  # 运动向量
speed = ||velocity||                         # 速度大小
heading_change = heading[t] - heading[t-1]   # 航向变化

# 相对特征 (旋转不变性)
relative_distance = ||pos_i - pos_j||
relative_angle = angle_between(heading_i, pos_i - pos_j)
```

#### 4.2.3 注意力机制

```latex
\begin{align}
\text{时序注意力:} \quad &H_t = \text{Attention}(H_{t-1}, H_{1:t-1}) \\
\text{地图注意力:} \quad &H_a = \text{Attention}(H_t, H_{map}) \\
\text{智能体注意力:} \quad &H_{final} = \text{Attention}(H_a, H_{other\_agents})
\end{align}
```

### 4.3 解码器 (QCNetDecoder)

#### 4.3.1 查询初始化

```python
# 模式嵌入
mode_queries = self.mode_emb.weight  # [K, hidden_dim]
mode_queries = mode_queries.repeat(N_agents, 1)  # [N_agents*K, hidden_dim]

# 智能体上下文
agent_context = scene_encoding['x_a'][:, -1]  # 最后时刻的智能体特征
agent_context = agent_context.repeat(K, 1)    # 复制K份对应K种模式
```

#### 4.3.2 两阶段预测

**提议阶段 (Propose Stage)**
```python
# 粗略预测，降采样输出
propose_output = self.propose_layers(queries, scene_features)
trajectory_propose = self.to_loc_propose(propose_output)  # [N*K, T_future//R, 2]
```

**精化阶段 (Refine Stage)**  
```python
# 精细预测，全分辨率输出
refine_input = torch.cat([queries, trajectory_propose_emb], dim=-1)
refine_output = self.refine_layers(refine_input, scene_features)
trajectory_refine = self.to_loc_refine(refine_output)  # [N*K, T_future, 2]
```

### 4.4 注意力层 (AttentionLayer)

#### 4.4.1 多头注意力机制

```latex
\begin{align}
Q &= XW_Q, \quad K = XW_K, \quad V = XW_V \\
\text{Attention}(Q,K,V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{align}
```

#### 4.4.2 位置编码增强

```python
# 相对位置编码
if self.has_pos_emb:
    K = K + self.to_k_r(relative_pos_emb)  # Key增强
    V = V + self.to_v_r(relative_pos_emb)  # Value增强
```

### 4.5 傅里叶嵌入 (FourierEmbedding)

#### 4.5.1 原理

```latex
\begin{align}
\text{FourierEmb}(x) &= \text{MLP}([\cos(2\pi f_1 x), \sin(2\pi f_1 x), \\
&\quad\quad\quad\quad\quad \cos(2\pi f_2 x), \sin(2\pi f_2 x), \ldots, x])
\end{align}
```

#### 4.5.2 优势
- 更好地编码连续数值特征
- 提供位置敏感性
- 增强模型的表达能力

---

## 5. 训练与评估

### 5.1 损失函数

#### 5.1.1 回归损失 (NLL Loss)
```python
# 拉普拉斯分布 + 冯·米塞斯分布
reg_loss = NLLLoss(
    component_distribution=['laplace'] * 2 + ['von_mises'] * 1,
    reduction='none'
)
```

#### 5.1.2 分类损失 (Mixture NLL Loss)
```python
# 多模态混合分布
cls_loss = MixtureNLLLoss(
    component_distribution=['laplace'] * 2 + ['von_mises'] * 1,
    reduction='none'
)
```

#### 5.1.3 自监督损失
```python
# 轨迹重构任务
ssl_loss = self.self_supervised_task(
    agent_encoding=scene_enc['x_a'],
    original_trajectory=historical_trajectory,
    mask=trajectory_mask
)
```

### 5.2 评估指标

#### 5.2.1 核心指标

```python
# 最小平均位移误差
minADE = min_k(ADE_k)  # Average Displacement Error

# 最小最终位移误差  
minFDE = min_k(FDE_k)  # Final Displacement Error

# 错失率
MR = P(min_k(FDE_k) > threshold)  # Miss Rate

# Brier评分 (概率校准)
Brier = (p - y)²  # y∈{0,1} 是否为最佳预测
```

#### 5.2.2 指标含义

- **minADE**: 所有时间步的平均预测误差
- **minFDE**: 最终时间步的预测误差  
- **MR**: 所有模式都未能准确预测的比例
- **Brier**: 衡量概率预测的校准程度

---

## 6. 从零开始学习指南

### 6.1 前置知识

#### 6.1.1 必需基础
- **深度学习基础**: PyTorch, 神经网络, 反向传播
- **图神经网络**: 消息传递, 注意力机制
- **几何学基础**: 2D/3D坐标变换, 角度计算
- **概率论**: 概率分布, 贝叶斯推理

#### 6.1.2 推荐学习路径

```
阶段1: 基础概念 (1-2周)
├── 轨迹预测问题定义
├── 多智能体系统基础  
├── 注意力机制原理
└── 图神经网络入门

阶段2: 核心技术 (2-3周)  
├── Transformer架构
├── DETR目标检测框架
├── 异构图神经网络
└── 时空数据处理

阶段3: 实践应用 (3-4周)
├── Argoverse数据集分析
├── QCNet代码实现
├── 训练调试技巧
└── 性能优化方法
```

### 6.2 关键概念理解

#### 6.2.1 多模态预测
```python
# 为什么需要多模态？
# 未来轨迹具有不确定性，一个智能体可能有多种合理的行为选择

# 示例：十字路口的车辆可能
modes = [
    "直行通过",      # 模式1
    "左转",          # 模式2  
    "右转",          # 模式3
    "停车等待",      # 模式4
    "变道超车",      # 模式5
    "减速让行"       # 模式6
]
```

#### 6.2.2 查询机制
```python
# 传统方法: 直接回归
trajectory = f(agent_features, map_features)

# QCNet方法: 查询-响应
for mode_i in range(num_modes):
    query_i = mode_embedding[mode_i] + agent_context
    trajectory_i = attention(query_i, scene_features)
```

#### 6.2.3 时空不变性

**旋转不变性**
```python
# 使用相对特征而非绝对坐标
relative_pos = target_pos - reference_pos
relative_angle = target_heading - reference_heading
```

**时间不变性**
```python
# 支持流式处理，不依赖固定时间窗口
def streaming_prediction(new_observation):
    # 更新历史缓冲区
    history_buffer.append(new_observation)
    # 保持预测能力
    return predict(history_buffer[-window_size:])
```

### 6.3 代码理解策略

#### 6.3.1 自顶向下理解
```python
# 1. 从主训练循环开始
def training_step(self, data, batch_idx):
    scene_enc = self.encoder(data)      # 场景编码
    pred = self.decoder(data, scene_enc) # 轨迹预测
    loss = self.compute_loss(pred, data) # 损失计算
    return loss

# 2. 深入每个模块
# 3. 理解数据流转换
# 4. 掌握关键算法
```

#### 6.3.2 关键调试点
```python
# 检查数据维度
print(f"Agent features: {data['agent']['position'].shape}")
print(f"Map features: {data['map_polygon']['position'].shape}")

# 检查注意力权重
attention_weights = self.attention_layer.get_attention_weights()
visualize_attention(attention_weights)

# 检查预测结果
pred_trajectories = model(data)
plot_trajectories(pred_trajectories, ground_truth)
```

---

## 7. 代码实践

### 7.1 环境配置

```bash
# 1. 创建conda环境
conda env create -f environment.yml
conda activate QCNet

# 2. 验证安装
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

### 7.2 数据准备

```bash
# 1. 下载Argoverse2数据集
# 数据会自动下载到指定目录
python train_qcnet.py --root /path/to/data --dataset argoverse_v2 [其他参数]

# 2. 数据预处理
# 首次运行会自动进行数据预处理，生成.pkl文件
```

### 7.3 训练实践

```bash
# 基础训练命令
python train_qcnet.py \
    --root /path/to/dataset \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --devices 8 \
    --dataset argoverse_v2 \
    --num_historical_steps 50 \
    --num_future_steps 60

# 关键超参数说明
# --num_modes: 预测模式数量 (默认6)
# --hidden_dim: 隐藏层维度 (默认128)
# --num_layers: 注意力层数量
# --lr: 学习率
# --ssl_weight: 自监督损失权重
```

### 7.4 模型验证

```bash
# 验证模型性能
python val.py \
    --model QCNet \
    --root /path/to/dataset \
    --ckpt_path /path/to/checkpoint.ckpt

# 生成测试结果
python test.py \
    --model QCNet \
    --root /path/to/dataset \
    --ckpt_path /path/to/checkpoint.ckpt
```

### 7.5 结果分析

```python
# 可视化预测结果
import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction(data, pred, agent_idx=0):
    """可视化单个智能体的预测结果"""
    
    # 历史轨迹
    hist_traj = data['agent']['position'][agent_idx, :50, :2]
    
    # 真实未来轨迹  
    gt_traj = data['agent']['target'][agent_idx, :, :2]
    
    # 预测轨迹 (多模态)
    pred_trajs = pred['loc_refine_pos'][agent_idx].reshape(6, 60, 2)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制历史轨迹
    plt.plot(hist_traj[:, 0], hist_traj[:, 1], 'b-', linewidth=3, label='History')
    
    # 绘制真实未来轨迹
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', linewidth=3, label='Ground Truth')
    
    # 绘制预测轨迹
    for i in range(6):
        plt.plot(pred_trajs[i, :, 0], pred_trajs[i, :, 1], 
                '--', alpha=0.7, label=f'Pred Mode {i+1}')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Agent {agent_idx} Trajectory Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
```

### 7.6 性能优化技巧

#### 7.6.1 内存优化
```python
# 1. 梯度累积减少batch size
accumulate_grad_batches = 4

# 2. 混合精度训练
precision = 16

# 3. 检查点保存策略
save_top_k = 3
```

#### 7.6.2 训练加速
```python
# 1. 数据加载优化
num_workers = 8
pin_memory = True
persistent_workers = True

# 2. 编译模型 (PyTorch 2.0+)
model = torch.compile(model)

# 3. 分布式训练
strategy = DDPStrategy(find_unused_parameters=False)
```

---

## 8. 进阶学习

### 8.1 扩展方向

#### 8.1.1 模型改进
- 增加更多模态预测
- 引入强化学习优化
- 集成语义分割信息
- 支持更长时间预测

#### 8.1.2 应用扩展  
- 多智能体协同规划
- 实时轨迹预测系统
- 异常行为检测
- 交通仿真集成

### 8.2 相关资源

#### 8.2.1 论文阅读
- **QCNet原论文**: "Query-Centric Trajectory Prediction"
- **DETR**: "End-to-End Object Detection with Transformers"  
- **Argoverse**: "Argoverse: 3D Tracking and Forecasting with Rich Maps"

#### 8.2.2 开源项目
- **Argoverse API**: 官方数据处理工具
- **PyTorch Geometric**: 图神经网络库
- **Hydra**: 配置管理框架

---

## 总结

QCNet通过查询中心的设计理念，将多智能体轨迹预测问题转化为查询-响应的交互过程。其核心创新在于：

1. **统一的编码-解码架构**: 有效整合地图和智能体信息
2. **多模态查询机制**: 自然处理未来轨迹的不确定性  
3. **时空不变性设计**: 提供更好的泛化能力
4. **两阶段预测策略**: 平衡计算效率和预测精度

通过本文档的学习，您应该能够：
- 理解QCNet的核心设计思想
- 掌握各模块的实现原理
- 具备代码调试和优化能力
- 为进一步研究打下基础

祝您学习愉快！🚀 