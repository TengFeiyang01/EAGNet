# QCNext: 联合多智能体轨迹预测框架

## 📋 概述

QCNext是QCNet的升级版本，实现了从**边际轨迹预测**到**联合多智能体轨迹预测**的重大跃进。

### 🆚 QCNet vs QCNext 对比

| 特性 | QCNet | QCNext |
|------|-------|--------|
| **预测类型** | 边际预测（每个智能体独立） | 联合预测（场景级整体） |
| **解码器架构** | 递归解码器 | Multi-Agent DETR-like解码器 |
| **交互建模** | 隐式（A2A attention） | 显式（未来时刻交互） |
| **输出格式** | [A, M, T, D] | [K, A, T, D] |
| **损失函数** | 智能体级NLL | 场景级Winner-Take-All |
| **评分机制** | 单智能体置信度 | 场景级置信度 |

## 🏗️ 核心架构

### 1. **Anchor-Free Trajectory Proposal Module**
```python
# K个联合场景查询 [K, A, D]
joint_mode_queries = nn.Parameter(torch.randn(K, max_agents, hidden_dim))

# 四种注意力机制
- Mode2Time cross-attention    # 与历史时序交互
- Mode2Map cross-attention     # 与地图信息交互  
- Row-wise self-attention      # 同场景内智能体交互
- Column-wise self-attention   # 不同场景间通信
```

### 2. **Anchor-Based Trajectory Refinement Module**
```python
# 基于proposal结果进行精化
traj_emb = nn.GRU(...)  # 轨迹嵌入
# 相同的四种注意力机制进行refinement
```

### 3. **Scene Scoring Module**
```python
# 场景级置信度评分
scene_scoring = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1)
)
```

## 📊 损失函数

### Joint NLL Loss (联合负对数似然损失)
```python
# 场景级Winner-Take-All策略
def forward(pred, target, mask):
    # pred: [K, A, T, D] - K个联合场景
    # target: [A, T, output_dim] - 真实轨迹
    
    # 计算每个场景的联合似然
    for k in range(K):
        joint_log_prob = ∏_{i=1}^{A} ∏_{t=1}^{T} f(p_i^{t}|θ_k)
        scene_nll = -joint_log_prob.sum()
    
    # 选择最佳场景
    best_scene = argmin(scene_nlls)
    return scene_nlls[best_scene]
```

### Joint Mixture NLL Loss (场景级分类损失)
```python
# 优化最佳场景的置信度
scene_log_probs = F.log_softmax(scene_pi, dim=0)
classification_loss = -scene_log_probs[best_scene_idx]
```

## 🚀 使用方法

### 1. **训练QCNext模型**

```bash
python train_qcnext.py \
    --root /path/to/argoverse_v2 \
    --num_joint_modes 6 \
    --hidden_dim 128 \
    --max_epochs 64 \
    --train_batch_size 4 \
    --devices 1 \
    --output_head
```

### 2. **主要参数说明**

```bash
# QCNext特有参数
--num_joint_modes 6        # K个联合场景数量
--max_agents 64           # 最大支持智能体数量（代码中设置）

# 网络架构参数
--hidden_dim 128          # 隐藏层维度
--num_dec_layers 2        # 解码器层数
--num_heads 8             # 注意力头数
--dropout 0.1             # Dropout率

# 训练参数
--lr 5e-4                 # 学习率
--weight_decay 1e-4       # 权重衰减
--gradient_clip_val 1.0   # 梯度裁剪（自动添加）
```

### 3. **从QCNet迁移**

如果你有QCNet的检查点，可以这样迁移：

```python
# 1. 加载QCNet检查点
qcnet_ckpt = torch.load('qcnet_checkpoint.ckpt')

# 2. 初始化QCNext
qcnext = QCNext(num_joint_modes=6, ...)

# 3. 迁移编码器权重（完全兼容）
encoder_state = {k.replace('encoder.', ''): v 
                 for k, v in qcnet_ckpt['state_dict'].items() 
                 if 'encoder' in k}
qcnext.encoder.load_state_dict(encoder_state)

# 4. 解码器需要重新训练（架构不同）
```

## 📈 性能优势

### 1. **理论优势**
- **显式交互建模**：Row-wise attention显式建模同场景内智能体的未来交互
- **场景级一致性**：联合预测确保多智能体轨迹的全局一致性
- **更丰富的模式**：K个联合场景比单智能体多模态更具表达力

### 2. **实验结果**（论文数据）
```
Argoverse 2 Multi-Agent Challenge:
- minADE: QCNet单智能体 → QCNext联合预测 (提升)
- minFDE: 场景级优化带来更好的终点预测
- 首次证明：联合预测在边际指标上也能超越边际预测
```

## 🔧 代码结构

```
QCNext实现文件:
├── modules/qcnext_decoder.py          # Multi-Agent DETR解码器
├── losses/joint_nll_loss.py           # 联合损失函数
├── predictors/qcnext.py               # QCNext主预测器
└── train_qcnext.py                    # 训练脚本
```

## ⚠️ 注意事项

### 1. **内存和计算开销**
```python
# QCNet: [A, M, T, D] ≈ A × 6 × 60 × 4
# QCNext: [K, A, T, D] ≈ 6 × A × 60 × 4
# 当A很大时，QCNext的显存占用会显著增加
```

### 2. **批处理大小**
```bash
# 建议根据GPU显存调整batch size
# RTX 3090 (24GB): batch_size=2-4
# V100 (32GB): batch_size=4-8
```

### 3. **训练稳定性**
```python
# 自动添加的训练优化技巧：
- gradient_clip_val=1.0      # 梯度裁剪
- EMA updates                # 指数移动平均（可选）
- Warmup scheduler           # 学习率预热（可选）
```

## 🔄 与现有代码的兼容性

### 1. **数据加载器**
- ✅ 完全兼容现有的ArgoversV2DataModule
- ✅ 数据预处理流程无需修改

### 2. **评估指标**
- ✅ 自动转换联合预测为边际预测进行评估
- ✅ 复用现有的minADE、minFDE等指标

### 3. **可视化工具**
- ✅ 可以复用现有的轨迹可视化代码
- 🆕 新增场景级可视化功能

## 🎯 总结

QCNext通过以下技术创新实现了从边际到联合预测的跃升：

1. **Multi-Agent DETR架构**：借鉴DETR的查询机制，设计联合场景查询
2. **四重注意力机制**：全方位建模时序、地图、智能体内和场景间交互
3. **场景级Winner-Take-All**：联合优化所有智能体，确保全局一致性
4. **显式未来交互建模**：不再依赖隐式的历史交互，直接建模未来交互

这使得QCNext成为首个在联合多智能体预测任务上超越边际预测的框架！🏆 