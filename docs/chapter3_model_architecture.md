# QCNet技术文档：第三章 模型架构

## 3.1 整体架构

QCNet采用编码器-解码器架构，主要包含以下模块：

1. **编码器模块**
   - 地图编码器：处理静态环境信息
   - 智能体编码器：处理动态对象信息
   - 场景融合：整合静态和动态信息

2. **解码器模块**
   - 查询生成：生成轨迹查询
   - 提议阶段：生成粗略轨迹
   - 精化阶段：优化轨迹细节

3. **预测头**
   - 轨迹预测：生成未来轨迹点
   - 不确定性估计：预测概率分布
   - 多模态融合：整合多个预测

## 3.2 编码器模块

### 3.2.1 地图编码器
```python
class MapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 注意力层
        self.self_attention = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.ReLU(),
            nn.Linear(config.ffn_dim, config.hidden_dim)
        )
        
        # 层标准化
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, map_data):
        # 特征提取
        features = self.feature_extractor(map_data)
        
        # 自注意力
        attended = self.self_attention(
            query=features,
            key=features,
            value=features
        )
        features = self.norm1(features + attended)
        
        # 前馈网络
        output = self.ffn(features)
        output = self.norm2(features + output)
        
        return output
```

### 3.2.2 智能体编码器
```python
class AgentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 时序特征提取
        self.temporal_encoder = TemporalFeatureExtractor(config)
        
        # 空间特征提取
        self.spatial_encoder = SpatialFeatureExtractor(config)
        
        # 交互特征提取
        self.interaction_encoder = InteractionFeatureExtractor(config)
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.total_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(self, agent_data):
        # 提取各类特征
        temporal_features = self.temporal_encoder(agent_data)
        spatial_features = self.spatial_encoder(agent_data)
        interaction_features = self.interaction_encoder(agent_data)
        
        # 特征融合
        features = torch.cat([
            temporal_features,
            spatial_features,
            interaction_features
        ], dim=-1)
        output = self.fusion_layer(features)
        
        return output
```

### 3.2.3 场景融合
```python
class SceneFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 交叉注意力
        self.cross_attention = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads
        )
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(self, map_features, agent_features):
        # 交叉注意力
        attended_map = self.cross_attention(
            query=agent_features,
            key=map_features,
            value=map_features
        )
        
        # 特征融合
        features = torch.cat([agent_features, attended_map], dim=-1)
        output = self.fusion_layer(features)
        
        return output
```

## 3.3 解码器模块

### 3.3.1 查询生成器
```python
class QueryGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 查询嵌入
        self.query_embedding = nn.Parameter(
            torch.randn(config.num_queries, config.hidden_dim)
        )
        
        # 查询变换
        self.query_transform = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(self, scene_features):
        batch_size = scene_features.shape[0]
        
        # 扩展查询嵌入
        queries = self.query_embedding.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # 变换查询
        queries = self.query_transform(queries)
        
        return queries
```

### 3.3.2 提议解码器
```python
class ProposalDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 交叉注意力
        self.cross_attention = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads
        )
        
        # 自注意力
        self.self_attention = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.ReLU(),
            nn.Linear(config.ffn_dim, config.hidden_dim)
        )
        
        # 层标准化
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, queries, scene_features):
        # 交叉注意力
        attended_scene = self.cross_attention(
            query=queries,
            key=scene_features,
            value=scene_features
        )
        queries = self.norm1(queries + attended_scene)
        
        # 自注意力
        attended_self = self.self_attention(
            query=queries,
            key=queries,
            value=queries
        )
        queries = self.norm2(queries + attended_self)
        
        # 前馈网络
        output = self.ffn(queries)
        output = self.norm3(queries + output)
        
        return output
```

### 3.3.3 精化解码器
```python
class RefinementDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 注意力层
        self.attention = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads
        )
        
        # 轨迹生成
        self.trajectory_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
    def forward(self, proposals, scene_features):
        # 特征提取
        features = torch.cat([proposals, scene_features], dim=-1)
        features = self.feature_extractor(features)
        
        # 注意力
        attended = self.attention(
            query=features,
            key=features,
            value=features
        )
        
        # 生成轨迹
        output = self.trajectory_generator(attended)
        
        return output
```

## 3.4 预测头

### 3.4.1 轨迹预测头
```python
class TrajectoryHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 轨迹生成
        self.trajectory_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_future_points * 3)
        )
        
    def forward(self, features):
        # 生成轨迹点
        trajectories = self.trajectory_generator(features)
        trajectories = trajectories.view(
            -1, self.config.num_queries,
            self.config.num_future_points, 3
        )
        
        return trajectories
```

### 3.4.2 不确定性预测头
```python
class UncertaintyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 位置不确定性
        self.position_uncertainty = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_future_points * 2)
        )
        
        # 航向不确定性
        self.heading_uncertainty = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_future_points * 2)
        )
        
    def forward(self, features):
        # 预测不确定性参数
        pos_params = self.position_uncertainty(features)
        heading_params = self.heading_uncertainty(features)
        
        # 重塑输出
        pos_params = pos_params.view(
            -1, self.config.num_queries,
            self.config.num_future_points, 2
        )
        heading_params = heading_params.view(
            -1, self.config.num_queries,
            self.config.num_future_points, 2
        )
        
        return pos_params, heading_params
```

### 3.4.3 多模态融合头
```python
class MultiModalHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 模态权重预测
        self.weight_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def forward(self, features, trajectories, uncertainties):
        # 预测模态权重
        weights = self.weight_predictor(features)
        weights = F.softmax(weights, dim=1)
        
        # 加权融合
        final_trajectory = (trajectories * weights.unsqueeze(-1)).sum(dim=1)
        final_uncertainty = (uncertainties * weights.unsqueeze(-1)).sum(dim=1)
        
        return final_trajectory, final_uncertainty, weights
```

## 3.5 完整模型

```python
class QCNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 编码器
        self.map_encoder = MapEncoder(config)
        self.agent_encoder = AgentEncoder(config)
        self.scene_fusion = SceneFusion(config)
        
        # 解码器
        self.query_generator = QueryGenerator(config)
        self.proposal_decoder = ProposalDecoder(config)
        self.refinement_decoder = RefinementDecoder(config)
        
        # 预测头
        self.trajectory_head = TrajectoryHead(config)
        self.uncertainty_head = UncertaintyHead(config)
        self.modal_head = MultiModalHead(config)
        
    def forward(self, data):
        # 编码
        map_features = self.map_encoder(data.map_data)
        agent_features = self.agent_encoder(data.agent_data)
        scene_features = self.scene_fusion(map_features, agent_features)
        
        # 查询生成
        queries = self.query_generator(scene_features)
        
        # 提议阶段
        proposals = self.proposal_decoder(queries, scene_features)
        
        # 精化阶段
        refined = self.refinement_decoder(proposals, scene_features)
        
        # 预测
        trajectories = self.trajectory_head(refined)
        uncertainties = self.uncertainty_head(refined)
        final_traj, final_uncert, weights = self.modal_head(
            refined, trajectories, uncertainties
        )
        
        return {
            'trajectories': trajectories,
            'uncertainties': uncertainties,
            'weights': weights,
            'final_trajectory': final_traj,
            'final_uncertainty': final_uncert
        }
``` 