# QCNet技术文档：第二章 数据处理与特征工程

## 2.1 数据结构设计

### 2.1.1 异构图数据结构
QCNet使用PyTorch Geometric的HeteroData来组织数据：

```python
class ArgoverseData(HeteroData):
    def __init__(self):
        super().__init__()
        self.agent = {
            'position': None,      # [N_agents, T_hist, 3]
            'heading': None,       # [N_agents, T_hist]
            'velocity': None,      # [N_agents, T_hist, 3]
            'valid_mask': None,    # [N_agents, T_hist]
            'predict_mask': None,  # [N_agents]
            'type': None,         # [N_agents]
            'target': None        # [N_agents, T_future, 3]
        }
        
        self.map_polygon = {
            'position': None,      # [N_polygons, max_points, 3]
            'type': None,         # [N_polygons]
            'valid_mask': None     # [N_polygons, max_points]
        }
        
        self.map_lane = {
            'position': None,      # [N_lanes, max_points, 3]
            'type': None,         # [N_lanes]
            'valid_mask': None     # [N_lanes, max_points]
        }
```

### 2.1.2 数据字段说明

1. **智能体数据**
   - position：历史轨迹点坐标
   - heading：航向角
   - velocity：速度向量
   - valid_mask：有效性掩码
   - predict_mask：预测目标掩码
   - type：智能体类型
   - target：未来轨迹（训练用）

2. **地图数据**
   - map_polygon：多边形要素
     * position：顶点坐标
     * type：多边形类型
     * valid_mask：顶点有效性
   - map_lane：车道线要素
     * position：采样点坐标
     * type：车道线类型
     * valid_mask：采样点有效性

## 2.2 数据预处理

### 2.2.1 坐标系转换
```python
def coordinate_transform(data):
    """坐标系转换
    
    1. 全局坐标 -> 相对坐标
    2. 绝对角度 -> 相对角度
    3. 速度分解
    """
    # 选择参考智能体
    ref_pos = data.agent['position'][data.agent['predict_mask']][0]
    ref_heading = data.agent['heading'][data.agent['predict_mask']][0]
    
    # 构建变换矩阵
    cos_h = torch.cos(ref_heading)
    sin_h = torch.sin(ref_heading)
    R = torch.stack([
        torch.stack([cos_h, -sin_h], dim=-1),
        torch.stack([sin_h, cos_h], dim=-1)
    ], dim=-2)
    
    # 应用变换
    data.agent['position'] = torch.matmul(
        data.agent['position'] - ref_pos.unsqueeze(1),
        R
    )
    data.agent['heading'] = data.agent['heading'] - ref_heading
    
    return data
```

### 2.2.2 数据增强
```python
class DataAugmentor:
    def __init__(self, config):
        self.config = config
        
    def random_rotation(self, data):
        """随机旋转增强"""
        angle = torch.rand(1) * 2 * math.pi
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 应用旋转
        data.agent['position'] = torch.matmul(data.agent['position'], R)
        data.agent['heading'] += angle
        
        return data
        
    def random_noise(self, data):
        """添加随机噪声"""
        if torch.rand(1) < self.config.noise_prob:
            noise = torch.randn_like(data.agent['position']) * self.config.noise_std
            data.agent['position'] += noise
            
        return data
        
    def __call__(self, data):
        """应用所有数据增强"""
        data = self.random_rotation(data)
        data = self.random_noise(data)
        return data
```

## 2.3 特征工程

### 2.3.1 时序特征提取
```python
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 时间编码
        self.time_encoder = FourierFeatureTransform(
            input_dim=1,
            hidden_dim=config.time_dim,
            num_bands=config.num_bands
        )
        
        # 轨迹编码
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(7, config.hidden_dim),  # [x, y, z, vx, vy, vz, heading]
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(self, data):
        # 提取时间特征
        t = torch.arange(self.config.num_history_frames)
        time_features = self.time_encoder(t)
        
        # 提取轨迹特征
        traj_features = torch.cat([
            data.agent['position'],
            data.agent['velocity'],
            data.agent['heading'].unsqueeze(-1)
        ], dim=-1)
        traj_features = self.trajectory_encoder(traj_features)
        
        # 组合特征
        features = traj_features + time_features.unsqueeze(0)
        
        return features
```

### 2.3.2 空间特征提取
```python
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 空间编码
        self.pos_encoder = FourierFeatureTransform(
            input_dim=3,
            hidden_dim=config.pos_dim,
            num_bands=config.num_bands
        )
        
        # 类型编码
        self.type_embedding = nn.Embedding(
            num_embeddings=config.num_types,
            embedding_dim=config.type_dim
        )
        
    def forward(self, data):
        # 位置特征
        pos_features = self.pos_encoder(data.agent['position'])
        
        # 类型特征
        type_features = self.type_embedding(data.agent['type'])
        
        # 组合特征
        features = torch.cat([pos_features, type_features], dim=-1)
        
        return features
```

### 2.3.3 交互特征提取
```python
class InteractionFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 相对位置编码
        self.rel_pos_encoder = FourierFeatureTransform(
            input_dim=3,
            hidden_dim=config.rel_pos_dim,
            num_bands=config.num_bands
        )
        
        # 相对速度编码
        self.rel_vel_encoder = nn.Linear(3, config.rel_vel_dim)
        
    def forward(self, data):
        N = data.agent['position'].shape[0]
        
        # 计算相对位置
        pos_i = data.agent['position'].unsqueeze(1)  # [N, 1, T, 3]
        pos_j = data.agent['position'].unsqueeze(0)  # [1, N, T, 3]
        rel_pos = pos_i - pos_j  # [N, N, T, 3]
        
        # 计算相对速度
        vel_i = data.agent['velocity'].unsqueeze(1)
        vel_j = data.agent['velocity'].unsqueeze(0)
        rel_vel = vel_i - vel_j
        
        # 提取特征
        pos_features = self.rel_pos_encoder(rel_pos)
        vel_features = self.rel_vel_encoder(rel_vel)
        
        # 组合特征
        features = torch.cat([pos_features, vel_features], dim=-1)
        
        return features
```

## 2.4 数据加载与批处理

### 2.4.1 数据集实现
```python
class TrajectoryDataset(Dataset):
    def __init__(self, data_root, split, config):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.config = config
        
        # 数据增强
        self.augmentor = DataAugmentor(config) if split == 'train' else None
        
        # 加载数据索引
        self.data_index = self._load_index()
        
    def _load_index(self):
        index_file = os.path.join(self.data_root, f'{self.split}_index.txt')
        with open(index_file, 'r') as f:
            index = [line.strip() for line in f]
        return index
        
    def __len__(self):
        return len(self.data_index)
        
    def __getitem__(self, idx):
        # 加载原始数据
        data_file = os.path.join(self.data_root, self.data_index[idx])
        data = torch.load(data_file)
        
        # 数据预处理
        data = coordinate_transform(data)
        
        # 数据增强
        if self.augmentor is not None:
            data = self.augmentor(data)
            
        return data
```

### 2.4.2 数据收集器
```python
def collate_fn(batch):
    """批处理数据收集器"""
    
    # 获取批次大小
    batch_size = len(batch)
    
    # 初始化批次数据
    batch_data = ArgoverseData()
    
    # 合并智能体数据
    agent_keys = ['position', 'heading', 'velocity', 'valid_mask', 
                  'predict_mask', 'type', 'target']
    for key in agent_keys:
        batch_data.agent[key] = torch.cat(
            [data.agent[key] for data in batch],
            dim=0
        )
        
    # 合并地图数据
    map_keys = ['position', 'type', 'valid_mask']
    for key in map_keys:
        batch_data.map_polygon[key] = torch.cat(
            [data.map_polygon[key] for data in batch],
            dim=0
        )
        batch_data.map_lane[key] = torch.cat(
            [data.map_lane[key] for data in batch],
            dim=0
        )
        
    return batch_data
```

### 2.4.3 数据加载器配置
```python
def create_dataloader(data_root, split, config):
    """创建数据加载器"""
    
    # 创建数据集
    dataset = TrajectoryDataset(data_root, split, config)
    
    # 配置数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
``` 