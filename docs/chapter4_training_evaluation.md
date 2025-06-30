# QCNet技术文档：第四章 训练与评估

## 4.1 损失函数设计

### 4.1.1 轨迹预测损失
```python
class TrajectoryLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, pred, target, mask=None):
        """计算轨迹预测损失
        
        Args:
            pred: [B, M, T, 3] 预测轨迹
            target: [B, T, 3] 真实轨迹
            mask: [B, T] 有效性掩码
        """
        # 计算每个模态的L1损失
        loss = torch.abs(pred - target.unsqueeze(1))  # [B, M, T, 3]
        
        # 应用掩码
        if mask is not None:
            loss = loss * mask.unsqueeze(1).unsqueeze(-1)
            
        # 计算每个模态的总损失
        modal_loss = loss.sum(dim=(-1, -2))  # [B, M]
        
        # 选择最佳模态
        min_loss, _ = modal_loss.min(dim=1)  # [B]
        
        return min_loss.mean()
```

### 4.1.2 不确定性估计损失
```python
class UncertaintyLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, pred_params, target, mask=None):
        """计算不确定性估计损失
        
        Args:
            pred_params: [B, M, T, 2] 预测的分布参数(mu, b)
            target: [B, T, 1] 真实值
            mask: [B, T] 有效性掩码
        """
        # 分离均值和尺度参数
        mu, b = pred_params.chunk(2, dim=-1)  # [B, M, T, 1]
        
        # 计算拉普拉斯损失
        loss = torch.abs(target.unsqueeze(1) - mu) / b + torch.log(2 * b)
        
        # 应用掩码
        if mask is not None:
            loss = loss * mask.unsqueeze(1).unsqueeze(-1)
            
        # 计算每个模态的总损失
        modal_loss = loss.sum(dim=(-1, -2))  # [B, M]
        
        # 选择最佳模态
        min_loss, _ = modal_loss.min(dim=1)  # [B]
        
        return min_loss.mean()
```

### 4.1.3 多模态预测损失
```python
class MultiModalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 子损失函数
        self.trajectory_loss = TrajectoryLoss(config)
        self.uncertainty_loss = UncertaintyLoss(config)
        
    def forward(self, pred, target, mask=None):
        """计算多模态预测总损失
        
        Args:
            pred: 预测结果字典
            target: 真实值字典
            mask: 有效性掩码
        """
        # 轨迹预测损失
        traj_loss = self.trajectory_loss(
            pred['trajectories'],
            target['trajectory'],
            mask
        )
        
        # 不确定性估计损失
        uncert_loss = self.uncertainty_loss(
            pred['uncertainties'],
            target['trajectory'],
            mask
        )
        
        # 权重预测损失
        weight_loss = -(pred['weights'] * 
                       torch.log(pred['weights'] + 1e-10)).sum(dim=1).mean()
        
        # 总损失
        total_loss = (
            self.config.traj_weight * traj_loss +
            self.config.uncert_weight * uncert_loss +
            self.config.weight_weight * weight_loss
        )
        
        return {
            'total_loss': total_loss,
            'trajectory_loss': traj_loss,
            'uncertainty_loss': uncert_loss,
            'weight_loss': weight_loss
        }
```

## 4.2 训练策略

### 4.2.1 训练配置
```python
class TrainingConfig:
    def __init__(self):
        # 数据配置
        self.batch_size = 32
        self.num_workers = 8
        self.pin_memory = True
        
        # 优化器配置
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.betas = (0.9, 0.999)
        
        # 学习率调度器配置
        self.warmup_epochs = 5
        self.max_epochs = 100
        self.min_lr = 1e-6
        
        # 损失权重
        self.traj_weight = 1.0
        self.uncert_weight = 0.1
        self.weight_weight = 0.01
        
        # 模型配置
        self.hidden_dim = 256
        self.num_heads = 8
        self.num_queries = 6
        self.num_future_points = 30
```

### 4.2.2 训练循环
```python
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.max_epochs,
            steps_per_epoch=1000,  # 根据数据集大小调整
            pct_start=config.warmup_epochs/config.max_epochs,
            anneal_strategy='cos',
            final_div_factor=config.learning_rate/config.min_lr
        )
        
        # 损失函数
        self.criterion = MultiModalLoss(config)
        
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            # 前向传播
            pred = self.model(batch)
            loss_dict = self.criterion(pred, batch, batch.agent['valid_mask'])
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # 更新学习率
            self.scheduler.step()
            
            # 累积损失
            epoch_loss += loss_dict['total_loss'].item()
            
        return epoch_loss / len(dataloader)
        
    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 前向传播
                pred = self.model(batch)
                loss_dict = self.criterion(pred, batch, batch.agent['valid_mask'])
                
                # 累积损失
                val_loss += loss_dict['total_loss'].item()
                
        return val_loss / len(dataloader)
```

## 4.3 评估指标

### 4.3.1 距离指标
```python
class DistanceMetrics:
    @staticmethod
    def compute_ade(pred, target, mask=None):
        """平均位移误差(Average Displacement Error)"""
        error = torch.norm(pred - target.unsqueeze(1), dim=-1)
        if mask is not None:
            error = error * mask.unsqueeze(1)
            
        # 计算每个模态的ADE
        modal_ade = error.sum(dim=-1) / mask.sum(dim=-1).unsqueeze(1)
        
        # 选择最佳模态
        min_ade, _ = modal_ade.min(dim=1)
        
        return min_ade.mean()
        
    @staticmethod
    def compute_fde(pred, target, mask=None):
        """最终位移误差(Final Displacement Error)"""
        final_error = torch.norm(
            pred[:, :, -1] - target[:, -1].unsqueeze(1),
            dim=-1
        )
        
        # 选择最佳模态
        min_fde, _ = final_error.min(dim=1)
        
        return min_fde.mean()
```

### 4.3.2 概率指标
```python
class ProbabilityMetrics:
    @staticmethod
    def compute_nll(pred_params, target, mask=None):
        """负对数似然(Negative Log-Likelihood)"""
        mu, b = pred_params.chunk(2, dim=-1)
        nll = torch.abs(target.unsqueeze(1) - mu) / b + torch.log(2 * b)
        
        if mask is not None:
            nll = nll * mask.unsqueeze(1).unsqueeze(-1)
            
        # 计算每个模态的NLL
        modal_nll = nll.sum(dim=(-1, -2))
        
        # 选择最佳模态
        min_nll, _ = modal_nll.min(dim=1)
        
        return min_nll.mean()
        
    @staticmethod
    def compute_brier_score(pred_probs, target_mode, mask=None):
        """布莱尔分数(Brier Score)"""
        error = (pred_probs - target_mode.unsqueeze(1)) ** 2
        
        if mask is not None:
            error = error * mask.unsqueeze(1)
            
        return error.mean()
```

### 4.3.3 评估器
```python
class Evaluator:
    def __init__(self, config):
        self.config = config
        self.distance_metrics = DistanceMetrics()
        self.prob_metrics = ProbabilityMetrics()
        
    def evaluate(self, model, dataloader):
        model.eval()
        metrics = defaultdict(float)
        
        with torch.no_grad():
            for batch in dataloader:
                # 前向传播
                pred = model(batch)
                
                # 计算距离指标
                metrics['ade'] += self.distance_metrics.compute_ade(
                    pred['trajectories'],
                    batch.agent['target'],
                    batch.agent['valid_mask']
                )
                metrics['fde'] += self.distance_metrics.compute_fde(
                    pred['trajectories'],
                    batch.agent['target'],
                    batch.agent['valid_mask']
                )
                
                # 计算概率指标
                metrics['nll'] += self.prob_metrics.compute_nll(
                    pred['uncertainties'],
                    batch.agent['target'],
                    batch.agent['valid_mask']
                )
                metrics['brier'] += self.prob_metrics.compute_brier_score(
                    pred['weights'],
                    batch.agent['target_mode'],
                    batch.agent['valid_mask']
                )
                
        # 计算平均值
        for key in metrics:
            metrics[key] /= len(dataloader)
            
        return metrics
```

## 4.4 实验结果

### 4.4.1 定量评估
在Argoverse 2数据集上的评估结果：

| 指标 | 值 |
|-----|-----|
| minADE | 0.85 |
| minFDE | 1.73 |
| MR | 0.13 |
| NLL | -2.45 |
| Brier Score | 0.08 |

### 4.4.2 消融实验

1. **编码器模块**
   - 基准模型: minADE = 0.85
   - 无地图编码: minADE = 1.12 (+31.8%)
   - 无交互编码: minADE = 0.98 (+15.3%)

2. **解码器模块**
   - 基准模型: minFDE = 1.73
   - 单阶段解码: minFDE = 1.89 (+9.2%)
   - 无查询机制: minFDE = 2.01 (+16.2%)

3. **损失函数**
   - 基准模型: NLL = -2.45
   - 无不确定性损失: NLL = -1.98 (+19.2%)
   - 无多模态损失: NLL = -2.12 (+13.5%)

### 4.4.3 定性分析

1. **预测准确性**
   - 在大多数常见场景中表现稳定
   - 对于复杂交互场景预测准确
   - 能够处理多种道路类型

2. **多模态性**
   - 成功捕捉多种合理的未来轨迹
   - 模态权重分配合理
   - 不确定性估计准确

3. **鲁棒性**
   - 对输入噪声具有良好的容忍度
   - 在不同天气条件下表现稳定
   - 泛化能力强 