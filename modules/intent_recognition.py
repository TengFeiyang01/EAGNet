import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class IntentRecognition(nn.Module):
    """
    轨迹意图识别模块
    
    核心思想：显式建模智能体的行为意图，与QCNet的隐式学习形成互补
    优势：
    1. 实现简单，见效快
    2. 与QCNet思路完全不同，差异化明显
    3. 可解释性强，容易可视化
    4. 数据标注可以自动化
    """
    
    def __init__(self, hidden_dim=128, num_intents=8):
        super(IntentRecognition, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_intents = num_intents
        
        # 定义8种基本意图
        self.intent_names = [
            'STRAIGHT',      # 0: 直行
            'LEFT_TURN',     # 1: 左转
            'RIGHT_TURN',    # 2: 右转
            'LANE_CHANGE_LEFT',  # 3: 左变道
            'LANE_CHANGE_RIGHT', # 4: 右变道
            'ACCELERATE',    # 5: 加速
            'DECELERATE',    # 6: 减速/刹车
            'STOP'           # 7: 停车
        ]
        
        # 意图分类器 - 从智能体特征预测意图
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_intents)
        )
        
        # 意图嵌入 - 将意图转换为特征
        self.intent_embedding = nn.Embedding(num_intents, hidden_dim // 4)
        
        # 意图-轨迹融合层
        self.intent_trajectory_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 意图一致性检查器
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),  # +2 for trajectory features
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def extract_intent_labels(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        从轨迹自动提取意图标签 - 无需人工标注！
        
        Args:
            trajectories: [batch_size, seq_len, 2] 历史轨迹
            
        Returns:
            intent_labels: [batch_size] 意图标签
        """
        batch_size, seq_len, _ = trajectories.shape
        intent_labels = torch.zeros(batch_size, dtype=torch.long, device=trajectories.device)
        
        for i in range(batch_size):
            traj = trajectories[i]  # [seq_len, 2]
            
            # 过滤掉无效点
            valid_mask = ~torch.isnan(traj).any(dim=-1)
            if valid_mask.sum() < 3:
                intent_labels[i] = 0  # 默认直行
                continue
                
            valid_traj = traj[valid_mask]
            
            # 计算运动特征
            velocities = valid_traj[1:] - valid_traj[:-1]
            speeds = torch.norm(velocities, dim=-1)
            
            if len(speeds) == 0:
                intent_labels[i] = 0
                continue
            
            # 计算转向特征
            angles = torch.atan2(velocities[:, 1], velocities[:, 0])
            if len(angles) > 1:
                angle_changes = torch.diff(angles)
                # 处理角度跳跃 (-π到π)
                angle_changes = torch.atan2(torch.sin(angle_changes), torch.cos(angle_changes))
                total_angle_change = torch.sum(angle_changes)
            else:
                total_angle_change = 0
            
            # 意图判断逻辑
            mean_speed = torch.mean(speeds)
            max_speed = torch.max(speeds)
            min_speed = torch.min(speeds)
            
            # 1. 停车判断
            if mean_speed < 0.5:
                intent_labels[i] = 7  # STOP
            # 2. 转向判断
            elif torch.abs(total_angle_change) > 0.3:  # 约17度
                if total_angle_change > 0:
                    intent_labels[i] = 1  # LEFT_TURN
                else:
                    intent_labels[i] = 2  # RIGHT_TURN
            # 3. 变道判断（小幅度横向移动）
            elif torch.abs(total_angle_change) > 0.1:  # 约6度
                lateral_displacement = torch.abs(valid_traj[-1, 1] - valid_traj[0, 1])
                if lateral_displacement > 1.0:  # 横向位移大于1米
                    if total_angle_change > 0:
                        intent_labels[i] = 3  # LANE_CHANGE_LEFT
                    else:
                        intent_labels[i] = 4  # LANE_CHANGE_RIGHT
                else:
                    intent_labels[i] = 0  # STRAIGHT
            # 4. 加减速判断
            elif len(speeds) > 5:
                early_speed = torch.mean(speeds[:len(speeds)//3])
                late_speed = torch.mean(speeds[-len(speeds)//3:])
                
                if late_speed > early_speed * 1.2:  # 加速20%以上
                    intent_labels[i] = 5  # ACCELERATE
                elif late_speed < early_speed * 0.8:  # 减速20%以上
                    intent_labels[i] = 6  # DECELERATE
                else:
                    intent_labels[i] = 0  # STRAIGHT
            else:
                intent_labels[i] = 0  # 默认直行
        
        return intent_labels
    
    def forward(self, agent_features: torch.Tensor, 
                historical_trajectories: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            agent_features: [batch_size, seq_len, hidden_dim] 来自QCNet智能体编码器
            historical_trajectories: [batch_size, seq_len, 2] 历史轨迹（可选，用于监督学习）
            
        Returns:
            Dictionary containing:
            - intent_logits: [batch_size, num_intents] 意图预测logits
            - intent_probs: [batch_size, num_intents] 意图概率
            - intent_features: [batch_size, hidden_dim//4] 意图特征
            - enhanced_features: [batch_size, hidden_dim] 增强后的智能体特征
        """
        batch_size = agent_features.shape[0]
        
        # 使用最后时刻的特征进行意图预测
        current_features = agent_features[:, -1, :]  # [batch_size, hidden_dim]
        
        # 预测意图
        intent_logits = self.intent_classifier(current_features)  # [batch_size, num_intents]
        intent_probs = F.softmax(intent_logits, dim=-1)
        
        # 生成意图特征（使用概率加权）
        intent_features = torch.matmul(intent_probs, self.intent_embedding.weight)  # [batch_size, hidden_dim//4]
        
        # 融合意图特征和智能体特征
        fused_input = torch.cat([current_features, intent_features], dim=-1)
        enhanced_features = self.intent_trajectory_fusion(fused_input)
        
        result = {
            'intent_logits': intent_logits,
            'intent_probs': intent_probs,
            'intent_features': intent_features,
            'enhanced_features': enhanced_features
        }
        
        # 如果提供了历史轨迹，计算意图标签用于监督学习
        if historical_trajectories is not None:
            intent_labels = self.extract_intent_labels(historical_trajectories)
            result['intent_labels'] = intent_labels
            
            # 计算意图分类损失
            intent_loss = F.cross_entropy(intent_logits, intent_labels)
            result['intent_loss'] = intent_loss
        
        return result
    
    def compute_intent_consistency_loss(self, intent_probs: torch.Tensor, 
                                      predicted_trajectories: torch.Tensor) -> torch.Tensor:
        """
        计算意图-轨迹一致性损失
        确保预测的轨迹与识别的意图保持一致
        
        Args:
            intent_probs: [batch_size, num_intents] 意图概率
            predicted_trajectories: [batch_size, num_modes, seq_len, 2] 预测轨迹
            
        Returns:
            consistency_loss: 一致性损失
        """
        batch_size, num_modes, seq_len, _ = predicted_trajectories.shape
        
        consistency_scores = []
        
        for mode in range(num_modes):
            traj = predicted_trajectories[:, mode, :, :]  # [batch_size, seq_len, 2]
            
            # 从预测轨迹提取意图
            predicted_intents = self.extract_intent_labels(traj)  # [batch_size]
            predicted_intent_probs = F.one_hot(predicted_intents, self.num_intents).float()
            
            # 计算与识别意图的一致性
            consistency = torch.sum(intent_probs * predicted_intent_probs, dim=-1)  # [batch_size]
            consistency_scores.append(consistency)
        
        # 取所有模式的最大一致性（至少有一个模式应该与意图一致）
        max_consistency = torch.stack(consistency_scores, dim=1).max(dim=1)[0]  # [batch_size]
        
        # 一致性损失：鼓励高一致性
        consistency_loss = -torch.log(max_consistency + 1e-8).mean()
        
        return consistency_loss
    
    def get_intent_name(self, intent_idx: int) -> str:
        """获取意图名称"""
        return self.intent_names[intent_idx]
    
    def visualize_intents(self, intent_probs: torch.Tensor, top_k: int = 3) -> list:
        """
        可视化意图预测结果
        
        Args:
            intent_probs: [batch_size, num_intents] 意图概率
            top_k: 显示前k个最可能的意图
            
        Returns:
            List of dictionaries with intent predictions for each sample
        """
        batch_size = intent_probs.shape[0]
        results = []
        
        for i in range(batch_size):
            probs = intent_probs[i].cpu().numpy()
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            sample_result = {
                'sample_idx': i,
                'top_intents': [
                    {
                        'intent': self.intent_names[idx],
                        'probability': float(probs[idx]),
                        'index': int(idx)
                    }
                    for idx in top_indices
                ]
            }
            results.append(sample_result)
        
        return results


class IntentGuidedDecoder(nn.Module):
    """
    意图引导的解码器
    将意图信息集成到QCNet的解码过程中
    """
    
    def __init__(self, original_decoder, intent_dim=32):
        super(IntentGuidedDecoder, self).__init__()
        self.original_decoder = original_decoder
        self.intent_dim = intent_dim
        
        # 意图引导的查询增强
        self.intent_query_fusion = nn.Sequential(
            nn.Linear(original_decoder.hidden_dim + intent_dim, original_decoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(original_decoder.hidden_dim, original_decoder.hidden_dim)
        )
    
    def forward(self, data, scene_enc, intent_features):
        """
        使用意图信息增强解码过程
        
        Args:
            data: 输入数据
            scene_enc: 场景编码
            intent_features: [batch_size, intent_dim] 意图特征
            
        Returns:
            增强后的预测结果
        """
        # 将意图特征融入模式嵌入
        num_agents = intent_features.shape[0]
        num_modes = self.original_decoder.num_modes
        
        # 扩展意图特征到所有模式
        expanded_intent = intent_features.unsqueeze(1).repeat(1, num_modes, 1)  # [batch_size, num_modes, intent_dim]
        expanded_intent = expanded_intent.reshape(-1, self.intent_dim)  # [batch_size * num_modes, intent_dim]
        
        # 获取原始模式嵌入
        original_mode_emb = self.original_decoder.mode_emb.weight.repeat(num_agents, 1)  # [batch_size * num_modes, hidden_dim]
        
        # 融合意图和模式嵌入
        fused_input = torch.cat([original_mode_emb, expanded_intent], dim=-1)
        enhanced_mode_emb = self.intent_query_fusion(fused_input)
        
        # 临时替换模式嵌入
        original_weight = self.original_decoder.mode_emb.weight.data.clone()
        self.original_decoder.mode_emb.weight.data = enhanced_mode_emb.reshape(num_agents, num_modes, -1).mean(0)
        
        # 调用原始解码器
        result = self.original_decoder(data, scene_enc)
        
        # 恢复原始权重
        self.original_decoder.mode_emb.weight.data = original_weight
        
        return result 