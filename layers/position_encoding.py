import numpy as np
import torch
import torch.nn as nn


# 假设智能体位置数据是通过输入的场景数据传递过来的
def compute_scene_complexity(agents_positions, obstacles_positions):
    """
    计算场景复杂度。这里使用简单的距离度量来评估密集程度。
    :param agents_positions: 当前智能体的位置，形状为 (num_agents, 2)
    :param obstacles_positions: 静态障碍物的位置，形状为 (num_obstacles, 2)
    :return: 计算得到的场景复杂度
    """
    # 确保输入是torch tensor
    if isinstance(agents_positions, np.ndarray):
        agents_positions = torch.from_numpy(agents_positions)
    if isinstance(obstacles_positions, np.ndarray):
        obstacles_positions = torch.from_numpy(obstacles_positions)
    
    # 计算智能体之间的平均距离
    agent_distances = torch.norm(agents_positions[:, None] - agents_positions, dim=-1)
    non_zero_mask = agent_distances > 0
    if non_zero_mask.any():
        avg_agent_distance = torch.mean(agent_distances[non_zero_mask])
    else:
        avg_agent_distance = torch.tensor(0.0)

    # 计算障碍物与智能体的平均距离
    if obstacles_positions.size(0) > 0:
        obstacle_distances = torch.norm(agents_positions[:, None] - obstacles_positions, dim=-1)
        avg_obstacle_distance = torch.mean(obstacle_distances)
    else:
        avg_obstacle_distance = torch.tensor(0.0)

    # 简单的复杂度评估：智能体的密集程度 + 障碍物的密集程度
    complexity = avg_agent_distance + avg_obstacle_distance
    return complexity.item()  # 返回python标量


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = torch.zeros(max_len, d_model)

        # 初始化位置编码
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x, agents_positions, obstacles_positions):
        """
        动态调整位置编码
        :param x: 输入的序列数据
        :param agents_positions: 智能体的位置
        :param obstacles_positions: 障碍物的位置
        :return: 动态调整后的位置编码
        """
        # 计算场景复杂度
        scene_complexity = compute_scene_complexity(agents_positions, obstacles_positions)

        # 根据场景复杂度动态调整 alpha
        alpha = 1 / (1 + torch.exp(torch.tensor(-scene_complexity)))  # 使用torch.exp来保持一致性

        # 计算位置编码
        position_encoding = self.pe[:, :x.size(1)].to(x.device)  # 截取适当长度的编码并移到正确设备

        # 动态调整位置编码
        adjusted_position_encoding = position_encoding * alpha

        return x + adjusted_position_encoding  # 将位置编码加到输入特征上