from typing import Dict, Mapping, Optional
import torch
import torch.nn as nn
import numpy as np
from torch_cluster import radius, radius_graph
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import dense_to_sparse, subgraph

from layers.attention_layer import AttentionLayer
from layers.edge_attention_layer import EdgeAttentionLayer
from layers.fourier_embedding import FourierEmbedding
from layers.position_encoding import PositionEncoding
from utils import angle_between_2d_vectors, weight_init, wrap_angle


class QCNetAgentEncoder(nn.Module):
    def __init__(self, dataset, input_dim, hidden_dim, num_historical_steps, time_span, pl2a_radius, a2a_radius,
                 num_freq_bands, num_layers, num_heads, head_dim, dropout):
        super(QCNetAgentEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        # 初始化位置编码
        self.position_encoding = PositionEncoding(d_model=hidden_dim)

        if dataset == 'argoverse_v2':
            input_dim_x_a = 4
            input_dim_r_t = 4
            input_dim_r_pl2a = 3
            input_dim_r_a2a = 3
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':
            self.type_a_emb = nn.Embedding(10, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        # Embeddings and attention layers
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)

        # Attention layers
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        # self.a2a_attn_layers = nn.ModuleList(
        #     [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
        #                     bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        # )
        # 结合 EdgeAttentionLayer - 使用优化后的高效版本
        self.a2a_attn_layers = nn.ModuleList(
            [EdgeAttentionLayer(hidden_dim=hidden_dim, edge_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def forward(self, data: HeteroData, map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        pos_a = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].contiguous()
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
        head_a = data['agent']['heading'][:, :self.num_historical_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['map_polygon']['orientation'].contiguous()

        if self.dataset == 'argoverse_v2':
            vel = data['agent']['velocity'][:, :self.num_historical_steps, :self.input_dim].contiguous()
            length = width = height = None
            categorical_embs = [
                self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.num_historical_steps,
                                                                                dim=0),
            ]
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        if self.dataset == 'argoverse_v2':
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                 torch.norm(vel[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)

        # 应用动态位置编码
        # 获取当前智能体位置和地图多边形作为"障碍物"
        current_agent_pos = pos_a[:, -1, :2]  # 使用最后一个时间步的位置
        map_polygon_pos = pos_pl[:, :2]  # 地图多边形位置
        
        # 对每个智能体的时间序列特征应用位置编码
        for agent_idx in range(x_a.size(0)):
            agent_features = x_a[agent_idx:agent_idx+1]  # (1, num_historical_steps, hidden_dim)
            # 使用当前智能体和其他智能体的位置来计算场景复杂度
            other_agents_pos = torch.cat([current_agent_pos[:agent_idx], current_agent_pos[agent_idx+1:]], dim=0)
            if other_agents_pos.size(0) == 0:  # 如果只有一个智能体
                other_agents_pos = current_agent_pos[agent_idx:agent_idx+1]
            
            x_a[agent_idx:agent_idx+1] = self.position_encoding(
                agent_features, 
                other_agents_pos, 
                map_polygon_pos
            )

        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        # 计算空间关系
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1)
        orient_pl = orient_pl.repeat(self.num_historical_steps)

        # 计算polygon-to-agent (pl2a) 边缘索引
        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(self.num_historical_steps)], dim=0)
            batch_pl = torch.cat([data['map_polygon']['batch'] + data.num_graphs * t
                                  for t in range(self.num_historical_steps)], dim=0)
        else:
            batch_s = torch.arange(self.num_historical_steps,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(self.num_historical_steps,
                                    device=pos_pl.device).repeat_interleave(data['map_polygon']['num_nodes'])

        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius, batch_x=batch_s, batch_y=batch_pl,
                                 max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        # 计算agent-to-agent (a2a) 边缘索引
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
            x_a = x_a.reshape(-1, self.num_historical_steps, self.hidden_dim).transpose(0, 1).reshape(-1,
                                                                                                      self.hidden_dim)
            x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'].transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a,
                                           edge_index_pl2a)
            # 使用 EdgeAttentionLayer：forward(x, edge_index, edge_attr)
            x_a = self.a2a_attn_layers[i](x_a, edge_index_a2a, r_a2a)

            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)

        return {'x_a': x_a}
