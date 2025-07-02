import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle


class QCNextDecoder(nn.Module):
    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_joint_modes: int,  # K: 联合场景数量
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNextDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_joint_modes = num_joint_modes  # K个联合场景
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        # ===== 嵌入层 =====
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        
        # ===== 动态联合查询初始化 =====
        # 使用MLP来为每个智能体生成初始查询，而不是固定大小的参数
        self.query_init = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joint_modes * hidden_dim)
        )
        
        # Mode2Time cross-attention: 联合查询与历史时序信息交互
        self.mode2time_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        
        # Mode2Map cross-attention: 联合查询与地图信息交互
        self.mode2map_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        
        # Row-wise self-attention: 同一联合场景内的智能体交互
        self.row_self_attn_propose_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=False) for _ in range(num_layers)]
        )
        
        # Column-wise self-attention: 不同联合场景间的通信
        self.col_self_attn_propose_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                          dropout=dropout, bipartite=False, has_pos_emb=False)
        
        # ===== Anchor-Based Trajectory Refinement Module =====
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        
        # Refinement attention layers
        self.mode2time_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.mode2map_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.row_self_attn_refine_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=False) for _ in range(num_layers)]
        )
        self.col_self_attn_refine_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                         dropout=dropout, bipartite=False, has_pos_emb=False)

        # ===== 输出头 =====
        # 检查递归步骤的整除性
        if num_future_steps % num_recurrent_steps != 0:
            raise ValueError(f"num_future_steps ({num_future_steps}) must be divisible by num_recurrent_steps ({num_recurrent_steps})")
        
        self.steps_per_recurrent = num_future_steps // num_recurrent_steps
        
        # Proposal outputs (递归预测)
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=self.steps_per_recurrent * output_dim)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=self.steps_per_recurrent * output_dim)
        
        # Refinement outputs (一次性预测)
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=self.steps_per_recurrent)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=self.steps_per_recurrent)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None

        # ===== Scene Scoring Module =====
        # 场景级置信度评分，使用注意力池化（按论文建议）
        self.attentive_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.scene_scoring = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # ===== 正确处理批次数据 =====
        is_batch = isinstance(data, Batch)
        agent_batch = data['agent']['batch'] if is_batch else torch.zeros(data['agent']['num_nodes'], dtype=torch.long, device=data['agent']['position'].device)
        batch_size = agent_batch.max().item() + 1 if is_batch else 1
        
        # 获取智能体基本信息
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        # 场景编码
        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1]
        x_a = scene_enc['x_a'][:, -1]

        # 构建掩码
        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True)

        # ===== 计算相对位置编码 =====
        # Time-to-Mode relations
        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)

        # Polyline-to-Mode relations
        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=agent_batch,
            batch_y=data['map_polygon']['batch'] if is_batch else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)

        # Agent-to-Mode relations
        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=agent_batch,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)

        # ===== 动态联合查询初始化 =====
        # 为每个智能体生成K个初始查询
        agent_queries_init = self.query_init(x_a)  # [A, K*D]
        agent_queries_init = agent_queries_init.view(-1, self.num_joint_modes, self.hidden_dim)  # [A, K, D]
        
        # 重新组织为 [K, A, D] 格式以便后续处理
        joint_queries = agent_queries_init.transpose(0, 1)  # [K, A, D]
        
        # 存储每个递归步骤的输出
        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps

        for t in range(self.num_recurrent_steps):
            # L_dec 层的注意力机制
            for i in range(self.num_layers):
                # 1. Mode2Time cross-attention
                queries_flat = joint_queries.reshape(-1, self.hidden_dim)  # [K*A, D]
                queries_flat = self.mode2time_propose_attn_layers[i]((x_t, queries_flat), r_t2m, edge_index_t2m)
                joint_queries = queries_flat.reshape(self.num_joint_modes, -1, self.hidden_dim)
                
                # 2. Mode2Map cross-attention
                queries_flat = joint_queries.reshape(-1, self.hidden_dim)  # [K*A, D]
                queries_flat = self.mode2map_propose_attn_layers[i]((x_pl, queries_flat), r_pl2m, edge_index_pl2m)
                joint_queries = queries_flat.reshape(self.num_joint_modes, -1, self.hidden_dim)
                
                # 3. Row-wise self-attention: 同一联合场景内的智能体交互
                # 需要按场景分别处理
                joint_queries_updated = joint_queries.clone()
                for b in range(batch_size):
                    # 获取当前场景的智能体掩码
                    scene_mask = (agent_batch == b)
                    if scene_mask.sum() == 0:
                        continue
                    
                    # 对每个联合模式应用自注意力
                    for k in range(self.num_joint_modes):
                        scene_queries = joint_queries[k, scene_mask]  # [A_scene, D]
                        if scene_queries.size(0) > 0:
                            updated_queries = self.row_self_attn_propose_layers[i](scene_queries, None, None)
                            joint_queries_updated[k, scene_mask] = updated_queries
                
                joint_queries = joint_queries_updated
            
            # 4. Column-wise self-attention: 不同联合场景间的通信
            # 对每个智能体的K个模式查询进行自注意力
            for a in range(joint_queries.size(1)):
                agent_modes = joint_queries[:, a, :]  # [K, D]
                updated_modes = self.col_self_attn_propose_layer(agent_modes, None, None)
                joint_queries[:, a, :] = updated_modes
            
            # 输出当前递归步骤的轨迹
            locs_propose_pos[t] = self.to_loc_propose_pos(joint_queries)
            scales_propose_pos[t] = self.to_scale_propose_pos(joint_queries)
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(joint_queries)
                concs_propose_head[t] = self.to_conc_propose_head(joint_queries)

        # 累积递归输出得到完整轨迹
        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(self.num_joint_modes, -1, self.num_future_steps, self.output_dim),
            dim=-2)
        scale_propose_pos = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos, dim=-1).view(self.num_joint_modes, -1, self.num_future_steps, self.output_dim),
                alpha=1.0) + 1.0,
            dim=-2) + 0.1
            
        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
                                            dim=-2)
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                                                    dim=-2) + 0.02)
            # 轨迹嵌入用于refinement
            joint_queries = self.y_emb(torch.cat([loc_propose_pos.detach(),
                                                  wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head = loc_propose_pos.new_zeros((self.num_joint_modes, loc_propose_pos.size(1), self.num_future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((self.num_joint_modes, scale_propose_pos.size(1), self.num_future_steps, 1))
            joint_queries = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))

        # ===== Anchor-Based Trajectory Refinement =====
        joint_queries = joint_queries.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        joint_queries = self.traj_emb(joint_queries, self.traj_emb_h0.unsqueeze(1).repeat(1, joint_queries.size(1), 1))[1].squeeze(0)
        joint_queries = joint_queries.reshape(self.num_joint_modes, -1, self.hidden_dim)
        
        # Refinement attention layers
        for i in range(self.num_layers):
            # Mode2Time refinement
            queries_flat = joint_queries.reshape(-1, self.hidden_dim)
            queries_flat = self.mode2time_refine_attn_layers[i]((x_t, queries_flat), r_t2m, edge_index_t2m)
            joint_queries = queries_flat.reshape(self.num_joint_modes, -1, self.hidden_dim)
            
            # Mode2Map refinement
            queries_flat = joint_queries.reshape(-1, self.hidden_dim)
            queries_flat = self.mode2map_refine_attn_layers[i]((x_pl, queries_flat), r_pl2m, edge_index_pl2m)
            joint_queries = queries_flat.reshape(self.num_joint_modes, -1, self.hidden_dim)
            
            # Row-wise self-attention需要按场景分别处理
            joint_queries_updated = joint_queries.clone()
            for b in range(batch_size):
                scene_mask = (agent_batch == b)
                if scene_mask.sum() == 0:
                    continue
                
                for k in range(self.num_joint_modes):
                    scene_queries = joint_queries[k, scene_mask]
                    if scene_queries.size(0) > 0:
                        updated_queries = self.row_self_attn_refine_layers[i](scene_queries, None, None)
                        joint_queries_updated[k, scene_mask] = updated_queries
            
            joint_queries = joint_queries_updated
        
        # Column-wise self-attention: 对每个智能体的K个模式查询进行自注意力
        for a in range(joint_queries.size(1)):
            agent_modes = joint_queries[:, a, :]  # [K, D]
            updated_modes = self.col_self_attn_refine_layer(agent_modes, None, None)
            joint_queries[:, a, :] = updated_modes

        # 精化输出
        loc_refine_pos = self.to_loc_refine_pos(joint_queries).view(self.num_joint_modes, -1, self.num_future_steps, self.output_dim)
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(joint_queries).view(self.num_joint_modes, -1, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
            
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(joint_queries).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(joint_queries).unsqueeze(-1)) + 1.0 + 0.02)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((self.num_joint_modes, loc_refine_pos.size(1), self.num_future_steps, 1))
            conc_refine_head = scale_refine_pos.new_zeros((self.num_joint_modes, scale_refine_pos.size(1), self.num_future_steps, 1))

        # ===== Scene Scoring with Attentive Pooling =====
        # 按场景计算置信度分数，使用注意力池化（按论文建议）
        scene_scores = []
        for k in range(self.num_joint_modes):
            batch_scores = []
            for b in range(batch_size):
                scene_mask = (agent_batch == b)
                if scene_mask.sum() == 0:
                    batch_scores.append(torch.tensor(0.0, device=joint_queries.device))
                    continue
                
                # 当前场景当前模式的智能体特征
                scene_queries = joint_queries[k, scene_mask]  # [A_scene, D]
                
                if scene_queries.size(0) == 1:
                    # 只有一个智能体，直接使用其特征
                    scene_embed = scene_queries.squeeze(0)  # [D]
                else:
                    # 使用注意力池化聚合多个智能体特征
                    # 计算注意力权重
                    attn_weights = self.attentive_pooling(scene_queries).squeeze(-1)  # [A_scene]
                    attn_weights = F.softmax(attn_weights, dim=0)  # [A_scene]
                    
                    # 加权聚合智能体特征
                    scene_embed = torch.sum(scene_queries * attn_weights.unsqueeze(-1), dim=0)  # [D]
                
                score = self.scene_scoring(scene_embed).squeeze()  # []
                batch_scores.append(score)
            
            # 所有场景的平均分数作为该模式的分数
            if len(batch_scores) > 0:
                mode_score = torch.stack(batch_scores).mean()
            else:
                mode_score = torch.tensor(0.0, device=joint_queries.device)
            scene_scores.append(mode_score)
        
        scene_pi = torch.stack(scene_scores)  # [K]

        return {
            # Proposal outputs: [K, A, T, D]
            'loc_propose_pos': loc_propose_pos,
            'scale_propose_pos': scale_propose_pos,
            'loc_propose_head': loc_propose_head,
            'conc_propose_head': conc_propose_head,
            # Refinement outputs: [K, A, T, D] 
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            # Scene-level confidence: [K]
            'scene_pi': scene_pi,
            # Joint queries for analysis: [K, A, D]
            'joint_queries': joint_queries,
            # Batch information for loss computation
            'agent_batch': agent_batch,
            'batch_size': batch_size,
        } 