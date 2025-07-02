# 文件路径：QCNet/layers/edge_attention_layer.py

import torch
import torch.nn as nn
from torch_geometric.utils import softmax

class EdgeAttentionLayer(nn.Module):
    """
    高效的边注意力层 - 优化版本
    使用PyTorch向量化操作，避免手动循环
    """

    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int, dropout: float):
        super(EdgeAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # 将节点特征投影到多头查询 (Q)、键 (K)、值 (V)
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        # 线性变换矩阵：Q、K、V
        self.lin_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 把边属性映射到每个 head 的权重空间
        self.lin_edge = nn.Linear(edge_dim, num_heads, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x: 节点特征，形状 (N, hidden_dim)
        edge_index: 边索引，形状 (2, E)
        edge_attr: 边属性，形状 (E, edge_dim)
        返回：更新后的节点特征 (N, hidden_dim)
        """
        N = x.size(0)
        E = edge_index.size(1)

        # 投影为多头 Q、K、V
        Q = self.lin_q(x).view(N, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)
        K = self.lin_k(x).view(N, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)
        V = self.lin_v(x).view(N, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)

        # edge_index 中 source、target
        src, tgt = edge_index[0], edge_index[1]  # 均形状 (E,)

        # 提取对应边的节点特征
        Q_src = Q[src]    # (E, num_heads, head_dim)
        K_tgt = K[tgt]    # (E, num_heads, head_dim)
        V_tgt = V[tgt]    # (E, num_heads, head_dim)

        # 计算注意力分数：Q·K^T / √head_dim
        attn_score = (Q_src * K_tgt).sum(dim=-1) / (self.head_dim ** 0.5)  # (E, num_heads)

        # 边属性映射到注意力偏置
        edge_bias = self.lin_edge(edge_attr)  # (E, num_heads)
        attn_score = attn_score + edge_bias

        # 🚀 关键优化：使用PyTorch Geometric的高效softmax
        # 为每个头分别计算softmax
        attn_output = []
        for h in range(self.num_heads):
            # 对每个头，在相同target节点下做softmax
            attn_h = softmax(attn_score[:, h], tgt, num_nodes=N)  # (E,)
            attn_h = self.attn_dropout(attn_h)
            
            # 加权聚合值
            out_h = attn_h.unsqueeze(-1) * V_tgt[:, h, :]  # (E, head_dim)
            
            # 按target聚合
            aggregated_h = torch.zeros(N, self.head_dim, device=x.device, dtype=x.dtype)
            aggregated_h.index_add_(0, tgt, out_h)  # (N, head_dim)
            
            attn_output.append(aggregated_h)

        # 拼接所有头的输出
        out = torch.cat(attn_output, dim=1)  # (N, hidden_dim)

        # 最终线性变换
        out = self.out_proj(out)  # (N, hidden_dim)
        return out
