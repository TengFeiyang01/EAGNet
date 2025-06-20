# 文件路径：QCNet/layers/edge_attention_layer.py

import torch
import torch.nn as nn

class EdgeAttentionLayer(nn.Module):
    """
    AgentFormer 风格的边注意力 (Edge‐Attention)：
    - 输入：
      • x (N, D): 节点特征向量矩阵（N 是节点数目，D 是隐藏维度）
      • edge_index (2, E): 边的索引，E 是边的数量，格式为 [source_nodes; target_nodes]
      • edge_attr (E, F): 每条边的属性（可以是相对位置编码、相对速度等，F 为属性维度）
    - 输出：
      • out (N, D): 经过边注意力更新后的节点特征
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
        Q = self.lin_q(x)  # (N, hidden_dim)
        K = self.lin_k(x)  # (N, hidden_dim)
        V = self.lin_v(x)  # (N, hidden_dim)

        # 分割到 (num_heads, head_dim)
        Q = Q.view(N, self.num_heads, self.head_dim)   # (N, num_heads, head_dim)
        K = K.view(N, self.num_heads, self.head_dim)   # (N, num_heads, head_dim)
        V = V.view(N, self.num_heads, self.head_dim)   # (N, num_heads, head_dim)

        # edge_index 中 source、target
        src, tgt = edge_index[0], edge_index[1]  # 均形状 (E,)

        # 提取对应边的节点特征
        K_tgt = K[tgt]    # (E, num_heads, head_dim)
        Q_src = Q[src]    # (E, num_heads, head_dim)
        V_tgt = V[tgt]    # (E, num_heads, head_dim)

        # 计算 Q·K^T / √head_dim
        attn_score = (Q_src * K_tgt).sum(dim=-1) / (self.head_dim ** 0.5)  # (E, num_heads)

        # 边属性映射到注意力偏置
        edge_bias = self.lin_edge(edge_attr)  # (E, num_heads)
        attn_score = attn_score + edge_bias

        # 对每个 head 分别 softmax，用于 target 聚合
        # 需要将 attention 得分从 (E, num_heads) 转到 (num_heads, N, ?)
        # 我们先把 (E, num_heads) 转置为 (num_heads, E)，再给 tgt 求 softmax
        attn_score = attn_score.transpose(0, 1)  # (num_heads, E)
        # 对每个 head，在同一个 tgt 节点下做 softmax
        attn_score = self._edge_softmax(attn_score, tgt, N)  # (num_heads, E)
        attn_score = attn_score.transpose(0, 1)  # (E, num_heads)

        attn_score = self.attn_dropout(attn_score)  # (E, num_heads)

        # 加权 V_tgt
        V_tgt = V_tgt  # (E, num_heads, head_dim)
        attn_score = attn_score.unsqueeze(-1)  # (E, num_heads, 1)
        out_edge = attn_score * V_tgt          # (E, num_heads, head_dim)

        # 将 (E, num_heads, head_dim) 聚合到目标节点 tgt
        out_edge = out_edge.view(E, -1)  # (E, hidden_dim)

        # 初始化节点输出为零
        out = x.new_zeros(N, self.hidden_dim)  # (N, hidden_dim)
        # 把 out_edge 按照 tgt 索引加到对应节点
        out = out.index_add(0, tgt, out_edge)  # 聚合 (N, hidden_dim)

        # 最后经过一个线性变换
        out = self.out_proj(out)  # (N, hidden_dim)
        return out

    @staticmethod
    def _edge_softmax(attn_score: torch.Tensor, tgt: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        对每个 head 上相同 tgt 节点下的边做 softmax。
        attn_score: (num_heads, E)
        tgt:       (E,) 对应边的 target 节点索引
        num_nodes: 节点总数 N
        返回：与 attn_score 形状相同 (num_heads, E)
        """
        num_heads, E = attn_score.size()
        # 1) 初始化输出
        out = torch.zeros_like(attn_score)  # (num_heads, E)

        # 2) 对每个 head 单独做
        for h in range(num_heads):
            # 将该 head 下 attn_score[h] 按 tgt 分类求 softmax
            # 先找到每条边的 tgt 节点
            scores_h = attn_score[h]  # (E,)

            # 为了对相同 tgt 做 softmax，我们可以先将 scores_h 分组，但这里直接用 scatter_softmax 更高效
            out[h] = EdgeAttentionLayer._scatter_softmax(scores_h, tgt, num_nodes)
        return out

    @staticmethod
    def _scatter_softmax(scores: torch.Tensor, tgt: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        在同一个 tgt 节点下，对 scores 做 softmax。
        scores: (E,) 这一 head 下的注意力得分
        tgt:    (E,) 每条边对应的 target 节点索引
        num_nodes: 节点总数 N
        返回：softmax 后的 scores，(E,)
        """
        E = scores.size(0)
        # 1) 计算每个 tgt 节点的最大值
        max_per_tgt = scores.new_full((num_nodes,), float('-inf'))
        
        # 手动实现 scatter max
        for i in range(E):
            node_idx = tgt[i]
            max_per_tgt[node_idx] = torch.max(max_per_tgt[node_idx], scores[i])
        
        # 将 -inf 替换为 0（对于没有边的节点）
        max_per_tgt = torch.where(max_per_tgt == float('-inf'), torch.zeros_like(max_per_tgt), max_per_tgt)

        # 将每条边的 scores 减去对应 tgt 聚合的最大值
        normalized_scores = scores - max_per_tgt[tgt]

        # 2) 指数化
        exp_scores = normalized_scores.exp()  # (E,)

        # 3) 对每个 tgt 分组做 sum
        sum_per_tgt = exp_scores.new_zeros((num_nodes,))
        sum_per_tgt.scatter_add_(0, tgt, exp_scores)
        sum_per_tgt = torch.where(sum_per_tgt == 0, torch.ones_like(sum_per_tgt), sum_per_tgt)

        # 4) 得到最终 softmax
        return exp_scores / sum_per_tgt[tgt]  # (E,)
