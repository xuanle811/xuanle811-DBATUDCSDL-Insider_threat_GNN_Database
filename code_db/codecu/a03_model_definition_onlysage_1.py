"""
a03_model_definition.py — ATHITD (SAGE) + RGCN + GAT baselines
===============================================================
SỬA LẠI DÙNG CẢ B + TSA CHO CẢ 3 THẰNG CÙNG CÔNG BẰNG
3 model dùng chung InsiderThreatDetector wrapper, phân biệt qua gnn_type:

    "sage" → SAGEKernel  + Lớp B (RelProj + SemanticAttn) + TSA   ← INDUCTIVE
    "rgcn" → RGCNKernel  (Relational GCN per relation type)         ← TRANSDUCTIVE
    "gat"  → GATKernel   (per-edge attention, không có Lớp B + TSA) ← TRANSDUCTIVE

Tại sao SAGE inductive, RGCN/GAT transductive:
    SAGEConv: h_i' = W·[h_i || mean(h_N(i))]
        → chỉ cần neighbor features tại inference time
        → node mới chỉ cần kéo neighbor hiện có → không retrain
    RGCNConv: H' = Σ_r W_r · Â_r · H   (Â_r = normalized adjacency per relation)
        → phụ thuộc Â toàn cục → node mới làm Â thay đổi → phải retrain
    GATConv: tương tự RGCN về graph-level dependency

Fair comparison đảm bảo:
    RGCN/GAT KHÔNG có Lớp B (SemanticAttention) và KHÔNG có TSA
    → SAGE được credit thêm từ 2 component này → kết quả hợp lệ cho paper

Lý do chọn RGCN thay GCN thuần:
    RGCN thiết kế cho heterogeneous/relational graph với W_r riêng per relation
    → fair comparison hơn GCN thuần (1 W chung cho mọi relation type)

Bugs đã sửa:
    C1: ModuleDict nhận tuple key → TypeError khi khởi tạo
        FIX: _ekey() chuyển tuple → string "src||rel||dst" (dùng || tránh conflict)
    C2: _pass() tra edge_index_dict bằng string, không match tuple key HeteroData
        FIX: lưu edge_types_list riêng, zip với conv, tra bằng tuple
    C3: RGCNConv API sai với hetero local index
        FIX: implement thủ công: concat global tensor, reindex, scatter_add
    C4: print(edge_index_dict.keys()) debug còn sót
        FIX: đã xóa
    C5: RGCN/GAT khởi tạo Lớp B + TSA → baseline bị bias
        FIX: chỉ SAGE có Lớp B + TSA
    C6 (MỚI): _ekey dùng "__" separator → split sai nếu rel chứa "__"
        FIX: dùng "||" làm separator, parse bằng split("||", maxsplit=2)
    C7 (MỚI): biến _ trong `src_t, _, dst_t = et` bị dùng lại trong str(k[1]) == _
        FIX: dùng tên biến rel_t rõ ràng
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — key conversion an toàn
# ══════════════════════════════════════════════════════════════════════════════

def _ekey(et: tuple) -> str:
    """
    Chuyển edge_type tuple (src, rel, dst) → string key cho ModuleDict.
    Dùng "||" làm separator (không bao giờ xuất hiện trong tên node/edge type).
    C6 FIX: tránh dùng "__" vì rel_name có thể chứa "__" (vd: rev_has_usb).
    """
    return f"{et[0]}||{et[1]}||{et[2]}"


def _ekey_parse(key: str) -> tuple:
    """Parse string key → (src, rel, dst) tuple."""
    parts = key.split("||", maxsplit=2)
    return tuple(parts)   # (src, rel, dst)


def _lookup_ei(edge_index_dict: dict, src_t: str, rel_t: str, dst_t: str):
    """
    Tra edge_index từ edge_index_dict.
    HeteroData lưu key là tuple của strings hoặc EdgeType object.
    Thử tuple string trước, fallback scan nếu cần.
    """
    # Thử trực tiếp với string tuple
    key_str = (src_t, rel_t, dst_t)
    ei = edge_index_dict.get(key_str)
    if ei is not None:
        return ei
    # Fallback: scan và so sánh string representation
    for k, v in edge_index_dict.items():
        if str(k[0]) == src_t and str(k[1]) == rel_t and str(k[2]) == dst_t:
            return v
    return None


# ══════════════════════════════════════════════════════════════════════════════
# INPUT PROJECTION — dùng chung cho cả 3 model
# ══════════════════════════════════════════════════════════════════════════════

class InputProjection(nn.Module):
    """Chiếu từng node type từ chiều thô → hidden_channels đồng nhất."""
    def __init__(self, in_channels_dict: dict, hidden_channels: int):
        super().__init__()
        self.projs = nn.ModuleDict({
            nt: nn.Linear(dim, hidden_channels)
            for nt, dim in in_channels_dict.items()
        })

    def forward(self, x_dict: dict) -> dict:
        return {nt: F.relu(self.projs[nt](x.float()))
                for nt, x in x_dict.items() if nt in self.projs}
#=====================Module convert edge attributes=======================
class EdgeAttrProjector(nn.Module):
    """Map edge features thành vector cùng hidden_dim với node embeddings."""
    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, edge_attr):
        # edge_attr: [num_edges, edge_dim]
        return self.mlp(edge_attr)  # output: [num_edges, hidden_dim]
    
# ══════════════════════════════════════════════════════════════════════════════
# SAGE KERNEL — INDUCTIVE
# ══════════════════════════════════════════════════════════════════════════════

class SAGEKernel(nn.Module):
    """
    2-layer SAGEConv trên heterogeneous graph.
    h_i' = W · [h_i || mean_{j∈N(i)}(h_j)]
    INDUCTIVE: chỉ cần neighbor features → node mới inference được.
    """
    def __init__(self, metadata, in_channels_dict: dict, hidden_channels: int, edge_dims: list[int]):
        super().__init__()
        self.node_types    = metadata[0]
        self.edge_types    = metadata[1]          # list of tuples, giữ nguyên
        self.input_proj    = InputProjection(in_channels_dict, hidden_channels)
        assert len(edge_dims) == len(self.edge_types), \
            "edge_dims phải có cùng độ dài với metadata[1] (edge_types)."

        # edge_dims: list số chiều của edge features theo thứ tự edge_types
        self.edge_projectors = nn.ModuleDict({
            _ekey(et): EdgeAttrProjector(edge_dim=edge_dims[i], hidden_dim=hidden_channels)
            for i, et in enumerate(self.edge_types)
        })
        # C1 FIX + C6 FIX: string key dùng "||" separator
        self.conv1 = nn.ModuleDict({
            _ekey(et): SAGEConv((hidden_channels, hidden_channels),
                                hidden_channels, normalize=True)
            for et in self.edge_types
        })
        self.conv2 = nn.ModuleDict({
            _ekey(et): SAGEConv((hidden_channels, hidden_channels),
                                hidden_channels, normalize=True)
            for et in self.edge_types
        })


    def _pass(self, conv_dict: nn.ModuleDict,
              h_dict: dict, edge_index_dict: dict, edge_attr_dict: dict) -> dict:
        accum = {nt: [] for nt in self.node_types}

        # C2 FIX: zip conv_dict keys với edge_types list để có tuple
        # C7 FIX: đặt tên rel_t rõ ràng, không dùng _
        for et, (ekey, conv) in zip(self.edge_types, conv_dict.items()):
            src_t, rel_t, dst_t = et

            ei = _lookup_ei(edge_index_dict, src_t, rel_t, dst_t)
            if ei is None or ei.numel() == 0:
                continue

            h_src = h_dict.get(src_t)
            h_dst = h_dict.get(dst_t)
            if h_src is None or h_dst is None:
                continue

            # Edge feature
            edge_attr = edge_attr_dict.get((src_t, rel_t, dst_t))
            if edge_attr is not None:
                edge_vec = self.edge_projectors[ekey](edge_attr)
                h_src_msg = h_src[ei[0]] + edge_vec
            else:
                h_src_msg = h_src[ei[0]]

            accum[dst_t].append(conv((h_src_msg, h_dst), ei))


        result = {}
        for nt in self.node_types:
            base = h_dict.get(nt)
            if accum[nt]:
                agg = torch.stack(accum[nt], dim=0).sum(dim=0)
                result[nt] = F.relu(agg + base) if base is not None else F.relu(agg)
            else:
                result[nt] = base
        return result

    def forward(self, x_dict: dict, edge_index_dict: dict, edge_attr_dict: dict) -> dict:
        h  = self.input_proj(x_dict)
        h1 = self._pass(self.conv1, h,  edge_index_dict,edge_attr_dict )
        h1 = {nt: F.dropout(v, p=0.3, training=self.training)
              for nt, v in h1.items()}
        h2 = self._pass(self.conv2, h1, edge_index_dict, edge_attr_dict)
        return h2


# ══════════════════════════════════════════════════════════════════════════════
# RGCN KERNEL — TRANSDUCTIVE
# ══════════════════════════════════════════════════════════════════════════════

class RGCNKernel(nn.Module):
    """
    Relational GCN (Schlichtkrull et al. 2018).
    h_i' = W_0·h_i  +  Σ_r  (1/c_{i,r}) · Σ_{j∈N_r(i)} W_r·h_j

    C3 FIX: implementation đúng cho hetero-graph với local index:
        1. Không concat global tensor (gây nhầm lẫn index).
        2. Với mỗi relation r: dùng scatter_add_ trực tiếp trên local dst index.
           Accumulator kích thước [N_dst, H] riêng per node type.
        3. Normalize, cộng W_0·h_i, relu.

    TRANSDUCTIVE: W_r tính trên adjacency toàn cục của graph hiện tại.
    Node mới → adjacency thay đổi → phải retrain.
    """
    def __init__(self, metadata, in_channels_dict: dict, hidden_channels: int, edge_dims: list[int]):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.hidden     = hidden_channels

        self.input_proj = InputProjection(in_channels_dict, hidden_channels)

        assert len(edge_dims) == len(self.edge_types), \
            "edge_dims phải khớp số lượng edge_types"
        
        self.edge_projectors = nn.ModuleDict({
            _ekey(et): EdgeAttrProjector(edge_dim=edge_dims[i],
                                         hidden_dim=hidden_channels)
            for i, et in enumerate(self.edge_types)
        })

        # W_r per relation: lưu theo index (tránh key conflict)
        self.W_r = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels, bias=False)
            for _ in range(len(self.edge_types))
        ])
        # W_0 per node type (self-loop)
        self.W_0 = nn.ModuleDict({
            nt: nn.Linear(hidden_channels, hidden_channels, bias=False)
            for nt in self.node_types
        })

    def _one_layer(self, h_dict: dict, edge_index_dict: dict, edge_attr_dict: dict) -> dict:
        H      = self.hidden
        device = next(iter(h_dict.values())).device

        # Accumulator per node type (local index)
        accum = {nt: torch.zeros_like(h_dict[nt]) for nt in h_dict}
        count = {nt: torch.zeros(h_dict[nt].shape[0], 1, device=device)
                 for nt in h_dict}

        for r_idx, et in enumerate(self.edge_types):
            src_t, rel_t, dst_t = et

            ei = _lookup_ei(edge_index_dict, src_t, rel_t, dst_t)
            if ei is None or ei.numel() == 0:
                continue

            h_src = h_dict.get(src_t)
            h_dst = h_dict.get(dst_t)
            if h_src is None or h_dst is None:
                continue

            src_idx = ei[0]   # local index of source nodes [E]
            dst_idx = ei[1]   # local index of dest nodes   [E]

            # Edge attributes
            edge_attr = edge_attr_dict.get((src_t, rel_t, dst_t))
            if edge_attr is not None:
                ekey = _ekey(et)
                edge_vec = self.edge_projectors[ekey](edge_attr)  # [E, H]
                msg_in = h_src[src_idx] + edge_vec
            else:
                msg_in = h_src[src_idx]
                
            msg = self.W_r[r_idx](msg_in)   # [E, H]

            # Scatter vào local accumulator của dst
            accum[dst_t].scatter_add_(
                0, dst_idx.unsqueeze(-1).expand(-1, H), msg)
            count[dst_t].scatter_add_(
                0, dst_idx.unsqueeze(-1),
                torch.ones(dst_idx.shape[0], 1, device=device))

        result = {}
        for nt in self.node_types:
            if nt not in h_dict:
                continue
            h_agg   = accum[nt] / count[nt].clamp(min=1.0)
            h_self  = self.W_0[nt](h_dict[nt])
            result[nt] = F.relu(h_agg + h_self)
        return result

    def forward(self, x_dict: dict, edge_index_dict: dict, edge_attr_dict: dict) -> dict:
        h  = self.input_proj(x_dict)
        h1 = self._one_layer(h,  edge_index_dict, edge_attr_dict)
        h1 = {nt: F.dropout(v, p=0.3, training=self.training)
              for nt, v in h1.items()}
        h2 = self._one_layer(h1, edge_index_dict, edge_attr_dict)
        return h2


# ══════════════════════════════════════════════════════════════════════════════
# GAT KERNEL — TRANSDUCTIVE
# ══════════════════════════════════════════════════════════════════════════════

class GATKernel(nn.Module):
    """
    2-layer GATConv trên heterogeneous graph.
    Có per-edge attention α(i,j) — gần với SAGE Lớp A.
    Nhưng KHÔNG có relation-level β (Lớp B) và KHÔNG có TSA.
    TRANSDUCTIVE: phụ thuộc graph structure toàn cục.
    """
    def __init__(self, metadata, in_channels_dict: dict, hidden_channels: int, edge_dims: list[int]):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.input_proj = InputProjection(in_channels_dict, hidden_channels)

        assert len(edge_dims) == len(self.edge_types), \
            "edge_dims phải khớp số lượng edge_types"
        
        self.edge_projectors = nn.ModuleDict({
            _ekey(et): EdgeAttrProjector(edge_dim=edge_dims[i],
                                         hidden_dim=hidden_channels)
            for i, et in enumerate(self.edge_types)
        })

        # C1+C6 FIX: "||" separator  |  C3/A3 FIX: add_self_loops=False
        self.conv1 = nn.ModuleDict({
            _ekey(et): GATConv((hidden_channels, hidden_channels),
                               hidden_channels, heads=1,
                               add_self_loops=False, concat=False, edge_dim=edge_dims[i])
            for i, et in enumerate(self.edge_types)
        })

        self.conv2 = nn.ModuleDict({
            _ekey(et): GATConv((hidden_channels, hidden_channels),
                               hidden_channels, heads=1,
                               add_self_loops=False, concat=False, edge_dim=edge_dims[i])
            for i, et in enumerate(self.edge_types)
        })

    def _pass(self, conv_dict: nn.ModuleDict,
              h_dict: dict, edge_index_dict: dict,
             edge_attr_dict: dict) -> dict:
        accum = {nt: [] for nt in self.node_types}

        # C2+C7 FIX: zip để có tuple et song song
        for et, (ekey, conv) in zip(self.edge_types, conv_dict.items()):
            src_t, rel_t, dst_t = et   # C7 FIX: rel_t thay vì _

            ei = _lookup_ei(edge_index_dict, src_t, rel_t, dst_t)
            if ei is None or ei.numel() == 0:
                continue

            h_src = h_dict.get(src_t)
            h_dst = h_dict.get(dst_t)
            if h_src is None or h_dst is None:
                continue

            # thêm thuộc tính cạnh
            edge_attr = edge_attr_dict.get((src_t, rel_t, dst_t))
            if edge_attr is not None:
                edge_vec = self.edge_projectors[ekey](edge_attr)
                h_src_in = h_src + 0  # đảm bảo không in-place
                # dùng index src của từng edge
                h_src_in = h_src_in.clone()
                h_src_in[ei[0]] = h_src_in[ei[0]] + edge_vec
                out = conv((h_src_in, h_dst), ei)
            else:
                out = conv((h_src, h_dst), ei)

            accum[dst_t].append(out)

        result = {}
        for nt in self.node_types:
            base = h_dict.get(nt)
            if accum[nt]:
                agg = torch.stack(accum[nt], dim=0).sum(dim=0)
                result[nt] = F.relu(agg + base) if base is not None else F.relu(agg)
            else:
                result[nt] = base
        return result

    def forward(self, x_dict: dict, edge_index_dict: dict, edge_attr_dict: dict) -> dict:
        h  = self.input_proj(x_dict)
        h1 = self._pass(self.conv1, h,  edge_index_dict, edge_attr_dict)
        h1 = {nt: F.dropout(F.elu(v), p=0.3, training=self.training)
              for nt, v in h1.items()}
        h2 = self._pass(self.conv2, h1, edge_index_dict, edge_attr_dict)
        return h2


# ══════════════════════════════════════════════════════════════════════════════
# LỚP B — chỉ SAGE dùng
# ══════════════════════════════════════════════════════════════════════════════

class RelationProjector(nn.Module):
    """
    Chiếu session_emb → n_relations embeddings KHÁC NHAU.
    Điều kiện cần để SemanticAttentionLayer học α có ý nghĩa.
    """
    def __init__(self, hidden_channels: int, n_relations: int):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(n_relations)
        ])

    def forward(self, x: torch.Tensor) -> list:
        return [F.relu(proj(x)) for proj in self.projectors]


class SemanticAttentionLayer(nn.Module):
    """
    β_r = softmax_r( mean_n(tanh(W·h_n^(r))) · w_r )
    H   = Σ_r β_r · H^(r)
    Ý nghĩa: DELETE/DROP tự động nhận β cao hơn SELECT.
    """
    def __init__(self, hidden_channels: int, num_relation_types: int):
        super().__init__()
        self.w_r  = nn.Parameter(torch.empty(num_relation_types, hidden_channels))
        nn.init.xavier_uniform_(self.w_r.unsqueeze(0))
        self.proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, emb_list: list) -> torch.Tensor:
        n_r     = len(emb_list)
        stacked = torch.stack(emb_list, dim=0)                     # [R, N, H]
        proj    = torch.tanh(self.proj(stacked))                   # [R, N, H]
        scores  = torch.einsum('rnh,rh->rn', proj, self.w_r[:n_r])# [R, N]
        alpha   = F.softmax(scores, dim=0).unsqueeze(-1)           # [R, N, 1]
        return (alpha * stacked).sum(dim=0)                        # [N, H]


# ══════════════════════════════════════════════════════════════════════════════
# THÀNH PHẦN 3: TSA — chỉ SAGE dùng
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div) if d % 2 == 0 else torch.cos(pos * div[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalSelfAttentionModule(nn.Module):
    """
    Input : [N_sess, T, H] — chuỗi T session embeddings theo thời gian
    Output: [N_sess, H]    — V_intent
    """
    def __init__(self, d: int, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        while d % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.pos_enc = PositionalEncoding(d, max_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads,
            dim_feedforward=d * 4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, seq: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        x = self.pos_enc(seq)
        x = self.transformer(x, src_key_padding_mask=mask)
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            return (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return x.mean(dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# InsiderThreatDetector — wrapper cho cả 3 model
# ══════════════════════════════════════════════════════════════════════════════

class InsiderThreatDetector(nn.Module):
    """
    API thống nhất: logits = model.forward_temporal(snapshot_list)

    gnn_type = "sage" → ATHITD đầy đủ     → INDUCTIVE
    gnn_type = "rgcn" → RGCN chỉ Lớp A    → TRANSDUCTIVE
    gnn_type = "gat"  → GAT chỉ Lớp A     → TRANSDUCTIVE
    """
    def __init__(self, metadata, in_channels_dict: dict,
                 hidden_channels: int = 64, num_heads: int = 4,
                 temporal_layers: int = 2, dropout: float = 0.1,
                 gnn_type: str = "sage",
                 edge_dims: list[int] = None):
        super().__init__()
        self.metadata    = metadata
        self.n_relations = len(metadata[1])
        self.gnn_type    = gnn_type
        if edge_dims is None:
            raise ValueError("Error: Edge dim = None")

        if gnn_type == "sage":
            self.encoder = SAGEKernel(metadata, in_channels_dict, hidden_channels, edge_dims)
        elif gnn_type == "rgcn":
            self.encoder = RGCNKernel(metadata, in_channels_dict, hidden_channels, edge_dims)
        elif gnn_type == "gat":
            self.encoder = GATKernel(metadata, in_channels_dict, hidden_channels, edge_dims)
        else:
            raise ValueError(f"gnn_type không hợp lệ: '{gnn_type}'. "
                             f"Chọn: 'sage', 'rgcn', 'gat'")

        # C5 FIX: Lớp B + TSA CHỈ cho SAGE
        if gnn_type == "sage":
            self.rel_projector = RelationProjector(hidden_channels, self.n_relations)
            self.semantic_attn = SemanticAttentionLayer(hidden_channels, self.n_relations)
            self.temporal      = TemporalSelfAttentionModule(
                hidden_channels, num_heads, temporal_layers, dropout)
        else:
            self.rel_projector = None
            self.semantic_attn = None
            self.temporal      = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def _encode_snapshot(self, snapshot, edge_attr_dict) -> torch.Tensor:
        node_emb = self.encoder(snapshot.x_dict, snapshot.edge_index_dict, edge_attr_dict=snapshot.edge_attr_dict) # << Bổ sung trường này vào huấn luyện
        sess_emb = node_emb['session']
        if self.gnn_type == "sage":
            sess_emb = self.semantic_attn(self.rel_projector(sess_emb))
        return sess_emb

    def forward_temporal(self, snapshot_list: list, edge_attr_dict: dict = None) -> torch.Tensor:
        """Trả về RAW LOGITS [N_sess] — trước sigmoid."""
        if edge_attr_dict is None:
            raise ValueError("Cần truyền edge_attr_dict vào forward_temporal.")
        
        if self.gnn_type == "sage":
            per_snap = [self._encode_snapshot(s, edge_attr_dict) for s in snapshot_list]
            seq      = torch.stack(per_snap, dim=1)   # [N_sess, T, H]
            repr_t   = self.temporal(seq)              # [N_sess, H]
        else:
            # RGCN/GAT: chỉ snapshot cuối, không có temporal
            repr_t = self._encode_snapshot(snapshot_list[-1], edge_attr_dict)
        return self.classifier(repr_t).squeeze(-1)
