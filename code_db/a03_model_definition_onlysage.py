"""
a03_model_definition.py — ATHITD (SAGE) + RGCN + GAT baselines
==============================================================

    "sage" → SAGEKernel  + Lớp B (RelProj + SemanticAttn) + TSA   ← INDUCTIVE
    "rgcn" → RGCNKernel  (Relational GCN per relation type)         ← TRANSDUCTIVE
    "gat"  → GATKernel   (per-edge attention, không có Lớp B + TSA) ← TRANSDUCTIVE

Thành phần 1: THGC - Temporal Heterogeneous Graph Constructor
             (Thực hiện ở a02 - xây dựng đồ thị con theo cửa sổ thời gian)

Thành phần 2: ASTHE Encoder (Asymmetric Spatial-semantic Heterogeneous Encoder)
   - Lớp A: Structural Message Passing (GraphSAGE quy nạp via to_hetero)
   - Lớp B: Semantic Attention Layer — học trọng số α cho từng loại cạnh

Thành phần 3: Temporal Self-Attention Module
   - Multi-Head Self-Attention trên chuỗi graph embeddings
   - Thêm Positional Encoding
   - Tầng Linear + Sigmoid để tính Anomaly Score
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

    Neighbor Sampling:
        num_neighbors_per_layer = [k1, k2]
        - Layer 1: mỗi cạnh chỉ sample tối đa k1 neighbor ngẫu nhiên
        - Layer 2: mỗi cạnh chỉ sample tối đa k2 neighbor ngẫu nhiên
        None → không sample (dùng toàn bộ neighbor, như cũ)
    """
    def __init__(self, metadata, in_channels_dict: dict, hidden_channels: int,
                 edge_dims: list[int],
                 num_neighbors_per_layer: list[int] = None):
        super().__init__()
        self.node_types    = metadata[0]
        self.edge_types    = metadata[1]          # list of tuples, giữ nguyên
        self.input_proj    = InputProjection(in_channels_dict, hidden_channels)
        # num_neighbors_per_layer: [k_layer1, k_layer2], None = không sample
        self.num_neighbors_per_layer = num_neighbors_per_layer or [None, None]
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


    @staticmethod
    def _sample_neighbors(ei: torch.Tensor, k: int) -> torch.Tensor:
        """
        Sample tối đa k neighbor ngẫu nhiên cho mỗi dst node.
        ei: [2, E] — (src_idx, dst_idx)
        Trả về ei_sampled: [2, E'] với E' ≤ E.
        Nếu k is None hoặc E ≤ k → trả nguyên ei.
        """
        if k is None or ei.size(1) <= k:
            return ei
        dst_idx = ei[1]
        # Permute ngẫu nhiên toàn bộ cạnh, sau đó giữ lại k cạnh đầu mỗi dst
        perm    = torch.randperm(ei.size(1), device=ei.device)
        ei_perm = ei[:, perm]
        dst_perm = ei_perm[1]
        # Sắp xếp theo dst để group, rồi lấy k đầu mỗi group
        order   = torch.argsort(dst_perm, stable=True)
        ei_sort = ei_perm[:, order]
        dst_sort = ei_sort[1]
        # Đếm và giới hạn k per dst
        _, counts = torch.unique_consecutive(dst_sort, return_counts=True)
        keep_mask = torch.zeros(ei_sort.size(1), dtype=torch.bool, device=ei.device)
        pos = 0
        for c in counts:
            c = c.item()
            keep_mask[pos: pos + min(c, k)] = True
            pos += c
        return ei_sort[:, keep_mask]

    def _pass(self, conv_dict: nn.ModuleDict,
              h_dict: dict, edge_index_dict: dict, edge_attr_dict: dict,
              k_neighbors: int = None) -> dict:
        accum = {nt: [] for nt in self.node_types}

        # C2 FIX: zip conv_dict keys với edge_types list để có tuple
        # C7 FIX: đặt tên rel_t rõ ràng, không dùng _
        for et, (ekey, conv) in zip(self.edge_types, conv_dict.items()):
            src_t, rel_t, dst_t = et

            ei_full = _lookup_ei(edge_index_dict, src_t, rel_t, dst_t)
            if ei_full is None or ei_full.numel() == 0:
                continue

            # ── Neighbor Sampling ────────────────────────────────────────────
            # Lấy mask vị trí cạnh được chọn để lọc edge_attr đồng bộ
            ei_full_size = ei_full.size(1)
            ei = self._sample_neighbors(ei_full, k_neighbors)
            # Tính mask vị trí cạnh còn lại (dùng để lọc edge_attr)
            sampled = (ei.size(1) < ei_full_size)

            h_src = h_dict.get(src_t)
            h_dst = h_dict.get(dst_t)
            if h_src is None or h_dst is None:
                continue

            # Edge feature: scatter edge_vec vào node ĐÍCH (dst) sau SAGEConv
            # Lý do dùng dst: cạnh user→session mang thông tin session nhận từ user
            # scatter vào src sẽ làm ô nhiễm embedding của nguồn gửi, không đúng chiều
            edge_attr_full = edge_attr_dict.get((src_t, rel_t, dst_t))

            # Nếu đã sample, cần xây dựng lại edge_attr tương ứng với ei đã sample.
            # Do _sample_neighbors shuffle + sort, ta tìm lại mapping qua src-dst pair.
            if edge_attr_full is not None and sampled:
                # Tạo key (src, dst) cho ei_full → map sang index
                src_full = ei_full[0]; dst_full = ei_full[1]
                full_keys = src_full * (h_dst.size(0) + 1) + dst_full  # unique hash
                samp_keys = ei[0]    * (h_dst.size(0) + 1) + ei[1]
                # Tìm index tương ứng trong ei_full (nearest match)
                idx_map = torch.bucketize(samp_keys, full_keys.sort()[0])
                idx_map = idx_map.clamp(0, ei_full_size - 1)
                sorted_idx = full_keys.argsort()
                edge_attr = edge_attr_full[sorted_idx[idx_map]]
            else:
                edge_attr = edge_attr_full

            if edge_attr is not None and edge_attr.size(0) != ei.size(1):
                # fallback: bỏ qua edge_attr khi kích thước không khớp sau sampling
                edge_attr = None

            # Bước 1: SAGEConv aggregate thông tin cấu trúc (src → dst)
            sage_out = conv((h_src, h_dst), ei)   # [N_dst, H]

            if edge_attr is not None:
                edge_vec = self.edge_projectors[ekey](edge_attr)   # [E, H]
                # Bước 2: scatter edge_vec vào dst — thêm thông tin ngữ nghĩa cạnh
                edge_bias = torch.zeros_like(sage_out)
                edge_bias.scatter_add_(
                    0,
                    ei[1].unsqueeze(-1).expand(-1, sage_out.size(-1)),  # dst index
                    edge_vec
                )
                accum[dst_t].append(sage_out + edge_bias)
            else:
                accum[dst_t].append(sage_out)


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
        k1, k2 = self.num_neighbors_per_layer[0], self.num_neighbors_per_layer[1]
        h  = self.input_proj(x_dict)
        h1 = self._pass(self.conv1, h,  edge_index_dict, edge_attr_dict, k_neighbors=k1)
        h1 = {nt: F.dropout(v, p=0.3, training=self.training)
              for nt, v in h1.items()}
        h2 = self._pass(self.conv2, h1, edge_index_dict, edge_attr_dict, k_neighbors=k2)
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
            if edge_attr is not None and edge_attr.size(0) != ei.size(1):
                raise ValueError(
                    f"edge_attr mismatch at {(src_t, rel_t, dst_t)}: "
                    f"edge_index has {ei.size(1)} edges, "
                    f"edge_attr has {edge_attr.size(0)} rows"
                )
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
            if edge_attr is not None and edge_attr.size(0) != ei.size(1):
                raise ValueError(
                    f"edge_attr mismatch at {(src_t, rel_t, dst_t)}: "
                    f"edge_index has {ei.size(1)} edges, "
                    f"edge_attr has {edge_attr.size(0)} rows"
                )
            # if edge_attr is not None:
            #     edge_vec = self.edge_projectors[ekey](edge_attr)
            #     h_src_in = h_src + 0  # đảm bảo không in-place
            #     # dùng index src của từng edge
            #     h_src_in = h_src_in.clone()
            #     h_src_in[ei[0]] = h_src_in[ei[0]] + edge_vec
            #     out = conv((h_src_in, h_dst), ei)
            # else:
            #     out = conv((h_src, h_dst), ei)
            if edge_attr is not None:
                out = conv((h_src, h_dst), ei, edge_attr=edge_attr.float())
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
# EMBEDDING CALIBRATOR — chuẩn hoá phân phối embedding về N(0,1)
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingCalibrator(nn.Module):
    """
    Đưa tất cả vector nhúng của mọi snapshot (cũ hay mới) về chung
    một không gian phân phối chuẩn N(0, 1).

    Cách hoạt động:
        1. Nhận danh sách embedding [E_1, E_2, ..., E_T] từ T snapshot
           (mỗi E_t có shape [N_t, H], N_t có thể khác nhau).
        2. Concat toàn bộ → tính global mean/std.
        3. Normalize từng E_t theo global mean/std (instance-like, nhưng global).
        4. Áp thêm learnable affine: γ · x̂ + β  (LayerNorm-style).

    Tại sao cần:
        - Train snapshot và test snapshot có phân phối node khác nhau
          (user/session mới, period khác).
        - Chuẩn hoá chung giảm covariate shift trước khi đưa vào TSA.
        - γ, β học cách "re-scale" phù hợp với downstream task.
    """
    def __init__(self, hidden_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(hidden_channels))
        self.beta  = nn.Parameter(torch.zeros(hidden_channels))

    def forward(self, emb_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        emb_list : list of [N_t, H] tensors (T snapshots, N_t có thể ≠ nhau)
        Returns  : list of [N_t, H] tensors đã calibrate
        """
        # Concat toàn bộ để tính global statistics
        all_emb = torch.cat(emb_list, dim=0)          # [ΣN_t, H]
        mean    = all_emb.mean(dim=0, keepdim=True)   # [1, H]
        std     = all_emb.std(dim=0, keepdim=True).clamp(min=self.eps)  # [1, H]

        # Normalize từng snapshot theo global stats
        calibrated = []
        for emb in emb_list:
            x_hat = (emb - mean) / std                # [N_t, H]
            calibrated.append(self.gamma * x_hat + self.beta)
        return calibrated




class InsiderThreatDetector(nn.Module):
    """
    API thống nhất: logits = model.forward_temporal(snapshot_list)

    gnn_type = "sage" → ATHITD đầy đủ     → INDUCTIVE
    gnn_type = "rgcn" → RGCN chỉ Lớp A    → TRANSDUCTIVE
    gnn_type = "gat"  → GAT chỉ Lớp A     → TRANSDUCTIVE

    Tham số mới (chỉ ảnh hưởng SAGE):
        num_neighbors_per_layer : [k1, k2] — neighbor sampling per layer.
                                  None = không sample.
        use_calibration         : bool — bật EmbeddingCalibrator trước TSA.
    """
    def __init__(self, metadata, in_channels_dict: dict,
                 hidden_channels: int = 64, num_heads: int = 4,
                 temporal_layers: int = 2, dropout: float = 0.1,
                 gnn_type: str = "sage",
                 edge_dims: list[int] = None,
                 num_neighbors_per_layer: list[int] = None,
                 use_calibration: bool = True):
        super().__init__()
        self.metadata    = metadata
        self.n_relations = len(metadata[1])
        self.gnn_type    = gnn_type
        if edge_dims is None:
            raise ValueError("Error: Edge dim = None")

        if gnn_type == "sage":
            self.encoder = SAGEKernel(metadata, in_channels_dict, hidden_channels,
                                      edge_dims, num_neighbors_per_layer)
        elif gnn_type == "rgcn":
            self.encoder = RGCNKernel(metadata, in_channels_dict, hidden_channels, edge_dims)
        elif gnn_type == "gat":
            self.encoder = GATKernel(metadata, in_channels_dict, hidden_channels, edge_dims)
        else:
            raise ValueError(f"gnn_type không hợp lệ: '{gnn_type}'. "
                             f"Chọn: 'sage', 'rgcn', 'gat'")

        # Lớp B + TSA CHỈ cho SAGE
        if gnn_type == "sage":
            self.rel_projector = RelationProjector(hidden_channels, self.n_relations)
            self.semantic_attn = SemanticAttentionLayer(hidden_channels, self.n_relations)
            self.temporal      = TemporalSelfAttentionModule(
                hidden_channels, num_heads, temporal_layers, dropout)
            # Calibration Embedding: chuẩn hoá phân phối qua các snapshot
            self.calibrator = EmbeddingCalibrator(hidden_channels) if use_calibration else None
        else:
            self.rel_projector = None
            self.semantic_attn = None
            self.temporal      = None
            self.calibrator    = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def _encode_snapshot(self, snapshot) -> torch.Tensor:
        """
        Encode 1 snapshot → user embedding [N_user, H].

        edge_attr đọc trực tiếp từ snapshot — đảm bảo mỗi snapshot
        Quy trình:
                Hetero GNN → session embedding
            Semantic Attention trên session
            Pool session → user qua cạnh user-opens-session
        """
        snap_edge_attr_dict = {
            et: snapshot[et].edge_attr
            for et in snapshot.edge_types
            if hasattr(snapshot[et], 'edge_attr') and snapshot[et].edge_attr is not None
        }
        node_emb = self.encoder(
            snapshot.x_dict,
            snapshot.edge_index_dict,
            edge_attr_dict=snap_edge_attr_dict,
        )
        sess_emb = node_emb['session'] # [N_sess, H]

        # Semantic attention chỉ cho SAGE
        if self.gnn_type == "sage":
            sess_emb = self.semantic_attn(self.rel_projector(sess_emb))
        
        # Pool session → user
        edge_index = snapshot['user', 'opens', 'session'].edge_index
        user_idx = edge_index[0]
        sess_idx = edge_index[1]

        N_user = snapshot['user'].x.size(0)
        H = sess_emb.size(1)

        user_emb = torch.zeros(N_user, H, device=sess_emb.device)
        count = torch.zeros(N_user, 1, device=sess_emb.device)

        if edge_index.numel() > 0:
            user_emb.scatter_add_(
                0,
                user_idx.unsqueeze(-1).expand(-1, H),
                sess_emb[sess_idx]
            )
            count.scatter_add_(
                0,
                user_idx.unsqueeze(-1),
                torch.ones(user_idx.size(0), 1, device=sess_emb.device)
            )

        user_emb = user_emb / count.clamp(min=1.0)

        # user mới ko có session thì embedding = zero, cần fallback bằng user.x
        # Sau khi pool session→user
        zero_mask = (count.squeeze(-1) == 0)
        if zero_mask.any():
            # Dùng static user feature làm fallback cho user chưa có session
            static = self.input_proj.projs['user'](snapshot['user'].x[zero_mask].float())
            user_emb[zero_mask] = static

        return user_emb
        

    def forward_temporal(self, snapshot_list: list) -> torch.Tensor:
        """
        Trả về RAW LOGITS [N_user] — trước sigmoid.

        SAGE:
        encode từng snapshot → user embedding [N_user, H]
        calibration
        padding user dimension nếu test có user mới
        TSA trên user sequence
        classifier → user anomaly logits

        GAT/RGCN:
        encode snapshot cuối → user embedding
        classifier → user anomaly logits
        """
        if self.gnn_type == "sage":
            # Mỗi phần tử: [N_user_t, H]
            per_snap = [self._encode_snapshot(s) for s in snapshot_list]

            if self.calibrator is not None:
                per_snap = self.calibrator(per_snap)

            # Padding theo số user, không phải session
            N_max = max(e.size(0) for e in per_snap)
            H = per_snap[0].size(1)

            padded, masks = [], []

            for emb in per_snap:
                n = emb.size(0)

                if n < N_max:
                    pad = torch.zeros(N_max - n, H, device=emb.device)
                    padded.append(torch.cat([emb, pad], dim=0))

                    m = torch.zeros(N_max, dtype=torch.bool, device=emb.device)
                    m[n:] = True
                else:
                    padded.append(emb)
                    m = torch.zeros(N_max, dtype=torch.bool, device=emb.device)

                masks.append(m)

            seq = torch.stack(padded, dim=1)      # [N_user_max, T, H]
            tmask = torch.stack(masks, dim=1)     # [N_user_max, T]

            repr_t = self.temporal(seq, mask=tmask)  # [N_user_max, H]

        else:
            # GAT/RGCN: chỉ dùng snapshot cuối, nhưng vẫn trả user embedding
            repr_t = self._encode_snapshot(snapshot_list[-1])

        return self.classifier(repr_t).squeeze(-1)
