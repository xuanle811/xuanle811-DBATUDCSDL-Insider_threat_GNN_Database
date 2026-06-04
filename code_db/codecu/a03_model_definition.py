"""
File: a03_model_definition.py
Định nghĩa kiến trúc mô hình 3 thành phần theo đặc tả PDF "12_mô_hình":

Thành phần 1: THGC - Temporal Heterogeneous Graph Constructor
             (Thực hiện ở a02 - xây dựng đồ thị con theo cửa sổ thời gian)

Thành phần 2: ASTHE Encoder (Asymmetric Spatial-semantic Heterogeneous Encoder)
   - Lớp A: Structural Message Passing (GraphSAGE quy nạp via to_hetero)
   - Lớp B: Semantic Attention Layer — học trọng số α cho từng loại cạnh

Thành phần 3: Temporal Self-Attention Module
   - Multi-Head Self-Attention trên chuỗi graph embeddings
   - Thêm Positional Encoding
   - Tầng Linear + Sigmoid để tính Anomaly Score
==========================================================
- So sánh thêm với các model GCN, GAT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero, GCNConv, GATConv, RGCNConv


# ══════════════════════════════════════════════════════════════════════════════
# THÀNH PHẦN 2 — LỚP A: InductiveGNNKernel (GraphSAGE)
# ══════════════════════════════════════════════════════════════════════════════

class InductiveGNNKernel(torch.nn.Module):
    def __init__(self, metadata, in_channels_dict: dict, hidden_channels: int, gnn_type="sage"):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.gnn_type = gnn_type

        # 1. Tầng chiếu đầu vào
        self.input_projs = torch.nn.ModuleDict({
            node_type: nn.Linear(in_channels_dict[node_type], hidden_channels)
            for node_type in self.node_types
        })

        # 2. Hàm tạo Layer linh hoạt
        def get_conv(in_dim, out_dim):
            if self.gnn_type == "sage":
                return SAGEConv((in_dim, in_dim), out_dim, normalize=True)
            elif self.gnn_type == "rgcn":
                return RGCNConv(in_dim, out_dim, num_relations=len(self.edge_types))
            elif self.gnn_type == "gat":
                return GATConv(in_dim, out_dim, heads=1, add_self_loops=True)
            return None
        
        # 3. Khởi tạo các tầng Conv
        self.conv1 = torch.nn.ModuleDict({
            et: get_conv(hidden_channels, hidden_channels) 
            for et in self.edge_types})
        self.conv2 = torch.nn.ModuleDict({
            et: get_conv(hidden_channels, hidden_channels) 
            for et in self.edge_types})

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        # Bước 0: Chiếu đặc trưng nút
        h_dict = {nt: F.relu(self.input_projs[nt](x.float())) for nt, x in x_dict.items()}
        #=============Debug=========================
        print(edge_index_dict.keys())
        # Tạo mapping edge_type -> relation_id
        self.edge_type2rel = {et: i for i, et in enumerate(self.edge_types)}

        # Hàm truyền tin dùng chung (Reusable Message Passing)
        def pass_layer(conv_layer, input_dict):
            out_dict = {nt: [] for nt in self.node_types}
            for edge_type, conv in conv_layer.items():
                src, _, dst = edge_type
                if edge_type in edge_index_dict and edge_index_dict[edge_type].numel() > 0:
                    if self.gnn_type == "rgcn":
                        rel_id = self.edge_type2rel[edge_type]
                        h_out = conv(h_dict[src], edge_index_dict[edge_type], edge_type=torch.full(
                            (edge_index_dict[edge_type].size(1),), rel_id, dtype=torch.long, device=h_dict[src].device))
                    else:
                        h_out = conv((input_dict[src], input_dict[dst]), edge_index_dict[edge_type])
                    out_dict[dst].append(h_out)

                
            
            # Gom lại
            new_h = {}
            for nt in self.node_types:
                if out_dict[nt]:
                    new_h[nt] = F.relu(torch.stack(out_dict[nt], dim=0).sum(dim=0) + input_dict[nt])
                else:
                    new_h[nt] = input_dict[nt]
            return new_h

        # Chạy 2 tầng truyền tin
        h1 = pass_layer(self.conv1, h_dict)
        h2 = pass_layer(self.conv2, h1)
        
        return h2
# ══════════════════════════════════════════════════════════════════════════════
# THÀNH PHẦN 2 — RELATION PROJECTOR
# ══════════════════════════════════════════════════════════════════════════════

class RelationProjector(nn.Module):
    """
    Chiếu session_emb [N, H] thành n_relations embeddings khác nhau.
    Mỗi relation type có W_r riêng → embeddings_per_relation[r] ≠ embeddings_per_relation[s].
    Đây là điều kiện cần để SemanticAttentionLayer học α có ý nghĩa.

    Công thức:
        h_r = ReLU(W_r · h_session + b_r)    r = 0..n_relations-1
    """
    def __init__(self, hidden_channels: int, n_relations: int):
        super().__init__()
        # ModuleList: n_relations linear layers độc lập
        self.projectors = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(n_relations)
        ])

    def forward(self, session_emb: torch.Tensor) -> list:
        """
        Args:
            session_emb: [N, H]
        Returns:
            List[Tensor[N, H]] — n_relations embeddings khác nhau
        """
        return [F.relu(proj(session_emb)) for proj in self.projectors]


# ══════════════════════════════════════════════════════════════════════════════
# THÀNH PHẦN 2 — LỚP B: SemanticAttentionLayer
# ══════════════════════════════════════════════════════════════════════════════

class SemanticAttentionLayer(nn.Module):
    """
    Tính trọng số α cho từng loại quan hệ.
    Nhận đầu vào là List[Tensor[N, H]] với mỗi tensor khác nhau
    (được tạo bởi RelationProjector) → α có ý nghĩa thực sự.

    Công thức (paper §3.2.B):
        e_r = mean_n( tanh(W_s · h_n^(r)) ) · w_r
        α_r = softmax_r( e_r )
        H   = Σ_r α_r · H^(r)
    """
    def __init__(self, hidden_channels: int, num_relation_types: int):
        super().__init__()
        self.semantic_vectors = nn.Parameter(
            torch.empty(num_relation_types, hidden_channels)
        )
        nn.init.xavier_uniform_(self.semantic_vectors.unsqueeze(0))
        self.proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.act  = nn.Tanh()

    def forward(self, embeddings_per_relation: list) -> torch.Tensor:
        n_r     = len(embeddings_per_relation)
        stacked = torch.stack(embeddings_per_relation, dim=0)          # [R, N, H]
        proj    = self.act(self.proj(stacked))                         # [R, N, H]
        scores  = torch.einsum('rnh,rh->rn', proj,
                               self.semantic_vectors[:n_r])            # [R, N]
        alpha   = F.softmax(scores, dim=0).unsqueeze(-1)               # [R, N, 1]
        return (alpha * stacked).sum(dim=0)                            # [N, H]


# ══════════════════════════════════════════════════════════════════════════════
# THÀNH PHẦN 3: PositionalEncoding + TemporalSelfAttentionModule
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_channels: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, hidden_channels)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_channels, 2, dtype=torch.float)
                        * (-math.log(10000.0) / hidden_channels))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = (torch.cos(pos * div) if hidden_channels % 2 == 0
                       else torch.cos(pos * div[:-1]))
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, H]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalSelfAttentionModule(nn.Module):
    """
    Multi-Head Self-Attention trên chuỗi graph embeddings.
    Input : [N_sess, T, H]
    Output: [N_sess, H]
    """
    def __init__(self, hidden_channels: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        # Tự điều chỉnh num_heads nếu hidden_channels không chia hết
        while hidden_channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.pos_enc = PositionalEncoding(hidden_channels, max_len=max_len,
                                          dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels, nhead=num_heads,
            dim_feedforward=hidden_channels * 4, dropout=dropout,
            batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, sequence: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """sequence: [B, T, H]  →  [B, H]"""
        x = self.pos_enc(sequence)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).float().unsqueeze(-1)
            x     = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# MÔ HÌNH TỔNG THỂ: InsiderThreatDetector
# ══════════════════════════════════════════════════════════════════════════════

class InsiderThreatDetector(torch.nn.Module):
    """
    Kiến trúc 3 thành phần:
        1. ASTHE Encoder: GNNKernel (Lớp A) + RelationProjector + SemanticAttention (Lớp B)
        2. Temporal Self-Attention Module (TSA)
        3. Linear Classifier → RAW LOGITS (BUG-10 FIX: không sigmoid ở đây)

    Dùng forward_temporal(snapshot_list) cho Temporal mode đầy đủ.
    snapshot_list: List[HeteroData]  (BUG-11 FIX)
    """

        
    def __init__(self, metadata, in_channels_dict: dict, hidden_channels: int = 64,
                 num_heads: int = 4, temporal_layers: int = 2, dropout: float = 0.1, gnn_type="sage"):
        super().__init__()
        self.metadata    = metadata
        self.n_relations = len(metadata[1])   # tổng số edge types
        self.gnn_type = gnn_type

        # THÀNH PHẦN 1: ASTHE Encoder (Lớp A)
        # Truyền trực tiếp gnn_type vào InductiveGNNKernel để nó tự tạo các tầng conv tương ứng
        self.encoder = InductiveGNNKernel(
            metadata=metadata, 
            in_channels_dict=in_channels_dict, 
            hidden_channels=hidden_channels, 
            gnn_type=gnn_type
        )

        # RelationProjector tạo embeddings khác nhau per relation
        self.rel_projector = RelationProjector(hidden_channels, self.n_relations)

        # Lớp B: Semantic Attention
        self.semantic_attn = SemanticAttentionLayer(
            hidden_channels=hidden_channels,
            num_relation_types=self.n_relations,
        )

        # Thành phần 3: Temporal Self-Attention
        self.temporal = TemporalSelfAttentionModule(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        # Output: trả về RAW LOGITS
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def _encode_one_snapshot(self, snapshot) -> torch.Tensor:
        """
        Encode 1 HeteroData snapshot → session embedding [N_sess, H].
        Áp dụng đủ Lớp A + Lớp B.
        """
        # Lớp A: message passing
        node_emb = self.encoder(snapshot.x_dict, snapshot.edge_index_dict)
        sess_emb = node_emb['session']                      # [N_sess, H]

        # Lớp B: Semantic Attention → [N_sess, H]
        # hiếu sess_emb thành n_relations embeddings khác nhau
        emb_per_rel = self.rel_projector(sess_emb)          # List[Tensor[N,H]]
        return self.semantic_attn(emb_per_rel)

    def forward_temporal(self, snapshot_list: list) -> torch.Tensor:
        """
        Temporal mode đầy đủ: chuỗi T snapshots → logits [N_sess].

        Args:
            snapshot_list: List[HeteroData]  — theo thứ tự thời gian
                           BUG-11 FIX: truy cập .x_dict qua snapshot trực tiếp

        Returns:
            Tensor [N_sess] — RAW LOGITS (BUG-10 FIX: chưa qua sigmoid)
            → dùng với BCEWithLogitsLoss trong a04
        """
        per_snapshot_emb = []
        for snapshot in snapshot_list:         # BUG-11 FIX: không unpack tuple
            sess_attended = self._encode_one_snapshot(snapshot)   # [N_sess, H]
            per_snapshot_emb.append(sess_attended)

        # Stack: [N_sess, T, H]
        sequence = torch.stack(per_snapshot_emb, dim=1)

        # TSA: [N_sess, H]
        temporal_repr = self.temporal(sequence)

        # Classifier: [N_sess, 1] → squeeze → [N_sess]  (RAW LOGITS)
        return self.classifier(temporal_repr).squeeze(-1)
