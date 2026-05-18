"""
model_architecture.py — File 3: Định nghĩa kiến trúc mô hình
=============================================================
Cập nhật theo schema 02_graph_construction.py (DB-aware):

Node types:
    user, session, pc, file_object, web_resource,
    email, device, topic, role, department

Edge types (forward):
    (user, logs_on,       session)
    (user, uses_device,   device)
    (user, exports_file,  file_object)
    (user, visits,        web_resource)
    (user, sends_email,   email)
    (user, has_role,      role)
    (user, belongs_to,    department)
    (user, reports_to,    user)
    (session,      on_pc,       pc)
    (device,       attached_to, pc)
    (file_object,  has_topic,   topic)
    (web_resource, has_topic,   topic)
    (email,        has_topic,   topic)
    (email,        sent_to,     user)
    + reverse edges do ToUndirected() sinh ra

user.x chiều: 3 (role/dept/team) + 7 baseline + 5 psychometric = 15

Pipeline chính (PDF §3 — Attentive SAGE):
    HeteroData
        → HeteroGraphSAGEEncoder          (Attentive SAGE: SAGEConv + NeighborAttention)
        → emb_dict["session"]             (session embedding sau GNN, KHÔNG dùng raw feature)
        → _make_behavior_seq()            (xây chuỗi k session gần nhất per user)
        → IntentionExtractor (GRU)        → V_intent
        → emb_dict["user"] → hist_proj    → V_history
        → SimilarityScorer                → logits, anomaly_score

Thay đổi so với phiên bản cũ (PDF §3):
    - Vẫn dùng SAGEConv làm backbone (giữ inductive capability).
    - Bổ sung NeighborAttentionSAGEConv: trước khi SAGEConv aggregate,
      học attention score α(i,j) cho từng cặp (center_node, neighbor).
      α = softmax(LeakyReLU(Linear([h_i || h_j]))) per neighbor.
      Neighbor được scale bởi α trước khi đưa vào SAGEConv.
      → Hành vi DELETE/GRANT tự động nhận α cao hơn SELECT (PDF §3).
    - Mọi API, forward(), _make_behavior_seq() đều giữ nguyên.
"""

# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax as pyg_softmax

from constants import (GNN_HIDDEN, GNN_OUT, GRU_HIDDEN, WINDOW_SIZE, DROPOUT)
# ══════════════════════════════════════════════════════════════════════════════
# 0. NeighborAttentionSAGEConv — SAGEConv + Neighbor Attention (PDF §3)
# ══════════════════════════════════════════════════════════════════════════════

class NeighborAttentionSAGEConv(MessagePassing):
    """
    Attentive SAGE: giữ nguyên SAGEConv làm backbone, bổ sung cơ chế
    attention để học trọng số α(i,j) cho từng cặp (center_i, neighbor_j).

    Tại sao không đổi sang GATConv? (PDF §3)
    - SAGEConv inductive: sample neighbor → inference trên node mới không cần retrain.
    - GATConv không thay đổi điều này, nhưng ta muốn giữ nguyên toàn bộ
      kiến trúc SAGEConv và chỉ bổ sung attention như một "wrapper" bên trên.

    Cơ chế Attentive SAGE:
        1. Với mỗi edge (j → i), tính attention score:
               e(i,j) = LeakyReLU( W_a · [h_i || h_j] )
           Trong đó h_i là center node, h_j là neighbor node.
        2. Normalize bằng softmax trên tất cả neighbor của i:
               α(i,j) = softmax_j( e(i,j) )
        3. Scale neighbor feature trước khi aggregate:
               h_j_scaled = α(i,j) * h_j
        4. SAGEConv aggregate h_j_scaled theo mean, concat với h_i, project:
               h_i_new = W · [h_i || mean(h_j_scaled)]

    Điểm khác biệt so với GATConv:
    - Vẫn dùng SAGEConv (mean aggregation + concat + linear) như cũ.
    - Attention chỉ scale neighbor trước khi đưa vào SAGEConv,
      không thay thế toàn bộ aggregation như GATConv.
    - Lazy input dim: không hardcode in_channels, dùng Linear lazy.

    Args:
        out_channels : chiều đầu ra sau SAGEConv
        dropout      : dropout trên attention weights
    """

    def __init__(self, out_channels: int, dropout: float = 0.0):
        # aggr="mean": SAGEConv dùng mean aggregation trên neighbor đã scale
        super().__init__(aggr="mean")
        self.out_channels = out_channels
        self.dropout      = dropout

        # Lazy Linear: tự suy in_channels khi forward lần đầu
        # W_a: chiều vào = 2 * in_channels (concat h_i và h_j)
        # → dùng nn.LazyLinear(1) để tự suy
        self._attn_lazy: nn.LazyLinear | None = nn.LazyLinear(1)

        # SAGEConv backbone: (-1, -1) tự suy chiều đầu vào
        self.sage = SAGEConv((-1, -1), out_channels)

    def forward(
        self,
        x: torch.Tensor | tuple,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        x          : node features [N, in_channels] hoặc tuple (x_src, x_dst)
        edge_index : [2, E] — cạnh từ neighbor j sang center i
        """
        # Chuẩn hóa x thành (x_src, x_dst) cho bipartite graph
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        # ── Bước 1: Tính attention score e(i,j) cho mỗi edge ─────────────────
        # Thu thập (h_i, h_j) cho mỗi edge theo edge_index
        row, col = edge_index   # row = neighbor j (src), col = center i (dst)

        h_src = x_src[row]     # [E, in_channels] — neighbor features
        h_dst = x_dst[col]     # [E, in_channels] — center features

        # Concat [h_dst || h_src] để attention biết cả center lẫn neighbor
        pair  = torch.cat([h_dst, h_src], dim=-1)   # [E, 2*in_channels]

        # Tính score qua lazy linear (tự suy 2*in_channels khi lần đầu)
        e     = self._attn_lazy(pair).squeeze(-1)   # [E]
        e     = F.leaky_relu(e, negative_slope=0.2)

        # ── Bước 2: Softmax normalize per center node i ───────────────────────
        # pyg_softmax: softmax trên tất cả neighbor của cùng 1 center node
        alpha = pyg_softmax(e, col, num_nodes=x_dst.size(0))   # [E]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # ── Bước 3: Scale neighbor features bởi attention weight ─────────────
        # h_j_scaled = α(i,j) * h_j
        # x_src_scaled = x_src.clone()
        # Scatter attention weights lên từng neighbor
        # Cách đơn giản: tạo tensor scaled per edge rồi scatter-mean
        # → thực hiện bằng cách pass vào SAGEConv qua x_src đã scale
        # Trick: tạo x_src mới trong đó mỗi node j nhận trung bình alpha của nó
        # Cách chuẩn: dùng propagate() của MessagePassing trực tiếp
        scaled = alpha.unsqueeze(-1) * x_src[row]   # [E, in_channels]

        # Tạo x_src_weighted: với mỗi node j, cộng dồn scaled messages về
        # Nhưng SAGEConv cần x_src nguyên để project — ta truyền thẳng scaled
        # qua cách override: build weighted x bằng scatter mean
        x_src_weighted = torch.zeros_like(x_src)
        x_src_weighted.scatter_add_(0, row.unsqueeze(-1).expand_as(scaled), scaled)
        # Đếm số lần mỗi neighbor được referenced để chuẩn hoá
        count = torch.zeros(x_src.size(0), 1, device=x_src.device)
        count.scatter_add_(0, row.unsqueeze(-1),
                           torch.ones(row.size(0), 1, device=x_src.device))
        count = count.clamp(min=1.0)
        x_src_weighted = x_src_weighted / count   # mean-scale per neighbor node

        # ── Bước 4: SAGEConv aggregate trên x_src đã scale ───────────────────
        # SAGEConv nhận (x_src, x_dst): aggregate x_src, concat x_dst, project
        return self.sage((x_src_weighted, x_dst), edge_index)


# ══════════════════════════════════════════════════════════════════════════════
# 1. HeteroGraphSAGEEncoder — dùng NeighborAttentionSAGEConv (PDF §3)
# ══════════════════════════════════════════════════════════════════════════════

class HeteroGraphSAGEEncoder(nn.Module):
    """
    2 lớp HeteroConv(NeighborAttentionSAGEConv) trên Heterogeneous Graph.

    Backbone vẫn là SAGEConv (inductive, tự suy chiều đầu vào).
    Bổ sung Neighbor Attention: α(i,j) = softmax(LeakyReLU(W_a·[h_i||h_j]))
    → scale neighbor h_j trước khi aggregate → hành vi nguy hiểm được
    attention cao hơn (PDF §3: DELETE/GRANT > SELECT).

    Fix NoneType crash: sau mỗi HeteroConv, node type không có edge trong
    batch → None. _fill_none() thay bằng zero tensor đúng chiều.
    """

    FORWARD_EDGE_TYPES = [
        ("user",         "logs_on",       "session"),
        # ("user",         "uses_device",   "device"),
        ("user",         "exports_file",  "file_object"),
        ("user",         "visits",        "web_resource"),
        ("user",         "sends_email",   "email"),
        ("user",         "has_role",      "role"),
        ("user",         "belongs_to",    "department"),
        ("user",         "reports_to",    "user"),
        ("session",      "on_pc",         "pc"),
        # ("device",       "attached_to",   "pc"),
        ("file_object",  "has_topic",     "topic"),
        ("web_resource", "has_topic",     "topic"),
        ("email",        "has_topic",     "topic"),
        ("email",        "sent_to",       "user"),
        ("pc",     "has_usb_activity", "user"), # bổ sung edge mới để truyền signal USB activity từ PC sang user, giúp GNN học được mối liên hệ giữa hành vi copy file ra USB và session/email liên quan
    ]

    # Reverse edges sinh bởi ToUndirected()
    REVERSE_EDGE_TYPES = [
        ("session",      "rev_logs_on",        "user"),
        # ("device",       "rev_uses_device",    "user"),
        ("file_object",  "rev_exports_file",   "user"),
        ("web_resource", "rev_visits",         "user"),
        ("email",        "rev_sends_email",    "user"),
        ("role",         "rev_has_role",       "user"),
        ("department",   "rev_belongs_to",     "user"),
        ("pc",           "rev_on_pc",          "session"),
        # ("pc",           "rev_attached_to",    "device"),
        ("topic",        "rev_has_topic",      "file_object"),
        ("topic",        "rev_has_topic",      "web_resource"),
        ("topic",        "rev_has_topic",      "email"),
        ("user",         "rev_sent_to",        "email"),
        ("user",   "rev_has_usb_activity", "pc"),
    ]

    def __init__(self, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.dropout    = dropout

        all_et = self.FORWARD_EDGE_TYPES + self.REVERSE_EDGE_TYPES

        # Dùng NeighborAttentionSAGEConv thay SAGEConv thuần:
        # backbone vẫn là SAGE, bổ sung attention α(i,j) per neighbor (PDF §3).
        self.conv1 = HeteroConv(
            {et: NeighborAttentionSAGEConv(hidden_dim, dropout=dropout) for et in all_et},
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {et: NeighborAttentionSAGEConv(out_dim, dropout=dropout) for et in all_et},
            aggr="sum",
        )

    def _fill_none(self, x_out: dict, x_in: dict, out_dim: int) -> dict:
        """Thay None (node type không có edge trong batch) bằng zero tensor."""
        result = {}
        for ntype, feat in x_in.items():
            val = x_out.get(ntype)
            if val is None:
                n   = feat.shape[0] if feat is not None else 1
                dev = feat.device   if feat is not None else torch.device("cpu")
                val = torch.zeros(n, out_dim, dtype=torch.float, device=dev)
            result[ntype] = val
        return result

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        known  = set(self.FORWARD_EDGE_TYPES + self.REVERSE_EDGE_TYPES)
        active = {k: v for k, v in edge_index_dict.items() if k in known}

        out1 = self.conv1(x_dict, active)
        out1 = self._fill_none(out1, x_dict, self.hidden_dim)
        out1 = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                for k, v in out1.items()}

        out2 = self.conv2(out1, active)
        out2 = self._fill_none(out2, out1, self.out_dim)
        return out2


# ══════════════════════════════════════════════════════════════════════════════
# 2. IntentionExtractor — GRU xử lý chuỗi hành vi
# ══════════════════════════════════════════════════════════════════════════════

class IntentionExtractor(nn.Module):
    """
    GRU xử lý chuỗi k session/hành vi gần nhất.

    Input:  [batch, k, input_dim]
    Output: [batch, hidden_dim]  — vector ý định V_intent
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru  = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(seq)       # [num_layers, batch, hidden]
        return F.relu(self.proj(h_n[-1]))


# ══════════════════════════════════════════════════════════════════════════════
# 3. SimilarityScorer
# ══════════════════════════════════════════════════════════════════════════════

class SimilarityScorer(nn.Module):
    """
    Cosine similarity giữa V_intent và V_history + MLP phân loại.

    anomaly_score = 1 - cosine_sim:
        ~ 0  → hành vi ý định khớp lịch sử → bình thường
        ~ 2  → hoàn toàn ngược chiều → suspicious
    """

    def __init__(self, embed_dim: int, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, v_intent: torch.Tensor,
                v_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sim       = F.cosine_similarity(v_intent, v_history, dim=-1, eps=1e-8)
        combined      = torch.cat([v_intent, v_history, cos_sim.unsqueeze(-1)], dim=-1)
        logits        = self.classifier(combined)
        anomaly_score = 1.0 - cos_sim
        return logits, anomaly_score


# ══════════════════════════════════════════════════════════════════════════════
# 4. InsiderThreatModel — ghép GNN + GRU + Scorer
# ══════════════════════════════════════════════════════════════════════════════

class InsiderThreatModel(nn.Module):
    """
    Pipeline:
        HeteroData → GNN encoder (Attentive SAGE) → emb_dict
                   → emb_dict["session"] → _make_behavior_seq → GRU → V_intent
                   → emb_dict["user"]   → hist_proj           → V_history
                   → Scorer(V_intent, V_history) → logits, anomaly_score

    GRU ăn session embedding từ GNN output (không dùng raw session.x).
    GNN dùng NeighborAttentionSAGEConv: SAGEConv + α(i,j) attention per neighbor.
    Caller không cần truyền behavior_seq từ bên ngoài.
    """

    def __init__(
        self,
        gnn_hidden:  int   = 128,
        gnn_out:     int   = 64,
        gru_hidden:  int   = 64,
        window_size: int   = 10,
        num_classes: int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.window_size = window_size
        self.gnn_out     = gnn_out

        self.gnn = HeteroGraphSAGEEncoder(
            hidden_dim=gnn_hidden, out_dim=gnn_out, dropout=dropout,
        )
        self.intention_extractor = IntentionExtractor(
            input_dim=gnn_out, hidden_dim=gru_hidden, num_layers=2, dropout=dropout,
        )
        self.scorer    = SimilarityScorer(embed_dim=gru_hidden, num_classes=num_classes)
        self.hist_proj = nn.Linear(gnn_out, gru_hidden)

    # ------------------------------------------------------------------
    def _make_behavior_seq(
        self,
        session_emb:     torch.Tensor,   # [N_sess, gnn_out] — GNN output của session nodes
        edge_index_dict: dict,           # để tìm edge (user, logs_on, session)
        user_indices:    torch.Tensor,   # [batch] — seed user indices trong batch
    ) -> torch.Tensor:
        """
        Xây chuỗi [batch, window_size, gnn_out] từ session GNN embedding.

        Với mỗi user trong batch:
            - Tra edge (user, logs_on, session) → lấy session indices của user đó
            - Lấy tối đa window_size session gần nhất (cuối danh sách = gần nhất)
            - Pad bằng 0 nếu thiếu
        GRU ăn đúng GNN embedding — không dùng raw session.x.
        """
        device     = session_emb.device
        batch_size = len(user_indices)
        result     = torch.zeros(batch_size, self.window_size, self.gnn_out, device=device)

        edge_key = ("user", "logs_on", "session")
        if edge_key not in edge_index_dict:
            return result

        ei = edge_index_dict[edge_key]   # [2, E]: row0=user_idx, row1=sess_idx

        for i, uid in enumerate(user_indices.tolist()):
            mask     = ei[0] == uid
            sess_ids = ei[1][mask]
            if len(sess_ids) == 0:
                continue
            sess_ids = sess_ids[-self.window_size:]           # lấy gần nhất
            seq      = session_emb[sess_ids]                  # [k, gnn_out]
            result[i, :seq.shape[0]] = seq

        return result

    # ------------------------------------------------------------------
    def forward(
        self,
        x_dict:          dict,
        edge_index_dict: dict,
        user_indices:    torch.Tensor,   # [batch] — seed user indices trong batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. GNN encode toàn bộ node types trong batch
        emb_dict = self.gnn(x_dict, edge_index_dict)

        # 2. Lấy session embedding từ GNN output → GRU
        #    (KHÔNG dùng raw session.x — đây là thay đổi chính)
        session_emb = emb_dict["session"]                     # [N_sess, gnn_out]
        behavior    = self._make_behavior_seq(
            session_emb, edge_index_dict, user_indices
        )                                                     # [batch, window, gnn_out]

        # 3. V_intent từ GRU
        v_intent  = self.intention_extractor(behavior)        # [batch, gru_hidden]

        # 4. V_history từ user embedding
        v_history = self.hist_proj(emb_dict["user"][user_indices])  # [batch, gru_hidden]

        return self.scorer(v_intent, v_history)


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════
# Hàm khởi tạo model, giữ graph_data làm tham số để tương thích với caller cũ.
def build_hetero_model(
    graph_data:  HeteroData,
    gnn_hidden:  int   = GNN_HIDDEN,
    gnn_out:     int   = GNN_OUT,
    gru_hidden:  int   = GRU_HIDDEN,
    window_size: int   = WINDOW_SIZE,
    dropout:     float = DROPOUT,
) -> InsiderThreatModel:
    """
    Khởi tạo InsiderThreatModel với Attentive SAGE (SAGEConv + NeighborAttention).
    graph_data được giữ làm tham số để tương thích với caller cũ.
    """
    return InsiderThreatModel(
        gnn_hidden=gnn_hidden,
        gnn_out=gnn_out,
        gru_hidden=gru_hidden,
        window_size=window_size,
        dropout=dropout,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Smoke test InsiderThreatModel (Attentive SAGE — NeighborAttention) ===")

    N = dict(user=50, session=120, pc=20, email=80, device=30,
             file_object=40, web_resource=60, topic=25, role=5, department=8)
    GNN_OUT, GRU_HID, WINDOW, BATCH = 64, 64, 10, 8

    x_dict = {
        "user":         torch.randn(N["user"],         15),  # 3+7+5
        "session":      torch.randn(N["session"],      3),
        "pc":           torch.randn(N["pc"],           2),
        "email":        torch.randn(N["email"],        3),
        # "device":       torch.randn(N["device"],       3),
        "file_object":  torch.randn(N["file_object"],  4),
        "web_resource": torch.randn(N["web_resource"], 4),
        "topic":        torch.eye(N["topic"]),
        "role":         torch.eye(N["role"]),
        "department":   torch.eye(N["department"]),
    }

    def re(a, b, e):
        return torch.stack([torch.randint(0, a, (e,)), torch.randint(0, b, (e,))])

    edge_index_dict = {
        ("user",         "logs_on",        "session"):      re(N["user"],        N["session"],      200),
        # ("user",         "uses_device",    "device"):       re(N["user"],        N["device"],       100),
        ("user",         "exports_file",   "file_object"):  re(N["user"],        N["file_object"],   80),
        ("user",         "visits",         "web_resource"): re(N["user"],        N["web_resource"], 150),
        ("user",         "sends_email",    "email"):        re(N["user"],        N["email"],        120),
        ("user",         "has_role",       "role"):         re(N["user"],        N["role"],         N["user"]),
        ("user",         "belongs_to",     "department"):   re(N["user"],        N["department"],   N["user"]),
        ("session",      "on_pc",          "pc"):           re(N["session"],     N["pc"],           200),
        # ("device",       "attached_to",    "pc"):           re(N["device"],      N["pc"],           100),
        ("file_object",  "has_topic",      "topic"):        re(N["file_object"], N["topic"],        120),
        ("web_resource", "has_topic",      "topic"):        re(N["web_resource"],N["topic"],        180),
        ("email",        "has_topic",      "topic"):        re(N["email"],       N["topic"],        160),
        ("email",        "sent_to",        "user"):         re(N["email"],       N["user"],         100),
        ("session",      "rev_logs_on",        "user"):     re(N["session"],     N["user"],         200),
        ("device",       "rev_uses_device",    "user"):     re(N["device"],      N["user"],         100),
        ("file_object",  "rev_exports_file",   "user"):     re(N["file_object"], N["user"],          80),
        ("web_resource", "rev_visits",         "user"):     re(N["web_resource"],N["user"],         150),
        ("email",        "rev_sends_email",    "user"):     re(N["email"],       N["user"],         120),
        ("role",         "rev_has_role",       "user"):     re(N["role"],        N["user"],         N["user"]),
        ("department",   "rev_belongs_to",     "user"):     re(N["department"],  N["user"],         N["user"]),
        ("pc",           "rev_on_pc",          "session"):  re(N["pc"],          N["session"],      200),
        ("pc",           "rev_attached_to",    "device"):   re(N["pc"],          N["device"],       100),
        ("topic",        "rev_has_topic",      "file_object"):re(N["topic"],     N["file_object"],  120),
        ("topic",        "rev_has_topic",      "web_resource"):re(N["topic"],    N["web_resource"], 180),
        ("topic",        "rev_has_topic",      "email"):    re(N["topic"],       N["email"],        160),
        ("user",         "rev_sent_to",        "email"):    re(N["user"],        N["email"],        100),
        ("pc", "has_usb_activity",     "user"): re(N["pc"], N["user"], 60),
        ("user", "rev_has_usb_activity", "pc"): re(N["user"], N["pc"], 60),
    }

    model = InsiderThreatModel(gnn_out=GNN_OUT, gru_hidden=GRU_HID, window_size=WINDOW)
    model.eval()
    with torch.no_grad():
        # forward không cần behavior_seq — model tự lấy session emb từ GNN
        logits, score = model(
            x_dict, edge_index_dict,
            torch.randint(0, N["user"], (BATCH,)),
        )
    print(f"  logits : {logits.shape}")
    print(f"  score  : {score.shape}")
    print(f"  params : {sum(p.numel() for p in model.parameters()):,}")
    print("✓ OK")
