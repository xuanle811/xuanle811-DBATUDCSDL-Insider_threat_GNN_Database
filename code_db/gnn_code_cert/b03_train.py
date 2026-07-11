"""
03_train.py — Giai đoạn huấn luyện
=====================================
Đầu vào  : output/graph_data.pt  (từ 02_graph_construction.py DB-aware)
Đầu ra   : output/best_model.pth
            output/training_log.csv
            output/split_masks.pt

Thay đổi so với phiên bản cũ:
    - Node types mới: session, pc, file_object, web_resource,
                      email, device, topic, role, department, user
    - Edge types mới khớp với 02_graph_construction.py
        Node types: ['role', 'department', 'user', 'session', 'device', 'file_object', 'web_resource', 'email', 'pc', 'topic']
        Edge types: [('user', 'has_role', 'role'), ('user', 'belongs_to', 'department'), ('user', 'reports_to', 'user'), 
            ('user', 'logs_on', 'session'), ('session', 'on_pc', 'pc'), ('user', 'uses_device', 'device'), 
            ('device', 'attached_to', 'pc'), ('user', 'exports_file', 'file_object'), ('file_object', 'has_topic', 'topic'), 
            ('user', 'visits', 'web_resource'), ('web_resource', 'has_topic', 'topic'), ('user', 'sends_email', 'email'), 
            ('email', 'sent_to', 'user'), ('email', 'has_topic', 'topic')]
    - behavior_seq được build bên trong InsiderThreatModel.forward từ GNN session embedding
    - train_epoch và evaluate không còn gọi make_behavior_seq từ bên ngoài
    - Caller chỉ truyền x_dict, edge_index_dict, user_indices — model tự xử lý phần còn lại
    - Main Step:
            session.x (raw)
                ↓
            GraphSAGE / HeteroGNN
                ↓
            emb_dict["session"]  ← session embedding sau GNN (không dùng raw)
                ↓
            _make_behavior_seq() bên trong model
                ↓
            GRU → V_intent


    - Sửa khi máy khoẻ: BATCH_SIZE = 32, EPOCHS =4, NUM_NEIGHBORS = [10,5 hoặc 5,3], WINDOW_SIZE = 10, PATIENCE = 10, 
    GNN_HIDDEN, GNN_OUT, GRU_HIDDEN = 128, 64, 64 (tăng kích thước model để học biểu diễn tốt hơn, nhưng cần máy khoẻ để train ổn định)
    - cài thêm thư viện: pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
"""

import gc
import csv
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from model_architecture import build_hetero_model
from constants import * # Hoặc liệt kê cụ thể các hằng số

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_DIR    = Path("output")
GRAPH_PT   = OUT_DIR / "model"/"graph_data.pt"
BEST_MODEL = OUT_DIR / "model"/"best_model.pth"
TRAIN_LOG  = OUT_DIR / "model"/"training_log.csv"

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS        = 2
LR            = 1e-3
WINDOW_SIZE   = 2     # 10   # k session gần nhất cho GRU
BATCH_SIZE    = 8         # Sửa thành 32 nếu máy khoe
# Số neighbor sample mỗi hop, theo từng edge type.
# Dùng dict để NeighborLoader biết sample bao nhiêu per relation.
# Giá trị -1 = lấy hết (dùng cho edge thưa như has_role, belongs_to).
# NUM_NEIGHBORS = {
#     ("user",         "logs_on",       "session"):      [3,2], #[10, 5], # lấy tối đa 10 session và 5 neighbor cho session đó (Example: pc, device)
#     ("user",         "uses_device",   "device"):       [2,1], #[5,  3],
#     ("user",         "exports_file",  "file_object"):  [2,1], #[5,  3],
#     ("user",         "visits",        "web_resource"): [2,1], #[5,  3],
#     ("user",         "sends_email",   "email"):        [2,1], #[5,  3],
#     ("user",         "has_role",      "role"):         [2,1], #[-1, -1],
#     ("user",         "belongs_to",    "department"):   [2,1], #[-1, -1],
#     ("user",         "reports_to",    "user"):         [2,1], #[5,  3],
#     ("session",      "on_pc",         "pc"):           [-1, -1],
#     ("device",       "attached_to",   "pc"):           [-1, -1],
#     ("file_object",  "has_topic",     "topic"):        [2,1], #[5,  3],
#     ("web_resource", "has_topic",     "topic"):        [2,1], #[5,  3],
#     ("email",        "has_topic",     "topic"):        [2,1], #[5,  3],
#     ("email",        "sent_to",       "user"):         [2,1], #[5,  3],
#     # reverse edges (ToUndirected)
#     ("session",      "rev_logs_on",       "user"):     [3,2], #[10, 5],
#     ("device",       "rev_uses_device",   "user"):     [2,1], #[5,  3],
#     ("file_object",  "rev_exports_file",  "user"):     [2,1], #[5,  3],
#     ("web_resource", "rev_visits",        "user"):     [2,1], #[5,  3],
#     ("email",        "rev_sends_email",   "user"):     [2,1], #[5,  3],
#     ("role",         "rev_has_role",      "user"):     [2,1], #[-1, -1],
#     ("department",   "rev_belongs_to",    "user"):     [2,1], #[-1, -1],
#     ("pc",           "rev_on_pc",         "session"):  [-1, -1],
#     ("pc",           "rev_attached_to",   "device"):   [-1, -1],
#     ("topic",        "rev_has_topic",     "file_object"): [2,1], #[5,  3],
#     ("topic",        "rev_has_topic",     "web_resource"):[2,1], #[5,  3],
#     ("topic",        "rev_has_topic",     "email"):    [2,1], #[5,  3],
#     ("user",         "rev_sent_to",       "email"):    [2,1], #[5,  3],
# }

PATIENCE    = 2 # sửa thành 10 nếu máy khoẻ
FOCAL_GAMMA = 2.0
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

# DEVICE = torch.device("cpu")
# Sửa cho chạy GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"CUDA version: {torch.version.cuda}")
else:
    DEVICE = torch.device("cpu")
    log.warning("CUDA không khả dụng -> dùng CPU")



# ══════════════════════════════════════════════════════════════════════════════
# Focal Loss
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.weight,
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Bước 3.1: không Chia theo thời gian, mà chia stratified theo nhãn insider/non-insider
# ══════════════════════════════════════════════════════════════════════════════

# def time_based_split(data: HeteroData) -> tuple[list, list, list]:
#     """
#     Chia user nodes theo thứ tự index (index nhỏ = gia nhập sớm hơn).
#     Không random để tránh data leakage: train quá khứ → test tương lai.
#     """
#     n      = data["user"].x.shape[0]
#     n_tr   = int(n * TRAIN_RATIO)
#     n_val  = int(n * VAL_RATIO)
#     tr     = list(range(0, n_tr))
#     val    = list(range(n_tr, n_tr + n_val))
#     te     = list(range(n_tr + n_val, n))
#     log.info(f"Split: train={len(tr)} | val={len(val)} | test={len(te)}")
#     return tr, val, te


def stratified_user_split(data: HeteroData) -> tuple[list, list, list]:
    """
    Chia user nodes theo nhãn insider/non-insider.
    Đảm bảo train/val/test đều có cả class 0 và class 1.
    """
    y = data["user"].y.cpu().numpy()
    idx = list(range(len(y)))

    train_idx, temp_idx = train_test_split(
        idx,
        train_size=TRAIN_RATIO,
        stratify=y,
        random_state=42,
    )

    temp_y = y[temp_idx]
    val_size = VAL_RATIO / (1.0 - TRAIN_RATIO)

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        stratify=temp_y,
        random_state=42,
    )

    log.info(
        f"Split: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}"
    )

    return train_idx, val_idx, test_idx
# ══════════════════════════════════════════════════════════════════════════════
# Bước 3.2: Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model:     nn.Module,
    loader:    NeighborLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    gnn_out:   int,
    scaler:    GradScaler,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        batch = batch.to(DEVICE)

        n_seed = batch["user"].batch_size
        user_idx_in_batch = torch.arange(n_seed, device=DEVICE)
        labels = batch["user"].y[:n_seed].long()

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(DEVICE.type == "cuda")):
            # Model tự lấy session GNN embedding bên trong forward
            # — không cần truyền behavior_seq từ ngoài
            logits, _ = model(
                x_dict=batch.x_dict,
                edge_index_dict=batch.edge_index_dict,
                user_indices=user_idx_in_batch,
            )

            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: NeighborLoader, gnn_out: int) -> dict:
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        n_seed            = batch["user"].batch_size
        user_idx_in_batch = torch.arange(n_seed, device=DEVICE)
        labels            = batch["user"].y[:n_seed]

        # Model tự build behavior_seq từ GNN session embedding bên trong forward
        logits, anomaly_score = model(
            x_dict=batch.x_dict,
            edge_index_dict=batch.edge_index_dict,
            user_indices=user_idx_in_batch,
        )
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_scores.extend(anomaly_score.cpu().tolist())

    f1   = f1_score(all_labels,  all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels,  all_preds, zero_division=0)
    acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)

    return {
        "f1": f1, "precision": prec, "recall": rec, "accuracy": acc,
        "all_preds": all_preds, "all_labels": all_labels,
        "anomaly_scores": all_scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline chính
# ══════════════════════════════════════════════════════════════════════════════

def train(data: HeteroData) -> nn.Module:
    # ── Split ─────────────────────────────────────────────────────────────────
    # chia node user theo tỉ lệ train:val:test = 70:10:20 đảm bảo data có cả 2 nhãn
    tr_idx, val_idx, te_idx = stratified_user_split(data)
    n_users = data["user"].x.shape[0]



    # # Phân chia data dựa trên MASk, đánh dấu vai trò của từng node mà ko cần cắt nhỏ graphtrong quá trình train/val/test. NeighborLoader sẽ dựa vào mask này để biết node nào thuộc tập nào.
    # 1. KHỞI TẠO (Bắt buộc phải có để không bị lỗi UnboundLocalError)
    train_mask = torch.zeros(n_users, dtype=torch.bool)
    val_mask   = torch.zeros(n_users, dtype=torch.bool)  # Mở comment dòng này
    test_mask  = torch.zeros(n_users, dtype=torch.bool) # Mở comment dòng này

    # 2. GÁN GIÁ TRỊ GỐC
    train_mask[tr_idx] = True
    val_mask[val_idx]   = True
    test_mask[te_idx]   = True

# ========================================================
# COMMENT LẠI KHI MÁY KHOẺ
    # 3. THU NHỎ (Chỉ giữ lại 20 node để test nhanh)
    val_indices = torch.where(val_mask)[0][:20]
    test_indices = torch.where(test_mask)[0][:20]

    new_val_mask = torch.zeros_like(val_mask)
    new_val_mask[val_indices] = True
    val_mask = new_val_mask

    new_test_mask = torch.zeros_like(test_mask)
    new_test_mask[test_indices] = True
    test_mask = new_test_mask

# hết phần cần comment khi máy khoẻ
#==================================================

    data["user"].train_mask = train_mask
    data["user"].val_mask   = val_mask
    data["user"].test_mask  = test_mask

    torch.save(
        {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask},
        OUT_DIR / "model"/"split_masks.pt",
    )
    

    # ── NeighborLoader ────────────────────────────────────────────────────────
    # Lọc NUM_NEIGHBORS chỉ lấy edge type thực sự có trong graph
    # Lấy danh sách các loại quan hệ thực tế có trong đồ thị
    existing_et = set(data.edge_types)

    # Chỉ giữ lại các cấu hình lấy mẫu cho những quan hệ thực sự tồn tại
    # Lưu ý: Key phải là Tuple (src, edge, dst), Value phải là List [int]
    active_neighbors = {
        k: list(v) for k, v in NUM_NEIGHBORS.items() if k in existing_et
    }
    # FIX QUAN TRỌNG: Xóa dấu phẩy ở cuối dòng này để tránh tạo thành Tuple
    neighbor_arg = active_neighbors if active_neighbors else [2, 1]

    common_kw = dict(
        num_neighbors=neighbor_arg,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    # NeighborLoader sẽ tự động lấy subgraph cho mỗi batch dựa trên user nodes có mask tương ứng (train/val/test).
    train_loader = NeighborLoader(data, input_nodes=("user", train_mask), shuffle=True,  **common_kw)
    val_loader   = NeighborLoader(data, input_nodes=("user", val_mask),   shuffle=False, **common_kw)
    test_loader  = NeighborLoader(data, input_nodes=("user", test_mask),  shuffle=False, **common_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    # Model sẽ tự xử lý phần tạo behavior_seq từ session GNN embedding bên trong forward, nên caller chỉ cần truyền x_dict, edge_index_dict, user_indices.
    model = build_hetero_model(
        graph_data=data, 
        gnn_hidden=GNN_HIDDEN, # GNN_HIDDEN=128: số chiều ẩn của GNN layers, ảnh hưởng đến khả năng học biểu diễn phức tạp từ graph. Giá trị này thường được chọn dựa trên độ lớn của graph và tài nguyên tính toán.
        gnn_out=GNN_OUT, # GNN_OUT=64: số chiều của embedding đầu ra từ GNN, sẽ được dùng làm input cho GRU. Giá trị này có thể nhỏ hơn GNN_HIDDEN để giảm độ phức tạp trước khi vào GRU.
        gru_hidden=GRU_HIDDEN, # GRU_HIDDEN=64: số chiều ẩn của GRU layer, ảnh hưởng đến khả năng học các mẫu tuần tự trong behavior sequence. Giá trị này thường được chọn dựa trên độ dài và tính phức tạp của chuỗi hành vi.
        window_size=WINDOW_SIZE, # WINDOW_SIZE=10: số session gần nhất được dùng để tạo behavior sequence cho mỗi user. Giá trị này nên đủ lớn để capture các mẫu hành vi dài hạn, nhưng không quá lớn để tránh quá tải thông tin và tính toán.
        dropout=DROPOUT, # DROPOUT=0.3: tỷ lệ dropout được áp dụng trong GNN và GRU để giảm overfitting. Giá trị này thường được chọn dựa trên độ lớn của dataset và độ phức tạp của model, có thể thử nghiệm với các giá trị như 0.1, 0.3, 0.5
    ).to(DEVICE)

    # +++ FIX: Khởi tạo tham số cho LazyModules trước khi đếm hoặc train +++
    # Lấy batch đầu tiên từ train_loader để "mồi" cho model
    for dummy_batch in train_loader:
        dummy_batch = dummy_batch.to(DEVICE)
        # Chạy forward pass giả để khởi tạo các chiều (dimension) cho Lazy layers
        model(
            dummy_batch.x_dict, 
            dummy_batch.edge_index_dict, 
            torch.tensor([0], device=DEVICE) # dummy user index
        )
        break # Chỉ cần 1 batch là đủ khởi tạo

    # Bây giờ bạn có thể đếm tham số mà không bị lỗi
    log.info(f"Model: {sum(p.numel() for p in model.parameters()):,} tham số")
    log.info(f"user.x chiều: {data['user'].x.shape[1]}")


    # ── Optimizer / Scheduler / Loss ──────────────────────────────────────────
    # LR=1e-3: learning rate ban đầu cho optimizer, ảnh hưởng đến tốc độ hội tụ và chất lượng model, có thể thử nghiệm với các giá trị như 1e-4, 1e-3, 1e-2.
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4) 
    # ReduceLROnPlateau sẽ giảm learning rate khi metric (ở đây là val_f1) không cải thiện sau một số epoch nhất định (patience=5), giúp model thoát khỏi local minima và cải thiện hiệu suất.
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5) 
    labels_all = data["user"].y
    num_normal = int((labels_all == 0).sum())
    num_insider = int((labels_all == 1).sum())

    # Tính class weight để xử lý imbalance: insider ít hơn normal → tăng trọng số cho insider trong loss function.
    class_weight = torch.tensor(
        [1.0, num_normal / max(num_insider, 1)],
        dtype=torch.float,
        device=DEVICE,
    )

    log.info(f"Class weight: normal=1.0 | insider={class_weight[1].item():.4f}")

    # Focal Loss với gamma=2.0 để tập trung vào các mẫu khó phân loại, kết hợp với class weight để xử lý imbalance.
    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
         # FOCAL_GAMMA=2.0: hệ số gamma trong Focal Loss, điều chỉnh mức độ tập trung vào các mẫu khó phân loại. Giá trị này thường được chọn dựa trên mức độ imbalance của dataset, có thể thử nghiệm với các giá trị như 1.0, 2.0, 5.0.
        weight=class_weight,
    ).to(DEVICE)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    # ── Loop ──────────────────────────────────────────────────────────────────
    # theo dõi best val_f1 để lưu model tốt nhất và đếm số epoch không cải thiện để early stopping
    best_val_f1, epochs_no_imp = -1.0, 0 
    log_rows = [] # lưu log mỗi epoch để ghi ra CSV sau khi training xong

    for epoch in range(1, EPOCHS + 1):
        loss   = train_epoch(model, train_loader, optimizer, criterion, GNN_OUT, scaler)
        val_m  = evaluate(model, val_loader, GNN_OUT)
        val_f1 = val_m["f1"]
        scheduler.step(val_f1)

        log.info(
            f"Epoch {epoch:3d}/{EPOCHS} | loss={loss:.4f} | "
            f"val_f1={val_f1:.4f} | prec={val_m['precision']:.4f} | rec={val_m['recall']:.4f}"
        )
        log_rows.append({
            "epoch": epoch, "loss": loss,
            "val_f1": val_f1, "val_prec": val_m["precision"], "val_rec": val_m["recall"],
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "hyperparams": {
                    "gnn_hidden":    GNN_HIDDEN,
                    "gnn_out":       GNN_OUT,
                    "gru_hidden":    GRU_HIDDEN,
                    "window_size":   WINDOW_SIZE,
                    "dropout":       DROPOUT,
                    "lr":            LR,
                    "user_feat_dim": int(data["user"].x.shape[1]),  # lưu để eval load đúng
                },
            }, BEST_MODEL)
            log.info(f"  ✓ Lưu best model (val_f1={val_f1:.4f})")
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= PATIENCE:
                log.info(f"Early stopping tại epoch {epoch}")
                break

        gc.collect()

    # ── Log CSV ───────────────────────────────────────────────────────────────
    with open(TRAIN_LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_f1", "val_prec", "val_rec"])
        w.writeheader()
        w.writerows(log_rows)
    log.info(f"Training log → {TRAIN_LOG}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    if BEST_MODEL.exists():
        ckpt = torch.load(BEST_MODEL, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Loaded best checkpoint from epoch {ckpt.get('epoch')} for final test.")

    test_m = evaluate(model, test_loader, GNN_OUT)

    log.info("═══ Final Test Result ═══")
    log.info(f"Test Accuracy : {test_m['accuracy']:.4f}")
    log.info(f"Test F1       : {test_m['f1']:.4f}")
    log.info(f"Test Precision: {test_m['precision']:.4f}")
    log.info(f"Test Recall   : {test_m['recall']:.4f}")

    print("\n===== FINAL TEST RESULT =====")
    print(f"Accuracy : {test_m['accuracy']:.4f}")
    print(f"F1       : {test_m['f1']:.4f}")
    print(f"Precision: {test_m['precision']:.4f}")
    print(f"Recall   : {test_m['recall']:.4f}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Load graph_data.pt ...")
    data = torch.load(GRAPH_PT, map_location="cpu", weights_only=False)
    log.info(f"Node types : {data.node_types}")
    log.info(f"Edge types : {data.edge_types}")

    log.info("=== KIỂM TRA METADATA TRƯỚC KHI HUẤN LUYỆN ===")
    
    for ntype in data.node_types:
        node_store = data[ntype]
        # Tìm các key mà giá trị không phải là torch.Tensor
        # Các thuộc tính như user_ids, names, urls thường là 'list'
        meta_attrs = [key for key, val in node_store.items() if not isinstance(val, torch.Tensor)]
        
        if meta_attrs:
            log.info(f"Nút '{ntype}' có các thuộc tính metadata: {meta_attrs}")
            # In thử một vài giá trị đầu tiên để xác nhận
            for attr in meta_attrs:
                sample_val = node_store[attr][:3] if isinstance(node_store[attr], list) else node_store[attr]
                log.info(f"  -> {attr} (mẫu): {sample_val}...")
            
            # Thực hiện xoá để tránh lỗi NeighborLoader
            for attr in meta_attrs:
                del node_store[attr]
            log.info(f"✓ Đã dọn dẹp xong metadata cho nút '{ntype}'")
        else:
            log.info(f"Nút '{ntype}' đã sạch (chỉ chứa Tensor đặc trưng).")


    log.info("═══ Bắt đầu Training ═══")
    train(data)
    log.info(f"✓ Best model → {BEST_MODEL}")


if __name__ == "__main__":
    main()
