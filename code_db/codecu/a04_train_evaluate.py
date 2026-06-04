"""
a04_train_evaluate.py — Huấn luyện & Đánh giá (DB tự sinh, Temporal Mode)
==========================================================================
    TÍNH NĂNG MỚI ĐỂ CHỨNG MINH LUẬN ĐIỂM KHOA HỌC:
  - PHASE 1 (Static Phase): Huấn luyện trên tập sạch (20 nhân viên cũ, 45 ngày), 
                            giữ nguyên cấu trúc masking ngẫu nhiên, lưu mô hình 
                            thành 'trained_insider_detector.pt'.
  - PHASE 2 (Inductive Phase): Nạp mô hình 'trained_insider_detector.pt', chạy trên 
                               tập dữ liệu OOD (10 nhân viên mới, data phình to), 
                               ép 100% test_mask = True, hoàn toàn không chạy Backprop.
    - Khi chạy: gán đường dẫn file .pt (tương ứng với 2 file graph )
    python a04_train_evaluate.py --phase 1

    tính quy nạp
    python a04_train_evaluate.py --phase 2

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import argparse

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from a02_graph_construction import build_heterogeneous_graph, HETERO_GRAPH_PATH
from codecu.a03_model_definition import InsiderThreatDetector

# MODEL_SAVE_PATH = "trained_insider_detector.pt"
THRESHOLD       = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS (thay BCEWithLogitsLoss — xử lý imbalance tốt hơn)
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    FL(p_t) = -α_t (1-p_t)^γ log(p_t)
    γ=2: tập trung vào anomaly session hiếm gặp.
    Nhận raw logits [N] + binary labels [N] (float).
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=torch.tensor(self.pos_weight, device=logits.device))
        p_t   = torch.exp(-bce)
        focal = (1 - p_t) ** self.gamma * bce
        return focal.mean()


def compute_pos_weight(labels: torch.Tensor) -> float:
    n_pos = labels.sum().item()
    n_neg = (labels == 0).sum().item()
    return n_neg / max(n_pos, 1)


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD — dùng model.forward_temporal() đầy đủ (FIX BUG-12)
# ══════════════════════════════════════════════════════════════════════════════

def _get_logits(model: InsiderThreatDetector,
                snapshot_list: list) -> torch.Tensor:
    """
    Trả về RAW LOGITS [N_sess_global] từ model.forward_temporal().
    BUG-12 FIX: gọi đủ 3 thành phần (GNN + SemanticAttn + TSA).
    BUG-13/14 FIX: shape [N_sess_global] khớp với mask từ snapshot cuối
    vì mọi snapshot dùng chung session_map toàn cục (xem a02 BUG-05 comment).
    """
    return model.forward_temporal(snapshot_list)   # [N_sess]


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE INTERFERENCE / TEST
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, snapshot_list, mask, criterion, threshold=THRESHOLD):
    model.eval()
    logits      = _get_logits(model, snapshot_list)          # [N_sess]
    probs       = torch.sigmoid(logits)                      # [N_sess]

    # Nhãn lấy từ snapshot cuối (target graph)
    true_labels = snapshot_list[-1]['session'].y[mask]
    loss        = criterion(logits[mask], true_labels).item()

    y_true = true_labels.cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float('nan')
    except ValueError:
        auc = float('nan')

    return dict(
        loss      = loss,
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall    = recall_score(y_true, y_pred, zero_division=0),
        f1        = f1_score(y_true, y_pred, zero_division=0),
        auc_roc   = auc,
        y_true    = y_true,
        y_prob    = y_prob,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    phase:        int   = 1,
    gnn_type:     str   = "sage",  # Tham số mới truyền từ tham số dòng lệnh
    num_epochs:   int   = 100,
    hidden_dim:   int   = 64,
    lr:           float = 0.005,
    weight_decay: float = 1e-4,
    patience:     int   = 15,
    threshold:    float = THRESHOLD,
    num_heads:    int   = 4,
    tsa_layers:   int   = 2,
    dropout:      float = 0.3,
):
    # Tự động thay đổi tên file checkpoint bám theo loại mô hình đang chạy
    current_model_save_path = f"trained_{gnn_type}_detector_phase_1.pt"

    print(f"\n==========================================================")
    print(f" KHỞI CHẠY TIẾN TRÌNH: GIAI ĐOẠN {phase}")
    print(f"==========================================================")

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("=== BƯỚC 3: Tải chuỗi snapshot ===")
    try:
        snapshot_list = torch.load(HETERO_GRAPH_PATH, weights_only=False)
        print(f"   Loaded '{HETERO_GRAPH_PATH}': {len(snapshot_list)} snapshots")
    except FileNotFoundError:
        print(f"   [LỖI] Không tìm thấy file '{HETERO_GRAPH_PATH}'!")
        print("   -> Bạn cần chạy pipeline tiền xử lý và xây dựng đồ thị tương ứng trước.")
        sys.exit(1)

    last      = snapshot_list[-1]
    # Thiết lập cơ chế Masking dựa trên Giai đoạn thực nghiệm
    if phase == 1:
        # Giai đoạn tĩnh: Bám theo phân tách train/val/test ngẫu nhiên của tập cũ
    # Mask + labels lấy từ snapshot cuối (target graph)
        train_mask = last['session'].train_mask
        val_mask   = last['session'].val_mask
        test_mask  = last['session'].test_mask
    else:
        # Giai đoạn Quy nạp: Ép buộc 100% nút session mới thành tập test, khóa chặt tập huấn luyện
        num_sessions = last['session'].x.size(0)
        test_mask  = torch.ones(num_sessions, dtype=torch.bool)
        train_mask = torch.zeros(num_sessions, dtype=torch.bool)
        val_mask   = torch.zeros(num_sessions, dtype=torch.bool)
        
        # Đồng bộ lại vào object đồ thị con cuối cùng tránh lỗi cục bộ
        last['session'].test_mask  = test_mask
        last['session'].train_mask = train_mask
        last['session'].val_mask   = val_mask

    labels     = last['session'].y
    n_total = labels.shape[0]
    n_pos   = int(labels.sum())
    print(f"   Sessions: {n_total} | Anomaly: {n_pos} | Normal: {n_total - n_pos}")
    print(f"   Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    # ── 2. Model ───────────────────────────────────────────────────────────────
    # ── 2. KHỞI TẠO KIẾN TRÚC MÔ HÌNH HẨN (MẠNG KHÔNG ĐỒNG NHẤT) ──────────────────
    print("=== BƯỚC 2: Cấu hình khung mạng học sâu InsiderThreatDetector ===")
    in_channels_dict = {
        'user': 4,          # [Role_Index, Department_Index, Clearance_Level, Target_Score]
        'session': 2,       # [Time_Window_Norm, Is_Internal_IP]
        'sql_template': 4,  # [Command_Type, Join_Count, Where_Count, Has_Subquery]
        'table': 2          # [Sensitivity_Level, Total_Records_Baseline]
    }
    
    model = InsiderThreatDetector(
        metadata        = last.metadata(),
        in_channels_dict= in_channels_dict, # Nạp số chiều tường minh tại đây
        hidden_channels = hidden_dim,
        num_heads       = num_heads,
        temporal_layers = tsa_layers,
        dropout         = dropout,
        gnn_type        = gnn_type, # Khai báo biến rẽ nhánh tại đây
    )

    # Dummy forward để khởi tạo tất cả params
    dummy_snapshot = snapshot_list[:1]  # 1 snapshot
    _ = model.forward_temporal(dummy_snapshot)

    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    # Tính toán trọng số Loss chống lệch lớp phân phối dữ liệu
    if phase == 1:
        pw        = compute_pos_weight(labels[train_mask])
    else:
        pw = compute_pos_weight(labels) # Dự phòng cho hàm Loss tĩnh trong Inference

    criterion = FocalLoss(gamma=2.0, pos_weight=pw)
    # ── 3. RẼ NHÁNH XỬ LÝ THEO KỊCH BẢN KHOA HỌC ──────────────────────────────────
    if phase == 1:
        print(f"\n=== BƯỚC 3: Huấn luyện mô hình tĩnh ({num_epochs} Epochs, Kỳ vọng Tối ưu Cơ sở) ===")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
        print(f"   pos_weight: {pw:.2f}")

        # ── 3. Training ────────────────────────────────────────────────────────────
        print(f"\n=== BƯỚC 5: Huấn luyện ({num_epochs} epochs, patience={patience}) ===")
        best_val_f1  = -1.0
        no_improve   = 0

        for epoch in range(1, num_epochs + 1):
            # Train step
            model.train()
            optimizer.zero_grad()
            logits     = _get_logits(model, snapshot_list)        # [N_sess]
            train_loss = criterion(logits[train_mask], labels[train_mask])
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation step
            val_res = evaluate(model, snapshot_list, val_mask, criterion, threshold)
            scheduler.step(val_res['f1'])

            if epoch % 10 == 0 or epoch == 1:
                print(f"Ep {epoch:03d} | "
                    f"Train: {train_loss.item():.4f} | "
                    f"Val Loss: {val_res['loss']:.4f} | "
                    f"Val F1: {val_res['f1']:.4f} | "
                    f"Val AUC: {val_res['auc_roc']:.4f}")

            # Early stopping (theo F1 tăng)
            if val_res['f1'] > best_val_f1:
                best_val_f1 = val_res['f1']
                no_improve  = 0
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': model.state_dict(),
                    'val_f1':           best_val_f1,
                    'hyperparams': {
                        'hidden_dim':  hidden_dim,
                        'num_heads':   num_heads,
                        'tsa_layers':  tsa_layers,
                        'dropout':     dropout,
                        'temporal':    'self_attention',
                    },
                }, current_model_save_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n[Early Stop] Epoch {epoch} | best val_f1={best_val_f1:.4f}")
                    break
        print(f"\n[THÀNH CÔNG GIAI ĐOẠN 1] Đã huấn luyện và lưu bộ não AI vào tệp: '{MODEL_SAVE_PATH}'")
        # Đánh giá trực tiếp ra báo cáo cho Bằng chứng số 1
        model.load_state_dict(torch.load(current_model_save_path, weights_only=False))
        test_res = evaluate(model, snapshot_list, test_mask, criterion, threshold)
        _print_report("BẰNG CHỨNG SỐ 1: KẾT QUẢ ĐÁNH GIÁ TĨNH (STATIC PHASE)", test_mask, test_res, threshold)

    elif phase == 2:
        print(f"\n=== BƯỚC 3: Kiểm thử năng lực Quy nạp Tổng quát hóa (OOD Môi trường Biến động) ===")
        print(f"   -> Đang nạp tệp trọng số đóng gói cứng: '{current_model_save_path}'")
        
        try:
            model.load_state_dict(torch.load(current_model_save_path, weights_only=False))
        except FileNotFoundError:
            print(f"   [THẤT BẠI CRITICAL] Không tìm thấy file trọng số '{current_model_save_path}'!")
            print("   -> Bạn bắt buộc phải chạy lệnh với tham số '--phase 1' trước để sinh bộ não AI.")
            sys.exit(1)
            
        model.eval() # Chuyển nghiêm chỉnh sang chế độ đóng băng suy luận, tắt hoàn toàn Dropout/BatchNorm
        
        # Đo đạc trực tiếp chuỗi biến động mới mà KHÔNG chạy hàm Loss hay Backpropagation tối ưu lại
        inductive_res = evaluate(model, snapshot_list, test_mask, criterion, threshold)
        _print_report("BẰNG CHỨNG SỐ 2: KẾT QUẢ KIỂM THỬ QUY NẠP (INDUCTIVE PROOF)", test_mask, inductive_res, threshold)

# ══════════════════════════════════════════════════════════════════════════════
# TIỆN ÍCH IN BÁO CÁO KHOA HỌC
# ══════════════════════════════════════════════════════════════════════════════

def _print_report(title, mask, res, threshold):
    cm = confusion_matrix(res['y_true'], (res['y_prob'] >= threshold).astype(int))
    print(f"\n════════════════ {title} ════════════════")
    print(f" Môi trường thực nghiệm : Hệ thống CSDL Thao tác Phức tạp (TPC-H Benchmark)")
    print(f" Tổng số Sessions quét  : {mask.sum().item()} bản ghi đồ thị tạm thời")
    print(f" Độ chính xác (Accuracy): {res['accuracy']*100:.2f}%")
    print(f" Độ chuẩn xác (Precision): {res['precision']:.4f}")
    print(f" Độ nhạy (Recall)       : {res['recall']:.4f}")
    print(f" Chỉ số F1-Score Gốc    : {res['f1']:.4f}  <-- [CHỈ SỐ CỐT LÕI BÀI BÁO]")
    print(f" Diện tích dưới đường cong (AUC-ROC): {res['auc_roc']:.4f}")
    print(f"\n Ma trận nhầm lẫn (Confusion Matrix):")
    print(f"   [Thực tế Normal]  TN = {cm[0,0]:<6} | FP = {cm[0,1]:<6} (Báo động giả)")
    print(f"   [Thực tế Attack]  FN = {cm[1,0]:<6} | TP = {cm[1,1]:<6} (Bắt được trộm)")
    print(f"══════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description="Pipeline Huấn luyện/Suy luận Quy nạp so sánh mô hình.")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Chọn 1: Huấn luyện tĩnh. Chọn 2: Đánh giá quy nạp.")
    # # Thêm cấu hình chọn model nền tảng so sánh
    # parser.add_argument("--model", type=str, default="sage", choices=["sage", "rgcn", "gat"],
    #                     help="Chọn lõi xử lý không gian đồ thị: sage (GraphSAGE), rgcn (RGCN), gat (GAT)")
    args = parser.parse_args()

    # Danh sách mô hình muốn chạy so sánh
    models_to_run = ["sage", "rgcn", "gat"]

    for model_name in models_to_run:
        print(f"\n\n================ Running model: {model_name.upper()} ================\n")
    
        run_pipeline(
            phase        = args.phase,
            gnn_type     = model_name, # Truyền cấu hình mô hình được lựa chọn
            num_epochs   = 100,
            hidden_dim   = 64,
            lr           = 0.005,
            weight_decay = 1e-4,
            patience     = 15,
            threshold    = THRESHOLD,
            num_heads    = 4,
            tsa_layers   = 2,
            dropout      = 0.3,
        )