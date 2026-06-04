"""
Ver 1: test trên snapshot cuối, chưa thêm module, KQ inductive thấp
a04_train_evaluate_onlysage.py — Huấn luyện & Đánh giá (chuẩn paper)
======================================================================
Thiết kế thí nghiệm:
    Phase 1 — train + test tĩnh CẢ 3 model trên train graph, so sánh
    Phase 2 — inductive inference CẢ 3 model trên test graph, so sánh

Cách chạy:
    python a04_train_evaluate_onlysage.py --phase 1
    python a04_train_evaluate_onlysage.py --phase 2

Routing forward_temporal (giữ nguyên logic a03):
    SAGE     : nhận TẤT CẢ T snapshot → TSA  (train T=7, test T=5)
    GAT/RGCN : nhận snapshot[-1]       → encode trực tiếp, không TSA

Fair comparison:
    Cả 3 model nhận cùng snapshot_list — SAGE tự dùng hết, GAT/RGCN tự lấy [-1]
    Cùng FocalLoss / Adam / ReduceLROnPlateau / EarlyStopping / threshold search
"""

import sys, time, argparse, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    average_precision_score,
)

from a02_graph_construction import (
    build_heterogeneous_graph, HETERO_GRAPH_PATH, HETERO_TEST_GRAPH_PATH
)
from code_db.a03_model_definition_onlysage_v1 import InsiderThreatDetector

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR     = RESULTS_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
GNN_TYPES   = ["sage", "rgcn", "gat"]
COLORS      = {"sage": "#3a86ff", "rgcn": "#ff6b6b", "gat": "#ffd166"}


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """FL(p_t) = -(1-p_t)^γ · log(p_t),  γ=2 theo Lin et al. ICCV 2017."""
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none',
            pos_weight=torch.tensor(self.pos_weight, device=logits.device))
        return ((1 - torch.exp(-bce)) ** self.gamma * bce).mean()


def _pos_weight(labels: torch.Tensor) -> float:
    n_pos = max(labels.sum().item(), 1)
    n_neg = (labels == 0).sum().item()
    return float(np.clip(n_neg / n_pos, 1.0, 50.0))


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD SEARCH — tối ưu F1 trên val set
# ══════════════════════════════════════════════════════════════════════════════

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    best_th, best_f1 = 0.5, -1.0
    for th in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (y_prob >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_th, best_f1


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_in_channels(snapshot) -> dict:
    return {nt: snapshot[nt].x.shape[1]
            for nt in snapshot.node_types
            if hasattr(snapshot[nt], 'x') and snapshot[nt].x is not None}


def _build_edge_dims(snapshot) -> list:
    """edge_dims theo đúng thứ tự metadata()[1]; edge không có attr → dim=1."""
    return [
        snapshot[et].edge_attr.shape[1]
        if (hasattr(snapshot[et], 'edge_attr') and snapshot[et].edge_attr is not None)
        else 1
        for et in snapshot.metadata()[1]
    ]


def _build_model(gnn_type: str, snapshot,
                 hidden_dim: int, num_heads: int,
                 tsa_layers: int, dropout: float) -> nn.Module:
    return InsiderThreatDetector(
        metadata         = snapshot.metadata(),
        in_channels_dict = _build_in_channels(snapshot),
        hidden_channels  = hidden_dim,
        num_heads        = num_heads,
        temporal_layers  = tsa_layers,
        dropout          = dropout,
        gnn_type         = gnn_type,
        edge_dims        = _build_edge_dims(snapshot),
    )


def _set_masks_test(snapshot_list: list) -> None:
    """Gán train/val=False, test=True cho session trong MỌI snapshot test."""
    for snap in snapshot_list:
        N = snap['session'].x.size(0)
        snap['session'].train_mask = torch.zeros(N, dtype=torch.bool)
        snap['session'].val_mask   = torch.zeros(N, dtype=torch.bool)
        snap['session'].test_mask  = torch.ones(N,  dtype=torch.bool)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module,
             snapshot_list: list,
             mask: torch.Tensor,
             criterion: nn.Module,
             threshold: float = 0.5) -> dict:
    """
    Đánh giá model trên tập chỉ định bởi mask.
    snapshot_list : toàn bộ danh sách snapshot (train hoặc test)
    mask          : boolean [N_sess] trên snapshot[-1]['session']
    threshold     : tìm trên val, áp dụng lại trên test
    """
    model.eval()
    last   = snapshot_list[-1]
    labels = last['session'].y
    logits = model.forward_temporal(snapshot_list)   # routing theo gnn_type trong a03
    probs  = torch.sigmoid(logits)

    loss   = criterion(logits[mask], labels[mask]).item()
    y_true = labels[mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')

    return dict(
        loss          = loss,
        accuracy      = accuracy_score(y_true, y_pred),
        precision     = precision_score(y_true, y_pred, zero_division=0),
        recall        = recall_score(y_true, y_pred, zero_division=0),
        f1            = f1_score(y_true, y_pred, zero_division=0),
        auc_roc       = auc,
        avg_precision = ap,
        y_true        = y_true,
        y_prob        = y_prob,
        threshold     = threshold,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _print_report(title: str, mask: torch.Tensor, res: dict) -> None:
    th     = res['threshold']
    y_true = np.asarray(res['y_true'])
    y_prob = np.asarray(res['y_prob'])
    cm     = confusion_matrix(y_true, (y_prob >= th).astype(int))
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")
    print(f"  Sessions      : {int(mask.sum()):,}")
    print(f"  Threshold     : {th:.3f}")
    print(f"  Accuracy      : {res['accuracy']*100:.2f}%")
    print(f"  Precision     : {res['precision']:.4f}")
    print(f"  Recall        : {res['recall']:.4f}")
    print(f"  F1-Score      : {res['f1']:.4f}  ← chỉ số cốt lõi paper")
    print(f"  AUC-ROC       : {res['auc_roc']:.4f}")
    print(f"  Avg Precision : {res['avg_precision']:.4f}  (PR-AUC)")
    print(f"  CM  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"{'═'*65}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP — dùng chung cho cả 3 model
# ══════════════════════════════════════════════════════════════════════════════

def _train_single(model: nn.Module, gnn_type: str,
                  snapshot_list: list, save_path: str,
                  num_epochs: int, lr: float, weight_decay: float,
                  patience: int, num_heads: int, tsa_layers: int,
                  dropout: float) -> dict:

    last       = snapshot_list[-1]
    train_mask = last['session'].train_mask
    val_mask   = last['session'].val_mask
    test_mask  = last['session'].test_mask
    labels     = last['session'].y

    pw        = _pos_weight(labels[train_mask])
    criterion = FocalLoss(gamma=2.0, pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    n_params     = sum(p.numel() for p in model.parameters())
    best_val_f1  = -1.0
    best_th      = 0.5
    no_improve   = 0
    t_total      = 0.0

    log.info(f"\n{'='*60}\n  {gnn_type.upper()} | Params={n_params:,} | pos_weight={pw:.2f}\n{'='*60}")

    for epoch in range(1, num_epochs + 1):
        # ── Train ───────────────────────────────────────────────────────
        model.train()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits = model.forward_temporal(snapshot_list)
        loss   = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        t_total += time.perf_counter() - t0

        # ── Validate: threshold search trên val set ──────────────────────
        val_raw = evaluate(model, snapshot_list, val_mask, criterion, threshold=0.5)
        best_th, cur_f1 = find_best_threshold(val_raw['y_true'], val_raw['y_prob'])
        scheduler.step(val_raw['loss'])

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  Ep {epoch:03d} | TrainLoss={loss.item():.4f} | "
                     f"ValLoss={val_raw['loss']:.4f} | ValF1={cur_f1:.4f} | "
                     f"AUC={val_raw['auc_roc']:.4f} | Th={best_th:.2f} | "
                     f"LR={optimizer.param_groups[0]['lr']:.2e}")

        # ── Early stopping ───────────────────────────────────────────────
        if cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            no_improve  = 0
            torch.save(dict(model_state_dict = model.state_dict(),
                            val_f1           = best_val_f1,
                            val_auc          = val_raw['auc_roc'],
                            epoch            = epoch,
                            best_threshold   = best_th,
                            gnn_type         = gnn_type,
                            n_params         = n_params,
                            hyperparams      = dict(hidden_dim  = model.hidden_channels
                                                    if hasattr(model, 'hidden_channels') else 128,
                                                    num_heads   = num_heads,
                                                    tsa_layers  = tsa_layers,
                                                    dropout     = dropout)),
                       save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  [EarlyStop] Ep={epoch} | best_val_f1={best_val_f1:.4f}")
                break

    # ── Evaluate test tĩnh với best checkpoint ───────────────────────────
    ckpt           = torch.load(save_path, weights_only=False)
    best_threshold = ckpt['best_threshold']
    model.load_state_dict(ckpt['model_state_dict'])
    log.info(f"  Best: epoch={ckpt['epoch']} | val_f1={ckpt['val_f1']:.4f} | "
             f"val_auc={ckpt['val_auc']:.4f} | threshold={best_threshold:.3f}")

    test_res = evaluate(model, snapshot_list, test_mask, criterion, threshold=best_threshold)
    test_res.update(train_time_s=round(t_total, 2), gnn_type=gnn_type,
                    n_params=n_params, best_epoch=ckpt['epoch'])
    return test_res


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — train CẢ 3 model trên train graph, so sánh
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(snapshot_list: list,
               num_epochs: int = 200, hidden_dim: int = 128,
               lr: float = 0.001, weight_decay: float = 1e-4,
               patience: int = 40, num_heads: int = 4,
               tsa_layers: int = 2, dropout: float = 0.3) -> dict:
    """
    Train + test tĩnh cả 3 model (SAGE, RGCN, GAT) trên train graph.
    Lưu checkpoint riêng cho từng model → dùng lại ở phase 2.
    In bảng so sánh + vẽ figure cuối cùng.
    """
    last        = snapshot_list[-1]
    all_results = {}

    log.info("\n" + "="*60)
    log.info("  PHASE 1 — TRAIN + TEST TĨNH (train graph, 60 ngày)")
    log.info("="*60)

    for gnn_type in GNN_TYPES:
        log.info(f"\n  → Đang train: {gnn_type.upper()}")
        model     = _build_model(gnn_type, last, hidden_dim, num_heads, tsa_layers, dropout)
        save_path = str(RESULTS_DIR / f"trained_{gnn_type}_detector_phase_1.pt")

        # Warm-up: kiểm tra forward pass trước khi train
        model.eval()
        with torch.no_grad():
            _ = model.forward_temporal(snapshot_list)
        log.info(f"    Warm-up OK | gnn_type={gnn_type}")

        res = _train_single(model, gnn_type, snapshot_list, save_path,
                            num_epochs, lr, weight_decay, patience,
                            num_heads, tsa_layers, dropout)
        all_results[gnn_type] = res
        _print_report(f"PHASE 1 TEST TĨNH — {gnn_type.upper()}",
                      last['session'].test_mask, res)

    # ── Bảng so sánh phase 1 ─────────────────────────────────────────────
    _print_comparison_table("PHASE 1 — SO SÁNH TEST TĨNH (train graph)", all_results)

    # ── Figure: F1 bar + ROC curve ────────────────────────────────────────
    _plot_results("phase1_train_graph", all_results,
                  title_bar="F1 — Test Tĩnh (Train Graph)",
                  title_roc="ROC — Test Tĩnh (Train Graph)")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS PHASE 2 — sliding-window inference
# ══════════════════════════════════════════════════════════════════════════════

def _infer_one_snapshot(model: nn.Module, gnn_type: str,
                        window: list, threshold: float,
                        criterion: nn.Module) -> dict:
    """
    Inference 1 window (list snapshot) → metrics trên snapshot[-1].
    Dùng chung cho cả SAGE (window dài) và GAT/RGCN (window bất kỳ, tự lấy [-1]).
    """
    last      = window[-1]
    test_mask = last['session'].test_mask
    return evaluate(model, window, test_mask, criterion, threshold=threshold)


def _aggregate_window_results(window_results: list) -> dict:
    """
    Tổng hợp kết quả qua các window bằng cách nối y_true / y_prob.
    Tính lại tất cả metrics từ đầu trên tập nối — không average metrics.
    KQ cuối là của window[-1] (đầy đủ context nhất), các window trước
    được log để quan sát độ hội tụ theo thời gian.
    """
    # Kết quả cuối = window cuối (đủ context nhất)
    final = window_results[-1]
    # Gắn thêm per-window F1 để vẽ biểu đồ temporal
    final['per_window_f1']     = [r['f1']     for r in window_results]
    final['per_window_auc']    = [r['auc_roc'] for r in window_results]
    final['per_window_recall'] = [r['recall']  for r in window_results]
    return final


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — sliding-window inductive inference trên test graph
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2(snapshot_list_test: list,
               hidden_dim: int = 128, num_heads: int = 4,
               tsa_layers: int = 2, dropout: float = 0.3) -> dict:
    """
    Sliding-window inference trên toàn bộ M snapshot của test graph.
    Không retrain — dùng checkpoint từ phase 1.

    Chiến lược từng model:
    ┌──────────┬───────────────────────────────────────────────────────────────┐
    │ SAGE     │ Growing window: [:1], [:2], ..., [:M]                        │
    │          │ Mỗi window forward_temporal → TSA học chuỗi tăng dần         │
    │          │ Window [:M] là kết quả cuối (context đầy đủ nhất)            │
    │          │ → INDUCTIVE: không retrain, generalize user/session mới      │
    ├──────────┼───────────────────────────────────────────────────────────────┤
    │ GAT/RGCN │ Single snapshot: [0], [1], ..., [M-1]                        │
    │          │ Mỗi lần chỉ encode snapshot đó (forward_temporal lấy [-1])   │
    │          │ Metrics cuối = snapshot M-1 (snapshot mới nhất)              │
    │          │ → TRANSDUCTIVE: graph mới → cần retrain về lý thuyết         │
    └──────────┴───────────────────────────────────────────────────────────────┘

    Fair comparison:
    - Cả 3 đều được thấy thông tin của snapshot M-1 (target mới nhất)
    - SAGE có thêm lợi thế temporal context từ 0..M-2 — đây là đóng góp của TSA
    - Số snapshot (M) bằng nhau giữa 3 model; chỉ cách sử dụng khác nhau

    Output per-window cho phép vẽ đường cong temporal F1 theo thời gian.
    """
    _set_masks_test(snapshot_list_test)
    M          = len(snapshot_list_test)
    last_test  = snapshot_list_test[-1]
    all_results = {}

    log.info("\n" + "="*60)
    log.info(f"  PHASE 2 — SLIDING-WINDOW INFERENCE ({M} test snapshots)")
    log.info("="*60)
    log.info(f"  SAGE     : growing window [:1]→[:2]→...→[:{M}]")
    log.info(f"  GAT/RGCN : single snapshot [0]→[1]→...→[{M-1}]")

    for gnn_type in GNN_TYPES:
        ckpt_path = RESULTS_DIR / f"trained_{gnn_type}_detector_phase_1.pt"
        if not ckpt_path.exists():
            log.error(f"  Chưa có '{ckpt_path}'. Chạy --phase 1 trước.")
            sys.exit(1)

        ckpt           = torch.load(ckpt_path, weights_only=False)
        best_threshold = ckpt.get('best_threshold', 0.5)

        # Build model với metadata test graph (user/session mới có thể xuất hiện)
        model = _build_model(gnn_type, last_test, hidden_dim, num_heads, tsa_layers, dropout)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        log.info(f"\n  [{gnn_type.upper()}] epoch={ckpt['epoch']} | "
                 f"val_f1={ckpt['val_f1']:.4f} | threshold={best_threshold:.3f}")

        # ── Dùng pos_weight của snapshot cuối để tính loss ──────────────
        criterion = FocalLoss(2.0, _pos_weight(last_test['session'].y))

        # ── Sliding-window inference ─────────────────────────────────────
        window_results = []
        t0_total = time.perf_counter()

        if gnn_type == "sage":
            # Growing window: [:1], [:2], ..., [:M]
            # SAGE dùng TẤT CẢ snapshot trong window → TSA có ngữ cảnh tăng dần
            for end in range(1, M + 1):
                window = snapshot_list_test[:end]
                res_w  = _infer_one_snapshot(model, gnn_type, window,
                                             best_threshold, criterion)
                window_results.append(res_w)
                log.info(f"    Window [:{end:2d}] | "
                         f"F1={res_w['f1']:.4f} | "
                         f"AUC={res_w['auc_roc']:.4f} | "
                         f"Rec={res_w['recall']:.4f}")
        else:
            # Single snapshot: [0], [1], ..., [M-1]
            # GAT/RGCN forward_temporal tự lấy snapshot[-1] → mỗi lần encode 1 snap
            for idx in range(M):
                window = [snapshot_list_test[idx]]   # list 1 phần tử
                res_w  = _infer_one_snapshot(model, gnn_type, window,
                                             best_threshold, criterion)
                window_results.append(res_w)
                log.info(f"    Snap [{idx:2d}]    | "
                         f"F1={res_w['f1']:.4f} | "
                         f"AUC={res_w['auc_roc']:.4f} | "
                         f"Rec={res_w['recall']:.4f}")

        inf_t = time.perf_counter() - t0_total

        # ── Kết quả cuối = window/snapshot cuối ─────────────────────────
        res = _aggregate_window_results(window_results)
        res.update(gnn_type=gnn_type, inf_time_ms=round(inf_t * 1000, 2),
                   n_windows=M)
        all_results[gnn_type] = res

        inductive_tag = ("INDUCTIVE ✓ — growing window, không retrain"
                         if gnn_type == "sage"
                         else "TRANSDUCTIVE ✗ — single snapshot per step")
        _print_report(f"PHASE 2 [{inductive_tag}] — {gnn_type.upper()} (snapshot cuối)",
                      last_test['session'].test_mask, res)
        log.info(f"    Tổng inference time: {inf_t*1000:.1f} ms ({M} windows)")

    # ── Bảng tổng hợp kết quả snapshot cuối ─────────────────────────────
    _print_comparison_table(
        f"PHASE 2 — SO SÁNH INDUCTIVE TEST (kết quả snapshot cuối, M={M})",
        all_results)

    # ── Figure: F1 bar + ROC + per-window temporal curve ─────────────────
    _plot_results("phase2_inductive_test", all_results,
                  title_bar=f"F1 — Inductive Test (snapshot cuối, M={M})",
                  title_roc=f"ROC — Inductive Test (M={M})")

    _plot_temporal_curve(all_results, M)

    return all_results


def _plot_temporal_curve(all_results: dict, M: int) -> None:
    """
    Vẽ đường cong F1 / Recall theo từng window/snapshot trong phase 2.
    Cho thấy SAGE cải thiện dần khi context tăng, GAT/RGCN ổn định từng snap.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x_sage = list(range(1, M + 1))      # window size 1..M
    x_base = list(range(0, M))          # snapshot index 0..M-1

    for gt in GNN_TYPES:
        if gt not in all_results: continue
        r    = all_results[gt]
        c    = COLORS[gt]
        f1s  = r.get('per_window_f1', [])
        recs = r.get('per_window_recall', [])
        xs   = x_sage if gt == "sage" else x_base
        label_f  = f"{gt.upper()} ({'growing window' if gt=='sage' else 'single snap'})"
        label_r  = label_f

        if f1s:
            axes[0].plot(xs, f1s,  marker='o', lw=2, ms=5, color=c, label=label_f)
            axes[1].plot(xs, recs, marker='s', lw=2, ms=5, color=c, label=label_r)

    for ax, ylabel, title in [
        (axes[0], 'F1-Score',   f'F1 theo window/snapshot (M={M})'),
        (axes[1], 'Recall',     f'Recall theo window/snapshot (M={M})'),
    ]:
        ax.set_xlabel('Window size (SAGE) / Snapshot index (GAT/RGCN)')
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out = FIG_DIR / 'phase2_temporal_curve.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  → Temporal curve: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# BẢNG SO SÁNH
# ══════════════════════════════════════════════════════════════════════════════

def _print_comparison_table(title: str, all_results: dict) -> None:
    inductive_map = {'sage': 'YES ✓', 'rgcn': 'NO ✗', 'gat': 'NO ✗'}
    sep = "─" * 80
    hdr = (f"  {'Model':<8} {'F1':>6} {'AUC':>6} {'AP':>6} "
           f"{'Prec':>6} {'Rec':>6} {'Acc':>6} {'Inductive':>10}")
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"{sep}\n{hdr}\n{sep}")
    for gt in GNN_TYPES:
        if gt not in all_results:
            continue
        r   = all_results[gt]
        pfx = "**" if gt == "sage" else "  "
        print(f"{pfx} {gt.upper():<7} "
              f"{r['f1']:>6.4f} {r['auc_roc']:>6.4f} "
              f"{r.get('avg_precision', float('nan')):>6.4f} "
              f"{r['precision']:>6.4f} {r['recall']:>6.4f} "
              f"{r['accuracy']:>6.4f} {inductive_map[gt]:>10}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _plot_results(fname_prefix: str, all_results: dict,
                  title_bar: str, title_roc: str) -> None:
    from sklearn.metrics import precision_recall_curve

    names  = [gt.upper() for gt in GNN_TYPES if gt in all_results]
    colors = [COLORS[gt] for gt in GNN_TYPES if gt in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    # ── Hình 1: F1 bar ────────────────────────────────────────────────────
    f1s  = [all_results[gt]['f1'] for gt in GNN_TYPES if gt in all_results]
    bars = axes[0].bar(names, f1s, color=colors, edgecolor='white', width=0.5)
    axes[0].bar_label(bars, fmt='%.4f', padding=3, fontsize=10)
    axes[0].set_ylim(0, 1.15); axes[0].set_ylabel('F1-Score')
    axes[0].set_title(title_bar); axes[0].grid(axis='y', alpha=0.3)

    # ── Hình 2: ROC curve ─────────────────────────────────────────────────
    for gt, c in zip(GNN_TYPES, colors):
        if gt not in all_results: continue
        r = all_results[gt]
        if len(np.unique(r['y_true'])) < 2: continue
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
        axes[1].plot(fpr, tpr, lw=2, color=c,
                     label=f"{gt.upper()} (AUC={r['auc_roc']:.3f})")
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
    axes[1].set_title(title_roc); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # ── Hình 3: PR curve ──────────────────────────────────────────────────
    for gt, c in zip(GNN_TYPES, colors):
        if gt not in all_results: continue
        r = all_results[gt]
        if len(np.unique(r['y_true'])) < 2: continue
        prec, rec, _ = precision_recall_curve(r['y_true'], r['y_prob'])
        ap = r.get('avg_precision', float('nan'))
        axes[2].plot(rec, prec, lw=2, color=c,
                     label=f"{gt.upper()} (AP={ap:.3f})")
    axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
    axes[2].set_title(title_roc.replace('ROC', 'PR-Curve'))
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / f'{fname_prefix}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  → Figure: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",      type=int,   default=1, choices=[1, 2])
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--patience",   type=int,   default=40)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--heads",      type=int,   default=4)
    parser.add_argument("--tsa_layers", type=int,   default=2)
    args = parser.parse_args()

    # ── Load train graph ──────────────────────────────────────────────────
    try:
        snapshot_list = torch.load(HETERO_GRAPH_PATH, weights_only=False)
        N_last = snapshot_list[-1]['session'].x.size(0)
        log.info(f"Train graph: '{HETERO_GRAPH_PATH}' | "
                 f"{len(snapshot_list)} snaps | {N_last} sessions (last)")
    except FileNotFoundError:
        log.error(f"Không tìm thấy '{HETERO_GRAPH_PATH}'. Chạy a02 trước.")
        sys.exit(1)

    # ── Load test graph ───────────────────────────────────────────────────
    snapshot_list_test = None
    try:
        snapshot_list_test = torch.load(HETERO_TEST_GRAPH_PATH, weights_only=False)
        _set_masks_test(snapshot_list_test)
        N_test = snapshot_list_test[-1]['session'].x.size(0)
        log.info(f"Test graph : '{HETERO_TEST_GRAPH_PATH}' | "
                 f"{len(snapshot_list_test)} snaps | {N_test} sessions (last)")
    except FileNotFoundError:
        log.warning(f"Không tìm thấy '{HETERO_TEST_GRAPH_PATH}' — chỉ chạy phase 1.")

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.phase == 1:
        run_phase1(snapshot_list,
                   num_epochs  = args.epochs,
                   hidden_dim  = args.hidden,
                   lr          = args.lr,
                   patience    = args.patience,
                   num_heads   = args.heads,
                   tsa_layers  = args.tsa_layers,
                   dropout     = args.dropout)

    elif args.phase == 2:
        if snapshot_list_test is None:
            log.error("--phase 2 cần test graph. Chạy build_inductive_test_dataset.py trước.")
            sys.exit(1)
        run_phase2(snapshot_list_test,
                   hidden_dim  = args.hidden,
                   num_heads   = args.heads,
                   tsa_layers  = args.tsa_layers,
                   dropout     = args.dropout)
