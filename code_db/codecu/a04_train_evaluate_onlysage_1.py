"""
a04_train_evaluate.py — Huấn luyện & Đánh giá (DB tự sinh)
===========================================================
Sửa lỗi so với phiên bản upload:
    BUG-D1: MODEL_SAVE_PATH bị comment → NameError
            FIX: dùng current_model_save_path nhất quán
    BUG-D2: in_channels_dict hardcode → crash nếu a01 thay đổi
            FIX: tự đọc từ snapshot
    BUG-D3: --model argument bị comment nhưng code vẫn cần
            FIX: bỏ comment, thêm mode so sánh rõ ràng

Cách chạy:
    # Train + test từng model riêng:
    python a04_train_evaluate.py --phase 1 
    # Chứng minh tính quy nạp (chạy sau phase 1):
    python a04_train_evaluate.py --phase 2 --model sage

    # So sánh cả 3 model (train + inductive):
    python a04_train_evaluate_onlysage.py --compare
"""

import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

from a02_graph_construction import build_heterogeneous_graph, HETERO_GRAPH_PATH, HETERO_TEST_GRAPH_PATH
from codecu.a03_model_definition_onlysage_1 import InsiderThreatDetector

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

THRESHOLD   = 0.5
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR     = RESULTS_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.gamma = gamma; self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none',
            pos_weight=torch.tensor(self.pos_weight, device=logits.device))
        p_t   = torch.exp(-bce)
        return ((1 - p_t) ** self.gamma * bce).mean()


def _pos_weight(labels):
    n_pos = labels.sum().item()
    n_neg = (labels == 0).sum().item()
    return np.sqrt(n_neg / max(n_pos, 1))


# BUG-D2 FIX: tự đọc chiều feature từ snapshot
def _build_in_channels(snapshot) -> dict:
    return {nt: snapshot[nt].x.shape[1]
            for nt in snapshot.node_types
            if hasattr(snapshot[nt], 'x') and snapshot[nt].x is not None}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, snapshot_list, mask, criterion, edge_attr_dict, threshold=THRESHOLD):
    model.eval()
    last   = snapshot_list[-1]
    labels = last['session'].y
    logits = model.forward_temporal(snapshot_list, edge_attr_dict=edge_attr_dict)
    probs  = torch.sigmoid(logits)
    loss   = criterion(logits[mask], labels[mask]).item()
    y_true = labels[mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float('nan')
    except ValueError:
        auc = float('nan')
    return dict(loss=loss,
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1=f1_score(y_true, y_pred, zero_division=0),
                auc_roc=auc, y_true=y_true, y_prob=y_prob)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP — fair comparison: cùng loss/optimizer/stopping cho cả 3
# ══════════════════════════════════════════════════════════════════════════════

def _train_single(model, gnn_type, snapshot_list, save_path,
                  num_epochs, lr, weight_decay, patience,
                  num_heads, tsa_layers, dropout, edge_attr_dict):
    last       = snapshot_list[-1]
    train_mask = last['session'].train_mask
    val_mask   = last['session'].val_mask
    test_mask  = last['session'].test_mask
    labels     = last['session'].y

    pw        = _pos_weight(labels[train_mask])
    criterion = FocalLoss(gamma=2.0, pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5) # ban đầu là max

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"\n{'='*55}\n  Model: {gnn_type.upper()} | "
             f"Params: {n_params:,} | pos_weight: {pw:.2f}\n{'='*55}")

    best_val_f1, no_improve, t_total = -1.0, 0, 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits = model.forward_temporal(snapshot_list, edge_attr_dict = edge_attr_dict)
        loss   = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        t_total += time.perf_counter() - t0

        val_res = evaluate(model, snapshot_list, val_mask, criterion, edge_attr_dict = edge_attr_dict)
        scheduler.step(val_res['loss'])
        # scheduler.step(val_res['f1'])

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  Ep {epoch:03d} | Loss: {loss.item():.4f} | "
                     f"Val F1: {val_res['f1']:.4f} | AUC: {val_res['auc_roc']:.4f}")

        if val_res['f1'] > best_val_f1:
            best_val_f1 = val_res['f1']; no_improve = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'val_f1': best_val_f1, 'epoch': epoch,
                        'gnn_type': gnn_type,
                        'hyperparams': dict(num_heads=num_heads,
                                            tsa_layers=tsa_layers,
                                            dropout=dropout)},
                       save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  [Early Stop] Ep {epoch} | best_f1={best_val_f1:.4f}")
                break

    ckpt = torch.load(save_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_res = evaluate(model, snapshot_list, test_mask, criterion, edge_attr_dict = edge_attr_dict)
    test_res.update(train_time_s=round(t_total, 2),
                    gnn_type=gnn_type, n_params=n_params,
                    best_epoch=ckpt['epoch'])
    _print_report(
        f"KẾT QUẢ ĐÁNH GIÁ TĨNH — {gnn_type.upper()}",
        test_mask, test_res)
    return test_res


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — train 1 model
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(gnn_type, snapshot_list, num_epochs=100, hidden_dim=64,
               lr=0.005, weight_decay=1e-4, patience=40,
               num_heads=4, tsa_layers=2, dropout=0.3):

    last             = snapshot_list[-1]
    in_channels_dict = _build_in_channels(last)     # Tự xác định chiều dựa vào lớp cuối cùng


    # Build edge_attr_dict từ snapshot
    edge_attr_dict = {
        ('user','opens','session'): last['user','opens','session'].edge_attr,
        ('user','executes','sql_template'): last['user','executes','sql_template'].edge_attr,
        ('sql_template','belongs_to','session'): last['sql_template','belongs_to','session'].edge_attr,
        ('sql_template','affects','table'): last['sql_template','affects','table'].edge_attr,
    }
    # Tạo list edge_dims theo thứ tự edge_types
    edge_dims = [
        edge_attr_dict[et].shape[1] if et in edge_attr_dict else 1
        for et in last.metadata()[1]
    ]
    
    log.info(f"in_channels_dict: {in_channels_dict}")
    log.info(f"edge_attr_dict: {edge_attr_dict}, \n edge_dim: {edge_dims}")

    # Gọi model và bắt đầu train
    model = InsiderThreatDetector(
        metadata=last.metadata(), in_channels_dict=in_channels_dict,
        hidden_channels=hidden_dim, num_heads=num_heads,
        temporal_layers=tsa_layers, dropout=dropout, gnn_type=gnn_type, 
        edge_dims=edge_dims )

    model.eval()
    with torch.no_grad():
        _ = model.forward_temporal(snapshot_list[:1], edge_attr_dict=edge_attr_dict)

    # BUG-D1 FIX: dùng tên biến nhất quán
    save_path = str(RESULTS_DIR / f"trained_{gnn_type}_detector_phase_1.pt")
    return _train_single(model, gnn_type, snapshot_list, save_path,
                         num_epochs, lr, weight_decay, patience,
                         num_heads, tsa_layers, dropout,
                         edge_attr_dict = edge_attr_dict)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — inductive inference (không retrain)
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2(gnn_type, snapshot_list, hidden_dim=64,
               num_heads=4, tsa_layers=2, dropout=0.3):
    """
    Nạp model đã train (phase 1), inference trên 100% sessions.
    SAGE: không cần retrain → INDUCTIVE.
    RGCN/GAT: inference được trên graph cũ, nhưng nếu graph thay đổi
    (thêm node/edge mới) → A cần được tính lại → TRANSDUCTIVE.
    """
    # BUG-D1 FIX: load đúng file phase 1
    load_path = str(RESULTS_DIR / f"trained_{gnn_type}_detector_phase_1.pt")
    if not Path(load_path).exists():
        log.error(f"Chưa có '{load_path}'. Chạy --phase 1 --model {gnn_type} trước.")
        sys.exit(1)

    last             = snapshot_list[-1]
    in_channels_dict = _build_in_channels(last)     # BUG-D2 FIX
        # Build edge_attr_dict từ snapshot
    edge_attr_dict = {
        ('user','opens','session'): last['user','opens','session'].edge_attr,
        ('user','executes','sql_template'): last['user','executes','sql_template'].edge_attr,
        ('sql_template','belongs_to','session'): last['sql_template','belongs_to','session'].edge_attr,
        ('sql_template','affects','table'): last['sql_template','affects','table'].edge_attr,
    }
    # Tạo list edge_dims theo thứ tự edge_types
    # edge_dims = [edge_attr_dict[et].shape[1] for et in edge_attr_dict]
    edge_dims = [
        edge_attr_dict[et].shape[1] if et in edge_attr_dict else 1
        for et in last.metadata()[1]
    ]
    log.info(f"in_channels_dict: {in_channels_dict}")
    log.info(f"edge_attr_dict: {edge_attr_dict}, \n edge_dim: {edge_dims}")

    model = InsiderThreatDetector(
        metadata=last.metadata(), in_channels_dict=in_channels_dict,
        hidden_channels=hidden_dim, num_heads=num_heads,
        temporal_layers=tsa_layers, dropout=dropout, 
        gnn_type=gnn_type, edge_dims=edge_dims )

    ckpt = torch.load(load_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    log.info(f"Loaded epoch={ckpt['epoch']} | val_f1={ckpt['val_f1']:.4f}")

    # 100% sessions vào test_mask
    N = last['session'].x.size(0)
    test_mask = torch.ones(N, dtype=torch.bool)
    labels    = last['session'].y
    criterion = FocalLoss(2.0, _pos_weight(labels))

    t0  = time.perf_counter()
    res = evaluate(model, snapshot_list, test_mask, 
                   criterion, edge_attr_dict = edge_attr_dict)
    inf_time = time.perf_counter() - t0

    inductive_label = ("INDUCTIVE — không cần retrain khi có session/user mới"
                       if gnn_type == "sage"
                       else "TRANSDUCTIVE — cần retrain khi graph thay đổi")
    _print_report(
        f"BẰNG CHỨNG QUY NẠP — {gnn_type.upper()} ({inductive_label})",
        test_mask, res)
    log.info(f"  Inference time: {inf_time:.4f}s | {inductive_label}")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE — train + test + inductive cả 3 model + vẽ biểu đồ
# ══════════════════════════════════════════════════════════════════════════════

def run_compare(snapshot_list, num_epochs=100, lr=0.005,
                weight_decay=1e-4, patience=15, hidden_dim=64,
                num_heads=4, tsa_layers=2, dropout=0.3):

    all_results = {}
    all_models  = {}

    for gnn_type in ["sage", "rgcn", "gat"]:
        res = run_phase1(gnn_type, snapshot_list, num_epochs=num_epochs,
                         hidden_dim=hidden_dim, lr=lr,
                         weight_decay=weight_decay, patience=patience,
                         num_heads=num_heads, tsa_layers=tsa_layers,
                         dropout=dropout)
        all_results[gnn_type] = res

        # Reload model cho inductive test
        last             = snapshot_list[-1]
        in_channels_dict = _build_in_channels(last)
            # Build edge_attr_dict từ snapshot
        edge_attr_dict = {
            ('user','opens','session'): last['user','opens','session'].edge_attr,
            ('user','executes','sql_template'): last['user','executes','sql_template'].edge_attr,
            ('sql_template','belongs_to','session'): last['sql_template','belongs_to','session'].edge_attr,
            ('sql_template','affects','table'): last['sql_template','affects','table'].edge_attr,
        }
        # Tạo list edge_dims theo thứ tự edge_types
        # edge_dims = [edge_attr_dict[et].shape[1] for et in edge_attr_dict]
        edge_dims = [
            edge_attr_dict[et].shape[1] if et in edge_attr_dict else 1
            for et in last.metadata()[1]
            ]
        log.info(f"in_channels_dict: {in_channels_dict}")
        log.info(f"edge_attr_dict: {edge_attr_dict}, \n edge_dim: {edge_dims}")
        m = InsiderThreatDetector(
            metadata=last.metadata(), in_channels_dict=in_channels_dict,
            hidden_channels=hidden_dim, gnn_type=gnn_type, edge_dims=edge_dims  )
        ckpt = torch.load(
            str(RESULTS_DIR / f"trained_{gnn_type}_detector_phase_1.pt"),
            weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        all_models[gnn_type] = m

    # ── Bảng so sánh ──────────────────────────────────────────────────────
    inductive_map = {'sage': 'YES ✓', 'rgcn': 'NO ✗', 'gat': 'NO ✗'}
    hdr = (f"{'Model':<8} {'F1':>6} {'AUC':>6} "
           f"{'Prec':>6} {'Rec':>6} {'Acc':>6} "
           f"{'Params':>9} {'Train(s)':>9} {'Inductive':>10}")
    sep = "─" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for gt, r in all_results.items():
        pfx = "**" if gt == "sage" else "  "
        print(f"{pfx}{gt.upper():<6} {r['f1']:>6.4f} {r['auc_roc']:>6.4f} "
              f"{r['precision']:>6.4f} {r['recall']:>6.4f} "
              f"{r['accuracy']:>6.4f} "
              f"{r['n_params']:>9,} {r['train_time_s']:>9.1f}s "
              f"{inductive_map[gt]:>10}")
    print(sep)

    # ── F1 bar + ROC ──────────────────────────────────────────────────────
    names  = [gt.upper() for gt in all_results]
    f1s    = [all_results[gt]['f1'] for gt in all_results]
    colors = ['#3a86ff', '#ff6b6b', '#ffd166']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars = axes[0].bar(names, f1s, color=colors, edgecolor='white', width=0.5)
    axes[0].bar_label(bars, fmt='%.4f', padding=3, fontsize=10)
    axes[0].set_ylim(0, 1.1); axes[0].set_ylabel('F1-Score')
    axes[0].set_title('So sánh F1: SAGE vs RGCN vs GAT')
    axes[0].grid(axis='y', alpha=0.3)

    for i, (gt, r) in enumerate(all_results.items()):
        if len(set(r['y_true'])) < 2: continue
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
        axes[1].plot(fpr, tpr, lw=2, color=colors[i],
                     label=f"{gt.upper()} (AUC={r['auc_roc']:.3f})")
    axes[1].plot([0,1],[0,1],'k--',lw=1)
    axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
    axes[1].set_title('ROC: SAGE vs RGCN vs GAT')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Inductive inference time ──────────────────────────────────────────
    log.info("\n── Inductive Inference Time ──")
    last      = snapshot_list[-1]
    N         = last['session'].x.size(0)
    inf_times = {}
    for gt, m in all_models.items():
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = m.forward_temporal(snapshot_list, edge_attr_dict=edge_attr_dict)
        inf_times[gt] = time.perf_counter() - t0
        log.info(f"  {gt.upper():<6}: {inf_times[gt]:.4f}s | "
                 f"{'INDUCTIVE ✓' if gt == 'sage' else 'TRANSDUCTIVE ✗'}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars2   = ax.bar([gt.upper() for gt in inf_times],
                     list(inf_times.values()),
                     color=colors, edgecolor='white', width=0.5)
    ax.bar_label(bars2, fmt='%.4fs', padding=3, fontsize=10)
    for i, (gt, t) in enumerate(inf_times.items()):
        ax.text(i, t + 0.0005,
                'Inductive ✓' if gt == 'sage' else 'Transductive ✗',
                ha='center', fontsize=8,
                color='#3a86ff' if gt == 'sage' else '#ff6b6b')
    ax.set_ylabel('Inference Time (s)')
    ax.set_title('Tính Quy Nạp: SAGE không cần retrain\nRGCN/GAT cần retrain khi graph thay đổi')
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(FIG_DIR / 'inductive_time.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"→ Figures: {FIG_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _print_report(title, mask, res):
    cm = confusion_matrix(res['y_true'],
                          (res['y_prob'] >= THRESHOLD).astype(int))
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")
    print(f"  Sessions   : {mask.sum().item():,}")
    print(f"  Accuracy   : {res['accuracy']*100:.2f}%")
    print(f"  Precision  : {res['precision']:.4f}")
    print(f"  Recall     : {res['recall']:.4f}")
    print(f"  F1-Score   : {res['f1']:.4f}  ← chỉ số cốt lõi bài báo")
    print(f"  AUC-ROC    : {res['auc_roc']:.4f}")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"{'═'*62}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",   type=int,   default=1, choices=[1, 2])
    # BUG-D3 FIX: bỏ comment --model argument
    parser.add_argument("--model",   type=str,   default="sage",
                        choices=["sage", "rgcn", "gat"],
                        help="sage=ATHITD (inductive), rgcn, gat (transductive)")
    parser.add_argument("--compare", action="store_true",
                        help="Train + test + inductive cả 3 model")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--hidden",  type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=0.001)
    parser.add_argument("--patience",type=int,   default=40)
    args = parser.parse_args()

    # Load snapshot
    try:
        snapshot_list = torch.load(HETERO_GRAPH_PATH, weights_only=False)
        # snapshot_list_test = torch.load(HETERO_TEST_GRAPH_PATH, weights_only=False)
        log.info(f"Loaded '{HETERO_GRAPH_PATH}': {len(snapshot_list)} snapshots")
    except FileNotFoundError:
        log.error(f"Không tìm thấy '{HETERO_GRAPH_PATH}'. Chạy a02 trước.")
        sys.exit(1)

    if args.compare:
        run_compare(snapshot_list, num_epochs=args.epochs,
                    lr=args.lr, patience=args.patience, hidden_dim=args.hidden)
    elif args.phase == 1:
        run_phase1(args.model, snapshot_list, num_epochs=args.epochs,
                   hidden_dim=args.hidden, lr=args.lr, patience=args.patience)
    elif args.phase == 2:
        run_phase2(args.model, snapshot_list, hidden_dim=args.hidden)
