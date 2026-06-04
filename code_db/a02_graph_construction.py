"""
a02_graph_construction.py — Bước 2: Xây dựng chuỗi Snapshot đồ thị
- Đọc các file csv sạch
- Chuyển đổi các định danh thành dạng số
- Đóng gói toàn bộ cấu trúc HeteroData của Torch Geometric
Node:
Nút User: [role_index, department_index, clearance_level, baseline_hour_1, baseline_hour_2]
Nút Table: [row_count, security_level]
Nút SQL_Template: [cmd_type, join_count, where_count, has_subquery]
Nút Session: [time_window_norm, is_internal_ip]

- Cắt audit log thành chuỗi danh sách Tên Snapshot con (Temporal Mode)
- Session features: dùng time_window_norm + is_internal_ip THỰC từ audit log (không random)
- Thêm đầy đủ edge features theo đặc tả PDF:
    * user → opens → session   : [is_after_hours, session_duration_norm]
    * user → executes → sql_t  : [execution_count_norm]
    * sql_t → belongs_to → sess: [sequence_order_norm]
    * sql_t → affects → table  : [rows_affected, permission_denied, success, execution_error]  (đã có)
    * ('table', 'relates_to', 'table'): ko có thuộc tính
- Xử lý tables_involved có thể chứa nhiều bảng (dấu phẩy)
- Thêm behavioral_baseline_hour vào user features
- Thêm mask train/val/test tại đây để file train dùng luôn
- CHUYỂN ĐỔI SOTA: Cắt audit log thành chuỗi danh sách Tên Snapshot con (Temporal Mode)
  để nạp nghiêm chỉnh vào module Transformer trục thời gian của file a03 và a04.
- Đầy đủ 4 node và đặc trưng node theo đúng đặc tả yêu cầu của bạn.
- Bổ sung toàn vẹn các edge features tương ứng với 4 loại cạnh động và 1 loại cạnh tĩnh PK-FK.
- Đồng bộ hóa bài toán gán mặt nạ phân chia Train/Val/Test lên Snapshot đích cuối cùng.
====================================================================
Sửa lỗi so với phiên bản upload:

    BUG-01/02/03/04 [NGHIÊM TRỌNG]
        Node features, edge building, session labels, session_agg
        đều dùng toàn bộ audit_df thay vì current_chunk.
        → Mọi snapshot giống nhau, temporal slicing vô nghĩa.
        FIX: tất cả tính toán trong vòng for t dùng current_chunk.

    BUG-05 [DESIGN]
        session_map xây từ toàn bộ audit_df → index toàn cục.
        Giữ nguyên (shared index) vì a03 cần session_emb[N_sess_global].
        Document rõ: mỗi snapshot có cùng N session nodes nhưng
        edge set và features khác nhau theo cửa sổ thời gian.

    BUG-07 [INDENT]
        torch.save + print thống kê nằm bên trong for t.
        FIX: đưa ra ngoài vòng lặp.

    BUG-08 [PERF]
        iterrows() O(T×N). FIX: group-by trước, dùng vectorized ops.
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split

from constant import (
    TABLE_METADATA_PATH,
    TABLE_RELATIONS_PATH,
    CLEAN_AUDIT_LOG_PATH,
    CLEAN_USER_METADATA_PATH,
    ANOMALY_GROUND_TRUTH_USER_PATH,
)

HETERO_GRAPH_PATH = "hetero_graph.pt"
HETERO_TEST_GRAPH_PATH = "hetero_test_graph.pt"


def _safe_norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE: group-by để build edges nhanh (BUG-08)
# ══════════════════════════════════════════════════════════════════════════════

def _precompute_edge_groups(chunk: pd.DataFrame,
                            user_map: dict, session_map: dict,
                            sql_map: dict, table_map: dict
                            ) -> dict:
    """
    Trả về dict chứa tất cả dữ liệu cần thiết để xây edges cho 1 chunk.
    Thực hiện 1 lần duyệt chunk thay vì iterrows() lặp lại.
    """
    rows = []
    for _, row in chunk.iterrows():
        u = user_map.get(row['db_user'])
        s = session_map.get(row['session_id'])
        qt = (row['cmd_type'], row['join_count'],
              row['where_count'], row['has_subquery'])
        q = sql_map.get(qt)
        if u is None or s is None or q is None:
            continue

        tables_raw = str(row.get('tables_involved', '')).strip().lower()
        involved   = [t.strip() for t in tables_raw.split(',') if t.strip()] \
                     or [tables_raw]

        rows.append(dict(
            u=u, s=s, q=q,
            is_after_hours=float(row.get('is_after_hours', 0)),
            session_duration_norm=float(row.get('session_duration_norm')),#
            execution_count_norm=float(row.get('execution_count_norm')),
            sequence_order_norm=float(row.get('sequence_order_norm')),
            rows_log=float(np.log1p(max(row.get('rows_affected', 0), 0))),
            permission_denied=float(row.get('permission_denied', 0)),
            success=float(row.get('success', 0)),
            execution_error=float(row.get('execution_error', 0)),
            involved_tables=involved,
        ))
    return rows


def _build_edges_from_rows(rows: list, table_map: dict):
    opens_src, opens_dst, opens_attr      = [], [], []
    exec_src,  exec_dst,  exec_attr       = [], [], []
    bel_src,   bel_dst,   bel_attr        = [], [], []
    aff_src,   aff_dst,   aff_attr        = [], [], []

    opened_pairs   = set()
    executed_pairs = set()

    for r in rows:
        u, s, q = r['u'], r['s'], r['q']

        # user → opens → session  (1 lần per (u,s))
        if (u, s) not in opened_pairs:
            opens_src.append(u); opens_dst.append(s)
            opens_attr.append([r['is_after_hours'], r['session_duration_norm']])
            opened_pairs.add((u, s))

        # user → executes → sql_template  (1 lần per (u,q))
        if (u, q) not in executed_pairs:
            exec_src.append(u); exec_dst.append(q)
            exec_attr.append([r['execution_count_norm']])
            executed_pairs.add((u, q))

        # sql_template → belongs_to → session  (mỗi event = 1 cạnh)
        bel_src.append(q); bel_dst.append(s)
        bel_attr.append([r['sequence_order_norm']])

        # sql_template → affects → table
        for tbl in r['involved_tables']:
            t_idx = table_map.get(tbl)
            if t_idx is not None:
                aff_src.append(q); aff_dst.append(t_idx)
                aff_attr.append([r['rows_log'], r['permission_denied'],
                                 r['success'],  r['execution_error']])

    def _ei(src, dst):
        if not src:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    def _ea(attr):
        if not attr:
            return torch.empty((0, len(attr[0]) if attr else 1), dtype=torch.float)
        return torch.tensor(attr, dtype=torch.float)

    return {
        'opens':      (_ei(opens_src, opens_dst), _ea(opens_attr)),
        'executes':   (_ei(exec_src,  exec_dst),  _ea(exec_attr)),
        'belongs_to': (_ei(bel_src,   bel_dst),   _ea(bel_attr)),
        'affects':    (_ei(aff_src,   aff_dst),   _ea(aff_attr)),
    }
# chia snapshot theo timewwindows + 5 ngày
def split_by_days(audit_df, days_per_snapshot=5):
    audit_df = audit_df.copy()
    audit_df['date'] = audit_df['timestamp'].dt.normalize()
    start_date = audit_df['date'].min()

    audit_df['snapshot_id'] = (
        (audit_df['date'] - start_date).dt.days // days_per_snapshot
    )

    chunks = []
    for sid, chunk in audit_df.groupby('snapshot_id'):
        chunks.append(chunk.sort_values('timestamp').copy())

    return chunks

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_heterogeneous_graph(days_per_snapshot: int = 5) -> list:
    print("=== BƯỚC 2.1: Tải dữ liệu sạch ===")
    audit_df    = pd.read_csv(CLEAN_AUDIT_LOG_PATH)
    user_df     = pd.read_csv(CLEAN_USER_METADATA_PATH)
    table_df    = pd.read_csv(TABLE_METADATA_PATH)
    relation_df = pd.read_csv(TABLE_RELATIONS_PATH)
    user_gt_df  = pd.read_csv(ANOMALY_GROUND_TRUTH_USER_PATH) # đọc thêm nhãn user

    audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
    audit_df = audit_df.sort_values('timestamp').reset_index(drop=True)

    # ── Index maps (toàn cục — shared across snapshots) ───────────────────────
    user_map  = {n: i for i, n in enumerate(user_df['db_user'].unique())}
    table_df['table_name'] = table_df['table_name'].str.strip().str.lower()
    table_map = {n: i for i, n in enumerate(table_df['table_name'].unique())}
    table_df['row_count'] = np.log1p (table_df["row_count"]) # đưa về phân bố 

    sql_cols      = ['cmd_type', 'join_count', 'where_count', 'has_subquery']
    sql_templates = audit_df[sql_cols].drop_duplicates().reset_index(drop=True)
    sql_map       = {tuple(r): i for i, r in sql_templates.iterrows()}

    # BUG-05 DESIGN: session index toàn cục — mỗi snapshot có CÙNG N session nodes,
    # features + edges phản ánh đúng cửa sổ thời gian của chunk đó.
    session_ids = audit_df['session_id'].unique()
    session_map = {sid: i for i, sid in enumerate(session_ids)}
    N_sess      = len(session_map)
    N_users     = len(user_map)
    N_sql       = len(sql_templates)
    N_tables    = len(table_map)

    # ── Pre-normalize liên tục trên toàn bộ (scale nhất quán) ────────────────
    audit_df['session_duration_norm'] = _safe_norm(audit_df['session_duration'].fillna(0))

    audit_df['execution_count_norm'] = _safe_norm(audit_df['execution_count'].fillna(1))

    audit_df['sequence_order_norm'] = audit_df.groupby('session_id')['sequence_order'] \
                                            .transform(_safe_norm)

    # ── Static node features (tính 1 lần — không đổi theo snapshot) ──────────
    feat_cols_user = ['role_index', 'department_index', 'clearance_level', "baseline_hour_1", "baseline_hour_2"]
    user_df_sorted = (pd.DataFrame({'db_user': list(user_map.keys())})
                      .merge(user_df, on='db_user', how='left'))
    user_feat_tensor = torch.tensor(
        user_df_sorted[feat_cols_user].values.astype(np.float32), dtype=torch.float)

    table_df_sorted = (pd.DataFrame({'table_name': list(table_map.keys())})
                       .merge(table_df, on='table_name', how='left'))
    table_feat_tensor = torch.tensor(
        table_df_sorted[['row_count', 'security_level']].values.astype(np.float32),
        dtype=torch.float)

    sql_feat_tensor = torch.tensor(
        sql_templates.values.astype(np.float32), dtype=torch.float)

    # ── Static PK-FK edges (xây 1 lần) ───────────────────────────────────────
    relation_df.columns = relation_df.columns.str.strip()
    table_rel_edges = []
    if 'source_table' in relation_df.columns and 'target_table' in relation_df.columns:
        for _, r in relation_df.iterrows():
            s = str(r['source_table']).strip().lower()
            t = str(r['target_table']).strip().lower()
            if s in table_map and t in table_map:
                table_rel_edges.append([table_map[s], table_map[t]])
    pk_fk_ei = (torch.tensor(table_rel_edges, dtype=torch.long).t().contiguous()
                if table_rel_edges else torch.empty((2, 0), dtype=torch.long))
    print(f"-> PK-FK edges: {pk_fk_ei.shape[1]}")

    # # ── Global session labels (anomaly nếu BẤT KỲ event trong session là anomaly)
    # KHÔNG DÙNG SESSION-LEVEL LABEL NỮA CHUYỂN SANG USER-LEVEL LABLE

    # ── Chia audit_df thành num_snapshots chunk theo thời gian ────────────────
    # chunks = np.array_split(audit_df, num_snapshots)
    chunks = split_by_days(audit_df, days_per_snapshot=days_per_snapshot)
    print(f"-> Phân rã {len(chunks)} snapshots "
      f"({days_per_snapshot} ngày/snapshot): "
      f"{[len(c) for c in chunks]} events/chunk")

    snapshot_chain = []

    for t, current_chunk in enumerate(chunks):
        # BUG-01/02/03/04 FIX: TẤT CẢ tính toán dùng current_chunk
        data = HeteroData()

        # ── Node features: User, Table, SQL (tĩnh — dùng global tensor) ──────
        data['user'].x         = user_feat_tensor          # [N_users, F_u]
        data['table'].x        = table_feat_tensor         # [N_tables, F_t]
        data['sql_template'].x = sql_feat_tensor           # [N_sql, 4]

        # ── Node features: Session (FIX BUG-04: từ current_chunk) ────────────
        sess_agg = (
    current_chunk.groupby('session_id')
        .agg(
            time_window_norm=('time_window_norm', 'first'),
            is_internal_ip=('is_internal_ip', 'min'),

            # thống kê session
            num_events=('session_id', 'size'),
            num_sql_templates=('cmd_type', 'nunique'),
            num_tables=('tables_involved', 'nunique'),

            # hành vi lỗi
            permission_denied_rate=('permission_denied', 'mean'),
            execution_error_rate=('execution_error', 'mean'),
            success_rate=('success', 'mean'),

            # thao tác SQL
            avg_join_count=('join_count', 'mean'),
            avg_where_count=('where_count', 'mean'),
            has_subquery_rate=('has_subquery', 'mean'),

            # dữ liệu tác động
            total_rows_affected=('rows_affected', 'sum'),
            max_rows_affected=('rows_affected', 'max'),

            # thời lượng
            session_duration=('session_duration', 'max'),

            # ngoài giờ
            after_hours_rate=('is_after_hours', 'mean')
        )
        .reset_index()
    )
        log_cols = ['num_events','num_sql_templates','num_tables','total_rows_affected','max_rows_affected','session_duration']
        for col in log_cols:
            sess_agg[col] = np.log1p(sess_agg[col])
        extra_cols = [
            'num_events','num_sql_templates','num_tables',
            'permission_denied_rate','execution_error_rate','success_rate',
            'avg_join_count','avg_where_count','has_subquery_rate',
            'total_rows_affected','max_rows_affected','session_duration','after_hours_rate'
        ]

        for col in extra_cols:
            sess_agg[col] = _safe_norm(sess_agg[col])
            
        sess_df = (pd.DataFrame({'session_id': list(session_map.keys())})
                   .merge(sess_agg, on='session_id', how='left')
                   .fillna(0))
        session_feature_cols = [
            'time_window_norm','is_internal_ip',
            'num_events','num_sql_templates','num_tables',
            'permission_denied_rate','execution_error_rate','success_rate',
            'avg_join_count','avg_where_count','has_subquery_rate',
            'total_rows_affected','max_rows_affected','session_duration','after_hours_rate'
        ]



        # ── Session labels: ko dùng
        # Gán user-level lable theo thời gian bằng max của cột is_anomaly
        snapshot_start = current_chunk['timestamp'].min().date()
        snapshot_end   = current_chunk['timestamp'].max().date()

        gt_chunk = user_gt_df[
            (user_gt_df['date'] >= str(snapshot_start)) &
            (user_gt_df['date'] <= str(snapshot_end))
        ]
        user_labels = np.zeros(N_users)

        for db_user, g in gt_chunk.groupby('db_user'):
            u_idx = user_map.get(db_user)
            if u_idx is not None:
                user_labels[u_idx] = g['is_anomaly'].max()
        
        data['session'].x = torch.tensor(
            sess_df[session_feature_cols].values.astype(np.float32),
            dtype=torch.float
        )
        
        data['user'].y = torch.tensor(user_labels, dtype=torch.float)

        # ── Edges (FIX BUG-02: từ current_chunk) ─────────────────────────────
        rows = _precompute_edge_groups(
            current_chunk, user_map, session_map, sql_map, table_map)
        edges = _build_edges_from_rows(rows, table_map)

        data['user', 'opens',    'session'].edge_index    = edges['opens'][0]
        data['user', 'opens',    'session'].edge_attr     = edges['opens'][1]
        data['user', 'executes', 'sql_template'].edge_index = edges['executes'][0]
        data['user', 'executes', 'sql_template'].edge_attr  = edges['executes'][1]
        data['sql_template', 'belongs_to', 'session'].edge_index = edges['belongs_to'][0]
        data['sql_template', 'belongs_to', 'session'].edge_attr  = edges['belongs_to'][1]
        data['sql_template', 'affects', 'table'].edge_index      = edges['affects'][0]
        data['sql_template', 'affects', 'table'].edge_attr       = edges['affects'][1]
        data['table', 'relates_to', 'table'].edge_index          = pk_fk_ei

        # ── Train/Val/Test mask: CHỈ snapshot cuối (target graph T) ──────────
        # ── Train/Val/Test mask: target là USER, không còn là SESSION ──────────
        # Context snapshots: không train/eval trực tiếp
        data['session'].train_mask = torch.zeros(N_sess, dtype=torch.bool)
        data['session'].val_mask   = torch.zeros(N_sess, dtype=torch.bool)
        data['session'].test_mask  = torch.zeros(N_sess, dtype=torch.bool)

        if t == len(chunks) - 1:
            idx = np.arange(N_users)
            y = data['user'].y.cpu().numpy()

            # Nếu snapshot cuối có cả 0 và 1 thì stratify.
            # Nếu không có anomaly ở snapshot cuối thì dùng random split để tránh lỗi.
            if len(np.unique(y)) == 2 and y.sum() >= 2:
                tr_idx, te_idx = train_test_split(
                    idx, test_size=0.4, stratify=y, random_state=42
                )

                if len(np.unique(y[te_idx])) == 2 and y[te_idx].sum() >= 1:
                    val_idx, te_idx = train_test_split(
                        te_idx, test_size=0.5, stratify=y[te_idx], random_state=42
                    )
                else:
                    val_idx, te_idx = train_test_split(
                        te_idx, test_size=0.5, random_state=42
                    )
            else:
                tr_idx, te_idx = train_test_split(
                    idx, test_size=0.4, random_state=42
                )
                val_idx, te_idx = train_test_split(
                    te_idx, test_size=0.5, random_state=42
                )

            tm = torch.zeros(N_users, dtype=torch.bool); tm[tr_idx]  = True
            vm = torch.zeros(N_users, dtype=torch.bool); vm[val_idx] = True
            xm = torch.zeros(N_users, dtype=torch.bool); xm[te_idx]  = True

            data['user'].train_mask = tm
            data['user'].val_mask   = vm
            data['user'].test_mask  = xm

            print(f"  [T={t}] ★ Target snapshot | "
                f"events={len(current_chunk):,} | "
                f"user Train/Val/Test={tm.sum()}/{vm.sum()}/{xm.sum()} | "
                f"anom_users={int(data['user'].y.sum())}")
        else:
            data['user'].train_mask = torch.zeros(N_users, dtype=torch.bool)
            data['user'].val_mask   = torch.zeros(N_users, dtype=torch.bool)
            data['user'].test_mask  = torch.zeros(N_users, dtype=torch.bool)

            n_edges = edges['opens'][0].shape[1]
            print(f"  [T={t}] Context snapshot | "
                f"events={len(current_chunk):,} | "
                f"user→session edges={n_edges} | "
                f"anom_users={int(data['user'].y.sum())}")
        snapshot_chain.append(data)

    num_events_per_snapshot = [len(c) for c in chunks]
    average_events_per_snapshot = np.mean(num_events_per_snapshot)

    # lưu + in thống kê RA NGOÀI vòng lặp
    total_user_time_anom = sum(int(s['user'].y.sum()) for s in snapshot_chain)
    print(f"\n--- Thống kê chuỗi ---")
    print(f"  Tổng snapshots          : {len(snapshot_chain)}")
    print(f"  User nodes              : {N_users}")
    print(f"  Session nodes           : {N_sess}")
    print(f"  User-time anomalies     : {total_user_time_anom}")
    torch.save(snapshot_chain, HETERO_GRAPH_PATH)
    print(f"  → Đã lưu '{HETERO_GRAPH_PATH}'")
    print("=== BƯỚC 2: Hoàn tất! ===")
    return snapshot_chain, average_events_per_snapshot


def build_heterogeneous_graph_test(days_per_snapshot: int = 5) -> list:
    print("=== BƯỚC 2.1: Tải dữ liệu sạch ===")
    audit_df    = pd.read_csv(CLEAN_AUDIT_LOG_PATH)
    user_df     = pd.read_csv(CLEAN_USER_METADATA_PATH)
    table_df    = pd.read_csv(TABLE_METADATA_PATH)
    relation_df = pd.read_csv(TABLE_RELATIONS_PATH)
    user_gt_df  = pd.read_csv(ANOMALY_GROUND_TRUTH_USER_PATH) # đọc thêm nhãn user

    audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
    audit_df = audit_df.sort_values('timestamp').reset_index(drop=True)

    # ── Index maps (toàn cục — shared across snapshots) ───────────────────────
    user_map  = {n: i for i, n in enumerate(user_df['db_user'].unique())}
    table_df['table_name'] = table_df['table_name'].str.strip().str.lower()
    table_map = {n: i for i, n in enumerate(table_df['table_name'].unique())}
    table_df['row_count'] = np.log1p (table_df["row_count"]) # đưa về phân bố 

    sql_cols      = ['cmd_type', 'join_count', 'where_count', 'has_subquery']
    sql_templates = audit_df[sql_cols].drop_duplicates().reset_index(drop=True)
    sql_map       = {tuple(r): i for i, r in sql_templates.iterrows()}

    # BUG-05 DESIGN: session index toàn cục — mỗi snapshot có CÙNG N session nodes,
    # features + edges phản ánh đúng cửa sổ thời gian của chunk đó.
    session_ids = audit_df['session_id'].unique()
    session_map = {sid: i for i, sid in enumerate(session_ids)}
    N_sess      = len(session_map)
    N_users     = len(user_map)
    N_sql       = len(sql_templates)
    N_tables    = len(table_map)

    # ── Pre-normalize liên tục trên toàn bộ (scale nhất quán) ────────────────
    audit_df['session_duration_norm'] = _safe_norm(audit_df['session_duration'].fillna(0))

    audit_df['execution_count_norm'] = _safe_norm(audit_df['execution_count'].fillna(1))

    audit_df['sequence_order_norm'] = audit_df.groupby('session_id')['sequence_order'] \
                                            .transform(_safe_norm)

    # ── Static node features (tính 1 lần — không đổi theo snapshot) ──────────
    feat_cols_user = ['role_index', 'department_index', 'clearance_level', "baseline_hour_1", "baseline_hour_2"]
    user_df_sorted = (pd.DataFrame({'db_user': list(user_map.keys())})
                      .merge(user_df, on='db_user', how='left'))
    user_feat_tensor = torch.tensor(
        user_df_sorted[feat_cols_user].values.astype(np.float32), dtype=torch.float)

    table_df_sorted = (pd.DataFrame({'table_name': list(table_map.keys())})
                       .merge(table_df, on='table_name', how='left'))
    table_feat_tensor = torch.tensor(
        table_df_sorted[['row_count', 'security_level']].values.astype(np.float32),
        dtype=torch.float)

    sql_feat_tensor = torch.tensor(
        sql_templates.values.astype(np.float32), dtype=torch.float)

    # ── Static PK-FK edges (xây 1 lần) ───────────────────────────────────────
    relation_df.columns = relation_df.columns.str.strip()
    table_rel_edges = []
    if 'source_table' in relation_df.columns and 'target_table' in relation_df.columns:
        for _, r in relation_df.iterrows():
            s = str(r['source_table']).strip().lower()
            t = str(r['target_table']).strip().lower()
            if s in table_map and t in table_map:
                table_rel_edges.append([table_map[s], table_map[t]])
    pk_fk_ei = (torch.tensor(table_rel_edges, dtype=torch.long).t().contiguous()
                if table_rel_edges else torch.empty((2, 0), dtype=torch.long))
    print(f"-> PK-FK edges: {pk_fk_ei.shape[1]}")

    # # ── Global session labels (anomaly nếu BẤT KỲ event trong session là anomaly)
    # KHÔNG DÙNG SESSION-LEVEL LABEL NỮA CHUYỂN SANG USER-LEVEL LABLE
    # # # Dùng để gán nhãn cho SNAPSHOT CUỐI (target graph)
    # global_labels = np.zeros(N_sess, dtype=np.float32)
    # for _, row in audit_df[audit_df['is_anomaly'] == 1].iterrows():
    #     s_idx = session_map.get(row['session_id'])
    #     if s_idx is not None:
    #         global_labels[s_idx] = 1.0

    # ── Chia audit_df thành num_snapshots chunk theo thời gian ────────────────
    # chunks = np.array_split(audit_df, num_snapshots)
    chunks = split_by_days(audit_df, days_per_snapshot=days_per_snapshot)
    print(f"-> Phân rã {len(chunks)} snapshots "
      f"({days_per_snapshot} ngày/snapshot): "
      f"{[len(c) for c in chunks]} events/chunk")

    snapshot_chain = []

    for t, current_chunk in enumerate(chunks):
        # BUG-01/02/03/04 FIX: TẤT CẢ tính toán dùng current_chunk
        data = HeteroData()

        # ── Node features: User, Table, SQL (tĩnh — dùng global tensor) ──────
        data['user'].x         = user_feat_tensor          # [N_users, F_u]
        data['table'].x        = table_feat_tensor         # [N_tables, F_t]
        data['sql_template'].x = sql_feat_tensor           # [N_sql, 4]

        # ── Node features: Session (FIX BUG-04: từ current_chunk) ────────────
        sess_agg = (
    current_chunk.groupby('session_id')
        .agg(
            time_window_norm=('time_window_norm', 'first'),
            is_internal_ip=('is_internal_ip', 'min'),

            # thống kê session
            num_events=('session_id', 'size'),
            num_sql_templates=('cmd_type', 'nunique'),
            num_tables=('tables_involved', 'nunique'),

            # hành vi lỗi
            permission_denied_rate=('permission_denied', 'mean'),
            execution_error_rate=('execution_error', 'mean'),
            success_rate=('success', 'mean'),

            # thao tác SQL
            avg_join_count=('join_count', 'mean'),
            avg_where_count=('where_count', 'mean'),
            has_subquery_rate=('has_subquery', 'mean'),

            # dữ liệu tác động
            total_rows_affected=('rows_affected', 'sum'),
            max_rows_affected=('rows_affected', 'max'),

            # thời lượng
            session_duration=('session_duration', 'max'),

            # ngoài giờ
            after_hours_rate=('is_after_hours', 'mean')
        )
        .reset_index()
    )
        log_cols = ['num_events','num_sql_templates','num_tables','total_rows_affected','max_rows_affected','session_duration']
        for col in log_cols:
            sess_agg[col] = np.log1p(sess_agg[col])
        extra_cols = [
            'num_events','num_sql_templates','num_tables',
            'permission_denied_rate','execution_error_rate','success_rate',
            'avg_join_count','avg_where_count','has_subquery_rate',
            'total_rows_affected','max_rows_affected','session_duration','after_hours_rate'
        ]

        for col in extra_cols:
            sess_agg[col] = _safe_norm(sess_agg[col])
            
        sess_df = (pd.DataFrame({'session_id': list(session_map.keys())})
                   .merge(sess_agg, on='session_id', how='left')
                   .fillna(0))
        session_feature_cols = [
            'time_window_norm','is_internal_ip',
            'num_events','num_sql_templates','num_tables',
            'permission_denied_rate','execution_error_rate','success_rate',
            'avg_join_count','avg_where_count','has_subquery_rate',
            'total_rows_affected','max_rows_affected','session_duration','after_hours_rate'
        ]



        # ── Session labels: ko dùng
        # Gán user-level lable theo thời gian bằng max của cột is_anomaly
        snapshot_start = current_chunk['timestamp'].min().date()
        snapshot_end   = current_chunk['timestamp'].max().date()

        gt_chunk = user_gt_df[
            (user_gt_df['date'] >= str(snapshot_start)) &
            (user_gt_df['date'] <= str(snapshot_end))
        ]
        user_labels = np.zeros(N_users)

        for db_user, g in gt_chunk.groupby('db_user'):
            u_idx = user_map.get(db_user)
            if u_idx is not None:
                user_labels[u_idx] = g['is_anomaly'].max()
        
        data['session'].x = torch.tensor(
            sess_df[session_feature_cols].values.astype(np.float32),
            dtype=torch.float
        )
        
        data['user'].y = torch.tensor(user_labels, dtype=torch.float)

        # ── Edges (FIX BUG-02: từ current_chunk) ─────────────────────────────
        rows = _precompute_edge_groups(
            current_chunk, user_map, session_map, sql_map, table_map)
        edges = _build_edges_from_rows(rows, table_map)

        data['user', 'opens',    'session'].edge_index    = edges['opens'][0]
        data['user', 'opens',    'session'].edge_attr     = edges['opens'][1]
        data['user', 'executes', 'sql_template'].edge_index = edges['executes'][0]
        data['user', 'executes', 'sql_template'].edge_attr  = edges['executes'][1]
        data['sql_template', 'belongs_to', 'session'].edge_index = edges['belongs_to'][0]
        data['sql_template', 'belongs_to', 'session'].edge_attr  = edges['belongs_to'][1]
        data['sql_template', 'affects', 'table'].edge_index      = edges['affects'][0]
        data['sql_template', 'affects', 'table'].edge_attr       = edges['affects'][1]
        data['table', 'relates_to', 'table'].edge_index          = pk_fk_ei

        # ── Train/Val/Test mask: CHỈ snapshot cuối (target graph T) ──────────
        # ── Train/Val/Test mask: target là USER, không còn là SESSION ──────────
        # Context snapshots: không train/eval trực tiếp
        data['session'].train_mask = torch.zeros(N_sess, dtype=torch.bool)
        data['session'].val_mask   = torch.zeros(N_sess, dtype=torch.bool)
        data['session'].test_mask  = torch.zeros(N_sess, dtype=torch.bool)
        # Test graph: tất cả user đều được evaluate
        data['user'].train_mask = torch.zeros(N_users, dtype=torch.bool)
        data['user'].val_mask   = torch.zeros(N_users, dtype=torch.bool)
        data['user'].test_mask  = torch.ones(N_users, dtype=torch.bool)

        print(
            f"  [T={t}] Test snapshot | "
            f"events={len(current_chunk):,} | "
            f"users={N_users} | "
            f"anom_users={int(data['user'].y.sum())}"
        )
        snapshot_chain.append(data)

    num_events_per_snapshot = [len(c) for c in chunks]
    average_events_per_snapshot = np.mean(num_events_per_snapshot)

    # lưu + in thống kê RA NGOÀI vòng lặp
    total_user_time_anom = sum(int(s['user'].y.sum()) for s in snapshot_chain)
    print(f"\n--- Thống kê chuỗi ---")
    print(f"  Tổng snapshots          : {len(snapshot_chain)}")
    print(f"  User nodes              : {N_users}")
    print(f"  Session nodes           : {N_sess}")
    print(f"  User-time anomalies     : {total_user_time_anom}")
    torch.save(snapshot_chain, HETERO_TEST_GRAPH_PATH)
    print(f"  → Đã lưu '{HETERO_TEST_GRAPH_PATH}'")
    print("=== BƯỚC 2: Hoàn tất! ===")
    return snapshot_chain, average_events_per_snapshot

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # a, average_events_per_snapshot = build_heterogeneous_graph(num_snapshots=5)
    # print("average_events_per_snapshot: ", average_events_per_snapshot)

    # lấy số event trung bình của từng snapshot rồi đưa vào thay thế 200
    a, average_events_per_snapshot = build_heterogeneous_graph(days_per_snapshot=5)






