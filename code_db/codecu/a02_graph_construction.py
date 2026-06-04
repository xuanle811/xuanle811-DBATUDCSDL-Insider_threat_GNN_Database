"""
File:
- Đọc các file csv sạch
- Chuyển đổi các định danh thành dạng số
- Đóng gói toàn bộ cấu trúc HeteroData của Torch Geometric
Node:
Nút User: [role_index, department_index, clearance_level, behavioral_baseline_hour]
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
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData


from constant import (
    TABLE_METADATA_PATH,
    TABLE_RELATIONS_PATH,
    CLEAN_AUDIT_LOG_PATH,
    CLEAN_USER_METADATA_PATH,
)


def _safe_norm(series: pd.Series) -> pd.Series:
    """Min-max normalize; trả về 0 nếu series là hằng số."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def build_heterogeneous_graph(num_snapshots: int = 5):
    # ================================================================
    # ĐỌC DỮ LIỆU SẠCH
    # ================================================================
    print("=== BƯỚC 2.1: Tải và ánh xạ dữ liệu kiểm toán ===")
    audit_df   = pd.read_csv(CLEAN_AUDIT_LOG_PATH)
    user_df    = pd.read_csv(CLEAN_USER_METADATA_PATH)
    table_df   = pd.read_csv(TABLE_METADATA_PATH)
    relation_df = pd.read_csv(TABLE_RELATIONS_PATH)

    # Chuẩn hóa kiểu dữ liệu
    audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
    audit_df = audit_df.sort_values('timestamp').reset_index(drop=True)

    # data = HeteroData()

    # ================================================================
    # 1. TẠO INDEX MAP CHO CÁC NÚT
    # ================================================================
    user_map = {name: i for i, name in enumerate(user_df['db_user'].unique())}

 # Chuẩn hóa triệt để tên bảng về chữ thường và loại bỏ dấu cách thừa
    table_df['table_name'] = table_df['table_name'].astype(str).str.strip().str.lower()
    table_map = {name: i for i, name in enumerate(table_df['table_name'].unique())}

    # SQL_Template: nhận diện duy nhất qua (cmd_type, join_count, where_count, has_subquery)
    sql_template_cols = ['cmd_type', 'join_count', 'where_count', 'has_subquery']
    sql_templates = (
        audit_df[sql_template_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    sql_map = {tuple(row): i for i, row in sql_templates.iterrows()}

    session_ids  = audit_df['session_id'].unique()
    session_map  = {sid: i for i, sid in enumerate(session_ids)}
    # Pre-normalize các thuộc tính liên tục trên toàn cục để giữ đúng phân phối dữ liệu
    audit_df['session_duration_norm'] = _safe_norm(audit_df['session_duration'].fillna(0)) if 'session_duration' in audit_df.columns else 0.0
    audit_df['execution_count_norm']  = _safe_norm(audit_df['execution_count'].fillna(1)) if 'execution_count' in audit_df.columns else 0.0
    audit_df['sequence_order_norm']   = audit_df.groupby('session_id')['sequence_order'].transform(lambda s: _safe_norm(s)) if 'sequence_order' in audit_df.columns else 0.0

    # Chia nhỏ dữ liệu audit log thành các phân đoạn thời gian (Chunks) bằng nhau
    chunks = np.array_split(audit_df, num_snapshots)
    snapshot_chain = []

    print(f"-> Khởi tạo tiến trình phân rã không gian - thời gian: {num_snapshots} Snapshots.")

    # Duyệt qua từng mốc thời gian t để xây dựng đồ thị con con độc lập G_t
    for t in range(num_snapshots):
        current_chunk = chunks[t]
        data = HeteroData()

    # ================================================================
    # 2. KHỞI TẠO ĐẶC TRƯNG NÚT (NODE FEATURES) CHO TỪNG SNAPSHOT
    # ================================================================

        # --- Nút User: [role_index, department_index, clearance_level, behavioral_baseline_hour]
        # Đặc trưng tĩnh, không chứa định danh db_user (để đảm bảo tính quy nạp)
        feat_cols_user = ['role_index', 'department_index', 'clearance_level']
        if 'behavioral_baseline_hour' in user_df.columns:
            feat_cols_user.append('behavioral_baseline_hour')
        # Sắp xếp user_df đúng thứ tự theo user_map
        user_df_sorted = (
            pd.DataFrame({'db_user': list(user_map.keys())})
            .merge(user_df, on='db_user', how='left')
        )
        user_feat = user_df_sorted[feat_cols_user].values.astype(np.float32)
        data['user'].x = torch.tensor(user_feat, dtype=torch.float)

        # --- Nút Table: [row_count, security_level]
        # Sắp xếp đúng thứ tự theo table_map
        table_df_sorted = (
            pd.DataFrame({'table_name': list(table_map.keys())})
            .merge(table_df, on='table_name', how='left')
        )
        table_feat = table_df_sorted[['row_count', 'security_level']].values.astype(np.float32)
        data['table'].x = torch.tensor(table_feat, dtype=torch.float)

        # --- Nút SQL_Template: [cmd_type, join_count, where_count, has_subquery]
        data['sql_template'].x = torch.tensor(
            sql_templates.values.astype(np.float32), dtype=torch.float
        )

        # --- Nút Session: [time_window_norm, is_internal_ip] — dùng giá trị THỰC từ audit log
        session_agg = (
            audit_df.groupby('session_id')
            .agg(
                time_window_norm=('time_window_norm', 'first'),
                is_internal_ip  =('is_internal_ip',   'min'),
            )
            .reset_index()
        )
        session_df_sorted = (
            pd.DataFrame({'session_id': list(session_map.keys())})
            .merge(session_agg, on='session_id', how='left')
            .fillna(0)
        )
        session_feat = session_df_sorted[['time_window_norm', 'is_internal_ip']].values.astype(np.float32)
        data['session'].x = torch.tensor(session_feat, dtype=torch.float)

        # ================================================================
        # 3. XÂY DỰNG CÁC CẠNH VÀ THUỘC TÍNH CẠNH
        # ================================================================
        user_opens_session_edges  = []   # [u_idx, s_idx]
        user_executes_sql_edges   = []   # [u_idx, q_idx]
        sql_belongs_session_edges = []   # [q_idx, s_idx]
        sql_affects_table_edges   = []   # [q_idx, t_idx]

        # Edge attributes
        opens_attr    = []  # [is_after_hours, session_duration_norm]
        executes_attr = []  # [execution_count_norm]
        belongs_attr  = []  # [sequence_order_norm]
        affects_attr  = []  # [rows_affected, permission_denied]



        # Tập hợp các (u_idx, s_idx) đã thêm để tránh lặp cạnh mở session
        opened_pairs = set()
        # Tập hợp các (u_idx, q_idx) đã thêm để tránh lặp cạnh executes
        executed_pairs = set()
        
    
        for _, row in audit_df.iterrows():
            if row['db_user'] not in user_map:
                continue
            u_idx = user_map[row['db_user']]
            s_idx = session_map[row['session_id']]
            sql_tuple = (row['cmd_type'], row['join_count'], row['where_count'], row['has_subquery'])
            if sql_tuple not in sql_map:
                continue
            q_idx = sql_map[sql_tuple]

            # --- Cạnh user → opens → session (mỗi cặp (u,s) 1 lần)
            if (u_idx, s_idx) not in opened_pairs:
                user_opens_session_edges.append([u_idx, s_idx])
                opens_attr.append([
                    float(row.get('is_after_hours', 0)),
                    float(row.get('session_duration_norm', 0)),
                ])
                opened_pairs.add((u_idx, s_idx))

            # --- Cạnh user → executes → sql_template (mỗi cặp (u,q) 1 lần)
            if (u_idx, q_idx) not in executed_pairs:
                user_executes_sql_edges.append([u_idx, q_idx])
                executes_attr.append([float(row.get('execution_count_norm', 0))])
                executed_pairs.add((u_idx, q_idx))

            # --- Cạnh sql_template → belongs_to → session (mỗi hàng audit log = 1 sự kiện)
            sql_belongs_session_edges.append([q_idx, s_idx])
            belongs_attr.append([float(row.get('sequence_order_norm', 0))])

            # --- Cạnh sql_template → affects → table
            # tables_involved có thể chứa nhiều bảng phân cách bởi dấu phẩy
            tables_raw = str(row.get('tables_involved', '')).strip().lower()
            involved_tables = [t.strip() for t in tables_raw.split(',') if t.strip()]
            if not involved_tables:
                involved_tables = [tables_raw]

            rows_log = float(np.log1p(row.get('rows_affected', 0)))
            perm_denied = float(row.get('permission_denied', 0))
            # Lấy thêm 2 thuộc tính mới từ file a01
            is_success = float(row.get('success', 0))
            is_error = float(row.get('execution_error', 0))

            for tbl in involved_tables:
                if tbl in table_map:
                    t_idx = table_map[tbl]
                    sql_affects_table_edges.append([q_idx, t_idx])
                    affects_attr.append([rows_log, perm_denied, is_success, is_error])

        # ================================================================
        # 4. ĐƯA CẠNH VÀO HETERODATA
        # ================================================================
        def _to_edge_index(edge_list):
            return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        def _to_edge_attr(attr_list):
            return torch.tensor(attr_list, dtype=torch.float)

        data['user', 'opens',       'session']     .edge_index = _to_edge_index(user_opens_session_edges)
        data['user', 'opens',       'session']     .edge_attr  = _to_edge_attr(opens_attr)

        data['user', 'executes',    'sql_template'].edge_index = _to_edge_index(user_executes_sql_edges)
        data['user', 'executes',    'sql_template'].edge_attr  = _to_edge_attr(executes_attr)

        data['sql_template', 'belongs_to', 'session']    .edge_index = _to_edge_index(sql_belongs_session_edges)
        data['sql_template', 'belongs_to', 'session']    .edge_attr  = _to_edge_attr(belongs_attr)

        if len(sql_affects_table_edges) > 0:
            data['sql_template', 'affects', 'table'].edge_index = _to_edge_index(sql_affects_table_edges)
            data['sql_template', 'affects', 'table'].edge_attr  = _to_edge_attr(affects_attr)
        else:
            print("[Cảnh báo] Không có cạnh sql_template -> affects -> table nào được tạo. "
                "Kiểm tra cột 'tables_involved' trong audit log.")

        # ================================================================
        # 5. CẠNH CẤU TRÚC TĨNH: table → relates_to → table (PK-FK)
        # ================================================================
        relation_df.columns = relation_df.columns.str.strip()
        table_rel_edges = []

        col_p, col_c = 'source_table', 'target_table'
        if col_p in relation_df.columns and col_c in relation_df.columns:
            print(f"-> Đang kết nối đồ thị tĩnh bảng: [{col_p}] → [{col_c}]")
            for _, rel_row in relation_df.iterrows():
                s_tbl = str(rel_row[col_p]).strip().lower()
                t_tbl = str(rel_row[col_c]).strip().lower()
                if s_tbl in table_map and t_tbl in table_map:
                    table_rel_edges.append([table_map[s_tbl], table_map[t_tbl]])
                else:
                    for tbl in (s_tbl, t_tbl):
                        if tbl not in table_map:
                            print(f"   [Lưu ý] Bảng '{tbl}' không tìm thấy trong table_metadata.csv")
        else:
            print(f"[LỖI] Không tìm thấy cột '{col_p}' hoặc '{col_c}'!")
            print(f"Các cột hiện có: {list(relation_df.columns)}")

        if len(table_rel_edges) > 0:
            data['table', 'relates_to', 'table'].edge_index = _to_edge_index(table_rel_edges)
            print(f"-> Đã nạp {len(table_rel_edges)} cạnh table → relates_to → table.")

        # ================================================================
        # 6. NHÃN GROUND TRUTH Ở MỨC SESSION CHO CHUỖI THỜI GIAN
        # ================================================================
        session_labels = np.zeros(len(session_map), dtype=np.float32)
        for _, row in audit_df.iterrows():
            s_idx = session_map[row['session_id']]
            if row['is_anomaly'] == 1:
                session_labels[s_idx] = 1.0

        data['session'].y = torch.tensor(session_labels, dtype=torch.float)

    # ================================================================
        # Chỉ áp dụng cờ mặt nạ huấn luyện thực tế cho Snapshot cuối cùng (Target Graph T)
    # ================================================================
        if t == (num_snapshots - 1):
            n_sessions = len(session_map)
            indices    = np.random.permutation(n_sessions)

            n_train = int(0.6 * n_sessions)
            n_val   = int(0.2 * n_sessions)

            train_idx = indices[:n_train]
            val_idx   = indices[n_train : n_train + n_val]
            test_idx  = indices[n_train + n_val:]

            train_mask = torch.zeros(n_sessions, dtype=torch.bool)
            val_mask   = torch.zeros(n_sessions, dtype=torch.bool)
            test_mask  = torch.zeros(n_sessions, dtype=torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx]     = True
            test_mask[test_idx]   = True

            data['session'].train_mask = train_mask
            data['session'].val_mask   = val_mask
            data['session'].test_mask  = test_mask
        
        else:
            # Các snapshot quá khứ đóng vai trò làm Context dữ liệu, đặt mask ẩn (False) để tránh lỗi gọi logic
            n_sessions = len(session_map)
            data['session'].train_mask = torch.zeros(n_sessions, dtype=torch.bool)
            data['session'].val_mask   = torch.zeros(n_sessions, dtype=torch.bool)
            data['session'].test_mask  = torch.zeros(n_sessions, dtype=torch.bool)

        snapshot_chain.append(data)
        print(f"  > Hoàn thành trích xuất Subgraph Snapshot [{t+1}/{num_snapshots}] thành công.")

    # Xuất thông tin thống kê chuỗi
        print("\n--- Thống kê chuỗi đồ thị thời gian ---")
        print(f"Đã đóng gói tổng cộng {len(snapshot_chain)} snapshots con.")
        print(f"Kiểu dữ liệu file lưu trữ: {type(snapshot_chain)}")
        
        # Lưu danh sách chuỗi thời gian sạch lỗi logic vĩ mô
        torch.save(snapshot_chain, 'hetero_graph.pt')
        print("=== BƯỚC 2: Xuất file 'hetero_graph.pt' dạng Temporal Snapshots hoàn tất! ===")
        return snapshot_chain


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    build_heterogeneous_graph(num_snapshots=5)
