"""
a01_data_prepocessing.py — Bước 1: Tiền xử lý dữ liệu DB tự sinh
==================================================================
Sửa lỗi so với phiên bản upload:
    BUG-16: Thêm clearance_level vào user_df (a02 cần cột này).
"""

import pandas as pd
import numpy as np
import re
import datetime
from sklearn.cluster import KMeans

from constant import (
    AUDIT_LOG_PATH,
    ANOMALY_GROUND_TRUTH_PATH,
    USER_METADATA_PATH,
    CLEAN_AUDIT_LOG_PATH,
    CLEAN_USER_METADATA_PATH,
    ANOMALY_GROUND_TRUTH_USER_PATH,
)
from session_builder import assign_session_ids, analyze_idle_distribution

# ── Cấu hình session windowing ─────────────────────────────────────────────────
# Điều chỉnh IDLE_THRESHOLD_MIN sau khi chạy analyze_idle_distribution()
# Quy tắc chọn: IDLE_THRESHOLD ≈ 1.5 × P90(idle_time) làm tròn 5 phút
IDLE_THRESHOLD_MIN = 80    # mặc định 30 phút — hợp lý cho DB doanh nghiệp
SESSION_STRATEGY   = 'auto'  # 'auto' | 'inactivity' | 'fixed' | 'connection' | 'login'

# ── Giờ hành chính ─────────────────────────────────────────────────────────────
WORK_HOUR_START = datetime.time(8,  0, 0)
WORK_HOUR_END   = datetime.time(17, 45, 0)

# ── Regex IP nội bộ ────────────────────────────────────────────────────────────
INTERNAL_IP_REGEX = (
    r'^(10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.|192\.168\.|127\.|::1)'
)


# ══════════════════════════════════════════════════════════════════════════════
# SQL FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_sql_features(sql_text: str) -> dict:
    sql_upper = str(sql_text).upper()
    if   "SELECT"                          in sql_upper: cmd = 1
    elif "INSERT"                          in sql_upper: cmd = 2
    elif "UPDATE"                          in sql_upper: cmd = 3
    elif "DELETE"                          in sql_upper: cmd = 4
    elif "DROP"                            in sql_upper: cmd = 5
    elif re.search(r'\b(GRANT|REVOKE)\b',  sql_upper):  cmd = 6
    elif re.search(r'\b(CREATE|ALTER)\b',  sql_upper):  cmd = 7
    else:                                                cmd = 0

    join_count = len(re.findall(r'\bJOIN\b', sql_upper))
    where_count = (1 + len(re.findall(r'\b(AND|OR)\b', sql_upper))
                   if "WHERE" in sql_upper else 0)
    has_subquery = 1 if len(re.findall(r'\bSELECT\b', sql_upper)) > 1 else 0

    return dict(cmd_type=cmd, join_count=join_count,
                where_count=where_count, has_subquery=has_subquery)


def normalize_time_window(ts: pd.Series) -> pd.Series:
    sec = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    return sec / 86400.0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_pipeline():
    # ── 1A: Đọc audit log ────────────────────────────────────────────────────
    audit_df = pd.read_csv(AUDIT_LOG_PATH)
    gt_df    = pd.read_csv(ANOMALY_GROUND_TRUTH_PATH)
    audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
    # gt_df['timestamp']    = pd.to_datetime(gt_df['timestamp'])

    # ── 1A-THÊM: Gán session_id nếu chưa có ─────────────────────────────────
    if 'session_id' not in audit_df.columns:
        print("-> 'session_id' chưa có → tự động phân chia session...")
        # Phân tích distribution để chọn idle_threshold đúng:
        analyze_idle_distribution(audit_df, user_col='db_user',
                                  time_col='timestamp')
        audit_df = assign_session_ids(
            audit_df,
            idle_threshold_min = IDLE_THRESHOLD_MIN,
            strategy           = SESSION_STRATEGY,
            user_col           = 'db_user',
            time_col           = 'timestamp',
        )
        print(f"   Đã gán {audit_df['session_id'].nunique():,} session IDs "
              f"(strategy={audit_df['session_strategy'].iloc[0]})")
    else:
        print(f"-> 'session_id' đã có: {audit_df['session_id'].nunique():,} sessions")

    


    # File có nhãn is_anomaly rồi"
    print("-> Đồng bộ nhãn độc hại...")
    # chỉ dùng để thống kê ko dùng để đưa vào xây graph bỏ scenerio type, attack_type, attack_phase trong a02 tránh lộ nhãn
    audit_df['attack_type'] = audit_df['scenario_type'].astype(str).str.lower() 
    # Dùng merge (how='left' giữ lại toàn bộ event_id của df1)
    # merged_df = pd.merge(audit_df, gt_df[['event_id', 'is_anomaly']], on='event_id', how='left')


    # Gán cột mới bằng giá trị vừa tìm được
    # audit_df['is_anomaly'] = merged_df['is_anomaly']

    audit_df = pd.merge(audit_df, gt_df[['event_id','is_anomaly']], on='event_id', how='left')

    # ── 1B: SQL features ───────────────────────────────────────────────────────
    sql_feat_df = pd.DataFrame(
        audit_df['sql_statement'].apply(extract_sql_features).tolist()
    )
    audit_df = pd.concat([audit_df.reset_index(drop=True), sql_feat_df], axis=1)

    # ── 1C: Thuộc tính cạnh + nút Session ────────────────────────────────────
    status_upper = audit_df['status'].astype(str).str.upper()
    audit_df['permission_denied'] = status_upper.str.contains('DENIED',  regex=False).astype(int)
    audit_df['success']           = status_upper.str.contains('SUCCESS', regex=False).astype(int)
    audit_df['execution_error']   = status_upper.str.contains('ERROR',   regex=False).astype(int)

    audit_df['is_after_hours'] = audit_df['timestamp'].apply(
        lambda ts: 0 if WORK_HOUR_START <= ts.time() < WORK_HOUR_END else 1
    )
    audit_df['time_window_norm'] = normalize_time_window(audit_df['timestamp'])

    if 'client_ip' in audit_df.columns:
        ips = audit_df['client_ip'].astype(str).fillna('')
        audit_df['is_internal_ip'] = ips.str.contains(INTERNAL_IP_REGEX, regex=True).astype(int)
        audit_df.loc[audit_df['client_ip'].isna(), 'is_internal_ip'] = -1
    else:
        print("[WARNING] Không có cột 'client_ip' → mặc định is_internal_ip = 1")
        audit_df['is_internal_ip'] = 1

    # Session duration
    sess_time = audit_df.groupby('session_id')['timestamp'].agg(['min', 'max'])
    sess_time['session_duration'] = (sess_time['max'] - sess_time['min']).dt.total_seconds()
    audit_df = audit_df.merge(
        sess_time[['session_duration']].reset_index(), on='session_id', how='left'
    )

    # Sequence order & execution count
    audit_df = audit_df.sort_values(['session_id', 'timestamp'])
    audit_df['sequence_order'] = audit_df.groupby('session_id').cumcount() + 1

    sql_grp_cols = ['session_id', 'db_user', 'cmd_type', 'join_count',
                    'where_count', 'has_subquery']
    exec_cnt = (audit_df.groupby(sql_grp_cols)
                .size().reset_index(name='execution_count'))
    audit_df = audit_df.merge(exec_cnt, on=sql_grp_cols, how='left')

    # ── 1D: User metadata ─────────────────────────────────────────────────────
    user_df = pd.read_csv(USER_METADATA_PATH)

    role_map = {
        "admin": 4, 
        # "lead_engineer": 4,
        "hr_specialist": 3,
        "developer": 2,
        "staff": 1,
    }
    dept_map = {
        "sales": 1, "it_engineering": 2,
        "human_resources": 3, "infrastructure": 4
    }
    user_df['role_index']       = user_df['role'].map(role_map).fillna(0).astype(int)
    user_df['department_index'] = user_df['department'].map(dept_map).fillna(0).astype(int)

    # DÙng kmeans chia 2 khung giờ thời gian tính behavirol_baseline_hour
    normal_logs = audit_df[audit_df['is_anomaly'] == 0].copy()

    # Khởi tạo từ điển lưu mốc thời gian nền cho từng user
    user_baseline_1 = {}
    user_baseline_2 = {}

    if not normal_logs.empty and 'db_user' in normal_logs.columns:
        # Lấy giờ thực tế từ timestamp
        normal_logs['hour_ext'] = normal_logs['timestamp'].dt.hour
        
        # Duyệt qua từng người dùng để phân cụm thời gian làm việc
        for db_user, group in normal_logs.groupby('db_user'):
            hours = group['hour_ext'].values
            
            # Nếu user có quá ít dữ liệu hoặc chỉ hoạt động duy nhất 1 khung giờ tĩnh
            if len(hours) < 2 or len(np.unique(hours)) == 1:
                user_baseline_1[db_user] = float(hours[0]) if len(hours) > 0 else 8.0
                user_baseline_2[db_user] = float(hours[0]) if len(hours) > 0 else 8.0
                continue
                
            # Biến đổi mảng giờ thành ma trận 2D nạp vào KMeans
            X = hours.reshape(-1, 1)
            
            # Phân cụm hành vi thành 2 cụm (đại diện cho Ca ngày và Ca đêm bảo trì)
            kmeans = KMeans(n_clusters=2, n_init=5, max_iter=50, random_state=42)
            kmeans.fit(X)
            
            # Lấy tọa độ 2 tâm cụm và sắp xếp tăng dần để nhất quán
            centers = sorted(kmeans.cluster_centers_.flatten())
            
            user_baseline_1[db_user] = float(centers[0]) # Thường rơi vào mốc ca ngày (~8h đến 12h)
            user_baseline_2[db_user] = float(centers[1]) # Thường rơi vào mốc ca muộn/đêm (~16h đến 22h)

    # 2. Ánh xạ ma trận đặc trưng tĩnh này vào DataFrame user_df
    user_df['baseline_hour_1'] = user_df['db_user'].map(user_baseline_1).fillna(8.0)
    user_df['baseline_hour_2'] = user_df['db_user'].map(user_baseline_2).fillna(18.0)
    # ── Lưu ───────────────────────────────────────────────────────────────────
    audit_df.to_csv(CLEAN_AUDIT_LOG_PATH, index=False)
    user_df.to_csv(CLEAN_USER_METADATA_PATH, index=False)
    print(f"   Audit events   : {len(audit_df):,}")
    print(f"   Sessions       : {audit_df['session_id'].nunique():,}")
    print(f"   Anomaly events : {int(audit_df['is_anomaly'].sum()):,}")
    print("=== BƯỚC 1: Tiền xử lý hoàn tất! ===")


if __name__ == "__main__":
    preprocess_pipeline()
