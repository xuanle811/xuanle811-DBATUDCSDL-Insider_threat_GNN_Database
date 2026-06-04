"""
Nhiệm vụ của File:
- Đọc các file csv trong thư mục db_data
- Trích xuất câu lệnh SQL thô thành vector số
- Gán nhãn cho các session id: Timestamp trong 1 khoảng thời gian + db_user

[Sửa đổi so với phiên bản gốc]:
- Thêm đặc trưng Behavioral_Baseline_Hour vào user_df (theo PDF node spec)
- Thêm trường is_after_hours, session_duration vào audit_df để phục vụ edge features
- Thêm trường is_internal_ip, time_window_norm cho nút Session
- Chuẩn hóa tên cột theo đúng đặc tả trong 2 file PDF
- Thêm cột sequence_order (thứ tự câu lệnh trong session) cho cạnh BELONGS_TO
"""

import pandas as pd
import numpy as np
import re
import datetime

from constant import (
    AUDIT_LOG_PATH,
    ANOMALY_GROUND_TRUTH_PATH,
    USER_METADATA_PATH,
    CLEAN_AUDIT_LOG_PATH,
    CLEAN_USER_METADATA_PATH,
)


# ============================================================
# BUSINESS HOURS CONFIGURATION
# Giờ hành chính: 8h - 18h.  Ngoài khoảng này => after_hours = 1
# ============================================================
WORK_HOUR_START = datetime.time(8, 00, 0)   # Ví dụ: 08h 30p 00s
WORK_HOUR_END   = datetime.time(17, 45, 0)  # Ví dụ: 17h 45p 00s


def extract_sql_features(sql_text):
    """
    Trích xuất vector đặc trưng cú pháp câu lệnh SQL.
    Trả về dict: cmd_type, join_count, where_count, has_subquery
    - cmd_type   : mã hóa loại lệnh (1–7, 0=UNKNOWN)
    - join_count : số phép JOIN
    - where_count: số điều kiện lọc (WHERE + AND/OR)
    - has_subquery: 1 nếu có SELECT lồng nhau
    """
    sql_upper = str(sql_text).upper()

    # 1. Mã hóa loại lệnh chính (theo thứ tự ưu tiên để tránh nhầm lẫn)
    if   "SELECT"              in sql_upper: cmd_type = 1
    elif "INSERT"              in sql_upper: cmd_type = 2
    elif "UPDATE"              in sql_upper: cmd_type = 3
    elif "DELETE"              in sql_upper: cmd_type = 4
    elif "DROP"                in sql_upper: cmd_type = 5
    elif re.search(r'\b(GRANT|REVOKE)\b', sql_upper): cmd_type = 6
    elif re.search(r'\b(CREATE|ALTER)\b',  sql_upper): cmd_type = 7
    else:                                              cmd_type = 0

    # 2. Đếm số phép JOIN
    join_count = len(re.findall(r'\bJOIN\b', sql_upper))

    # 3. Đếm điều kiện lọc WHERE (WHERE + số AND/OR bên trong)
    if "WHERE" in sql_upper:
        where_count = 1 + len(re.findall(r'\b(AND|OR)\b', sql_upper))
    else:
        where_count = 0

    # 4. Kiểm tra subquery: SELECT xuất hiện nhiều hơn 1 lần
    has_subquery = 1 if len(re.findall(r'\bSELECT\b', sql_upper)) > 1 else 0

    return {
        'cmd_type':    cmd_type,
        'join_count':  join_count,
        'where_count': where_count,
        'has_subquery': has_subquery,
    }


def normalize_time_window(timestamps: pd.Series) -> pd.Series:
    """
    Chuẩn hóa timestamp về [0, 1] theo khoảng thời gian trong ngày.
    Công thức: (hour * 3600 + minute * 60 + second) / 86400
    """
    seconds_in_day = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
    return seconds_in_day / 86400.0


def preprocess_pipeline():
    # ================================================================
    # BƯỚC 1A: ĐỌC VÀ GÁN NHÃN AUDIT LOG
    # ================================================================
    audit_df = pd.read_csv(AUDIT_LOG_PATH)
    gt_df    = pd.read_csv(ANOMALY_GROUND_TRUTH_PATH)

    audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
    gt_df['timestamp']    = pd.to_datetime(gt_df['timestamp'])

    # Mặc định: an toàn
    audit_df['is_anomaly']  = 0
    audit_df['attack_type'] = 'normal'

    print("-> Đang tiến hành đồng bộ nhãn độc hại qua thời gian và tài khoản...")
    for _, gt_row in gt_df.iterrows():
        attack_time  = gt_row['timestamp']
        attacker     = gt_row['db_user']
        attack_name  = gt_row['attack_type']

        start_window = attack_time - pd.Timedelta(seconds=2)
        end_window   = attack_time + pd.Timedelta(seconds=2)

        matched_events = audit_df[
            (audit_df['db_user']   == attacker) &
            (audit_df['timestamp'] >= start_window) &
            (audit_df['timestamp'] <= end_window)
        ]

        if not matched_events.empty:
            bad_session_ids = matched_events['session_id'].unique()
            audit_df.loc[audit_df['session_id'].isin(bad_session_ids), 'is_anomaly']  = 1
            audit_df.loc[audit_df['session_id'].isin(bad_session_ids), 'attack_type'] = attack_name

    # ================================================================
    # BƯỚC 1B: TRÍCH XUẤT ĐẶC TRƯNG SQL
    # ================================================================
    sql_features = audit_df['sql_statement'].apply(extract_sql_features)
    sql_feat_df  = pd.DataFrame(sql_features.tolist())
    audit_df     = pd.concat([audit_df.reset_index(drop=True), sql_feat_df], axis=1)

    # ================================================================
    # BƯỚC 1C: TÍNH TOÁN CÁC THUỘC TÍNH CẠNH VÀ NÚT SESSION
    # ================================================================

    # --- Thuộc tính cạnh: is_permission_denied (cho cạnh sql_template -> affects -> table)
    # 1. Quyền truy cập bị từ chối (DENIED)
    status_upper = audit_df['status'].astype(str).str.upper()
    # Khớp chính xác 3 trạng thái rõ ràng, độc lập
    audit_df['permission_denied'] = status_upper.str.contains('DENIED', regex=False).astype(int)
    audit_df['success']           = status_upper.str.contains('SUCCESS', regex=False).astype(int)
    audit_df['execution_error']   = status_upper.str.contains('ERROR', regex=False).astype(int)

    # --- Thuộc tính cạnh: is_after_hours (cho cạnh user -> opens -> session)
    # Lấy ra phần giờ:phút:giây từ cột timestamp để so sánh với cấu hình
    audit_df['is_after_hours'] = audit_df['timestamp'].apply(
        lambda ts: 0 if WORK_HOUR_START <= ts.time() < WORK_HOUR_END else 1
    )

    # --- Thuộc tính nút Session: time_window_norm
    audit_df['time_window_norm'] = normalize_time_window(audit_df['timestamp'])

    # --- Thuộc tính nút Session: is_internal_ip
    # Nếu cột 'client_ip' tồn tại, phân loại IP nội bộ (10.x, 172.16.x, 192.168.x)
    # sửa lại hàm kiểm tra internal và client ip

    # Định nghĩa chuỗi Regex tối ưu: bao gồm cả dải IP Private (10.x, 172.16-31.x, 192.168.x) và Localhost (127.x, ::1)
    INTERNAL_IP_REGEX = r'^(10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.|192\.168\.|127\.|::1)'

    if 'client_ip' in audit_df.columns:
        # Bước 1: Ép kiểu về chuỗi và điền các giá trị trống (NaN) thành chuỗi rỗng để tránh lỗi dữ liệu
        client_ips = audit_df['client_ip'].astype(str).fillna('')
        
        # Bước 2: Dùng Vectorization với .str.contains() chạy nhanh hơn .apply(lambda) gấp nhiều lần
        audit_df['is_internal_ip'] = client_ips.str.contains(INTERNAL_IP_REGEX, regex=True).astype(int)
        
        # Bước 3 (Nâng cao): Nếu IP bị rỗng/không xác định, không nên coi là nội bộ an toàn (gán bằng -1 để mô hình chú ý)
        audit_df.loc[audit_df['client_ip'].isna(), 'is_internal_ip'] = -1

    else:
        # Khi thiếu hẳn cột dữ liệu IP: Đặt giá trị mặc định là 1 cho môi trường DB doanh nghiệp như bạn mong muốn
        # Nhưng khuyến khích bạn in ra một cảnh báo (Warning) để log lại quá trình tracking.
        print("[WARNING] Cột 'client_ip' không tồn tại trong dữ liệu audit. Mặc định gán is_internal_ip = 1.")
        audit_df['is_internal_ip'] = 1

    # --- Thuộc tính cạnh: session_duration (giây) — tính theo khoảng thời gian trong session
    session_time = audit_df.groupby('session_id')['timestamp'].agg(['min', 'max'])
    session_time['session_duration'] = (session_time['max'] - session_time['min']).dt.total_seconds()
    audit_df = audit_df.merge(
        session_time[['session_duration']].reset_index(),
        on='session_id', how='left'
    )

    # --- Thuộc tính cạnh: sequence_order — thứ tự câu lệnh trong session (cho BELONGS_TO)
    audit_df = audit_df.sort_values(['session_id', 'timestamp'])
    audit_df['sequence_order'] = audit_df.groupby('session_id').cumcount() + 1

    # --- Thuộc tính cạnh: execution_count — số lần user gọi cùng SQL template trong session
    # (dùng cho cạnh user -> executes -> sql_template)
    sql_template_cols = ['session_id', 'db_user', 'cmd_type', 'join_count', 'where_count', 'has_subquery']
    exec_count = (
        audit_df.groupby(sql_template_cols)
        .size()
        .reset_index(name='execution_count')
    )
    audit_df = audit_df.merge(exec_count, on=sql_template_cols, how='left')

    # ================================================================
    # BƯỚC 1D: XỬ LÝ USER METADATA (TĨNH)
    # ================================================================
    print("-> Đang xử lý metadata người dùng...")
    user_df = pd.read_csv(USER_METADATA_PATH)

    # Định nghĩa cấu trúc ánh xạ sạch sẽ (Tránh lỗi so sánh chuỗi trên Series)
    role_map = {
        "DB_Admin": 3, "Lead_Engineer": 3,
        "HR_Specialist": 2, "Developer": 2,
        "Accountant": 1, "Sales_Agent": 1
    }
    user_df['role_index'] = user_df['role'].map(role_map).fillna(0).astype(int)

    dept_map = {
        "IT_Operations": 1,
        "Finance": 2,
        "Human_resources": 3,
        "Business": 4,
        "IT_Engineering": 5
    }
    user_df['department_index'] = user_df['department'].map(dept_map).fillna(0).astype(int)
    # behavioral_baseline_hour: giờ làm việc bình thường (8=bình thường, 22=đêm cho admin)
    # Gán dựa trên role để phục vụ phát hiện hành vi out-of-hours
    #ktl@#
    # Tính toán khoa học hơn cho behavioral_baseline_hour thay vì gán cứng
    # Thử tính toán dựa trên dữ liệu lịch sử đăng nhập thực tế của user trong audit log (nếu có)
    normal_logs = audit_df[audit_df['is_anomaly'] == 0]
    if not normal_logs.empty and 'db_user' in normal_logs.columns:
        # Lấy giờ hoạt động trung bình thực tế từ các bản ghi không độc hại
        normal_logs['hour_extracted'] = normal_logs['timestamp'].dt.hour
        baseline_hours_map = normal_logs.groupby('db_user')['hour_extracted'].mean().to_dict()
        user_df['behavioral_baseline_hour'] = user_df['db_user'].map(baseline_hours_map)
    
    # Nếu user không có log hoặc quá trình tính toán trên bị trống, fallback về logic cũ dựa vào role
    if 'behavioral_baseline_hour' not in user_df.columns or user_df['behavioral_baseline_hour'].isna().any():
        user_df['behavioral_baseline_hour'] = user_df['behavioral_baseline_hour'].fillna(
            user_df['db_user'].apply(lambda x: 22.0 if 'admin' in str(x).lower() else 8.0)
        )

    # ================================================================
    # LƯU DỮ LIỆU ĐÃ XỬ LÝ
    # ================================================================
    audit_df.to_csv(CLEAN_AUDIT_LOG_PATH, index=False)
    user_df.to_csv(CLEAN_USER_METADATA_PATH, index=False)
    print(f"   Đã xử lý {len(audit_df)} sự kiện audit log.")
    print(f"   Số phiên làm việc (sessions): {audit_df['session_id'].nunique()}")
    print(f"   Số sự kiện bất thường: {audit_df['is_anomaly'].sum()}")
    print("=== BƯỚC 1: Tiền xử lý dữ liệu hoàn tất! ===")


if __name__ == "__main__":
    preprocess_pipeline()
