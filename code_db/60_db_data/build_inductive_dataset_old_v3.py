#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: build_compressed_dataset1.py (Phiên bản v12.0 - SQL ĐA DẠNG / MULTI-ATTACKER / DAYTIME INSIDER)
G xác định các khung giờ cố định
Điểm khác biệt so với v10.0:
  - Kẻ tấn công KHÔNG cố định là 'insider_threat' mà được chọn NGẪU NHIÊN
    từ bất kỳ user Cấp 1-3 nào → Mô phỏng thực tế hơn cho Inductive GNN
  - Hành vi bình thường đa dạng theo vai trò (staff / developer / manager)
  - Không có user 'insider_threat' cố định trong danh sách
"""

import pandas as pd
import numpy as np
import random
import os
# import attack
from datetime import datetime, timedelta

try:
    import attack
    HAS_ATTACK_MODULE = True
    print("✅ Đã tải module attack.py với 10 kịch bản tấn công.")
except ImportError:
    HAS_ATTACK_MODULE = False
    print("⚠️  Không tìm thấy attack.py. Sử dụng fallback đơn giản.")

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =====================================================================
DATA_DIR          = "raw"
USER_META_PATH    = os.path.join(DATA_DIR, "user_metadata.csv")
TABLE_META_PATH   = os.path.join(DATA_DIR, "table_metadata.csv")
AUDIT_LOG_PATH    = os.path.join(DATA_DIR, "audit_log.csv")
TABLE_REL_PATH    = os.path.join(DATA_DIR, "table_relations.csv")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "anomaly_ground_truth.csv")

# =====================================================================
# CẤU HÌNH GRAPH FRAMES
# =====================================================================
GRAPH_FRAMES      = ["G1", "G2", "G3", "G4", "G5"]
# G1-G3: trong giờ hành chính | G4-G5: ngoài giờ / ban đêm
FRAME_AFTER_HOURS = {"G1": 0, "G2": 0, "G3": 0, "G4": 1, "G5": 1}
FRAME_HOUR        = {"G1": (8, 11), "G2": (11, 14), "G3": (14, 18), "G4": (18, 2), "G5": (2, 8)}


#===================================================================
# TEMPORAL WINDOW SAMPLING
# THÊM HÀM Random hoàn toàn hour, minute, second
#====================================================================
def random_time_in_frame(frame: str):
    """
    Sinh timestamp ngẫu nhiên trong temporal window thực.
    """

    start_h, end_h = FRAME_HOUR[frame]

    # Window bình thường
    if start_h < end_h:
        valid_hours = list(range(start_h, end_h))

    # Window qua nửa đêm
    else:
        valid_hours = list(range(start_h, 24)) + list(range(0, end_h))

    hour = random.choice(valid_hours)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    return hour, minute, second

# =====================================================================
# SINH HÀNH VI BÌNH THƯỜNG ĐA DẠNG THEO VAI TRÒ
# =====================================================================
LV1_TABLES = ["customer", "orders", "lineitem", "region", "nation", "part", "supplier"]
LV2_TABLES = LV1_TABLES + ["partsupp"]
LV3_TABLES = LV2_TABLES + ["salary_details"]
# Thêm giá trị "Daily_Report"
NORMAL_SCENARIOS = ["Normal", "Normal_Admin", "Normal_Admin_Maintenance", "Daily-Report"]


def _normal_sql(user_meta: dict, step: int, table_df: pd.DataFrame) -> dict:
    """Sinh câu SQL bình thường đa dạng theo vai trò: SELECT/INSERT/UPDATE/JOIN/subquery."""
    clv           = user_meta["clearance_level"]
    scenario_type = random.choice(NORMAL_SCENARIOS)

    if clv == 1:  # Staff – chỉ đọc dữ liệu nghiệp vụ cơ bản
        tbl = random.choice(LV1_TABLES)
        t   = tbl[:2]
        
        templates = [
            f"SELECT * FROM public.{tbl} WHERE {t}_key = {random.randint(1, 5000)};",
            f"SELECT COUNT(*) FROM public.{tbl};",
            f"SELECT * FROM public.{tbl} ORDER BY 1 LIMIT {random.randint(10, 50)};",
            f"SELECT * FROM public.{tbl} WHERE 1=1 LIMIT 20;",
            f"SELECT {t}_key, {t}_comment FROM public.{tbl} WHERE {t}_key BETWEEN {random.randint(1,100)} AND {random.randint(200,500)};",
            f"SELECT COUNT(*), MAX({t}_key) FROM public.{tbl} WHERE {t}_key > 0;",
            f"SELECT c.c_name, o.o_totalprice FROM public.customer c JOIN public.orders o ON c.c_custkey = o.o_custkey LIMIT {random.randint(10,40)};",
            f"SELECT n.n_name, COUNT(c.c_custkey) FROM public.nation n JOIN public.customer c ON n.n_nationkey = c.c_nationkey GROUP BY n.n_name LIMIT 20;",
            f"SELECT * FROM public.{tbl} WHERE {t}_key IN (SELECT {t}_key FROM public.{tbl} ORDER BY 1 LIMIT 50);",
            f"SELECT p.p_name, ps.ps_supplycost FROM public.part p JOIN public.partsupp ps ON p.p_partkey = ps.ps_partkey LIMIT {random.randint(10, 40)};",

            # template mới tăng đa dạng
            f"SELECT * FROM public.{tbl} WHERE {t}_key % 2 = 0 LIMIT {random.randint(10,50)};",
            f"SELECT * FROM public.{tbl} WHERE {t}_key > {random.randint(100,1000)};",
            f"SELECT * FROM public.{tbl} WHERE {t}_key < {random.randint(1000,5000)};",
            f"SELECT {t}_key, {t}_comment FROM public.{tbl} ORDER BY {t}_key DESC LIMIT {random.randint(5,50)};",
            f"SELECT * FROM public.{tbl} WHERE {t}_key BETWEEN {random.randint(1,50)} AND {random.randint(51,150)};",
            f"SELECT * FROM public.{tbl} ORDER BY random() LIMIT {random.randint(5,30)};",
            f"SELECT COUNT(*), AVG({t}_key) FROM public.{tbl} WHERE {t}_key > 0;",
        ]
        # Lấy row_count bảng hiện tại
        tbl_row_count = table_df.loc[table_df["table_name"]==tbl, "row_count"].values[0]
        # Tạo rows_affected tỷ lệ thuận với row_count bảng
        rows = random.randint(max(1,int(0.01*tbl_row_count)), max(1,int(0.05*tbl_row_count)))

    elif clv == 2:  # Developer – đọc/ghi nghiệp vụ, explain, index check
        tbl = random.choice(LV2_TABLES)
        t   = tbl[:2]
        templates = [
            f"SELECT * FROM public.{tbl} LIMIT {random.randint(20, 100)};",
            f"SELECT COUNT(*) FROM public.{tbl} WHERE 1=1;",
            f"EXPLAIN ANALYZE SELECT * FROM public.{tbl} LIMIT 50;",
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{tbl}';",
            f"SELECT * FROM public.{tbl} WHERE {t}_key = {random.randint(1, 1000)};",
            f"INSERT INTO public.{tbl} SELECT * FROM public.{tbl} WHERE {t}_key = {random.randint(1, 100)};",
            f"UPDATE public.{tbl} SET {t}_comment = 'updated_{step}' WHERE {t}_key = {random.randint(1, 500)};",
            f"SELECT o.o_orderkey, l.l_quantity, l.l_shipdate FROM public.orders o JOIN public.lineitem l ON o.o_orderkey = l.l_orderkey WHERE o.o_totalprice > {random.randint(1000,9000)} LIMIT 30;",
            f"SELECT s.s_name, ps.ps_supplycost FROM public.supplier s JOIN public.partsupp ps ON s.s_suppkey = ps.ps_suppkey ORDER BY ps.ps_supplycost DESC LIMIT 20;",
            f"SELECT {t}_key FROM public.{tbl} WHERE {t}_key NOT IN (SELECT {t}_key FROM public.{tbl} WHERE {t}_key < {random.randint(100,500)}) LIMIT 30;",
            f"SELECT pg_size_pretty(pg_total_relation_size('public.{tbl}')) AS table_size;",
            f"SELECT indexname, indexdef FROM pg_indexes WHERE tablename = '{tbl}';",

            # template mới
            f"SELECT * FROM public.{tbl} WHERE {t}_key % 3 = 0 LIMIT {random.randint(10,50)};",
            f"SELECT * FROM public.{tbl} WHERE {t}_key BETWEEN {random.randint(50,200)} AND {random.randint(201,500)};",
            f"SELECT * FROM public.{tbl} ORDER BY random() LIMIT {random.randint(5,30)};",
        ]
        # Lấy row_count bảng hiện tại
        tbl_row_count = table_df.loc[table_df["table_name"]==tbl, "row_count"].values[0]
        # Tạo rows_affected tỷ lệ thuận với row_count bảng
        rows = random.randint(max(1,int(0.01*tbl_row_count)), max(1,int(0.05*tbl_row_count)))

    else:  # clv == 3 – HR Specialist – đọc lương, nhân sự, báo cáo
        tbl = random.choice(LV3_TABLES)
        t   = tbl[:2]
        templates = [
            f"SELECT * FROM public.{tbl} LIMIT {random.randint(10, 80)};",
            f"SELECT COUNT(*) FROM public.{tbl};",
            f"SELECT * FROM public.salary_details WHERE department = 'Finance' LIMIT 30;",
            f"SELECT AVG(salary), MAX(salary), MIN(salary) FROM public.salary_details;",
            f"SELECT * FROM public.{tbl} ORDER BY 1 DESC LIMIT 20;",
            f"SELECT db_user, salary FROM public.salary_details WHERE salary > {random.randint(50000, 200000)} LIMIT 20;",
            f"SELECT department, COUNT(*) as headcount, AVG(salary) FROM public.salary_details GROUP BY department;",
            f"SELECT sd.db_user, sd.salary, c.c_name FROM public.salary_details sd JOIN public.customer c ON sd.db_user = c.c_name LIMIT 20;",
            f"UPDATE public.salary_details SET salary = salary * 1.05 WHERE department = 'HR' AND db_user = '{user_meta['db_user']}';",
            f"SELECT * FROM public.salary_details WHERE salary BETWEEN {random.randint(30000,80000)} AND {random.randint(100000,300000)};",
            f"INSERT INTO public.salary_details (db_user, department, salary) VALUES ('new_emp_{step}', 'HR', {random.randint(40000, 90000)});",

            # template mới
            f"SELECT * FROM public.{tbl} ORDER BY random() LIMIT {random.randint(5,30)};",
            f"SELECT * FROM public.{tbl} WHERE salary % 2 = 0 LIMIT {random.randint(5,20)};",
            f"SELECT department, MAX(salary) FROM public.salary_details GROUP BY department;",
        ]
        # rows = random.randint(40, 150) #(10, 100)
        # Lấy row_count bảng hiện tại
        tbl_row_count = table_df.loc[table_df["table_name"]==tbl, "row_count"].values[0]
        # Tạo rows_affected tỷ lệ thuận với row_count bảng
        rows = random.randint(max(1,int(0.01*tbl_row_count)), max(1,int(0.05*tbl_row_count))
    )

    stmt = random.choice(templates)
    return {"sql": stmt, "table": tbl, "rows": rows, "status": "SUCCESS", "scenario_type": scenario_type}

#===== HÀNH VI CỦA ADMIN===============
def _admin_sql(step: int, is_maintenance: bool = False) -> dict:
    """
    Sinh câu lệnh DBA bình thường cho tài khoản postgres (Cấp 4).
 
    Phân loại hành vi:
      - Ban ngày (G1, G2, G3): Giám sát hiệu năng, kiểm tra kết nối, xem cấu hình
      - Bảo trì ngoài giờ (G4 xác suất thấp): VACUUM, ANALYZE, REINDEX, backup
 
    Đặc điểm quan trọng cho GNN:
      - rows_affected thường = 0 (lệnh quản trị không trả về dữ liệu nghiệp vụ)
      - tables_involved là system_catalog hoặc pg_catalog (đặc thù admin)
      - Tần suất thấp → nếu đột ngột tăng vọt = bất thường
    """
    scenario_type = "Normal_Admin"
    if is_maintenance:
        # Lệnh bảo trì định kỳ ngoài giờ: ít gặp, rows_affected lớn (pages xử lý)
        tbl = random.choice(["orders", "lineitem", "customer", "partsupp"])
        templates = [
            f"VACUUM ANALYZE public.{tbl};",
            f"VACUUM FULL public.{tbl};",
            f"ANALYZE public.{tbl};",
            f"REINDEX TABLE public.{tbl};",
            f"CLUSTER public.{tbl} USING {tbl[:2]}_pkey;",
            "CHECKPOINT;",
            "SELECT pg_rotate_logfile();",
        ]
        rows = random.randint(0, 50)   # pages processed, không phải data rows
        tbl_log = "system_catalog"
    else:
        # Giám sát & vận hành ban ngày
        tbl_log = "system_catalog"        
        templates = [
            # Kiểm tra kết nối đang hoạt động
            "SELECT pid, usename, application_name, state, query_start FROM pg_stat_activity WHERE state = 'active';",
            # Giám sát hiệu năng truy vấn chậm
            "SELECT query, calls, total_exec_time, rows FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 10;",
            # Kiểm tra kích thước bảng
            "SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) AS size FROM pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC LIMIT 10;",
            # Kiểm tra lock đang chờ
            "SELECT pid, relation::regclass, mode, granted FROM pg_locks WHERE NOT granted;",
            # Kiểm tra replication lag
            "SELECT client_addr, state, sent_lsn, write_lsn, replay_lsn FROM pg_stat_replication;",
            # Xem cấu hình hệ thống
            f"SHOW max_connections;",
            f"SHOW work_mem;",
            f"SHOW shared_buffers;",
            # Kiểm tra index usage
            "SELECT indexrelname, idx_scan, idx_tup_read FROM pg_stat_user_indexes ORDER BY idx_scan DESC LIMIT 10;",
            # Xem log lỗi gần nhất
            "SELECT log_time, error_severity, message FROM pg_catalog.pg_log ORDER BY log_time DESC LIMIT 20;",
        ]
        rows = random.randint(0, 20)
        scenario_type = "Normal_Admin" # Bổ sung dòng này
 
    stmt = random.choice(templates)
    return {"sql": stmt, "table": tbl_log, "rows": rows, "status": "SUCCESS", "scenario_type": scenario_type}


#=======================================================
# USER
#==============================================================
def build_user_metadata() -> pd.DataFrame:
    users = []

    # Nhân viên bình thường – Sales & Finance tăng từ 20 LÊN 25
    for i in range(1, 25):
        users.append({
            "db_user": f"normal_user_{i}",
            "department": "sales",
            "clearance_level": 1,
            "role": "staff",
            "is_privileged": 0,
        })

    # Lập trình viên / DBA junior
    for i in range(1, 5): # TỪ 4 LÊN 5
        users.append({
            "db_user": f"dev_user_{i}",
            "department": "it_engineering",
            "clearance_level": 2,
            "role": "developer",
            "is_privileged": 0,
        })

    for i in range(1, 5): # TỪ 3 LÊN 5
        users.append({
            "db_user": f"hr_specialist_{i}",
            "department": "human_resources",
            "clearance_level": 3,
            "role": "hr_specialist",
            "is_privileged": 0,
        })

       # Quản trị viên hệ thống
    users.append({
        "db_user": "postgres",
        "department": "infrastructure",
        "clearance_level": 4,
        "role": "admin",
        "is_privileged": 1,
    })

    df = pd.DataFrame(users)
    df.to_csv(USER_META_PATH, index=False)
    print(f"  [1/4] user_metadata.csv: {len(df)} người dùng")
    return df
# =====================================================================
# PIPELINE CHÍNH (thêm cơ chế chèn chuỗi)
# =====================================================================
def generate_dynamic_insider_dataset(num_days: int = 30, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("🚀 PIPELINE v10.2 – ĐỒNG BỘ CHUỖI TUẦN TỰ TRÊN ĐỒ THỊ SNAPSOT – Dynamic Insider Threat (Cấp 1-3 làm kẻ tấn công)")
    print(f"   Thời gian mô phỏng : {num_days} ngày | 5 graph frames/ngày")
    print(f"   Kẻ tấn công        : Chọn ngẫu nhiên từ Cấp 1-3 mỗi đêm")
    print(f"   Admin baseline     : Giám sát ban ngày (G1, G2, G3) + bảo trì ngoài giờ (G4, G5)")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. DANH MỤC NGƯỜI DÙNG: Cấp 1 → 4, không có 'insider_threat' cố định
    # ------------------------------------------------------------------
    #print(f"  [1/4] user_metadata.csv : {len(users)} người dùng (không có insider cố định)")
    users_df = build_user_metadata()
    # Chuyển DataFrame -> list[dict]
    users = users_df.to_dict(orient="records")
    print(f"  [1/4] user_metadata.csv : {len(users)} người dùng (không có insider cố định)")
    # ------------------------------------------------------------------
    # 2. DANH MỤC BẢNG
    # ------------------------------------------------------------------
    tables = [
        {"table_name": "region",        "security_level": 1, "row_count": 10,         "schema": "public"},
        {"table_name": "nation",         "security_level": 1, "row_count": 40,        "schema": "public"},
        {"table_name": "customer",       "security_level": 1, "row_count": 200_000,   "schema": "public"},
        {"table_name": "orders",         "security_level": 1, "row_count": 2_000_000, "schema": "public"},
        {"table_name": "lineitem",       "security_level": 1, "row_count": 6_500_000, "schema": "public"},
        {"table_name": "part",           "security_level": 1, "row_count": 240_000,   "schema": "public"},
        {"table_name": "supplier",       "security_level": 1, "row_count": 14_000,    "schema": "public"},
        {"table_name": "partsupp",       "security_level": 2, "row_count": 890_000,   "schema": "public"},
        {"table_name": "salary_details", "security_level": 3, "row_count": 5_500,     "schema": "public"},
        {"table_name": "system_catalog", "security_level": 4, "row_count": 1_200,     "schema": "pg_catalog"},
    ]

    table_df = pd.DataFrame(tables)
    table_df.to_csv(TABLE_META_PATH, index=False)

    relations = [
        {"source_table": "region",        "target_table": "nation",          "relation_type": "HAS_NATION"},
        {"source_table": "nation",         "target_table": "customer",        "relation_type": "HAS_CUSTOMER"},
        {"source_table": "customer",       "target_table": "orders",          "relation_type": "PLACES_ORDER"},
        {"source_table": "orders",         "target_table": "lineitem",        "relation_type": "CONTAINS_LINE"},
        {"source_table": "part",           "target_table": "partsupp",        "relation_type": "SUPPLIED_AS"},
        {"source_table": "supplier",       "target_table": "partsupp",        "relation_type": "PROVIDES"},
        {"source_table": "customer",       "target_table": "salary_details",  "relation_type": "HAS_SALARY"},
        {"source_table": "salary_details", "target_table": "system_catalog",  "relation_type": "CATALOG_PROTECT"},
    ]
    pd.DataFrame(relations).to_csv(TABLE_REL_PATH, index=False)
    print(f"  [2/4] table_metadata.csv: {len(tables)} bảng | table_relations.csv: {len(relations)} quan hệ")

    # ------------------------------------------------------------------
    # 3. SINH LOG CHUỖI THỜI GIAN
    # ------------------------------------------------------------------
    all_logs         = []
    all_ground_truth = []
    event_counter    = 1
    start_date_old       = datetime(2026, 5, 1, 0, 0, 0)

    # Bắt đầu dataset mới sau 60 ngày
    start_date = start_date_old + timedelta(days=60)

    # Chỉ user Cấp 1-3 mới có thể trở thành kẻ tấn công (không phải admin)
    potential_attackers = [u for u in users if u["clearance_level"] in (1, 2, 3)]
    # User Cấp 4 (admin) không tham gia hoạt động thường ngày thủ công
    daytime_users = [u for u in users if u["db_user"] != "postgres"]
    admin_user    = next(u for u in users if u["db_user"] == "postgres")

    for day in range(num_days):
        current_day  = start_date + timedelta(days=day)
        date_str     = current_day.strftime("%Y-%m-%d")
        is_weekend   = current_day.weekday() >= 5
        base_commands = random.randint(12,20) if not is_weekend else random.randint(2, 6)  # SỬA: cuối tuần vẫn có hoạt động để tạo context

        for frame in GRAPH_FRAMES:
            is_after_hours = FRAME_AFTER_HOURS[frame]
            # hour           = FRAME_HOUR[frame]
            #====THAY========
            hour, minute, second = random_time_in_frame(frame)

            # ------------------------------------------------------
            # PHA BAN NGÀY (G1-G3): Mọi user hoạt động hợp pháp
            # ------------------------------------------------------
            if is_after_hours == 0 and base_commands > 0:
                for u in daytime_users:
                    #session_id = f"SESS_{u['db_user'].upper()}_{date_str}_{frame}"

                    for step in range(base_commands):
                        evt_id = f"EVT_{event_counter:07d}"
                        # =====Sửa=======
                        # minute = (step * 4 + random.randint(0, 2)) % 60
                        # second = random.randint(10, 59)
                        # v_ts   = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"

                        # THÊM 2 DÒNG======
                        hour, minute, second = random_time_in_frame(frame)
                        v_ts = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
                        #THÊM========
                        normal = _normal_sql(u, step,table_df)
                        # Xác định baseline_hour theo user và frame
                        
                        all_logs.append({
                            "event_id":        evt_id,
                            "timestamp":       v_ts,
                            #"session_id":      session_id,
                            "db_user":         u["db_user"],
                            "sql_statement":   normal["sql"],
                            "tables_involved": normal["table"],
                            "status":          normal["status"],
                            "rows_affected":   normal["rows"],
                            "graph_frame":     frame,
                            "is_after_hours":  0,
                            #"scenario_type":   "Normal",
                            "scenario_type":   normal["scenario_type"],
                        })
                        all_ground_truth.append({"event_id": evt_id, "is_anomaly": 0})
                        event_counter += 1

            # ------------------------------------------------------
            # PHA TẤN CÔNG BAN NGÀY (G1-G3): Slow insider – xác suất thấp
            # Mô phỏng nội gián thực hiện hành vi bất thường ẩn trong giờ làm việc
            # (Slow-Rate, Lateral Movement, Data Staging, Off-Hours bypass)
            # ------------------------------------------------------
            DAY_ATTACK_SCENARIOS = [5, 8, 9, 2]  # chỉ kịch bản ít "ồn" nhất
            if is_after_hours == 0 and base_commands > 0 and random.random() < 0.18:
                day_attacker     = random.choice(potential_attackers)
                day_atk_user     = day_attacker["db_user"]
                day_atk_steps    = random.randint(8, 15)
                day_scenario_id  = random.choice(DAY_ATTACK_SCENARIOS)

                if HAS_ATTACK_MODULE:
                    day_atk_seq = attack.get_attack_sequence(
                        day_atk_user, frame, 0, day_scenario_id, day_atk_steps
                    )
                else:
                    day_atk_seq = [{
                        "sql_statement":   "SELECT * FROM public.orders LIMIT 5000 OFFSET 0;",
                        "tables_involved": "orders",
                        "status":          "SUCCESS",
                        "rows_affected":   5000,
                        "scenario_type":   "Slow_Rate_Exfiltration",
                    }] * day_atk_steps

                for step, atk_data in enumerate(day_atk_seq):
                    evt_id = f"EVT_{event_counter:07d}"
                    hour, minute, second = random_time_in_frame(frame)
                    v_ts = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
                    all_logs.append({
                        "event_id":        evt_id,
                        "timestamp":       v_ts,
                        "db_user":         day_atk_user,
                        "sql_statement":   atk_data["sql_statement"],
                        "tables_involved": atk_data["tables_involved"],
                        "status":          atk_data["status"],
                        "rows_affected":   atk_data["rows_affected"],
                        "graph_frame":     frame,
                        "is_after_hours":  0,
                        "scenario_type":   atk_data.get("scenario_type", "Unknown"),
                    })
                    all_ground_truth.append({"event_id": evt_id, "is_anomaly": 1})
                    event_counter += 1

            # ------------------------------------------------------
            # PHA ADMIN BAN NGÀY (G1, G2, G3): Giám sát & vận hành
            # Admin hoạt động vào G1, G2, G3 với tần suất thấp
            # với tần suất thấp (2-4 lệnh/frame) → baseline thưa, ổn định
            # ------------------------------------------------------
            if is_after_hours == 0 and frame in ("G1", "G2", "G3") and base_commands > 0:
                #session_id   = f"SESS_POSTGRES_{date_str}_{frame}"
                admin_cmds   = random.randint(2, 4)   # ít hơn user thường
 
                for step in range(admin_cmds):
                    evt_id = f"EVT_{event_counter:07d}"
                    # THAY BẰNG 2 DÒNG
                    hour, minute, second = random_time_in_frame(frame)
                    v_ts = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
                    #=======================
                    adm = _admin_sql(step, is_maintenance=False)
                    all_logs.append({
                        "event_id":        evt_id,
                        "timestamp":       v_ts,
                        #"session_id":      session_id,
                        "db_user":         "postgres",
                        "sql_statement":   adm["sql"],
                        "tables_involved": adm["table"],
                        "status":          adm["status"],
                        "rows_affected":   adm["rows"],
                        "graph_frame":     frame,
                        "is_after_hours":  0,
                        #"scenario_type":   "Normal_Admin",
                        "scenario_type":   adm["scenario_type"],
                    })
                    all_ground_truth.append({"event_id": evt_id, "is_anomaly": 0})
                    event_counter += 1
                        
            #===================================================
            # PHA ADMIN BẢO TRÌ NGOÀI GIỜ (G4 only, xác suất 40%):
            # VACUUM, ANALYZE, REINDEX — hợp lệ nhưng cần phân biệt
            # với tấn công bằng ngữ cảnh (scheduled_maintenance flag)
            #=========================================================
            if frame == "G4" and random.random() < 0.40:
                #session_id  = f"SESS_POSTGRES_MAINT_{date_str}_{frame}"
                maint_cmds  = random.randint(2, 5)

                for step in range(maint_cmds):
                    evt_id = f"EVT_{event_counter:07d}"
                    #minute = (step * 10 + random.randint(0, 4)) % 60
                    #second = random.randint(0, 59)
                    #v_ts   = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"

                    #=============THAY 2 DÒNG================
                    hour, minute, second = random_time_in_frame(frame)
                    v_ts = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
                    adm = _admin_sql(step, is_maintenance=True)

                    all_logs.append({
                        "event_id":        evt_id,
                        "timestamp":       v_ts,
                        #"session_id":      session_id,
                        "db_user":         "postgres",
                        "sql_statement":   adm["sql"],
                        "tables_involved": adm["table"],
                        "status":          adm["status"],
                        "rows_affected":   adm["rows"],
                        "graph_frame":     frame,
                        "is_after_hours":  1,
                        #"scenario_type":   "Normal_Admin_Maintenance",
                        "scenario_type":   adm["scenario_type"],
                    })
                    all_ground_truth.append({"event_id": evt_id, "is_anomaly": 0})
                    event_counter += 1
            

            # ------------------------------------------------------
            # PHA BAN ĐÊM (G4-G5): Chọn ngẫu nhiên 1 user Cấp 1-3
            # làm kẻ tấn công đêm nay → Dynamic Insider
            # ------------------------------------------------------
            # SỬA 6: Multi-attacker – chọn 1 hoặc 2 kẻ tấn công mỗi đêm
            if is_after_hours == 1 and random.random() < 0.75:
                # 25% cơ hội có 2 kẻ tấn công đồng thời (coordinated insider)
                num_attackers = 2 if random.random() < 0.25 else 1
                chosen_attackers = random.sample(potential_attackers,
                                                  min(num_attackers, len(potential_attackers)))

                for attacker_meta in chosen_attackers:
                    user_attacker    = attacker_meta["db_user"]
                    num_attack_steps = random.randint(30, 50)

                    if HAS_ATTACK_MODULE:
                        scenario_id     = random.randint(1, 10)
                        attack_sequence = attack.get_attack_sequence(
                            user_attacker, frame, is_after_hours, scenario_id, num_attack_steps
                        )
                    else:
                        attack_sequence = [{
                            "sql_statement":   "SELECT * FROM public.orders LIMIT 450000;",
                            "tables_involved": "orders",
                            "status":          "SUCCESS",
                            "rows_affected":   450_000,
                            "scenario_type":   "Data_Exfiltration",
                        }] * num_attack_steps

                    for step, atk_step_data in enumerate(attack_sequence):
                        evt_id = f"EVT_{event_counter:07d}"
                        hour, minute, second = random_time_in_frame(frame)
                        v_ts = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
                        all_logs.append({
                            "event_id":        evt_id,
                            "timestamp":        v_ts,
                            "db_user":          user_attacker,
                            "sql_statement":    atk_step_data["sql_statement"],
                            "tables_involved":  atk_step_data["tables_involved"],
                            "status":           atk_step_data["status"],
                            "rows_affected":    atk_step_data["rows_affected"],
                            "graph_frame":      frame,
                            "is_after_hours":   is_after_hours,
                            "scenario_type":    atk_step_data.get("scenario_type", "Unknown"),
                        })
                        all_ground_truth.append({"event_id": evt_id, "is_anomaly": 1})
                        event_counter += 1

    # ------------------------------------------------------------------
    # 4. XUẤT FILE
    # ------------------------------------------------------------------
    log_df = pd.DataFrame(all_logs)
    gt_df  = pd.DataFrame(all_ground_truth)

    log_df.to_csv(AUDIT_LOG_PATH,   index=False)
    gt_df.to_csv(GROUND_TRUTH_PATH, index=False)

    # ------------------------------------------------------------------
    # 5. THỐNG KÊ
    # ------------------------------------------------------------------
    total_events  = len(log_df)
    total_anomaly = int(gt_df["is_anomaly"].sum())
    total_normal  = total_events - total_anomaly
    anomaly_pct   = total_anomaly / total_events * 100

    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH v12.0 – Thống kê dataset:")
    print(f"   📋 Tổng số sự kiện     : {total_events:,}")
    print(f"   ✅ Sự kiện bình thường  : {total_normal:,}  ({100 - anomaly_pct:.1f}%)")
    print(f"   🚨 Sự kiện tấn công    : {total_anomaly:,}  ({anomaly_pct:.1f}%)")

    # Phân bố theo graph_frame
    print(f"\n   Phân bố is_anomaly theo graph_frame:")
    for frm in GRAPH_FRAMES:
        frm_df  = log_df[log_df["graph_frame"] == frm]
        frm_gt  = gt_df.loc[frm_df.index]
        n_atk   = int(frm_gt["is_anomaly"].sum())
        n_tot   = len(frm_df)
        pct     = n_atk / n_tot * 100 if n_tot > 0 else 0
        print(f"     {frm}: {n_tot:>6,} events  |  anomaly {n_atk:>5,}  ({pct:.1f}%)")

    print(f"\n   Phân bố scenario_type (bình thường):")
    normal_logs = log_df[log_df["scenario_type"].isin(NORMAL_SCENARIOS)]
    for sc, cnt in normal_logs["scenario_type"].value_counts().items():
        print(f"     • {sc:<40}: {cnt:>5,} sự kiện")

    print(f"\n   Phân bố kịch bản tấn công:")
    attack_logs = log_df[~log_df["scenario_type"].isin(NORMAL_SCENARIOS)]
    for sc, cnt in attack_logs["scenario_type"].value_counts().items():
        print(f"     • {sc:<35}: {cnt:>5,} sự kiện")

    print(f"\n   Phân bố kẻ tấn công theo user (clearance level):")
    for usr, cnt in attack_logs["db_user"].value_counts().items():
        clv = next((u["clearance_level"] for u in users if u["db_user"] == usr), "?")
        pct_u = cnt / total_anomaly * 100
        print(f"     • {usr:<30} (Cấp {clv}): {cnt:>5,} sự kiện  ({pct_u:.1f}%)")

    print(f"\n   Đa dạng SQL (unique statements):")
    print(f"     Normal  : {log_df[log_df['scenario_type'].isin(NORMAL_SCENARIOS)]['sql_statement'].nunique():,} unique")
    print(f"     Attack  : {attack_logs['sql_statement'].nunique():,} unique")

    print("\n   📁 File đã xuất:")
    for path in [USER_META_PATH, TABLE_META_PATH, AUDIT_LOG_PATH, TABLE_REL_PATH, GROUND_TRUTH_PATH]:
        size_kb = os.path.getsize(path) / 1024
        print(f"     → {path:<45} ({size_kb:,.1f} KB)")

    print("=" * 70)
    print("🎯 Dataset Dynamic Insider v12.0 sẵn sàng cho pipeline Inductive GNN!")
    print("=" * 70)


if __name__ == "__main__":
    generate_dynamic_insider_dataset(num_days=15, seed=42)  # SỬA: 60 ngày → nhiều data hơn
