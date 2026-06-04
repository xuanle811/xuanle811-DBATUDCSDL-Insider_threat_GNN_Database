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


def _normal_sql(user_meta: dict, step: int) -> dict:
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
        rows = random.randint(5, 80)

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
        rows = random.randint(10, 120)

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
        rows = random.randint(10, 100)

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

    # Nhân viên bình thường – Sales & Finance
    for i in range(1, 20):
        users.append({
            "db_user": f"normal_user_{i}",
            "department": "sales",
            "clearance_level": 1,
            "role": "staff",
            "is_privileged": 0,
        })

    # Lập trình viên / DBA junior
    for i in range(1, 4):
        users.append({
            "db_user": f"dev_user_{i}",
            "department": "it_engineering",
            "clearance_level": 2,
            "role": "developer",
            "is_privileged": 0,
        })

    for i in range(1, 3):
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
def _build_kill_chain_schedule(potential_attackers, num_days, seed,
                                num_fixed_attackers=6,
                                normal_ratio=0.60,
                                cooldown_days=10):
    """
    Lập lịch kill chain cho tất cả campaign trong num_days ngày.

    Thiết kế theo CERT v4.2:
      - Chọn CỐ ĐỊNH 3 attacker từ pool (không đổi suốt 60 ngày)
      - Mỗi attacker có thể thực hiện NHIỀU campaign (cooldown >= 10 ngày) (trước hết là 1 attacker - 1 scenario trước)
        - Cả 3 attacker đều đi qua cùng timeline phase:
            Day 36-40 : recon
            Day 41-50 : preparation
            Day 51-55 : execution
            Day 56-60 : cover
        - Đảm bảo snapshot cuối T11 có đủ 3 anomaly users.

      - Attack xảy ra ưu tiên ngoài giờ (G4/G5) nhưng có thể vào ban ngày
        với xác suất thấp hơn (slow insider)

    Returns:
        schedule: dict[db_user] → list of CampaignEntry
        CampaignEntry = {
            "scenario_id": int,
            "phase_days":  dict[day_index] → phase_name (recon/preparation/execution/cover)
        }
        Ngày không có key trong phase_days = ngày bình thường (is_anomaly=0)
        Ngày có key = ngày tấn công (is_anomaly=1)
    """
    rng = random.Random(seed)

    # Chọn cố định 3 attacker
    fixed_attackers = rng.sample(potential_attackers,
                                 min(num_fixed_attackers, len(potential_attackers)))

    schedule = {u["db_user"]: [] for u in fixed_attackers}

    for attacker in fixed_attackers:
        uname        = attacker["db_user"]
        last_end_day = -cooldown_days   # cho phép bắt đầu sớm

        while True:
            # Khoảng cách tối thiểu giữa 2 campaign
            earliest_start = last_end_day + cooldown_days
            if earliest_start >= num_days:
                break

            # normal_ratio ngày đầu là bình thường → campaign bắt đầu muộn
            normal_days   = int(num_days * normal_ratio)
            # campaign_start = rng.randint(
            #     max(earliest_start, normal_days),
            #     min(num_days - 8, num_days - 1)   # cần ít nhất 4 phase × 1-2 ngày
            # )
            # tính ngày bắt đầu campaign đảm bảo snapshot cuối có nhãn của 3 user
            min_attack_days = 4 * 5  # 4 phase, mỗi phase tối thiểu 5 ngày

            campaign_start = rng.randint(
                max(earliest_start, normal_days),
                min(num_days - min_attack_days, normal_days + 2)
            )
            if campaign_start >= num_days - 4:
                break

            scenario_id = rng.choice([1, 2, 3, 5, 9, 10])

            # Phân bổ ngày cho từng phase (mỗi phase 1-3 ngày)
            from attack import KILL_CHAIN_PHASES
            phase_days = {}
            cursor = campaign_start
            for phase in KILL_CHAIN_PHASES:
                duration = rng.randint(5,7)
                for d in range(cursor, min(cursor + duration, num_days)):
                    phase_days[d] = phase
                cursor += duration
                if cursor >= num_days:
                    break

            campaign_end = max(phase_days.keys()) if phase_days else campaign_start
            schedule[uname].append({
                "scenario_id": scenario_id,
                "phase_days":  phase_days,
            })
            last_end_day = campaign_end

            # Xác suất có thêm campaign nữa (~50%)
            # if rng.random() > 0.5:
            #     break
            # tạm thời break
            break

    return fixed_attackers, schedule


def generate_dynamic_insider_dataset(num_days: int = 60, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("🚀 PIPELINE v13.0 – KILL CHAIN CERT v4.2 – Dynamic Insider Threat")
    print(f"   Thời gian mô phỏng : {num_days} ngày | 5 graph frames/ngày")
    print(f"   Kill chain         : NORMAL(~75%) → RECON → PREP → EXEC → COVER")
    print(f"   Attackers          : 3 user cố định, mỗi người có thể tái phạm")
    print(f"   Nhãn is_anomaly    : từ phase RECON trở đi = 1")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. DANH MỤC NGƯỜI DÙNG
    # ------------------------------------------------------------------
    users_df = build_user_metadata()
    users    = users_df.to_dict(orient="records")
    print(f"  [1/4] user_metadata.csv : {len(users)} người dùng")
    # ------------------------------------------------------------------
    # 2. DANH MỤC BẢNG
    # ------------------------------------------------------------------
    tables = [
        {"table_name": "region",        "security_level": 1, "row_count": 5,         "schema": "public"},
        {"table_name": "nation",         "security_level": 1, "row_count": 25,        "schema": "public"},
        {"table_name": "customer",       "security_level": 1, "row_count": 150_000,   "schema": "public"},
        {"table_name": "orders",         "security_level": 1, "row_count": 1_500_000, "schema": "public"},
        {"table_name": "lineitem",       "security_level": 1, "row_count": 6_000_000, "schema": "public"},
        {"table_name": "part",           "security_level": 1, "row_count": 200_000,   "schema": "public"},
        {"table_name": "supplier",       "security_level": 1, "row_count": 10_000,    "schema": "public"},
        {"table_name": "partsupp",       "security_level": 2, "row_count": 800_000,   "schema": "public"},
        {"table_name": "salary_details", "security_level": 3, "row_count": 5_000,     "schema": "public"},
        {"table_name": "system_catalog", "security_level": 4, "row_count": 1_200,     "schema": "pg_catalog"},
    ]
    pd.DataFrame(tables).to_csv(TABLE_META_PATH, index=False)

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
    all_logs              = []
    all_ground_truth      = []          # event-level
    all_user_attack_keys  = set()       # (db_user, date_str, graph_frame) → anomaly
    all_user_phase_keys   = {}          # (db_user, date_str, graph_frame) → phase name
    event_counter         = 1
    start_date            = datetime(2025, 5, 1, 0, 0, 0)

    potential_attackers = [u for u in users if u["clearance_level"] in (1, 2, 3)]
    daytime_users       = [u for u in users if u["db_user"] != "postgres"]
    admin_user          = next(u for u in users if u["db_user"] == "postgres")

    # ── Lập lịch kill chain cho toàn bộ 60 ngày ──────────────────────
    fixed_attackers, kill_chain_schedule = _build_kill_chain_schedule(
        potential_attackers, num_days, seed=seed)
    attacker_names = {u["db_user"] for u in fixed_attackers}

    print(f"\n  [Kill Chain]Các attacker cố định:")
    for u in fixed_attackers:
        campaigns = kill_chain_schedule[u["db_user"]]
        for ci, c in enumerate(campaigns):
            days_sorted = sorted(c["phase_days"].keys())
            phases_str  = " → ".join(sorted(set(c["phase_days"].values()),
                                            key=list(c["phase_days"].values()).index))
            print(f"    • {u['db_user']:<25} campaign {ci+1}: "
                  f"ngày {days_sorted[0]+1}–{days_sorted[-1]+1}  "
                  f"scenario={c['scenario_id']}  [{phases_str}]")

    # ── Duyệt từng ngày × frame ───────────────────────────────────────
    for day in range(num_days):
        current_day  = start_date + timedelta(days=day)
        date_str     = current_day.strftime("%Y-%m-%d")
        is_weekend   = current_day.weekday() >= 5
        base_commands = random.randint(12, 20) if not is_weekend else random.randint(2, 6)

        for frame in GRAPH_FRAMES:
            is_after_hours = FRAME_AFTER_HOURS[frame]

            # ── Hành vi bình thường tất cả user (kể cả attacker) ─────
            if is_after_hours == 0 and base_commands > 0:
                for u in daytime_users:
                    for step in range(base_commands):
                        evt_id = f"EVT_{event_counter:07d}"
                        hour, minute, second = random_time_in_frame(frame)
                        v_ts   = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
                        normal = _normal_sql(u, step)
                        all_logs.append({
                            "event_id":        evt_id,
                            "timestamp":       v_ts,
                            "db_user":         u["db_user"],
                            "sql_statement":   normal["sql"],
                            "tables_involved": normal["table"],
                            "status":          normal["status"],
                            "rows_affected":   normal["rows"],
                            "graph_frame":     frame,
                            "is_after_hours":  0,
                            "scenario_type":   normal["scenario_type"],
                            "attack_phase":    "normal",
                        })
                        all_ground_truth.append({"event_id": evt_id, "is_anomaly": 0})
                        event_counter += 1

            # ── Admin ban ngày ────────────────────────────────────────
            if is_after_hours == 0 and frame in ("G1", "G2", "G3") and base_commands > 0:
                for step in range(random.randint(2, 4)):
                    evt_id = f"EVT_{event_counter:07d}"
                    hour, minute, second = random_time_in_frame(frame)
                    v_ts   = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
                    adm    = _admin_sql(step, is_maintenance=False)
                    all_logs.append({
                        "event_id":        evt_id,
                        "timestamp":       v_ts,
                        "db_user":         "postgres",
                        "sql_statement":   adm["sql"],
                        "tables_involved": adm["table"],
                        "status":          adm["status"],
                        "rows_affected":   adm["rows"],
                        "graph_frame":     frame,
                        "is_after_hours":  0,
                        "scenario_type":   adm["scenario_type"],
                        "attack_phase":    "normal",
                    })
                    all_ground_truth.append({"event_id": evt_id, "is_anomaly": 0})
                    event_counter += 1

            # ── Admin bảo trì ngoài giờ (G4, 40%) ────────────────────
            if frame in ("G4", "G5"):
                maintenance_prob = 0.40 if frame == "G4" else 0.25
                if random.random() < maintenance_prob:
                    for step in range(random.randint(2, 5)):
                        evt_id = f"EVT_{event_counter:07d}"
                        h, m, s = random_time_in_frame(frame)
                        v_ts = f"{date_str} {h:02d}:{m:02d}:{s:02d}"
                        adm  = _admin_sql(step, is_maintenance=True)
                        all_logs.append({
                            "event_id":        evt_id, "timestamp":       v_ts,
                            "db_user":         "postgres",
                            "sql_statement":   adm["sql"], "tables_involved": adm["table"],
                            "status":          adm["status"], "rows_affected": adm["rows"],
                            "graph_frame":     frame, "is_after_hours": 1,
                            "scenario_type":   adm["scenario_type"], "attack_phase": "normal",
                        })
                        all_ground_truth.append({"event_id": evt_id, "is_anomaly": 0})
                        event_counter += 1

            # ── KILL CHAIN: inject attack phase cho từng attacker ─────
            for attacker in fixed_attackers:
                uname     = attacker["db_user"]
                campaigns = kill_chain_schedule[uname]

                for campaign in campaigns:
                    phase = campaign["phase_days"].get(day)
                    if phase is None:
                        continue   # ngày này user hoàn toàn bình thường

                    sid = campaign["scenario_id"]

                    # Slow phases (recon, preparation) ưu tiên ban ngày
                    # để ẩn trong traffic bình thường → khó phát hiện hơn
                    # Execution / Cover ưu tiên ngoài giờ
                    if phase in ("recon", "preparation"):
                        target_frames = ["G1", "G2", "G3"]
                        if frame not in target_frames:
                            continue
                    else:  # execution, cover
                        target_frames = ["G4", "G5"]
                        if frame not in target_frames:
                            continue

                    steps = random.randint(5, 12)
                    atk_records = attack.get_phase_sequence(
                        uname, frame, is_after_hours, sid, phase, steps)

                    for rec in atk_records:
                        evt_id = f"EVT_{event_counter:07d}"
                        hour, minute, second = random_time_in_frame(frame)
                        v_ts   = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
                        all_logs.append({
                            "event_id":        evt_id,
                            "timestamp":       v_ts,
                            "db_user":         rec["db_user"],
                            "sql_statement":   rec["sql_statement"],
                            "tables_involved": rec["tables_involved"],
                            "status":          rec["status"],
                            "rows_affected":   rec["rows_affected"],
                            "graph_frame":     frame,
                            "is_after_hours":  is_after_hours,
                            "scenario_type":   rec["scenario_type"],
                            "attack_phase":    rec["attack_phase"],
                        })
                        all_ground_truth.append({"event_id": evt_id, "is_anomaly": 1})
                        # Đánh dấu (user, date, frame) → anomaly + phase
                        key = (uname, date_str, frame)
                        all_user_attack_keys.add(key)
                        all_user_phase_keys[key] = phase
                        event_counter += 1

    # ------------------------------------------------------------------
    # 4. XUẤT FILE
    # ------------------------------------------------------------------
    log_df = pd.DataFrame(all_logs)
    gt_df  = pd.DataFrame(all_ground_truth)

    # Build nhãn USER-LEVEL: (db_user, date, graph_frame) → is_anomaly + phase
    user_frame_rows = []
    for u in users:
        for day_i in range(num_days):
            date_str_i = (start_date + timedelta(days=day_i)).strftime("%Y-%m-%d")
            for frm in GRAPH_FRAMES:
                key       = (u["db_user"], date_str_i, frm)
                is_anom   = 1 if key in all_user_attack_keys else 0
                phase_val = all_user_phase_keys.get(key, "normal")
                user_frame_rows.append({
                    "db_user":      u["db_user"],
                    "date":         date_str_i,
                    "graph_frame":  frm,
                    "is_anomaly":   is_anom,
                    "attack_phase": phase_val,
                    # is_attacker = 1 nếu user này là 1 trong 3 attacker cố định
                    "is_attacker":  1 if u["db_user"] in attacker_names else 0,
                })
    user_gt_df = pd.DataFrame(user_frame_rows)

    GROUND_TRUTH_USER_PATH = os.path.join(DATA_DIR, "anomaly_ground_truth_user.csv")

    log_df.to_csv(AUDIT_LOG_PATH,          index=False)
    gt_df.to_csv(GROUND_TRUTH_PATH,        index=False)
    user_gt_df.to_csv(GROUND_TRUTH_USER_PATH, index=False)

    # ------------------------------------------------------------------
    # 5. THỐNG KÊ
    # ------------------------------------------------------------------
    total_events  = len(log_df)
    total_anomaly = int(gt_df["is_anomaly"].sum())
    total_normal  = total_events - total_anomaly
    anomaly_pct   = total_anomaly / total_events * 100

    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH v13.0 – Kill Chain CERT v4.2:")
    print(f"   📋 Tổng số sự kiện       : {total_events:,}")
    print(f"   ✅ Sự kiện bình thường    : {total_normal:,}  ({100 - anomaly_pct:.1f}%)")
    print(f"   🚨 Sự kiện tấn công      : {total_anomaly:,}  ({anomaly_pct:.1f}%)")

    # Nhãn user-level
    n_uf        = len(user_gt_df)
    n_uf_anom   = int(user_gt_df["is_anomaly"].sum())
    uf_pct      = n_uf_anom / n_uf * 100
    print(f"\n   📊 Nhãn USER-LEVEL (user × date × frame):")
    print(f"     Tổng bản ghi   : {n_uf:,}")
    print(f"     Bình thường    : {n_uf - n_uf_anom:,}  ({100-uf_pct:.1f}%)")
    print(f"     Bất thường     : {n_uf_anom:,}  ({uf_pct:.1f}%)")
    print(f"     3 attacker cố định: {', '.join(sorted(attacker_names))}")

    # Phân bố phase
    print(f"\n   Phân bố attack_phase trong user-level labels:")
    for ph, cnt in user_gt_df[user_gt_df["is_anomaly"]==1]["attack_phase"].value_counts().items():
        print(f"     • {ph:<15}: {cnt:>5,} (user×frame)")

    # Phân bố theo graph_frame
    print(f"\n   Phân bố event-level theo graph_frame:")
    for frm in GRAPH_FRAMES:
        frm_df = log_df[log_df["graph_frame"] == frm]
        frm_gt = gt_df.loc[frm_df.index]
        n_atk  = int(frm_gt["is_anomaly"].sum())
        n_tot  = len(frm_df)
        pct    = n_atk / n_tot * 100 if n_tot > 0 else 0
        u_anom = int(user_gt_df[(user_gt_df["graph_frame"]==frm) &
                                 (user_gt_df["is_anomaly"]==1)].shape[0])
        print(f"     {frm}: {n_tot:>6,} events | event-anom {n_atk:>5,} ({pct:.1f}%) "
              f"| user-anom {u_anom:>4,}")

    # Timeline tấn công từng user
    print(f"\n   Timeline kill chain (ngày bắt đầu anomaly trên mỗi attacker):")
    for uname in sorted(attacker_names):
        anom_days = sorted(user_gt_df[(user_gt_df["db_user"]==uname) &
                                      (user_gt_df["is_anomaly"]==1)]["date"].unique())
        phases_by_day = (user_gt_df[(user_gt_df["db_user"]==uname) &
                                     (user_gt_df["is_anomaly"]==1)]
                         .groupby("date")["attack_phase"]
                         .first().to_dict())
        if anom_days:
            timeline = "  ".join(
                f"{d}({phases_by_day.get(d,'?')[:4]})" for d in anom_days[:8])
            print(f"     • {uname:<25}: {timeline}"
                  + (" ..." if len(anom_days) > 8 else ""))

    print("\n   📁 File đã xuất:")
    for path in [USER_META_PATH, TABLE_META_PATH, AUDIT_LOG_PATH,
                 TABLE_REL_PATH, GROUND_TRUTH_PATH, GROUND_TRUTH_USER_PATH]:
        size_kb = os.path.getsize(path) / 1024
        print(f"     → {path:<50} ({size_kb:,.1f} KB)")

    print("=" * 70)
    print("🎯 Dataset v13.0 Kill Chain sẵn sàng cho TSA trên user sequence!")
    print("=" * 70)


if __name__ == "__main__":
    generate_dynamic_insider_dataset(num_days=60, seed=42)
