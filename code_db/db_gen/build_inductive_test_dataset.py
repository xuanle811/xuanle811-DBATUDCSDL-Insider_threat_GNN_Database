#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_inductive_test_dataset.py (v1.0)
=======================================
Sinh tập TEST INDUCTIVE 15 ngày (ngày 61–75) cho pipeline GNN.

Khác biệt so với tập train (60 ngày):
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. THÊM 5 USER MỚI (node chưa từng xuất hiện trong train)      │
  │    → Test khả năng inductive của SAGE                           │
  │                                                                 │
  │ 2. ROW COUNT CÁC BẢNG TĂNG (~10–25%)                           │
  │    → Phân phối dữ liệu thay đổi, test covariate shift          │
  │                                                                 │
  │ 3. 3 ATTACKER THEO KILL CHAIN CERT v4.2:                        │
  │    A) 1 attacker cũ (đã tấn công trong train) tái phạm         │
  │    B) 1 normal user cũ (train) bị "turn" thành insider          │
  │    C) 1 nhân viên mới (hoàn toàn chưa có trong train)          │
  └─────────────────────────────────────────────────────────────────┘

Output (raw_inductive/):
  • user_metadata_test.csv          — user train + 5 user mới
  • table_metadata_test.csv         — row count tăng
  • audit_log_test.csv              — log 15 ngày
  • anomaly_ground_truth_test.csv   — event-level label
  • anomaly_ground_truth_user_test.csv — USER-LEVEL label theo kill chain
                                        (db_user, date, graph_frame, is_anomaly,
                                         attack_phase, attacker_type)
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

try:
    import attack
    HAS_ATTACK_MODULE = True
    print("✅ Đã tải attack.py")
except ImportError:
    HAS_ATTACK_MODULE = False
    print("⚠️  Không tìm thấy attack.py, dùng fallback.")

# ─── Đường dẫn ────────────────────────────────────────────────────────────────
TRAIN_DATA_DIR  = "raw"
TEST_DATA_DIR   = "raw_inductive"
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# ─── Graph frames (giữ nguyên schema) ────────────────────────────────────────
GRAPH_FRAMES      = ["G1", "G2", "G3", "G4", "G5"]
FRAME_AFTER_HOURS = {"G1": 0, "G2": 0, "G3": 0, "G4": 1, "G5": 1}
FRAME_HOUR        = {
    "G1": (8,  11), "G2": (11, 14),
    "G3": (14, 18), "G4": (18,  2), "G5": (2, 8),
}
NORMAL_SCENARIOS  = ["Normal", "Normal_Admin", "Normal_Admin_Maintenance", "Daily-Report"]

# ─── Cấu hình inductive test ──────────────────────────────────────────────────
TEST_START_DATE   = datetime(2025, 7,  1)   # ngày 61 (train kết thúc 30/5 + 60 ngày)
TEST_NUM_DAYS     = 15
SEED              = 99                       # seed khác train để tránh rò rỉ


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def random_time_in_frame(frame: str):
    start_h, end_h = FRAME_HOUR[frame]
    valid_hours = (list(range(start_h, 24)) + list(range(0, end_h))
                   if start_h >= end_h else list(range(start_h, end_h)))
    return random.choice(valid_hours), random.randint(0, 59), random.randint(0, 59)


LV1_TABLES = ["customer", "orders", "lineitem", "region", "nation", "part", "supplier"]
LV2_TABLES = LV1_TABLES + ["partsupp"]
LV3_TABLES = LV2_TABLES + ["salary_details"]


def _normal_sql(user_meta: dict, step: int) -> dict:
    """Giữ nguyên logic normal SQL từ train (copy để standalone)."""
    clv  = user_meta["clearance_level"]
    scen = random.choice(NORMAL_SCENARIOS)

    if clv == 1:
        tbl = random.choice(LV1_TABLES); t = tbl[:2]
        templates = [
            f"SELECT * FROM public.{tbl} WHERE {t}_key = {random.randint(1,5000)};",
            f"SELECT COUNT(*) FROM public.{tbl};",
            f"SELECT * FROM public.{tbl} ORDER BY 1 LIMIT {random.randint(10,50)};",
            f"SELECT c.c_name, o.o_totalprice FROM public.customer c "
            f"JOIN public.orders o ON c.c_custkey=o.o_custkey LIMIT {random.randint(10,40)};",
            f"SELECT * FROM public.{tbl} WHERE {t}_key IN "
            f"(SELECT {t}_key FROM public.{tbl} ORDER BY 1 LIMIT 50);",
        ]
        rows = random.randint(1, 100)
    elif clv == 2:
        tbl = random.choice(LV2_TABLES); t = tbl[:2]
        templates = [
            f"SELECT * FROM public.{tbl} LIMIT {random.randint(50,500)};",
            f"EXPLAIN ANALYZE SELECT * FROM public.{tbl} WHERE {t}_key > 0;",
            f"SELECT table_name, pg_size_pretty(pg_total_relation_size(table_name::regclass)) "
            f"FROM information_schema.tables WHERE table_schema='public';",
            f"INSERT INTO public.{tbl} SELECT * FROM public.{tbl} WHERE 1=0;",
            f"UPDATE public.{tbl} SET {t}_comment='reviewed' WHERE {t}_key={random.randint(1,1000)};",
        ]
        rows = random.randint(1, 500)
    elif clv == 3:
        tbl = random.choice(LV3_TABLES); t = tbl[:2]
        templates = [
            f"SELECT * FROM public.salary_details LIMIT {random.randint(10,50)};",
            f"SELECT COUNT(*), AVG(salary) FROM public.salary_details;",
            f"UPDATE public.salary_details SET salary=salary WHERE db_user='{user_meta['db_user']}';",
            f"SELECT * FROM public.{tbl} WHERE {t}_key > 0 LIMIT 20;",
        ]
        rows = random.randint(1, 50)
    else:
        tbl = "system_catalog"
        templates = [
            "SELECT pid, usename, state FROM pg_stat_activity;",
            "VACUUM ANALYZE;",
            "SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) "
            "FROM pg_statio_user_tables ORDER BY 2 DESC LIMIT 5;",
        ]
        rows = random.randint(0, 20)
        scen = "Normal_Admin"

    return {
        "sql":           random.choice(templates),
        "table":         tbl,
        "rows":          rows,
        "status":        "SUCCESS",
        "scenario_type": scen,
    }


def _admin_sql(step: int, is_maintenance: bool) -> dict:
    if is_maintenance:
        pool = [
            ("VACUUM ANALYZE public.orders;",    "orders",         0,  "Normal_Admin_Maintenance"),
            ("REINDEX TABLE public.lineitem;",   "lineitem",       0,  "Normal_Admin_Maintenance"),
            ("CHECKPOINT;",                      "system_catalog", 0,  "Normal_Admin_Maintenance"),
            ("SELECT pg_stat_reset();",          "system_catalog", 0,  "Normal_Admin_Maintenance"),
            ("ANALYZE public.customer;",         "customer",       0,  "Normal_Admin_Maintenance"),
        ]
    else:
        pool = [
            ("SELECT pid, usename, state FROM pg_stat_activity WHERE state='active';",
             "system_catalog", random.randint(0,15), "Normal_Admin"),
            ("SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) "
             "FROM pg_statio_user_tables ORDER BY 2 DESC LIMIT 10;",
             "system_catalog", 10, "Normal_Admin"),
            ("SHOW max_connections;", "system_catalog", 1, "Normal_Admin"),
        ]
    sql, tbl, rows, scen = random.choice(pool)
    return {"sql": sql, "table": tbl, "rows": rows, "status": "SUCCESS", "scenario_type": scen}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD USER LIST
# ══════════════════════════════════════════════════════════════════════════════

def build_test_users() -> tuple:
    """
    Trả về (all_users, train_users, new_users).
    all_users = train_users (25 người) + new_users (5 người mới).
    """
    # ── Đọc lại user train nếu file tồn tại, không thì build lại ──
    train_path = os.path.join(TRAIN_DATA_DIR, "user_metadata.csv")
    if os.path.exists(train_path):
        train_df   = pd.read_csv(train_path)
        train_users = train_df.to_dict(orient="records")
    else:
        # Fallback: tái tạo đúng danh sách train
        train_users = []
        for i in range(1, 20):
            train_users.append({
                "db_user": f"normal_user_{i}", "department": "sales",
                "clearance_level": 1, "role": "staff", "is_privileged": 0,
            })
        for i in range(1, 4):
            train_users.append({
                "db_user": f"dev_user_{i}", "department": "it_engineering",
                "clearance_level": 2, "role": "developer", "is_privileged": 0,
            })
        for i in range(1, 3):
            train_users.append({
                "db_user": f"hr_specialist_{i}", "department": "human_resources",
                "clearance_level": 3, "role": "hr_specialist", "is_privileged": 0,
            })
        train_users.append({
            "db_user": "postgres", "department": "infrastructure",
            "clearance_level": 4, "role": "admin", "is_privileged": 1,
        })

    # ── 5 user hoàn toàn mới (chưa có trong train) ────────────────
    new_users = [
        # 2 nhân viên mới phòng sales
        {"db_user": "new_sales_1", "department": "sales",
         "clearance_level": 1, "role": "staff",        "is_privileged": 0},
        {"db_user": "new_sales_2", "department": "sales",
         "clearance_level": 1, "role": "staff",        "is_privileged": 0},
        # 1 dev mới
        {"db_user": "new_dev_1",   "department": "it_engineering",
         "clearance_level": 2, "role": "developer",    "is_privileged": 0},
        # 1 HR mới
        {"db_user": "new_hr_1",    "department": "human_resources",
         "clearance_level": 3, "role": "hr_specialist","is_privileged": 0},
        # 1 analyst mới (Cấp 2)
        {"db_user": "new_analyst_1","department": "data_analytics",
         "clearance_level": 2, "role": "analyst",      "is_privileged": 0},
    ]

    all_users = train_users + new_users
    return all_users, train_users, new_users


# ══════════════════════════════════════════════════════════════════════════════
# BUILD TABLE METADATA (row count tăng 10–25%)
# ══════════════════════════════════════════════════════════════════════════════

def build_test_tables() -> list:
    """
    Row count tăng ~10–25% so với train để mô phỏng data tăng trưởng.
    Điều này gây covariate shift cho feature node → test calibration embedding.
    """
    return [
        {"table_name": "region",        "security_level": 1, "schema": "public",
         "row_count": 5},         # vùng địa lý — không tăng
        {"table_name": "nation",        "security_level": 1, "schema": "public",
         "row_count": 25},
        {"table_name": "customer",      "security_level": 1, "schema": "public",
         "row_count": 172_000},   # +14.7% (train: 150k)
        {"table_name": "orders",        "security_level": 1, "schema": "public",
         "row_count": 1_720_000}, # +14.7%
        {"table_name": "lineitem",      "security_level": 1, "schema": "public",
         "row_count": 6_900_000}, # +15%
        {"table_name": "part",          "security_level": 1, "schema": "public",
         "row_count": 238_000},   # +19%
        {"table_name": "supplier",      "security_level": 1, "schema": "public",
         "row_count": 11_500},    # +15%
        {"table_name": "partsupp",      "security_level": 2, "schema": "public",
         "row_count": 945_000},   # +18%
        {"table_name": "salary_details","security_level": 3, "schema": "public",
         "row_count": 5_850},     # +17% (có nhân viên mới)
        {"table_name": "system_catalog","security_level": 4, "schema": "pg_catalog",
         "row_count": 1_380},     # +15%
    ]


# ══════════════════════════════════════════════════════════════════════════════
# KILL CHAIN SCHEDULER CHO TEST
# ══════════════════════════════════════════════════════════════════════════════

def build_test_attack_schedule(train_users: list, new_users: list,
                                num_days: int, seed: int) -> dict:
    """
    Lập lịch kill chain cho 3 loại attacker đặc trưng của inductive test:

    Loại A — RECIDIVIST (attacker cũ tái phạm):
        Là 1 trong 3 attacker cố định của train.
        Trong test: tiếp tục hoặc bắt đầu campaign mới.
        Mục đích: test khả năng nhớ pattern của model (đã thấy user này).

    Loại B — TURNED INSIDER (normal user cũ chuyển xấu):
        Là user bình thường trong 60 ngày train, bây giờ mới bắt đầu tấn công.
        Quan trọng nhất cho inductive: model phải phát hiện được
        dù trước đây user này luôn bình thường.
        → Normal ratio cao hơn (80%) → bắt đầu tấn công muộn hơn.

    Loại C — NEW EMPLOYEE TURNED ATTACKER (nhân viên mới):
        Hoàn toàn mới, model chưa từng thấy node này.
        Test true inductive: không có lịch sử → chỉ dựa vào hành vi hiện tại.
        → Tấn công sớm hơn (ngày 5–10) với execution mạnh.

    Returns:
        schedule: dict[db_user] → list of campaign dict
        attacker_info: dict[db_user] → attacker_type ("recidivist"/"turned"/"new_employee")
    """
    rng = random.Random(seed)

    # ── Xác định TRAIN ATTACKERS từ file nếu có ───────────────────
    train_attacker_path = os.path.join(TRAIN_DATA_DIR, "anomaly_ground_truth_user.csv")
    if os.path.exists(train_attacker_path):
        gt_df = pd.read_csv(train_attacker_path)
        train_attacker_names = set(
            gt_df[gt_df["is_attacker"] == 1]["db_user"].unique()
        )
    else:
        # Fallback: dùng seed 42 để tái tạo danh sách train attackers
        _potential = [u for u in train_users if u["clearance_level"] in (1, 2, 3)]
        _rng42     = random.Random(42)
        train_attacker_names = {
            u["db_user"] for u in _rng42.sample(
                _potential, min(3, len(_potential))
            )
        }

    # ── Loại A: Recidivist — chọn 1 trong train attackers ─────────
    recidivist_candidates = [u for u in train_users
                             if u["db_user"] in train_attacker_names]
    recidivist = rng.choice(recidivist_candidates)

    # ── Loại B: Turned insider — normal user cũ, KHÔNG phải attacker ─
    turned_candidates = [u for u in train_users
                         if u["db_user"] not in train_attacker_names
                         and u["clearance_level"] in (1, 2, 3)
                         and u["db_user"] != "postgres"]
    turned = rng.choice(turned_candidates)

    # ── Loại C: New employee attacker ─────────────────────────────
    new_employee_candidates = [u for u in new_users if u["clearance_level"] in (1, 2, 3)]
    new_attacker = rng.choice(new_employee_candidates)

    attacker_info = {
        recidivist["db_user"]:    "recidivist",
        turned["db_user"]:        "turned_insider",
        new_attacker["db_user"]:  "new_employee",
    }

    print(f"\n  [Kill Chain Test] 3 attacker:")
    print(f"    A) Recidivist      : {recidivist['db_user']:<25} (đã tấn công trong train)")
    print(f"    B) Turned Insider  : {turned['db_user']:<25} (normal user cũ → insider)")
    print(f"    C) New Employee    : {new_attacker['db_user']:<25} (nhân viên mới → attacker)")

    # ── Lập lịch cho từng attacker ────────────────────────────────
    schedule = {}

    # A — Recidivist: bắt đầu tương đối sớm, 1 campaign rõ ràng
    #     normal_ratio thấp hơn (0.4) vì đây là tái phạm → ít do dự hơn
    schedule[recidivist["db_user"]] = _schedule_one_attacker(
        rng, num_days, normal_ratio=0.35, scenario_ids=[1, 9, 5],
        phase_duration=(1, 3), label="recidivist"
    )

    # B — Turned insider: normal_ratio cao (0.55) → tấn công bắt đầu muộn
    #     Scenario đa dạng hơn, bao gồm sabotage và privilege escalation
    schedule[turned["db_user"]] = _schedule_one_attacker(
        rng, num_days, normal_ratio=0.55, scenario_ids=[2, 3, 10],
        phase_duration=(1, 2), label="turned_insider"
    )

    # C — New employee: bắt đầu sớm (ngày 4–8), execution mạnh
    #     Model chỉ thấy vài ngày bình thường rồi tấn công → khó nhất
    schedule[new_attacker["db_user"]] = _schedule_one_attacker(
        rng, num_days, normal_ratio=0.25, scenario_ids=[1, 9],
        phase_duration=(1, 2), label="new_employee"
    )

    return schedule, attacker_info, train_attacker_names


def _schedule_one_attacker(rng, num_days, normal_ratio, scenario_ids,
                            phase_duration, label) -> list:
    """Sinh danh sách campaign cho 1 attacker."""
    from attack import KILL_CHAIN_PHASES
    campaigns     = []
    normal_days   = max(1, int(num_days * normal_ratio))
    campaign_start = rng.randint(normal_days, max(normal_days, num_days - 6))

    if campaign_start >= num_days - 3:
        # Không đủ ngày → campaign ngắn ép vào cuối
        campaign_start = max(0, num_days - 5)

    sid        = rng.choice(scenario_ids)
    phase_days = {}
    cursor     = campaign_start

    for phase in KILL_CHAIN_PHASES:
        dur = rng.randint(*phase_duration)
        for d in range(cursor, min(cursor + dur, num_days)):
            phase_days[d] = phase
        cursor += dur
        if cursor >= num_days:
            break

    if phase_days:
        campaigns.append({"scenario_id": sid, "phase_days": phase_days})

    return campaigns


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_inductive_test_dataset(num_days: int = TEST_NUM_DAYS,
                                    seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("🚀 INDUCTIVE TEST DATASET v1.0 — 15 ngày kế tiếp (ngày 61–75)")
    print(f"   Bắt đầu từ   : {TEST_START_DATE.strftime('%Y-%m-%d')}")
    print(f"   Số ngày       : {num_days}")
    print(f"   Thêm users    : 5 user mới (new_sales_1/2, new_dev_1, new_hr_1, new_analyst_1)")
    print(f"   Row count     : tăng 10–25% so với train (covariate shift)")
    print(f"   Kill chain    : Recidivist | Turned Insider | New Employee")
    print("=" * 70)

    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # ── 1. Users ──────────────────────────────────────────────────────────
    all_users, train_users, new_users = build_test_users()
    pd.DataFrame(all_users).to_csv(
        os.path.join(TEST_DATA_DIR, "user_metadata_test.csv"), index=False)
    print(f"\n  [1/4] user_metadata_test.csv: {len(all_users)} users "
          f"({len(train_users)} cũ + {len(new_users)} mới)")

    # ── 2. Tables ─────────────────────────────────────────────────────────
    tables = build_test_tables()
    pd.DataFrame(tables).to_csv(
        os.path.join(TEST_DATA_DIR, "table_metadata_test.csv"), index=False)

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
    pd.DataFrame(relations).to_csv(
        os.path.join(TEST_DATA_DIR, "table_relations_test.csv"), index=False)
    print(f"  [2/4] table_metadata_test.csv: {len(tables)} bảng | row count tăng 10–25%")

    # ── 3. Attack schedule ────────────────────────────────────────────────
    schedule, attacker_info, train_attacker_names = build_test_attack_schedule(
        train_users, new_users, num_days, seed)
    attacker_names = set(attacker_info.keys())

    # ── 4. Sinh log ───────────────────────────────────────────────────────
    all_logs             = []
    all_ground_truth     = []
    all_user_attack_keys = set()
    all_user_phase_keys  = {}
    event_counter        = 1

    daytime_users = [u for u in all_users if u["db_user"] != "postgres"]
    admin_user    = next(u for u in all_users if u["db_user"] == "postgres")

    print(f"\n  [3/4] Sinh log 15 ngày × 5 frames × {len(all_users)} users ...")

    for day in range(num_days):
        current_day   = TEST_START_DATE + timedelta(days=day)
        date_str      = current_day.strftime("%Y-%m-%d")
        is_weekend    = current_day.weekday() >= 5
        base_commands = random.randint(12, 20) if not is_weekend else random.randint(2, 6)

        for frame in GRAPH_FRAMES:
            is_after_hours = FRAME_AFTER_HOURS[frame]

            # ── Hành vi bình thường tất cả user ───────────────────────
            if is_after_hours == 0 and base_commands > 0:
                for u in daytime_users:
                    for step in range(base_commands):
                        evt_id = f"EVT_{event_counter:07d}"
                        h, m, s = random_time_in_frame(frame)
                        v_ts = f"{date_str} {h:02d}:{m:02d}:{s:02d}"
                        norm = _normal_sql(u, step)
                        all_logs.append({
                            "event_id":        evt_id,
                            "timestamp":       v_ts,
                            "db_user":         u["db_user"],
                            "sql_statement":   norm["sql"],
                            "tables_involved": norm["table"],
                            "status":          norm["status"],
                            "rows_affected":   norm["rows"],
                            "graph_frame":     frame,
                            "is_after_hours":  0,
                            "scenario_type":   norm["scenario_type"],
                            "attack_phase":    "normal",
                        })
                        all_ground_truth.append({"event_id": evt_id, "is_anomaly": 0})
                        event_counter += 1

            # ── Admin ban ngày ─────────────────────────────────────────
            if is_after_hours == 0 and frame in ("G1", "G2", "G3") and base_commands > 0:
                for step in range(random.randint(2, 4)):
                    evt_id = f"EVT_{event_counter:07d}"
                    h, m, s = random_time_in_frame(frame)
                    v_ts = f"{date_str} {h:02d}:{m:02d}:{s:02d}"
                    adm  = _admin_sql(step, is_maintenance=False)
                    all_logs.append({
                        "event_id":        evt_id, "timestamp":       v_ts,
                        "db_user":         "postgres",
                        "sql_statement":   adm["sql"], "tables_involved": adm["table"],
                        "status":          adm["status"], "rows_affected": adm["rows"],
                        "graph_frame":     frame, "is_after_hours": 0,
                        "scenario_type":   adm["scenario_type"], "attack_phase": "normal",
                    })
                    all_ground_truth.append({"event_id": evt_id, "is_anomaly": 0})
                    event_counter += 1

            # ──  Admin bảo trì ngoài giờ G4/G5 để tránh G5 toàn anomaly────────────────────────────────
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

            # ── Kill chain inject ──────────────────────────────────────
            for uname, campaigns in schedule.items():
                for campaign in campaigns:
                    phase = campaign["phase_days"].get(day)
                    if phase is None:
                        continue

                    sid = campaign["scenario_id"]

                    # Recon/preparation ẩn vào ban ngày
                    # Execution/cover ngoài giờ
                    if phase in ("recon", "preparation"):
                        if frame not in ("G1", "G2", "G3"):
                            continue
                    else:
                        if frame not in ("G4", "G5"):
                            continue

                    steps = random.randint(5, 12)
                    if HAS_ATTACK_MODULE:
                        atk_records = attack.get_phase_sequence(
                            uname, frame, is_after_hours, sid, phase, steps)
                    else:
                        atk_records = [{
                            "db_user": uname,
                            "sql_statement":   f"SELECT * FROM public.orders LIMIT 10000;",
                            "tables_involved": "orders",
                            "status":          "SUCCESS",
                            "rows_affected":   10000,
                            "scenario_type":   "Data_Exfiltration",
                            "attack_phase":    phase,
                        }] * steps

                    for rec in atk_records:
                        evt_id = f"EVT_{event_counter:07d}"
                        h, m, s = random_time_in_frame(frame)
                        v_ts = f"{date_str} {h:02d}:{m:02d}:{s:02d}"
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
                        key = (uname, date_str, frame)
                        all_user_attack_keys.add(key)
                        all_user_phase_keys[key] = phase
                        event_counter += 1

    # ── 5. Xuất file ──────────────────────────────────────────────────────
    log_df = pd.DataFrame(all_logs)
    gt_df  = pd.DataFrame(all_ground_truth)

    # User-level ground truth với attacker_type
    user_frame_rows = []
    for u in all_users:
        for day_i in range(num_days):
            date_str_i = (TEST_START_DATE + timedelta(days=day_i)).strftime("%Y-%m-%d")
            for frm in GRAPH_FRAMES:
                key     = (u["db_user"], date_str_i, frm)
                is_anom = 1 if key in all_user_attack_keys else 0
                phase_v = all_user_phase_keys.get(key, "normal")
                user_frame_rows.append({
                    "db_user":       u["db_user"],
                    "date":          date_str_i,
                    "graph_frame":   frm,
                    "is_anomaly":    is_anom,
                    "attack_phase":  phase_v,
                    # is_attacker: 1 nếu là 1 trong 3 attacker test
                    "is_attacker":   1 if u["db_user"] in attacker_names else 0,
                    # attacker_type: recidivist / turned_insider / new_employee / normal
                    "attacker_type": attacker_info.get(u["db_user"], "normal"),
                    # is_new_user: 1 nếu user chưa có trong train
                    "is_new_user":   1 if u["db_user"] in {nu["db_user"] for nu in new_users} else 0,
                })
    user_gt_df = pd.DataFrame(user_frame_rows)

    log_path      = os.path.join(TEST_DATA_DIR, "audit_log_test.csv")
    gt_path       = os.path.join(TEST_DATA_DIR, "anomaly_ground_truth_test.csv")
    user_gt_path  = os.path.join(TEST_DATA_DIR, "anomaly_ground_truth_user_test.csv")

    log_df.to_csv(log_path,     index=False)
    gt_df.to_csv(gt_path,       index=False)
    user_gt_df.to_csv(user_gt_path, index=False)

    # ── 6. Thống kê ───────────────────────────────────────────────────────
    total_events  = len(log_df)
    total_anomaly = int(gt_df["is_anomaly"].sum())
    total_normal  = total_events - total_anomaly
    anomaly_pct   = total_anomaly / total_events * 100

    n_uf       = len(user_gt_df)
    n_uf_anom  = int(user_gt_df["is_anomaly"].sum())
    uf_pct     = n_uf_anom / n_uf * 100

    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH — Inductive Test Dataset v1.0:")
    print(f"   📋 Tổng sự kiện    : {total_events:,}")
    print(f"   ✅ Bình thường      : {total_normal:,}  ({100-anomaly_pct:.1f}%)")
    print(f"   🚨 Tấn công        : {total_anomaly:,}  ({anomaly_pct:.1f}%)")
    print(f"\n   📊 Nhãn USER-LEVEL:")
    print(f"     Tổng bản ghi    : {n_uf:,}")
    print(f"     Bình thường     : {n_uf - n_uf_anom:,}  ({100-uf_pct:.1f}%)")
    print(f"     Bất thường      : {n_uf_anom:,}  ({uf_pct:.1f}%)")

    print(f"\n   🎯 Chi tiết 3 attacker:")
    for uname, atype in attacker_info.items():
        anom_rows  = user_gt_df[(user_gt_df["db_user"]==uname) &
                                 (user_gt_df["is_anomaly"]==1)]
        n_anom     = len(anom_rows)
        anom_days  = sorted(anom_rows["date"].unique())
        is_new     = uname in {nu["db_user"] for nu in new_users}
        phases     = anom_rows["attack_phase"].value_counts().to_dict()
        print(f"     [{atype:<16}] {uname:<22} | {'NEW USER' if is_new else 'TRAIN USER'}")
        print(f"       Frames bất thường: {n_anom}  |  Ngày: {anom_days}")
        print(f"       Phases: {phases}")

    print(f"\n   Phân bố phase trong user-level anomaly:")
    for ph, cnt in user_gt_df[user_gt_df["is_anomaly"]==1]["attack_phase"].value_counts().items():
        print(f"     • {ph:<15}: {cnt:>4,}")

    print(f"\n   Phân bố theo frame (event-level):")
    for frm in GRAPH_FRAMES:
        frm_df = log_df[log_df["graph_frame"] == frm]
        frm_gt = gt_df.loc[frm_df.index]
        n_tot  = len(frm_df)
        n_atk  = int(frm_gt["is_anomaly"].sum())
        pct    = n_atk / n_tot * 100 if n_tot else 0
        u_anom = int(user_gt_df[(user_gt_df["graph_frame"]==frm) &
                                 (user_gt_df["is_anomaly"]==1)].shape[0])
        print(f"     {frm}: {n_tot:>6,} events | event-anom {n_atk:>5,} ({pct:.1f}%) "
              f"| user-anom {u_anom:>4,}")

    print(f"\n   📁 File đã xuất (thư mục: {TEST_DATA_DIR}/):")
    for path in [os.path.join(TEST_DATA_DIR, f)
                 for f in ["user_metadata_test.csv", "table_metadata_test.csv",
                            "table_relations_test.csv", "audit_log_test.csv",
                            "anomaly_ground_truth_test.csv",
                            "anomaly_ground_truth_user_test.csv"]]:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"     → {os.path.basename(path):<45} ({size_kb:,.1f} KB)")

    print("=" * 70)
    print("🎯 Inductive test dataset sẵn sàng cho a02 → a03 → a04 phase 2!")
    print("=" * 70)


if __name__ == "__main__":
    generate_inductive_test_dataset(num_days=TEST_NUM_DAYS, seed=SEED)
