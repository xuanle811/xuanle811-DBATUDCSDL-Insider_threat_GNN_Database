"""
query_builder.py — Sinh SQL normal theo role + scenario + tham số động.

Mục tiêu:
- Sinh log normal theo nghiệp vụ thực tế trong 60 ngày.
- Query bám theo PERMISSION_MATRIX L1–L5.
- Hạn chế query làm bẩn dữ liệu seed.
- Một số role có RW sẽ có no-op UPDATE an toàn để tạo hành vi ghi hợp lệ.
"""

import random
from datetime import date, timedelta

# ============================================================
# QUERY TEMPLATES
# Mỗi role có danh sách (sql_template, tables_involved, is_write)
# tables_involved phải khớp với table_metadata.csv khi có thể.
# ============================================================
CATALOG_TABLE_MAP = {
    "information_schema": "system_catalog",
    "pg_indexes": "system_catalog",
    "pg_stat_user_tables": "system_catalog",
    "pg_tables": "system_catalog",
    "pg_stat_activity": "system_catalog",
    "pg_stat_database": "system_catalog",
    "pg_user": "system_catalog",
    "pg_roles": "system_catalog",
}


QUERY_TEMPLATES = {

    # ========================================================
    # SALES — L1/L2 READ
    # sales_staff: R L1, R L2, X L3-L5
    # ========================================================
    "sales_staff": [
        ("SELECT c_custkey, c_name, c_phone, c_mktsegment "
         "FROM customer WHERE c_nationkey = {nation} LIMIT {lim}",
         ["customer"], False),

        ("SELECT o_orderkey, o_totalprice, o_orderdate, o_orderstatus "
         "FROM orders WHERE o_custkey = {cust} LIMIT {lim}",
         ["orders"], False),

        ("SELECT o_orderstatus, COUNT(*) "
         "FROM orders WHERE o_orderdate BETWEEN '{dt_from}' AND '{dt}' "
         "GROUP BY o_orderstatus",
         ["orders"], False),

        ("SELECT l_orderkey, l_partkey, l_quantity, l_extendedprice "
         "FROM lineitem WHERE l_orderkey = {ord} LIMIT {lim}",
         ["lineitem"], False),

        ("SELECT p_partkey, p_name, p_retailprice "
         "FROM part WHERE p_size = {size} LIMIT {lim}",
         ["part"], False),

        ("SELECT s_suppkey, s_name, s_phone "
         "FROM supplier WHERE s_nationkey = {nation} LIMIT {lim}",
         ["supplier"], False),

        ("SELECT c.c_name, o.o_orderkey, o.o_totalprice "
         "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
         "WHERE c.c_mktsegment = '{seg}' LIMIT {lim}",
         ["customer", "orders"], False),
    ],

    # ========================================================
    # SALES MANAGER — R L1-L4, X L5
    # ========================================================
    "sales_manager": [
        ("SELECT c_name, c_acctbal, c_mktsegment "
         "FROM customer WHERE c_mktsegment = '{seg}' LIMIT {lim}",
         ["customer"], False),

        ("SELECT o_orderkey, o_totalprice, o_orderdate "
         "FROM orders WHERE o_orderdate >= '{dt_from}' LIMIT {lim}",
         ["orders"], False),

        ("SELECT SUM(o_totalprice) AS revenue, o_orderdate "
         "FROM orders WHERE o_orderdate BETWEEN '{dt_from}' AND '{dt}' "
         "GROUP BY o_orderdate ORDER BY o_orderdate DESC LIMIT {lim}",
         ["orders"], False),

        ("SELECT e.full_name, e.job_role, d.department_name "
         "FROM hr.employee e JOIN hr.department d ON e.department_id = d.department_id "
         "WHERE d.department_name = 'Sales Department' LIMIT {lim}",
         ["hr.employee", "hr.department"], False),

        ("SELECT c.c_mktsegment, COUNT(o.o_orderkey), SUM(o.o_totalprice) "
         "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
         "GROUP BY c.c_mktsegment",
         ["customer", "orders"], False),
    ],

    # ========================================================
    # ACCOUNTANT — R L1-L3, R* L4 payroll only, X L5
    # ========================================================
    "accountant": [
        ("SELECT o_orderkey, o_totalprice, o_orderstatus "
         "FROM orders WHERE o_totalprice > {price} LIMIT {lim}",
         ["orders"], False),

        ("SELECT l_suppkey, SUM(l_extendedprice) AS supplier_amount "
         "FROM lineitem WHERE l_shipdate BETWEEN '{dt_from}' AND '{dt}' "
         "GROUP BY l_suppkey LIMIT {lim}",
         ["lineitem"], False),

        ("SELECT s_suppkey, s_name, s_acctbal "
         "FROM supplier WHERE s_acctbal < {bal} LIMIT {lim}",
         ["supplier"], False),

        ("SELECT c_custkey, c_name, c_acctbal "
         "FROM customer WHERE c_acctbal < 0 LIMIT {lim}",
         ["customer"], False),

        ("SELECT ps_partkey, ps_suppkey, ps_supplycost "
         "FROM partsupp WHERE ps_supplycost > {cost} LIMIT {lim}",
         ["partsupp"], False),

        ("SELECT employee_id, net_salary, payroll_month "
         "FROM hr.payroll WHERE payroll_month = '{month}' LIMIT {lim}",
         ["hr.payroll"], False),

        ("SELECT SUM(net_salary) AS total_net, AVG(base_salary) AS avg_base "
         "FROM hr.payroll WHERE payroll_month = '{month}'",
         ["hr.payroll"], False),
    ],

    # ========================================================
    # FINANCE MANAGER — R L1-L4, X L5
    # ========================================================
    "finance_manager": [
        ("SELECT SUM(o_totalprice) AS total, o_orderstatus "
         "FROM orders WHERE o_orderdate BETWEEN '{dt_from}' AND '{dt}' "
         "GROUP BY o_orderstatus",
         ["orders"], False),

        ("SELECT c.c_name, c.c_acctbal, n.n_name "
         "FROM customer c JOIN nation n ON c.c_nationkey = n.n_nationkey "
         "LIMIT {lim}",
         ["customer", "nation"], False),

        ("SELECT e.full_name, e.job_role, p.net_salary "
         "FROM hr.employee e JOIN hr.payroll p ON e.employee_id = p.employee_id "
         "WHERE p.payroll_month = '{month}' LIMIT {lim}",
         ["hr.employee", "hr.payroll"], False),

        ("SELECT SUM(net_salary), COUNT(*) "
         "FROM hr.payroll WHERE payroll_month = '{month}'",
         ["hr.payroll"], False),

        ("SELECT d.department_name, SUM(p.net_salary) AS payroll_total "
         "FROM hr.department d JOIN hr.employee e ON d.department_id = e.department_id "
         "JOIN hr.payroll p ON e.employee_id = p.employee_id "
         "WHERE p.payroll_month = '{month}' "
         "GROUP BY d.department_name",
         ["hr.department", "hr.employee", "hr.payroll"], False),
    ],

    # ========================================================
    # PROCUREMENT — staff R L1/L2, manager R L1-L4
    # ========================================================
    "procurement_staff": [
        ("SELECT p_partkey, p_name, p_retailprice, p_size "
         "FROM part WHERE p_type LIKE '%{ptype}%' LIMIT {lim}",
         ["part"], False),

        ("SELECT n_nationkey, n_name FROM nation LIMIT {lim}",
         ["nation"], False),

        ("SELECT r_regionkey, r_name FROM region",
         ["region"], False),

        ("SELECT s_suppkey, s_name, s_phone, s_acctbal "
         "FROM supplier LIMIT {lim}",
         ["supplier"], False),

        ("SELECT ps_partkey, ps_suppkey, ps_availqty, ps_supplycost "
         "FROM partsupp WHERE ps_availqty > {qty} LIMIT {lim}",
         ["partsupp"], False),
    ],

    "procurement_manager": [
        ("SELECT p_partkey, p_name, p_retailprice "
         "FROM part WHERE p_retailprice < {price} LIMIT {lim}",
         ["part"], False),

        ("SELECT s_suppkey, s_name, s_acctbal "
         "FROM supplier WHERE s_acctbal > {bal} LIMIT {lim}",
         ["supplier"], False),

        ("SELECT ps_partkey, ps_suppkey, ps_supplycost, ps_availqty "
         "FROM partsupp ORDER BY ps_supplycost DESC LIMIT {lim}",
         ["partsupp"], False),

        ("SELECT e.full_name, e.job_role "
         "FROM hr.employee e JOIN hr.department d ON e.department_id = d.department_id "
         "WHERE d.department_name = 'Procurement Department' LIMIT {lim}",
         ["hr.employee", "hr.department"], False),

        ("SELECT s.s_name, p.p_name, ps.ps_supplycost "
         "FROM supplier s JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey "
         "JOIN part p ON ps.ps_partkey = p.p_partkey "
         "ORDER BY ps.ps_supplycost DESC LIMIT {lim}",
         ["supplier", "partsupp", "part"], False),
    ],

    # ========================================================
    # HR — RW L3, R L4, X L5
    # ========================================================
    "hr_staff": [
        ("SELECT employee_id, full_name, job_role, employment_status "
         "FROM hr.employee WHERE employment_status = 'ACTIVE' LIMIT {lim}",
         ["hr.employee"], False),

        ("SELECT e.full_name, e.job_role, d.department_name "
         "FROM hr.employee e JOIN hr.department d ON e.department_id = d.department_id "
         "LIMIT {lim}",
         ["hr.employee", "hr.department"], False),

        # Safe write: không đổi bản chất dữ liệu, chỉ tạo hành vi ghi hợp lệ L3.
        ("UPDATE hr.employee SET employment_status = employment_status WHERE employee_id = {emp}",
         ["hr.employee"], True),

        ("SELECT employee_id, net_salary, payroll_month "
         "FROM hr.payroll WHERE employee_id = {emp} LIMIT {lim}",
         ["hr.payroll"], False),

        ("SELECT job_role, COUNT(*) FROM hr.employee GROUP BY job_role",
         ["hr.employee"], False),
    ],

    "hr_manager": [
        ("SELECT e.full_name, e.job_role, p.base_salary, p.net_salary "
         "FROM hr.employee e JOIN hr.payroll p ON e.employee_id = p.employee_id "
         "WHERE p.payroll_month = '{month}' LIMIT {lim}",
         ["hr.employee", "hr.payroll"], False),

        ("SELECT d.department_name, COUNT(e.employee_id) AS total_employee "
         "FROM hr.department d JOIN hr.employee e ON d.department_id = e.department_id "
         "GROUP BY d.department_name",
         ["hr.department", "hr.employee"], False),

        ("SELECT e.employee_id, e.full_name, e.job_role, e.employment_status, d.department_name "
         "FROM hr.employee e JOIN hr.department d ON e.department_id = d.department_id "
         "WHERE e.employment_status = 'ACTIVE' LIMIT {lim}",
         ["hr.employee", "hr.department"], False),

        # Safe write: không update job_role nữa.
        ("UPDATE hr.employee SET employment_status = employment_status WHERE employee_id = {emp}",
         ["hr.employee"], True),
    ],

    # ========================================================
    # DATA ANALYST — R L1-L3, X L4/L5
    # Không được query hr.payroll.
    # ========================================================
    "data_analyst": [
        ("SELECT c.c_name, SUM(o.o_totalprice) AS total_order "
         "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
         "GROUP BY c.c_name ORDER BY total_order DESC LIMIT {lim}",
         ["customer", "orders"], False),

        ("SELECT p.p_name, SUM(l.l_quantity) AS total_qty "
         "FROM part p JOIN lineitem l ON p.p_partkey = l.l_partkey "
         "WHERE l.l_shipdate BETWEEN '{dt_from}' AND '{dt}' "
         "GROUP BY p.p_name ORDER BY total_qty DESC LIMIT {lim}",
         ["part", "lineitem"], False),

        ("SELECT e.job_role, COUNT(*) "
         "FROM hr.employee e GROUP BY e.job_role",
         ["hr.employee"], False),

        ("EXPLAIN SELECT * FROM orders WHERE o_orderdate BETWEEN '{dt_from}' AND '{dt}'",
         ["orders"], False),

        ("SELECT n.n_name, COUNT(c.c_custkey) "
         "FROM nation n JOIN customer c ON n.n_nationkey = c.c_nationkey "
         "GROUP BY n.n_name",
         ["nation", "customer"], False),
    ],

    "senior_analyst": [
        ("SELECT c.c_name, o.o_totalprice, l.l_quantity, p.p_name "
         "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
         "JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
         "JOIN part p ON l.l_partkey = p.p_partkey "
         "LIMIT {lim}",
         ["customer", "orders", "lineitem", "part"], False),

        ("SELECT e.job_role, SUM(p.net_salary) AS payroll_total "
         "FROM hr.employee e JOIN hr.payroll p ON e.employee_id = p.employee_id "
         "WHERE p.payroll_month = '{month}' "
         "GROUP BY e.job_role",
         ["hr.employee", "hr.payroll"], False),

        ("SELECT r.r_name, n.n_name, COUNT(s.s_suppkey) "
         "FROM region r JOIN nation n ON r.r_regionkey = n.n_regionkey "
         "JOIN supplier s ON n.n_nationkey = s.s_nationkey "
         "GROUP BY r.r_name, n.n_name",
         ["region", "nation", "supplier"], False),

        ("EXPLAIN SELECT c.c_name, o.o_totalprice "
         "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
         "WHERE o.o_orderdate >= '{dt_from}'",
         ["customer", "orders"], False),
    ],

    # ========================================================
    # DEVELOPER / DATA ENGINEER — R L1, RW L2, X L3-L5
    # Normal chủ yếu SELECT/EXPLAIN/catalog; write dùng no-op an toàn L2.
    # ========================================================
    "developer": [
        ("SELECT o_orderkey, o_custkey, o_orderstatus, o_totalprice "
         "FROM orders WHERE o_orderkey = {ord}",
         ["orders"], False),

        ("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
         ["information_schema"], False),

        ("SELECT column_name, data_type "
         "FROM information_schema.columns "
         "WHERE table_schema = 'public' AND table_name = '{tbl}'",
         ["information_schema"], False),

        ("EXPLAIN SELECT * FROM lineitem WHERE l_shipdate >= '{dt_from}'",
         ["lineitem"], False),

        ("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = '{tbl}'",
         ["pg_indexes"], False),

        ("SELECT o.o_orderkey, o.o_totalprice, l.l_quantity "
         "FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
         "WHERE o.o_orderdate >= '{dt_from}' LIMIT {lim}",
         ["orders", "lineitem"], False),

        # Safe write L2: không đổi giá trị.
        ("UPDATE orders SET o_comment = o_comment WHERE o_orderkey = {ord}",
         ["orders"], True),
    ],

    "data_engineer": [
        ("SELECT COUNT(*) FROM {tbl}",
         ["{tbl}"], False),

        ("SELECT column_name, data_type "
         "FROM information_schema.columns WHERE table_schema = 'public' AND table_name = '{tbl}'",
         ["information_schema"], False),

        ("EXPLAIN SELECT l_partkey, SUM(l_quantity) "
         "FROM lineitem WHERE l_shipdate BETWEEN '{dt_from}' AND '{dt}' "
         "GROUP BY l_partkey",
         ["lineitem"], False),

        ("SELECT schemaname, relname, n_live_tup "
         "FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT {lim}",
         ["pg_stat_user_tables"], False),

        ("SELECT o.o_orderdate, COUNT(*) "
         "FROM orders o WHERE o.o_orderdate BETWEEN '{dt_from}' AND '{dt}' "
         "GROUP BY o.o_orderdate ORDER BY o.o_orderdate DESC LIMIT {lim}",
         ["orders"], False),

        # Safe write L2.
        ("UPDATE lineitem SET l_comment = l_comment WHERE l_orderkey = {ord}",
         ["lineitem"], True),
    ],

    # ========================================================
    # SECURITY — R L1-L5, không write L5 trong normal.
    # ========================================================
    "security_analyst": [
        ("SELECT db_user, event_type, severity_index, event_time "
         "FROM sys.system_security_log ORDER BY event_time DESC LIMIT {lim}",
         ["sys.system_security_log"], False),

        ("SELECT db_user, account_status, last_login "
         "FROM sys.db_account LIMIT {lim}",
         ["sys.db_account"], False),

        ("SELECT pid, usename, application_name, state "
         "FROM pg_stat_activity WHERE state = 'active'",
         ["pg_stat_activity"], False),

        ("SELECT e.full_name, a.clearance_level, a.is_privileged "
         "FROM hr.employee e JOIN sys.db_account a ON e.employee_id = a.employee_id "
         "LIMIT {lim}",
         ["hr.employee", "sys.db_account"], False),

        ("SELECT event_type, COUNT(*) "
         "FROM sys.system_security_log GROUP BY event_type",
         ["sys.system_security_log"], False),

        ("SELECT db_user, COUNT(*) "
         "FROM sys.system_security_log "
         "WHERE event_type = 'FAILED_LOGIN' "
         "GROUP BY db_user ORDER BY COUNT(*) DESC LIMIT {lim}",
         ["sys.system_security_log"], False),
    ],

    "security_manager": [
        ("SELECT event_type, severity_index, db_user "
         "FROM sys.system_security_log WHERE severity_index >= 3 "
         "ORDER BY event_time DESC LIMIT {lim}",
         ["sys.system_security_log"], False),

        ("SELECT db_user, is_privileged, account_status "
         "FROM sys.db_account WHERE is_privileged = TRUE",
         ["sys.db_account"], False),

        ("SELECT COUNT(*) AS failed_logins, db_user "
         "FROM sys.system_security_log WHERE event_type = 'FAILED_LOGIN' "
         "GROUP BY db_user ORDER BY failed_logins DESC LIMIT {lim}",
         ["sys.system_security_log"], False),

        ("SELECT a.db_user, a.db_role, a.clearance_level, e.job_role "
         "FROM sys.db_account a JOIN hr.employee e ON a.employee_id = e.employee_id "
         "WHERE a.account_status = 'ACTIVE' LIMIT {lim}",
         ["sys.db_account", "hr.employee"], False),
    ],

    # ========================================================
    # DBA — junior RW L1-L3, R L4/L5. senior RW all.
    # ========================================================
    "junior_dba": [
        ("SELECT schemaname, relname, n_live_tup, n_dead_tup "
         "FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT {lim}",
         ["pg_stat_user_tables"], False),

        ("SELECT db_user, account_status, last_login "
         "FROM sys.db_account LIMIT {lim}",
         ["sys.db_account"], False),

        ("SELECT schemaname, tablename, "
         "pg_size_pretty(pg_total_relation_size(format('%I.%I', schemaname, tablename)::regclass)) AS size "
         "FROM pg_tables WHERE schemaname IN ('public', 'hr', 'sys') "
         "ORDER BY pg_total_relation_size(format('%I.%I', schemaname, tablename)::regclass) DESC LIMIT {lim}",
         ["pg_tables"], False),

        ("SELECT event_time, cpu_utilization, memory_utilization, active_connections "
         "FROM sys.system_monitor_log ORDER BY event_time DESC LIMIT {lim}",
         ["sys.system_monitor_log"], False),

        ("SELECT COUNT(*) FROM {tbl}",
        ["{tbl}"], False),
    ],

    "senior_dba": [
        ("SELECT pid, usename, application_name, state, query_start "
         "FROM pg_stat_activity",
         ["pg_stat_activity"], False),

        ("SELECT * FROM sys.system_security_log ORDER BY event_time DESC LIMIT {lim}",
         ["sys.system_security_log"], False),

        ("SELECT db_user, clearance_level, is_privileged, account_status "
         "FROM sys.db_account LIMIT {lim}",
         ["sys.db_account"], False),

        ("SELECT relname, n_live_tup, n_dead_tup "
         "FROM pg_stat_user_tables ORDER BY n_dead_tup DESC LIMIT {lim}",
         ["pg_stat_user_tables"], False),

        ("SELECT datname, numbackends, xact_commit, xact_rollback "
         "FROM pg_stat_database LIMIT {lim}",
         ["pg_stat_database"], False),

        # Safe write L5, allowed by senior_dba RW L5.
        ("UPDATE sys.db_account SET account_status = account_status WHERE db_user = '{dbu}'",
         ["sys.db_account"], True),
    ],

    "admin": [
        ("SELECT usename, usesuper FROM pg_user",
         ["pg_user"], False),

        ("SELECT pid, usename, application_name, state, wait_event "
         "FROM pg_stat_activity WHERE wait_event IS NOT NULL",
         ["pg_stat_activity"], False),

        ("SELECT rolname, rolsuper FROM pg_roles LIMIT {lim}",
         ["pg_roles"], False),

        ("SELECT db_user, db_role, is_privileged "
         "FROM sys.db_account WHERE is_privileged = TRUE",
         ["sys.db_account"], False),

        ("SELECT event_type, severity_index, db_user, event_time "
         "FROM sys.system_security_log ORDER BY event_time DESC LIMIT {lim}",
         ["sys.system_security_log"], False),

        ("SELECT schemaname, relname, n_live_tup, n_dead_tup "
         "FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT {lim}",
         ["pg_stat_user_tables"], False),
    ],
}

# ============================================================
# PARAM HELPERS
# ============================================================

TPCH_TABLES = ["orders", "lineitem", "customer", "part", "supplier", "partsupp"]

# Một số db_user seed thực tế để DBA/security query có rows trả về.
SAMPLE_DB_USERS = (
    [f"sales_staff_{i:03d}" for i in range(1, 21)] +
    [f"sales_manager_{i:03d}" for i in range(21, 24)] +
    [f"accountant_{i:03d}" for i in range(24, 34)] +
    [f"finance_manager_{i:03d}" for i in range(34, 36)] +
    [f"procurement_staff_{i:03d}" for i in range(36, 44)] +
    [f"procurement_manager_{i:03d}" for i in range(44, 46)] +
    [f"hr_staff_{i:03d}" for i in range(46, 52)] +
    ["hr_manager_052"] +
    [f"data_analyst_{i:03d}" for i in range(53, 58)] +
    [f"senior_analyst_{i:03d}" for i in range(58, 60)] +
    [f"developer_{i:03d}" for i in range(60, 68)] +
    [f"data_engineer_{i:03d}" for i in range(68, 71)] +
    [f"security_analyst_{i:03d}" for i in range(71, 73)] +
    ["security_manager_073"] +
    [f"junior_dba_{i:03d}" for i in range(74, 77)] +
    [f"senior_dba_{i:03d}" for i in range(77, 79)] +
    ["admin_079"]
)

# Edit because date in TPCH 1992-1998
TPCH_DATE_START = date(1992, 1, 1)
TPCH_DATE_END   = date(1998, 12, 31)

PAYROLL_MONTHS = [
    "2025-12-01",
    "2026-01-01",
    "2026-02-01",
]

def random_tpch_date_range():
    max_days = (TPCH_DATE_END - TPCH_DATE_START).days
    start_offset = random.randint(0, max_days - 60)

    d1 = TPCH_DATE_START + timedelta(days=start_offset)
    d2 = d1 + timedelta(days=random.randint(1, 60))

    return d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")

def random_params(sim_date: date, sample_db_users: list = None):
    tpch_from, tpch_to = random_tpch_date_range()
    # Dùng list từ DB nếu có, fallback về SAMPLE_DB_USERS hardcode
    dbu_pool = sample_db_users if sample_db_users else SAMPLE_DB_USERS

    return {
        

        "nation": random.randint(0, 24),
        "cust":   random.randint(1, 75000),
        "ord":    random.randint(1, 1500000),
        "size":   random.randint(1, 50),
        "price":  random.randint(1000, 100000),
        "bal":    random.randint(-500, 5000),
        "cost":   random.randint(10, 1000),
        "qty":    random.randint(1, 9999),
        "lim":    random.choice([10, 20, 25, 50]),

        # Dùng cho điều kiện WHERE trên bảng TPC-H
        "dt":      tpch_to,
        "dt_from": tpch_from,
        "dt2":     tpch_to,

        # Dùng cho hr.payroll
        "month": random.choice(PAYROLL_MONTHS),

        "emp":    random.randint(1, 79),
        "alert":  random.randint(1, 20),
        "seg":    random.choice(["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]),
        "ptype":  random.choice(["BRASS", "COPPER", "NICKEL", "STEEL", "TIN"]),
        "role":   random.choice(["sales_staff", "accountant", "developer", "hr_staff"]),
        "tbl":    random.choice(TPCH_TABLES),
        "dbu": random.choice(dbu_pool),
        "status": random.choice(["ACTIVE", "LOCKED"]),
    }


def build_query(job_role: str, sim_date: date,
                sample_db_users: list = None,):
    """
    Trả về (sql, tables_involved, is_write).
Được session_manager gọi trực tiếp để sinh query và resolve table/catalog.
    """
    templates = QUERY_TEMPLATES.get(job_role, QUERY_TEMPLATES["sales_staff"])
    params    = random_params(sim_date, sample_db_users)  # truyền vào đây

    tmpl, tables, is_write = random.choice(templates)

    try:
        sql = tmpl.format(**params)
    except KeyError:
        sql = tmpl

    # tables_resolved = []
    # for t in tables:
    #     try:
    #         tables_resolved.append(t.format(**params))
    #     except KeyError:
    #         tables_resolved.append(t)

    tables_resolved = []
    for t in tables:
        try:
            tt = t.format(**params)
        except KeyError:
            tt = t

        tt = CATALOG_TABLE_MAP.get(tt, tt)
        tables_resolved.append(tt)

    return sql, tables_resolved, is_write



# Thêm system_catalog,pg_catalog,4,0,PostgreSQL system catalog and metadata views vào cuối file table_metadata.csv
