"""
attack_queries.py — SQL templates cho 4 attack scenarios chính.

Trả về list tuple:
    (sql, tables_involved, is_write, description)

Lưu ý:
- Quyền SUCCESS/DENIED không quyết định tại đây, mà do attack_runner ép theo role/table.
- Các write nguy hiểm được kiểm soát bằng SAVEPOINT/rollback trong attack_runner.
- S4 staging được phép persist tạm thời để mô phỏng staging rồi DROP ở cover.
"""
import random
from datetime import date, timedelta

TPCH_START = date(1992, 1, 1)
TPCH_END = date(1998, 12, 31)


def random_tpch_date() -> str:
    delta = (TPCH_END - TPCH_START).days
    return (TPCH_START + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def random_payroll_month() -> str:
    return random.choice(["2025-12-01", "2026-01-01", "2026-02-01"])


def _lim(small: bool = False) -> int:
    return random.randint(5, 20) if small else random.randint(50, 500)


# ============================================================
# S1 — Unauthorized Data Exploration
# ============================================================
def s1_queries(phase: str, job_role: str) -> list[tuple[str, list[str], bool, str]]:
    """S1: thăm dò ngoài phạm vi; có một số probe bị DENIED."""
    queries: list[tuple[str, list[str], bool, str]] = []

    if phase == "recon":
        queries.extend([
            (f"SELECT o_orderkey, o_totalprice FROM orders WHERE o_orderdate >= '{random_tpch_date()}' LIMIT {_lim(small=True)}",
             ["orders"], False, "recon_normal"),
            (f"SELECT c_name, c_acctbal FROM customer LIMIT {_lim(small=True)}",
             ["customer"], False, "recon_normal"),
            (f"SELECT p_name, p_retailprice FROM part LIMIT {_lim(small=True)}",
             ["part"], False, "recon_normal"),
        ])
        queries.extend(random.sample([
            ("SELECT * FROM hr.employee LIMIT 5",
             ["hr.employee"], False, "recon_probe"),
            ("SELECT * FROM hr.payroll LIMIT 5",
             ["hr.payroll"], False, "recon_probe"),
            ("SELECT * FROM sys.db_account LIMIT 5",
             ["sys.db_account"], False, "recon_probe"),
        ], k=random.randint(1, 2)))

    elif phase == "execution":
        queries.extend([
            ("SELECT COUNT(*) FROM hr.employee",
             ["hr.employee"], False, "exec_probe"),
            (f"SELECT full_name, job_role FROM hr.employee LIMIT {_lim()}",
             ["hr.employee"], False, "exec_read"),
            (f"SELECT employee_id, net_salary FROM hr.payroll WHERE payroll_month = '{random_payroll_month()}' LIMIT {_lim()}",
             ["hr.payroll"], False, "exec_read"),
            (f"SELECT * FROM sys.system_security_log ORDER BY event_time DESC LIMIT {_lim(small=True)}",
             ["sys.system_security_log"], False, "exec_probe"),
            ("SELECT table_name, table_schema FROM information_schema.tables "
             "WHERE table_schema NOT IN ('pg_catalog','information_schema')",
             ["information_schema"], False, "exec_schema_map"),
        ])

    return queries


# ============================================================
# S2 — Sensitive Data Theft
# ============================================================
def s2_queries(phase: str, job_role: str) -> list[tuple[str, list[str], bool, str]]:
    queries: list[tuple[str, list[str], bool, str]] = []

    if phase == "recon":
        queries.extend([
            ("SELECT COUNT(*) FROM hr.payroll",
             ["hr.payroll"], False, "recon_count"),
            ("SELECT column_name FROM information_schema.columns WHERE table_schema='hr' AND table_name='payroll'",
             ["information_schema"], False, "recon_schema"),
            ("SELECT COUNT(*) FROM sys.system_security_log",
             ["sys.system_security_log"], False, "recon_count"),
            (f"SELECT e.full_name, e.job_role FROM hr.employee e LIMIT {_lim(small=True)}",
             ["hr.employee"], False, "recon_normal"),
        ])

    elif phase == "preparation":
        queries.extend([
            (f"EXPLAIN SELECT * FROM hr.payroll WHERE payroll_month = '{random_payroll_month()}'",
             ["hr.payroll"], False, "prep_explain"),
            ("SELECT MIN(payroll_month), MAX(payroll_month) FROM hr.payroll",
             ["hr.payroll"], False, "prep_range"),
            ("SELECT e.job_role, COUNT(*), AVG(p.net_salary) "
             "FROM hr.payroll p JOIN hr.employee e ON p.employee_id = e.employee_id "
             "GROUP BY e.job_role",
             ["hr.payroll", "hr.employee"], False, "prep_aggregate"),
            ("SELECT db_user, account_status FROM sys.db_account WHERE account_status = 'ACTIVE' LIMIT 20",
             ["sys.db_account"], False, "prep_privilege_recon"),
        ])

    elif phase == "execution":
        queries.extend([
            (f"SELECT * FROM hr.payroll WHERE payroll_month = '{random_payroll_month()}'",
             ["hr.payroll"], False, "exec_bulk_read"),
            ("SELECT e.full_name, e.job_role, p.net_salary, p.base_salary "
             "FROM hr.employee e JOIN hr.payroll p ON e.employee_id = p.employee_id",
             ["hr.employee", "hr.payroll"], False, "exec_bulk_join"),
            ("SELECT * FROM sys.system_security_log ORDER BY event_time DESC",
             ["sys.system_security_log"], False, "exec_bulk_read"),
            ("SELECT * FROM hr.payroll ORDER BY net_salary DESC",
             ["hr.payroll"], False, "exec_bulk_read"),
            ("SELECT * FROM sys.db_account",
             ["sys.db_account"], False, "exec_bulk_read"),
        ])

    elif phase == "cover":
        # Không dùng DISCARD ALL vì PostgreSQL không cho chạy trong transaction block.
        queries.extend([
            (f"SELECT o_orderkey FROM orders LIMIT {_lim(small=True)}",
             ["orders"], False, "cover_normal"),
            (f"SELECT c_name FROM customer LIMIT {_lim(small=True)}",
             ["customer"], False, "cover_normal"),
            ("RESET ALL",
             ["system_catalog"], False, "cover_reset_session"),
        ])

    return queries


# ============================================================
# S3 — Slow-rate Data Theft
# ============================================================
def s3_queries(phase: str, job_role: str, day_offset: int) -> list[tuple[str, list[str], bool, str]]:
    """Mỗi ngày chỉ 1–3 query nhỏ, không DENIED, không spike rõ."""
    queries: list[tuple[str, list[str], bool, str]] = []

    emp_start = ((max(day_offset, 66) - 66) * 3) % 77 + 1
    month_idx = (max(day_offset, 66) - 66) % 3

    if phase == "recon":
        queries.extend([
            ("SELECT COUNT(*) FROM hr.employee",
             ["hr.employee"], False, "recon_light"),
            ("SELECT job_role, COUNT(*) FROM hr.employee GROUP BY job_role",
             ["hr.employee"], False, "recon_light"),
        ])

    elif phase == "execution":
        payroll_months = ["2025-12-01", "2026-01-01", "2026-02-01"]
        month = payroll_months[month_idx]
        queries.extend([
            (f"SELECT employee_id, net_salary FROM hr.payroll "
             f"WHERE payroll_month = '{month}' AND employee_id BETWEEN {emp_start} AND {emp_start + 2}",
             ["hr.payroll"], False, "exec_slow_batch"),
            (f"SELECT full_name, job_role FROM hr.employee "
             f"WHERE employee_id BETWEEN {emp_start} AND {emp_start + 2}",
             ["hr.employee"], False, "exec_slow_batch"),
        ])
        if random.random() < 0.30:
            queries.append(
                (f"SELECT c_name, c_acctbal FROM customer WHERE c_custkey % 50 = {day_offset % 50} LIMIT 10",
                 ["customer"], False, "exec_slow_batch")
            )

    return queries


# ============================================================
# S4 — Data Staging & Aggregation
# ============================================================
def s4_queries(phase: str, job_role: str) -> list[tuple[str, list[str], bool, str]]:
    """S4 dùng customer/orders để hợp với developer/data_engineer/senior_dba."""
    queries: list[tuple[str, list[str], bool, str]] = []

    if phase == "recon":
        queries.extend([
            ("SELECT table_name, table_schema FROM information_schema.tables "
             "WHERE table_schema NOT IN ('pg_catalog','information_schema')",
             ["information_schema"], False, "recon_schema_map"),
            ("SELECT COUNT(*) FROM customer",
             ["customer"], False, "recon_count"),
            ("SELECT COUNT(*) FROM orders",
             ["orders"], False, "recon_count"),
            ("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'orders'",
             ["information_schema"], False, "recon_schema"),
        ])

    elif phase == "preparation":
        queries.extend([
            ("CREATE TABLE IF NOT EXISTS public.staging_data "
             "(c_custkey INT, c_name TEXT, o_orderkey INT, o_totalprice NUMERIC, o_orderdate DATE)",
             ["public.staging_data"], True, "prep_create_staging"),
            ("INSERT INTO public.staging_data "
             "SELECT c.c_custkey, c.c_name, o.o_orderkey, o.o_totalprice, o.o_orderdate "
             "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
             "LIMIT 50",
             ["public.staging_data", "customer", "orders"], True, "prep_test_insert"),
            ("SELECT COUNT(*) FROM public.staging_data",
             ["public.staging_data"], False, "prep_verify"),
        ])

    elif phase == "execution":
        tpch_date = random_tpch_date()
        queries.extend([
            ("INSERT INTO public.staging_data "
             "SELECT c.c_custkey, c.c_name, o.o_orderkey, o.o_totalprice, o.o_orderdate "
             "FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey "
             f"WHERE o.o_orderdate >= '{tpch_date}' LIMIT 5000",
             ["public.staging_data", "customer", "orders"], True, "exec_stage_insert"),
            ("SELECT COUNT(*) FROM public.staging_data",
             ["public.staging_data"], False, "exec_verify"),
            (f"SELECT * FROM public.staging_data ORDER BY o_totalprice DESC LIMIT {_lim()}",
             ["public.staging_data"], False, "exec_read_staged"),
            ("SELECT SUM(o_totalprice), AVG(o_totalprice) FROM public.staging_data",
             ["public.staging_data"], False, "exec_aggregate"),
        ])

    elif phase == "cover":
        queries.extend([
            ("DROP TABLE IF EXISTS public.staging_data",
             ["public.staging_data"], True, "cover_drop_staging"),
            ("SELECT COUNT(*) FROM orders",
             ["orders"], False, "cover_normal"),
        ])

    return queries


# ============================================================
# PUBLIC
# ============================================================
def get_attack_queries(
    scenario: str,
    phase: str,
    job_role: str,
    day_offset: int = 0,
) -> list[tuple[str, list[str], bool, str]]:
    """Trả về list of (sql, tables_involved, is_write, description)."""
    if scenario == "S1":
        return s1_queries(phase, job_role)
    if scenario == "S2":
        return s2_queries(phase, job_role)
    if scenario == "S3":
        return s3_queries(phase, job_role, day_offset)
    if scenario == "S4":
        return s4_queries(phase, job_role)
    return []
