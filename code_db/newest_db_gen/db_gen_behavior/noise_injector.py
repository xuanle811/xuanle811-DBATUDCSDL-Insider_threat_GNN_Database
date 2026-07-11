"""
noise_injector.py — WFH/permission denied/negligent insider noise.
Tách từ normal_behavior.py, giữ nguyên logic gốc.
"""

import random
import uuid
from config import fake, random_time_in_frame, get_graph_frame
from datetime import timedelta

# ============================================================
# ROLE-AWARE DENIED TARGETS
# ============================================================

DENIED_TARGETS_BY_ROLE = {
    # Nhân viên nghiệp vụ: thỉnh thoảng click nhầm HR/system
    "sales_staff": [
        "hr.employee",
        "hr.payroll",
        "sys.system_monitor_log",
        "sys.system_security_log",
    ],
    "procurement_staff": [
        "hr.employee",
        "hr.payroll",
        "sys.system_security_log",
    ],

    # Accountant có quyền payroll, nên không nên denied payroll.
    # Chủ yếu denied khi đụng system/security.
    "accountant": [
        "sys.db_account",
        "sys.system_security_log",
        "sys.system_monitor_log",
    ],
    "finance_manager": [
        "sys.db_account",
        "sys.system_security_log",
    ],

    # HR có quyền employee/payroll, nên denied chủ yếu ở L5/system.
    "hr_staff": [
        "sys.db_account",
        "sys.system_security_log",
    ],
    "hr_manager": [
        "sys.db_account",
        "sys.system_security_log",
    ],

    # Analyst thường query nhầm payroll/system.
    "data_analyst": [
        "hr.payroll",
        "sys.system_security_log",
        "sys.db_account",
    ],
    "senior_analyst": [
        "sys.db_account",
        "sys.system_security_log",
    ],

    # Developer/data engineer dễ query nhầm HR/system.
    "developer": [
        "hr.employee",
        "hr.payroll",
        "sys.db_account",
        "sys.system_security_log",
    ],
    "data_engineer": [
        "hr.employee",
        "hr.payroll",
        "sys.db_account",
        "sys.system_security_log",
    ],

    # Security có quyền đọc L5, nhưng có thể bị denied khi thử WRITE.
    "security_analyst": [
        "sys.db_account",
        "sys.system_security_log",
    ],
    "security_manager": [
        "sys.db_account",
        "sys.system_security_log",
    ],

    # Junior DBA đọc được L5 nhưng không nên write L5.
    "junior_dba": [
        "sys.db_account",
        "sys.system_security_log",
    ],

    # senior_dba/admin gần như không có unauthorized normal.
    # Nếu bị gọi thì function sẽ bỏ qua.
    "senior_dba": [],
    "admin": [],
}


# ============================================================
# SQL TEMPLATE CHO DENIED NORMAL NOISE
# ============================================================

def _build_denied_sql(table: str, job_role: str) -> str:
    """
    Sinh SQL DENIED đa dạng hơn.
    Không execute thật, chỉ ghi synthetic audit event.
    """

    # Một số role có quyền đọc nhưng không có quyền ghi L5/system.
    write_attempt_roles = {
        "junior_dba",
        "security_analyst",
        "security_manager",
    }

    if table == "hr.payroll":
        templates = [
            "SELECT * FROM hr.payroll LIMIT 5;",
            "SELECT COUNT(*) FROM hr.payroll;",
            "SELECT employee_id, payroll_month, net_salary FROM hr.payroll LIMIT 10;",
            "SELECT employee_id, net_salary FROM hr.payroll WHERE payroll_month = '2026-01-01' LIMIT 10;",
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'hr' AND table_name = 'payroll';",
        ]

    elif table == "hr.employee":
        templates = [
            "SELECT * FROM hr.employee LIMIT 5;",
            "SELECT employee_id, full_name, job_role FROM hr.employee LIMIT 10;",
            "SELECT COUNT(*) FROM hr.employee;",
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'hr' AND table_name = 'employee';",
        ]

    elif table == "sys.db_account":
        if job_role in write_attempt_roles:
            templates = [
                "UPDATE sys.db_account SET account_status = 'ACTIVE' WHERE db_user = current_user;",
                "UPDATE sys.db_account SET failed_login_count = 0 WHERE db_user = current_user;",
                "DELETE FROM sys.db_account WHERE db_user = current_user;",
            ]
        else:
            templates = [
                "SELECT * FROM sys.db_account LIMIT 5;",
                "SELECT db_user, account_status, last_login FROM sys.db_account LIMIT 10;",
                "SELECT COUNT(*) FROM sys.db_account;",
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'sys' AND table_name = 'db_account';",
            ]

    elif table == "sys.system_security_log":
        if job_role in write_attempt_roles:
            templates = [
                "UPDATE sys.system_security_log SET severity_index = 1 WHERE db_user = current_user;",
                "DELETE FROM sys.system_security_log WHERE db_user = current_user;",
            ]
        else:
            templates = [
                "SELECT * FROM sys.system_security_log ORDER BY event_time DESC LIMIT 5;",
                "SELECT event_type, severity_index FROM sys.system_security_log ORDER BY event_time DESC LIMIT 10;",
                "SELECT COUNT(*) FROM sys.system_security_log;",
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'sys' AND table_name = 'system_security_log';",
            ]

    elif table == "sys.system_monitor_log":
        templates = [
            "SELECT * FROM sys.system_monitor_log ORDER BY event_time DESC LIMIT 5;",
            "SELECT db_user, cpu_utilization, memory_utilization FROM sys.system_monitor_log ORDER BY event_time DESC LIMIT 10;",
            "SELECT COUNT(*) FROM sys.system_monitor_log;",
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'sys' AND table_name = 'system_monitor_log';",
        ]

    else:
        templates = [
            f"SELECT * FROM {table} LIMIT 5;",
            f"SELECT COUNT(*) FROM {table};",
        ]

    return random.choice(templates)


def _choose_denied_target(job_role: str):
    targets = DENIED_TARGETS_BY_ROLE.get(job_role, [
        "hr.payroll",
        "sys.db_account",
        "sys.system_security_log",
    ])

    if not targets:
        return None

    return random.choice(targets)


def _choose_noise_time(sim_date, base_ts=None):
    """
    Ưu tiên sinh noise ngay sau query normal trong cùng session.
    Nếu không có base_ts thì chọn frame ngẫu nhiên, chủ yếu trong giờ làm việc.
    """
    if base_ts is not None:
        return base_ts + timedelta(seconds=random.randint(10, 45))

    frame = random.choices(
        ["G1", "G2", "G3", "G4", "G5"],
        weights=[20, 35, 30, 10, 5]
    )[0]
    return random_time_in_frame(sim_date, frame)




# ============================================================
# PUBLIC FUNCTION
# ============================================================
def inject_denied_noise(cur, db_user, job_role, sim_date,
                        session_id, event_index,
                        session_ip=None, base_ts=None):
    """
    Normal permission noise:
    - role-aware
    - SQL đa dạng
    - không execute thật
    - status = DENIED
    - rows_affected = 0
    - scenario_type = noise_denied
    """

    table = _choose_denied_target(job_role)

    # senior_dba/admin gần như không có denied normal
    if table is None:
        return 0

    ts = _choose_noise_time(sim_date, base_ts)
    frame_actual = get_graph_frame(ts.hour)
    is_after = (frame_actual == "G5")

    sql = _build_denied_sql(table, job_role)
    session_ip = session_ip or fake.ipv4_private()

    cur.execute("""
        INSERT INTO public.audit_log_raw
            (event_id, timestamp, db_user, session_id, sql_statement,
             tables_involved, status, rows_affected,
             graph_frame, sim_date, is_after_hours,
             session_start_time, session_ip, session_event_index,
             scenario_type)
        VALUES (%s,%s,%s,%s,%s,%s,'DENIED',0,%s,%s,%s,%s,%s,%s,'noise_denied')
    """, (
        str(uuid.uuid4()),
        ts,
        db_user,
        session_id,
        sql,
        table,
        frame_actual,
        sim_date,
        is_after,
        ts,
        session_ip,
        event_index
    ))

    return 1


# def inject_denied_noise(cur, db_user, sim_date, session_id, event_index, session_ip=None):
#     """3% chance: user query sai bảng → DENIED (không thực thi thật)."""
#     denied_tables = ["sys.system_security_log", "hr.payroll", "sys.db_account"]
#     table = random.choice(denied_tables)
#     ts    = random_time_in_frame(sim_date, "G2")
#     sql   = f"SELECT * FROM {table} LIMIT 5"
#     cur.execute("""
#         INSERT INTO public.audit_log_raw
#             (event_id, timestamp, db_user, session_id, sql_statement,
#              tables_involved, status, rows_affected,
#              graph_frame, sim_date, is_after_hours,
#              session_start_time, session_ip, session_event_index,
#              scenario_type)
#         VALUES (%s,%s,%s,%s,%s,%s,'DENIED',0,%s,%s,%s,%s,%s,%s,'noise_denied')
#     """, (str(uuid.uuid4()), ts, db_user, session_id, sql,
#           table, get_graph_frame(ts.hour), sim_date,
#           False, ts, session_ip or fake.ipv4_private(), event_index))

# def inject_negligent(cur, db_user, sim_date, session_id, event_index):
#     """0.05% chance: vô tình query L5, 90% bị DENIED."""
#     table  = random.choice(["sys.system_security_log", "sys.db_account"])
#     sql    = f"SELECT * FROM {table} LIMIT 10"
#     status = "DENIED" if random.random() < 0.90 else "SUCCESS"
#     rows   = 0 if status == "DENIED" else random.randint(1, 5)
#     ts     = random_time_in_frame(sim_date, "G2")
#     cur.execute("""
#         INSERT INTO public.audit_log_raw
#             (event_id, timestamp, db_user, session_id, sql_statement,
#              tables_involved, status, rows_affected,
#              graph_frame, sim_date, is_after_hours,
#              session_start_time, session_ip, session_event_index,
#              scenario_type)
#         VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'noise_negligent')
#     """, (str(uuid.uuid4()), ts, db_user, session_id, sql,
#           table, status, rows, get_graph_frame(ts.hour), sim_date,
#           False, ts, fake.ipv4_private(), event_index))

