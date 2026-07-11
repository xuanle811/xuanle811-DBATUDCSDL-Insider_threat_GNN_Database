"""
post_process.py — Sinh system_monitor_log và system_security_log
từ audit_log_raw sau khi simulate xong toàn bộ session của 1 ngày.

Được gọi từ day_simulator.py sau mỗi ngày:
    post_process_system_logs(cur, sim_date)
"""

import random
import uuid
from datetime import date, datetime, timedelta
from faker import Faker

fake = Faker()

# ── Bảng L5 để detect L5_ACCESS / L5_TAMPERING ──────────────
L5_TABLES = {"sys.db_account", "sys.system_security_log"}

# ── Security level để detect UNAUTHORIZED_ACCESS ─────────────
SENSITIVE_TABLES = {
    "hr.payroll", "sys.system_monitor_log",
    "sys.db_account", "sys.system_security_log",
    "hr.employee", "hr.department",
}

# ── Severity mapping ─────────────────────────────────────────
# 1=low, 2=medium, 3=high, 4=critical
SEVERITY = {
    # "FAILED_LOGIN":         1, tạm thời bỏ
    "UNAUTHORIZED_ACCESS":  2,
    "PRIVILEGE_CHANGE":     3,
    "L5_ACCESS":            3,
    "L5_TAMPERING":         4,
}

# ── Roles có quyền L5 hợp lệ ─────────────────────────────────
L5_AUTHORIZED_ROLES = {
    "security_analyst", "security_manager",
    "junior_dba", "senior_dba", "admin"
}

# ============================================================
# HELPER: lấy job_role từ db_user (cache để tránh query lặp)
# ============================================================
_role_cache: dict = {}

def _get_job_role(cur, db_user: str) -> str:
    if db_user not in _role_cache:
        cur.execute("""
            SELECT e.job_role
            FROM sys.db_account a
            JOIN hr.employee e ON a.employee_id = e.employee_id
            WHERE a.db_user = %s
        """, (db_user,))
        row = cur.fetchone()
        _role_cache[db_user] = row[0] if row else "unknown"
    return _role_cache[db_user]


# ============================================================
# 1. system_monitor_log
# Mỗi session → 1 record tổng hợp metrics
# ============================================================
def _generate_monitor_logs(cur, sim_date: date):
    """
    Đọc các session trong audit_log_raw của sim_date,
    tổng hợp metrics thực tế và insert vào system_monitor_log.
    """

    # Lấy thông tin từng session: db_user, session_id, frame,
    # số events, tổng rows_affected, thời điểm đầu/cuối
    cur.execute("""
        SELECT
            db_user,
            session_id,
            session_start_time,
            COUNT(*)                    AS event_count,
            SUM(rows_affected)          AS total_rows,
            MAX(timestamp)              AS session_end,
            SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) AS error_count
        FROM public.audit_log_raw
        WHERE sim_date = %s
          AND scenario_type IN ('normal', 'noise_denied', 'noise_negligent')
        GROUP BY db_user, session_id, session_start_time
        ORDER BY session_start_time
    """, (sim_date,))

    sessions = cur.fetchall()
    if not sessions:
        return

    # Tổng active connections trong ngày (để simulate active_connections)
    total_sessions = len(sessions)

    for row in sessions:
        (db_user, session_id, session_start,
         event_count, total_rows, session_end, error_count) = row

        # event_time = giữa session
        if session_start and session_end:
            mid_delta = (session_end - session_start) / 2
            event_time = session_start + mid_delta
        else:
            event_time = session_start or datetime(
                sim_date.year, sim_date.month, sim_date.day, 9, 0, 0
            )

        # CPU tăng theo số events và rows
        base_cpu = random.uniform(5, 20)
        load_cpu = min(event_count * 0.8 + (total_rows or 0) * 0.001, 60)
        cpu = round(base_cpu + load_cpu + random.uniform(-2, 2), 2)
        cpu = max(1.0, min(cpu, 95.0))

        # Memory tương tự
        base_mem = random.uniform(20, 40)
        load_mem = min(event_count * 0.5, 40)
        memory = round(base_mem + load_mem + random.uniform(-3, 3), 2)
        memory = max(10.0, min(memory, 95.0))

        # active_connections = số sessions đang chạy song song gần thời điểm này
        active_conn = random.randint(
            max(1, total_sessions // 4),
            max(2, total_sessions // 2)
        )

        # Disk I/O tỉ lệ với rows_affected
        disk_read  = round((total_rows or 0) * random.uniform(0.5, 2.0)
                           + random.uniform(50, 500), 2)
        disk_write = round((total_rows or 0) * random.uniform(0.1, 0.5)
                           + random.uniform(10, 100), 2)

        # backup_status: chỉ DBA có backup, xác suất thấp
        job_role = _get_job_role(cur, db_user)
        if job_role in ("senior_dba", "admin"):
            backup_status = random.choices(
                ["OK", "RUNNING", "SKIPPED", "FAILED"],
                weights=[60, 10, 25, 5]
            )[0]
        else:
            backup_status = "N/A"

        """
        OK       = backup chạy thành công
        RUNNING  = backup đang chạy
        FAILED   = backup thất bại
        SKIPPED  = có lịch/đáng lẽ có backup nhưng bị bỏ qua
        N/A      = không áp dụng, user/session này không liên quan backup
"""

        cur.execute("""
            INSERT INTO sys.system_monitor_log
                (event_time, db_user, session_id,
                 cpu_utilization, memory_utilization, active_connections,
                 disk_read_kb, disk_write_kb, backup_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            event_time, db_user, session_id,
            cpu, memory, active_conn,
            disk_read, disk_write, backup_status
        ))


# ============================================================
# 2. system_security_log
# Đọc audit_log_raw → phát hiện các event đáng ghi nhận
# ============================================================
def _insert_security_event(cur, event_time, db_user, event_type,
                           severity_index, source_ip,
                           target_object, related_event_id,
                           related_session_id):
    """Helper insert 1 record vào system_security_log."""
    """Helper insert 1 record vào system_security_log."""
    # Thêm tiền tố [SIM_GENERATED] cố định để đánh dấu dữ liệu giả lập
    sim_description = f"[SIM_GENERATED] {event_type} by {db_user} on {target_object}"
    cur.execute("""
        INSERT INTO sys.system_security_log
            (event_time, db_user, event_type, severity_index,
             source_ip, target_object,
             related_event_id, related_session_id,
             description)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        event_time, db_user, event_type, severity_index,
        source_ip, target_object,
        related_event_id, related_session_id,
        sim_description
    ))


def _generate_security_logs(cur, sim_date: date):
    """
    Đọc audit_log_raw của sim_date, sinh security events dựa trên
    hành vi SQL thực tế đã xảy ra.
    """

    # ── 2b. UNAUTHORIZED_ACCESS: DENIED vào bảng nhạy cảm ──
    cur.execute("""
        SELECT event_id, db_user, session_id, sql_statement,
               tables_involved, timestamp, session_ip
        FROM public.audit_log_raw
        WHERE sim_date = %s
          AND status = 'DENIED'
          AND scenario_type IN ('noise_denied', 'noise_negligent')
    """, (sim_date,))

    for event_id, db_user, session_id, sql, tables, ts, source_ip in cur.fetchall():
        table_list = [t.strip() for t in (tables or "").split(",")]
        for tbl in table_list:
            if tbl in SENSITIVE_TABLES:
                _insert_security_event(
                    cur, ts, db_user, "UNAUTHORIZED_ACCESS",
                    SEVERITY["UNAUTHORIZED_ACCESS"],
                    source_ip, tbl, event_id, session_id
                )
                break   # 1 event per denied query

    # ── 2c. L5_ACCESS: user có quyền query vào L5 table ────
    cur.execute("""
        SELECT event_id, db_user, session_id, tables_involved,
               timestamp, session_ip, status
        FROM public.audit_log_raw
        WHERE sim_date = %s
          AND status = 'SUCCESS'
          AND scenario_type = 'normal'
    """, (sim_date,))

    for event_id, db_user, session_id, tables, ts, source_ip, status in cur.fetchall():
        table_list = [t.strip() for t in (tables or "").split(",")]
        job_role   = _get_job_role(cur, db_user)

        for tbl in table_list:
            if tbl in L5_TABLES:
                if job_role in L5_AUTHORIZED_ROLES:
                    _insert_security_event(
                        cur, ts, db_user, "L5_ACCESS",
                        SEVERITY["L5_ACCESS"],
                        source_ip, tbl, event_id, session_id
                    )
                else:
                    # User không có quyền nhưng SUCCESS → escalate
                    _insert_security_event(
                        cur, ts, db_user, "L5_TAMPERING",
                        SEVERITY["L5_TAMPERING"],
                        source_ip, tbl, event_id, session_id
                    )
                break

    # ── 2d. PRIVILEGE_CHANGE: DBA write vào sys.db_account ──
    cur.execute("""
        SELECT event_id, db_user, session_id, sql_statement,
               timestamp, session_ip
        FROM public.audit_log_raw
        WHERE sim_date = %s
          AND status = 'SUCCESS'
          AND tables_involved LIKE '%%sys.db_account%%'
          AND sql_statement ILIKE '%%UPDATE%%'
    """, (sim_date,))

    for event_id, db_user, session_id, sql, ts, source_ip in cur.fetchall():
        job_role = _get_job_role(cur, db_user)
        if job_role in ("senior_dba", "admin"):
            _insert_security_event(
                cur, ts, db_user, "PRIVILEGE_CHANGE",
                SEVERITY["PRIVILEGE_CHANGE"],
                source_ip, "sys.db_account", event_id, session_id
            )





# ============================================================
# PUBLIC ENTRY POINT — gọi từ day_simulator.py
# ============================================================
def post_process_system_logs(cur, sim_date: date):
    """
    Sinh monitor và security logs từ audit_log_raw của sim_date.
    Gọi sau khi toàn bộ sessions của ngày đó đã được simulate.
    Chống sinh trùng
    """
    global _role_cache
    _role_cache.clear() # Giải phóng cache ngày cũ để nạp tài nguyên sạch cho ngày mới

    cur.execute("""
        DELETE FROM sys.system_monitor_log
        WHERE DATE(event_time) = %s
    """, (sim_date,))

    # Xóa chính xác tuyệt đối các bản ghi có tag giả lập, không lo xóa nhầm log thật
    cur.execute("""
        DELETE FROM sys.system_security_log
        WHERE DATE(event_time) = %s
            AND description LIKE '[SIM_GENERATED]%%'
    """, (sim_date,))

    _generate_monitor_logs(cur, sim_date)
    _generate_security_logs(cur, sim_date)
