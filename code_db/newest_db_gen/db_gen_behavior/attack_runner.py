"""
attack_runner.py — Điều phối toàn bộ dataset 90 ngày:
  - Ngày 1–60 : normal only
  - Ngày 61–90: normal background + attack events theo 4 scenarios song song

Run:
    python attack_runner.py

Chế độ chạy:
- RESET_AUDIT_LOG = True: xóa log cũ và sinh lại full 90 ngày.
- RESET_AUDIT_LOG = False: giữ 60 ngày normal đã có và chỉ sinh thêm ngày 61–90.
"""
import inspect
import json
import random
import uuid
from datetime import timedelta
import os
import csv

import psycopg2

from config import (
    DB_CONFIG,
    CREATE_LOG_TABLE,
    ACTIVE_RATE,
    ABSENT_RATE,
    SESSION_COUNT,
    get_day_type,
    is_end_of_month,
    get_normal_frame,
    should_do_g5,
    get_graph_frame,
    random_time_in_frame,
    fake,
    can_access,
)
from session_manager import simulate_user_session
from noise_injector import inject_denied_noise
from post_process import post_process_system_logs
from attack_config import (
    SIM_START,
    NORMAL_DAYS,
    TOTAL_DAYS,
    ATTACKER_ROLES,
    SCENARIO_TIMELINE,
    get_active_scenarios,
)
from attack_queries import get_attack_queries

# True: chạy lại full 90 ngày từ đầu.
# False: giữ 60 ngày normal đã có và chỉ sinh ngày 61–90.
RESET_AUDIT_LOG = True
RANDOM_SEED = 42
EXPORT_CSV = True
EXPORT_DIR = "exports"

# Object kỹ thuật của attack staging — không dùng permission matrix thường.
STAGING_TABLES = {"public.staging_data"}
SYSTEM_CATALOG_ALIASES = {"information_schema", "system_catalog", "pg_catalog"}
L5_TABLES = {"sys.db_account", "sys.system_security_log"}
SENSITIVE_TABLES = {
    "hr.employee",
    "hr.payroll",
    "sys.db_account",
    "sys.system_security_log",
    "sys.system_monitor_log",
}


# ============================================================
# SETUP
# ============================================================
def ensure_extra_columns(cur) -> None:
    """Bổ sung các cột phụ nếu bảng hiện tại chưa có."""
    cur.execute("""
        ALTER TABLE sys.system_security_log
        ADD COLUMN IF NOT EXISTS related_event_id TEXT;
    """)
    cur.execute("""
        ALTER TABLE sys.system_security_log
        ADD COLUMN IF NOT EXISTS related_session_id TEXT;
    """)


def reset_generated_tables(cur) -> None:
    """Xóa log để sinh lại dataset chính thức từ đầu."""
    cur.execute("TRUNCATE TABLE public.audit_log_raw;")
    cur.execute("TRUNCATE TABLE public.anomaly_ground_truth;")
    cur.execute("TRUNCATE TABLE sys.system_monitor_log;")
    cur.execute("TRUNCATE TABLE sys.system_security_log;")
    cur.execute("DROP TABLE IF EXISTS public.staging_data;")


# ============================================================
# CHỌN ATTACKER USERS TỪ DB
# ============================================================
ATTACKER_RANDOM = random.Random(RANDOM_SEED)
def select_attackers(cur) -> dict[str, dict[str, str]]:
    """
    Với mỗi scenario, chọn ngẫu nhiên 1 user ACTIVE có job_role phù hợp.
    Đảm bảo không chọn trùng db_user giữa các scenario.
    """
    attackers: dict[str, dict[str, str]] = {}
    selected_users: set[str] = set()

    for scenario, roles in ATTACKER_ROLES.items():
        placeholders = ",".join(["%s"] * len(roles))
        cur.execute(f"""
            SELECT a.db_user, e.job_role
            FROM sys.db_account a
            JOIN hr.employee e ON a.employee_id = e.employee_id
            WHERE e.job_role IN ({placeholders})
              AND a.account_status = 'ACTIVE'
            ORDER BY a.db_user
        """, roles)

        candidates = [
            {"db_user": r[0], "job_role": r[1]}
            for r in cur.fetchall()
            if r[0] not in selected_users
        ]

        if not candidates:
            print(f"  [WARN] Không tìm được attacker riêng cho {scenario} với roles {roles}")
            continue
        
        attacker = ATTACKER_RANDOM.choice(candidates)
        # attacker = random.choice(candidates)
        attackers[scenario] = attacker
        selected_users.add(attacker["db_user"])

    return attackers


# ============================================================
# NORMAL SESSION WRAPPER
# ============================================================
def run_normal_session(cur, user, sim_date, frame, session_ip, eom, sample_db_users, session_count=1) -> int:
    """
    Gọi simulate_user_session tương thích cả 2 phiên bản:
    - bản cũ không có session_count
    - bản mới có session_count
    """
    params = inspect.signature(simulate_user_session).parameters
    kwargs = {}
    if "session_count" in params:
        kwargs["session_count"] = session_count

    return simulate_user_session(
        cur,
        user,
        sim_date,
        frame,
        session_ip,
        eom,
        sample_db_users,
        **kwargs,
    )


# ============================================================
# PERMISSION / DENIED LOGIC CHO ATTACK SYNTHETIC EVENT
# ============================================================
def _needs_permission_check(table: str) -> bool:
    """Bỏ qua object không thuộc bảng nghiệp vụ cần kiểm tra quyền."""
    return table not in STAGING_TABLES and table not in SYSTEM_CATALOG_ALIASES


def should_force_denied(job_role: str, tables: list[str], is_write: bool) -> bool:
    """
    Vì simulator kết nối DB bằng tài khoản kỹ thuật, DB không tự enforce quyền
    theo db_user trong log. Hàm này ép status='DENIED' theo permission matrix.
    """
    for table in tables:
        if not _needs_permission_check(table):
            continue

        table_write = is_write and table in STAGING_TABLES

        if not can_access(job_role, table, write=table_write):
            return True

    return False

# ============================================================
# ATTACK SESSION
# ============================================================
def run_attack_session(cur, attacker: dict, sim_date, phase: str, scenario: str, day_offset: int) -> int:
    """
    Sinh attack events cho 1 scenario trong 1 ngày.

    - Attack event append vào public.audit_log_raw.
    - status có thể là SUCCESS, DENIED, ERROR.
    - DENIED là synthetic, không execute thật.
    - S4 staging được phép persist trong preparation/execution/cover.
    """
    db_user = attacker["db_user"]
    job_role = attacker["job_role"]

    queries = get_attack_queries(scenario, phase, job_role, day_offset)
    if not queries:
        return 0

    # Frame theo cấu hình normal hiện tại:
    # G1 6–9, G2 9–12, G3 12–14, G4 14–18, G5 18–6.
    if phase == "execution":
        frame = random.choices(["G5", "G4", "G2"], weights=[60, 25, 15])[0]
    elif phase == "cover":
        frame = random.choices(["G5", "G4"], weights=[70, 30])[0]
    else:
        frame = random.choices(["G2", "G3", "G4"], weights=[35, 35, 30])[0]

    session_id = f"sess_atk_{scenario}_{db_user}_{sim_date}_{uuid.uuid4().hex[:6]}"
    session_start = random_time_in_frame(sim_date, frame)
    session_ip = fake.ipv4_private()
    scenario_type = f"{scenario}_{phase}"

    event_count = 0
    current_ts = session_start

    for event_index, (sql, tables, is_write, description) in enumerate(queries):
        # Timestamp tăng dần trong session.
        if event_index == 0:
            ts = current_ts
        else:
            current_ts = current_ts + timedelta(
                minutes=random.randint(1, 4),
                seconds=random.randint(0, 59),
            )
            ts = current_ts

        frame_actual = get_graph_frame(ts.hour)
        is_after = frame_actual == "G5"

        status = "SUCCESS"
        rows_affected = 0

        force_denied = should_force_denied(job_role, tables, is_write)
        allow_persist_staging = scenario == "S4" and phase in {"preparation", "execution", "cover"}
        should_rollback = is_write and not allow_persist_staging

        if force_denied:
            status = "DENIED"
            rows_affected = 0
        else:
            sp_name = f"sp_atk_{uuid.uuid4().hex[:8]}"
            try:
                cur.execute(f"SAVEPOINT {sp_name}")
                cur.execute(sql)

                if cur.description:
                    rows_affected = len(cur.fetchall())
                else:
                    rows_affected = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

                if should_rollback:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
                cur.execute(f"RELEASE SAVEPOINT {sp_name}")

            except Exception:
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
                    cur.execute(f"RELEASE SAVEPOINT {sp_name}")
                except Exception:
                    pass
                status = "ERROR"
                rows_affected = 0

        cur.execute("""
            INSERT INTO public.audit_log_raw
                (event_id, timestamp, db_user, session_id, sql_statement,
                 tables_involved, status, rows_affected,
                 graph_frame, sim_date, is_after_hours,
                 session_start_time, session_ip, session_event_index,
                 scenario_type)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            str(uuid.uuid4()),
            ts,
            db_user,
            session_id,
            sql,
            ",".join(tables),
            status,
            rows_affected,
            frame_actual,
            sim_date,
            is_after,
            session_start,
            session_ip,
            event_index,
            scenario_type,
        ))
        event_count += 1

    return event_count


# ============================================================
# GROUND TRUTH + ATTACK SECURITY EVENTS
# ============================================================
CREATE_GROUND_TRUTH = """
CREATE TABLE IF NOT EXISTS public.anomaly_ground_truth (
    event_id        TEXT,
    db_user         TEXT,
    sim_date        DATE,
    graph_frame     TEXT,
    scenario_type   TEXT,
    attack_phase    TEXT,
    is_anomaly      BOOLEAN DEFAULT TRUE
);
"""


def write_ground_truth(cur, sim_date, active_scenarios: dict[str, str]) -> None:
    """Ghi ground truth event-level cho attack events của ngày đó."""
    scenario_types = [f"{sc}_{ph}" for sc, ph in active_scenarios.items()]
    if not scenario_types:
        return

    placeholders = ",".join(["%s"] * len(scenario_types))
    cur.execute(f"""
        SELECT event_id, db_user, graph_frame, scenario_type
        FROM public.audit_log_raw
        WHERE sim_date = %s
          AND scenario_type IN ({placeholders})
    """, [sim_date] + scenario_types)

    for event_id, db_user, graph_frame, scenario_type in cur.fetchall():
        parts = scenario_type.split("_", 1)
        attack_phase = parts[1] if len(parts) > 1 else ""
        cur.execute("""
            INSERT INTO public.anomaly_ground_truth
                (event_id, db_user, sim_date, graph_frame,
                 scenario_type, attack_phase, is_anomaly)
            VALUES (%s,%s,%s,%s,%s,%s,TRUE)
        """, (event_id, db_user, sim_date, graph_frame, scenario_type, attack_phase))


def _insert_attack_security_event(cur, event_time, db_user, event_type, severity_index,
                                  source_ip, target_object, related_event_id,
                                  related_session_id) -> None:
    cur.execute("""
        INSERT INTO sys.system_security_log
            (event_time, db_user, event_type, severity_index,
             source_ip, target_object,
             related_event_id, related_session_id, description)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        event_time,
        db_user,
        event_type,
        severity_index,
        source_ip,
        target_object,
        related_event_id,
        related_session_id,
        f"[ATTACK_GENERATED] ATTACK_{event_type} by {db_user} on {target_object}",
    ))


def write_attack_security_events(cur, sim_date, active_scenarios: dict[str, str]) -> None:
    """
    Sinh security events cho attack events.
    Lý do đặt ở đây: post_process.py cũ thường chỉ detect noise_denied/normal L5,
    nên các attack DENIED/SUCCESS L5 có thể không được ghi vào system_security_log.
    """
    scenario_types = [f"{sc}_{ph}" for sc, ph in active_scenarios.items()]
    if not scenario_types:
        return

    placeholders = ",".join(["%s"] * len(scenario_types))
    cur.execute(f"""
        SELECT event_id, timestamp, db_user, session_id, session_ip,
               status, tables_involved, scenario_type
        FROM public.audit_log_raw
        WHERE sim_date = %s
          AND scenario_type IN ({placeholders})
    """, [sim_date] + scenario_types)

    for event_id, ts, db_user, session_id, source_ip, status, tables_str, scenario_type in cur.fetchall():
        tables = [t.strip() for t in (tables_str or "").split(",") if t.strip()]
        table_set = set(tables)

        if status == "DENIED" and table_set & SENSITIVE_TABLES:
            target = sorted(table_set & SENSITIVE_TABLES)[0]
            _insert_attack_security_event(
                cur, ts, db_user, "UNAUTHORIZED_ACCESS", 2,
                source_ip, target, event_id, session_id,
            )
            continue

        if status == "SUCCESS" and table_set & L5_TABLES:
            target = sorted(table_set & L5_TABLES)[0]
            _insert_attack_security_event(
                cur, ts, db_user, "L5_ACCESS", 3,
                source_ip, target, event_id, session_id,
            )

# ============================================================
# EXPORT CSV
# ============================================================
def _copy_query_to_csv(cur, query: str, output_path: str) -> None:
    """
    Export kết quả SELECT ra CSV bằng PostgreSQL COPY.
    File CSV được ghi trên máy đang chạy Python.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    copy_sql = f"""
        COPY (
            {query}
        ) TO STDOUT WITH CSV HEADER
    """

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        cur.copy_expert(copy_sql, f)

def export_audit_by_day(cur, export_dir: str = EXPORT_DIR) -> None:
    """
    Xuất audit_log_raw thành từng file CSV theo sim_date.
    """
    audit_day_dir = os.path.join(export_dir, "audit_by_day")
    os.makedirs(audit_day_dir, exist_ok=True)

    cur.execute("""
        SELECT DISTINCT sim_date
        FROM public.audit_log_raw
        ORDER BY sim_date
    """)

    dates = cur.fetchall()

    for (sim_date,) in dates:
        date_str = sim_date.strftime("%Y-%m-%d")

        _copy_query_to_csv(cur, f"""
            SELECT *
            FROM public.audit_log_raw
            WHERE sim_date = DATE '{date_str}'
            ORDER BY timestamp, session_id, session_event_index
        """, os.path.join(audit_day_dir, f"audit_{date_str}.csv"))

    print(f"✓ Daily audit CSV exported to: {os.path.abspath(audit_day_dir)}")

def export_dataset_csv(cur, export_dir: str = EXPORT_DIR) -> None:
    """
    Xuất các bảng chính ra CSV để dễ kiểm tra ngoài DB.
    """

    os.makedirs(export_dir, exist_ok=True)

    # # 1. Full audit log quá nặng
    # _copy_query_to_csv(cur, """
    #     SELECT *
    #     FROM public.audit_log_raw
    #     ORDER BY timestamp, session_id, session_event_index
    # """, os.path.join(export_dir, "audit_log_full.csv"))
    
    # 1.2. Full audit log by day
    export_audit_by_day(cur, export_dir)

    # 2. Train normal: ngày 1–60
    _copy_query_to_csv(cur, """
        SELECT *
        FROM public.audit_log_raw
        WHERE sim_date < DATE '2026-01-01' + INTERVAL '60 days'
        ORDER BY timestamp, session_id, session_event_index
    """, os.path.join(export_dir, "audit_log_train_normal_60d.csv"))

    # 3. Test: ngày 61–90
    _copy_query_to_csv(cur, """
        SELECT *
        FROM public.audit_log_raw
        WHERE sim_date >= DATE '2026-01-01' + INTERVAL '60 days'
        ORDER BY timestamp, session_id, session_event_index
    """, os.path.join(export_dir, "audit_log_test_30d.csv"))

    # 4. Chỉ attack events
    _copy_query_to_csv(cur, """
        SELECT *
        FROM public.audit_log_raw
        WHERE scenario_type LIKE 'S1_%'
           OR scenario_type LIKE 'S2_%'
           OR scenario_type LIKE 'S3_%'
           OR scenario_type LIKE 'S4_%'
        ORDER BY timestamp, session_id, session_event_index
    """, os.path.join(export_dir, "audit_log_attack_only.csv"))

    # 5. Ground truth event-level
    _copy_query_to_csv(cur, """
        SELECT *
        FROM public.anomaly_ground_truth
        ORDER BY sim_date, db_user, graph_frame, event_id
    """, os.path.join(export_dir, "anomaly_ground_truth_event.csv"))

    # 6. User-frame label: gom từ event-level attack
    _copy_query_to_csv(cur, """
        SELECT
            db_user,
            sim_date,
            graph_frame,
            TRUE AS is_anomaly,
            STRING_AGG(DISTINCT scenario_type, ',') AS scenario_types,
            STRING_AGG(DISTINCT attack_phase, ',') AS attack_phases,
            COUNT(*) AS anomaly_event_count
        FROM public.anomaly_ground_truth
        GROUP BY db_user, sim_date, graph_frame
        ORDER BY sim_date, db_user, graph_frame
    """, os.path.join(export_dir, "anomaly_ground_truth_user_frame.csv"))

    # 7. System monitor log
    _copy_query_to_csv(cur, """
        SELECT *
        FROM sys.system_monitor_log
        ORDER BY event_time, db_user, session_id
    """, os.path.join(export_dir, "system_monitor_log.csv"))

    # 8. System security log
    _copy_query_to_csv(cur, """
        SELECT *
        FROM sys.system_security_log
        ORDER BY event_time, db_user, event_type
    """, os.path.join(export_dir, "system_security_log.csv"))

    # 9. Summary theo ngày
    _copy_query_to_csv(cur, """
        SELECT
            sim_date,
            COUNT(*) AS total_events,
            COUNT(DISTINCT db_user) AS active_users,
            COUNT(DISTINCT session_id) AS total_sessions,
            SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) AS success_events,
            SUM(CASE WHEN status = 'DENIED' THEN 1 ELSE 0 END) AS denied_events,
            SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) AS error_events,
            SUM(CASE WHEN scenario_type LIKE 'S1_%'
                      OR scenario_type LIKE 'S2_%'
                      OR scenario_type LIKE 'S3_%'
                      OR scenario_type LIKE 'S4_%'
                     THEN 1 ELSE 0 END) AS attack_events
        FROM public.audit_log_raw
        GROUP BY sim_date
        ORDER BY sim_date
    """, os.path.join(export_dir, "summary_by_day.csv"))

    # 10. Summary theo scenario
    _copy_query_to_csv(cur, """
        SELECT
            scenario_type,
            status,
            COUNT(*) AS total_events,
            MIN(timestamp) AS first_time,
            MAX(timestamp) AS last_time,
            COUNT(DISTINCT db_user) AS users,
            COUNT(DISTINCT session_id) AS sessions,
            ROUND(AVG(rows_affected), 2) AS avg_rows,
            MAX(rows_affected) AS max_rows
        FROM public.audit_log_raw
        GROUP BY scenario_type, status
        ORDER BY scenario_type, status
    """, os.path.join(export_dir, "summary_by_scenario.csv"))

    print(f"✓ CSV exported to: {os.path.abspath(export_dir)}")

# export ds user
def export_attackers_csv(attackers: dict, export_dir: str = EXPORT_DIR) -> None:
    """
    Xuất danh sách attacker được chọn cho từng scenario.
    """
    os.makedirs(export_dir, exist_ok=True)

    output_path = os.path.join(export_dir, "attack_users.csv")

    rows = []
    for scenario, attacker in attackers.items():
        timeline = SCENARIO_TIMELINE.get(scenario, {})
        phases = timeline.get("phases", {})

        phase_text = "; ".join(
            f"{phase}:{start}-{end}"
            for phase, (start, end) in phases.items()
        )

        rows.append({
            "scenario": scenario,
            "db_user": attacker["db_user"],
            "job_role": attacker["job_role"],
            "start_day": timeline.get("start"),
            "end_day": timeline.get("end"),
            "start_date": SIM_START + timedelta(days=timeline.get("start", 1) - 1),
            "end_date": SIM_START + timedelta(days=timeline.get("end", 1) - 1),
            "phase_plan": phase_text,
        })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "db_user",
                "job_role",
                "start_day",
                "end_day",
                "start_date",
                "end_date",
                "phase_plan",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Attack users exported to: {os.path.abspath(output_path)}")

# ============================================================
# MAIN SIMULATION
# ============================================================
def run_simulation() -> None:
    random.seed(RANDOM_SEED)

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(CREATE_LOG_TABLE)
    cur.execute(CREATE_GROUND_TRUTH)
    ensure_extra_columns(cur)
    conn.commit()

    if RESET_AUDIT_LOG:
        reset_generated_tables(cur)
        conn.commit()
        start_day_offset = 1
    else:
        # Giữ 60 ngày normal đã có, chỉ sinh thêm ngày 61–90.
        start_day_offset = NORMAL_DAYS + 1

    cur.execute("""
        SELECT a.db_user, e.job_role
        FROM sys.db_account a
        JOIN hr.employee e ON a.employee_id = e.employee_id
        WHERE a.account_status = 'ACTIVE'
        ORDER BY a.db_user
    """)
    rows = cur.fetchall()
    users = [{"db_user": r[0], "job_role": r[1]} for r in rows]
    sample_db_users = [r[0] for r in rows]
    print(f"Loaded {len(users)} active users")

    attackers = select_attackers(cur)
    print("Attackers selected:")
    for sc, atk in attackers.items():
        print(f"  {sc}: {atk['db_user']} ({atk['job_role']})")

    with open("attack_config_run.json", "w", encoding="utf-8") as f:
        json.dump(attackers, f, indent=2, ensure_ascii=False)

    total_events = 0

    for day_offset in range(start_day_offset, TOTAL_DAYS + 1):
        sim_date = SIM_START + timedelta(days=day_offset - 1)
        day_type = get_day_type(sim_date)
        eom = is_end_of_month(sim_date)
        day_label = sim_date.strftime("%Y-%m-%d (%a)")
        is_attack_day = day_offset > NORMAL_DAYS
        day_events = 0

        # 1) Normal background sessions — full 90 ngày khi RESET=True,
        # hoặc chỉ ngày 61–90 khi append mode.
        for user in users:
            job_role = user["job_role"]
            rate = ACTIVE_RATE.get(job_role, {}).get(day_type, 0.5)

            if random.random() > rate:
                continue
            if random.random() < ABSENT_RATE:
                continue

            session_ip = fake.ipv4_private()

            # Lấy số session theo đúng SESSION_COUNT như day_simulator.py
            day_type_vol = "weekend" if day_type != "weekday" else "weekday"
            sess_min, sess_max = SESSION_COUNT.get(job_role, {}).get(day_type_vol, (1, 2))

            if sess_min == 0 and sess_max == 0:
                n_sessions = 0
            else:
                n_sessions = random.randint(sess_min, sess_max)

            for _ in range(n_sessions):
                frame = get_normal_frame(job_role)
                day_events += run_normal_session(
                    cur, user, sim_date, frame, session_ip, eom,
                    sample_db_users, session_count=n_sessions,
                )

            # G5 ngoài giờ — truyền session_count giống day_simulator.py
            if should_do_g5(job_role, sim_date):
                day_events += run_normal_session(
                    cur, user, sim_date, "G5", session_ip, eom,
                    sample_db_users, session_count=n_sessions,
                )


        # 2) Attack sessions — chỉ ngày 61–90.
        active_scenarios: dict[str, str] = {}
        if is_attack_day:
            active_scenarios = get_active_scenarios(day_offset)

            for scenario, phase in active_scenarios.items():
                attacker = attackers.get(scenario)
                if not attacker:
                    continue

                n = run_attack_session(cur, attacker, sim_date, phase, scenario, day_offset)
                day_events += n
                if n > 0:
                    print(f"    [{scenario}/{phase}] {attacker['db_user']} — {n} attack events")

            write_ground_truth(cur, sim_date, active_scenarios)
            # write_attack_security_events(cur, sim_date, active_scenarios)

        # 3) Post-process monitor + normal security logs cuối ngày.
        post_process_system_logs(cur, sim_date)

        # 4) Attack security events ghi sau post_process để không bị xóa.
        if is_attack_day:
            write_attack_security_events(cur, sim_date, active_scenarios)


        # 4) Commit cuối ngày.
        conn.commit()
        total_events += day_events
        tag = " [ATTACK]" if is_attack_day else ""
        print(f"  {day_label}{tag} — {day_events} events")

    # export data ra csv
    if EXPORT_CSV:
        export_dataset_csv(cur, EXPORT_DIR)
        export_attackers_csv(attackers, EXPORT_DIR)

    cur.close()
    conn.close()
    print(f"\n✓ Simulation complete: {total_events} total events over days {start_day_offset}–{TOTAL_DAYS}")


if __name__ == "__main__":
    run_simulation()
