"""
session_manager.py — Tạo session, sinh query trong session và ghi session boundary vào audit_log_raw.
Tách từ normal_behavior.py, giữ nguyên logic gốc.
"""

import random
import uuid
from datetime import date, timedelta, datetime

from config import QUERY_VOLUME, get_day_type, random_time_in_frame, get_graph_frame
from query_builder import build_query

from noise_injector import inject_denied_noise


def simulate_user_session(cur, user: dict, sim_date: date,
                           frame: str, session_ip: str,
                           end_of_month: bool,
                           sample_db_users: list = None,
                           session_count: int = 2,):
    """Sinh 1 session cho 1 user trong 1 graph frame."""
    job_role    = user["job_role"]
    db_user     = user["db_user"]
    session_id  = f"sess_{db_user}_{sim_date}_{frame}_{uuid.uuid4().hex[:6]}"
    session_start = random_time_in_frame(sim_date, frame)

    # Số query trong session
    day_type_vol = "weekend" if get_day_type(sim_date) != "weekday" else "weekday"
    vol_min, vol_max = QUERY_VOLUME.get(job_role, {}).get(day_type_vol, (5, 10))

    # Nếu role weekend volume = (0, 0) thì không sinh query
    if vol_min == 0 and vol_max == 0:
        return 0

    # Cuối tháng: tăng volume cho kế toán và HR
    if end_of_month and job_role in ("accountant", "hr_staff",
                                     "finance_manager", "hr_manager"):
        vol_min = int(vol_min * 1.5)
        vol_max = int(vol_max * 2.0)

    # n_queries = random.randint(max(1, vol_min // 2), max(2, vol_max // 2))
    # nhận session count rồi chia ra
    session_count = max(1, session_count)

    # Giảm workload ngoài giờ
    if frame == "G5":
        vol_min = max(1, int(vol_min * 0.6))
        vol_max = max(2, int(vol_max * 0.6))

    low = max(1, vol_min // session_count)
    high = max(2, vol_max // session_count)

    if high < low:
        high = low

    n_queries = random.randint(low, high)

    event_index = 0

    # sửa thành dạng cộng dồn
    current_ts = session_start

    for _ in range(n_queries):
        sql, tables_resolved, is_write = build_query(
            job_role=job_role,
            sim_date=sim_date,
            sample_db_users=sample_db_users
        )

        # ts = session_start + timedelta(
        #     minutes=event_index * random.randint(1, 5),
        #     seconds=random.randint(0, 59)
        # )
        current_ts = current_ts + timedelta(
            minutes=random.randint(1, 5),
            seconds=random.randint(0, 59)
        )

        ts = current_ts

        # Đảm bảo không bị lố sang ngày hôm sau
        eod = datetime(sim_date.year, sim_date.month, sim_date.day, 23, 59, 59)
        ts = min(ts, eod)
        current_ts = ts

        frame_actual = get_graph_frame(ts.hour)
        is_after = (frame_actual == "G5")

        status = "SUCCESS"
        rows_affected = 0
        # error_message = None

        sp_name = f"sp_{uuid.uuid4().hex[:8]}"
        cur.execute(f"SAVEPOINT {sp_name}")

        try:
            cur.execute(sql)

            if cur.description:
                rows_affected = len(cur.fetchall())
            else:
                rows_affected = cur.rowcount if cur.rowcount > 0 else 0

            # Nếu là query write trong normal, rollback để không làm bẩn DB seed.
            if is_write:
                cur.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")

            cur.execute(f"RELEASE SAVEPOINT {sp_name}")

        except Exception as e:
            cur.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
            cur.execute(f"RELEASE SAVEPOINT {sp_name}")
            status = "ERROR"
            rows_affected = 0
            # error_message = str(e)[:200]

        # Ghi log normal vào audit_log_raw
        cur.execute("""
            INSERT INTO public.audit_log_raw
                (event_id, timestamp, db_user, session_id, sql_statement,
                 tables_involved, status, rows_affected,
                 graph_frame, sim_date, is_after_hours,
                 session_start_time, session_ip, session_event_index,
                 scenario_type)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'normal')
        """, (
            str(uuid.uuid4()), ts, db_user, session_id, sql,
            ",".join(tables_resolved), status, rows_affected,
            frame_actual, sim_date, is_after,
            session_start, session_ip, event_index,
        ))


        # Noise DENIED normal: xác suất thấp, role-aware
        # Không coi là attack, chỉ là lỗi quyền/người dùng query nhầm.
        if random.random() < 0.01:
            event_index += 1
            injected = inject_denied_noise(
                cur=cur,
                db_user=db_user,
                job_role=job_role,
                sim_date=sim_date,
                session_id=session_id,
                event_index=event_index,
                session_ip=session_ip,
                base_ts=ts
            )

            # Nếu role như senior_dba/admin không sinh noise thì trả 0.
            # Khi đó lùi lại event_index để không bị nhảy số.
            if injected == 0:
                event_index -= 1

        event_index += 1

    return event_index


