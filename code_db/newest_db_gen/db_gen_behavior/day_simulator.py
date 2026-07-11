"""
day_simulator.py — Điều phối từng ngày: ai active, bao nhiêu session, chạy simulation.
Tách từ normal_behavior.py, giữ nguyên logic gốc.

Run:
    python day_simulator.py
"""
"""PIpeline:
Kết nối PostgreSQL insider_db
→ tạo bảng public.audit_log_raw nếu chưa có
→ nếu RESET_AUDIT_LOG=True thì xóa log cũ trong audit_log_raw
→ đọc danh sách user thật từ sys.db_account + hr.employee
→ sinh session/query theo role
→ thực thi SQL thật trên DB
→ ghi kết quả vào public.audit_log_raw
→ commit theo từng ngày
"""

import psycopg2
import random
from datetime import timedelta

from config import (
    DB_CONFIG, CREATE_LOG_TABLE, SIM_START, SIM_DAYS,
    ACTIVE_RATE, ABSENT_RATE,SESSION_COUNT,
    get_day_type, is_end_of_month, get_normal_frame,
    should_do_g5, fake,
)
from session_manager import simulate_user_session
# Import ở đầu file
from post_process import post_process_system_logs

RESET_AUDIT_LOG = True # reset audit_log_raw khi chạy lại
def run_simulation():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    # Tạo bảng log tạm nếu chưa có
    cur.execute(CREATE_LOG_TABLE)
    conn.commit()



    if RESET_AUDIT_LOG:
        cur.execute("TRUNCATE TABLE public.audit_log_raw;")
        cur.execute("TRUNCATE TABLE sys.system_monitor_log;")
        cur.execute("TRUNCATE TABLE sys.system_security_log;")
        conn.commit()

    # Lấy danh sách users từ DB
    cur.execute("""
        SELECT a.db_user, e.job_role
        FROM sys.db_account a
        JOIN hr.employee e ON a.employee_id = e.employee_id
        WHERE a.account_status = 'ACTIVE'
        ORDER BY a.db_user
    """)
    users = [{"db_user": row[0], "job_role": row[1]} for row in cur.fetchall()]
    sample_db_users = [u["db_user"] for u in users]
    print(f"Loaded {len(users)} active users")

    total_events = 0
    for day_offset in range(SIM_DAYS):
        sim_date  = SIM_START + timedelta(days=day_offset)
        day_type  = get_day_type(sim_date)
        eom       = is_end_of_month(sim_date)
        day_label = sim_date.strftime("%Y-%m-%d (%a)")

        day_events = 0

        # 1. Sinh toàn bộ audit_log_raw của ngày sim_date
        for user in users:
            job_role = user["job_role"]
            rate     = ACTIVE_RATE.get(job_role, {}).get(day_type, 0.5)

            # Bỏ qua nếu không active hoặc vắng ngẫu nhiên
            if random.random() > rate:
                continue
            if random.random() < ABSENT_RATE:
                continue

            session_ip = fake.ipv4_private()

            # Chọn 1–2 frames ban ngày
            # n_sessions = random.randint(1, 2) 
            # sửa lại, lấy số session từ config
            day_type_vol = "weekend" if day_type != "weekday" else "weekday"
            sess_min, sess_max = SESSION_COUNT.get(job_role, {}).get(day_type_vol, (1, 2))

            if sess_min == 0 and sess_max == 0:
                n_sessions = 0
            else:
                n_sessions = random.randint(sess_min, sess_max)
                
            for _ in range(n_sessions):
                frame = get_normal_frame(job_role)
                n = simulate_user_session(cur, user, sim_date,
                                          frame, session_ip, eom, sample_db_users, session_count=n_sessions)
                day_events += n

            # G5 ngoài giờ (DBA/dev/WFH)
            if should_do_g5(job_role, sim_date):
                n = simulate_user_session(cur, user, sim_date,
                                          "G5", session_ip, eom, sample_db_users, session_count=n_sessions)
                day_events += n

        # 2. Sinh monitor/security log từ audit_log_raw của ngày đó
        post_process_system_logs(cur, sim_date)
        
        # 3. Commit cuối ngày
        conn.commit()
        total_events += day_events
        print(f"  {day_label} — {day_events} events")

    cur.close()
    conn.close()
    print(f"\n✓ Simulation complete: {total_events} total events over {SIM_DAYS} days")


if __name__ == "__main__":
    run_simulation()
