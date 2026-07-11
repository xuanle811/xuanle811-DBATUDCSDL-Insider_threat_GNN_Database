"""
config.py — Constants/config dùng chung cho normal behavior workload.
Tách từ normal_behavior.py, giữ nguyên logic gốc.
"""

import random
from faker import Faker
from datetime import date, datetime, timedelta

fake = Faker()
random.seed(42)
Faker.seed(42)

# ============================================================
# CONFIG
# ============================================================
DB_CONFIG = dict(dbname="insider_db", user="postgres",
                 password="123456", host="localhost", port=5432)

SIM_START  = date(2026, 1, 1)
SIM_DAYS   = 60
ABSENT_RATE = 0.05   # xác suất vắng ngẫu nhiên đè lên mọi role

ACTIVE_RATE = {
    "sales_staff":         {"weekday": 0.90, "saturday": 0.15, "sunday": 0.03},
    "procurement_staff":   {"weekday": 0.90, "saturday": 0.15, "sunday": 0.03},
    "accountant":          {"weekday": 0.85, "saturday": 0.10, "sunday": 0.02},
    "hr_staff":            {"weekday": 0.85, "saturday": 0.10, "sunday": 0.02},
    "sales_manager":       {"weekday": 0.90, "saturday": 0.22, "sunday": 0.07},
    "finance_manager":     {"weekday": 0.90, "saturday": 0.22, "sunday": 0.07},
    "procurement_manager": {"weekday": 0.90, "saturday": 0.22, "sunday": 0.07},
    "hr_manager":          {"weekday": 0.90, "saturday": 0.22, "sunday": 0.07},
    "senior_analyst":      {"weekday": 0.90, "saturday": 0.22, "sunday": 0.07},
    "developer":           {"weekday": 0.90, "saturday": 0.27, "sunday": 0.10},
    "data_engineer":       {"weekday": 0.90, "saturday": 0.27, "sunday": 0.10},
    "data_analyst":        {"weekday": 0.88, "saturday": 0.27, "sunday": 0.10},
    "security_analyst":    {"weekday": 0.87, "saturday": 0.40, "sunday": 0.30},
    "security_manager":    {"weekday": 0.90, "saturday": 0.40, "sunday": 0.30},
    "junior_dba":          {"weekday": 0.92, "saturday": 0.50, "sunday": 0.35},
    "senior_dba":          {"weekday": 0.95, "saturday": 0.60, "sunday": 0.45},
    "admin":               {"weekday": 0.95, "saturday": 0.60, "sunday": 0.45},
}

# Query volume (min, max) theo role và loại ngày
QUERY_VOLUME = {
    "sales_staff":         {"weekday": (12, 20), "weekend": (2,  6)},
    "sales_manager":       {"weekday": (15, 25), "weekend": (5, 10)},
    "accountant":          {"weekday": (15, 25), "weekend": (10,15)},
    "finance_manager":     {"weekday": (20, 35), "weekend": (10,20)},
    "procurement_staff":   {"weekday": (10, 18), "weekend": (0,  0)},
    "procurement_manager": {"weekday": (15, 25), "weekend": (5, 10)},
    "hr_staff":            {"weekday": (15, 25), "weekend": (10,15)},
    "hr_manager":          {"weekday": (20, 35), "weekend": (10,20)},
    "data_analyst":        {"weekday": (20, 40), "weekend": (20,30)},
    "senior_analyst":      {"weekday": (25, 45), "weekend": (20,35)},
    "developer":           {"weekday": (15, 30), "weekend": (0,  0)},
    "data_engineer":       {"weekday": (20, 35), "weekend": (5, 10)},
    "security_analyst":    {"weekday": (20, 35), "weekend": (10,20)},
    "security_manager":    {"weekday": (15, 25), "weekend": (10,15)},
    "junior_dba":          {"weekday": (20, 35), "weekend": (10,20)},
    "senior_dba":          {"weekday": (25, 40), "weekend": (15,25)},
    "admin":               {"weekday": (8,  15), "weekend": (5, 10)},
}

# Graph frame theo giờ
def get_graph_frame(hour):
    if  6 <= hour <  9: return "G1"
    if  9 <= hour < 12: return "G2"
    if 12 <= hour < 14: return "G3"
    if 14 <= hour < 18: return "G4"
    return "G5"   # 18–06

# Loại ngày
def get_day_type(d: date):
    wd = d.weekday()   # 0=Mon … 6=Sun
    if wd == 5: return "saturday"
    if wd == 6: return "sunday"
    return "weekday"

def is_end_of_month(d: date):
    return d.day >= 25

def is_friday(d: date):
    return d.weekday() == 4

# ============================================================
# PERMISSION MATRIX (dùng để kiểm tra quyền trước khi gắn nhãn)
# R / RW / R* / X / A  cho L1..L5
# ============================================================
PERMISSION_MATRIX = {
    "sales_staff":         ["R",  "R",  "X",  "X",  "X"],
    "sales_manager":       ["R",  "R",  "R",  "R",  "X"],
    "accountant":          ["R",  "R",  "R",  "R*", "X"],
    "finance_manager":     ["R",  "R",  "R",  "R",  "X"],
    "procurement_staff":   ["R",  "R",  "X",  "X",  "X"],
    "procurement_manager": ["R",  "R",  "R",  "R",  "X"],
    "hr_staff":            ["R",  "R",  "RW", "R",  "X"],
    "hr_manager":          ["R",  "R",  "RW", "R",  "X"],
    "data_analyst":        ["R",  "R",  "R",  "X",  "X"],
    "senior_analyst":      ["R",  "R",  "R",  "R",  "X"],
    "developer":           ["R",  "RW", "X",  "X",  "X"],
    "data_engineer":       ["R",  "RW", "X",  "X",  "X"],
    "security_analyst":    ["R",  "R",  "R",  "R",  "R"],
    "security_manager":    ["R",  "R",  "R",  "R",  "R"],
    "junior_dba":          ["RW", "RW", "RW", "R",  "R"],
    "senior_dba":          ["RW", "RW", "RW", "RW", "RW"],
    "admin":               ["A",  "A",  "A",  "A",  "A"],
}

TABLE_SECURITY_LEVEL = {
    "region": 1, "nation": 1, "part": 1,
    "customer": 2, "orders": 2, "lineitem": 2, "supplier": 2, "partsupp": 2,
    "hr.department": 3, "hr.employee": 3,
    "hr.payroll": 4, "sys.system_monitor_log": 4,
    "sys.db_account": 5, "sys.system_security_log": 5,
}

def can_access(job_role, table, write=False):
    level = TABLE_SECURITY_LEVEL.get(table, 1) - 1
    perm  = PERMISSION_MATRIX.get(job_role, ["X"]*5)[level]
    if perm == "X":  return False
    if perm == "A":  return True
    if write:        return perm == "RW"
    return perm in ("R", "RW", "R*")


# ============================================================
# G5 ELIGIBILITY (ngoài giờ)
# ============================================================
G5_RATE = {
    "junior_dba":    0.40,
    "senior_dba":    0.50,
    "admin":         0.40,
    "developer":     0.10,    # thứ 6 deploy
    "data_engineer": 0.15,
}
WFH_RATE = 0.05   # 5% mọi user có thể query G5

def should_do_g5(job_role: str, sim_date: date):
    base = G5_RATE.get(job_role, 0)
    if job_role in ("developer", "data_engineer") and not is_friday(sim_date):
        base = 0
    return random.random() < max(base, WFH_RATE)

# ============================================================
# SESSION TIMING
# ============================================================
FRAME_HOURS = {
    "G1": (6,  9),
    "G2": (9,  12),
    "G3": (12, 14),
    "G4": (14, 18),
    "G5": None  # ngoài giờ: 00–06 hoặc 18–24 trong cùng sim_date
    # "G5": (18, 6),  # Sửa lại bảo dưỡng từ 18h hôm trc- 6h hôm sau
}

# số session_count theo ngày+ role
SESSION_COUNT = {
    "sales_staff":         {"weekday": (2, 3), "weekend": (1, 1)},
    "sales_manager":       {"weekday": (2, 3), "weekend": (1, 2)},

    "accountant":          {"weekday": (2, 4), "weekend": (1, 2)},
    "finance_manager":     {"weekday": (2, 3), "weekend": (1, 2)},

    "procurement_staff":   {"weekday": (2, 3), "weekend": (0, 0)},
    "procurement_manager": {"weekday": (2, 3), "weekend": (1, 2)},

    "hr_staff":            {"weekday": (2, 4), "weekend": (1, 2)},
    "hr_manager":          {"weekday": (2, 3), "weekend": (1, 2)},

    "data_analyst":        {"weekday": (3, 4), "weekend": (1, 2)},
    "senior_analyst":      {"weekday": (3, 4), "weekend": (1, 2)},

    "developer":           {"weekday": (2, 4), "weekend": (0, 1)},
    "data_engineer":       {"weekday": (3, 4), "weekend": (1, 2)},

    "security_analyst":    {"weekday": (3, 5), "weekend": (2, 3)},
    "security_manager":    {"weekday": (2, 4), "weekend": (1, 2)},

    "junior_dba":          {"weekday": (3, 5), "weekend": (2, 3)},
    "senior_dba":          {"weekday": (3, 5), "weekend": (2, 4)},
    "admin":               {"weekday": (2, 4), "weekend": (1, 2)},
}


# Edit to remain frame G5 and not change to another day
def random_time_in_frame(sim_date: date, frame: str):
    if frame == "G5":
        # G5 cùng ngày: hoặc rạng sáng, hoặc buổi tối
        if random.random() < 0.4:
            h = random.randint(0, 5)      # 00:00–05:59
        else:
            h = random.randint(18, 23)    # 18:00–23:59
        d = sim_date
    else:
        h_start, h_end = FRAME_HOURS[frame]
        h = random.randint(h_start, h_end - 1)
        d = sim_date

    m = random.randint(0, 59)
    s = random.randint(0, 59)
    return datetime(d.year, d.month, d.day, h, m, s)


def get_normal_frame(job_role: str):
    """Frame mà role này hay làm việc nhất trong giờ hành chính."""
    if job_role in ("junior_dba", "senior_dba", "admin"):
        return random.choice(["G1", "G2", "G3", "G4"])
    if job_role in ("data_analyst", "senior_analyst"):
        return random.choice(["G1", "G2", "G3", "G4"])
    return random.choice(["G2", "G3", "G4"])


# ============================================================
# AUDIT LOG TABLE (tạm thời lưu vào DB để parse sau)
# ============================================================
CREATE_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS public.audit_log_raw (
    event_id            TEXT,
    timestamp           TIMESTAMP,
    db_user             TEXT,
    session_id          TEXT,
    sql_statement       TEXT,
    tables_involved     TEXT,
    status              TEXT,
    rows_affected       INTEGER,
    graph_frame         TEXT,
    sim_date            DATE,
    is_after_hours      BOOLEAN,
    session_start_time  TIMESTAMP,
    session_ip          TEXT,
    session_event_index INTEGER,
    scenario_type       TEXT DEFAULT 'normal'
);
"""

