"""
seed_data.py — Sinh data cho 6 bảng hr/sys trong insider_db
             + Xuất table_metadata.csv, table_relations.csv, user_metadata.csv
Requires: psycopg2-binary, faker
Install : pip install psycopg2-binary faker --break-system-packages
Run     : python seed_data.py
"""

import psycopg2
import random
import csv
from faker import Faker
from datetime import date, datetime, timedelta

fake = Faker()
random.seed(42)
Faker.seed(42)

# ── Kết nối DB ──────────────────────────────────────────────
conn = psycopg2.connect(
    dbname="insider_db",
    user="postgres",
    password="123456",
    host="localhost",
    port=5432
)
cur = conn.cursor()

# ============================================================
# TRUNCATE toàn bộ 6 bảng (chạy lại không bị nhân đôi)
# ============================================================
print("Truncating existing data ...")
cur.execute("""
    TRUNCATE
        sys.system_security_log,
        sys.system_monitor_log,
        sys.db_account,
        hr.payroll,
        hr.employee,
        hr.department
    RESTART IDENTITY CASCADE;
""")
conn.commit()

# ============================================================
# PERMISSION MATRIX
# Mỗi role → quyền trên từng security level [L1, L2, L3, L4, L5]
# R=read, RW=read/write, X=no access, A=admin, R*=restricted read
# ============================================================
PERMISSION_MATRIX = {
    #                         L1    L2    L3    L4    L5
    "sales_staff":           ["R",  "R",  "X",  "X",  "X"],
    "sales_manager":         ["R",  "R",  "R",  "R",  "X"],
    "accountant":            ["R",  "R",  "R",  "R*", "X"],
    "finance_manager":       ["R",  "R",  "R",  "R",  "X"],
    "procurement_staff":     ["R",  "R",  "X",  "X",  "X"],
    "procurement_manager":   ["R",  "R",  "R",  "R",  "X"],
    "hr_staff":              ["R",  "R",  "RW", "R",  "X"],
    "hr_manager":            ["R",  "R",  "RW", "R",  "X"],
    "data_analyst":          ["R",  "R",  "R",  "X",  "X"],
    "senior_analyst":        ["R",  "R",  "R",  "R",  "X"],
    "developer":             ["R",  "RW", "X",  "X",  "X"],
    "data_engineer":         ["R",  "RW", "X",  "X",  "X"],
    "security_analyst":      ["R",  "R",  "R",  "R",  "R"],
    "security_manager":      ["R",  "R",  "R",  "R",  "R"],
    "junior_dba":            ["RW", "RW", "RW", "R",  "R"],
    "senior_dba":            ["RW", "RW", "RW", "RW", "RW"],
    "admin":                 ["A",  "A",  "A",  "A",  "A"],
}

# ── Security level từng bảng ─────────────────────────────────
TABLE_SECURITY_LEVEL = {
    # L1 - Public
    "region":               1,
    "nation":               1,
    "part":                 1,
    # L2 - Business
    "customer":             2,
    "orders":               2,
    "lineitem":             2,
    "supplier":             2,
    "partsupp":             2,
    # L3 - Internal Sensitive
    "department":           3,
    "employee":             3,
    # L4 - Confidential
    "payroll":              4,
    "system_monitor_log":   4,
    # L5 - Restricted
    "db_account":           5,
    "system_security_log":  5,
}

# ── Mapping bảng → schema thực trong DB ─────────────────────
TABLE_SCHEMA = {
    "region": "public", "nation": "public", "part": "public",
    "customer": "public", "orders": "public", "lineitem": "public",
    "supplier": "public", "partsupp": "public",
    "department": "hr",
    "employee": "hr",
    "payroll": "hr",
    "db_account": "sys",
    "system_monitor_log": "sys",
    "system_security_log": "sys",
}

# ── Table relations (FK + cross-schema, bỏ LOGICAL) ──────────
# (source_table, source_column, target_table, target_column, relation_type, cardinality)
TABLE_RELATIONS = [
    # TPC-H
    ("orders",              "o_custkey",    "customer",     "c_custkey",    "FK", "many-to-one"),
    ("lineitem",            "l_orderkey",   "orders",       "o_orderkey",   "FK", "many-to-one"),
    ("lineitem",            "l_partkey",    "part",         "p_partkey",    "FK", "many-to-one"),
    ("lineitem",            "l_suppkey",    "supplier",     "s_suppkey",    "FK", "many-to-one"),
    ("partsupp",            "ps_partkey",   "part",         "p_partkey",    "FK", "many-to-one"),
    ("partsupp",            "ps_suppkey",   "supplier",     "s_suppkey",    "FK", "many-to-one"),
    ("supplier",            "s_nationkey",  "nation",       "n_nationkey",  "FK", "many-to-one"),
    ("customer",            "c_nationkey",  "nation",       "n_nationkey",  "FK", "many-to-one"),
    ("nation",              "n_regionkey",  "region",       "r_regionkey",  "FK", "many-to-one"),
    # HR internal
    ("employee",            "department_id","department",   "department_id","FK", "many-to-one"),
    ("payroll",             "employee_id",  "employee",     "employee_id",  "FK", "many-to-one"),
    ("db_account",          "employee_id",  "employee",     "employee_id",  "FK", "one-to-one"),
    # Cross-schema
    ("system_monitor_log",  "db_user",      "db_account",   "db_user",      "REF","many-to-one"),
    ("system_security_log", "db_user",      "db_account",   "db_user",      "REF","many-to-one"),
]

# ── Cấu trúc phòng ban ───────────────────────────────────────
DEPARTMENTS = [
    ("Sales Department",            "Revenue generation and customer relations", [
        ("sales_staff",             20, 1),
        ("sales_manager",            3, 3),
    ]),
    ("Finance Department",          "Financial planning and accounting", [
        ("accountant",              10, 2),
        ("finance_manager",          2, 3),
    ]),
    ("Procurement Department",      "Vendor management and purchasing", [
        ("procurement_staff",        8, 1),
        ("procurement_manager",      2, 3),
    ]),
    ("HR Department",               "Human resources and talent management", [
        ("hr_staff",                 6, 2),
        ("hr_manager",               1, 3),
    ]),
    ("Data Analytics Department",   "Data analysis and business intelligence", [
        ("data_analyst",             5, 2),
        ("senior_analyst",           2, 3),
    ]),
    ("IT / Engineering Department", "Software development and infrastructure", [
        ("developer",                8, 2),
        ("data_engineer",            3, 2),
    ]),
    ("Security Department",         "Cybersecurity and compliance monitoring", [
        ("security_analyst",         2, 2),
        ("security_manager",         1, 3),
    ]),
    ("Database Administration",     "Database management and administration", [
        ("junior_dba",               3, 4),
        ("senior_dba",               2, 4),
        ("admin",                    1, 4),
    ]),
]

# ── Helpers ──────────────────────────────────────────────────
def get_db_role(job_role):
    perms = PERMISSION_MATRIX.get(job_role, ["R"] * 5)
    if "A"  in perms: return "admin"
    if "RW" in perms: return "readwrite"
    return "readonly"

def is_privileged(job_role):
    return job_role in ("junior_dba", "senior_dba", "admin")

# Chỉ các bảng L5 để L5_ACCESS/L5_TAMPERING trỏ đúng
L5_TABLES = [t for t, lvl in TABLE_SECURITY_LEVEL.items() if lvl == 5]

SIM_START      = date(2026, 1, 1)
PAYROLL_MONTHS = [date(2025, 12, 1), date(2026, 1, 1), date(2026, 2, 1)]

# ============================================================
# 1. hr.department
# ============================================================
print("[1/6] Inserting hr.department ...")
dept_ids = {}
for dept_name, biz_func, _ in DEPARTMENTS:
    cur.execute("""
        INSERT INTO hr.department (department_name, business_function, description)
        VALUES (%s, %s, %s) RETURNING department_id
    """, (dept_name, biz_func, f"Handles {biz_func.lower()}"))
    dept_ids[dept_name] = cur.fetchone()[0]

# ============================================================
# 2. hr.employee
# ============================================================
print("[2/6] Inserting hr.employee ...")
employees = []
for dept_name, _, roles in DEPARTMENTS:
    dept_id  = dept_ids[dept_name]
    managers = []
    for job_role, count, clearance in roles:
        for _ in range(count):
            hire_date = fake.date_between(start_date="-5y", end_date=SIM_START)
            cur.execute("""
                INSERT INTO hr.employee
                    (full_name, department_id, job_role, hire_date,
                     manager_id, employment_status)
                VALUES (%s, %s, %s, %s, NULL, 'ACTIVE')
                RETURNING employee_id
            """, (fake.name(), dept_id, job_role, hire_date))
            emp_id = cur.fetchone()[0]
            employees.append({
                "employee_id": emp_id,
                "department":  dept_name,
                "job_role":    job_role,
                "clearance":   clearance,
                "permissions": PERMISSION_MATRIX.get(job_role, ["R"] * 5),
            })
            if "manager" in job_role or job_role in ("senior_dba", "admin"):
                managers.append(emp_id)

    staff_in_dept = [e["employee_id"] for e in employees
                     if e["department"] == dept_name]
    if managers:
        for emp_id in staff_in_dept:
            mgr = random.choice(managers)
            if mgr != emp_id:
                cur.execute(
                    "UPDATE hr.employee SET manager_id=%s WHERE employee_id=%s",
                    (mgr, emp_id))

# ============================================================
# 3. hr.payroll
# ============================================================
print("[3/6] Inserting hr.payroll ...")
SALARY_RANGE = {
    1: (8_000_000,  15_000_000),
    2: (15_000_000, 25_000_000),
    3: (25_000_000, 45_000_000),
    4: (45_000_000, 80_000_000),
}
for emp in employees:
    lo, hi = SALARY_RANGE[emp["clearance"]]
    base   = random.randint(lo, hi)
    for month in PAYROLL_MONTHS:
        bonus     = random.randint(0, int(base * 0.3))
        tax       = int((base + bonus) * 0.1)
        deduction = random.randint(0, 500_000)
        net       = base + bonus - tax - deduction
        cur.execute("""
            INSERT INTO hr.payroll
                (employee_id, payroll_month, base_salary, bonus,
                 tax, deduction, net_salary)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (emp["employee_id"], month, base, bonus, tax, deduction, net))

# ============================================================
# 4. sys.db_account
# ============================================================
print("[4/6] Inserting sys.db_account ...")
db_users = []
for emp in employees:
    role_slug  = emp["job_role"].replace("/", "_").replace(" ", "_")
    db_user    = f"{role_slug}_{emp['employee_id']:03d}"
    db_role    = get_db_role(emp["job_role"])
    priv       = is_privileged(emp["job_role"])
    created_at = datetime.combine(
        fake.date_between(start_date="-3y", end_date=SIM_START),
        datetime.min.time()
    )
    last_login = fake.date_time_between(start_date="-30d", end_date="now")
    cur.execute("""
        INSERT INTO sys.db_account
            (employee_id, db_user, db_role, clearance_level,
             is_privileged, account_status, created_at, last_login)
        VALUES (%s, %s, %s, %s, %s, 'ACTIVE', %s, %s)
    """, (emp["employee_id"], db_user, db_role,
          emp["clearance"], priv, created_at, last_login))
    db_users.append({
        "db_user":     db_user,
        "job_role":    emp["job_role"],
        "clearance":   emp["clearance"],
        "permissions": emp["permissions"],
    })

# ============================================================
# 5. sys.system_monitor_log
# ============================================================
print("[5/6] Inserting sys.system_monitor_log ...")
for day_offset in range(7):
    log_date = SIM_START + timedelta(days=day_offset)
    sample   = random.sample(db_users, k=min(30, len(db_users)))
    for u in sample:
        for session_num in range(random.randint(1, 3)):
            session_id = f"sess_{u['db_user']}_{log_date}_{session_num}"
            event_time = datetime.combine(log_date, datetime.min.time()) + \
                         timedelta(hours=random.randint(8, 17),
                                   minutes=random.randint(0, 59))
            cur.execute("""
                INSERT INTO sys.system_monitor_log
                    (event_time, db_user, session_id, cpu_utilization,
                     memory_utilization, active_connections,
                     disk_read_kb, disk_write_kb, backup_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                event_time, u["db_user"], session_id,
                round(random.uniform(5, 80), 2),
                round(random.uniform(20, 90), 2),
                random.randint(1, 50),
                round(random.uniform(100, 5000), 2),
                round(random.uniform(50, 2000), 2),
                random.choice(["OK", "OK", "OK", "FAILED", "SKIPPED"])
            ))

# ============================================================
# 6. sys.system_security_log
# ============================================================
print("[6/6] Inserting sys.system_security_log ...")
EVENT_TYPES = ["FAILED_LOGIN", "UNAUTHORIZED_ACCESS", "PRIVILEGE_CHANGE",
               "L5_ACCESS", "L5_TAMPERING"]

# 4 mức severity: 1=low, 2=medium, 3=high, 4=critical
SEVERITY = {
    "FAILED_LOGIN":         1,   # low   — thường gặp, chưa nguy hiểm
    "UNAUTHORIZED_ACCESS":  2,   # medium
    "PRIVILEGE_CHANGE":     3,   # high
    "L5_ACCESS":            3,   # high  — truy cập đúng người, bình thường
    "L5_TAMPERING":         4,   # critical
}

# Target pool theo event type
# FAILED_LOGIN → gắn với database connection, không phải bảng cụ thể
EVENT_TARGET_POOL = {
    "FAILED_LOGIN":         ["database_connection"],
    "UNAUTHORIZED_ACCESS":  [t for t, l in TABLE_SECURITY_LEVEL.items() if l >= 3],
    "PRIVILEGE_CHANGE":     ["db_account"],
    "L5_ACCESS":            L5_TABLES,
    "L5_TAMPERING":         L5_TABLES,
}

# Users có quyền truy cập L5 (perm_L5 != X)
l5_authorized_users = [
    u for u in db_users
    if u["permissions"][4] in ("R", "RW", "A", "R*")
]

for _ in range(20):
    event_type = random.choices(EVENT_TYPES, weights=[50, 20, 15, 10, 5])[0]
    event_time = fake.date_time_between(
        start_date=SIM_START,
        end_date=SIM_START + timedelta(days=7)
    )

    # L5_ACCESS và L5_TAMPERING trong log normal → chỉ user có quyền L5
    if event_type in ("L5_ACCESS", "L5_TAMPERING"):
        u = random.choice(l5_authorized_users)
    else:
        u = random.choice(db_users)

    target = random.choice(EVENT_TARGET_POOL[event_type])
    cur.execute("""
        INSERT INTO sys.system_security_log
            (event_time, db_user, event_type, severity_index,
             source_ip, target_object, related_event_id, description)
        VALUES (%s, %s, %s, %s, %s, %s, NULL, %s)
    """, (
        event_time, u["db_user"], event_type, SEVERITY[event_type],
        fake.ipv4_private(), target,
        f"{event_type} by {u['db_user']} on {target}"
    ))

conn.commit()
cur.close()
conn.close()
print("\n✓ Insert 6 bảng hoàn tất")

# ============================================================
# XUẤT table_metadata.csv
# ============================================================
print("\nXuất table_metadata.csv ...")
conn2 = psycopg2.connect(dbname="insider_db", user="postgres",
                          password="123456", host="localhost", port=5432)
cur2  = conn2.cursor()

LEVEL_LABEL = {
    1: "PUBLIC",
    2: "BUSINESS",
    3: "INTERNAL_SENSITIVE",
    4: "CONFIDENTIAL",
    5: "RESTRICTED",
}
meta_rows = []
for table, level in TABLE_SECURITY_LEVEL.items():
    schema = TABLE_SCHEMA[table]
    try:
        cur2.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        row_count = cur2.fetchone()[0]
    except Exception:
        row_count = 0
    meta_rows.append({
        "table_name":     table,
        "schema":         schema,
        "security_level": level,
        "level_label":    LEVEL_LABEL[level],
        "row_count":      row_count,
    })

cur2.close()
conn2.close()

with open("table_metadata.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["table_name", "schema",
                                       "security_level", "level_label", "row_count"])
    w.writeheader()
    w.writerows(meta_rows)
print("✓ table_metadata.csv")

# ============================================================
# XUẤT table_relations.csv
# ============================================================
print("Xuất table_relations.csv ...")
with open("table_relations.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["source_table", "source_column",
                                       "target_table", "target_column",
                                       "relation_type", "cardinality"])
    w.writeheader()
    for src_t, src_c, tgt_t, tgt_c, rtype, card in TABLE_RELATIONS:
        w.writerow({
            "source_table":  src_t,
            "source_column": src_c,
            "target_table":  tgt_t,
            "target_column": tgt_c,
            "relation_type": rtype,
            "cardinality":   card,
        })
print("✓ table_relations.csv")

# ============================================================
# XUẤT user_metadata.csv
# ============================================================
print("Xuất user_metadata.csv ...")
with open("user_metadata.csv", "w", newline="") as f:
    fieldnames = ["db_user", "job_role", "clearance_level", "db_role",
                  "is_privileged", "perm_L1", "perm_L2", "perm_L3",
                  "perm_L4", "perm_L5"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for u in db_users:
        perms = u["permissions"]
        w.writerow({
            "db_user":         u["db_user"],
            "job_role":        u["job_role"],
            "clearance_level": u["clearance"],
            "db_role":         get_db_role(u["job_role"]),
            "is_privileged":   is_privileged(u["job_role"]),
            "perm_L1":         perms[0],
            "perm_L2":         perms[1],
            "perm_L3":         perms[2],
            "perm_L4":         perms[3],
            "perm_L5":         perms[4],
        })
print("✓ user_metadata.csv")
print("\n✓ Hoàn tất — 3 file CSV đã được xuất cùng thư mục script")