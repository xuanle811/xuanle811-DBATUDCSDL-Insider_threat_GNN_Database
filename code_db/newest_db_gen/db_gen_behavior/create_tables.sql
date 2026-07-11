-- ============================================================
-- SCHEMA: hr (nhân sự)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS hr;

CREATE TABLE hr.department (
    department_id       SERIAL,
    department_name     VARCHAR(100)    NOT NULL,
    business_function   VARCHAR(200),
    description         TEXT
);

CREATE TABLE hr.employee (
    employee_id         SERIAL,
    full_name           VARCHAR(200)    NOT NULL,
    department_id       INTEGER,
    job_role            VARCHAR(100),
    hire_date           DATE,
    manager_id          INTEGER,
    employment_status   VARCHAR(50)     -- ACTIVE, RESIGNED, TERMINATED
);

CREATE TABLE hr.payroll (
    payroll_id          SERIAL,
    employee_id         INTEGER,
    payroll_month       DATE,           -- lưu dạng YYYY-MM-01
    base_salary         NUMERIC(15,2),
    bonus               NUMERIC(15,2),
    tax                 NUMERIC(15,2),
    deduction           NUMERIC(15,2),
    net_salary          NUMERIC(15,2)
);

-- ============================================================
-- SCHEMA: sys (hệ thống)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS sys;

CREATE TABLE sys.db_account (
    employee_id         INTEGER,
    db_user             VARCHAR(100),   -- PK sau này
    db_role             VARCHAR(50),    -- readonly, readwrite, admin
    clearance_level     INTEGER,        -- 1=PUBLIC, 2=INTERNAL, 3=MEDIUM, 4=HIGH, 5=CRITICAL
    is_privileged       BOOLEAN         DEFAULT FALSE,
    account_status      VARCHAR(50)     DEFAULT 'ACTIVE',  -- ACTIVE, SUSPENDED, DISABLED
    created_at          TIMESTAMP,
    last_login          TIMESTAMP
);

CREATE TABLE sys.system_monitor_log (
    monitor_id          SERIAL,
    event_time          TIMESTAMP       NOT NULL,
    db_user             VARCHAR(100),
    session_id          VARCHAR(100),
    cpu_utilization     NUMERIC(5,2),   -- %
    memory_utilization  NUMERIC(5,2),   -- %
    active_connections  INTEGER,
    disk_read_kb        NUMERIC(15,2),
    disk_write_kb       NUMERIC(15,2),
    backup_status       VARCHAR(50)     -- OK, FAILED, RUNNING, SKIPPED
);

CREATE TABLE sys.system_security_log (
    alert_id            SERIAL,
    event_time          TIMESTAMP       NOT NULL,
    db_user             VARCHAR(100),
    event_type          VARCHAR(50),    -- FAILED_LOGIN, UNAUTHORIZED_ACCESS,
                                        -- PRIVILEGE_CHANGE, L5_ACCESS, L5_TAMPERING
    severity_index      INTEGER,        -- 1 (thấp) → 5 (critical)
    source_ip           VARCHAR(50),
    target_object       VARCHAR(200),   -- tên bảng hoặc object bị tác động
    related_event_id    VARCHAR(100),   -- liên kết với event_id trong audit_log
    related_session_id TEXT, --liên kết với session_id trong audit_log
    description         TEXT
);
