"""
attack_config.py — Cấu hình timeline, attacker users, phase mapping cho 4 scenarios.

Thiết kế dataset:
- Ngày 1–60 : normal only
- Ngày 61–90: normal background + attack events
- 4 scenario chính chạy bán song song, có offset lệch nhau
"""
from datetime import date

# ── Simulation period ────────────────────────────────────────
SIM_START = date(2026, 1, 1)
NORMAL_DAYS = 60
ATTACK_DAYS = 30
TOTAL_DAYS = NORMAL_DAYS + ATTACK_DAYS  # 90 ngày

# ── Timeline từng scenario (day_offset tính từ SIM_START, 1-indexed) ────
# Không scenario nào bắt đầu trước ngày 61 để giữ train normal sạch.
SCENARIO_TIMELINE = {
    "S1": {  # Unauthorized Data Exploration
        "start": 61,
        "end": 80,
        "phases": {
            "recon": (61, 65),
            "execution": (66, 80),
        },
    },
    "S2": {  # Sensitive Data Theft
        "start": 63,
        "end": 90,
        "phases": {
            "recon": (63, 68),
            "preparation": (69, 74),
            "execution": (75, 85),
            "cover": (86, 90),
        },
    },
    "S3": {  # Slow-rate Data Theft
        "start": 61,
        "end": 90,
        "phases": {
            "recon": (61, 65),
            "execution": (66, 90),
        },
    },
    "S4": {  # Data Staging & Aggregation
        "start": 67,
        "end": 88,
        "phases": {
            "recon": (67, 71),
            "preparation": (72, 76),
            "execution": (77, 86),
            "cover": (87, 88),
        },
    },
}

# ── Role ứng viên cho attacker từng scenario ─────────────────
# attack_runner sẽ chọn 1 user ACTIVE từ DB cho mỗi scenario.
# Chọn role phải khớp quyền để scenario không bị DENIED ngoài ý muốn.
ATTACKER_ROLES = {
    # S1: quyền thấp/kỹ thuật, thăm dò ngoài phạm vi nên có DENIED vừa phải.
    "S1": ["sales_staff", "developer"],

    # S2: người có lý do tiếp cận payroll, nhưng có thể lạm dụng để theft.
    "S2": ["hr_staff", "accountant"],

    # S3: slow-rate nên hạn chế DENIED; chọn role có thể đọc payroll/L4 hợp lệ.
    "S3": ["accountant", "hr_staff", "finance_manager", "senior_analyst"],

    # S4: staging cần khả năng kỹ thuật/RW hợp lý, không dùng sales_manager.
    "S4": ["developer", "data_engineer", "senior_dba"],
}


def get_phase(scenario: str, day_offset: int) -> str | None:
    """Trả về phase hiện tại của scenario tại day_offset (1-indexed)."""
    timeline = SCENARIO_TIMELINE.get(scenario)
    if not timeline:
        return None
    if not (timeline["start"] <= day_offset <= timeline["end"]):
        return None
    for phase, (start, end) in timeline["phases"].items():
        if start <= day_offset <= end:
            return phase
    return None


def get_active_scenarios(day_offset: int) -> dict[str, str]:
    """Trả về {scenario: phase} cho tất cả scenarios active ngày đó."""
    active: dict[str, str] = {}
    for scenario in SCENARIO_TIMELINE:
        phase = get_phase(scenario, day_offset)
        if phase:
            active[scenario] = phase
    return active


def sim_date_to_offset(sim_date: date) -> int:
    """Đổi sim_date sang day_offset 1-indexed."""
    return (sim_date - SIM_START).days + 1
