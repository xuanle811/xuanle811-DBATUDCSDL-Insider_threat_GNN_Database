"""
02_graph_construction_dbaware.py
================================
Build DB-aware Heterogeneous Graph from sampled CERT v4.2 parquet output.

Expected input:
    output/users_selected.csv
    output/logon/part_*.parquet
    output/file/part_*.parquet
    output/http/part_*.parquet
    output/email/part_*.parquet
    output/device/part_*.parquet
    cert_data/LDAP/*.csv
    cert_data/psychometric.csv     (optional)

Output:
    output/graph_data.pt

Schema:
    Node types:
        user, session, pc, file_object, web_resource, email,
        device, topic, role, department

    Edge types:
        (user, logs_on, session)
        (session, on_pc, pc)

        (user, uses_device, device)
        (device, attached_to, pc)
        (pc, has_usb_activity, user)   ← NEW (PDF §1): USB edge gắn PC→User
                                          edge_attr: [connect_ratio, disconnect_ratio, after_hours_ratio]

        (user, exports_file, file_object)
                                          edge_attr: [tw_norm, response_time, failed_attempts]  ← NEW (PDF §3)
        (file_object, has_topic, topic)

        (user, visits, web_resource)      web_resource node = Domain (không phải URL thô)  ← NEW (PDF §1)
                                          edge_attr: [tw_norm, response_time, failed_attempts]  ← NEW (PDF §3)
        (web_resource, has_topic, topic)

        (user, sends_email, email)
        (email, sent_to, user)
        (email, has_topic, topic)

        (user, has_role, role)
        (user, belongs_to, department)
        (user, reports_to, user)              # if LDAP has supervisor/manager column

Temporal Subgraph (PDF §2):
    - Mỗi event được gán time_window_id = (ts - t0).days // TIME_WINDOW_DAYS
    - Thuộc tính tw_norm lưu trong node.x của web_resource để mô hình
      học temporal dynamics qua chuỗi subgraph G1, G2, ..., GT

Notes:
    - Baseline_Behavior is NOT a node. It is appended to user features.
    - Psychometric O,C,E,A,N are appended to user features if available.
    - web_resource node gom theo Domain (không phải từng URL) để giảm sparsity.
"""

import gc
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import json


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()
CERT_DIR = ROOT_DIR / "cert_data"
LDAP_DIR = CERT_DIR / "LDAP"
PSY_FILE = CERT_DIR / "psychometric.csv"
OUT_DIR = ROOT_DIR / "output"
OUT_PT = OUT_DIR / "model"/"graph_data.pt"
SHARED_PC_THRESHOLD = 3


# ── Config ────────────────────────────────────────────────────────────────────
WORK_START_HOUR = 8
WORK_END_HOUR = 18
TOPIC_MAX_PER_EVENT = 8
TOPIC_MIN_LEN = 2

# ── Temporal Subgraph Config (PDF mục 2) ──────────────────────────────────────
# Thay vì 1 đồ thị khổng lồ, chia dữ liệu log thành chuỗi đồ thị con G1..GT
# theo cửa sổ thời gian TIME_WINDOW_DAYS (ví dụ: 1 ngày = 1 subgraph).
# Khi build graph, mỗi event được gán time_window_id = (timestamp - t0).days // TIME_WINDOW_DAYS
# và lưu vào node/edge attribute để mô hình có thể học temporal dynamics.
TIME_WINDOW_DAYS = 1   # độ rộng mỗi cửa sổ thời gian (ngày). Đặt = 1 để tách theo ngày làm việc.


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

# normalize user id: strip, upper, ensure string type
def normalize_user_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

# Try multiple candidate column names and return the first match. This helps adapt to varying schemas.
def safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def extract_domain(url) -> str:
    """
    PDF mục 1 — Website: Gom nhóm theo Domain Name thay vì URL thô.
    Mục đích: Giảm độ thưa thớt (sparsity) của web_resource node.
    Ví dụ: 'https://news.bbc.co.uk/sport/football' → 'news.bbc.co.uk'
    Nếu parse thất bại (URL không hợp lệ) → trả về chuỗi gốc để tránh mất data.
    """
    if pd.isna(url):
        return "unknown"
    try:
        parsed = urlparse(str(url).strip())
        domain = parsed.netloc or parsed.path  # netloc chứa hostname
        domain = domain.lower().strip()
        # Bỏ tiền tố www. để chuẩn hoá: 'www.bbc.co.uk' → 'bbc.co.uk'
        if domain.startswith("www."):
            domain = domain[4:]
        return domain if domain else str(url).strip().lower()
    except Exception:
        return str(url).strip().lower()


def assign_time_window(ts: pd.Series, t0=None) -> np.ndarray:
    """
    PDF mục 2 — Temporal Subgraph: gán time_window_id cho mỗi event.
    time_window_id = số nguyên đại diện cho cửa sổ thời gian chứa event đó.
    Công thức: (timestamp - t0).days // TIME_WINDOW_DAYS
    Dùng làm attribute trong node/edge để mô hình phân biệt các subgraph G_t.
    t0: thời điểm gốc (mặc định = timestamp nhỏ nhất trong chuỗi).
    """
    if ts is None or len(ts) == 0:
        return np.zeros(0, dtype=int)
    ts_clean = pd.to_datetime(ts, errors="coerce")
    if t0 is None:
        t0 = ts_clean.min()
    delta_days = (ts_clean - t0).dt.days.fillna(0).astype(int)
    return (delta_days // TIME_WINDOW_DAYS).values


def read_parquet_dir(name: str) -> pd.DataFrame:
    p_dir = OUT_DIR / name
    if not p_dir.exists():
        log.warning(f"Không tìm thấy thư mục parquet: {p_dir}")
        return pd.DataFrame()

    files = sorted(p_dir.glob("*.parquet"))
    if not files:
        log.warning(f"Không có part_*.parquet trong: {p_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    return df


def load_ldap_df() -> pd.DataFrame:
    if not LDAP_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục LDAP: {LDAP_DIR.resolve()}")

    csv_files = sorted(LDAP_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Không có file .csv nào trong: {LDAP_DIR.resolve()}")

    dfs = [] 
    for f in csv_files:
        log.info(f"Đọc LDAP file: {f}")
        dfs.append(pd.read_csv(f, low_memory=False)) # df: 1 list chứa nhiều dataframe

    df = pd.concat(dfs, ignore_index=True) # nối tất cả dataframe trong list thành 1 dataframe duy nhất
    del dfs
    gc.collect()
    return df


def load_psychometric_df() -> pd.DataFrame:
    if not PSY_FILE.exists():
        log.warning(f"Không tìm thấy psychometric.csv: {PSY_FILE} — bỏ qua OCEAN features.")
        return pd.DataFrame()
    return pd.read_csv(PSY_FILE, low_memory=False)


def to_datetime_col(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def hour_feature(ts: pd.Series) -> np.ndarray:
    if ts is None or len(ts) == 0:
        return np.zeros(0, dtype=float)
    return ts.dt.hour.fillna(0).astype(float).values / 23.0


def is_after_hours(ts: pd.Series) -> np.ndarray:
    if ts is None or len(ts) == 0:
        return np.zeros(0, dtype=float)
    h = ts.dt.hour.fillna(12).astype(int)
    return ((h < WORK_START_HOUR) | (h >= WORK_END_HOUR)).astype(float).values


def safe_log1p_norm(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.log1p(np.maximum(arr, 0.0))
    mx = arr.max() if len(arr) else 0.0
    return arr / (mx + 1e-8) if mx > 0 else arr


def split_topics(content) -> list[str]:
    if pd.isna(content):
        return []
    text = str(content)
    # file.csv content often starts with a hex header. Keep readable keywords only.
    tokens = re.split(r"\s+", text.strip())
    topics = []
    for t in tokens:
        t = t.strip().lower()
        if not t:
            continue
        if len(t) < TOPIC_MIN_LEN:
            continue
        if re.fullmatch(r"[0-9a-fA-F]{6,}", t):
            continue
        if not re.search(r"[a-zA-Z]", t):
            continue
        topics.append(t)
        if len(topics) >= TOPIC_MAX_PER_EVENT:
            break
    return topics


class SafeLabelEncoder:
    def __init__(self):
        self.le = LabelEncoder()
        self._classes: set | None = None

    def fit(self, values):
        vals = [str(v) if pd.notna(v) else "unknown" for v in values]
        if not vals:
            vals = ["unknown"]
        self.le.fit(vals)
        self._classes = set(self.le.classes_)
        return self

    def transform(self, values):
        if self._classes is None:
            raise RuntimeError("SafeLabelEncoder must be fitted first.")
        vals = [str(v) if pd.notna(v) else "unknown" for v in values]
        safe = [v if v in self._classes else "__unknown__" for v in vals]
        if "__unknown__" in safe and "__unknown__" not in self._classes:
            self.le.classes_ = np.append(self.le.classes_, "__unknown__")
            self._classes.add("__unknown__")
        return self.le.transform(safe)

    def fit_transform(self, values):
        return self.fit(values).transform(values)

    def __len__(self):
        return len(self.le.classes_)


class NodeIndexer:
    def __init__(self):
        self.map: dict[str, int] = {}
        self.items: list[str] = []

    def add(self, key) -> int:
        key = str(key)
        if key not in self.map:
            self.map[key] = len(self.items)
            self.items.append(key)
        return self.map[key]

    def get(self, key):
        return self.map.get(str(key))

    def __len__(self):
        return len(self.items)


def edge_tensor(src: list[int], dst: list[int]) -> torch.Tensor:
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2.1 — Load and map raw sampled data
# ══════════════════════════════════════════════════════════════════════════════

def load_and_map() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}

    # logon.csv -> session_events
    df = read_parquet_dir("logon")
    if not df.empty:
        rename = {}
        if "id" in df.columns: rename["id"] = "session_id"
        if "date" in df.columns: rename["date"] = "timestamp"
        if "user" in df.columns: rename["user"] = "user_id"
        if "pc" in df.columns: rename["pc"] = "pc_id"
        if "activity" in df.columns: rename["activity"] = "activity"
        df = df.rename(columns=rename)
        if "user_id" in df.columns:
            df["user_id"] = normalize_user_id_series(df["user_id"])
        df = to_datetime_col(df) # đảm bảo cột timestamp là datetime để sort và extract hour sau này, nếu có
        result["session_events"] = df
        log.info(f"session_events: {len(df):,} dòng")

    # file.csv -> file_events / data export events
    df = read_parquet_dir("file")
    if not df.empty:
        rename = {}
        if "id" in df.columns: rename["id"] = "file_id"
        if "date" in df.columns: rename["date"] = "timestamp"
        if "user" in df.columns: rename["user"] = "user_id"
        if "pc" in df.columns: rename["pc"] = "pc_id"
        if "filename" in df.columns: rename["filename"] = "filename"
        if "content" in df.columns: rename["content"] = "content_keywords"
        df = df.rename(columns=rename)
        if "user_id" in df.columns:
            df["user_id"] = normalize_user_id_series(df["user_id"])
        df = to_datetime_col(df)
        result["file_events"] = df
        log.info(f"file_events: {len(df):,} dòng")

    # http.csv -> web_events / external access events
    df = read_parquet_dir("http")
    if not df.empty:
        rename = {}
        if "id" in df.columns: rename["id"] = "url_id"
        if "date" in df.columns: rename["date"] = "timestamp"
        if "user" in df.columns: rename["user"] = "user_id"
        if "pc" in df.columns: rename["pc"] = "pc_id"
        if "url" in df.columns: rename["url"] = "url"
        if "content" in df.columns: rename["content"] = "content_keywords"
        df = df.rename(columns=rename)
        if "user_id" in df.columns:
            df["user_id"] = normalize_user_id_series(df["user_id"])
        df = to_datetime_col(df)
        result["web_events"] = df
        log.info(f"web_events: {len(df):,} dòng")

    # email.csv -> email_events
    df = read_parquet_dir("email")
    if not df.empty:
        rename = {}
        if "id" in df.columns: rename["id"] = "email_id"
        if "date" in df.columns: rename["date"] = "timestamp"
        if "user" in df.columns: rename["user"] = "user_id"
        if "pc" in df.columns: rename["pc"] = "pc_id"
        if "from" in df.columns: rename["from"] = "from_user"
        if "to" in df.columns: rename["to"] = "to"
        if "cc" in df.columns: rename["cc"] = "cc"
        if "bcc" in df.columns: rename["bcc"] = "bcc"
        if "size" in df.columns: rename["size"] = "size"
        if "attachment_count" in df.columns: rename["attachment_count"] = "attachment_count"
        if "content" in df.columns: rename["content"] = "content_keywords"
        df = df.rename(columns=rename)
        if "user_id" in df.columns:
            df["user_id"] = normalize_user_id_series(df["user_id"])
        if "from_user" in df.columns:
            df["from_user"] = normalize_user_id_series(df["from_user"])
        df = to_datetime_col(df)
        result["email_events"] = df
        log.info(f"email_events: {len(df):,} dòng")

    # device.csv -> device_events
    df = read_parquet_dir("device")
    if not df.empty:
        rename = {}
        if "id" in df.columns: rename["id"] = "device_id"
        if "date" in df.columns: rename["date"] = "timestamp"
        if "user" in df.columns: rename["user"] = "user_id"
        if "pc" in df.columns: rename["pc"] = "pc_id"
        if "activity" in df.columns: rename["activity"] = "activity"
        df = df.rename(columns=rename)
        if "user_id" in df.columns:
            df["user_id"] = normalize_user_id_series(df["user_id"])
        df = to_datetime_col(df)
        result["device_events"] = df
        log.info(f"device_events: {len(df):,} dòng")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Feature construction
# ══════════════════════════════════════════════════════════════════════════════
# Function to build baseline features for users based on their activities in different event types
def build_baseline_features(user_ids: list[str], mapped: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = pd.DataFrame({"user_id": user_ids}).set_index("user_id")

    # Session baseline
    sess = mapped.get("session_events", pd.DataFrame())
    if not sess.empty and "user_id" in sess.columns and "timestamp" in sess.columns:
        logon_mask = sess.get("activity", pd.Series([""] * len(sess))).astype(str).str.lower().str.contains("logon")
        logon_df = sess[logon_mask].copy()
        if not logon_df.empty:
            logon_df["hour"] = logon_df["timestamp"].dt.hour
            tmp = logon_df.groupby("user_id")["hour"].mean()
            base["avg_login_hour"] = tmp
            tmp = logon_df.assign(after=is_after_hours(logon_df["timestamp"])).groupby("user_id")["after"].mean()
            base["after_hours_ratio"] = tmp

        # ← THÊM: avg_logout_hour
        logoff_mask = sess.get("activity", pd.Series([""] * len(sess))).astype(str).str.lower().str.contains("logoff")
        logoff_df = sess[logoff_mask].copy()
        if not logoff_df.empty:
            logoff_df["hour"] = logoff_df["timestamp"].dt.hour
            tmp = logoff_df.groupby("user_id")["hour"].mean()
            base["avg_logout_hour"] = tmp

    # Device baseline
    dev = mapped.get("device_events", pd.DataFrame())
    if not dev.empty and "user_id" in dev.columns and "timestamp" in dev.columns:
        tmp = dev.assign(day=dev["timestamp"].dt.date).groupby(["user_id", "day"]).size().groupby("user_id").mean()
        base["avg_device_usage_per_day"] = tmp

    # File baseline
    fe = mapped.get("file_events", pd.DataFrame())
    if not fe.empty and "user_id" in fe.columns and "timestamp" in fe.columns:
        tmp = fe.assign(day=fe["timestamp"].dt.date).groupby(["user_id", "day"]).size().groupby("user_id").mean()
        base["avg_file_copy_per_day"] = tmp

    # Email baseline
    em = mapped.get("email_events", pd.DataFrame())
    if not em.empty and "user_id" in em.columns and "timestamp" in em.columns:
        tmp = em.assign(day=em["timestamp"].dt.date).groupby(["user_id", "day"]).size().groupby("user_id").mean()
        base["avg_email_sent_per_day"] = tmp

    # Web baseline
    web = mapped.get("web_events", pd.DataFrame())
    if not web.empty and "user_id" in web.columns and "timestamp" in web.columns:
        tmp = web.assign(day=web["timestamp"].dt.date).groupby(["user_id", "day"]).size().groupby("user_id").mean()
        base["avg_web_visit_per_day"] = tmp

    for col in [
        "avg_login_hour",
        "avg_logout_hour",  
        "after_hours_ratio",
        "avg_device_usage_per_day",
        "avg_file_copy_per_day",
        "avg_email_sent_per_day",
        "avg_web_visit_per_day",
    ]:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = base[col].fillna(0.0)

    # Normalize numeric baseline
    base["avg_login_hour"]  = base["avg_login_hour"]  / 23.0
    base["avg_logout_hour"] = base["avg_logout_hour"] / 23.0
    # avg_logout_hour đã chia /23 ở trên → KHÔNG đưa vào safe_log1p_norm lần nữa
    for col in [
        "avg_device_usage_per_day",
        "avg_file_copy_per_day",
        "avg_email_sent_per_day",
        "avg_web_visit_per_day",
    ]:
        base[col] = safe_log1p_norm(base[col].values)

    return base.reset_index()


def parse_recipients(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    # CERT commonly separates recipients by ; or comma.
    parts = re.split(r"[;,]", text)
    return [p.strip() for p in parts if p.strip()]


def recipient_to_user_id(recipient: str) -> str | None:
    # Internal employees use DTAA email addresses; usually user id before @.
    r = recipient.strip()
    if "@" in r:
        local, domain = r.split("@", 1)
        if "dtaa" in domain.lower():
            return local.strip().upper()
        return None
    # Some data may contain direct user ids.
    if re.fullmatch(r"[A-Za-z]{2,4}\d{3,5}", r):
        return r.upper()
    return None


def file_extension(filename) -> str:
    if pd.isna(filename):
        return "unknown"
    name = str(filename)
    if "." in name:
        return name.rsplit(".", 1)[-1].lower()
    return "unknown"


def file_header(content) -> str:
    if pd.isna(content):
        return "unknown"
    token = str(content).strip().split()[0] if str(content).strip() else "unknown"
    return token[:12].lower()


# ══════════════════════════════════════════════════════════════════════════════
# Step 2.2 — Build HeteroData
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(mapped: dict[str, pd.DataFrame]) -> HeteroData:
    # return data có type: torch_geometric.data.HeteroData, 10 types of node and 14 types of edge
    data = HeteroData()

    # ── Load selected users ───────────────────────────────────────────────────
    sel_df = pd.read_csv(OUT_DIR / "users_selected.csv")
    sel_df["user_id"] = normalize_user_id_series(sel_df["user_id"])
    selected_users = set(sel_df["user_id"].tolist())
    insider_set = set(sel_df.loc[sel_df["is_insider"] == 1, "user_id"].tolist())

    # ── User from LDAP ────────────────────────────────────────────────────────
    ldap_df = load_ldap_df()
    col_user = safe_col(ldap_df, ["user_id", "user", "uid", "employee_name"]) # tìm cột user trong LDAP, trong TH này là user_id
    if col_user is None:
        raise ValueError(f"Không tìm thấy cột user trong LDAP. Columns={ldap_df.columns.tolist()}")
    ldap_df[col_user] = normalize_user_id_series(ldap_df[col_user])
    ldap_df = ldap_df[ldap_df[col_user].isin(selected_users)].copy() # chỉ giữ những user có trong users_selected.csv
    ldap_df = ldap_df.drop_duplicates(subset=[col_user]) # loại bỏ dòng trùng user_id nếu có

    if len(ldap_df) == 0:
        raise ValueError("LDAP không match user nào với output/users_selected.csv")

    # keep only valid  in LDAP for graph.
    user_id_list = sorted(ldap_df[col_user].tolist())
    user2idx = {uid: i for i, uid in enumerate(user_id_list)} # tạo mapping user_id → index liên tục từ 0 đến N-1, để dùng làm node index trong graph
    log.info(f"User nodes from LDAP: {len(user_id_list)} / selected={len(selected_users)}")

    col_role = safe_col(ldap_df, ["role", "functional_unit", "department", "dept"])
    col_dept = safe_col(ldap_df, ["department", "dept", "team", "bu"])
    col_team = safe_col(ldap_df, ["team", "business_unit", "bu"])
    col_supervisor = safe_col(ldap_df, ["supervisor", "manager", "manager_id", "supervisor_id"])

    # Guard: nếu col_role và col_dept trỏ cùng cột → role node và department node
    # sẽ giống hệt nhau, graph dư thừa. Log cảnh báo để kiểm tra lại LDAP.
    if col_role and col_dept and col_role == col_dept:
        log.warning(
            f"col_role và col_dept cùng trỏ vào cột '{col_role}' — "
            "role node và department node sẽ có giá trị giống nhau. "
            "Kiểm tra lại tên cột LDAP."
        )

    # Role / Department nodes
    # Nếu cột role/dept/team không tồn tại trong LDAP, vẫn tạo node với 1 giá trị "unknown" để tránh lỗi graph construction sau này. Còn nếu có thì lấy giá trị từ LDAP, fillna "unknown" nếu có giá trị thiếu.
    role_vals = ldap_df[col_role].fillna("unknown").astype(str).tolist() if col_role else ["unknown"] * len(ldap_df)
    dept_vals = ldap_df[col_dept].fillna("unknown").astype(str).tolist() if col_dept else ["unknown"] * len(ldap_df)
    team_vals = ldap_df[col_team].fillna("unknown").astype(str).tolist() if col_team else ["unknown"] * len(ldap_df)

    # chuyển role/department dạng text thành node index số nguyên
    role_indexer = NodeIndexer() # tạo indexer để map giá trị role → index liên tục từ 0 đến M-1, để dùng làm node index trong graph
    dept_indexer = NodeIndexer()
    role_ids = [role_indexer.add(v) for v in role_vals]
    dept_ids = [dept_indexer.add(v) for v in dept_vals] 

    # tạo feature cho role/depart node
    # data["role"].x = torch.eye(max(len(role_indexer), 1), dtype=torch.float)
    # data["role"].names = role_indexer.items
    # data["department"].x = torch.eye(max(len(dept_indexer), 1), dtype=torch.float)
    # data["department"].names = dept_indexer.items
    data["role"].x = torch.tensor(role_ids, dtype=torch.long).view(-1, 1)
    data["role"].names = role_indexer.items
    
    data["department"].x = torch.tensor(dept_ids, dtype=torch.long).view(-1, 1)
    data["department"].names = dept_indexer.items
    log.info(f"✓ Đã tối ưu RAM cho Role/Dept bằng Label Indexing thay vì Identity Matrix.")

    # Baseline features
    # lấy baseline features đã tính ở hàm build_baseline_features, chỉ giữ những user_id có trong user_id_list (đã được lọc từ LDAP) để đảm bảo đúng thứ tự node index. 
    # Nếu user nào không có baseline feature thì fill 0. Chuẩn bị feature cho node user
    baseline_df = build_baseline_features(user_id_list, mapped).set_index("user_id") # 
    baseline_feat = baseline_df.loc[user_id_list, [
        "avg_login_hour",
        "avg_logout_hour",
        "after_hours_ratio",
        "avg_device_usage_per_day",
        "avg_file_copy_per_day",
        "avg_email_sent_per_day",
        "avg_web_visit_per_day",
    ]].values.astype(float)

    # Psychometric features
    # Biến file psychometric OCEAN thành feature numeric, 1 feature cho node user
    psy_feat = np.zeros((len(user_id_list), 5), dtype=float)
    psy_df = load_psychometric_df()
    if not psy_df.empty:
        psy_user = safe_col(psy_df, ["user_id", "user", "uid", "employee_name"])
        if psy_user:
            psy_df[psy_user] = normalize_user_id_series(psy_df[psy_user])
            psy_df = psy_df.drop_duplicates(subset=[psy_user]).set_index(psy_user)
            for i, uid in enumerate(user_id_list):
                if uid in psy_df.index:
                    vals = []
                    for c in ["O", "C", "E", "A", "N"]:
                        vals.append(float(psy_df.loc[uid, c]) if c in psy_df.columns and pd.notna(psy_df.loc[uid, c]) else 0.0)
                    psy_feat[i] = vals
            # Simple robust normalization if scores are likely 0-100.
            maxv = np.nanmax(psy_feat) if psy_feat.size else 0
            if maxv > 1:
                psy_feat = psy_feat / 100.0

    # Encode user role/dept/team categorical as compact numeric features
    role_enc = SafeLabelEncoder()
    dept_enc = SafeLabelEncoder()
    team_enc = SafeLabelEncoder()
    role_arr = role_enc.fit_transform(role_vals) / max(len(role_enc) - 1, 1)
    dept_arr = dept_enc.fit_transform(dept_vals) / max(len(dept_enc) - 1, 1)
    team_arr = team_enc.fit_transform(team_vals) / max(len(team_enc) - 1, 1)

    user_x = np.concatenate([
        role_arr.reshape(-1, 1),
        dept_arr.reshape(-1, 1),
        team_arr.reshape(-1, 1),
        baseline_feat,
        psy_feat,
    ], axis=1)

    data["user"].x = torch.tensor(user_x, dtype=torch.float)
    data["user"].y = torch.tensor([int(uid in insider_set) for uid in user_id_list], dtype=torch.long)
    data["user"].user_ids = user_id_list
    log.info(f"User nodes: {len(user_id_list)} | features={data['user'].x.shape}")

    # User -> Role / Department
    u_src = list(range(len(user_id_list)))
    data["user", "has_role", "role"].edge_index = edge_tensor(u_src, role_ids)
    data["user", "belongs_to", "department"].edge_index = edge_tensor(u_src, dept_ids)

    # User -> User hierarchy
    if col_supervisor:
        sup_src, sup_dst = [], []
        sup_vals = ldap_df[col_supervisor].astype(str).str.strip().str.upper().tolist()
        for uid, sup in zip(ldap_df[col_user].tolist(), sup_vals):
            if uid in user2idx and sup in user2idx and uid != sup:
                sup_src.append(user2idx[uid])
                sup_dst.append(user2idx[sup])
        if sup_src:
            data["user", "reports_to", "user"].edge_index = edge_tensor(sup_src, sup_dst)
            log.info(f"Edge (user→reports_to→user): {len(sup_src)}")
        else:
            data["user", "reports_to", "user"].edge_index = torch.empty((2, 0), dtype=torch.long)

    # Shared indexers
    pc_indexer = NodeIndexer()
    topic_indexer = NodeIndexer()

    # Helper functions to add PC and topic nodes with consistent indexing across different event types.
    def add_pc(pc):
        if pd.isna(pc):
            pc = "unknown_pc"
        return pc_indexer.add(str(pc).strip().upper())

    def add_topics_from_content(content):
        return [topic_indexer.add(t) for t in split_topics(content)]

    # ── Session and PC ────────────────────────────────────────────────────────
    sess = mapped.get("session_events", pd.DataFrame()).copy()
    if not sess.empty:
        sess = sess.dropna(subset=["user_id"])
        sess["user_id"] = normalize_user_id_series(sess["user_id"])
        sess = sess[sess["user_id"].isin(user2idx)].copy()

        # Chỉ giữ logon events làm session node.
        # readme CERT: "After-hours logins are intended to be significant" —
        # logoff không phải signal bất thường, đưa vào gây noise.
        if "activity" in sess.columns:
            logon_mask = sess["activity"].astype(str).str.lower().str.contains("logon")
            sess = sess[logon_mask].copy()

        # Sắp xếp session theo thời gian để behavior_seq lấy đúng session gần nhất
        if "timestamp" in sess.columns:
            sess = sess.sort_values(
                ["user_id", "timestamp"]
            ).reset_index(drop=True)

        if len(sess) > 0:
            pc_col = "pc_id" if "pc_id" in sess.columns else None
            pc_ids = [add_pc(v) for v in (sess[pc_col].tolist() if pc_col else ["unknown_pc"] * len(sess))]

            act_enc = SafeLabelEncoder()
            act_vals = sess["activity"].fillna("unknown").astype(str).tolist() if "activity" in sess.columns else ["unknown"] * len(sess)
            act_arr = act_enc.fit_transform(act_vals) / max(len(act_enc) - 1, 1)

            h_arr = hour_feature(sess["timestamp"]) if "timestamp" in sess.columns else np.zeros(len(sess))
            ah_arr = is_after_hours(sess["timestamp"]) if "timestamp" in sess.columns else np.zeros(len(sess))

            # Bổ sung tw_norm
            tw_arr = assign_time_window(sess["timestamp"]) if "timestamp" in sess.columns else np.zeros(len(sess), dtype=int)
            tw_norm = tw_arr / max(tw_arr.max(), 1)

            # Thêm tw_norm vào cột thứ 4
            data["session"].x = torch.tensor(np.stack([h_arr, ah_arr, act_arr, tw_norm], axis=1), dtype=torch.float)
            data["session"].event_ids = sess["session_id"].astype(str).tolist() if "session_id" in sess.columns else [str(i) for i in range(len(sess))]

            src = [user2idx[u] for u in sess["user_id"].tolist()]
            dst = list(range(len(sess)))
            data["user", "logs_on", "session"].edge_index = edge_tensor(src, dst)
            data["session", "on_pc", "pc"].edge_index = edge_tensor(dst, pc_ids)
            log.info(f"Session nodes: {len(sess)} | Edge user→session={len(src)} | session→pc={len(pc_ids)}")
        del sess
        gc.collect()

    # ── Device (USB) ─────────────────────────────────────────────────────────
    # PDF mục 1 — USB: Thay vì để device là node độc lập, gom các hành vi USB 
    # trên cùng 1 PC vào edge_attr của cạnh (pc → user).
    # Lý do: Giảm node thưa, tiết kiệm RAM, tăng tính diễn giải cho Insider Threat.

    dev = mapped.get("device_events", pd.DataFrame()).copy()
    if not dev.empty:
        # 1. Tiền xử lý dữ liệu
        dev = dev.dropna(subset=["user_id", "pc_id"])
        dev["user_id"] = normalize_user_id_series(dev["user_id"])
        dev = dev[dev["user_id"].isin(user2idx)].copy()

        if len(dev) > 0:
            # 2. Tạo các đặc trưng mức sự kiện (Vectorized)
            dev["is_con"] = (dev["activity"].str.lower() == "connect").astype(int)
            dev["is_dis"] = (dev["activity"].str.lower() == "disconnect").astype(int)
            dev["is_ah"]  = is_after_hours(dev["timestamp"])

            # 3. Gom nhóm theo cặp (User, PC) để tạo Edge Attributes
            # Tối ưu: Tính toán tất cả thống kê trong 1 lần group duy nhất
            usb_summary = dev.groupby(["user_id", "pc_id"]).agg(
                cnt_con=('is_con', 'sum'),
                cnt_dis=('is_dis', 'sum'),
                cnt_ah=('is_ah', 'sum'),
                total=('activity', 'count')
            ).reset_index()

            # 4. Tính toán các Ratio và Đặc trưng độ lệch (Deviation)
            usb_summary["con_rate"] = usb_summary["cnt_con"] / usb_summary["total"]
            usb_summary["dis_rate"] = usb_summary["cnt_dis"] / usb_summary["total"]
            usb_summary["ah_rate"]  = usb_summary["cnt_ah"]  / usb_summary["total"]
            
            # Đặc trưng log-scale cho tổng số hành vi để mô hình nhận diện sự "đột biến"
            usb_summary["total_norm"] = np.log1p(usb_summary["total"])

            # 5. Đảm bảo tất cả PC ID đều được index trước khi map
            unique_pcs = usb_summary["pc_id"].unique()
            for p in unique_pcs:
                if p not in pc_indexer:
                    add_pc(p)

            # 6. Mapping ID sang Index và đẩy vào đồ thị PyG
            # Chuyển ID sang số nguyên (indices) nhanh chóng bằng .map()
            src_pc = usb_summary["pc_id"].map(pc_indexer).values
            dst_user = usb_summary["user_id"].map(user2idx).values

            # Khởi tạo cạnh (pc -> has_usb_activity -> user)
            data["pc", "has_usb_activity", "user"].edge_index = edge_tensor(src_pc, dst_user)

            # Gán thuộc tính cạnh: [connect_rate, disconnect_rate, after_hours_rate, total_normalized]
            features = ["con_rate", "dis_rate", "ah_rate", "total_norm"]
            edge_attr_values = usb_summary[features].values
            
            data["pc", "has_usb_activity", "user"].edge_attr = torch.tensor(
                edge_attr_values, dtype=torch.float
            )

            log.info(f"USB: Hoàn tất tạo {len(usb_summary)} cạnh (pc->user) với 4 edge features.")

            # LƯU Ý QUAN TRỌNG: 
            # Chúng ta đã bỏ hoàn toàn việc tạo nút data["device"] và các cạnh uses_device.
            # Điều này giúp giảm độ thưa (sparsity) của đồ thị và tiết kiệm bộ nhớ cực lớn.

            log.info(
                f"USB edges (pc->user): {len(src_pc)}"
            )

        del dev
        gc.collect()

    # ── File Object ───────────────────────────────────────────────────────────
    fe = mapped.get("file_events", pd.DataFrame()).copy()
    if not fe.empty:
        fe = fe.dropna(subset=["user_id"])
        fe["user_id"] = normalize_user_id_series(fe["user_id"])
        fe = fe[fe["user_id"].isin(user2idx)].copy()

        if len(fe) > 0:
            ext_vals = fe["filename"].apply(file_extension).tolist() if "filename" in fe.columns else ["unknown"] * len(fe)
            hdr_vals = fe["content_keywords"].apply(file_header).tolist() if "content_keywords" in fe.columns else ["unknown"] * len(fe)
            ext_enc = SafeLabelEncoder()
            hdr_enc = SafeLabelEncoder()
            ext_arr = ext_enc.fit_transform(ext_vals) / max(len(ext_enc) - 1, 1)
            hdr_arr = hdr_enc.fit_transform(hdr_vals) / max(len(hdr_enc) - 1, 1)
            h_arr = hour_feature(fe["timestamp"]) if "timestamp" in fe.columns else np.zeros(len(fe))

            # Bổ sung tw_norm
            tw_arr = assign_time_window(fe["timestamp"]) if "timestamp" in fe.columns else np.zeros(len(fe), dtype=int)
            tw_norm = tw_arr / max(tw_arr.max(), 1)

            data["file_object"].x = torch.tensor(np.stack([h_arr, ext_arr, hdr_arr, tw_norm], axis=1), dtype=torch.float)
            data["file_object"].names = fe["filename"].astype(str).tolist() if "filename" in fe.columns else [str(i) for i in range(len(fe))]

            src = [user2idx[u] for u in fe["user_id"].tolist()]
            dst = list(range(len(fe)))
            data["user", "exports_file", "file_object"].edge_index = edge_tensor(src, dst)

            # ── PDF mục 3: Edge Features cho cạnh (user → file_object) ──────────
            # Bổ sung 3 thuộc tính định lượng vào cạnh để mô hình phân biệt
            # mức độ nghiêm trọng của hành vi copy file ra USB:
            #   response_time   : thời gian phản hồi (proxy = 0.0 vì CERT không log; giữ slot để mở rộng)
            #   failed_attempts : số lần thử thất bại trước khi copy thành công
            #                     (proxy = is_anomaly flag từ bước sampling)
            response_time_arr  = np.zeros(len(fe))                         # placeholder
            failed_arr         = fe.get("is_anomaly", pd.Series(np.zeros(len(fe)))).fillna(0).astype(float).values if "is_anomaly" in fe.columns else np.zeros(len(fe))
            file_edge_attr     = np.stack([response_time_arr, failed_arr], axis=1)
            data["user", "exports_file", "file_object"].edge_attr = torch.tensor(file_edge_attr, dtype=torch.float)
            log.info(f"File edge_attr shape: {file_edge_attr.shape} "
                     f"[ response_time, failed_attempts]")

            ft_src, ft_dst = [], []
            content_col = "content_keywords" if "content_keywords" in fe.columns else None
            if content_col:
                for i, content in enumerate(fe[content_col].tolist()):
                    for tid in add_topics_from_content(content):
                        ft_src.append(i)
                        ft_dst.append(tid)
            data["file_object", "has_topic", "topic"].edge_index = edge_tensor(ft_src, ft_dst)
            log.info(f"File_Object nodes: {len(fe)} | Edge user→file={len(src)} | file→topic={len(ft_src)}")
        del fe
        gc.collect()

    # ── Web Resource ──────────────────────────────────────────────────────────
    # PDF mục 1: Gom nhóm theo Domain Name (extract_domain) thay vì URL thô
    # để giảm độ thưa thớt. Ví dụ: hàng nghìn URL của bbc.co.uk → 1 node domain.
    web = mapped.get("web_events", pd.DataFrame()).copy()
    if not web.empty:
        web = web.dropna(subset=["user_id"])
        web["user_id"] = normalize_user_id_series(web["user_id"])
        web = web[web["user_id"].isin(user2idx)].copy()

        if len(web) > 0:
            # Trích xuất domain từ URL — dùng hàm extract_domain() chuẩn hoá
            web["_domain"] = web["url"].apply(extract_domain) if "url" in web.columns else "unknown"

            # Gom nhóm: mỗi node web_resource = 1 domain (không phải 1 URL riêng lẻ)
            # → giảm số node từ hàng nghìn URL xuống vài trăm domain
            dom_enc = SafeLabelEncoder()
            domains = web["_domain"].tolist()
            dom_arr = dom_enc.fit_transform(domains) / max(len(dom_enc) - 1, 1)

            # Path length vẫn giữ để phân biệt deep vs shallow URL trên cùng domain
            path_lens = []
            for url in web["url"].astype(str).tolist() if "url" in web.columns else [""] * len(web):
                parsed = urlparse(url if "://" in url else "http://" + url)
                path_lens.append(len(parsed.path or ""))
            path_arr = safe_log1p_norm(path_lens)
            h_arr    = hour_feature(web["timestamp"]) if "timestamp" in web.columns else np.zeros(len(web))

            # PDF mục 2 — Temporal Subgraph: gán time_window_id cho mỗi web event
            # để mô hình biết event nào thuộc subgraph G_t nào
            tw_arr = assign_time_window(web["timestamp"]) if "timestamp" in web.columns else np.zeros(len(web), dtype=int)
            tw_norm = tw_arr / max(tw_arr.max(), 1)  # normalize về [0,1]

            data["web_resource"].x = torch.tensor(
                np.stack([h_arr, dom_arr, path_arr, tw_norm], axis=1), dtype=torch.float
            )
            # Lưu domain (đã gom nhóm) thay vì URL thô
            data["web_resource"].urls = web["_domain"].tolist()

            src = [user2idx[u] for u in web["user_id"].tolist()]
            dst = list(range(len(web)))
            data["user", "visits", "web_resource"].edge_index = edge_tensor(src, dst)

            # ── PDF mục 3: Edge Features cho cạnh (user → web_resource) ─────────
            # Bổ sung 2 thuộc tính vào cạnh để phản ánh mức độ bất thường của truy cập:
            #   response_time   : placeholder 0.0 (CERT không log response time)
            #   failed_attempts : proxy = is_anomaly từ bước sampling (anomaly event = thử sai)
            # domain_visit_count = web.groupby("_domain")["user_id"].transform("count").values
            # web_rows_arr       = safe_log1p_norm(domain_visit_count.astype(float))
            web_resp_arr       = np.zeros(len(web))
            web_failed_arr     = web.get("is_anomaly", pd.Series(np.zeros(len(web)))).fillna(0).astype(float).values if "is_anomaly" in web.columns else np.zeros(len(web))
            web_edge_attr      = np.stack([web_resp_arr, web_failed_arr], axis=1)
            data["user", "visits", "web_resource"].edge_attr = torch.tensor(web_edge_attr, dtype=torch.float)
            log.info(f"Web edge_attr shape: {web_edge_attr.shape} "
                     f"[response_time, failed_attempts]")

            wt_src, wt_dst = [], []
            content_col = "content_keywords" if "content_keywords" in web.columns else None
            if content_col:
                for i, content in enumerate(web[content_col].tolist()):
                    for tid in add_topics_from_content(content):
                        wt_src.append(i)
                        wt_dst.append(tid)
            data["web_resource", "has_topic", "topic"].edge_index = edge_tensor(wt_src, wt_dst)
            log.info(f"Web_Resource nodes: {len(web)} (gom theo domain) | "
                     f"Edge user→web={len(src)} | web→topic={len(wt_src)}")
        del web
        gc.collect()

    # ── Email ─────────────────────────────────────────────────────────────────
    em = mapped.get("email_events", pd.DataFrame()).copy()
    if not em.empty:
        em = em.dropna(subset=["user_id"])
        em["user_id"] = normalize_user_id_series(em["user_id"])
        em = em[em["user_id"].isin(user2idx)].copy()

        if len(em) > 0:
            size_arr   = safe_log1p_norm(pd.to_numeric(em["size"], errors="coerce").fillna(0).values) if "size" in em.columns else np.zeros(len(em))
            attach_arr = safe_log1p_norm(pd.to_numeric(em["attachment_count"], errors="coerce").fillna(0).values) if "attachment_count" in em.columns else np.zeros(len(em))
            h_arr      = hour_feature(em["timestamp"]) if "timestamp" in em.columns else np.zeros(len(em))

            # Bổ sung tw_norm
            tw_arr     = assign_time_window(em["timestamp"]) if "timestamp" in em.columns else np.zeros(len(em), dtype=int)
            tw_norm    = tw_arr / max(tw_arr.max(), 1)

            # External recipient flag: email gửi ra ngoài tổ chức là signal exfiltration quan trọng.
            # readme CERT: "Emails can have a mix of employees and non-employees in dist list"
            def _is_external_recipient(value) -> bool:
                for r in parse_recipients(value):
                    r = r.strip()
                    if "@" in r:
                        _, domain = r.split("@", 1)
                        if "dtaa" not in domain.lower():
                            return True
                return False

            if any(c in em.columns for c in ["to", "cc", "bcc"]):
                ext_flag = []
                for _, row in em.iterrows():
                    found = False
                    for col in ["to", "cc", "bcc"]:
                        if col in em.columns and _is_external_recipient(row.get(col, "")):
                            found = True
                            break
                    ext_flag.append(1.0 if found else 0.0)
                ext_arr = np.array(ext_flag, dtype=float)
            else:
                ext_arr = np.zeros(len(em))

            data["email"].x = torch.tensor(np.stack([h_arr, size_arr, attach_arr, ext_arr], axis=1), dtype=torch.float)
            data["email"].event_ids = em["email_id"].astype(str).tolist() if "email_id" in em.columns else [str(i) for i in range(len(em))]

            src = [user2idx[u] for u in em["user_id"].tolist()]
            dst = list(range(len(em)))
            data["user", "sends_email", "email"].edge_index = edge_tensor(src, dst)

            # Email -> internal user recipients
            # efficient re-loop with positional index
            er_src, er_dst = [], []
            for pos, (_, row) in enumerate(em.iterrows()):
                for col in ["to", "cc", "bcc"]:
                    if col not in em.columns:
                        continue
                    for r in parse_recipients(row.get(col)):
                        rid = recipient_to_user_id(r)
                        if rid and rid in user2idx:
                            er_src.append(pos)
                            er_dst.append(user2idx[rid])
            data["email", "sent_to", "user"].edge_index = edge_tensor(er_src, er_dst)

            et_src, et_dst = [], []
            content_col = "content_keywords" if "content_keywords" in em.columns else None
            if content_col:
                for i, content in enumerate(em[content_col].tolist()):
                    for tid in add_topics_from_content(content):
                        et_src.append(i)
                        et_dst.append(tid)
            data["email", "has_topic", "topic"].edge_index = edge_tensor(et_src, et_dst)
            log.info(f"Email nodes: {len(em)} | Edge user→email={len(src)} | email→user={len(er_src)} | email→topic={len(et_src)}")
        del em
        gc.collect()

    # ── Finalize PC and Topic features after all indexers are filled ──────────
    # n_pc = max(len(pc_indexer), 1)
    # data["pc"].x = torch.ones((n_pc, 1), dtype=torch.float)
    # data["pc"].pc_ids = pc_indexer.items

    pc_user_map = {}

    for df_name in ["session_events", "device_events", "file_events", "web_events", "email_events"]:
        df_tmp = mapped.get(df_name, pd.DataFrame())
        if df_tmp.empty:
            continue
        if "pc_id" not in df_tmp.columns or "user_id" not in df_tmp.columns:
            continue

        tmp = df_tmp[["pc_id", "user_id"]].dropna().copy()
        tmp["pc_id"] = tmp["pc_id"].astype(str).str.strip().str.upper()
        tmp["user_id"] = normalize_user_id_series(tmp["user_id"])

        for pc, users in tmp.groupby("pc_id")["user_id"]:
            pc_user_map.setdefault(pc, set()).update(users.unique().tolist())

    n_pc = max(len(pc_indexer), 1)
    max_users = max([len(v) for v in pc_user_map.values()] or [1])

    # Dùng LDAP assigned_pc để xác định PC nào là dedicated (assigned cho 1 user cụ thể).
    # readme CERT: "1k users, each with an assigned PC. 100 shared machines" —
    # shared PC là thuộc tính cố định, không nên chỉ suy từ heuristic đếm user.
    col_pc_ldap = safe_col(ldap_df, ["pc", "workstation", "assigned_pc", "computer"])
    assigned_pcs: set[str] = set()
    if col_pc_ldap:
        assigned_pcs = set(
            ldap_df[col_pc_ldap].dropna().astype(str).str.strip().str.upper().tolist()
        )
        log.info(f"Assigned PCs từ LDAP ('{col_pc_ldap}'): {len(assigned_pcs)}")
    else:
        log.warning("LDAP không có cột assigned_pc — fallback heuristic đếm user cho is_shared.")

    pc_features = []
    for pc in pc_indexer.items:
        n_users = len(pc_user_map.get(pc, set()))
        # PC là shared nếu: không phải assigned PC của ai trong LDAP
        # HOẶC (khi không có LDAP pc col) có quá nhiều user dùng (fallback)
        if assigned_pcs:
            is_shared = 1.0 if pc not in assigned_pcs else 0.0
        else:
            is_shared = 1.0 if n_users > SHARED_PC_THRESHOLD else 0.0
        n_users_norm = n_users / max(max_users, 1)
        pc_features.append([is_shared, n_users_norm])

    if not pc_features:
        pc_features = [[0.0, 0.0]]

    data["pc"].x = torch.tensor(pc_features, dtype=torch.float)
    data["pc"].pc_ids = pc_indexer.items

    #------------------------

    n_topic = max(len(topic_indexer), 1)
    data["topic"].x = torch.eye(n_topic, dtype=torch.float)
    data["topic"].names = topic_indexer.items

    # If no edge was created for optional node types, ensure minimal x exists.
    for nt in ["session", "device", "file_object", "web_resource", "email"]:
        if nt not in data.node_types:
            data[nt].x = torch.empty((0, 1), dtype=torch.float)

    log.info(f"PC nodes: {len(pc_indexer)}")
    log.info(f"Topic nodes: {len(topic_indexer)}")
    log.info(f"Node types: {data.node_types}")
    log.info(f"Edge types: {data.edge_types}")

    return data

# trích xuất các thuộc tính metadata (ID, names) ra file JSON trước khi xóa chúng khỏi đồ thị để huấn luyện. Điều này giúp lưu lại thông tin gốc của các node để có thể tra cứu sau này, đồng thời tránh lưu trữ dữ liệu thô không cần thiết trong graph data dùng cho GNN.

def save_mappings(data, out_dir):
    """
    Trích xuất các thuộc tính metadata (ID, names) ra file JSON 
    trước khi xóa chúng khỏi đồ thị để huấn luyện.
    """
    mappings = {}
    
    # Danh sách các nút và thuộc tính metadata tương ứng
    meta_config = {
        'user': 'user_ids',
        'pc': 'pc_ids',
        'role': 'names',
        'department': 'names',
        'session': 'event_ids',
        'device': 'event_ids',
        'file_object': 'names',
        'web_resource': 'urls',
        'email': 'event_ids',
        'topic': 'names'
    }

    for node_type, attr in meta_config.items():
        if node_type in data.node_types and hasattr(data[node_type], attr):
            # Lấy dữ liệu list/array
            values = getattr(data[node_type], attr)
            # Chuyển thành dictionary index: value
            mappings[node_type] = {i: str(val) for i, val in enumerate(values)}
            
    # Lưu ra file JSON
    mapping_path = out_dir / "node_mappings.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    
    print(f"[INFO] Đã lưu file mapping tra cứu tại: {mapping_path}")


def save_graph(data: HeteroData) -> None:
    # 1. Đảm bảo thư mục tồn tại
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Xóa file cũ nếu tồn tại để tránh lỗi ghi đè (corruption)
    if OUT_PT.exists():
        OUT_PT.unlink()
        
    log.info(f"Đang ghi file vào: {OUT_PT}...")
    try:
        # Thử lưu mặc định
        torch.save(data, OUT_PT)
    except RuntimeError as e:
        log.error(f"Lỗi khi lưu mặc định: {e}")
        log.info("Thử lưu với phương thức serialization cũ...")
        # Cách này đôi khi giải quyết được lỗi 'unexpected pos' trên Windows
        torch.save(data, OUT_PT, _use_new_zipfile_serialization=False)

    if OUT_PT.exists():
        size_mb = OUT_PT.stat().st_size / 1024 / 1024
        log.info(f"Đã lưu thành công graph_data.pt ({size_mb:.2f} MB) → {OUT_PT}")
    else:
        log.error("File không tồn tại sau khi lưu!")


def main():
    log.info("═══ Bước 2.1: Load + mapping sampled Parquet ═══")
    mapped = load_and_map()

    log.info("═══ Bước 2.2: Build DB-aware heterogeneous graph ═══")
    data = build_graph(mapped)
    del mapped; gc.collect()
    #======================
    # --- BẮT ĐẦU DỌN DẸP METADATA ---
    log.info("═══ Bước 2.3: Lưu Mapping và dọn dẹp Metadata ═══")
    
    # 1. Gọi hàm lưu mapping (đã định nghĩa ở trên)
    save_mappings(data, OUT_DIR)

    # 2. Xóa các thuộc tính metadata khỏi đối tượng data để tránh lỗi NeighborLoader
    # và tiết kiệm RAM (16GB RAM cần bước này)
    metadata_to_delete = {
        'role': ['names'],
        'department': ['names'],
        'user': ['user_ids'],
        'session': ['event_ids'],
        'device': ['event_ids'],
        'file_object': ['names'],
        'web_resource': ['urls'],
        'email': ['event_ids'],
        'pc': ['pc_ids'],
        'topic': ['names']
    }

    for node_type, attrs in metadata_to_delete.items():
        if node_type in data.node_types:
            for attr in attrs:
                if hasattr(data[node_type], attr):
                    delattr(data[node_type], attr)
                    log.info(f"✓ Đã xóa {attr} khỏi nút {node_type}")

    gc.collect() # Ép giải phóng bộ nhớ ngay lập tức
    # --- KẾT THÚC DỌN DẸP ---

    log.info("═══ Thêm reverse edges bằng ToUndirected ═══")
    data = T.ToUndirected()(data) # tự động thêm edge ngược cho tất cả edge types, để GNN có thể truyền thông tin 2 chiều giữa các node. Ví dụ: nếu có edge user→session thì sẽ tự động thêm session→user.

    log.info("═══ Graph Schema Summary ═══")

    for nt in data.node_types:
        x_shape = tuple(data[nt].x.shape) if "x" in data[nt] else None
        log.info(f"NODE {nt}: x={x_shape}, num_nodes={data[nt].num_nodes}")

    for et in data.edge_types:
        num_edges = data[et].edge_index.shape[1]
        log.info(f"EDGE {et}: num_edges={num_edges:,}")

    log.info("═══ Bước 2.3: Lưu PyG HeteroData ═══")
    save_graph(data)
    log.info("✓ Hoàn thành 02_graph_construction_dbaware.py")


if __name__ == "__main__":
    main()
