"""
data_sampling.py — File 1: User Sampling + Raw Data Filtering
=============================================================
Dataset: CERT Insider Threat Dataset v4.2

This module extracts a small, usable subset from the full CERT dataset
and writes per-event Parquet partitions for downstream processing.

Pipeline Summary
- Step A — Select users (Bước 1.1):
  1. Read `answers/insiders.csv` to get candidate insiders.
  2. Read `cert_data/LDAP/*.csv` to get all LDAP users.
  3. Sample N_INSIDERS insiders and N_NORMAL normal users (deterministic
     via `RANDOM_SEED`) and save `output/users_selected.csv`.

- Step B — Filter events (Bước 1.2):
  1. Load anomaly scenario events from `answers/r4.2-*` (if present).
  2. For each large CSV (logon/file/http/email/device):
     - Stream-read by chunk (`READ_CHUNK`).
     - Normalize and keep only rows for selected users.
     - Optionally filter by date range (`DATE_START`/`DATE_END`).
     - Tag rows that match anomaly events (`is_anomaly`).
     - Write kept rows into small Parquet partitions (`ROWS_PER_PART`).

Output
- `output/users_selected.csv`
- `output/{logon,file,http,email,device}/part_XXXX.parquet`

Notes
- Designed to avoid OOM by streaming CSVs and writing small Parquet files.
- All user IDs are normalized to uppercase strings for stable matching.


kHI MÁY KHOẻ, sửa lại:
N_INSIDERS     = 10 #50 #
N_NORMAL       = 20 #100, DATE_START, DATE_END
"""

import gc
import logging
import random
from pathlib import Path

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR     = Path(__file__).parent.resolve()
ANSWERS_DIR  = ROOT_DIR / "answers"
INSIDER_FILE = ANSWERS_DIR / "insiders.csv"
CERT_DIR     = ROOT_DIR / "cert_data"
LDAP_DIR     = CERT_DIR / "LDAP"
OUT_DIR      = ROOT_DIR / "output"

CSV_FILES = [
    (CERT_DIR / "logon.csv",  "user", "logon"),
    (CERT_DIR / "file.csv",   "user", "file"),
    (CERT_DIR / "http.csv",   "user", "http"),
    (CERT_DIR / "email.csv",  "user", "email"),
    (CERT_DIR / "device.csv", "user", "device"),
]

N_INSIDERS     = 10 #50 #
N_NORMAL       = 20 #100
RANDOM_SEED    = 42
READ_CHUNK     = 20_000   # dòng/lần đọc — giảm nếu thiếu RAM
ROWS_PER_PART  = 50_000   # dòng/file Parquet output

# Giới hạn thời gian (None = lấy hết).
# Với máy 16GB RAM nên bật: DATE_START="2010-05-01", DATE_END="2010-06-30"
# DATE_START = None
# DATE_END   = None

DATE_START = "2010-05-01"
DATE_END   = "2010-06-30"
DATE_COL   = "date"

# đường dẫn anomaly scenario
ANOMALY_DIRS = [
    ANSWERS_DIR / "r4.2-1",
    ANSWERS_DIR / "r4.2-2",
    ANSWERS_DIR / "r4.2-3",
]

ANOMALY_TIME_TOLERANCE_MINUTES = 0


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _detect_user_col(df: pd.DataFrame, filepath: Path) -> str:
    """Tự động tìm cột user ID trong DataFrame."""
    for col in ["user_id", "employee_name", "user", "User", "uid", "name"]:
        if col in df.columns:
            return col
    raise ValueError(
        f"Không tìm thấy cột user trong {filepath}.\n"
        f"Các cột: {df.columns.tolist()}\n"
        "Thêm tên cột đúng vào danh sách trong _detect_user_col()."
    )

# hàm chuẩn hoá chuỗi
def normalize_user_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

# hàm đọc anomaly events
def load_anomaly_events() -> pd.DataFrame:
    """
    Đọc anomaly event files trong answers/r4.2-1, r4.2-2, r4.2-3.

    Format thực tế không có header:
        logon,{id},date,user,pc,activity
        device,{id},date,user,pc,activity
        http,{id},date,user,pc,url,content

    Output chuẩn:
        event_type, event_id, date, user, pc, scenario, source_file
    """
    rows = []

    col_names = [
        "event_type",
        "id",
        "date",
        "user",
        "pc",
        "field1",
        "field2",
    ]

    for d in ANOMALY_DIRS:
        if not d.exists():
            log.warning(f"Không tìm thấy anomaly dir: {d}")
            continue

        for f in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(
                    f,
                    header=None,
                    names=col_names,
                    dtype=str,
                    # low_memory=False,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception as e:
                log.warning(f"Không đọc được anomaly file {f}: {e}")
                continue

            if df.empty:
                continue

            df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
            df["user"] = normalize_user_id(df["user"])
            df["date"] = pd.to_datetime(
                df["date"],
                format="%m/%d/%Y %H:%M:%S",
                errors="coerce"
            )
            df["pc"] = df["pc"].astype(str).str.strip().str.upper()
            df["source_file"] = f.name
            df["scenario"] = d.name

            df = df.dropna(subset=["event_type", "user", "date"])

            rows.append(df[[
                "event_type",
                "id",
                "date",
                "user",
                "pc",
                "scenario",
                "source_file",
            ]])

    if not rows:
        log.warning("Không load được anomaly events nào.")
        return pd.DataFrame(columns=[
            "event_type", "id", "date", "user", "pc", "scenario", "source_file"
        ])

    out = pd.concat(rows, ignore_index=True).drop_duplicates()
    log.info(f"[anomaly] Loaded {len(out):,} anomaly events")
    return out

# # hàm gắn nhãn cho chunk dựa trên anomaly events (date + user + pc)
# def add_event_labels(
#     keep: pd.DataFrame,
#     anomaly_df: pd.DataFrame,
#     user_col: str,
#     out_name: str,
# ) -> pd.DataFrame:
#     """
#     Gắn nhãn is_anomaly cho từng dòng event.

#     Match theo:
#         out_name/event_type + user + date + pc nếu có
#     """
#     keep = keep.copy()
#     keep["is_anomaly"] = 0
#     keep["anomaly_scenario"] = ""
#     keep["anomaly_source_file"] = ""

#     if anomaly_df.empty:
#         return keep

#     if "date" not in keep.columns:
#         return keep

#     keep["_event_type"] = out_name.lower()
#     keep["_user_norm"] = normalize_user_id(keep[user_col])
#     keep["_date_norm"] = pd.to_datetime(keep["date"], errors="coerce")

#     if "pc" in keep.columns:
#         keep["_pc_norm"] = keep["pc"].astype(str).str.strip().str.upper()
#     else:
#         keep["_pc_norm"] = ""

#     right = anomaly_df.copy()
#     right["_event_type"] = right["event_type"].astype(str).str.lower()
#     right["_user_norm"] = normalize_user_id(right["user"])
#     right["_date_norm"] = pd.to_datetime(right["date"], errors="coerce")

#     if "pc" in right.columns:
#         right["_pc_norm"] = right["pc"].astype(str).str.strip().str.upper()
#     else:
#         right["_pc_norm"] = ""

#     left = keep.reset_index().rename(columns={"index": "_orig_idx"})

#     merged = left[["_orig_idx", "_event_type", "_user_norm", "_date_norm", "_pc_norm"]].merge(
#         right[["_event_type", "_user_norm", "_date_norm", "_pc_norm", "scenario", "source_file"]],
#         on=["_event_type", "_user_norm", "_date_norm", "_pc_norm"],
#         how="left",
#     )

#     hit = merged.dropna(subset=["scenario"])

#     if not hit.empty:
#         idx = hit["_orig_idx"].values
#         keep.loc[idx, "is_anomaly"] = 1
#         keep.loc[idx, "anomaly_scenario"] = hit["scenario"].values
#         keep.loc[idx, "anomaly_source_file"] = hit["source_file"].values

#     keep = keep.drop(
#         columns=["_event_type", "_user_norm", "_date_norm", "_pc_norm"],
#         errors="ignore"
#     )

#     return keep



def normalize_event_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

# Đổi sang match theo event_type + id vì:
# - id là định danh duy nhất của event trong CERT v4.2, có thể match trực tiếp mà không cần quan tâm user/date/pc (đôi khi có mismatch hoặc thiếu thôngtin user/date/pc giữa anomaly events và raw events).
# hàm gắn nhãn cho chunk
def add_event_labels(
    keep: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    user_col: str,
    out_name: str,
) -> pd.DataFrame:
    """
    Gắn nhãn is_anomaly cho từng dòng event.

    Match theo:
        event_type + id

    Lý do:
    - id là định danh duy nhất của event trong CERT v4.2
    - không cần match thêm user/date/pc
    """
    keep = keep.copy()
    keep["is_anomaly"] = 0
    keep["anomaly_scenario"] = ""
    keep["anomaly_source_file"] = ""

    if anomaly_df.empty:
        return keep

    if "id" not in keep.columns:
        return keep

    if "id" not in anomaly_df.columns:
        return keep

    # Chuẩn hóa bên trái: dữ liệu gốc logon/file/http/email/device
    keep["_event_type"] = out_name.lower()
    keep["_id_norm"] = normalize_event_id(keep["id"])

    # Chuẩn hóa bên phải: anomaly events trong answers/r4.2-*
    right = anomaly_df.copy()
    right["_event_type"] = right["event_type"].astype(str).str.strip().str.lower()
    right["_id_norm"] = normalize_event_id(right["id"])

    # Chỉ giữ các cột cần merge
    right = right[[
        "_event_type",
        "_id_norm",
        "scenario",
        "source_file",
    ]].drop_duplicates()

    # Giữ index gốc để lát nữa gán nhãn lại vào keep
    left = keep.reset_index().rename(columns={"index": "_orig_idx"})

    merged = left[["_orig_idx", "_event_type", "_id_norm"]].merge(
        right,
        on=["_event_type", "_id_norm"],
        how="left",
    )

    hit = merged.dropna(subset=["scenario"])

    if not hit.empty:
        idx = hit["_orig_idx"].values
        keep.loc[idx, "is_anomaly"] = 1
        keep.loc[idx, "anomaly_scenario"] = hit["scenario"].values
        keep.loc[idx, "anomaly_source_file"] = hit["source_file"].values

    keep = keep.drop(
        columns=["_event_type", "_id_norm"],
        errors="ignore"
    )

    return keep





def _apply_date_filter(chunk: pd.DataFrame) -> pd.DataFrame:
    """Lọc DATE_START / DATE_END nếu được cấu hình."""
    if not (DATE_START or DATE_END):
        return chunk
    if DATE_COL not in chunk.columns:
        return chunk
    ts = pd.to_datetime(chunk[DATE_COL], errors="coerce")
    mask = pd.Series(True, index=chunk.index)
    if DATE_START:
        mask &= ts >= pd.Timestamp(DATE_START)
    if DATE_END:
        mask &= ts <= pd.Timestamp(DATE_END)
    return chunk[mask]


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1.1 — Chọn Users
# ══════════════════════════════════════════════════════════════════════════════

def load_all_insiders() -> set:
    """Đọc answers/insiders.csv → trả về TẤT CẢ insider user ID."""
    if not INSIDER_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy: {INSIDER_FILE}")
    df = pd.read_csv(INSIDER_FILE)
    col = _detect_user_col(df, INSIDER_FILE)
    # result = set(df[col].dropna().str.strip().unique())
    result = set(normalize_user_id(df[col].dropna()).unique())
    log.info(f"[insiders.csv] Tổng insider trong file: {len(result)}")
    return result


def sample_insiders(all_insiders: set) -> set:
    """Random chọn N_INSIDERS từ tập toàn bộ insider."""
    pool = sorted(all_insiders)              # sort trước → tái lặp được
    rng  = random.Random(RANDOM_SEED)
    n    = min(N_INSIDERS, len(pool))
    sampled = set(rng.sample(pool, n)) # random.sample() trả về list, convert lại thành set
    log.info(f"[1.1] Insider: chọn {len(sampled)}/{len(pool)} (seed={RANDOM_SEED})")
    if len(pool) < N_INSIDERS:
        log.warning(f"  Chỉ có {len(pool)} insider < N_INSIDERS={N_INSIDERS} → lấy hết")
    return sampled


def load_all_ldap_users() -> set:
    """
    Đọc tất cả *.csv trong cert_data/LDAP/.
    Hỗ trợ cả 1 file LDAP.csv lẫn nhiều file split.
    """
    ldap_files = sorted(LDAP_DIR.glob("*.csv"))
    if not ldap_files:
        raise FileNotFoundError(
            f"Không có file CSV trong {LDAP_DIR}.\n"
            "Kiểm tra lại đường dẫn LDAP_DIR."
        )
    all_users = set()
    for fpath in ldap_files:
        # df  = pd.read_csv(fpath, low_memory=False)
        df  = pd.read_csv(fpath)
        col = _detect_user_col(df, fpath)
        # u   = set(df[col].dropna().str.strip().unique())
        u = set(normalize_user_id(df[col].dropna()).unique())
        all_users |= u
        log.info(f"  LDAP {fpath.name}: {len(u):,} users")
    log.info(f"[LDAP] Tổng: {len(all_users):,} user ID duy nhất")
    return all_users


def sample_normal_users(all_ldap: set, all_insiders: set) -> set:
    """
    Loại TẤT CẢ insider (không chỉ 50 cái đã chọn) khỏi LDAP pool,
    random chọn N_NORMAL user bình thường.
    """
    pool = sorted(all_ldap - all_insiders)
    rng  = random.Random(RANDOM_SEED + 1)   # seed khác để normal ≠ insider
    n    = min(N_NORMAL, len(pool))
    sampled = set(rng.sample(pool, n))
    log.info(f"[1.1] Normal: chọn {len(sampled)}/{len(pool)} (seed={RANDOM_SEED + 1})")
    if len(pool) < N_NORMAL:
        log.warning(f"  Chỉ có {len(pool)} normal < N_NORMAL={N_NORMAL} → lấy hết")
    return sampled


def select_users() -> tuple:
    """
    Bước 1.1 đầy đủ. Trả về (sel_insiders, sel_normals).
    Lưu users_selected.csv với cột user_id và is_insider.

    This function implements the "Step A" of the pipeline: it ensures
    insiders exist in the LDAP set, samples insiders and normal users,
    and persists the selection for downstream filtering.
    """
    # load tất cả insider + LDAP users để kiểm tra và sampling
    all_insiders = load_all_insiders()
    all_ldap     = load_all_ldap_users()

    # FIX QUAN TRỌNG: chỉ giữ insider có trong LDAP
    valid_insiders = all_insiders & all_ldap
    missing_insiders = all_insiders - all_ldap

    log.info(f"[check] Insider có trong LDAP: {len(valid_insiders)}/{len(all_insiders)}")
    log.info(f"[check] Insider thiếu trong LDAP: {len(missing_insiders)}")

    # 
    sel_insiders = sample_insiders(valid_insiders)
    # sel_normals  = sample_normal_users(all_ldap, valid_insiders)
    # loại toàn bộ all_insider cho an toàn hơn
    sel_normals = sample_normal_users(all_ldap, all_insiders)

    selected = sel_insiders | sel_normals # tổng hợp để lưu file chung
    log.info(f"[1.1] Tổng selected: {len(selected)} "
             f"(insider={len(sel_insiders)}, normal={len(sel_normals)})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "users_selected.csv"
    pd.DataFrame([
        {"user_id": uid, "is_insider": int(uid in sel_insiders)}
        for uid in sorted(selected)
    ]).to_csv(out_path, index=False)
    log.info(f"  → {out_path}")

    return sel_insiders, sel_normals


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1.2 — Lọc và lưu file nhỏ (PartitionWriter)
# ══════════════════════════════════════════════════════════════════════════════

class PartitionWriter:
    """
    Ghi DataFrame vào nhiều file Parquet nhỏ.
    Mỗi file ≤ ROWS_PER_PART dòng → tránh file lớn gây OOM khi load.

    Cách dùng:
        writer = PartitionWriter(out_dir)
        writer.write(chunk_df)   # gọi nhiều lần với chunk
        total = writer.flush()   # gọi 1 lần cuối để ghi phần còn lại
    """

    def __init__(self, out_dir: Path, rows_per_part: int = ROWS_PER_PART):
        self.out_dir       = out_dir
        self.rows_per_part = rows_per_part
        out_dir.mkdir(parents=True, exist_ok=True)
        self._buf: list  = []
        self._buf_rows   = 0
        self._part_idx   = 0
        self._total      = 0

    def _flush_buf(self) -> None:
        if not self._buf:
            return
        df = pd.concat(self._buf, ignore_index=True)
        fpath = self.out_dir / f"part_{self._part_idx:04d}.parquet"
        df.to_parquet(fpath, index=False, compression="snappy")
        self._total    += len(df)
        self._part_idx += 1
        self._buf       = []
        self._buf_rows  = 0
        log.info(f"    → {fpath.name} ({len(df):,} dòng)")
        del df
        gc.collect()

    def write(self, df: pd.DataFrame) -> None:
        """Thêm chunk, tự flush khi buffer đủ ROWS_PER_PART."""
        if df.empty:
            return
        self._buf.append(df)
        self._buf_rows += len(df)
        if self._buf_rows >= self.rows_per_part:
            self._flush_buf()

    def flush(self) -> int:
        """Ghi phần còn dư. Trả về tổng dòng đã ghi."""
        self._flush_buf()
        return self._total


def filter_one_csv(
    csv_path: Path,
    user_col: str,
    out_name: str,
    selected_users: set,
    anomaly_df: pd.DataFrame,
) -> None:
    """
    Đọc CSV theo chunk READ_CHUNK dòng, lọc user + ngày, ghi Parquet nhỏ.
    Không bao giờ load toàn bộ CSV vào RAM.
    """
    if not csv_path.exists():
        log.warning(f"[skip] Không tìm thấy {csv_path}")
        return

    log.info(f"[1.2] {csv_path.name} → output/{out_name}/")
    writer           = PartitionWriter(OUT_DIR / out_name)
    total_in         = 0
    total_out        = 0
    resolved_col     = None

    for chunk in pd.read_csv(
        csv_path,
        chunksize=READ_CHUNK,
        # low_memory=False,
        dtype=str,       # đọc tất cả là str → tránh type mismatch
    ):
        total_in += len(chunk)

        # Xác định cột user lần đầu tiên
        if resolved_col is None:
            if user_col in chunk.columns:
                resolved_col = user_col
            else:
                try:
                    resolved_col = _detect_user_col(chunk, csv_path)
                    log.warning(f"  Cột '{user_col}' không có, dùng '{resolved_col}'")
                except ValueError as e:
                    log.error(str(e))
                    return

        # Lọc user
        chunk[resolved_col] = normalize_user_id(chunk[resolved_col])
        keep = chunk[chunk[resolved_col].isin(selected_users)].copy()
        if keep.empty:
            continue

        # Lọc ngày
        keep = _apply_date_filter(keep)
        if keep.empty:
            continue

        # Gắn nhãn event-level anomaly
        keep = add_event_labels(
            keep=keep,
            anomaly_df=anomaly_df,
            user_col=resolved_col,
            out_name=out_name,
        )

        n_anom = int(keep["is_anomaly"].sum())
        if n_anom > 0:
            log.info(f"  [{out_name}] anomaly matched trong chunk: {n_anom}")

        total_out += len(keep)
        writer.write(keep)
        del keep

    final = writer.flush()

    if final == 0:
        log.warning(f"  {csv_path.name}: 0 dòng sau lọc")
    else:
        pct = total_out / total_in * 100 if total_in else 0
        log.info(f"  {csv_path.name}: {total_in:,} → {total_out:,} dòng "
                 f"({pct:.1f}%) | {writer._part_idx} partition files")
    gc.collect()


def filter_all(selected_users: set, anomaly_df: pd.DataFrame) -> None:

    """Bước 1.2: Lọc tuần tự từng file, giải phóng RAM giữa các file.

    This function implements the "Step B" of the pipeline: it iterates
    through the large CSV files defined in `CSV_FILES` and calls
    `filter_one_csv` to stream, filter, label, and write partitions.
    """
    for csv_path, user_col, out_name in CSV_FILES:
        filter_one_csv(csv_path, user_col, out_name, selected_users, anomaly_df)
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# TÓM TẮT KẾT QUẢ
# ══════════════════════════════════════════════════════════════════════════════

def print_summary() -> None:
    log.info("\n── Output Summary ──────────────────────────────────")
    uf = OUT_DIR / "users_selected.csv"
    if uf.exists():
        df = pd.read_csv(uf)
        log.info(f"  users_selected.csv : {len(df)} users "
                 f"(insider={df['is_insider'].sum()}, "
                 f"normal={(df['is_insider']==0).sum()})")
    for _, _, name in CSV_FILES:
        d = OUT_DIR / name
        parts = sorted(d.glob("*.parquet")) if d.exists() else []
        if parts:
            mb = sum(p.stat().st_size for p in parts) / 1024 / 1024
            log.info(f"  {name:8s} : {len(parts):3d} files | {mb:.1f} MB")
        else:
            log.info(f"  {name:8s} : (trống)")
    log.info("────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("═══════════════════════════════════════════════")
    log.info("  CERT v4.2 — Data Sampling & Filtering")
    log.info(f"  N_INSIDERS={N_INSIDERS} | N_NORMAL={N_NORMAL} | SEED={RANDOM_SEED}")
    log.info(f"  READ_CHUNK={READ_CHUNK:,} | ROWS_PER_PART={ROWS_PER_PART:,}")
    log.info(f"  DATE: {DATE_START or 'all'} → {DATE_END or 'all'}")
    log.info("═══════════════════════════════════════════════")

    log.info("\n── Bước 1.1: Chọn Users ──")
    
    sel_ins, sel_nor = select_users() # chọn insider + normal user, lưu file users_selected.csv
    selected = sel_ins | sel_nor

    log.info("\n── Load anomaly event labels ──")
    anomaly_df = load_anomaly_events()

    log.info("\n── Bước 1.2: Lọc dữ liệu thô ──")
    # 
    filter_all(selected, anomaly_df) 

    print_summary()
    log.info("\n✓ Xong. Tiếp theo: python graph_construction.py")


if __name__ == "__main__":
    main()
