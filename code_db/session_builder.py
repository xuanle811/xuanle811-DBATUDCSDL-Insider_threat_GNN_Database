"""
session_builder.py — Gom event_id thành session_id
====================================================
Giải quyết bài toán: audit log chỉ có event_id theo từng dòng,
không có session_id sẵn → cần tự sinh session_id hợp lý.

4 chiến lược, tự động chọn chiến lược tốt nhất theo cột có sẵn:

    Chiến lược A — Fixed Time Window (fallback)
        Gom mọi event của cùng db_user trong cùng khoảng N phút
        thành 1 session.
        Dùng khi: chỉ có timestamp + db_user.
        Hạn chế: nếu user login 2 lần trong cùng window → bị ghép nhầm.

    Chiến lược B — Inactivity Gap (KHUYẾN NGHỊ cho DB tự sinh)
        Cắt session mới khi khoảng cách giữa 2 event liên tiếp
        của cùng db_user > IDLE_THRESHOLD phút.
        Dùng khi: chỉ có timestamp + db_user nhưng biết idle time thực tế.
        Ưu điểm: phản ánh đúng "user bỏ bàn phím" hơn Fixed Window.
        Là chuẩn công nghiệp (Google Analytics, Splunk dùng cách này).

    Chiến lược C — Connection-based (TỐT NHẤT)
        session_id = db_user + connection_id (hoặc process_id).
        Dùng khi: có cột connection_id / process_id / thread_id.
        Chính xác nhất vì 1 connection = 1 phiên làm việc thực sự.

    Chiến lược D — Login/Logout Event-based
        Dùng login event làm mốc bắt đầu session mới.
        Dùng khi: có cột activity_type với giá trị login/logon/connect.

Cách dùng trong a01_data_prepocessing.py:
    from session_builder import assign_session_ids
    audit_df = assign_session_ids(audit_df)
    # → audit_df có thêm cột 'session_id'

Justification cho paper (§4.1 Dataset):
    "Since raw audit logs contain individual event records without
     explicit session boundaries, we apply an inactivity-based
     session segmentation [REF]: a new session is created when the
     idle time between consecutive events of the same database user
     exceeds θ minutes (θ=30). This approach is consistent with
     standard database session analysis practices and avoids the
     boundary ambiguity of fixed-time-window methods."
"""

import hashlib
import pandas as pd
import numpy as np

# ── Cấu hình mặc định ─────────────────────────────────────────────────────────
IDLE_THRESHOLD_MIN   = 80    # Chiến lược B: cắt session nếu idle > 30 phút
FIXED_WINDOW_MIN     = 60    # Chiến lược A: khoảng cửa sổ cố định = 60 phút
MAX_SESSION_EVENTS   = 500   # Giới hạn tối đa event/session (tránh session vô hạn)
SESSION_ID_PREFIX    = "S"   # Prefix để session_id dễ đọc: S_0001, S_0002...


# ══════════════════════════════════════════════════════════════════════════════
# CHIẾN LƯỢC B — Inactivity Gap (KHUYẾN NGHỊ)
# ══════════════════════════════════════════════════════════════════════════════

def _session_by_inactivity(
    group: pd.DataFrame,
    idle_threshold_min: int,
    max_events: int,
    user_session_counter: dict,
    db_user: str,
) -> pd.Series:
    """
    Gom events của 1 db_user theo inactivity gap.
    Trả về Series chứa session_id cho từng row trong group.

    Logic:
        - Sắp xếp theo timestamp
        - Tính delta giữa event liên tiếp
        - Nếu delta > idle_threshold HOẶC đã > max_events → cắt session mới
        - session_id = f"{db_user}__{counter:04d}"
    """
    group     = group.sort_values('timestamp').copy()
    sess_ids  = []
    counter   = user_session_counter.get(db_user, 0)
    curr_sess = f"{SESSION_ID_PREFIX}_{db_user}__{counter:04d}"
    curr_count = 0

    prev_ts = None
    for ts in group['timestamp']:
        if prev_ts is not None:
            delta_min = (ts - prev_ts).total_seconds() / 60.0
            if delta_min > idle_threshold_min or curr_count >= max_events:
                counter  += 1
                curr_sess = f"{SESSION_ID_PREFIX}_{db_user}__{counter:04d}"
                curr_count = 0
        sess_ids.append(curr_sess)
        curr_count += 1
        prev_ts = ts

    user_session_counter[db_user] = counter + 1
    result = pd.Series(sess_ids, index=group.index)
    return result


def assign_sessions_inactivity(
    df: pd.DataFrame,
    idle_threshold_min: int = IDLE_THRESHOLD_MIN,
    max_events: int         = MAX_SESSION_EVENTS,
    user_col: str           = 'db_user',
    time_col: str           = 'timestamp',
) -> pd.DataFrame:
    """
    Chiến lược B: Inactivity Gap — KHUYẾN NGHỊ cho DB tự sinh.

    Công thức phân chia:
        new_session = True  khi  Δt(eᵢ, eᵢ₋₁) > θ  hoặc  |session| > max_events
        session_id  = f"S_{db_user}__{counter:04d}"

    Args:
        df                : DataFrame với cột timestamp và db_user
        idle_threshold_min: ngưỡng idle (phút) để cắt session
        max_events        : số event tối đa trong 1 session

    Returns:
        df với cột 'session_id' mới
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    user_session_counter = {}
    session_col = pd.Series(index=df.index, dtype=str)

    for db_user, group in df.groupby(user_col, sort=False):
        sess_series = _session_by_inactivity(
            group, idle_threshold_min, max_events,
            user_session_counter, str(db_user)
        )
        session_col.loc[sess_series.index] = sess_series

    df['session_id'] = session_col
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHIẾN LƯỢC A — Fixed Time Window (fallback đơn giản)
# ══════════════════════════════════════════════════════════════════════════════

def assign_sessions_fixed_window(
    df: pd.DataFrame,
    window_min: int = FIXED_WINDOW_MIN,
    user_col:   str = 'db_user',
    time_col:   str = 'timestamp',
) -> pd.DataFrame:
    """
    Chiến lược A: Fixed Time Window.
    session_id = f"S_{db_user}__{floor(timestamp / window_min)}"

    Ưu điểm: đơn giản, deterministic, dễ reproduce.
    Nhược điểm: nếu user login 2 lần trong cùng window → bị ghép nhầm.
                Nếu 1 session dài hơn window → bị cắt nhầm.

    Nên dùng làm baseline để so sánh với Chiến lược B.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Số phút kể từ epoch → floor về window_min → đây là "slot" thời gian
    epoch     = df[time_col].min()
    delta_min = (df[time_col] - epoch).dt.total_seconds() / 60.0
    slot      = (delta_min // window_min).astype(int)

    df['session_id'] = (
        df[user_col].astype(str)
        + "__W"
        + slot.astype(str).str.zfill(5)
    )
    df['session_id'] = SESSION_ID_PREFIX + "_" + df['session_id']
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHIẾN LƯỢC C — Connection-based (tốt nhất khi có connection_id)
# ══════════════════════════════════════════════════════════════════════════════

def assign_sessions_connection(
    df: pd.DataFrame,
    user_col:       str = 'db_user',
    connection_col: str = 'connection_id',
) -> pd.DataFrame:
    """
    Chiến lược C: session_id = db_user + connection_id.
    Chính xác nhất vì 1 TCP connection = 1 phiên thực sự.

    Dùng khi audit log có cột: connection_id, process_id, thread_id, spid.
    Nếu connection_id không unique toàn cục (DB tái sử dụng ID),
    thêm db_user vào key để đảm bảo duy nhất.
    """
    if connection_col not in df.columns:
        raise ValueError(
            f"Cột '{connection_col}' không tồn tại. "
            f"Dùng assign_sessions_inactivity() thay thế."
        )
    df = df.copy()
    df['session_id'] = (
        SESSION_ID_PREFIX + "_"
        + df[user_col].astype(str)
        + "__C"
        + df[connection_col].astype(str)
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHIẾN LƯỢC D — Login/Logout Event-based
# ══════════════════════════════════════════════════════════════════════════════

def assign_sessions_login_event(
    df: pd.DataFrame,
    user_col:     str  = 'db_user',
    time_col:     str  = 'timestamp',
    activity_col: str  = 'activity_type',
    login_keywords: tuple = ('login', 'logon', 'connect', 'authenticate'),
) -> pd.DataFrame:
    """
    Chiến lược D: Cắt session mới mỗi khi gặp login event.
    Phù hợp khi audit log có cột activity_type / event_type.

    Nếu không có login event rõ ràng (giữa chừng) → session đầu tiên
    của mỗi user bắt đầu từ event đầu tiên.
    """
    if activity_col not in df.columns:
        raise ValueError(
            f"Cột '{activity_col}' không tồn tại. "
            f"Dùng assign_sessions_inactivity() thay thế."
        )
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    login_kw  = tuple(k.lower() for k in login_keywords)
    is_login  = df[activity_col].astype(str).str.lower().str.contains(
        '|'.join(login_kw), regex=True
    )

    session_col = pd.Series(index=df.index, dtype=str)

    for db_user, group in df.groupby(user_col, sort=False):
        group    = group.sort_values(time_col)
        counter  = 0
        curr_sid = f"{SESSION_ID_PREFIX}_{db_user}__L{counter:04d}"
        sids     = []
        for idx, row in group.iterrows():
            if is_login[idx] and len(sids) > 0:
                counter += 1
                curr_sid = f"{SESSION_ID_PREFIX}_{db_user}__L{counter:04d}"
            sids.append(curr_sid)
        session_col.loc[group.index] = sids

    df['session_id'] = session_col
    return df


# ══════════════════════════════════════════════════════════════════════════════
# AUTO SELECTOR — tự chọn chiến lược tốt nhất theo cột có sẵn
# ══════════════════════════════════════════════════════════════════════════════

def assign_session_ids(
    df: pd.DataFrame,
    idle_threshold_min: int   = IDLE_THRESHOLD_MIN,
    window_min:         int   = FIXED_WINDOW_MIN,
    user_col:           str   = 'db_user',
    time_col:           str   = 'timestamp',
    strategy:           str   = 'auto',   # 'auto' | 'inactivity' | 'fixed' | 'connection' | 'login'
) -> pd.DataFrame:
    """
    Entry point chính. Gọi hàm này từ a01_data_prepocessing.py.

    strategy='auto': ưu tiên C > D > B > A theo thứ tự.

    Returns:
        df với cột 'session_id' + cột 'session_strategy' (ghi nhớ cách dùng)
    """
    chosen = strategy

    if strategy == 'auto':
        # Ưu tiên connection-based nếu có
        for conn_col in ('connection_id', 'process_id', 'thread_id', 'spid'):
            if conn_col in df.columns:
                chosen = 'connection'
                print(f"[session_builder] Chiến lược C: connection-based "
                      f"(cột '{conn_col}')")
                df = assign_sessions_connection(df, user_col, conn_col)
                df['session_strategy'] = f'C:{conn_col}'
                return _postprocess(df)

        # Thứ 2: login event-based
        for act_col in ('activity_type', 'event_type', 'activity', 'action'):
            if act_col in df.columns:
                acts = df[act_col].astype(str).str.lower()
                has_login = acts.str.contains(
                    'login|logon|connect|authenticate', regex=True).any()
                if has_login:
                    chosen = 'login'
                    print(f"[session_builder] Chiến lược D: login-event-based "
                          f"(cột '{act_col}')")
                    df = assign_sessions_login_event(df, user_col, time_col, act_col)
                    df['session_strategy'] = f'D:{act_col}'
                    return _postprocess(df)

        # Fallback: inactivity gap (khuyến nghị cho DB tự sinh)
        chosen = 'inactivity'
        print(f"[session_builder] Chiến lược B: inactivity gap "
              f"(idle>{idle_threshold_min}min) — KHUYẾN NGHỊ cho DB tự sinh")

    if chosen == 'inactivity':
        df = assign_sessions_inactivity(df, idle_threshold_min,
                                        MAX_SESSION_EVENTS, user_col, time_col)
        df['session_strategy'] = f'B:idle>{idle_threshold_min}min'

    elif chosen == 'fixed':
        df = assign_sessions_fixed_window(df, window_min, user_col, time_col)
        df['session_strategy'] = f'A:fixed{window_min}min'

    elif chosen == 'connection':
        # Tự tìm cột connection
        for c in ('connection_id', 'process_id', 'thread_id', 'spid'):
            if c in df.columns:
                df = assign_sessions_connection(df, user_col, c)
                df['session_strategy'] = f'C:{c}'
                break
        else:
            raise ValueError("Không tìm thấy cột connection. "
                             "Dùng strategy='inactivity'.")

    elif chosen == 'login':
        for c in ('activity_type', 'event_type', 'activity', 'action'):
            if c in df.columns:
                df = assign_sessions_login_event(df, user_col, time_col, c)
                df['session_strategy'] = f'D:{c}'
                break
        else:
            raise ValueError("Không tìm thấy cột activity. "
                             "Dùng strategy='inactivity'.")

    return _postprocess(df)


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thống kê và validate session_id sau khi gán.
    In cảnh báo nếu có session bất thường (quá nhiều event, 1 event).
    """
    n_sess  = df['session_id'].nunique()
    n_users = df['db_user'].nunique() if 'db_user' in df.columns else '?'
    sess_counts = df.groupby('session_id').size()

    print(f"\n── Session Statistics ──────────────────────────────")
    print(f"  Tổng sessions          : {n_sess:,}")
    print(f"  Tổng users             : {n_users:,}")
    print(f"  Sessions / user (avg)  : {n_sess / max(int(str(n_users).replace(',','')), 1):.1f}")
    print(f"  Events / session (avg) : {sess_counts.mean():.1f}")
    print(f"  Events / session (med) : {sess_counts.median():.1f}")
    print(f"  Events / session (max) : {sess_counts.max()}")
    print(f"  Events / session (min) : {sess_counts.min()}")

    # Cảnh báo session 1 event (thường là noise)
    single = (sess_counts == 1).sum()
    if single > 0:
        pct = single / n_sess * 100
        print(f"  [Cảnh báo] Session 1 event: {single} ({pct:.1f}%) "
              f"— cân nhắc tăng idle_threshold hoặc lọc bỏ")

    # Cảnh báo session quá lớn
    giant = (sess_counts > MAX_SESSION_EVENTS * 0.8).sum()
    if giant > 0:
        print(f"  [Cảnh báo] Session gần đạt max_events: {giant} "
              f"— cân nhắc tăng MAX_SESSION_EVENTS hoặc giảm idle_threshold")

    print(f"────────────────────────────────────────────────────\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HÀM PHÂN TÍCH: Giúp chọn idle_threshold phù hợp
# ══════════════════════════════════════════════════════════════════════════════

def analyze_idle_distribution(
    df: pd.DataFrame,
    user_col: str = 'db_user',
    time_col: str = 'timestamp',
    percentiles: list = [50, 75, 90, 95, 99],
) -> pd.DataFrame:
    """
    Phân tích phân phối khoảng cách idle giữa các event liên tiếp.
    Dùng để chọn idle_threshold_min phù hợp trước khi gán session.

    In ra bảng percentile để dễ quyết định:
        Nếu P90 = 25 phút → chọn idle_threshold = 30 phút là hợp lý.
        Nếu P90 = 3 phút  → hệ thống query liên tục, dùng 5 phút.

    Returns:
        DataFrame thống kê per user.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([user_col, time_col])

    records = []
    for user, grp in df.groupby(user_col):
        ts     = grp[time_col].sort_values()
        deltas = ts.diff().dt.total_seconds().dropna() / 60.0   # phút
        if len(deltas) == 0:
            continue
        rec = {'db_user': user, 'n_events': len(grp)}
        for p in percentiles:
            rec[f'p{p}_min'] = np.percentile(deltas, p)
        rec['max_idle_min'] = deltas.max()
        rec['mean_idle_min'] = deltas.mean()
        records.append(rec)

    result = pd.DataFrame(records)
    if result.empty:
        return result

    print("\n── Idle Time Distribution (phút) ────────────────────")
    global_deltas = []
    for user, grp in df.groupby(user_col):
        ts = grp[time_col].sort_values()
        d  = ts.diff().dt.total_seconds().dropna() / 60.0
        global_deltas.extend(d.tolist())

    global_deltas = np.array(global_deltas)
    for p in percentiles:
        print(f"  P{p:2d}: {np.percentile(global_deltas, p):.1f} phút")
    print(f"  Max : {global_deltas.max():.1f} phút")
    print()

    # Đề xuất threshold
    p90 = np.percentile(global_deltas, 90)
    suggested = max(5, round(p90 * 1.5 / 5) * 5)   # làm tròn 5 phút, nhân 1.5
    print(f"  → Đề xuất idle_threshold = {suggested} phút "
          f"(≈ 1.5 × P90={p90:.1f}min, làm tròn 5min)")
    print(f"────────────────────────────────────────────────────\n")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TÍCH HỢP VÀO a01 — snippet thay thế trực tiếp
# ══════════════════════════════════════════════════════════════════════════════

"""
HƯỚNG DẪN TÍCH HỢP vào a01_data_prepocessing.py:

    # Thêm ở đầu file:
    from session_builder import assign_session_ids, analyze_idle_distribution

    # Thêm vào preprocess_pipeline(), SAU khi đọc audit_df, TRƯỚC gán nhãn:

    # (Tuỳ chọn) Phân tích distribution trước để chọn threshold đúng:
    analyze_idle_distribution(audit_df, idle_threshold_min=30)

    # Gán session_id tự động (chọn chiến lược tốt nhất theo cột có sẵn):
    audit_df = assign_session_ids(
        audit_df,
        idle_threshold_min = 30,   # điều chỉnh theo kết quả analyze_idle_distribution()
        strategy           = 'auto',
    )

    # Sau đó tiếp tục pipeline bình thường (gán nhãn, SQL features, v.v.)
"""
