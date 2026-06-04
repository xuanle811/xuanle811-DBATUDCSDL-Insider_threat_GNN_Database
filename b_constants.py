# Di chuyển các hằng số từ a03_train.py sang đây
# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS        = 2
LR            = 1e-3
WINDOW_SIZE   = 2     # 10   # k session gần nhất cho GRU
BATCH_SIZE    = 8        # Sửa thành 32 nếu máy khoe    
GNN_HIDDEN    = 64 
GNN_OUT       = 32 
GRU_HIDDEN    = 32 
DROPOUT       = 0.3
NUM_WORKERS   = 0
NUM_NEIGHBORS = {
    ("user",         "logs_on",       "session"):      [3,2], #[10, 5], # lấy tối đa 10 session và 5 neighbor cho session đó (Example: pc, device)
    # ("user",         "uses_device",   "device"):       [2,1], #[5,  3],
    ("user",         "exports_file",  "file_object"):  [2,1], #[5,  3],
    ("user",         "visits",        "web_resource"): [2,1], #[5,  3],
    ("user",         "sends_email",   "email"):        [2,1], #[5,  3],
    ("user",         "has_role",      "role"):         [2,1], #[-1, -1],
    ("user",         "belongs_to",    "department"):   [2,1], #[-1, -1],
    ("user",         "reports_to",    "user"):         [2,1], #[5,  3],
    ("session",      "on_pc",         "pc"):           [-1, -1],
    # ("device",       "attached_to",   "pc"):           [-1, -1],
    ("file_object",  "has_topic",     "topic"):        [2,1], #[5,  3],
    ("web_resource", "has_topic",     "topic"):        [2,1], #[5,  3],
    ("email",        "has_topic",     "topic"):        [2,1], #[5,  3],
    ("email",        "sent_to",       "user"):         [2,1], #[5,  3],
    # reverse edges (ToUndirected)
    ("session",      "rev_logs_on",       "user"):     [3,2], #[10, 5],
    # ("device",       "rev_uses_device",   "user"):     [2,1], #[5,  3],
    ("file_object",  "rev_exports_file",  "user"):     [2,1], #[5,  3],
    ("web_resource", "rev_visits",        "user"):     [2,1], #[5,  3],
    ("email",        "rev_sends_email",   "user"):     [2,1], #[5,  3],
    ("role",         "rev_has_role",      "user"):     [2,1], #[-1, -1],
    ("department",   "rev_belongs_to",    "user"):     [2,1], #[-1, -1],
    ("pc",           "rev_on_pc",         "session"):  [-1, -1],
    # ("pc",           "rev_attached_to",   "device"):   [-1, -1],
    ("topic",        "rev_has_topic",     "file_object"): [2,1], #[5,  3],
    ("topic",        "rev_has_topic",     "web_resource"):[2,1], #[5,  3],
    ("topic",        "rev_has_topic",     "email"):    [2,1], #[5,  3],
    ("user",         "rev_sent_to",       "email"):    [2,1], #[5,  3],
    ("pc",   "has_usb_activity",     "user"):   [2, 1],  # USB edge mới (a02)
    ("user", "rev_has_usb_activity", "pc"):     [2, 1],  # reverse USB
}

