"""
pe_inject_pipeline.py

Pipeline chèn opcode benign vào PE binary theo kỹ thuật trampoline,
nhằm gây nhiễu CFG và đánh lừa mô hình GNN phân loại malware.

Tổng quan pipeline
------------------
Phase 1+2 (file_to_cfg_main.ipynb + a1_cfg_graph_utils.py):
    PE → CFG → centrality + GNNExplainer → important node/edge

Phase 3 (file này):
    important node/edge
    → xác định basic block cần patch (Step 1)
    → đọc raw bytes của block (Step 2)
    → decode instruction bằng Capstone (Step 3)
    → chọn vùng opcode_i an toàn (Step 4 + 5)
    → lấy benign opcode pool (Step 6)
    → tiền xử lý sequence_b → sequence_clean → sequence'_b (Step 7 + 8)
    → tạo trampoline stub (Step 9)
    → thêm section mới vào PE bằng LIEF (Step 10)
    → patch JMP E9 tại code gốc (Step 11)
    → validate PE / disassembly / CFG / semantics (Step 12)
    → đưa patched.exe vào GNN để đánh giá attack (Step 13)

Yêu cầu:
    pip install lief capstone angr torch torch-geometric

Phụ thuộc (import):
    a1_cfg_graph_utils.py      — run_cfg_pipeline, get_node_addr, va_to_rva, rva_to_file_offset,
                              get_section_name, get_block, node_to_dict, make_block_id,
                              NODE_FEATURE_DIM, FEATURE_NAMES, build_node_feature_vector
    benign_pool_builder.py  — build_benign_pool, select_best_sequence,
                              compute_cosine_dissimilarity
    a4_train_gnn_models.py     — build_pyg_data (truyền vào qua tham số build_pyg_data_fn)

Thứ tự gọi đúng:
    1. a4_train_gnn_models.py  → tạo model + scaler
    2. a4_train_gnn_models.export_malware_feature_vector()
    → tạo malware_feature_vec.npy cho malware adversarial target
    3. benign_pool_builder.py → build pool dùng malware_feature_vec.npy
    4. a2_file_to_cfg_main.ipynb → chọn target_node
    5. a5_pe_inject_multi_budget_pipeline.py → run_inject_pipeline(...)
"""

import os
import json
import logging
import struct
import hashlib
import math
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import shutil
import copy

import lief
import capstone
import angr
import networkx as nx
import numpy as np
import pickle
import torch

# Import từ Phase 1/2
# a1_cfg_graph_utils: pipeline chính + feature vector helpers
from a1_cfg_graph_utils import (
    run_cfg_pipeline,
    build_cfg,
    node_to_dict,
    get_node_addr,
    va_to_rva,
    rva_to_file_offset,
    get_section_name,
    get_block,
    make_block_id,
    NODE_FEATURE_DIM,       # số chiều feature vector (49) — dùng để build PyG data
    FEATURE_NAMES,          # tên 49 feature — dùng để debug/log
    build_node_feature_vector,  # build feature vector cho node mới khi simulate CFG
)

# benign_pool_builder: build pool + chọn sequence
from a3_benign_pool_builder import (
    build_benign_pool,
    select_best_sequence,
    compute_cosine_dissimilarity,
)

# train_gnn_models: load model/scaler + build PyG data + export malware feature vector
from a4_train_gnn_models import (
    load_model,
    load_scaler,
    build_pyg_data,
    export_malware_feature_vector, FeaturePreprocessor,

)
logger = logging.getLogger(__name__)


# ============================================================
# Hằng số
# ============================================================

# Nhóm mnemonic nguy hiểm — không được chọn làm opcode_i (Step 4/5)
UNSAFE_MNEMONICS_PATCH = {
    "jmp", "je", "jne", "jz", "jnz", "jg", "jge", "jl", "jle",
    "ja", "jae", "jb", "jbe", "js", "jns", "jo", "jno", "jp", "jnp",
    "call", "ret", "retn", "retf",
    "int", "int3", "ud2",
    "syscall", "sysenter", "sysexit",
    "loop", "loope", "loopne",
    "cli", "sti",
    "in", "out", "ins", "outs",
    "div", "idiv",
    # stack frame
    "push", "pop", "leave", "enter",
}

# Mnemonic cần tránh trong sequence benign (Step 7)
UNSAFE_MNEMONICS_BENIGN = {
    "jmp", "je", "jne", "jz", "jnz", "jg", "jge", "jl", "jle",
    "ja", "jae", "jb", "jbe", "js", "jns", "jo", "jno", "jp", "jnp",
    "call", "ret", "retn",
    "loop", "loope", "loopne",
    "int", "int3", "ud2",
    "syscall", "sysenter", "div", "idiv",
    "in", "out", "cli", "sti",
    # stack
    "push", "pop", "leave", "enter",
}

# Mnemonic thao tác stack/frame — cảnh báo thêm
STACK_MNEMONICS = {"mov", "add", "sub", "lea"}

# Kích thước lệnh JMP E9 (relative 32-bit)
JMP_SIZE = 5

# Tên section mới sẽ thêm vào PE
TRAMP_SECTION_NAME = ".adv"

# Quyền section trampoline: READ | EXECUTE | CODE
TRAMP_SECTION_FLAGS = (
    lief.PE.SECTION_CHARACTERISTICS.MEM_READ
    | lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE
    | lief.PE.SECTION_CHARACTERISTICS.CNT_CODE
)

CONTEXT_SIM_MIN = 0.15 # Sửa tăng lên nếu muốn chặt hơn
CONTEXT_SIM_MAX = 0.95 # Sửa giảm xuống nếu muốn chặt hơn



def _cosine_dissimilarity_safe(
    vec_a: Optional[List[float]],
    vec_b: Optional[List[float]],
    neutral: float = 0.5,
) -> float:
    """
    Tính cosine dissimilarity an toàn cho pipeline attack.

    benign_pool_builder có thể còn chứa pool cũ 8 chiều hoặc vector mới
    NODE_FEATURE_DIM chiều. Nếu hai vector khác số chiều, hàm trả về neutral
    thay vì làm crash budget loop. Điều này giữ backward compatibility với pool
    JSON cũ nhưng vẫn ưu tiên vector GNN đúng schema khi có.
    """
    if vec_a is None or vec_b is None:
        return neutral
    if len(vec_a) == 0 or len(vec_b) == 0 or len(vec_a) != len(vec_b):
        return neutral
    return compute_cosine_dissimilarity(vec_a, vec_b)


# ============================================================
# Dataclass-like dicts — kết quả trung gian
# ============================================================

def make_patch_block(va: int, rva: int, file_offset: int, size: int) -> Dict:
    """
    Tạo dict chuẩn mô tả basic block sẽ bị patch.

    Hàm chỉ gom metadata địa chỉ, không thay đổi PE. Dữ liệu này được truyền
    qua Step 2–4 để đọc byte thật, decode instruction và chọn vùng overwrite.
    """
    return {
        "patch_block_va": va,
        "patch_block_rva": rva,
        "patch_block_file_offset": file_offset,
        "patch_block_size": size,
    }


def make_opcode_region(
    start_va: int,
    file_offset: int,
    length: int,
    stolen_bytes: bytes,
    instruction_list: List[Dict],
    return_va: int,
) -> Dict:
    """
    Tạo dict chuẩn cho vùng opcode_i an toàn được chọn để ghi đè bằng JMP E9.

    stolen_bytes là các instruction gốc bị overwrite; chúng sẽ được copy sang
    trampoline rồi nhảy về return_va để bảo toàn luồng thực thi ban đầu.
    """
    return {
        "start_va": start_va,
        "file_offset": file_offset,
        "length": length,
        "stolen_bytes": stolen_bytes,
        "instruction_list": instruction_list,
        "return_va": return_va,
    }


# ============================================================
# Step 1 — Xác định basic block cần patch
# ============================================================

def resolve_patch_block(
    project: angr.Project,
    cfg,
    target_node=None,
    target_edge: Optional[Tuple] = None,
    target_nodes: Optional[List[Any]] = None,
) -> Dict:
    """
    Từ target_node hoặc target_edge, xác định block cần patch.

    Ưu tiên:
    - target_node  → block tương ứng với node đó
    - target_edge = (u, v) → chọn block u (nguồn của cạnh quan trọng)

    Returns:
        patch_block dict với va, rva, file_offset, size
    Raises:
        ValueError nếu không resolve được địa chỉ
    """
    if target_edge is not None:
        node = target_edge[0]
        logger.info(f"[Step 1] Dùng target_edge → chọn block u: {node}")
    elif target_node is not None:
        node = target_node
        logger.info(f"[Step 1] Dùng target_node: {node}")
    else:
        raise ValueError("Phải cung cấp ít nhất target_node hoặc target_edge.")

    va = get_node_addr(node)
    if va is None:
        raise ValueError(f"Không lấy được VA của node {node}")

    rva = va_to_rva(project, va)
    if rva is None:
        raise ValueError(f"Không convert VA→RVA cho node {node}")

    file_offset = rva_to_file_offset(project, rva)
    if file_offset is None:
        raise ValueError(f"Không convert RVA→file_offset cho node {node}")

    block = get_block(project, node)
    size = block.size if block else 0
    if size < JMP_SIZE:
        raise ValueError(f"Block size={size} tại {hex(va)} < JMP_SIZE={JMP_SIZE}, không thể patch.")

    section = get_section_name(project, va)
    logger.info(
        f"[Step 1] patch_block: VA={hex(va)}, RVA={hex(rva)}, "
        f"offset={hex(file_offset)}, size={size}, section={section}"
    )

    result = make_patch_block(va, rva, file_offset, size)
    result["section_name"] = section
    result["node"] = node
    return result


# ============================================================
# Step 2 — Đọc raw bytes của basic block từ PE
# ============================================================

def read_block_bytes(pe_path: str, patch_block: Dict) -> bytes:
    """
    Đọc raw bytes của block từ file PE dựa vào file_offset và block_size.

    Không patch trực tiếp từ CFG — CFG chỉ là biểu diễn logic.
    Muốn sửa file PE phải thao tác trên byte thật.
    """
    offset = patch_block["patch_block_file_offset"]
    size = patch_block["patch_block_size"]

    with open(pe_path, "rb") as f:
        f.seek(offset)
        block_bytes = f.read(size)

    if len(block_bytes) != size:
        raise IOError(
            f"Đọc được {len(block_bytes)} bytes, expected {size} "
            f"tại offset {hex(offset)}"
        )

    logger.info(f"[Step 2] Đọc {len(block_bytes)} bytes tại offset {hex(offset)}")
    return block_bytes


# ============================================================
# Step 3 — Decode instruction bằng Capstone
# ============================================================

def decode_instructions(
    block_bytes: bytes,
    base_va: int,
    arch: str = "x86",
    bits: int = 32,
) -> List[Dict]:
    """
    Dùng Capstone để decode raw bytes thành danh sách instruction.

    Mỗi instruction dict gồm:
        address, mnemonic, op_str, size, bytes

    Quan trọng: khi patch JMP E9, không được ghi đè giữa một instruction.
    Nếu ghi đè giữa instruction thì byte còn lại bị hiểu thành instruction
    rác và chương trình crash.
    """
    if arch == "x86":
        if bits == 32:
            cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
        else:
            cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    else:
        raise ValueError(f"Arch không hỗ trợ: {arch}")

    cs.detail = True

    instructions = []
    for insn in cs.disasm(block_bytes, base_va):
        instructions.append({
            "address": insn.address,
            "mnemonic": insn.mnemonic.lower(),
            "op_str": insn.op_str,
            "size": insn.size,
            "bytes": bytes(insn.bytes),
        })

    logger.info(f"[Step 3] Decode được {len(instructions)} instructions tại VA={hex(base_va)}")
    return instructions


# ============================================================
# Step 4 + 5 — Chọn vùng opcode_i an toàn và safety filter
# ============================================================

def _is_safe_instruction(insn: Dict) -> bool:
    """
    Kiểm tra một instruction có an toàn để relocate không.

    Loại bỏ:
    - jump/call/ret → thay đổi control flow
    - int/int3/ud2 → trap/exception
    - loop → phụ thuộc ECX
    - cli/sti/in/out → interrupt/hardware
    - div/idiv → có thể divide-by-zero
    - push/pop/leave/enter → thay đổi stack frame

    Cảnh báo thêm với mov esp/ebp, add/sub esp/ebp.
    """
    mnem = insn["mnemonic"]
    op = insn["op_str"].lower()

    if mnem in UNSAFE_MNEMONICS_PATCH:
        return False

    # mov/add/sub với esp hoặc ebp → nguy hiểm cho stack frame
    if mnem in STACK_MNEMONICS:
        if "esp" in op or "ebp" in op or "rsp" in op or "rbp" in op:
            return False

    return True


def find_safe_opcode_region(
    instructions: List[Dict],
    base_va: int,
    base_file_offset: int,
    min_length: int = JMP_SIZE,
) -> Optional[Dict]:
    """
    Tìm vùng instruction liên tiếp đủ dài và an toàn để overwrite bằng JMP E9.

    Thuật toán sliding window:
    - Duyệt từng vị trí bắt đầu có thể
    - Tích lũy instruction an toàn liên tiếp cho đến khi đủ min_length bytes
    - Dừng lại ngay khi gặp instruction không an toàn

    Ưu tiên: region đầu tiên thoả mãn (bảo thủ).

    Không cố xử lý mọi trường hợp — nếu không tìm được region
    thì trả về None, caller chọn node/edge khác.
    """
    # Tích lũy bytes_before O(n) — không tính lại từ đầu mỗi vòng
    bytes_before = 0
    for start_idx, start_insn in enumerate(instructions):
        if not _is_safe_instruction(start_insn):
            bytes_before += start_insn["size"]
            continue

        # Tích lũy instruction liên tiếp an toàn từ start_idx
        region_insns = []
        total_bytes = 0

        for j in range(start_idx, len(instructions)):
            insn = instructions[j]
            if not _is_safe_instruction(insn):
                break
            region_insns.append(insn)
            total_bytes += insn["size"]
            if total_bytes >= min_length:
                break

        if total_bytes < min_length:
            bytes_before += start_insn["size"]
            continue

        start_va = start_insn["address"]
        file_offset = base_file_offset + bytes_before
        stolen_bytes = b"".join(i["bytes"] for i in region_insns)
        return_va = start_va + total_bytes

        region = make_opcode_region(
            start_va=start_va,
            file_offset=file_offset,
            length=total_bytes,
            stolen_bytes=stolen_bytes,
            instruction_list=region_insns,
            return_va=return_va,
        )

        logger.info(
            f"[Step 4] opcode_i: VA={hex(start_va)}, length={total_bytes}, "
            f"return_va={hex(return_va)}, instructions={len(region_insns)}"
        )
        return region

    logger.warning("[Step 4] Không tìm được vùng an toàn trong block này.")
    return None


# ============================================================
# Step 6 — Load pool từ benign_pool_builder
# ============================================================

# Step 6 được xử lý hoàn toàn bởi benign_pool_builder.py.
# File này chỉ import: build_benign_pool, select_best_sequence.
# (đã import ở đầu file)


# ============================================================
# Encoding cố định (dùng cho Step 8 và Step 9)
# ============================================================

PUSHFD = bytes([0x9C])
PUSHAD = bytes([0x60])
POPAD  = bytes([0x61])
POPFD  = bytes([0x9D])
NOP    = bytes([0x90])


# ============================================================
# Step 8 — Chọn sequence từ pool + budget loop + CFG simulation
# ============================================================

def _score_sequence(seq: Dict, malware_feature_vec: Optional[List[float]]) -> float:
    """
    Tính điểm tổng hợp để rank sequence trong pool.

    Tiêu chí (đúng spec):
        - cosine_dissimilarity lớn  → feature xa malware   (trọng số 0.6)
        - sequence_frequency_norm cao → pattern phổ biến   (trọng số 0.4)

    Score cao hơn = tốt hơn.
    """
    fv = seq.get("gnn_stub_feature_vector", seq.get("feature_vector", []))
    if malware_feature_vec and fv:
        dissim = _cosine_dissimilarity_safe(
            fv,
            malware_feature_vec,
            neutral=seq.get("cosine_dissimilarity", 0.5),
        )
    else:
        dissim = seq.get("cosine_dissimilarity", 0.5)

    freq_norm = seq.get("sequence_frequency_norm", 0.0)
    return 0.6 * dissim + 0.4 * freq_norm


def filter_pool_by_budget(
    pool: List[Dict],
    binary_size_bytes: int,
    budget_ratio: float,
) -> List[Dict]:
    """
    Lọc pool chỉ giữ sequence có size_bytes <= budget_ratio * binary_size.

    Đây là ràng buộc kích thước: sequence'_b <= budget% file malware.
    """
    max_bytes = max(1, int(binary_size_bytes * budget_ratio))
    filtered = [
        seq for seq in pool
        if seq.get("sequence_prime", {}).get("size_bytes", 0) <= max_bytes
    ]
    logger.info(
        f"[Step 8] Budget {budget_ratio*100:.0f}%: "
        f"max={max_bytes}B, eligible={len(filtered)}/{len(pool)}"
    )
    return filtered


def rank_pool(
    pool: List[Dict],
    malware_feature_vec: Optional[List[float]],
) -> List[Dict]:
    """
    Xếp hạng pool theo score tổng hợp (dissimilarity + frequency).
    Trả về bản copy đã sắp xếp, giảm dần theo score.
    """
    scored = [(seq, _score_sequence(seq, malware_feature_vec)) for seq in pool]
    scored.sort(key=lambda x: -x[1])
    return [s for s, _ in scored]


def simulate_cfg_bypass(
    G: nx.DiGraph,
    node_types: Dict,
    centrality_dict: Dict,
    sequence_record: Dict,
    model,
    build_pyg_data_fn,
    device,
    original_prob_malware: float,
    scaler=None,
    target_node=None,
) -> Tuple[bool, float, nx.DiGraph]:
    """
    Giả lập chèn node benign vào CFG (không patch binary thật),
    rồi chạy GNN để kiểm tra có bypass không.

    Cách giả lập:
    1. Copy graph G
    2. Thêm một "benign_stub" node mới vào graph copy
    3. Nối nó đúng vào target_node đã chọn ở Phase 1/2
       bằng 2 edge: target_node → stub → successor đầu tiên của target_node
       (mô phỏng trampoline làm thêm path trong CFG).
       Lưu ý: bản này KHÔNG dùng target_edge theo yêu cầu hiện tại.
    4. Tạo feature vector cho node mới dựa vào sequence'_b
    5. Convert sang PyG → chạy GNN inference
    6. Trả về (bypassed, prob_malware_new, G_sim)

    Args:
        G                   : pruned CFG gốc
        node_types          : dict node → type
        centrality_dict     : dict centrality scores
        sequence_record     : một sequence_b' từ pool
        model               : GNN model
        build_pyg_data_fn   : hàm build_pyg_data
        device              : torch device
        original_prob_malware: xác suất malware của graph gốc
        target_node          : node thật sẽ bị patch JMP E9 ở binary.
                               Simulation phải bám đúng node này để kết quả
                               gần với patch thật, không tự chọn node khác.

    Returns:
        (bypassed, prob_malware_after, G_sim)
    """


    G_sim = G.copy()  # copy đầy đủ node/edge attributes, tránh sửa nhầm CFG gốc

    # ============================================================
    # Chọn anchor cho CFG simulation
    # ============================================================
    # Yêu cầu hiện tại: simulation phải dùng ĐÚNG target_node sẽ patch thật.
    # Không dùng target_edge và cũng không tự động chọn node có score cao nhất,
    # vì như vậy mô phỏng một vị trí nhưng Step 11 lại patch vị trí khác.
    if target_node is None:
        logger.warning(
            "[Step 8] target_node=None nên không thể simulation đúng vị trí patch. "
            "Bỏ qua sequence này."
        )
        return False, original_prob_malware, G_sim

    if target_node not in G_sim:
        logger.warning(
            f"[Step 8] target_node không tồn tại trong G_sim: {target_node}. "
            "Có thể node object không khớp với pruned_graph hoặc cfg_result khác."
        )
        return False, original_prob_malware, G_sim

    anchor = target_node
    anchor_succs = list(G_sim.successors(anchor))

    # Tạo node id mới
    stub_id = f"benign_stub_{sequence_record.get('block_addr', 'x')}"

    # Tính feature vector cho stub từ sequence'_b
    seq_bytes_hex = sequence_record.get("sequence_prime", {}).get("bytes_hex", "")
    try:
        seq_bytes = bytes.fromhex(seq_bytes_hex)
    except Exception:
        seq_bytes = bytes([0x90]) * 10

    fv_raw = sequence_record.get("gnn_stub_feature_vector")

    if fv_raw is None:
        fv_raw = sequence_record.get("feature_vector")

    if fv_raw is None or len(fv_raw) != NODE_FEATURE_DIM:
        raise ValueError(
            f"Sequence không có GNN stub vector {NODE_FEATURE_DIM} chiều. "
            f"Hiện tại len={len(fv_raw) if fv_raw is not None else None}"
        )

    # Số chiều feature của graph gốc — lấy từ node đầu tiên để debug tương thích.
    # Không ép resize fv_raw vì làm vậy sẽ phá schema FEATURE_NAMES từ a1_cfg_graph_utils.
    sample_node = next(iter(G_sim.nodes()))
    existing_fv = G_sim.nodes[sample_node].get("feature_vector", [])
    target_dim = len(existing_fv)
    if target_dim not in (0, NODE_FEATURE_DIM):
        logger.warning(
            f"[Step 8] Graph feature dim={target_dim}, expected={NODE_FEATURE_DIM}. "
            "Vẫn dùng vector từ a1_cfg_graph_utils để không làm lệch schema."
        )

    # Thêm node stub vào graph
    G_sim.add_node(stub_id)
    G_sim.nodes[stub_id]["node_type"] = "benign_stub"
    G_sim.nodes[stub_id]["feature_vector"] = fv_raw
    G_sim.nodes[stub_id]["structural_score"] = 0.0
    G_sim.nodes[stub_id]["block_size"] = len(seq_bytes)
    G_sim.nodes[stub_id]["instruction_count"] = sequence_record.get(
        "sequence_clean", {}
    ).get("size_bytes", len(seq_bytes)) // 3 or 1

    # Nối: anchor → stub → anchor_succ (nếu có)
    G_sim.add_edge(anchor, stub_id)
    if anchor_succs:
        G_sim.add_edge(stub_id, anchor_succs[0])

    # Chạy GNN trên CFG giả lập
    try:
        data, _, _ = build_pyg_data_fn(G_sim, scaler=scaler)
        data = data.to(device)
        model.eval()
        with torch.no_grad():
            # Thêm:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            logits = model(data.x, data.edge_index, data.batch)
            probs = torch.softmax(logits, dim=-1)
            prob_malware_new = float(probs.view(-1)[1].item())
    except Exception as e:
        logger.warning(f"[Step 8] GNN sim error: {e}")
        return False, original_prob_malware, G_sim

    # Bypass nếu GNN phân loại thành benign (< 0.5)
    bypassed = prob_malware_new < 0.5
    return bypassed, prob_malware_new, G_sim


def run_budget_loop(
    pool: List[Dict],
    binary_size_bytes: int,
    G: nx.DiGraph,
    node_types: Dict,
    centrality_dict: Dict,
    model,
    build_pyg_data_fn,
    device,
    malware_feature_vec: Optional[List[float]],
    original_prob_malware: float,
    budget_levels: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10),
    scaler = None,
    target_node=None,
) -> Tuple[Optional[Dict], Dict]:
    """
    Step 8 — Budget loop: thử từng budget level, tăng dần.

    Với mỗi budget (1% → 3% → 5% → 10%):
        1. Lọc pool theo kích thước <= budget * binary_size
        2. Rank pool theo (dissimilarity ↑, frequency ↑)
        3. Thử từng sequence theo thứ tự rank:
            - Giả lập chèn vào CFG tại đúng target_node (không patch binary)
            - Chạy GNN trên CFG giả lập
            - Nếu bypass (prob_malware < 0.5) → dừng, trả về sequence này
        4. Nếu hết sequence trong budget → tăng lên budget tiếp theo
        5. Nếu hết 10% mà vẫn chưa bypass → trả về sequence tốt nhất đã thử

    Returns:
        (chosen_sequence, budget_report)
        chosen_sequence: sequence_b' được chọn (hoặc None nếu pool rỗng)
        budget_report  : dict ghi lại kết quả từng budget level

    Ghi chú logic:
        - target_node được truyền từ run_inject_pipeline xuống simulate_cfg_bypass.
        - Không dùng target_edge trong phiên bản này theo yêu cầu test hiện tại.
    """
    budget_report = {
        "budget_levels_tried": [],
        "bypass_found": False,
        "bypass_budget": None,
        "bypass_sequence_addr": None,
        "trials": [],
    }

    best_candidate = None          # sequence tốt nhất tìm được dù chưa bypass
    best_prob_malware = original_prob_malware

    for budget in budget_levels:
        eligible = filter_pool_by_budget(pool, binary_size_bytes, budget)
        if not eligible:
            logger.info(f"[Step 8] Budget {budget*100:.0f}%: pool rỗng, bỏ qua.")
            budget_report["budget_levels_tried"].append({
                "budget": budget,
                "eligible_count": 0,
                "status": "skipped_empty_pool",
            })
            continue

        ranked = rank_pool(eligible, malware_feature_vec)
        bypass_found_this_budget = False

        budget_entry = {
            "budget": budget,
            "eligible_count": len(ranked),
            "trials": [],
            "status": "exhausted",
        }

        logger.info(
            f"[Step 8] ── Budget {budget*100:.0f}% ──  "
            f"{len(ranked)} sequences, prob_malware_orig={original_prob_malware:.4f}"
        )

        for idx, seq in enumerate(ranked):
            bypassed, prob_new, _ = simulate_cfg_bypass(
                G=G,
                node_types=node_types,
                centrality_dict=centrality_dict,
                sequence_record=seq,
                model=model,
                build_pyg_data_fn=build_pyg_data_fn,
                device=device,
                original_prob_malware=original_prob_malware,
                scaler=scaler,
                target_node=target_node,
            )

            trial = {
                "rank": idx + 1,
                "source": seq.get("source_file"),
                "block_addr": seq.get("block_addr"),
                "prob_malware_after": prob_new,
                "bypassed": bypassed,
                "score": _score_sequence(seq, malware_feature_vec),
                "size_bytes": seq.get("sequence_prime", {}).get("size_bytes", 0),
            }
            budget_entry["trials"].append(trial)

            logger.info(
                f"  [trial {idx+1}] {seq.get('block_addr')} | "
                f"prob={prob_new:.4f} | bypass={bypassed} | "
                f"score={trial['score']:.3f}"
            )

            # Cập nhật candidate tốt nhất (prob malware thấp nhất)
            if prob_new < best_prob_malware or best_candidate is None:
                best_prob_malware = prob_new
                best_candidate = seq

            if bypassed:
                budget_entry["status"] = "bypassed"
                bypass_found_this_budget = True
                budget_report["bypass_found"] = True
                budget_report["bypass_budget"] = budget
                budget_report["bypass_sequence_addr"] = seq.get("block_addr")
                budget_report["budget_levels_tried"].append(budget_entry)
                budget_report["trials"] = budget_entry["trials"]
                logger.info(
                    f"[Step 8] ✓ BYPASS tại budget {budget*100:.0f}%, "
                    f"sequence {seq.get('block_addr')}, prob_malware={prob_new:.4f}"
                )
                return seq, budget_report

        budget_report["budget_levels_tried"].append(budget_entry)

        if bypass_found_this_budget:
            break

    # Hết tất cả budget mà chưa bypass
    logger.warning(
        f"[Step 8] Hết budget, chưa bypass. "
        f"Best prob_malware={best_prob_malware:.4f}. "
        f"Dùng candidate tốt nhất để patch."
    )
    budget_report["trials"] = budget_report["budget_levels_tried"][-1].get("trials", []) if budget_report["budget_levels_tried"] else []
    return best_candidate, budget_report


def wrap_sequence_prime(sequence_record: Dict) -> bytes:
    """
    Lấy bytes của sequence'_b từ record pool (đã được bọc pushfd/pushad/popad/popfd
    bởi benign_pool_builder).

    Không cần bọc lại — pool builder đã làm rồi.
    """
    hex_str = sequence_record.get("sequence_prime", {}).get("bytes_hex", "")
    if not hex_str:
        raise ValueError(
            f"sequence_prime.bytes_hex rỗng cho record {sequence_record.get('block_addr')}"
        )
    result = bytes.fromhex(hex_str)
    if len(result) == 0:
        raise ValueError(
            f"sequence_prime bytes rỗng sau fromhex cho record {sequence_record.get('block_addr')}"
        )
    return result


# ============================================================
# Step 9 — Tạo trampoline stub
# ============================================================

def _encode_jmp_rel32(src_va: int, dst_va: int) -> bytes:
    """
    Encode lệnh JMP E9 relative 32-bit.
    rel32 = destination - (source + 5)
    """
    rel32 = dst_va - (src_va + 5)
    # Dùng signed 32-bit để xử lý backward jump
    rel32_packed = struct.pack("<i", rel32)
    return b"\xe9" + rel32_packed


def build_trampoline(
    sequence_prime_b: bytes,
    stolen_bytes: bytes,
    addr_b: int,
    return_va: int,
    opcode_region: Dict,
) -> bytes:
    """
    Step 9: Tạo trampoline stub đặt vào section mới.

    Cấu trúc cơ bản (phiên bản opcode_i bình thường — không phải jmp/call/ret):

        addr_b:
            <sequence'_b>       ; benign sequence đã bọc context
            <stolen_bytes>      ; các instruction gốc bị overwrite
            jmp return_va       ; nhảy về instruction tiếp theo sau opcode_i

    Lưu ý:
    - Trong bản đầu giả sử opcode_i không chứa relative branch/call.
    - Các trường hợp jmp/call/conditional/trap xử lý sau (Step 12.x trong PDF).
    - addr_b là địa chỉ ảo của trampoline trong section mới — cần biết
      trước khi encode JMP back.

    Returns:
        bytes toàn bộ trampoline stub
    """
    # sequence'_b
    stub = sequence_prime_b

    # stolen instructions (các byte gốc bị overwrite)
    stub += stolen_bytes

    # JMP về return_va
    # src_va = addr_b + len(sequence'_b) + len(stolen_bytes)
    jmp_back_src = addr_b + len(sequence_prime_b) + len(stolen_bytes)
    stub += _encode_jmp_rel32(jmp_back_src, return_va)

    logger.info(
        f"[Step 9] Trampoline: {len(stub)} bytes, "
        f"addr_b={hex(addr_b)}, return_va={hex(return_va)}"
    )
    return stub


# ============================================================
# Step 10 — Thêm section mới vào PE bằng LIEF
# ============================================================

def add_trampoline_section(
    pe_path: str,
    trampoline_bytes: bytes,
    out_path: str,
    section_name: str = TRAMP_SECTION_NAME,
) -> int:
    """
    Step 10: Thêm section mới chứa trampoline vào PE bằng LIEF.

    Section cần quyền: READ | EXECUTE | CODE.
    Nội dung: trampoline_bytes + NOP padding đến alignment.

    Returns:
        addr_b — VA của byte đầu tiên của trampoline trong section mới
                 (= image_base + section.virtual_address, stub đặt ở offset 0)
    """
    binary = lief.parse(pe_path)
    if binary is None:
        raise RuntimeError(f"LIEF không parse được: {pe_path}")

    # Tạo section mới
    section = lief.PE.Section()
    section.name = section_name

    # Padding đến bội số của file_alignment
    align = binary.optional_header.file_alignment
    pad_size = (align - len(trampoline_bytes) % align) % align
    content = list(trampoline_bytes) + [0x90] * pad_size  # NOP padding

    section.content = content
    section.characteristics = (
        lief.PE.SECTION_CHARACTERISTICS.MEM_READ
        | lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE
        | lief.PE.SECTION_CHARACTERISTICS.CNT_CODE
    )

    # Thêm section vào binary
    added = binary.add_section(section)

    # Lấy VA thực của section sau khi LIEF thêm vào
    image_base = binary.optional_header.imagebase
    section_rva = added.virtual_address
    addr_b = image_base + section_rva

    logger.info(
        f"[Step 10] Section '{section_name}' thêm: "
        f"RVA={hex(section_rva)}, VA={hex(addr_b)}, size={len(content)}"
    )

    # Ghi file tạm để lấy addr_b chính xác trước khi patch
    binary.write(out_path)
    logger.info(f"[Step 10] Ghi file tạm: {out_path}")

    return addr_b


# ============================================================
# Step 11 — Patch code gốc bằng JMP E9
# ============================================================

def patch_jmp_e9(
    pe_path: str,
    opcode_region: Dict,
    addr_b: int,
) -> None:
    """
    Step 11: Ghi đè vùng opcode_i bằng JMP E9 tới addr_b (trampoline).

    Công thức:
        E9 rel32
        rel32 = addr_b - (opcode_i.start_va + 5)

    Nếu opcode_i.length > 5 → padding NOP cho các byte dư.

    Ghi thẳng vào pe_path (file đã có section .adv) theo r+b.
    Phải patch trên file patched (không phải file gốc) vì file_offset
    của opcode_i vẫn valid — LIEF chỉ append section, không dịch data cũ.
    """
    start_va = opcode_region["start_va"]
    file_offset = opcode_region["file_offset"]
    length = opcode_region["length"]

    jmp_bytes = _encode_jmp_rel32(src_va=start_va, dst_va=addr_b)
    assert len(jmp_bytes) == JMP_SIZE

    # NOP padding cho các byte dư sau JMP
    nop_padding = NOP * (length - JMP_SIZE)
    patch_bytes = jmp_bytes + nop_padding

    assert len(patch_bytes) == length, (
        f"patch_bytes len {len(patch_bytes)} != opcode_region length {length}"
    )

    with open(pe_path, "r+b") as f:
        f.seek(file_offset)
        f.write(patch_bytes)

    logger.info(
        f"[Step 11] Patch JMP E9: VA={hex(start_va)}, "
        f"addr_b={hex(addr_b)}, nop_pad={len(nop_padding)}"
    )


# ============================================================
# Step 12 — Validate
# ============================================================

def validate_pe_structure(patched_path: str) -> Dict[str, bool]:
    """
    Step 12.1: Kiểm tra PE structure sau patch bằng LIEF.

    Checks:
    - LIEF parse được
    - Section .adv/.tramp tồn tại
    - Section có quyền execute
    - Entry point hợp lệ
    - File alignment hợp lệ
    """
    results = {}

    binary = lief.parse(patched_path)
    results["lief_parse_ok"] = binary is not None
    if binary is None:
        return results

    tramp_sec = binary.get_section(TRAMP_SECTION_NAME)
    results["tramp_section_exists"] = tramp_sec is not None

    if tramp_sec is not None:
        chars = tramp_sec.characteristics
        has_exec = bool(
            chars & lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE
        )
        results["tramp_section_executable"] = has_exec
    else:
        results["tramp_section_executable"] = False

    ep = binary.optional_header.addressof_entrypoint
    results["entry_point_nonzero"] = ep != 0

    fa = binary.optional_header.file_alignment
    results["file_alignment_valid"] = fa > 0 and (fa & (fa - 1)) == 0  # power of 2

    results["all_ok"] = all(results.values())
    logger.info(f"[Step 12.1] PE structure: {results}")
    return results


def validate_disassembly(
    patched_path: str,
    opcode_region: Dict,
    addr_b: int,
    return_va: int,
    arch: str = "x86",
    bits: int = 32,
) -> Dict[str, bool]:
    """
    Step 12.2: Dùng Capstone kiểm tra disassembly sau patch.

    Checks:
    - Tại vị trí gốc: có JMP tới addr_b
    - Không có instruction rác (decode fail)
    """
    results = {}

    binary = lief.parse(patched_path)
    if binary is None:
        results["parse_ok"] = False
        return results

    image_base = binary.optional_header.imagebase
    start_va = opcode_region["start_va"]
    length = opcode_region["length"]

    # Đọc bytes tại vị trí patch
    rva = start_va - image_base
    sec = binary.section_from_rva(rva)
    if sec is None:
        results["patch_site_readable"] = False
        return results

    offset_in_sec = rva - sec.virtual_address
    raw = bytes(sec.content)[offset_in_sec: offset_in_sec + length]

    insns = decode_instructions(raw, start_va, arch=arch, bits=bits)

    if insns and insns[0]["mnemonic"] == "jmp":
        # Kiểm tra địa chỉ đích (E9 relative)
        if raw[:1] == b"\xe9":
            rel32 = struct.unpack_from("<i", raw, 1)[0]
            decoded_dst = start_va + 5 + rel32
            results["jmp_to_trampoline_ok"] = (decoded_dst == addr_b)
        else:
            results["jmp_to_trampoline_ok"] = False
    else:
        results["jmp_to_trampoline_ok"] = False

    results["no_garbage_insn"] = len(insns) > 0

    # Kiểm tra JMP back trong section .adv (byte[-5] phải là 0xE9)
    tramp_sec = binary.get_section(TRAMP_SECTION_NAME)
    if tramp_sec is not None:
        content = bytes(tramp_sec.content)
        results["trampoline_has_jmp_back"] = (
            len(content) >= JMP_SIZE and content[-JMP_SIZE] == 0xE9
        )
    else:
        results["trampoline_has_jmp_back"] = False

    results["all_ok"] = all(results.values())
    logger.info(f"[Step 12.2] Disassembly validate: {results}")
    return results


def validate_cfg(
    original_path: str,
    patched_path: str,
    out_dir: str,
) -> Dict[str, Any]:
    """
    Step 12.3: Build lại CFG của patched.exe và so sánh với original.

    Checks:
    - node count, edge count có thay đổi hợp lý
    - Không crash khi build CFG
    """
    results = {}

    try:
        result_orig = run_cfg_pipeline(
            binary_path=original_path,
            out_dir=os.path.join(out_dir, "orig"),
            verbose=False,
        )
        G_orig = result_orig["pruned_graph"]
        results["orig_nodes"] = G_orig.number_of_nodes()
        results["orig_edges"] = G_orig.number_of_edges()
    except Exception as e:
        logger.warning(f"[Step 12.3] Không build CFG gốc: {e}")
        results["orig_build_ok"] = False
        return results

    try:
        result_patched = run_cfg_pipeline(
            binary_path=patched_path,
            out_dir=os.path.join(out_dir, "patched"),
            verbose=False,
        )
        G_patched = result_patched["pruned_graph"]
        results["patched_nodes"] = G_patched.number_of_nodes()
        results["patched_edges"] = G_patched.number_of_edges()
        results["cfg_build_ok"] = True
    except Exception as e:
        logger.warning(f"[Step 12.3] Không build CFG patched: {e}")
        results["cfg_build_ok"] = False
        return results

    results["node_delta"] = results["patched_nodes"] - results["orig_nodes"]
    results["edge_delta"] = results["patched_edges"] - results["orig_edges"]

    logger.info(f"[Step 12.3] CFG validate: {results}")
    return results


# ============================================================
# Step 13 — Đưa patched.exe vào lại GNN
# ============================================================

def evaluate_attack(
    original_path: str,
    patched_path: str,
    model,
    out_dir: str,
    build_pyg_data_fn,
    scaler=None,
    device=None,
) -> Dict[str, Any]:
    """
    Step 13: Chạy GNN inference trên original và patched, ghi lại kết quả.

    Args:
        model: MalwareGCN đã load weights
        build_pyg_data_fn: hàm build_pyg_data từ notebook
        device: torch device

    Returns:
        dict ghi lại label, confidence trước/sau, node/edge count thay đổi
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = {0: "benign", 1: "malware"}

    def _infer(binary_path, tag):
        result = run_cfg_pipeline(
            binary_path=binary_path,
            out_dir=os.path.join(out_dir, tag),
            verbose=False,
        )
        G = result["pruned_graph"]
        data, _, _ = build_pyg_data_fn(G, scaler=scaler)
        data = data.to(device)
        model.eval()
        with torch.no_grad():
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            logits = model(data.x, data.edge_index, data.batch)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        return {
            "prediction_id": pred,
            "prediction_label": label_map.get(pred, str(pred)),
            "prob_benign": float(probs.view(-1)[0].item()),
            "prob_malware": float(probs.view(-1)[1].item()),
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
        }

    orig_result = _infer(original_path, "orig")
    patched_result = _infer(patched_path, "patched")

    attack_result = {
        "original": orig_result,
        "patched": patched_result,
        "label_changed": orig_result["prediction_id"] != patched_result["prediction_id"],
        "confidence_delta": patched_result["prob_malware"] - orig_result["prob_malware"],
        "node_delta": patched_result["graph_nodes"] - orig_result["graph_nodes"],
        "edge_delta": patched_result["graph_edges"] - orig_result["graph_edges"],
        "patch_success": True,
        "semantic_preserved": None,  # cần sandbox để verify
    }

    logger.info(f"[Step 13] Attack result: {attack_result}")
    return attack_result


# ============================================================
# Multi-budget / multi-node helpers
# ============================================================

def _node_sort_score(node: Any, centrality_dict: Dict) -> float:
    """
    Lấy điểm quan trọng của node theo kết quả Phase 1/2.

    Ưu tiên các key thường có từ a2_file_to_cfg_main/a1_cfg_graph_utils:
        - combined_node_score / node_importance / explainer_node_score
        - structural_score
    Nếu không có thì trả về 0.0 để không làm crash pipeline.
    """
    candidate_keys = [
        "combined_node_score",
        "node_importance",
        "explainer_node_score",
        "gnnexplainer_score",
        "importance_score",
        "structural_score",
    ]
    for key in candidate_keys:
        table = centrality_dict.get(key, {}) if isinstance(centrality_dict, dict) else {}
        if isinstance(table, dict) and node in table:
            try:
                return float(table.get(node, 0.0))
            except Exception:
                return 0.0
    return 0.0


def get_ranked_internal_nodes_from_cfg(
    G: nx.DiGraph,
    node_types: Dict,
    centrality_dict: Dict,
    target_nodes: Optional[List[Any]] = None,
) -> List[Any]:
    """
    Chỉ sử dụng target_nodes đã được file_to_cfg_main ranking sẵn.

    Không fallback, không tự sort lại.
    Nếu không có target_nodes thì thông báo cho user.
    """

    if not target_nodes:
        print("[WARNING] file_to_cfg_main không trả về target_nodes.")
        print("[WARNING] Không thể tiếp tục chọn target node.")
        return []

    internal_set = {
        n for n in G.nodes()
        if node_types.get(n) == "internal"
    }

    ranked = [
        n for n in target_nodes
        if n in internal_set
    ]

    if not ranked:
        print("[WARNING] target_nodes tồn tại nhưng không có node nào thuộc internal nodes.")
        return []

    # Bình thường ranked <= target_nodes vì một số target_nodes có thể là external.
    if len(ranked) < len(target_nodes):
        print(f"[INFO] {len(target_nodes) - len(ranked)} target_nodes không thuộc internal, bỏ qua.")

    print(f"[INFO] Nhận {len(ranked)}/{len(target_nodes)} ranked internal target_nodes từ file_to_cfg_main.")

    return ranked


def select_topk_nodes_for_budget(
    ranked_internal_nodes: List[Any],
    total_internal_nodes: int,
    budget_ratio: float,
) -> List[Any]:
    """
    Selection đúng yêu cầu:
        k = ceil(budget * số internal nodes)
        chọn top-k node từ danh sách đã ranking.
    """
    if total_internal_nodes <= 0:
        return []
    k = max(1, int(math.ceil(budget_ratio * total_internal_nodes)))
    k = min(k, len(ranked_internal_nodes))
    return ranked_internal_nodes[:k]


def _sequence_size(seq: Dict) -> int:
    return int(seq.get("sequence_prime", {}).get("size_bytes", 0) or 0)


def _score_sequence_for_matching(
    seq: Dict,
    malware_feature_vec: Optional[List[float]],
    binary_size_bytes: int,
) -> float:
    """
    Matching score theo đúng nhóm tiêu chí đã nêu:
        - dissimilarity cao hơn tốt hơn
        - frequency cao hơn tốt hơn
        - risk thấp hơn tốt hơn
        - size nhỏ hơn tốt hơn

    Không bịa field mới bắt buộc: nếu pool chưa có risk_score hoặc
    sequence_frequency_norm thì dùng default an toàn.
    """
    fv = seq.get("gnn_stub_feature_vector", seq.get("feature_vector", []))
    if malware_feature_vec and fv:
        dissim = _cosine_dissimilarity_safe(
            fv,
            malware_feature_vec,
            neutral=seq.get("cosine_dissimilarity", 0.5),
        )
    else:
        dissim = float(seq.get("cosine_dissimilarity", 0.5) or 0.5)

    freq = float(seq.get("sequence_frequency_norm", 0.0) or 0.0)
    risk = float(seq.get("risk_score", 0.0) or 0.0)
    size = _sequence_size(seq)
    size_norm = size / max(1, binary_size_bytes)

    # score cao hơn = tốt hơn
    return (0.55 * dissim) + (0.25 * freq) - (0.15 * risk) - (0.05 * size_norm)


def rank_pool_for_matching(
    pool: List[Dict],
    malware_feature_vec: Optional[List[float]],
    binary_size_bytes: int,
    max_sequence_bytes: Optional[int] = None,
) -> List[Dict]:
    """
    Rank pool theo dissimilarity, frequency, risk, size.
    max_sequence_bytes dùng để loại sequence quá lớn nếu muốn giới hạn kích thước.
    """
    candidates = []
    for seq in pool:
        size = _sequence_size(seq)
        if size <= 0:
            continue
        if max_sequence_bytes is not None and size > max_sequence_bytes:
            continue
        candidates.append((seq, _score_sequence_for_matching(seq, malware_feature_vec, binary_size_bytes)))
    candidates.sort(key=lambda x: -x[1])
    return [seq for seq, _ in candidates]

# ============================================================
# Match the most suitable sequence to each selected node, with candidate filtering and final scoring.
def match_sequences_to_nodes(
    selected_nodes: List[Any],
    pool: List[Dict],
    malware_feature_vec: Optional[List[float]],
    binary_size_bytes: int,
    total_budget_bytes: int = 0,
    G: nx.DiGraph = None,
    tau_risk: float = 0.5,
    context_sim_min: float = CONTEXT_SIM_MIN,
    context_sim_max: float = CONTEXT_SIM_MAX,
    per_node_max_ratio: float = 0.03,
) -> List[Dict[str, Any]]:
    """
    Matching per-node with candidate filtering and Final_Score per spec.

    - Candidate Filtering: Risk_Score < tau_risk,
      ContextSim(seq, node) in [context_sim_min, context_sim_max],
      sequence size <= per_node_max_ratio * binary_size_bytes
    - Final_Score = 0.4 * Dissimilarity_to_Malware
                    + 0.3 * Similarity_to_Benign
                    + 0.2 * Frequency_Norm
                    - 0.1 * Risk_Score

    Assign sequences greedily per-node, avoid reusing sequence if possible,
    """
    # Precompute benign centroid for Similarity_to_Benign
    benign_vectors = []
    for s in pool:
        fv = s.get("gnn_stub_feature_vector") or s.get("feature_vector")
        if fv and len(fv) == NODE_FEATURE_DIM:
            benign_vectors.append(np.array(fv, dtype=float))
    if benign_vectors:
        benign_centroid = np.mean(np.stack(benign_vectors, axis=0), axis=0).tolist()
    else:
        benign_centroid = None

    used_ids = set()
    used_bytes = 0
    pairs: List[Dict[str, Any]] = []

    per_node_max_bytes = max(1, int(binary_size_bytes * per_node_max_ratio))

    for idx, node in enumerate(selected_nodes):
        # build per-node candidate list
        node_feat = None
        if G is not None and node in G.nodes():
            node_feat = G.nodes[node].get("feature_vector")

        candidates = []
        for seq in pool:
            seq_fv = seq.get("gnn_stub_feature_vector") or seq.get("feature_vector")
            if not seq_fv or len(seq_fv) != NODE_FEATURE_DIM:
                continue

            risk = float(seq.get("risk_score", 0.0) or 0.0)
            if risk >= tau_risk:
                continue

            size = _sequence_size(seq)
            if size <= 0 or size > per_node_max_bytes:
                continue

            # Context similarity: higher = more similar
            if node_feat and len(node_feat) == len(seq_fv):
                dissim_node = _cosine_dissimilarity_safe(seq_fv, node_feat, neutral=1.0)
                context_sim = 1.0 - dissim_node
            else:
                # cannot compute context sim -> skip to be conservative
                continue

            if not (context_sim_min <= context_sim <= context_sim_max):
                continue

            # Dissimilarity to malware
            dissim_mal = _cosine_dissimilarity_safe(seq_fv, malware_feature_vec, neutral=0.5)
            # Similarity to benign centroid
            if benign_centroid is not None:
                dissim_benign = _cosine_dissimilarity_safe(seq_fv, benign_centroid, neutral=0.5)
                sim_benign = 1.0 - dissim_benign
            else:
                sim_benign = 0.0

            freq = float(seq.get("sequence_frequency_norm", 0.0) or 0.0)

            final_score = (
                0.4 * dissim_mal + 0.3 * sim_benign + 0.2 * freq - 0.1 * risk
            )

            candidates.append((seq, final_score, size, context_sim))

        # sort candidates for this node
        candidates.sort(key=lambda x: -x[1])

        chosen = None
        for seq, score, size, ctx in candidates:
            seq_id = (seq.get("source_file"), seq.get("block_addr"), seq.get("sequence_prime", {}).get("bytes_hex"))
            if seq_id in used_ids:
                continue
            if total_budget_bytes > 0 and used_bytes + size > total_budget_bytes:
                continue
            used_ids.add(seq_id)
            used_bytes += size
            chosen = seq
            chosen_score = score
            chosen_size = size
            break

        if chosen is None:
            logger.warning(f"[Matching] Không tìm được candidate phù hợp cho node rank={idx+1}.")
            break

        pairs.append({
            "rank": idx + 1,
            "node": node,
            "sequence": chosen,
            "sequence_score": chosen_score,
            "sequence_size_bytes": chosen_size,
        })

    return pairs


def infer_graph_prob_malware(
    G: nx.DiGraph,
    model,
    build_pyg_data_fn,
    device,
    scaler=None,
) -> Tuple[float, int, str]:
    """Chạy GNN trên một NetworkX graph và trả về prob_malware, pred_id, pred_label."""

    label_map = {0: "benign", 1: "malware"}
    data, _, _ = build_pyg_data_fn(G, scaler=scaler)
    data = data.to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.batch)
        probs = torch.softmax(logits, dim=-1)
        pred = int(torch.argmax(probs, dim=-1).item())
    return float(probs.view(-1)[1].item()), pred, label_map.get(pred, str(pred))


def simulate_multi_node_cfg(
    G: nx.DiGraph,
    node_sequence_pairs: List[Dict[str, Any]],
    model,
    build_pyg_data_fn,
    device,
    scaler=None,
) -> Dict[str, Any]:
    """
    CFG Simulation đúng yêu cầu:
        Copy CFG gốc, với mỗi A_i thêm benign_stub_i và nối
        A_i → benign_stub_i → successor(A_i), sau đó chạy GNN.
    """
    G_sim = G.copy()
    inserted = []
    skipped = []

    for idx, pair in enumerate(node_sequence_pairs):
        node = pair["node"]
        seq = pair["sequence"]
        if node not in G_sim:
            skipped.append({"rank": pair.get("rank"), "reason": "node_not_in_graph", "node": str(node)})
            continue

        fv_raw = seq.get("gnn_stub_feature_vector")
        if fv_raw is None:
            fv_raw = seq.get("feature_vector")
        if fv_raw is None or len(fv_raw) != NODE_FEATURE_DIM:
            skipped.append({
                "rank": pair.get("rank"),
                "reason": f"invalid_feature_dim_{len(fv_raw) if fv_raw is not None else None}",
                "node": str(node),
            })
            continue

        seq_hex = seq.get("sequence_prime", {}).get("bytes_hex", "")
        try:
            seq_bytes = bytes.fromhex(seq_hex)
        except Exception:
            seq_bytes = b""

        succs = list(G_sim.successors(node))
        stub_id = f"benign_stub_budget_{idx}_{seq.get('block_addr', 'x')}"
        while stub_id in G_sim:
            stub_id += "_x"

        G_sim.add_node(stub_id)
        G_sim.nodes[stub_id]["node_type"] = "benign_stub"
        G_sim.nodes[stub_id]["feature_vector"] = fv_raw
        G_sim.nodes[stub_id]["structural_score"] = 0.0
        G_sim.nodes[stub_id]["block_size"] = len(seq_bytes)
        G_sim.nodes[stub_id]["instruction_count"] = max(1, seq.get("sequence_clean", {}).get("size_bytes", len(seq_bytes)) // 3)

        G_sim.add_edge(node, stub_id)
        if succs:
            G_sim.add_edge(stub_id, succs[0])

        inserted.append({
            "rank": pair.get("rank"),
            "node": str(node),
            "stub_id": stub_id,
            "successor": str(succs[0]) if succs else None,
            "sequence_block_addr": seq.get("block_addr"),
            "sequence_source": seq.get("source_file"),
        })

    prob, pred, label = infer_graph_prob_malware(
        G_sim,
        model=model,
        build_pyg_data_fn=build_pyg_data_fn,
        device=device,
        scaler=scaler,
    )
    return {
        "G_sim": G_sim,
        "inserted_count": len(inserted),
        "skipped_count": len(skipped),
        "inserted": inserted,
        "skipped": skipped,
        "simulated_prob_malware": prob,
        "simulated_prediction_id": pred,
        "simulated_prediction_label": label,
        "simulated_bypass": prob < 0.5,
    }


def prepare_patch_plan_for_pair(
    binary_path: str,
    project: angr.Project,
    cfg,
    pair: Dict[str, Any],
    arch: str,
    bits: int,
    min_opcode_length: int,
) -> Dict[str, Any]:
    """
    Resolve A_i → basic block, tìm opcode_region an toàn và gắn B_i.
    Chưa ghi file PE ở hàm này.
    """
    node = pair["node"]
    seq = pair["sequence"]

    patch_block = resolve_patch_block(project, cfg, target_node=node, target_edge=None)
    block_bytes = read_block_bytes(binary_path, patch_block)
    instructions = decode_instructions(
        block_bytes,
        base_va=patch_block["patch_block_va"],
        arch=arch,
        bits=bits,
    )
    if not instructions:
        raise RuntimeError("Không decode được instruction nào từ block.")

    opcode_region = find_safe_opcode_region(
        instructions,
        base_va=patch_block["patch_block_va"],
        base_file_offset=patch_block["patch_block_file_offset"],
        min_length=min_opcode_length,
    )
    if opcode_region is None:
        raise RuntimeError("Không tìm được opcode_region an toàn.")

    sequence_prime = wrap_sequence_prime(seq)
    tramp_size = len(sequence_prime) + len(opcode_region["stolen_bytes"]) + JMP_SIZE

    return {
        "rank": pair.get("rank"),
        "node": node,
        "node_str": str(node),
        "sequence": seq,
        "sequence_prime": sequence_prime,
        "sequence_size_bytes": len(sequence_prime),
        "patch_block": patch_block,
        "opcode_region": opcode_region,
        "trampoline_size": tramp_size,
        "status": "planned",
    }


def add_multi_trampoline_section(
    pe_path: str,
    patch_plans: List[Dict[str, Any]],
    out_path: str,
    section_name: str = TRAMP_SECTION_NAME,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Thêm một section .adv duy nhất chứa nhiều trampoline.

    Hai lượt để có VA thật của section:
        1. Tính tổng kích thước placeholder và add section để biết base VA.
        2. Build từng trampoline với addr_b = base VA + offset, rồi ghi lại PE.
    """
    if not patch_plans:
        shutil.copy2(pe_path, out_path)
        return patch_plans, 0

    total_len = sum(int(plan["trampoline_size"]) for plan in patch_plans)
    placeholder = bytes(total_len)

    tmp_probe = out_path + ".probe"
    base_va = add_trampoline_section(
        pe_path=pe_path,
        trampoline_bytes=placeholder,
        out_path=tmp_probe,
        section_name=section_name,
    )

    offset = 0
    final_blob = b""
    for plan in patch_plans:
        addr_b = base_va + offset
        trampoline_bytes = build_trampoline(
            sequence_prime_b=plan["sequence_prime"],
            stolen_bytes=plan["opcode_region"]["stolen_bytes"],
            addr_b=addr_b,
            return_va=plan["opcode_region"]["return_va"],
            opcode_region=plan["opcode_region"],
        )
        if len(trampoline_bytes) != plan["trampoline_size"]:
            raise RuntimeError(
                f"Trampoline size lệch ở rank={plan.get('rank')}: "
                f"expected {plan['trampoline_size']}, got {len(trampoline_bytes)}"
            )
        plan["addr_b"] = addr_b
        plan["trampoline_offset"] = offset
        plan["trampoline_bytes"] = trampoline_bytes
        final_blob += trampoline_bytes
        offset += len(trampoline_bytes)

    base_va_check = add_trampoline_section(
        pe_path=pe_path,
        trampoline_bytes=final_blob,
        out_path=out_path,
        section_name=section_name,
    )

    try:
        os.remove(tmp_probe)
    except OSError:
        pass

    if base_va_check != base_va:
        # Ít khi xảy ra nếu tổng length không đổi; nếu xảy ra thì rebuild một lần.
        logger.warning(
            f"[Physical Patch] base VA thay đổi {hex(base_va)} -> {hex(base_va_check)}. Rebuild trampoline."
        )
        base_va = base_va_check
        offset = 0
        final_blob = b""
        for plan in patch_plans:
            addr_b = base_va + offset
            trampoline_bytes = build_trampoline(
                sequence_prime_b=plan["sequence_prime"],
                stolen_bytes=plan["opcode_region"]["stolen_bytes"],
                addr_b=addr_b,
                return_va=plan["opcode_region"]["return_va"],
                opcode_region=plan["opcode_region"],
            )
            plan["addr_b"] = addr_b
            plan["trampoline_offset"] = offset
            plan["trampoline_bytes"] = trampoline_bytes
            final_blob += trampoline_bytes
            offset += len(trampoline_bytes)
        base_va_final = add_trampoline_section(
            pe_path=pe_path,
            trampoline_bytes=final_blob,
            out_path=out_path,
            section_name=section_name,
        )
        if base_va_final != base_va:
            raise RuntimeError("Không ổn định được base VA của multi-trampoline section.")

    return patch_plans, base_va


def physical_patch_budget(
    binary_path: str,
    patched_path: str,
    project: angr.Project,
    cfg,
    node_sequence_pairs: List[Dict[str, Any]],
    arch: str,
    bits: int,
    min_opcode_length: int,
) -> Dict[str, Any]:
    """
    Physical Patching đúng yêu cầu:
        Trên PE gốc, patch đồng thời k vị trí.
    Node nào không resolve/không có opcode_region an toàn thì ghi failed,
    các node còn lại vẫn được patch.
    """
    patch_plans = []
    failed = []

    occupied_ranges = []

    for pair in node_sequence_pairs:
        try:
            plan = prepare_patch_plan_for_pair(
                binary_path=binary_path,
                project=project,
                cfg=cfg,
                pair=pair,
                arch=arch,
                bits=bits,
                min_opcode_length=min_opcode_length,
            )

            start = plan["opcode_region"]["file_offset"]
            end = start + plan["opcode_region"]["length"]
            overlap = any(not (end <= a or start >= b) for a, b in occupied_ranges)
            if overlap:
                raise RuntimeError(
                    f"opcode_region overlap với patch trước đó: offset={hex(start)} len={end-start}"
                )
            occupied_ranges.append((start, end))
            patch_plans.append(plan)
        except Exception as e:
            failed.append({
                "rank": pair.get("rank"),
                "node": str(pair.get("node")),
                "sequence_block_addr": pair.get("sequence", {}).get("block_addr"),
                "error": str(e),
            })
            logger.warning(f"[Physical Patch] Bỏ qua node rank={pair.get('rank')}: {e}")

    if not patch_plans:
        shutil.copy2(binary_path, patched_path)
        return {
            "patched_path": patched_path,
            "patch_success_count": 0,
            "patch_failed_count": len(failed),
            "patch_plans": [],
            "failed": failed,
            "section_base_va": None,
        }

    patch_plans, section_base_va = add_multi_trampoline_section(
        pe_path=binary_path,
        patch_plans=patch_plans,
        out_path=patched_path,
        section_name=TRAMP_SECTION_NAME,
    )

    for plan in patch_plans:
        patch_jmp_e9(
            pe_path=patched_path,
            opcode_region=plan["opcode_region"],
            addr_b=plan["addr_b"],
        )
        plan["status"] = "patched"

    serializable_plans = []
    for plan in patch_plans:
        seq = plan["sequence"]
        serializable_plans.append({
            "rank": plan.get("rank"),
            "node": plan.get("node_str"),
            "sequence_block_addr": seq.get("block_addr"),
            "sequence_source": seq.get("source_file"),
            "sequence_size_bytes": plan.get("sequence_size_bytes"),
            "patch_va": hex(plan["opcode_region"]["start_va"]),
            "patch_file_offset": hex(plan["opcode_region"]["file_offset"]),
            "patch_len": plan["opcode_region"]["length"],
            "addr_b": hex(plan["addr_b"]),
            "return_va": hex(plan["opcode_region"]["return_va"]),
            "status": plan["status"],
        })

    return {
        "patched_path": patched_path,
        "patch_success_count": len(patch_plans),
        "patch_failed_count": len(failed),
        "patch_plans": serializable_plans,
        "failed": failed,
        "section_base_va": hex(section_base_va) if section_base_va else None,
    }


def evaluate_patched_binary_once(
    original_graph_nodes: int,
    original_graph_edges: int,
    patched_path: str,
    model,
    out_dir: str,
    build_pyg_data_fn,
    scaler=None,
    device=None,
) -> Dict[str, Any]:
    """
    Re-extract CFG từ patched.exe và đưa vào GNN đánh giá.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        result_patched = run_cfg_pipeline(
            binary_path=patched_path,
            out_dir=out_dir,
            verbose=False,
        )
        G_patched = result_patched["pruned_graph"]
        prob, pred, label = infer_graph_prob_malware(
            G_patched,
            model=model,
            build_pyg_data_fn=build_pyg_data_fn,
            device=device,
            scaler=scaler,
        )
        return {
            "cfg_build_ok": True,
            "real_prob_malware": prob,
            "real_prediction_id": pred,
            "real_prediction_label": label,
            "real_bypass": prob < 0.5,
            "patched_nodes": G_patched.number_of_nodes(),
            "patched_edges": G_patched.number_of_edges(),
            "node_delta": G_patched.number_of_nodes() - original_graph_nodes,
            "edge_delta": G_patched.number_of_edges() - original_graph_edges,
        }
    except Exception as e:
        logger.warning(f"[Real Eval] Không đánh giá được patched binary: {e}")
        return {
            "cfg_build_ok": False,
            "real_prob_malware": None,
            "real_prediction_id": None,
            "real_prediction_label": None,
            "real_bypass": False,
            "patched_nodes": None,
            "patched_edges": None,
            "node_delta": None,
            "edge_delta": None,
            "error": str(e),
        }


def write_experiment_table(rows: List[Dict[str, Any]], csv_path: str) -> None:
    """Ghi bảng dữ liệu thực nghiệm theo từng budget."""
    if not rows:
        return
    fieldnames = [
        "budget_ratio",
        "budget_percent",
        "k_nodes_requested",
        "matched_nodes",
        "patch_success_count",
        "patch_failed_count",
        "original_prob_malware",
        "simulated_prob_malware",
        "simulated_bypass",
        "real_prob_malware",
        "real_bypass",
        "orig_nodes",
        "orig_edges",
        "patched_nodes",
        "patched_edges",
        "node_delta",
        "edge_delta",
        "patched_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


# ============================================================
# Pipeline gộp — hàm chính
# ============================================================

def run_inject_pipeline(
    binary_path: str,
    out_dir: str,
    cfg_result: Dict,
    # target_node=None,
    target_nodes: Optional[List[Any]] = None,
    target_edge: Optional[Tuple] = None,
    benign_pool: Optional[List[Dict]] = None,
    benign_dir: Optional[str] = None,
    pool_cache_path: str = "./output/benign_pool.json",
    malware_feature_vec: Optional[List[float]] = None,
    arch: str = "x86",
    bits: int = 32,
    min_opcode_length: int = JMP_SIZE,
    budget_levels: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10),
    model=None,
    build_pyg_data_fn=None,
    device=None,
    verbose: bool = True,
    scaler=None,
) -> Dict[str, Any]:
    """
    Multi-budget / multi-node pipeline đúng quy trình chuẩn:

    Với mỗi budget ∈ {1%, 3%, 5%, 10%}:
        1. Selection: chọn top-k internal nodes theo budget,
           k = ceil(budget * số internal nodes), dựa trên target_nodes/ranking
           từ file_to_cfg_main.
        2. Matching: với mỗi A_i chọn B_i từ benign pool theo
           dissimilarity, frequency, risk, size.
        3. CFG Simulation: copy CFG gốc, thêm k benign_stub_i, chạy GNN.
        4. Physical Patching: patch đồng thời k vị trí trên PE gốc.
        5. Re-extract CFG: build lại CFG từ patched.exe.
        6. Real GNN Evaluation: đưa CFG patched vào GNN.
        7. Ghi bảng thực nghiệm theo từng budget.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    os.makedirs(out_dir, exist_ok=True)

    project = cfg_result.get("project")
    cfg = cfg_result.get("cfg")
    if project is None or cfg is None:
        raise ValueError(
            "cfg_result thiếu project/cfg. Khi chuẩn bị cfg_result cho inject, "
            "hãy gọi run_cfg_pipeline(..., release_after=False)."
        )

    if model is None or build_pyg_data_fn is None:
        raise RuntimeError(
            "Pipeline multi-budget yêu cầu model và build_pyg_data_fn để simulation/evaluation."
        )


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_orig = cfg_result["pruned_graph"]
    node_types = cfg_result["node_types"]
    centrality_dict = cfg_result["centrality"]
    binary_size = os.path.getsize(binary_path)

    if benign_pool is None:
        if benign_dir is None:
            raise ValueError("Cần cung cấp benign_pool hoặc benign_dir.")
        benign_pool = build_benign_pool(
            benign_dir=benign_dir,
            out_pool_path=pool_cache_path,
            arch=arch,
            bits=bits,
            malware_feature_vec=malware_feature_vec,
        )
    if not benign_pool:
        raise RuntimeError("[Step 6] Benign pool rỗng.")

    # Nếu code cũ chỉ truyền target_node, vẫn chạy được nhưng chỉ ưu tiên node đó trước.
    if target_nodes is None:
        target_nodes = []
    # if target_node is not None and target_node not in target_nodes:
    #     target_nodes = [target_node] + list(target_nodes)

    # xếp hạng internal nodes dựa trên target_nodes đã có, không tự sort lại nếu đã có ranking từ file_to_cfg_main.
    ranked_internal_nodes = get_ranked_internal_nodes_from_cfg(
        G=G_orig,
        node_types=node_types,
        centrality_dict=centrality_dict,
        target_nodes=target_nodes,
    )
    total_internal_nodes = sum(1 for n in G_orig.nodes() if node_types.get(n) == "internal")
    if total_internal_nodes <= 0 or not ranked_internal_nodes:
        raise RuntimeError("Không có internal nodes để chọn theo budget.")

    # Đánh giá baseline trên graph gốc trước khi patch.
    original_prob_malware, original_pred, original_label = infer_graph_prob_malware(
        G_orig,
        model=model,
        build_pyg_data_fn=build_pyg_data_fn,
        device=device,
        scaler=scaler,
    )

    summary: Dict[str, Any] = {
        "binary_path": binary_path,
        "out_dir": out_dir,
        "pool_cache_path": pool_cache_path,
        "pool_size": len(benign_pool),
        "total_internal_nodes": total_internal_nodes,
        "ranked_internal_nodes_count": len(ranked_internal_nodes),
        "original": {
            "prob_malware": original_prob_malware,
            "prediction_id": original_pred,
            "prediction_label": original_label,
            "graph_nodes": G_orig.number_of_nodes(),
            "graph_edges": G_orig.number_of_edges(),
        },
        "budget_results": [],
        "experiment_rows": [],
    }

    logger.info(
        f"[Pipeline] original_prob_malware={original_prob_malware:.4f}, "
        f"internal_nodes={total_internal_nodes}, pool={len(benign_pool)}"
    )

    # Duyệt từng budget, thực hiện selection → matching → simulation → physical patch → real evaluation, ghi kết quả.
    for budget in budget_levels:
        budget_percent = int(round(budget * 100))
        budget_dir = os.path.join(out_dir, f"budget_{budget_percent:02d}pct")
        os.makedirs(budget_dir, exist_ok=True)
        patched_path = os.path.join(budget_dir, f"patched_budget_{budget_percent:02d}pct.exe")

        # chọn số nodes theo từng budget, dựa trên ranking đã có.
        selected_nodes = select_topk_nodes_for_budget(
            ranked_internal_nodes=ranked_internal_nodes,
            total_internal_nodes=total_internal_nodes,
            budget_ratio=budget,
        )


        logger.info(
            f"[Budget {budget_percent}%] Selection: k={len(selected_nodes)}, "
        )


        # Tìm sequence phù hợp cho từng node đã chọn, với filtering và scoring theo spec.
        total_budget_bytes = max(1, int(binary_size * budget))
        pairs = match_sequences_to_nodes(
            selected_nodes=selected_nodes,
            pool=benign_pool,
            malware_feature_vec=malware_feature_vec,
            binary_size_bytes=binary_size,
            total_budget_bytes=total_budget_bytes,
            G=G_orig,
            tau_risk=0.5,
            context_sim_min=CONTEXT_SIM_MIN,
            context_sim_max=CONTEXT_SIM_MAX,
            per_node_max_ratio=0.03,
        )

        # Simulate CFG sau khi thêm benign_stub theo từng cặp node-sequence đã matched, đánh giá với GNN để có prob_malware giả định sau patch.
        sim_result = simulate_multi_node_cfg(
            G=G_orig,
            node_sequence_pairs=pairs,
            model=model,
            build_pyg_data_fn=build_pyg_data_fn,
            device=device,
            scaler=scaler,
        )

        # Thực hiện patch đồng thời k vị trí trên PE gốc, ghi file patched.exe.
        patch_result = physical_patch_budget(
            binary_path=binary_path,
            patched_path=patched_path,
            project=project,
            cfg=cfg,
            node_sequence_pairs=pairs,
            arch=arch,
            bits=bits,
            min_opcode_length=min_opcode_length,
        )

        # Re-extract CFG từ patched.exe và đánh giá thực tế với GNN.
        real_eval = evaluate_patched_binary_once(
            original_graph_nodes=G_orig.number_of_nodes(),
            original_graph_edges=G_orig.number_of_edges(),
            patched_path=patched_path,
            model=model,
            out_dir=os.path.join(budget_dir, "cfg_patched"),
            build_pyg_data_fn=build_pyg_data_fn,
            scaler=scaler,
            device=device,
        )

        row = {
            "budget_ratio": budget,
            "budget_percent": budget_percent,
            "k_nodes_requested": len(selected_nodes),
            "matched_nodes": len(pairs),
            "patch_success_count": patch_result["patch_success_count"],
            "patch_failed_count": patch_result["patch_failed_count"],
            "original_prob_malware": original_prob_malware,
            "simulated_prob_malware": sim_result["simulated_prob_malware"],
            "simulated_bypass": sim_result["simulated_bypass"],
            "real_prob_malware": real_eval["real_prob_malware"],
            "real_bypass": real_eval["real_bypass"],
            "orig_nodes": G_orig.number_of_nodes(),
            "orig_edges": G_orig.number_of_edges(),
            "patched_nodes": real_eval.get("patched_nodes"),
            "patched_edges": real_eval.get("patched_edges"),
            "node_delta": real_eval.get("node_delta"),
            "edge_delta": real_eval.get("edge_delta"),
            "patched_path": patched_path,
        }

        budget_result = {
            "budget_ratio": budget,
            "budget_percent": budget_percent,
            "k_nodes_requested": len(selected_nodes),
            "selected_nodes": [str(n) for n in selected_nodes],
            "matching": [
                {
                    "rank": p.get("rank"),
                    "node": str(p.get("node")),
                    "sequence_block_addr": p.get("sequence", {}).get("block_addr"),
                    "sequence_source": p.get("sequence", {}).get("source_file"),
                    "sequence_score": p.get("sequence_score"),
                    "sequence_size_bytes": p.get("sequence_size_bytes"),
                }
                for p in pairs
            ],
            "simulation": {k: v for k, v in sim_result.items() if k != "G_sim"},
            "physical_patch": patch_result,
            "real_evaluation": real_eval,
            "experiment_row": row,
        }

        summary["budget_results"].append(budget_result)
        summary["experiment_rows"].append(row)

        logger.info(
            f"[Budget {budget_percent}%] sim_prob={row['simulated_prob_malware']:.4f}, "
            f"real_prob={row['real_prob_malware']}, "
            f"patch={row['patch_success_count']}/{row['matched_nodes']}"
        )

    summary_path = os.path.join(out_dir, "inject_summary.json")
    csv_path = os.path.join(out_dir, "experiment_table.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        def _safe(obj):
            if isinstance(obj, bytes):
                return obj.hex()
            return str(obj)
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_safe)


    # Lưu lại kết quả
    write_experiment_table(summary["experiment_rows"], csv_path)
    summary["summary_path"] = summary_path
    summary["experiment_table_csv"] = csv_path

    logger.info(f"[Pipeline] Summary → {summary_path}")
    logger.info(f"[Pipeline] Experiment table → {csv_path}")
    return summary


# ============================================================
# Ví dụ sử dụng (chạy trực tiếp để test)
# ============================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    binary_path = "./files/test/malware/malware_inject_test.exe"
    out_dir = "./output/output_inject"
    benign_dir   = "./files/benign"          # hoặc truyền benign_pool trực tiếp
    pool_cache   = "./output/benign_pool.json"      # từ bước 2
    model_path   = "./models/gcn_model.pt"  # từ bước 1
    scaler_path  = "./models/scaler.pkl"
    malvec_out_dir = "./output/malware_feature_vector"             # nơi lưu malware_feature_vec.npy/.json



    # Phase 1+2: chạy file_to_cfg_main để lấy danh sách target_nodes đã ranking
    from ipynb.fs.full.a2_file_to_cfg_main import predict_and_explain_cfg, IMPORTANT_NODE_RATIO
    explain_result = predict_and_explain_cfg(
        binary_path=binary_path,
        model_path=model_path,
        scaler_path=scaler_path,
        out_dir="./output/output_cfg_adversarial_files",
        release_after=False, # ko giai phong bo nho, dung de test
        important_node_ratio=IMPORTANT_NODE_RATIO,
    )

    cfg_result = explain_result["cfg_result"]

    # lấy target_nodes từ kết quả của file_to_cfg_main
    target_nodes = explain_result.get("target_nodes", [])

    if not target_nodes:
        print("Không có target_nodes từ predict_and_explain_cfg. Thoát.")
        sys.exit(1)

    # target_node = target_nodes[0]

    print(f"Số target node top {IMPORTANT_NODE_RATIO*100:.0f}%: {len(target_nodes)}")
    print("Target nodes:", target_nodes)
    # print("Target node được chọn:", target_node)

    # Phase 3: chạy inject pipeline
    # Cần truyền model đã load weights để budget loop chạy CFG simulation.
    model  = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # Export malware feature vector trực tiếp từ file a4_train_gnn_models.py
    # Input là chính malware đang attack, không cần đọc sẵn malware_feature_vec.npy cũ.
    malvec_export = export_malware_feature_vector(
        malware_adversarial_path=binary_path,
        out_dir=malvec_out_dir,
        scaler=scaler,
        arch="x86",
        bits=32,
        quiet=False,
        out_name="malware_feature_vec",
    )
    malvec_path = malvec_export["npy_paths"][-1]
    malvec = np.load(malvec_path).tolist()
    print(f"Malware feature vector: {malvec_path}")

    summary = run_inject_pipeline(
        binary_path=binary_path,
        out_dir=out_dir,
        cfg_result=cfg_result,
        # target_node=target_node,          # backward compatibility
        target_nodes=target_nodes,        # multi-budget selection dùng danh sách này
        benign_dir=benign_dir,
        pool_cache_path=pool_cache,
        malware_feature_vec=malvec,
        arch="x86",
        bits=32,
        min_opcode_length=5,
        budget_levels=(0.01, 0.03, 0.05, 0.10),
        model=model,
        build_pyg_data_fn=build_pyg_data,
        scaler=scaler,
        verbose=True,
    )

    print("\n=== Inject Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
