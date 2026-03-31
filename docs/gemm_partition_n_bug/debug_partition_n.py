#!/usr/bin/env python3
"""Reproduce partition_N=4 GEMM bug: C_0 correct, C_1-C_3 wrong.

Usage:
    cd /home/jiajli/apps/IRON
    source ironenv/bin/activate && source /opt/xilinx/xrt/setup.sh
    python debug_partition_n.py
"""

import torch
import numpy as np
from pathlib import Path
from ml_dtypes import bfloat16

from iron.operators.gemm.op import AIEGEMM
from iron.operators.gemm.reference import generate_golden_reference
from iron.common import AIEContext
from iron.common.utils import torch_to_numpy

# ---------- Configuration (matches Llama model_with_json.py) ----------
M, K, N = 2048, 2048, 128256
PARTITION_N = 4
N_PER_PART = N // PARTITION_N  # 32064

# ---------- Setup ----------
ctx = AIEContext()
ctx.build_dir = Path("build_llama").resolve()

ref = generate_golden_reference(M=M, K=K, N=N, b_col_maj=True, partition_N=PARTITION_N)

op = AIEGEMM(
    M=M, K=K, N=N,
    tile_m=64, tile_k=64, tile_n=64,
    num_aie_columns=8,
    prio_accuracy=False,
    emulate_bf16_mmul_with_bfp16=True,
    b_col_maj=True,
    use_static_weight=True,
    partition_N=PARTITION_N,
    context=ctx,
)

# Load weights (same as model: op.weight = out_head.T)
full_B = torch.cat(ref["input_b"], dim=0)  # (N, K) in b_col_maj format
op.weight = full_B.T  # weight setter expects (K, N)

print(f"Problem size: M={M}, K={K}, N={N}, partition_N={PARTITION_N}")
print(f"N per partition (actual): {N_PER_PART}")
print(f"N per partition (padded): {op.N}")
print(f"Padding per row: {op.N - N_PER_PART}")
print()

# ---------- Compile & Run ----------
ctx.compile_all()
ctx.prepare_runtime()
op.write_buffer("A", torch_to_numpy(ref["input"]))
op.run_runlist()

# ---------- Analysis ----------
print("=" * 80)
print("FLAT comparison (what verify_buffer does — WRONG due to row misalignment)")
print("=" * 80)
for i in range(PARTITION_N):
    buf_size = op.buffers[f"C_{i}"] // 2
    out_flat = np.array(op.read_buffer(f"C_{i}", (buf_size,)), dtype=np.float32)
    ref_flat = torch_to_numpy(ref["output"][i]).reshape(-1).astype(np.float32)
    n = min(len(out_flat), len(ref_flat))
    corr = float(np.corrcoef(out_flat[:n], ref_flat[:n])[0, 1])
    print(f"  C_{i}: corr = {corr:.5f}   ← misleading (rows shift by {op.N - N_PER_PART} each)")
print()

print("=" * 80)
print("2D comparison (correct — reads as (M, N_padded), slices to (M, N_per_part))")
print("=" * 80)
for i in range(PARTITION_N):
    out_2d = np.array(op.read_buffer(f"C_{i}", (op.M, op.N)), dtype=np.float32)
    out_valid = out_2d[:M, :N_PER_PART].reshape(-1)
    ref_valid = torch_to_numpy(ref["output"][i]).reshape(-1).astype(np.float32)
    corr = float(np.corrcoef(out_valid, ref_valid)[0, 1])
    max_err = float(np.max(np.abs(out_valid - ref_valid)))
    mean_err = float(np.mean(np.abs(out_valid - ref_valid)))
    print(f"  C_{i}: corr = {corr:.5f}, max_err = {max_err:.1f}, mean_err = {mean_err:.2f}")
print()

print("=" * 80)
print("Per-row analysis for C_0 vs C_1 (first 5 rows)")
print("=" * 80)
for i in [0, 1]:
    out_2d = np.array(op.read_buffer(f"C_{i}", (op.M, op.N)), dtype=np.float32)
    ref_2d = torch_to_numpy(ref["output"][i]).reshape(M, N_PER_PART).astype(np.float32)
    for row in range(5):
        row_out = out_2d[row, :N_PER_PART]
        row_ref = ref_2d[row, :]
        corr = float(np.corrcoef(row_out, row_ref)[0, 1])
        max_err = float(np.max(np.abs(row_out - row_ref)))
        print(f"  C_{i} row {row}: corr = {corr:.5f}, max_err = {max_err:.1f}")
    print()

print("=" * 80)
print("Cross-partition check: is C_1 output actually C_0's data repeated?")
print("=" * 80)
c0_2d = np.array(op.read_buffer("C_0", (op.M, op.N)), dtype=np.float32)[:M, :N_PER_PART]
c1_2d = np.array(op.read_buffer("C_1", (op.M, op.N)), dtype=np.float32)[:M, :N_PER_PART]
c0_ref = torch_to_numpy(ref["output"][0]).reshape(M, N_PER_PART).astype(np.float32)
c1_ref = torch_to_numpy(ref["output"][1]).reshape(M, N_PER_PART).astype(np.float32)
# Is C_1 output correlated with C_0's reference (i.e., did partition 1 compute partition 0's answer)?
corr_c1_vs_c0ref = float(np.corrcoef(c1_2d.reshape(-1), c0_ref.reshape(-1))[0, 1])
# Is C_1 output correlated with C_0's actual output?
corr_c1_vs_c0out = float(np.corrcoef(c1_2d.reshape(-1), c0_2d.reshape(-1))[0, 1])
print(f"  C_1 output vs C_0 reference:  corr = {corr_c1_vs_c0ref:.5f}")
print(f"  C_1 output vs C_0 output:     corr = {corr_c1_vs_c0out:.5f}")
print(f"  C_1 output vs C_1 reference:  corr = {float(np.corrcoef(c1_2d.reshape(-1), c1_ref.reshape(-1))[0, 1]):.5f}")
print()

print("=" * 80)
print("Buffer address check: are B partitions actually different data?")
print("=" * 80)
for i in range(PARTITION_N):
    b_size = op.buffers[f"B_{i}"] // 2
    b_data = np.array(op.read_buffer(f"B_{i}", (b_size,)), dtype=np.float32)
    b_ref = torch_to_numpy(ref["input_b"][i]).reshape(-1).astype(np.float32)
    n = min(len(b_data), len(b_ref))
    corr = float(np.corrcoef(b_data[:n], b_ref[:n])[0, 1])
    print(f"  B_{i} vs reference: corr = {corr:.5f}, first 5 values: {b_data[:5]}")
