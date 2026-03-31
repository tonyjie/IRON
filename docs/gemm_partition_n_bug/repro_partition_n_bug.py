#!/usr/bin/env python3
"""Reproduce partition_N=4 GEMM bugs in IRON.

This script demonstrates two bugs in the GEMM operator when using partition_N > 1,
which is used by the Llama 3.2 1B final vocab projection (model_with_json.py line 186).

Usage:
    cd /path/to/IRON
    source ironenv/bin/activate && source /opt/xilinx/xrt/setup.sh
    python repro_partition_n_bug.py

Requirements: IRON with mlir_aie and XRT installed. No test framework changes needed.
"""

import torch
import numpy as np
from pathlib import Path
from ml_dtypes import bfloat16

from iron.operators.gemm.op import AIEGEMM
from iron.operators.gemm.reference import generate_golden_reference
from iron.common import AIEContext
from iron.common.utils import torch_to_numpy


def check_partition(output_2d, ref_2d, label):
    """Compare NPU output vs CPU reference (both as float32 2D arrays)."""
    out = output_2d.reshape(-1)
    ref = ref_2d.reshape(-1)
    n = min(len(out), len(ref))
    corr = float(np.corrcoef(out[:n], ref[:n])[0, 1])
    max_err = float(np.max(np.abs(out[:n] - ref[:n])))
    mean_err = float(np.mean(np.abs(out[:n] - ref[:n])))
    status = "PASS" if corr > 0.99 else "FAIL"
    print(f"  {label}: corr={corr:.5f}, max_err={max_err:.1f}, mean_err={mean_err:.2f}  [{status}]")
    return corr


# ---------- Configuration (matches Llama model_with_json.py lines 178-196) ----------
M, K, N = 2048, 2048, 128256
PARTITION_N = 4
N_PER_PART = N // PARTITION_N  # 32064
BUILD_DIR = Path("build_repro").resolve()

print("=" * 70)
print("IRON GEMM partition_N Bug Reproduction")
print("=" * 70)
print(f"Problem: M={M}, K={K}, N={N}, partition_N={PARTITION_N}")
print(f"Matches: Llama 3.2 1B final vocab projection (model_with_json.py)")
print(f"Build dir: {BUILD_DIR}")
print()

ref = generate_golden_reference(M=M, K=K, N=N, b_col_maj=True, partition_N=PARTITION_N)

# ======================== BUG 1: forward() returns wrong shape ========================
print("=" * 70)
print("BUG 1: forward() returns wrong output shape with partition_N > 1")
print("=" * 70)

ctx1 = AIEContext()
ctx1.build_dir = BUILD_DIR

op1 = AIEGEMM(
    M=M, K=K, N=N,
    tile_m=64, tile_k=64, tile_n=64,
    num_aie_columns=8,
    prio_accuracy=False, emulate_bf16_mmul_with_bfp16=True,
    b_col_maj=True, use_static_weight=True, partition_N=PARTITION_N,
    context=ctx1,
)

full_B = torch.cat(ref["input_b"], dim=0)  # (N, K) in b_col_maj format
op1.weight = full_B.T  # Model does: op.weight = out_head.T

ctx1.compile_all()
ctx1.prepare_runtime()

A_input = torch.randn(1, M, K, dtype=torch.bfloat16) * 4
result = op1.forward(A_input)

print(f"  Expected output shape: (1, {M}, {N})")
print(f"  Actual output shape:   {tuple(result.shape)}")
print()
if result.shape[-1] != N:
    print(f"  BUG CONFIRMED: forward() returns {result.shape[-1]} columns instead of {N}.")
    print(f"  Root cause: _partition_B() (op.py) overwrites self.static_weight_shape")
    print(f"  to single-partition size ({op1.N}, {K}), then forward() divides")
    print(f"  by partition_N again, yielding N_part = {op1.N // PARTITION_N}.")
    print()
    print(f"  The Llama model calls out_head_prefill(x) which hits this path.")
    print(f"  Logits shape is (batch, seq_len, {result.shape[-1]}) instead of")
    print(f"  (batch, seq_len, {N}), silently truncating the vocabulary.")
else:
    print("  Shape is correct. If running on unpatched code, expect shape")
    print(f"  (1, {M}, {op1.N}) instead — see bug description.")
print()

# ======================== BUG 2: Only partition 0 produces correct results ========================
print("=" * 70)
print("BUG 2: Only C_0 is correct when partition_N > 1 in single context")
print("=" * 70)
print()
print("Reading individual partition buffers directly (bypassing forward())...")

ctx2 = AIEContext()
ctx2.build_dir = BUILD_DIR

op2 = AIEGEMM(
    M=M, K=K, N=N,
    tile_m=64, tile_k=64, tile_n=64,
    num_aie_columns=8,
    prio_accuracy=False, emulate_bf16_mmul_with_bfp16=True,
    b_col_maj=True, use_static_weight=True, partition_N=PARTITION_N,
    context=ctx2,
)
full_B = torch.cat(ref["input_b"], dim=0)
op2.weight = full_B.T

ctx2.compile_all()
ctx2.prepare_runtime()
op2.write_buffer("A", torch_to_numpy(ref["input"]))
op2.run_runlist()

print(f"  N_per_partition={N_PER_PART}, N_padded={op2.N}, padding={op2.N - N_PER_PART}")
print()

# Read each C_i with correct 2D shape (accounting for N padding)
for i in range(PARTITION_N):
    out_2d = np.array(op2.read_buffer(f"C_{i}", (op2.M, op2.N)), dtype=np.float32)
    out_valid = out_2d[:M, :N_PER_PART]
    ref_valid = torch_to_numpy(ref["output"][i]).reshape(M, N_PER_PART).astype(np.float32)
    check_partition(out_valid, ref_valid, f"C_{i} (vocab {i*N_PER_PART}-{(i+1)*N_PER_PART-1})")
print()

# ======================== CONTROL: Standalone partitions all work ========================
print("=" * 70)
print("CONTROL: Each partition works correctly as standalone GEMM (partition_N=1)")
print("=" * 70)
print()

for i in range(PARTITION_N):
    ctx_i = AIEContext()
    ctx_i.build_dir = BUILD_DIR

    op_i = AIEGEMM(
        M=M, K=K, N=N_PER_PART,
        tile_m=64, tile_k=64, tile_n=64,
        num_aie_columns=8,
        prio_accuracy=False, emulate_bf16_mmul_with_bfp16=True,
        b_col_maj=True, use_static_weight=True, partition_N=1,
        context=ctx_i,
    )
    op_i.weight = ref["input_b"][i].T  # Single partition weight

    ctx_i.compile_all()
    ctx_i.prepare_runtime()
    op_i.write_buffer("A", torch_to_numpy(ref["input"]))
    op_i.run_runlist()

    out_2d = np.array(op_i.read_buffer("C_0", (op_i.M, op_i.N)), dtype=np.float32)
    out_valid = out_2d[:M, :N_PER_PART]
    ref_valid = torch_to_numpy(ref["output"][i]).reshape(M, N_PER_PART).astype(np.float32)
    check_partition(out_valid, ref_valid, f"Standalone partition {i}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Bug 1 (forward() shape): _partition_B() overwrites self.static_weight_shape
  to single-partition size. forward() reads N from this corrupted shape and
  divides by partition_N again, returning (M, N_padded_per_part) instead of
  (M, N_full). The Llama model's final vocab GEMM silently operates on a
  truncated vocabulary.

Bug 2 (partition correctness): When partition_N > 1, all 4 runlist entries
  share the same XRT kernel handle and instruction binary (insts.bin). The
  NPU's DMA descriptors bind to buffer addresses from the first invocation
  and are not re-resolved for subsequent entries. Only C_0 (first partition)
  produces correct results; C_1-C_3 read wrong buffer data.

  Each partition works perfectly when run as a standalone GEMM operator with
  its own AIEContext (separate XRT kernel handle + instruction binary).

Impact: The Llama 3.2 1B model's final vocab projection (128256 outputs,
  partition_N=4) produces correct logits only for vocab indices 0-32063.
  The model generates coherent text because common tokens have low indices
  and argmax is noise-tolerant, but output quality is degraded.

No existing test covers partition_N > 1. The Llama app test (test.py) only
  checks returncode == 0 with no output correctness validation.
""")
