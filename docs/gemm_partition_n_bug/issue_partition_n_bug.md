# GEMM `partition_N > 1` produces incorrect results (affects Llama final vocab projection)

## Summary

The GEMM operator produces incorrect results when `partition_N > 1`. Only the first partition (`C_0`) computes correctly; partitions `C_1` through `C_{N-1}` produce wrong output. This directly affects the Llama 3.2 1B model, which uses `partition_N=4` for its final vocab projection (128256 outputs).

There are two separate bugs:
1. **`forward()` returns wrong output shape** when `partition_N > 1` with static weights
2. **Runlist entries 1+ produce wrong results** when multiple entries share the same XRT kernel handle

## Affected Code

- `iron/operators/gemm/op.py` — `_partition_B()`, `forward()`, `_execute_aie_operation()`
- `iron/applications/llama_3.2_1b/src/model_with_json.py` line 186 — `partition_N=4`

## Reproduction

```bash
cd /path/to/IRON
source ironenv/bin/activate && source /opt/xilinx/xrt/setup.sh
python repro_partition_n_bug.py
```

<details>
<summary>repro_partition_n_bug.py (click to expand)</summary>

```python
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
    print(f"  (1, {M}, {op1.N}) instead -- see bug description.")
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
```

</details>

## Expected Output

```
BUG 1: forward() returns wrong output shape with partition_N > 1
  Expected output shape: (1, 2048, 128256)
  Actual output shape:   (1, 2048, 32256)     <-- should be 128256

BUG 2: Only C_0 is correct when partition_N > 1 in single context
  C_0 (vocab 0-32063):       corr=0.99994  [PASS]
  C_1 (vocab 32064-64127):   corr=0.74833  [FAIL]
  C_2 (vocab 64128-96191):   corr=0.74844  [FAIL]
  C_3 (vocab 96192-128255):  corr=0.74172  [FAIL]

CONTROL: Each partition works correctly as standalone GEMM (partition_N=1)
  Standalone partition 0:    corr=0.99994  [PASS]
  Standalone partition 1:    corr=0.99994  [PASS]
  Standalone partition 2:    corr=0.99994  [PASS]
  Standalone partition 3:    corr=0.99994  [PASS]
```

## Bug 1: `forward()` returns wrong output shape

### Root Cause

`_partition_B()` ([op.py line 383](https://github.com/nod-ai/IRON/blob/devel/iron/operators/gemm/op.py#L383)) overwrites `self.static_weight_shape` to the single-partition size:

```python
def _partition_B(self, B):
    ...
    self.static_weight_shape = B_parts[0].shape  # <-- overwrites to (32256, 2048)
```

Later, `forward()` reads N from this corrupted shape:

```python
def forward(self, A, B=None):
    B_shape = B.shape if B is not None else self.static_weight_shape  # (32256, 2048)
    K2, N = self._get_B_dims(B_shape)  # N = 32256 (should be 128256)
    N_part = N // self.partition_N     # 32256 / 4 = 8064 (should be 32064)
```

The output shape becomes `(M, 32256)` instead of `(M, 128256)`.

### Impact on Llama

The model calls `self.out_head_prefill(x)` which returns logits of shape `(batch, seq_len, 32256)` instead of `(batch, seq_len, 128256)`. The model then does `argmax(logits[:, -1, :])` over only 32256 values -- a scrambled mix of 4 partition results reassembled into the wrong column positions.

### Fix

Three changes in `op.py`:
1. Initialize `static_weight_shape` with full dimensions in the correct layout
2. Remove the `self.static_weight_shape = B_parts[0].shape` overwrite from `_partition_B()`
3. Fix the applicability check `N <= self.N` to `N <= self.N * self.partition_N`, and fix `_execute_aie_operation()` to use `self.K, self.N` directly for static weights

## Bug 2: Only first partition produces correct results

### Root Cause

When `partition_N=4`, `set_up_runtime()` creates 4 runlist entries sharing the same XRT kernel handle and instruction binary (`insts.bin`):

```python
for i in range(partition_N):
    self.add_to_runlist("gemm", "A", f"B_{i}", f"C_{i}")
```

All 4 entries use the same `xrt_kernel` object and `insts_bo`. The NPU's instruction sequence contains DMA descriptors that bind to buffer addresses on first execution. When the kernel is re-invoked with different B/C buffer objects, the NPU does not re-resolve the DMA addresses -- it reuses the cached descriptors from the first invocation.

### Evidence

- Partition 0: corr=0.99994 (correct)
- Partitions 1-3: corr=0.74 (wrong -- not random, not C_0's result, but corrupted data)
- Each partition as a **standalone GEMM** with its own `AIEContext`: all 4 produce corr=0.99994

### What was tried (none fixed Bug 2)

| Approach | Result |
|---|---|
| `AIEContext(use_runlist=False)` (sequential kernel calls) | Same -- partitions 1-3 wrong |
| Fresh `get_kernel_handle()` per partition | Same -- partitions 1-3 wrong |
| Per-partition runlist swap (`self.runlist = [runlist[i]]`) | Same -- partitions 1-3 wrong |
| Separate A buffer per partition | Worse -- all 4 partitions wrong (buffer pool aliasing) |
| **Separate `AIEContext` per partition** (partition_N=1 each) | **Works -- all 4 correct** |

### Why it's not caught by existing tests

- No GEMM test uses `partition_N > 1`
- The Llama app test (`iron/applications/llama_3.2_1b/test.py`) only checks `returncode == 0` with no output correctness validation
- The model generates coherent-looking text despite the bug because common English tokens have low vocab indices (within partition 0's range) and `argmax` is noise-tolerant

## Environment

- IRON: devel branch
- mlir-aie: v1.2.1
- XRT: /opt/xilinx/xrt
- Device: Ryzen AI NPU (npu2)
