# GEMV Tiling Scheme

The core operation is a matrix-vector multiply: `C[M] = A[M×K] · B[K]`.

Vector-matrix multiplication (`y = x · A`) is also supported: `op.py` transposes the matrix before dispatch, reducing it to the standard MV case (`A^T x = y`). The MLIR design and C++ kernel always operate in MV mode.

## Parameters

| Parameter | Variable in code | Description |
|-----------|-----------------|-------------|
| `M` | matrix rows (output size) | Number of output elements |
| `K` | matrix cols / vector length | Reduction dimension |
| `num_aie_columns` / `cols` | column parallelism | AIE columns used (1–8) |
| `tile_size_input` / `m_input` | rows per kernel call | Granularity of A streaming |
| `tile_size_output` / `m_output` | rows per output FIFO buffer | Granularity of C writeback |

Constraints (enforced in `design.py`):
- `m_output` must be a multiple of `m_input`
- `m_output <= M / cols` and evenly divides it
- `m_input <= M / cols` and evenly divides it

---

## Llama 3.2 1B Decode Configurations

From `test.py` (`llama_params`), the two main FFN shapes:

| Config | M | K | cols | m_input | m_output | is_mv |
|--------|------|------|------|---------|----------|-------|
| FFN up/gate (`llama_ffn_fc12`) | 8192 | 2048 | 8 | 1 | 512 | False |
| FFN down (`llama_ffn_fc3`) | 2048 | 8192 | 8 | 1 | 128 | False |

Other Llama decode shapes:

| Config | M | K | cols | m_input | m_output | is_mv |
|--------|------|--------|------|---------|----------|-------|
| GQA Q projection | 2048 | 2048 | 8 | 1 | 128 | False |
| GQA K/V projection | 512 | 2048 | 8 | 1 | 32 | False |
| GQA output projection | 2048 | 2048 | 8 | 1 | 128 | False |
| Final vocab projection | 128256 | 2048 | 8 | 4 | 32 | True |

---

## Three-Level Tiling

### Level 1: Column Parallelism (output row partitioning)

The `M` output rows are split evenly across `cols` AIE columns. Each column owns an independent, non-overlapping contiguous chunk of rows:

```
Column 0: rows    0 ..  1023   (M / cols = 8192 / 8 = 1024 rows)
Column 1: rows 1024 ..  2047
...
Column 7: rows 7168 ..  8191
```

This is implemented via `TensorAccessPattern` (`design.py:131-138`). The DDR offset for column `col` is `col * (M / cols) * K` elements into the flat row-major matrix buffer. Each column's shim DMA independently streams its chunk of A to its core.

The vector B is **broadcast** — every column gets the full K-element vector (`design.py:140-141`). Each column has its own `B_L3L1` FIFO with `depth=1`.

### Level 2: Output FIFO Buffering (`m_output`)

Within each column's row chunk, work is divided into `m_output`-sized output tiles:

```
tiles per column = (M / cols) / m_output
```

Each output tile corresponds to one acquire/release cycle on the `C_L1L3` FIFO (`design.py:103, 111`). The FIFO has `depth=2` for double-buffering — the shim can drain one tile back to DDR while the core is filling the next.

### Level 3: Kernel Call Granularity (`m_input`)

Within each `m_output`-sized tile, the kernel is called `m_output / m_input` times (`design.py:105`):

```
kernel calls per tile = m_output / m_input
```

Each call processes `m_input` rows of A, writing results into the appropriate slot of the L1 output buffer via the `row_offset` parameter (`j_i32 * m_input`, `design.py:107`). The A FIFO has `depth=2` for double-buffering at the `m_input`-row granularity.

---

## Full Loop Nest (both Llama FFN cases)

### Case 1: M=8192, K=2048 (FFN up/gate, `llama_ffn_fc12`)

```
[DDR → Shim]  Split A into 8 chunks of 1024 rows → 8 columns in parallel
  [Column, outer]  2 output tiles of 512 rows each       (1024/512 = 2 iterations)
    [Column, inner]  512 kernel calls of 1 row each       (512/1 = 512 iterations)
      [Kernel, vector]  32 vector MACs over K=2048        (2048/64 = 32 iterations)
```

Total work per column: 1024 rows × 2048 MACs/row = ~2.1M MACs.

### Case 2: M=2048, K=8192 (FFN down, `llama_ffn_fc3`)

```
[DDR → Shim]  Split A into 8 chunks of 256 rows → 8 columns in parallel
  [Column, outer]  2 output tiles of 128 rows each       (256/128 = 2 iterations)
    [Column, inner]  128 kernel calls of 1 row each       (128/1 = 128 iterations)
      [Kernel, vector]  128 vector MACs over K=8192       (8192/64 = 128 iterations)
```

Total work per column: 256 rows × 8192 MACs/row = ~2.1M MACs (same total, different shape).

---

## The Vectorized Kernel (`mv.cc`)

`matvec_vectorized<64>` processes one output row per outer iteration using AIE2P vector intrinsics with vector width `r=64` bf16 elements. For each output row:

1. Zero a 64-lane `accfloat` accumulator
2. Iterate over K in steps of 64: load 64 elements of A and B, call `aie::mac`
3. Reduce the 64-lane accumulator to a scalar with `aie::reduce_add`
4. Store one bf16 result

Vector MAC iterations per output row:
- K=2048: 2048 / 64 = **32 vector MAC operations**
- K=8192: 8192 / 64 = **128 vector MAC operations**

The `AIE_LOOP_MIN_ITERATION_COUNT(2)` pragma enables loop pipelining assuming K >= 128 (always true for Llama shapes).

The `row_offset` parameter (`mv.cc:70-72`) allows multiple sub-chunks of `m_input` rows to write into different positions of the same output tile buffer without pointer arithmetic in MLIR.

---

## Data Shapes: L3 → L1 (no explicit L2 tiling)

**There is no explicit L2 (MemTile) tiling in this design.** The ObjectFIFOs connect shim DMA directly to compute tiles. The compiler may route data through the MemTile physically (as a relay), but it does not perform any additional tiling or reshaping at L2. This is unlike the GEMM design which explicitly tiles at L2.

### Case 1: M=8192, K=2048 (FFN up/gate)

| Buffer | L3 (DDR) shape | L1 (per core) shape | FIFO depth | L1 bytes |
|--------|---------------|---------------------|------------|----------|
| A (matrix) | 8192 × 2048 bf16 (32 MB) | **1 × 2048** bf16 (4 KB) | 2 (double-buf) | 8 KB |
| B (vector) | 2048 bf16 (4 KB) | **2048** bf16 (4 KB) | 1 | 4 KB |
| C (output) | 8192 bf16 (16 KB) | **512** bf16 (1 KB) | 2 (double-buf) | 2 KB |
| **Total L1 per core** | | | | **~14 KB** |

### Case 2: M=2048, K=8192 (FFN down)

| Buffer | L3 (DDR) shape | L1 (per core) shape | FIFO depth | L1 bytes |
|--------|---------------|---------------------|------------|----------|
| A (matrix) | 2048 × 8192 bf16 (32 MB) | **1 × 8192** bf16 (16 KB) | 2 (double-buf) | 32 KB |
| B (vector) | 8192 bf16 (16 KB) | **8192** bf16 (16 KB) | 1 | 16 KB |
| C (output) | 2048 bf16 (4 KB) | **128** bf16 (256 B) | 2 (double-buf) | 512 B |
| **Total L1 per core** | | | | **~48.5 KB** |

AIE2P compute tiles have 64 KB of local memory. Case 1 uses ~22% of L1; Case 2 uses ~76%.

### Data flow diagram (per column)

```
              L3 (DDR)                              L1 (per core)
         ┌──────────────┐                      ┌─────────────────┐
         │  A: M×K      │  A_L3L1 FIFO         │ A: m_input × K  │ ← one row at a time
         │  (32 MB)     │ ──── depth=2 ────►   │ (4 or 16 KB)    │
         ├──────────────┤                      ├─────────────────┤
         │  B: K        │  B_L3L1 FIFO         │ B: K            │ ← full vector, read once
         │  (4/16 KB)   │ ──── depth=1 ────►   │ (4 or 16 KB)    │
         ├──────────────┤                      ├─────────────────┤
         │  C: M        │  C_L1L3 FIFO         │ C: m_output     │ ← accumulate, then drain
         │  (4/16 KB)   │ ◄─── depth=2 ────   │ (256 B / 1 KB)  │
         └──────────────┘                      └─────────────────┘
```

Each core streams through the matrix one row at a time (`m_input=1`), reuses the full vector B held in L1, and batches output into `m_output`-sized chunks before draining back to DDR. The double-buffered A FIFO allows the next row's DMA to overlap with the current row's computation.

---

## Hardware Layout (Llama shapes, cols=8)

```
           Col 0    Col 1    ...    Col 7
Row 0   [Shim]   [Shim]          [Shim]    ← DMA ingress/egress
Row 1   [MemTile]                [MemTile]  ← pass-through (no explicit tiling)
Row 2   [Core]   [Core]          [Core]    ← compute (1 worker per column)
Row 3   [Core]   [Core]          [Core]    ← unused
Row 4   [Core]   [Core]          [Core]    ← unused
Row 5   [Core]   [Core]          [Core]    ← unused
```

The `SequentialPlacer` assigns one `Worker` per column to a single core (one row used per column). Total compute cores in use: **8** out of 32 available (8 columns × 4 rows). Only the 8×1 topology is used.

> Note: The design instantiates one `Worker` per column (`design.py:114-125`), so only one core row per column is active. The remaining 3 core rows per column are idle.

---

## Vector-Matrix Multiplication Support

When `is_mv=False` (used for all Llama GQA and FFN projections except final vocab), the operator performs vector-matrix multiplication `y = x · A`. This is handled at the Python level in `op.py:155-158`: the matrix is transposed before being sent to the NPU, reducing it to the standard MV case. The MLIR design and C++ kernel are always matrix-vector.

The weight matrix shape in `op.py`:
- `is_mv=True`: weight is `(M, K)`, stored transposed as `(K, M)` column-major
- `is_mv=False`: weight is `(K, M)`, the double-transpose at forward time cancels out
