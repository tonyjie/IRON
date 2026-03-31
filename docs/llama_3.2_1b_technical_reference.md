# Llama 3.2 1B -- Technical Reference

> Architecture overview, file structure, operator pattern, high-level compilation flow,
> and prefill execution trace: see `CLAUDE.md`.
> Performance numbers and profiling: see `llama_3.2_1b_profile_prefill.md`.

---

## 1. App Internals

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `weights_file_path` | (required) | Path to safetensors model weights |
| `tokenizer_file_path` | (required) | Path to tiktoken model |
| `--num_tokens` | 1 | Tokens to generate |
| `--prompt` | reads `prompt.txt` | Custom prompt text |
| `--prompt_len` | 2048 | Truncate prompt to N tokens |
| `--use_prompt_template` | off | Enable chat template formatting |
| `--chat` | off | Interactive mode |
| `--profile` | off | Log all function calls with timestamps |
| `-v` to `-vvvv` | none | Verbosity levels (1-4) |

### JSON Config Contents (`configs/llama32_1b.json`)

**Model parameters:**
- `context_length`: 131072
- `rope_base`: 500000.0
- `rope_freq`: `factor=32.0`, `low_freq_factor=1.0`, `high_freq_factor=4.0`, `original_context_length=8192` -- extends effective context from 8k to 131k tokens via frequency scaling
- `dtype`: "bfloat16", `use_kv_cache`: true

**AIE toggle flags** (all enabled by default, each independently falls back to CPU when disabled):

| Flag | Component |
|------|-----------|
| `use_aie_attn_projection_gemm` | Attention QKV/output GEMM (prefill) |
| `use_aie_gqa_gemv` | Attention QKV/output GEMV (decode) |
| `use_aie_rope` | Rotary position embedding |
| `use_aie_fused_mha` | Fused multi-head attention (prefill) |
| `use_aie_ffn_swiglu` | Fused SwiGLU FFN |
| `use_aie_ffn_gemv` | Individual FFN GEMV projections |
| `use_aie_norm1` | Pre-attention RMSNorm |
| `use_aie_norm2` | Pre-FFN RMSNorm |
| `use_aie_final_norm` | Final RMSNorm before output head |
| `use_aie_residual` | Residual additions |
| `use_aie_final_gemm` | Output head GEMM (prefill) |
| `use_aie_final_gemv` | Output head GEMV (decode) |

### `generate()` Function Details

- **Iteration 0 (Prefill):** Process entire prompt `idx[:, -context_size:]`, no `input_pos`, populate KV cache
- **Iteration 1+ (Decode):** Process single token `idx[:, -1:]`, pass `input_pos` for KV cache position tracking
- **Sampling:** Top-k filtering (keep top 50) -> temperature scaling (0.7) -> softmax -> multinomial sampling
- **EOS detection:** Stop if model generates end-of-sequence token
- **Hook removal:** After prefill completes, removes layer forward hooks to reduce Python overhead during decode

### Decode Dispatch Trace (Per Transformer Block)

```
1.  RMSNorm (pre-norm)         -> rms_norm xclbin, 1 AIE column
2.  GEMV (Q projection)        -> gemv xclbin
3.  GEMV (K projection)        -> gemv xclbin
4.  GEMV (V projection)        -> gemv xclbin
5.  RoPE (query + key)         -> rope xclbin (2 calls)
6.  Attention                  -> CPU (scaled_dot_product_attention with cached K/V)
7.  GEMV (output projection)   -> gemv xclbin
8.  ElementwiseAdd (residual)  -> add xclbin
9.  RMSNorm (post-norm)        -> rms_norm xclbin
10. SwiGLU Decode (fused FFN)  -> swiglu_decode xclbin
11. ElementwiseAdd (residual)  -> add xclbin
```

~11 dispatches/block x 16 blocks per decode token. Decode attention is on CPU because the growing KV cache makes NPU tiling inefficient for variable-length attention.

### Design Patterns

1. **Dual operator pattern** -- Every operation has a prefill variant (8 AIE cols, GEMM-based) and a decode variant (1 AIE col, GEMV-based). Created separately at init, selected at runtime by phase.

2. **Config-driven offloading** -- Each operation independently toggled between AIE and CPU via JSON config flags. Enables incremental development and debugging.

3. **Shape-based phase detection** -- Forward methods detect prefill vs decode from input tensor dimensions (`seq_len > 1` = prefill, `seq_len == 1` with KV cache = decode). No explicit phase flag is passed.

4. **Declarative compilation** -- Operators declare what artifacts they need; compilation rules figure out how to build them, with caching to skip already-built artifacts.

5. **Fused operators** -- SwiGLU combines 4 sub-operations into one xclbin via `--xclbin-input` chaining. Fused MHA similarly combines Q@K, softmax, and attn@V.

---

## 2. NPU Execution

### XRT Context Caching

`CachedXRTRuntime` manages hardware contexts:
- Each xclbin gets its own `pyxrt.hw_context` when first loaded
- Up to 32 simultaneous contexts on NPU2
- Contexts are cached by xclbin path -- switching between operators is an implicit XRT context switch with negligible overhead after first load

**Lazy loading during `prepare_runtime()`:**

```
for each operator:
  for each kernel in operator:
    device_manager.get_kernel_handle(xclbin_path, kernel_name, insts_path)
      -> CachedXRTRuntime.load()
         -> if xclbin_path in cache: reuse hw_context
         -> else: pyxrt.xclbin(path)
                  device.register_xclbin(xclbin)
                  pyxrt.hw_context(device, xclbin_uuid)
                  cache[xclbin_path] = hw_context
```

After `prepare_runtime()` completes, all xclbins are loaded and cached.

### Memory Hierarchy

```
+-------------------------------------------------+
|                  Host DDR                       |
|  (shared between CPU and NPU)                  |
|                                                 |
|  +---------+ +---------+ +---------+          |
|  | Input   | | Weights | | Output  |          |
|  | Buffer  | | Buffer  | | Buffer  |  <- XRT BOs (host_only)
|  +----+----+ +----+----+ +----+----+          |
|       |           |           |                |
+-------+-----------+-----------+----------------+
        |           |           |
   bo.sync(TO_DEVICE)      bo.sync(FROM_DEVICE)
        |           |           ^
+-------+-----------+-----------+----------------+
|       v           v           |     NPU        |
|  +--------------------------------------------+|
|  |            ShimDMA Engines                 ||
|  |   (move data between DDR and L2)           ||
|  +-------------------+------------------------+|
|                      |                          |
|  +-------------------v------------------------+|
|  |         L2 Memory (per column)             ||
|  |    (ObjectFIFO staging, ~64 KB/col)        ||
|  +-------------------+------------------------+|
|                      |                          |
|  +-------------------v------------------------+|
|  |      L1 Memory (per AIE core)              ||
|  |    (local scratchpad, ~32 KB/core)         ||
|  |    Vector ALU: 16 bfloat16/cycle           ||
|  +--------------------------------------------+|
+-------------------------------------------------+
```

### Per-Dispatch Data Movement Trace

```
Host CPU (PyTorch tensor)
  |
  | write_buffer(): np.copyto() into pre-allocated BO
  v
Host DDR (XRT buffer object)
  |
  | bo.sync(TO_DEVICE): flush CPU cache, ensure NPU can see the data
  v
ShimDMA reads from DDR -> streams into L2 via ObjectFIFO
  |
  | DMA tiles data according to TensorAccessPattern
  v
L2 -> L1: per-core local memory
  |
  | Vector ALU computes (matrix multiply, SiLU, etc.)
  v
L1 -> L2 -> ShimDMA writes results back to DDR
  |
  | bo.sync(FROM_DEVICE): invalidate CPU cache
  v
Host DDR (result in XRT buffer object)
  |
  | read_buffer(): np.frombuffer() from BO, then numpy_to_torch()
  v
Host CPU (PyTorch tensor) -- passed to next operator
```

Every operator call round-trips through host DDR. There is no mechanism to keep activations on-chip between operators (each operator has its own xclbin with its own ObjectFIFO layout).

### Buffer Reuse Strategy

- **Static buffers** (weights): Allocated in a `static_data_pool`. Written once, synced TO_DEVICE on each operator call. Same weight BO is shared across all 16 layers that use the same operator.
- **Dynamic buffers** (activations): Allocated in size-pooled groups. Conflict analysis ensures buffers that appear in the same runlist entry get separate BOs, while non-conflicting buffers share a BO.
- **No malloc/free during inference** -- only `write_buffer` -> `sync` -> `run` -> `sync` -> `read_buffer`.

Buffer objects use `pyxrt.bo(device, buffer_size, pyxrt.bo.host_only, 0x10000)` -- allocated in host DDR, page-aligned.

### SwiGLU Runlist API

```python
xrt_runlist = pyxrt.runlist(context)
xrt_runlist.add(run_gemm_1_with_W1)     # kernel_id=0x901, insts_1
xrt_runlist.add(run_gemm_1_with_W2)     # kernel_id=0x901, insts_1 (same kernel, different weight buffer)
xrt_runlist.add(run_silu)                # kernel_id=0x902, insts_2
xrt_runlist.add(run_mul)                 # kernel_id=0x903, insts_3
xrt_runlist.add(run_gemm_2)             # kernel_id=0x904, insts_4
xrt_runlist.execute()                    # single submission to XRT
xrt_runlist.wait()                       # single wait
```

Reduces 5 separate write-sync-dispatch-wait-sync-read cycles into a single cycle.

---

## 3. Kernel Math

### RMSNorm

**Formula:**

```
rms = sqrt( (1/d) * sum_{i=0}^{d-1} x_i^2 + epsilon )
output_i = (x_i / rms) * gamma_i
```

Where `d = emb_dim = 2048`, `gamma` is a learnable weight vector, `epsilon = 1e-5`.

**Two-stage pipeline on NPU:**
1. **Stage 1 (RMS Norm cores):** Computes `sum(x^2)`, then `x * invsqrt(sum/d + eps)`
2. **Stage 2 (Multiply cores):** Multiplies normalized output by `gamma` weights

C++ kernel (`aie_kernels/aie2p/rms_norm.cc`):
```cpp
float rms = sum_sq / cols + epsilon;
float inv_rms = aie::invsqrt(rms);
output[i] = input[i] * inv_rms;
```

### RoPE (Rotary Position Embedding)

**Inverse frequencies:**

```
theta_i = rope_base ^ (-2i / head_dim)     for i = 0, 1, ..., head_dim/2 - 1

With rope_base = 500,000 and head_dim = 64:
  theta_i = 500000 ^ (-2i / 64)            for i = 0, 1, ..., 31
```

**Llama 3.2 frequency scaling** (from `rope_freq` config):
- `factor = 32.0`, `low_freq_factor = 1.0`, `high_freq_factor = 4.0`, `original_context_length = 8192`
- Extends effective context from 8k to 131k tokens

**Angle at position m:**

```
angle(m, i) = m * theta_i
```

**Interleaved rotation** (method_type=1, used by Llama):

```
x_even = x[0], x[2], x[4], ...
x_odd  = x[1], x[3], x[5], ...

cos = cos(angle(m, 0)), cos(angle(m, 1)), ...     (head_dim/2 values)
sin = sin(angle(m, 0)), sin(angle(m, 1)), ...     (head_dim/2 values)

output[2i]   = x_even[i] * cos[i] - x_odd[i] * sin[i]
output[2i+1] = x_even[i] * sin[i] + x_odd[i] * cos[i]
```

**Angle LUT format** (interleaved cos/sin):
```
LUT[m, 2i]   = cos(angle(m, i))
LUT[m, 2i+1] = sin(angle(m, i))
```

**Constraints:** `cols >= 32` and must be a multiple of 32 (kernel processes two 16-element vectors at a time).

C++ kernel (`aie_kernels/generic/rope.cc`):
```cpp
x_even = filter_even(x);
x_odd  = filter_odd(x);
cos_val = filter_even(lut);
sin_val = filter_odd(lut);

output_even = x_even * cos_val - x_odd * sin_val;
output_odd  = x_even * sin_val + x_odd * cos_val;
output = interleave(output_even, output_odd);
```

### Fused MHA (Flash Attention)

**Configuration:** `num_heads=32, B_q=64, B_kv=64, num_of_pipelines=8`

8 pipelines process Q blocks in parallel. For S=2048: `num_q_blocks = S/64 = 32`, each pipeline handles 4 Q blocks.

**Softmax scale factor** (bakes in `log2(e)` for fast `exp2()`):
```
scale = (1 / sqrt(d)) * log2(e) = 0.125 * 1.4453125 = 0.1806640625
```

**Block masking logic:**
- `kv_block > q_block`: entire block = -inf (future tokens)
- `kv_block == q_block`: mask upper triangle within block
- `kv_block < q_block`: no masking needed

**Online softmax (running statistics per row across KV blocks):**

```
m_i = max of all scores seen so far for row i
l_i = sum of all exp(score - m_i) seen so far for row i

When processing a new KV block:
  m_new = max(m_old, max(new_scores))
  rescale = exp(m_old - m_new)
  l_new = rescale * l_old + sum(exp(new_scores - m_new))
  O_new = rescale * O_old + exp(new_scores - m_new) @ V_block

Final: O = O / l
```

### SwiGLU Step-by-Step

```
FFN(x) = W_down @ ( SiLU(W_gate @ x) * (W_up @ x) )
```

**Prefill (S = prompt_len):**

```
Input x:          (S, 2048)
Step 1 - Gate:    gate = x @ W_gate^T = (S, 2048) @ (2048, 8192) = (S, 8192)
Step 2 - Up:      up   = x @ W_up^T   = (S, 2048) @ (2048, 8192) = (S, 8192)
Step 3 - SiLU:    gate_act = gate * sigmoid(gate)                 = (S, 8192)
Step 4 - Mul:     inter = gate_act * up                          = (S, 8192)
Step 5 - Down:    out  = inter @ W_down^T = (S, 8192) @ (8192, 2048) = (S, 2048)
```

**Decode (single token):**

```
Input x:          (2048,)
Step 1 - Gate:    gate = W_gate @ x = (8192,)
Step 2 - Up:      up   = W_up @ x   = (8192,)
Step 3 - SiLU:    gate_act = gate * sigmoid(gate) = (8192,)
Step 4 - Mul:     inter = gate_act * up            = (8192,)
Step 5 - Down:    out  = W_down @ inter = (2048,)
```

### SiLU Tanh Approximation

The C++ kernel (`aie_kernels/aie2p/silu.cc`) uses:
```
sigmoid(x) = (tanh(x/2) + 1) / 2
SiLU(x) = x * (tanh(x/2) + 1) / 2
```

Processes 16 bfloat16 elements per vector operation.

### Elementwise Multiply

C++ kernel (`aie_kernels/generic/mul.cc`):
```cpp
auto A = aie::load_v<16>(a + i);
auto B = aie::load_v<16>(b + i);
auto C = aie::mul(A, B).to_vector<bfloat16>();
aie::store_v(c + i, C);
```

### Weight Transposition

- **GEMM** (prefill): Weights stored transposed -- `x @ W^T` with W in transposed layout for efficient tiled access
- **GEMV** (decode): Weights stored un-transposed (row-major) -- `W @ x` reads rows directly

The same logical weights are reformatted for each operator's memory access pattern.

---

## 4. Compilation Details

### Three-Layer Source Structure

Every AIE operator has three source layers:

```
op.py          -- Python operator class
                  Declares compilation artifacts, buffers, runlist, forward()

design.py      -- Python MLIR generator
                  Defines ObjectFIFOs, Workers, tile placement, data movement
                  Called at compile time to produce a .mlir file

C++ kernel     -- AIE core compute logic
                  The actual vector ALU code running on each AIE core
                  Lives in aie_kernels/aie2p/ or aie_kernels/generic/
```

The `op.py` file ties everything together: it references both the `design.py` callback and the C++ kernel source files, declaring them as compilation artifacts with their dependency relationships.

### Compiler Flags Per Operator

| Operator | Flags |
|----------|-------|
| GEMM | `-DDIM_M=64 -DDIM_K=64 -DDIM_N=64 -DROUND_CONV_EVEN -Dbf16_bf16_ONLY -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` |
| RoPE | `-DINTERLEAVED` (Llama uses method 1) |
| MHA passThrough | `-DBIT_WIDTH=16` |
| MHA mm.cc (col-major) | `-DB_COL_MAJ` (for Q @ K^T) |
| MHA mm.cc (row-major) | (default, for Attn @ V) |
| GEMV, RMSNorm, SiLU, Add, Mul | (none) |

### Exact Peano clang++ Command

```bash
<peano>/bin/clang++ \
  -O2 \
  -std=c++20 \
  --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined \
  -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw \
  -I<mlir_aie>/include \
  [operator-specific -D flags] \
  -c <source.cc> \
  -o build/<kernel>.o
```

### Exact llvm-ar Command

```bash
<peano>/bin/llvm-ar rcs build/<archive>.a build/<kernel1>.o build/<kernel2>.o ...
```

### Exact aiecc.py Command

```bash
python <mlir_aie>/bin/aiecc.py \
  --no-compile-host \
  --no-xchesscc \
  --no-xbridge \
  --peano <peano_dir> \
  [--dynamic-objFifos] \
  --aie-generate-xclbin --xclbin-name=<name>.xclbin \
  --xclbin-kernel-name=MLIR_AIE \
  --aie-generate-npu --npu-insts-name=<name>.bin \
  build/<name>.mlir
```

`--dynamic-objFifos` is required for MHA (runtime-parameterized ObjectFIFO sizes).

### GEMM Artifact Dependency Graph

```
aie_kernels/aie2p/mm.cc                aie_kernels/generic/convert_copy.cc
        |                                          |
   PeanoCompilationRule                       PeanoCompilationRule
   clang++ -DDIM_M=64 -DDIM_K=64 ...        clang++ -c convert_copy.cc
        |                                          |
        v                                          v
  gemm_64x64x64_0_0.o                      convert_copy.o
        |                                          |
        +------------------+-------------------+
                           |
                  ArchiveCompilationRule
                  llvm-ar rcs
                           |
                           v
                  gemm_64x64x64_0_0.a
                           |
                           |     iron/operators/gemm/design.py
                           |              |
                           |     GenerateMLIRFromPythonCompilationRule
                           |     my_matmul(M=2048, K=2048, N=2048, ...)
                           |              |
                           |              v
                           |     gemm_2048x2048x2048_64x64x64_0_0.mlir
                           |              |
                           +------+-------+
                                  |
                         AieccCompilationRule
                         aiecc.py --aie-generate-xclbin --aie-generate-npu
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
  gemm_2048x2048x2048_64x64x64_0_0.xclbin   gemm_2048x2048x2048_64x64x64_0_0.bin
  (NPU bitstream)                             (NPU instructions)
```

### MHA: 5-Object Compilation

MHA compiles 5 object files from 4 source files (`mm.cc` compiled twice):

```
aie2p/mm.cc -----------> mha_mm.o          (column-major: -DB_COL_MAJ)
aie2p/mm.cc -----------> mha_mm_rowmaj.o   (row-major, symbols renamed)
aie2p/softmax.cc ------> mha_softmax.o
aie2p/mha.cc ----------> mha_mha.o
generic/passThrough.cc > mha_passThrough.o  (-DBIT_WIDTH=16)
        |
        v
    mha_kernels.a
        |
        v (+ mha.mlir from design.py)
    mha.xclbin + mha.bin
```

**Symbol rename** (avoids collision between the two mm.cc variants):

```bash
llvm-objcopy-18 \
  --redefine-sym matmul_bf16_bf16=matmul_bf16_bf16_rowmaj \
  --redefine-sym zero_bf16=zero_bf16_rowmaj \
  ... \
  mha_mm_rowmaj.o
```

### SwiGLU xclbin Chaining

Compilation chains each sub-operator's xclbin as input to the next:

```
Step 1: aiecc.py gemm_1.mlir
        -> gemm_1.xclbin                             (kernel_id=0x901)

Step 2: aiecc.py silu.mlir --xclbin-input=gemm_1.xclbin
        -> combined.xclbin                            (now has 0x901 + 0x902)

Step 3: aiecc.py mul.mlir --xclbin-input=combined.xclbin
        -> combined.xclbin                            (now has 0x901 + 0x902 + 0x903)

Step 4: aiecc.py gemm_2.mlir --xclbin-input=combined.xclbin
        -> combined.xclbin                            (final: 0x901 + 0x902 + 0x903 + 0x904)
```

| Sub-Operator | Kernel ID | Instance Name |
|-------------|-----------|---------------|
| GEMM gate/up | 0x901 | swiglu_gemm_1 |
| SiLU | 0x902 | swiglu_silu |
| Elementwise Mul | 0x903 | swiglu_eltwise_mul |
| GEMM down | 0x904 | swiglu_gemm_2 |

**op.py setup code:**

```python
# Get artifacts from sub-operators
gemm_1_xclbin, gemm_1_insts = gemm_1.get_artifacts(prefix="swiglu_gemm_1_")
silu_xclbin,   silu_insts   = silu.get_artifacts(prefix="swiglu_silu_")
mul_xclbin,    mul_insts     = eltwise_mul.get_artifacts(prefix="swiglu_eltwise_mul_")
gemm_2_xclbin, gemm_2_insts = gemm_2.get_artifacts(prefix="swiglu_gemm_2_")

# Chain: each xclbin takes the previous as input
silu_xclbin.xclbin_input = gemm_1_xclbin
mul_xclbin.xclbin_input  = silu_xclbin
gemm_2_xclbin.xclbin_input = mul_xclbin

# Assign kernel IDs
gemm_1_xclbin.extra_flags += ["--xclbin-kernel-id=0x901"]
silu_xclbin.extra_flags   += ["--xclbin-kernel-id=0x902"]
mul_xclbin.extra_flags    += ["--xclbin-kernel-id=0x903"]
gemm_2_xclbin.extra_flags += ["--xclbin-kernel-id=0x904"]
```

SwiGLU Decode uses the identical pattern but with GEMV instead of GEMM.

### File Naming Convention

- **Kernel archive:** `{op}_{tile_m}x{tile_k}x{tile_n}_{b_col_maj}_{c_col_maj}.a` -- tile dimensions only (reusable across problem sizes)
- **xclbin/insts:** `{op}_{M}x{K}x{N}_{tile_m}x{tile_k}x{tile_n}_{b_col_maj}_{c_col_maj}.xclbin` -- full problem size (each unique size gets its own bitstream)

Operators with identical parameters share the same xclbin (e.g., all 16 layers' Q-projection GEMMs).
