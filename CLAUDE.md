# IRON - Claude Code Context

IRON is AMD's open-source, close-to-metal Python API for fast execution on AMD Ryzen AI NPUs. It provides language bindings around the MLIR-AIE dialect.

## Environment Setup

```bash
cd /home/jiajli/apps/IRON
source ironenv/bin/activate        # Python 3.12.3 venv
source /opt/xilinx/xrt/setup.sh    # XRT runtime (required for device access)
```

## Project Structure

```
iron/                          # Main Python package
  common/                      # Shared infrastructure
    compilation.py             # Build system: artifact DAG + compilation rules
    aie_base.py                # AIEOperatorBase: base class for all operators
    aie_context.py             # AIEContext: compilation + runtime lifecycle manager
    aie_device_manager.py      # Singleton XRT device manager (DefaultNPURuntime)
    utils.py                   # torch<->numpy conversion, buffer helpers
    test_utils.py              # Test harness utilities
  operators/                   # 23+ operator implementations (GEMM, ReLU, SiLU, MHA, etc.)
  applications/                # ML applications (e.g., Llama 3.2 1B inference)
aie_kernels/                   # C++ kernel source for NPU cores
  aie2/                        # AIE2 architecture kernels
  aie2p/                       # AIE2P architecture kernels (primary target)
  generic/                     # Architecture-independent kernels
ironenv/                       # Python virtual environment
conftest.py                    # pytest fixture: aie_context
```

## Operator Pattern

Each operator in `iron/operators/<name>/` follows this structure:

| File | Purpose |
|------|---------|
| `op.py` | `AIEOperatorBase` subclass — defines artifacts, buffers, kernels, runlist, forward() |
| `design.py` | MLIR generation callback — defines ObjectFIFOs, Workers, Runtime data movement |
| `reference.py` | CPU golden reference implementation for validation |
| `test.py` | pytest tests with `aie_context` fixture |

## Compilation Flow

```
Python design.py callback
        │  (GenerateMLIRFromPythonCompilationRule)
        ▼
   .mlir file ─────────────────────┐
                                   │
C++ kernel (.cc)                   │  (AieccCompilationRule via aiecc.py)
        │  (PeanoCompilationRule)  │
        ▼                          ▼
   .o → .a  ──────────────►  .xclbin + insts.bin
                              (bitstream) (NPU instructions)
```

**Compilation rules** (applied iteratively in `compilation.py`):
1. `GenerateMLIRFromPythonCompilationRule` — imports design.py, calls callback, writes .mlir
2. `PeanoCompilationRule` — `clang++ --target=aie2p-none-unknown-elf` compiles C++ kernels to .o
3. `ArchiveCompilationRule` — `llvm-ar` bundles .o files into .a
4. `AieccCompilationRule` — `aiecc.py` compiles MLIR + kernel archives to xclbin + insts.bin

**Runtime execution** (in `aie_base.py`):
- `write_buffer()` → `bo.sync(TO_DEVICE)` → `xrt_kernel(opcode=3, insts, *buffers)` → `run.wait()` → `bo.sync(FROM_DEVICE)` → `read_buffer()`

## Key Concepts

- **ObjectFIFO**: Hardware-managed DMA queues for data movement between host, L2, and AIE cores
- **Worker**: Task assigned to an AIE core; acquires/releases FIFO elements
- **TensorAccessPattern (TAP)**: Describes how tiles are extracted from tensors
- **Program / SequentialPlacer**: Wraps design, assigns workers to physical tiles

## Development Commands

```bash
# Run all operator tests (excluding extensive)
pytest iron/operators/ -m "not extensive"

# Run all operator tests (including extensive)
pytest iron/operators/

# Run a specific operator's tests
pytest iron/operators/gemm/

# Lint / format
black .

# Install in editable mode
pip install -e .
```

## Dependencies

- `mlir_aie==v1.2.1` — MLIR dialect for AIE (provides `aie.iron`, `aie.utils`, `aiecc.py`)
- `llvm-aie` (nightly) — Peano toolchain (`clang++` targeting aie2p)
- `torch` (CPU-only) — tensor operations on host
- `numpy`, `ml_dtypes` — data type support (bfloat16)
- XRT runtime (`/opt/xilinx/xrt/`) — device access via `pyxrt`

## Git

- **Main branch**: `devel`
- **License**: Apache 2.0

---

## Llama 3.2 1B on IRON NPU

### Model Architecture

Llama 3.2 1B: 16 transformer layers, vocab 128256, emb_dim 2048, hidden_dim 8192,
32 attention heads, 8 KV groups, head_dim 64, bfloat16, RoPE base 500000.

```
Input IDs → Embedding (CPU)
  → 16x TransformerBlock:
      RMSNorm → GQA → Residual Add → RMSNorm → SwiGLU FFN → Residual Add
  → Final RMSNorm → Output Projection → Logits
```

Each block has two paths: **Prefill** (full prompt, GEMM-based) and **Decode** (single token with KV cache, GEMV-based).

### Running Inference

```bash
cd /home/jiajli/apps/IRON
source ironenv/bin/activate && source /opt/xilinx/xrt/setup.sh
pip install -r requirements_examples.txt  # tiktoken, torchtune, torchao, etc.

python iron/applications/llama_3.2_1b/inference.py \
    /path/to/model.safetensors /path/to/tokenizer.model \
    --num_tokens 10 --prompt_len 13
```

Or use `./run_llama.sh` (pre-configured wrapper script).

### Runtime Execution Model

**Operators execute sequentially.** Each AIE operator has its own xclbin (NPU bitstream).
Per operator call: sync buffers TO device → dispatch kernel (opcode=3) → wait → sync FROM device.
No cross-operator parallelism — NPU is reconfigured for each operator type.

**Within** each operator: data parallelism across up to 8 AIE columns × 4 rows = 32 cores.

**Exception: SwiGLU** fuses 4 sub-operators (2× GEMM + SiLU + Mul) into one combined xclbin
via `--xclbin-input` chaining, enabling runlist batching without NPU reconfiguration.

### Execution Trace Per Transformer Block (Prefill)

```
RMSNorm (pre-norm)           → rms_norm xclbin
GEMM (Q projection)          → gemm xclbin (M=prompt_len, K=2048, N=2048)
GEMM (K projection)          → gemm xclbin (M=prompt_len, K=2048, N=512)
GEMM (V projection)          → gemm xclbin (M=prompt_len, K=2048, N=512)
RoPE (queries)               → rope xclbin
RoPE (keys)                  → rope xclbin
Fused MHA (QK→softmax→·V)   → mha xclbin (8 pipelines)
GEMM (output projection)     → gemm xclbin (M=prompt_len, K=2048, N=2048)
ElementwiseAdd (residual)     → add xclbin
RMSNorm (post-norm)          → rms_norm xclbin
SwiGLU Prefill (fused FFN)   → combined xclbin (5 runlist entries, 4 kernel-ids)
ElementwiseAdd (residual)     → add xclbin
```

~12 NPU dispatches/block × 16 blocks = ~192 dispatches per prefill forward pass.

### Kernel Source Code Locations

| Operator | Python op.py | Python design.py | C++ kernel source |
|----------|-------------|-----------------|-------------------|
| GEMM | `iron/operators/gemm/op.py` | `iron/operators/gemm/design.py` | `aie_kernels/aie2p/mm.cc`, `aie_kernels/generic/convert_copy.cc` |
| GEMV | `iron/operators/gemv/op.py` | `iron/operators/gemv/design.py` | `aie_kernels/generic/mv.cc` |
| RoPE | `iron/operators/rope/op.py` | `iron/operators/rope/design.py` | `aie_kernels/generic/rope.cc` |
| MHA | `iron/operators/mha/op.py` | `iron/operators/mha/design.py` | `aie_kernels/aie2p/mha.cc`, `mm.cc`, `softmax.cc`, `generic/passThrough.cc` |
| RMSNorm | `iron/operators/rms_norm/op.py` | `iron/operators/rms_norm/design_weighted.py` | `aie_kernels/aie2p/rms_norm.cc`, `aie_kernels/generic/mul.cc` |
| Add | `iron/operators/elementwise_add/op.py` | `iron/operators/elementwise_add/design.py` | `aie_kernels/generic/add.cc` |
| SiLU | `iron/operators/silu/op.py` | `iron/operators/silu/design.py` | `aie_kernels/aie2p/silu.cc` |
| Mul | `iron/operators/elementwise_mul/op.py` | `iron/operators/elementwise_mul/design.py` | `aie_kernels/generic/mul.cc` |
| Softmax | `iron/operators/softmax/op.py` | `iron/operators/softmax/design.py` | `aie_kernels/aie2p/softmax.cc` |
| SwiGLU Prefill | `iron/operators/swiglu_prefill/op.py` | Composite (reuses GEMM, SiLU, Mul) | Reuses above |
| SwiGLU Decode | `iron/operators/swiglu_decode/op.py` | Composite (reuses GEMV, SiLU, Mul) | Reuses above |

### Build Artifacts

Default build dir: `build/` (relative to cwd). Each operator generates:

| Artifact | Naming Pattern | Example |
|----------|---------------|---------|
| MLIR | `{op}_{params}.mlir` | `gemm_2048x2048x2048_64x64x64_1_0.mlir` |
| xclbin | `{op}_{params}.xclbin` | `gemm_2048x2048x2048_64x64x64_1_0.xclbin` |
| Instructions | `{op}_{params}.bin` | `gemm_2048x2048x2048_64x64x64_1_0.bin` |
| Kernel object | `{op}.o` | `gemm_64x64x64_1_0.o` (from mm.cc) |
| Kernel archive | `{op}.a` | `gemm_64x64x64_1_0.a` |
| Project dir | `{mlir_name}.prj/` | Intermediate aiecc.py compilation artifacts |

### Operator Configurations in Llama

**Prefill** (multi-token): `num_aie_columns=8`, tile 64³, GEMM-based
**Decode** (single-token): `num_aie_columns=1-8`, GEMV-based, KV cache on CPU

| Operator | Prefill Config | Decode Config |
|----------|---------------|---------------|
| GEMM (QKV) | M=prompt_len, K=2048, N=2048/512, 8 cols | N/A |
| GEMV (QKV) | N/A | M=2048/512, K=2048, 8 cols |
| RoPE | rows=prompt_len×heads, cols=64 | rows=num_heads, angle_rows=1 |
| MHA | 32 heads, seq_len=prompt_len, 8 pipelines | N/A (CPU attention) |
| RMSNorm | size=prompt_len×2048, 8 cols, 2 ch | size=2048, 1 col, 2 ch |
| Add | size=prompt_len×2048, 8 cols | size=2048, 1 col |
| SwiGLU | seq=prompt_len, emb=2048, hidden=8192 | emb=2048, hidden=8192 |
| Final GEMM | M=prompt_len, N=128256, partition_N=4 | N/A |
| Final GEMV | N/A | M=128256, K=2048 |

### Profiling Results and Operator Test Mapping

Full profiling guide: `docs/llama_3.2_1b_profiling.md`

**Run with profiling** (`prompt_len=2048`, `num_tokens=100`):

```bash
WEIGHTS="/home/jiajli/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/model.safetensors"
TOKENIZER="/home/jiajli/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/original/tokenizer.model"
python iron/applications/llama_3.2_1b/inference.py "$WEIGHTS" "$TOKENIZER" \
    --num_tokens 100 --prompt_len 2048 --profile -vv
python iron/applications/llama_3.2_1b/analyze_profile.py \
    iron/applications/llama_3.2_1b/logs/profile_<timestamp>.log --function forward --sort total
```

**Wall-clock timing** (warm cache, `prompt_len=2048`, `num_tokens=100`):

```
Total time: 39.96 s  |  Prefill: 2.91 s  |  Decode: 37.05 s  |  2.67 tok/s
```

**Per-operator timing and test entry points** (decode-dominated run):

| Operator | Calls | Total (s) | Avg/call (ms) | C++ kernel | Individual test |
|----------|-------|-----------|----------------|------------|-----------------|
| `gqa.forward` | 1,600 | 15.69 | 9.8 | `mv.cc`, `rope.cc` | — (composite) |
| `feed_forward.forward` | 1,584 | 12.68 | 8.0 | `mv.cc`, `silu.cc`, `mul.cc` | — (composite) |
| `swiglu_decode.forward` | 1,584 | 12.59 | 7.9 | `mv.cc`, `silu.cc`, `mul.cc` | `iron/operators/swiglu_decode/test.py` |
| `gemv.forward` | 6,435 | 10.19 | 1.6 | `aie_kernels/generic/mv.cc` | `iron/operators/gemv/test.py` |
| `elementwise_add.forward` | 3,200 | 3.75 | 1.2 | `aie_kernels/generic/add.cc` | `iron/operators/elementwise_add/test.py` |
| `rms_norm.forward` | 3,300 | 3.41 | 1.0 | `aie_kernels/aie2p/rms_norm.cc` | `iron/operators/rms_norm/test.py` |
| `rope.forward` | 3,200 | 2.46 | 0.8 | `aie_kernels/generic/rope.cc` | `iron/operators/rope/test.py` |
| `run_runlist` (all NPU dispatches) | 17,816 | 25.82 | 1.4 | — | — |
| **Prefill:** `swiglu_prefill.forward` | 16 | 0.92 | 57.6 | `mm.cc`, `silu.cc`, `mul.cc` | `iron/operators/swiglu_prefill/test.py` |
| **Prefill:** `mha.forward` | 16 | 0.69 | 43.3 | `aie_kernels/aie2p/mha.cc` | `iron/operators/mha/test.py` |
| **Prefill:** `gemm.forward` | 65 | 0.69 | 10.6 | `aie_kernels/aie2p/mm.cc` | `iron/operators/gemm/test.py` |

**Key dispatch facts:**
- `run_runlist` timer covers only `xrt_kernel()` + `run.wait()`. Buffer sync (`bo.sync`) is outside the timer and accounts for the ~10 s gap between `run_runlist` total (25.8 s) and `aie_base.__call__` total (~35 s).
- Intermediate activations always round-trip through host DDR between operators, including within SwiGLU. The SwiGLU "fusion" eliminates XRT context switches and Python-side bo.sync overhead, not DDR traffic.
- GQA calls each sub-kernel (GEMV Q/K/V, RoPE Q/K, GEMV output) sequentially with separate DDR round-trips. No runlist batching.

**Run individual operator tests:**

```bash
pytest iron/operators/gemv/
pytest iron/operators/rms_norm/
pytest iron/operators/rope/
pytest iron/operators/mha/
pytest iron/operators/swiglu_decode/
pytest iron/operators/swiglu_prefill/
pytest iron/operators/elementwise_add/
pytest iron/operators/gemm/
```

### Llama App Files

```
iron/applications/llama_3.2_1b/
  inference.py                 # CLI entry point
  configs/llama32_1b.json      # Model + AIE operator config
  prompt.txt                   # Default prompt (King Lear)
  test.py                      # pytest (prompt_len 13/2048, tokens 1/40)
  torch_to_npy.py              # Weight format converter
  analyze_profile.py           # Parse --profile logs
  src/
    model_with_json.py         # Top-level model class
    tokenizer.py               # tiktoken wrapper
    utils.py                   # generate(), weight loading
    block/
      transformer.py           # TransformerBlock (norm→attn→add→norm→ffn→add)
      gqa.py                   # Grouped Query Attention
      feed_forward.py          # SwiGLU FFN
```
