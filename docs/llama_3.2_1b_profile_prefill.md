# IRON Llama 3.2 1B Profiling & Precision Verification

## Prerequisites

```bash
cd /home/jiajli/apps/IRON
source ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
```

## Build Cache Warning

The GEMM kernel archive filename (`gemm_64x64x64_0_0.a`) does **not** encode
`prio_accuracy` or `emulate_bf16_mmul_with_bfp16` flags. Regular GEMM tests use
`prio_accuracy=True`, while the Llama model uses `prio_accuracy=False`. Running
both against the same `build/` directory causes linker errors
(`undefined symbol: zero_bf16` or `zero_f32`).

**Solutions:**
- Use `--build-dir build_llama` for per-kernel Llama tests (separate cache).
- Before end-to-end model runs, clean stale GEMM archives if regular tests were
  run in the default `build/` directory:
  ```bash
  find build/ -name 'gemm_*64x64x64*' -delete
  find build/ -name 'gemm_*64x64x64*' -type d -exec rm -r {} +
  ```

---

## 1. End-to-End Model Profiling

### Command

```bash
WEIGHTS="/home/jiajli/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/model.safetensors"
TOKENIZER="/home/jiajli/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/original/tokenizer.model"

python iron/applications/llama_3.2_1b/inference.py \
    "$WEIGHTS" "$TOKENIZER" \
    --num_tokens 1 --prompt_len 2048 --profile -vv
```

### Analyze Profile Log

```bash
python iron/applications/llama_3.2_1b/analyze_profile.py \
    iron/applications/llama_3.2_1b/logs/profile_<timestamp>.log \
    --function forward --sort total --top 30
```

### Results (prefill seq_len=2048, 1 decode token)

```
Total time:   2.75 s
Prefill time: 2.75 s
```

| Function | Calls | Total (s) | Avg/call (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| `model.forward` | 1 | 2.744 | 2744.0 | 2744.0 | 2744.0 |
| `transformer.forward` | 16 | 2.437 | 152.3 | 145.5 | 161.2 |
| `gqa.forward` | 16 | 1.231 | 76.9 | 73.3 | 84.6 |
| `feed_forward.forward` | 16 | 0.919 | 57.5 | 55.6 | 59.1 |
| `swiglu_prefill.forward` | 16 | 0.919 | 57.4 | 55.5 | 59.1 |
| `gemm.forward` | 65 | 0.687 | 10.6 | 2.4 | 274.5 |
| `mha.forward` | 16 | 0.592 | 37.0 | 35.8 | 38.7 |
| `rope.forward` | 32 | 0.170 | 5.3 | 2.4 | 8.8 |
| `elementwise_add.forward` | 32 | 0.143 | 4.5 | 3.7 | 6.3 |
| `rms_norm.forward` | 33 | 0.142 | 4.3 | 3.6 | 6.1 |

---

## 2. Per-Kernel Llama Benchmarks

### Command

```bash
# Clean build dir for Llama-config tests (avoids archive conflicts)
mkdir -p build_llama

# Run all non-extensive Llama-marked tests
pytest iron/operators/ -m "llama and not extensive" \
    --build-dir build_llama --iterations 1 -v -s

# Run extensive Llama tests (2048-token configs)
pytest iron/operators/ -m "llama and extensive" \
    --build-dir build_llama --iterations 1 -v -s
```

### Results — Full Prefill (seq_len=2048, `llama and extensive`)

These are the actual Llama 3.2 1B shapes at full 2048-token prefill scale.

#### Elementwise Add

| Test | Shape | Latency (us) | Bandwidth (GB/s) | Corr | Max Err | Mean Err | 4%-fail |
|---|---|---|---|---|---|---|---|
| `llama_prefill_add_2048tok` | 4194304 elems, 8 cols | 429.4 | 58.6 | 0.99998 | 0.03 | 0.0038 | 0.0% |

#### GEMM (prio_accuracy=False, emulate_bfp16=True)

| Test | Shape (MxKxN) | Latency (us) | Throughput (GFLOP/s) | Bandwidth (GB/s) | Corr | Max Err | Mean Err | 4%-fail |
|---|---|---|---|---|---|---|---|---|
| `llama_kv_proj_2048tok` | 2048x2048x512 | 762.3 | 5634 | 16.5 | 0.99994 | 32.0 | 3.44 | 7.5% |
| `llama_q_out_proj_2048tok` | 2048x2048x2048 | 3539.1 | 4854 | 7.1 | 0.99994 | 40.0 | 3.44 | 7.5% |
| `llama_ffn_gate_up_2048tok` | 2048x2048x8192 | 14321.8 | 4798 | 5.3 | 0.99994 | 40.0 | 3.44 | 7.5% |
| `llama_ffn_down_2048tok` | 2048x8192x2048 | 12536.2 | 5482 | 6.0 | 0.99988 | 160.0 | 9.76 | 8.8% |
| `llama_final_vocab_2048tok` | 2048x2048x128256, N_part=4 | 216833.9 | 4962 | 4.9 | see below | — | — | — |

**Final vocab GEMM `partition_N=4` — bugs found and fixed:**

No existing test validated `partition_N > 1`. The Llama app test (`test.py`) only
checks `returncode == 0` with no output correctness validation. The standalone GEMM
test suite had no `partition_N > 1` cases before this profiling work added them.

**Bug 1 (FIXED): `forward()` returns wrong shape with `partition_N > 1` + static weights**

`_partition_B()` overwrote `self.static_weight_shape` to the single-partition size
`(N_padded, K) = (32256, 2048)`. When `forward()` was later called, it read N from
`self.static_weight_shape`, getting `N = 32256` instead of the full `N = 128256`,
then divided by `partition_N=4` again, yielding `N_part = 8064`. The returned
output had shape `(M, 32256)` instead of the expected `(M, 128256)`.

**Fix** (3 changes in `op.py`):
1. Initialize `static_weight_shape` with layout-aware full dimensions
   (`(N, K)` for b_col_maj, `(K, N)` for row-major) in `__init__`
2. Remove the `static_weight_shape` overwrite from `_partition_B()`
3. Fix `_execute_aie_operation()` to use `self.K, self.N` directly for static
   weights (per-partition dims), and fix the `N <= self.N` applicability check
   in `forward()` to `N <= self.N * self.partition_N`

After fix, `forward()` correctly returns `(1, 2048, 128256)` with partition 0
producing corr=0.99993.

**Bug 2 (OPEN — XRT/firmware level): Partitions 1-3 produce wrong results**

With the `forward()` fix applied, partition 0 is correct but partitions 1-3 are
still wrong (corr ~0.00). Investigation showed:

- Each partition run as a **standalone GEMM** (separate AIEContext, partition_N=1)
  produces correct results (corr=0.99994 for all 4).
- When 4 partitions share the same XRT kernel handle (same xclbin) and are invoked
  with different B/C buffer arguments, only the first invocation is correct.
- This happens both with XRT runlist batching AND sequential individual kernel
  invocations — the XRT kernel handle appears to cache buffer bindings from the
  first call.
- C_1-C_3 output is not a copy of C_0's result, but corrupted data (corr ~0.75
  with any partition's reference), suggesting the NPU reads wrong B data.

**Workaround**: Use separate AIEContext instances per partition (proven to work).
A proper fix requires changes to the XRT runtime or NPU firmware to properly
reset DMA state between kernel re-invocations with different buffer arguments.

**Impact on the Llama model**: The model's prefill final vocab GEMM operates on a
reduced effective vocabulary. The model still generates coherent text for common
tokens, but output quality is degraded for rare tokens (vocab indices 32064+).
A perplexity measurement or long-form generation would reveal the degradation.

**Reproduce:** `python docs/gemm_partition_n_bug/repro_partition_n_bug.py`
**GitHub issue:** Filed upstream with full reproduction script and analysis.
See `docs/gemm_partition_n_bug/` for all related files.

GEMM Llama tests (non-partitioned) use correlation-based pass criterion
(`corr >= 0.999`) rather than strict element-wise checks. The
`prio_accuracy=False` + `emulate_bf16_mmul_with_bfp16=True` configuration
matches the actual Llama model defaults and produces expected numerical
differences vs. the f32 reference.

#### MHA

| Test | Shape | Latency (us) | Bandwidth (GB/s) | Corr | Max Err | Mean Err | 4%-fail | Errors |
|---|---|---|---|---|---|---|---|---|
| `llama_prefill_2048tok` | 2048 seq, 32 heads, 64 dim, 8 pipelines | 30989.2 | 1.08 | 0.99758 | 0.25 | 0.0210 | 0.1% | 110/20971 max |

#### RMSNorm

| Test | Shape | Latency (us) | Bandwidth (GB/s) | Corr | Max Err | Mean Err | 4%-fail |
|---|---|---|---|---|---|---|---|
| `llama_prefill_rms_norm_2048tok` | 4194304 elems, 8 cols, weighted | 843.4 | 19.9 | 0.99998 | 0.09 | 0.0125 | 0.0% |

#### RoPE

| Test | Shape (rows x cols) | Latency (us) | Bandwidth (GB/s) | Corr | Max Err | Mean Err | 4%-fail |
|---|---|---|---|---|---|---|---|
| `llama_prefill_q_2048tok` | 65536x64 (32 heads x 2048 tok) | 844.7 | 20.2 | 0.99999 | 0.06 | 0.0068 | 0.6% |
| `llama_prefill_k_2048tok` | 16384x64 (8 KV groups x 2048 tok) | 256.8 | 17.4 | 0.99999 | 0.06 | 0.0068 | 0.6% |

#### SwiGLU Prefill (prio_accuracy=False)

| Test | Shape | Latency (us) | Bandwidth (GB/s) | Corr | 4%-fail |
|---|---|---|---|---|---|
| `llama_swiglu_prefill_2048tok_2048x8192` | seq=2048, emb=2048, hidden=8192 | 48100.2 | 0.17 | 0.99972 | 16.5% |

Note: Higher fail rate is expected for cascaded bf16 GEMM with K=2048. The
correlation (0.99972) confirms the computation is directionally correct.

---

### Results — Short Prompt & Decode (seq_len=13, `llama and not extensive`)

These are faster-running configs for quick validation.

#### GEMM (short prompt)

| Test | Shape (MxKxN) | Latency (us) | Throughput (GFLOP/s) | Corr | 4%-fail |
|---|---|---|---|---|---|
| `llama_kv_proj_13tok` | 13x2048x512 | 148.2 | 184 | 0.99995 | 5.7% |
| `llama_q_out_proj_13tok` | 13x2048x2048 | 491.6 | 222 | 0.99995 | 5.3% |

#### GEMV (decode, single token)

| Test | Shape (MxK) | Latency (us) | Throughput (GFLOP/s) | Corr | Max Err | 4%-fail |
|---|---|---|---|---|---|---|
| `llama_gqa_q` | 2048x2048 | 207.8 | 40.4 | 1.00000 | 0.00 | 0.0% |
| `llama_gqa_kv` | 512x2048 | 100.1 | 21.0 | 1.00000 | 0.00 | 0.0% |
| `llama_gqa_out` | 2048x2048 | 215.1 | 39.0 | 1.00000 | 0.00 | 0.0% |
| `llama_ffn_fc12` | 8192x2048 | 682.0 | 49.2 | 1.00000 | 0.12 | 0.0% |
| `llama_ffn_fc3` | 2048x8192 | 685.0 | 49.0 | 1.00000 | 0.00 | 0.0% |
| `llama_final_vocab` | 128256x2048 | 9526.4 | 55.1 | 1.00000 | 4.00 | 0.0% |

#### SwiGLU Decode (single token)

| Test | Shape | Latency (us) | Corr | 4%-fail |
|---|---|---|---|---|
| `llama_swiglu_decode_1x2048x8192` | emb=2048, hidden=8192 | 4645.1 | 0.99999 | 3.1% |

Note: 4 known DMA boundary zeros in intermediate buffer (allowed by `max_intermediate_errors=4`).

---

## 3. Precision Metrics

All operator tests now print standardized precision metrics:

```
Precision -- corr: 0.99995, max_err: 24.00, mean_err: 3.4903, 4%-fail: 5.7%
```

| Metric | Description |
|---|---|
| `corr` | Pearson correlation between NPU output and CPU f32 reference |
| `max_err` | Maximum absolute error across all elements |
| `mean_err` | Mean absolute error |
| `4%-fail` | Percentage of elements failing at 4% relative tolerance |

These metrics are captured by the CSV reporter via `@pytest.mark.metrics` patterns
and written to `tests_latest.csv`.

---

## 4. Key Configuration Differences

| Parameter | Regular GEMM Tests | Llama Model / Llama Tests |
|---|---|---|
| `prio_accuracy` | `True` | `False` |
| `emulate_bf16_mmul_with_bfp16` | `False` | `True` |
| Kernel symbols | `zero_f32`, `matmul_bf16_f32` | `zero_bf16`, `matmul_bf16_bf16` |
| Archive flags | `-Dbf16_f32_ONLY` | `-Dbf16_bf16_ONLY -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` |
| Element-wise tol | `rel_tol=0.005, abs_tol=0.005` | `rel_tol=0.04, abs_tol=1.0` |
| Pass criterion | `assert not errors` | `assert corr >= 0.999` |

---

## 5. File Changes

### Bug fixes

| File | Change |
|---|---|
| `iron/operators/gemm/op.py` | Fixed `forward()` shape bug with `partition_N > 1` + static weights (3 changes: layout-aware `static_weight_shape` init, removed overwrite in `_partition_B`, fixed applicability check and `_execute_aie_operation`) |

### Test infrastructure

| File | Change |
|---|---|
| `iron/common/test_utils.py` | Added `compute_precision_metrics()`, `print_precision_metrics()`, `report_precision()` |
| `iron/operators/gemm/test.py` | Added `prio_accuracy`/`emu_bfp16` params; Llama configs use `False/True` matching model defaults; `b_col_maj=True` for final vocab; correlation-based assertion for low-precision mode |
| `iron/operators/swiglu_prefill/test.py` | Changed Llama configs to `prio_accuracy=False` to match model defaults |
| `iron/operators/*/test.py` (8 files) | Added `report_precision()` calls and CSV metric patterns |
| `conftest.py` | Added `--build-dir` CLI option for build cache isolation |

### Documentation and bug reports

| File | Purpose |
|---|---|
| `docs/llama_3.2_1b_profile_prefill.md` | This file — prefill profiling results and precision analysis |
| `docs/gemm_partition_n_bug/issue_partition_n_bug.md` | GitHub issue filed upstream |
| `docs/gemm_partition_n_bug/repro_partition_n_bug.py` | Self-contained reproduction script |
| `docs/gemm_partition_n_bug/debug_partition_n.py` | Detailed diagnostic script |
