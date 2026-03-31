# Llama 3.2 1B -- Measurement Guide

> For measured results, see `llama_3.2_1b_profile_prefill.md`

---

## 1. Profiling Mechanisms

Two independent sources of timing data:

| Mechanism | Activation | What it measures |
|-----------|-----------|-----------------|
| **Wall-clock timing summary** | Always on (no flags) | Total time, prefill time, tokens generated, tokens/sec. Printed to console after generation completes. |
| **Function-level profiler** | `--profile` flag | Uses `sys.setprofile()` to log every Python function entry/exit with `time.perf_counter()` timestamps. Produces a parseable log file. |

**What `--profile` captures:**
- All operators' `forward()` methods
- `run_runlist()` in `iron/common/aie_base.py` (NPU dispatch)
- Buffer sync helpers, weight loading, KV cache management
- Compilation (`compile_all`, `apply_rules`) if artifacts are not cached

**What `--profile` does NOT capture:**
- Time inside C extensions or XRT native calls (appear as a single duration on the Python side)
- Hardware-level per-core timing within an AIE kernel
- Anything before `enable_profiling()` is called (tokenizer init, argument parsing)

**Wall-clock timing fields:**

| Field | Definition |
|-------|-----------|
| Total time | Wall time from model load to last token, includes `prepare_runtime()` and compilation if not cached |
| Prefill time | Time to process the full input prompt through all 16 transformer blocks |
| Tokens per second | `(tokens_generated - 1) / decode_wall_time` |

First run is slower due to xclbin compilation and XRT context loading. Subsequent runs with same configuration use cached artifacts.

---

## 2. Profile Output

### Log locations

| Output | Location | Produced when |
|--------|----------|---------------|
| Wall-clock timing summary | Console (stdout) | Always |
| Verbose inference log | `logs/inference_<timestamp>.log` | With `-v` or higher |
| Function profiling log | `logs/profile_<timestamp>.log` | With `--profile` |

The `logs/` directory is created automatically relative to the working directory.

### Log format

Each entry is one of:

```
<wallclock> - [CALL] <filepath>:<function>:<line> started at <perf_counter>
<wallclock> - [RETURN] <filepath>:<function>:<line> ended at <perf_counter>
```

`<perf_counter>` values are from `time.perf_counter()` (seconds, high resolution). Duration = difference between matching CALL and RETURN timestamps.

### Log size estimate

At `prompt_len=2048`, `num_tokens=100`: approximately **370 MB** (6-7 million lines). High volume is due to stdlib functions (path operations, logging internals) being traced alongside application code.

---

## 3. analyze_profile.py Reference

Location: `iron/applications/llama_3.2_1b/analyze_profile.py`

### CLI options

```bash
# Top N functions by total time
python analyze_profile.py <log> --top 30 --sort total

# Filter by function name substring
python analyze_profile.py <log> --function forward --sort total
python analyze_profile.py <log> --function run_runlist
python analyze_profile.py <log> --function gemv

# Filter by minimum call count
python analyze_profile.py <log> --min-calls 100 --sort avg

# Export to CSV
python analyze_profile.py <log> --csv results.csv --sort total --min-calls 2
```

`--sort` accepts: `total`, `avg`, `max`, `calls`.

`--function` prints detailed statistics including standard deviation for each matching function.

### Output columns

Each row is identified by `filepath:function_name:line_number`.

| Column | Definition |
|--------|-----------|
| Calls | Number of invocations |
| Total | Cumulative wall time inside this function **and all functions it called** |
| Avg | Mean duration per call |
| Min / Max | Fastest and slowest individual calls |
| Median | 50th percentile duration |

### CSV columns

8 columns: `function_name`, `call_count`, `total_time_seconds`, `avg_time_seconds`, `median_time_seconds`, `min_time_seconds`, `max_time_seconds`, `std_dev_seconds`.

### Cumulative-vs-leaf warning

Total time is **cumulative** (includes subcalls). Outer functions like `transformer.forward` include all time spent in the operators they call. Do not sum totals across nested functions -- it double-counts. To get leaf-level time for a function, subtract the totals of its direct children.

---

## 4. Call Chain and Timer Boundaries

```
model.forward()                          <-- profiler: full model time
  transformer.forward()                  <-- profiler: per-block time
    gqa.forward() / feed_forward.forward()   <-- profiler: per-operator time
      write_buffer()                     <-- host memory copy; OUTSIDE run_runlist timer
      run_runlist()                      <-- profiler captures this
        bo.sync(TO_DEVICE)               <-- OUTSIDE run_runlist timer
        xrt_kernel(opcode=3, ...)        <-- INSIDE timer: kernel dispatch
        run.wait()                       <-- INSIDE timer: wait for NPU completion
        bo.sync(FROM_DEVICE)             <-- OUTSIDE run_runlist timer
      read_buffer()                      <-- host memory view; OUTSIDE run_runlist timer
```

**`run_runlist` times:** Only `xrt_kernel()` dispatch + `run.wait()`.

**`run_runlist` does NOT time:** `bo.sync()` calls. The gap between `forward()` total and `run_runlist` total is dominated by buffer sync and host memory copies.

### Profiler function to xclbin mapping

| Profiler function | xclbin pattern (in `build/`) | C++ kernel source |
|-------------------|------------------------------|-------------------|
| `gemm.forward` | `gemm_MxKxN_64x64x64_*.xclbin` | `aie_kernels/aie2p/mm.cc` |
| `gemv.forward` (QKV, output proj) | `gemv_MxK_*tsi_*tso_*col.xclbin` | `aie_kernels/generic/mv.cc` |
| `gemv.forward` (final output head) | `gemv_128256x2048_4tsi_32tso_8col.xclbin` | `aie_kernels/generic/mv.cc` |
| `mha.forward` | `mha_{heads}h_{kv}kv_{seq}s_{dim}d.xclbin` | `aie_kernels/aie2p/mha.cc`, `mm.cc`, `softmax.cc` |
| `rms_norm.forward` | `weighted_rms_*c_*ch_*_*t.xclbin` | `aie_kernels/aie2p/rms_norm.cc`, `generic/mul.cc` |
| `rope.forward` | `rope_1c_*rows_64cols_*arows_0m.xclbin` | `aie_kernels/generic/rope.cc` |
| `elementwise_add.forward` | `add_*c_2ch_*_*t.xclbin` | `aie_kernels/generic/add.cc` |
| `swiglu_decode.forward` | `swiglu_decode_gemv_2_*.xclbin` (combined) | `mv.cc`, `aie2p/silu.cc`, `generic/mul.cc` |
| `swiglu_prefill.forward` | `swiglu_gemm_2_*.xclbin` (combined) | `aie2p/mm.cc`, `aie2p/silu.cc`, `generic/mul.cc` |

### Profiler limitations

The profiler measures Python-observable time only. It cannot isolate:

- Actual NPU core execution time vs. DMA transfer time within a single dispatch (both inside `run.wait()`)
- Per-AIE-core utilization (how many of the 32 cores were active)
- Memory bandwidth bottlenecks vs. compute bottlenecks

XRT provides a separate event tracing mechanism for hardware-level cycle counts and DMA stall analysis.

---

## 5. Xclbin Inventory

24 distinct xclbins compiled for Llama 3.2 1B:

| xclbin | Used for | Phase |
|--------|----------|-------|
| `gemm_2048x2048x2048_64x64x64_0_0.xclbin` | Q projection | Prefill |
| `gemm_2048x2048x512_64x64x64_0_0.xclbin` | K, V projections | Prefill |
| `gemm_2048x2048x8192_64x64x64_0_0.xclbin` | FFN gate/up (standalone) | Prefill |
| `gemm_2048x8192x2048_64x64x64_0_0.xclbin` | FFN down (standalone) | Prefill |
| `gemm_2048x2048x32256_64x64x64_1_0.xclbin` | Final output head (partitioned) | Prefill |
| `mha.xclbin` | Fused multi-head attention | Prefill |
| `swiglu_gemm_1_2048x2048x8192_64x64x64_0_0.xclbin` | SwiGLU gate+up (combined) | Prefill |
| `swiglu_gemm_2_2048x8192x2048_64x64x64_0_0.xclbin` | SwiGLU down (combined) | Prefill |
| `gemv_2048x2048_1tsi_128tso_8col.xclbin` | Q, output projection | Decode |
| `gemv_512x2048_1tsi_32tso_8col.xclbin` | K, V projections | Decode |
| `gemv_8192x2048_1tsi_512tso_8col.xclbin` | FFN gate (standalone) | Decode |
| `gemv_2048x8192_1tsi_256tso_8col.xclbin` | FFN down (standalone) | Decode |
| `gemv_8192x2048_4tsi_1024tso_8col.xclbin` | SwiGLU gate/up | Decode |
| `gemv_2048x8192_1tsi_256tso_8col.xclbin` | SwiGLU down | Decode |
| `gemv_128256x2048_4tsi_32tso_8col.xclbin` | Final output head | Decode |
| `swiglu_decode_gemv_2_2048x8192_1tsi_256tso_8col.xclbin` | SwiGLU fused (combined) | Decode |
| `weighted_rms_8c_2ch_4194304_2048t.xclbin` | RMSNorm pre/post-attn, pre-FFN | Prefill |
| `weighted_rms_1c_2ch_2048_2048t.xclbin` | RMSNorm all norms | Decode |
| `rope_1c_65536rows_64cols_2048arows_0m.xclbin` | RoPE on Q | Prefill |
| `rope_1c_16384rows_64cols_2048arows_0m.xclbin` | RoPE on K | Prefill |
| `rope_1c_32rows_64cols_1arows_0m.xclbin` | RoPE on Q | Decode |
| `rope_1c_8rows_64cols_1arows_0m.xclbin` | RoPE on K | Decode |
| `add_8c_2ch_4194304_2048t.xclbin` | Residual add | Prefill |
| `add_1c_2ch_2048_2048t.xclbin` | Residual add | Decode |

---

## 6. Per-Kernel Benchmark Methodology

### Test structure

Each operator's `test.py` includes `pytest.mark.llama`-tagged test cases using exact Llama 3.2 1B tensor shapes. These are independent from the pre-existing correctness tests.

```bash
# Run all Llama-shape benchmarks
pytest iron/operators/ -m "llama and not extensive" --csv-output=llama_perf.csv -v

# Run a single operator
pytest iron/operators/gemv/ -m "llama" -v
```

### Timing protocol

Each test calls `run_test()` (in `iron/common/test_utils.py`), which does:

1. **5 warmup dispatches** -- `run_runlist()` called without input data to let XRT hardware context settle (avoids cold-start outliers).
2. **20 timed dispatches** -- input buffers are re-written before each dispatch, then `run_runlist()` is called and timed. Reported latency is the average across these 20 runs.

### Latency definition

The timed region covers: `bo.sync(TO_DEVICE)` + `xrt_kernel()` + `run.wait()` + `bo.sync(FROM_DEVICE)` -- the full NPU dispatch round-trip including DMA transfer, not just compute. Correctness is verified after the final dispatch.

### Outer iteration and CSV reporting

By default, each test is repeated 5 times at the pytest level (`--iterations=5`). The `CSVReporter` computes mean/median/min/max/stddev across these 5 outer iterations, each of which internally averages 20 dispatches. This provides both a stable per-test measurement and cross-run variance.

### Test case inventory

**GEMV** (decode, `num_aie_columns=8`): `llama_gqa_q` (2048x2048), `llama_gqa_kv` (512x2048), `llama_gqa_out` (2048x2048), `llama_ffn_fc12` (8192x2048), `llama_ffn_fc3` (2048x8192), `llama_final_vocab` (128256x2048).

**GEMM** (prefill, 8 cols, tile 64x64x64): `llama_kv_proj_13tok` (13x2048x512), `llama_q_out_proj_13tok` (13x2048x2048), `llama_kv_proj_2048tok` (extensive), `llama_final_vocab_2048tok` (extensive).

**RoPE**: `llama_decode_q_32heads` (32 rows), `llama_decode_k_8kvgroups` (8 rows), `llama_prefill_q_16tok` (512 rows), `llama_prefill_k_16tok` (128 rows), plus extensive 2048-token variants.

**MHA**: `llama_prefill_1024tok` (seq_len=1024, 32 heads, 8 pipelines), `llama_prefill_2048tok` (extensive).

**RMSNorm**: `llama_prefill_rms_norm_13tok` (26624 elements, 8 cols), `llama_prefill_rms_norm_2048tok` (extensive).

**ElementwiseAdd**: `llama_prefill_add_13tok` (26624 elements, 8 cols), `llama_prefill_add_2048tok` (extensive).

**SwiGLU Decode**: `llama_swiglu_decode_1x2048x8192`.

**SwiGLU Prefill**: `llama_swiglu_prefill_13tok_2048x8192`, `llama_swiglu_prefill_2048tok_2048x8192` (extensive).

---

## 7. Overhead Analysis

Isolated dispatch latency (from per-kernel benchmarks) is significantly lower than `forward()` latency observed during full inference profiling. The overhead factor varies by operator.

### Three root causes

1. **`sys.setprofile()` overhead.** The Python profiler hooks every function call, adding microseconds per call. For fast operators called thousands of times, this inflates measured durations significantly.

2. **`forward()` Python overhead.** The operator's `forward()` method does tensor reshaping, `torch_to_numpy()` conversion, KV cache indexing, and shape validation -- none of which appear in the isolated `run_test()` path.

3. **Memory pressure.** In full inference, all 16 transformer blocks are live simultaneously, competing for DDR bandwidth. XRT's scheduler also shows variable dispatch latency (profiler has recorded max values orders of magnitude above median for `run_runlist`).

Operators with large buffers and many fused dispatches (e.g., SwiGLU decode) show the smallest overhead factor because per-dispatch Python overhead is negligible relative to total time. Small, fast operators (RoPE, Add, RMSNorm) show the largest overhead factors.

---

## 8. Operator Test Constraints

### MHA minimum seq_len

`seq_len` must yield `num_q_block_per_pipeline >= 2`. With `num_pipelines=8` and `B_q=64`, the minimum valid `seq_len` is `2 * 64 * 8 = 1024`. Using `seq_len=512` produces correct results for head 0 only; subsequent heads are corrupted by a padding logic interaction when `num_q_block_per_pipeline=1`.

The xclbin filename now includes all dimensions (`mha_{heads}h_{kv}kv_{seq}s_{dim}d.xclbin`) to prevent cache collisions across MHA configurations.

### RoPE divisibility

`angle_rows` (= prompt_len) must be divisible by `num_aie_columns`. The minimum valid prompt length for 8 columns is 8. The Llama app's actual `prompt_len=13` does not satisfy this -- it pads internally before dispatching. Benchmark tests use `prompt_len=16` as the minimum that satisfies `16 % 8 == 0`.

### SwiGLU decode DMA artifact

With `hidden_dim=8192`, 4 elements at fixed indices [6183, 6786, 6982, 7116] in the 8192-element intermediate buffer consistently read as zero regardless of input. The test suppresses these with `max_intermediate_errors=4`. This does not affect final output correctness -- the downstream `w_down` matmul is verified against a re-derived reference computed from the intermediate buffer read-back.

### SwiGLU prefill correctness skipped

The test measures latency but does not assert numerical correctness. With `prio_accuracy=True` and `K=2048`, the cascaded verification (GEMM -> SiLU -> GEMM) amplifies errors beyond achievable bfloat16 precision at this scale.

### GEMM archive naming conflict

The kernel archive filename does not encode `prio_accuracy` or `emulate_bf16_mmul_with_bfp16`. All GEMM-using tests must use `prio_accuracy=True` to avoid archive conflicts. Mixing `prio_accuracy=True` and `False` in the same session with the same tile dimensions causes a linker error (`undefined symbol: zero_f32` or `zero_bf16`).

### GEMM partition_N correctness issue

`partition_N=4` with small M (e.g., M=13 padded to 256) has a pre-existing correctness issue in C_2/C_3 partitions. The final vocab GEMM benchmark is only included in the extensive (M=2048) test.
