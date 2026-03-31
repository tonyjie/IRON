# Llama 3.2 1B Decode Profiling Results

## Configuration

- **Prefill:** prompt_len=2048
- **Decode:** num_tokens=100
- **Model:** Llama 3.2 1B, bfloat16, 16 layers, 32 heads, 8 KV groups, head_dim=64
- **Device:** Ryzen AI NPU (npu2)

## 1. End-to-End Timing

```
Total time:      36.58 s
Prefill time:     2.87 s
Decode time:     33.71 s  (100 tokens)
Tokens/second:    2.94
Time/token:       0.370 s  (= 370 ms/token)
```

## 2. Per-Function Profiling Breakdown

Profile from `--profile` flag captures all function calls with `sys.setprofile()`.

### Decode-Only Operators

These operators are ONLY called during decode (not prefill), so their stats
represent pure decode cost:

| Function | Calls | Total (s) | Avg/call (ms) | Median (ms) | Notes |
|---|---|---|---|---|---|
| `swiglu_decode.forward` | 1,584 | 11.95 | 7.54 | 7.00 | 99 tokens x 16 blocks |
| `gemv.forward` | 6,435 | 9.52 | 1.48 | 1.25 | Q/K/V/out proj + FFN + final vocab |

### Shared Operators (decode-dominated)

These operators are called during both prefill (16 calls) and decode (1584-3200
calls). Decode dominates by call count.

| Function | Calls | Total (s) | Avg/call (ms) | Median (ms) | Decode calls |
|---|---|---|---|---|---|
| `elementwise_add.forward` | 3,200 | 3.31 | 1.03 | 0.87 | 3,168 (99%) |
| `rms_norm.forward` | 3,300 | 3.01 | 0.91 | 0.76 | 3,267 (99%) |
| `rope.forward` | 3,200 | 2.18 | 0.68 | 0.52 | 3,168 (99%) |

### Prefill-Only Operators (for reference)

| Function | Calls | Total (s) | Avg/call (ms) |
|---|---|---|---|
| `swiglu_prefill.forward` | 16 | 0.92 | 57.3 |
| `gemm.forward` | 65 | 0.77 | 11.8 |
| `mha.forward` | 16 | 0.59 | 36.9 |

### Composite Breakdown

| Function | Calls | Total (s) | Avg/call (ms) | Notes |
|---|---|---|---|---|
| `model.forward` | 100 | 36.14 | 361.4 | 1 prefill + 99 decode |
| `transformer.forward` | 1,600 | 33.55 | 21.0 | Avg decode block: ~17.7 ms |
| `gqa.forward` | 1,600 | 13.85 | 8.66 | Avg decode GQA: ~6.9 ms |
| `feed_forward.forward` (decode) | 1,584 | 12.03 | 7.59 | Avg decode FFN: ~7.0 ms |
| `run_runlist` (all dispatches) | 17,816 | 24.14 | 1.36 | NPU kernel time only |

### Estimated Decode-Only Per-Token Breakdown

Decode time per token = 370 ms. Approximate split (from profiler medians):

| Component | Per-token (ms) | % of token | Source |
|---|---|---|---|
| GQA (4x GEMV + 2x RoPE + CPU attn + GEMV out) | ~110 | 30% | gqa.forward median x16 |
| SwiGLU Decode (fused 5 dispatches) | ~112 | 30% | swiglu_decode.forward median x16 |
| RMSNorm (3 instances) | ~37 | 10% | rms_norm.forward median x33 |
| ElementwiseAdd (2 instances) | ~28 | 8% | add.forward median x32 |
| RoPE (2 instances) | ~17 | 5% | rope.forward median x32 |
| Final norm + final GEMV | ~11 | 3% | 1x rms_norm + 1x gemv |
| Python/profiler overhead | ~55 | 15% | difference to 370 ms |

## 3. Decode Dispatch Trace Per Transformer Block

```
RMSNorm (pre-norm, 1 col)        --> rms_norm xclbin (size=2048)
GEMV (Q projection)              --> gemv xclbin (M=2048, K=2048)
GEMV (K projection)              --> gemv xclbin (M=512, K=2048)
GEMV (V projection)              --> gemv xclbin (M=512, K=2048)
RoPE (Q, 1 col)                  --> rope xclbin (rows=32, cols=64, angle_rows=1)
RoPE (K, 1 col)                  --> rope xclbin (rows=8, cols=64, angle_rows=1)
CPU Attention (QK->softmax->*V)  --> no xclbin (runs on host CPU)
GEMV (output projection)         --> gemv xclbin (M=2048, K=2048)
ElementwiseAdd (residual, 1 col) --> add xclbin (size=2048)
RMSNorm (post-norm, 1 col)       --> rms_norm xclbin (size=2048)
SwiGLU Decode (fused FFN)        --> combined xclbin (5 runlist entries)
ElementwiseAdd (residual, 1 col) --> add xclbin (size=2048)
```

Plus per token: 1x final RMSNorm + 1x final GEMV (vocab projection).

~12 NPU dispatches/block x 16 blocks + 2 final = ~194 NPU dispatches per decode token.

## 4. Standalone Kernel Benchmarks (Decode Shapes)

Run with: `pytest iron/operators/ -m "llama and not extensive" --build-dir build_llama --iterations 1 -v -s`

### GEMV (decode projections)

| Test | Shape (MxK) | Role | Latency (us) | Throughput (GFLOP/s) | Corr | Max Err | 4%-fail |
|---|---|---|---|---|---|---|---|
| `llama_gqa_q` | 2048x2048 | Q projection | 213.7 | 39.2 | 1.00000 | 0.00 | 0.0% |
| `llama_gqa_kv` | 512x2048 | K/V projection | 97.9 | 21.4 | 1.00000 | 0.00 | 0.0% |
| `llama_gqa_out` | 2048x2048 | Output projection | 214.3 | 39.1 | 1.00000 | 0.00 | 0.0% |
| `llama_ffn_fc12` | 8192x2048 | FFN gate/up | 656.8 | 51.1 | 1.00000 | 0.12 | 0.0% |
| `llama_ffn_fc3` | 2048x8192 | FFN down | 660.1 | 50.8 | 1.00000 | 0.00 | 0.0% |
| `llama_final_vocab` | 128256x2048 | Final vocab | 9442.9 | 55.6 | 1.00000 | 4.00 | 0.0% |

### RoPE (decode, single token)

| Test | Shape (rows x cols) | Latency (us) | Corr | Max Err | 4%-fail |
|---|---|---|---|---|---|
| `llama_decode_q_32heads` | 32x64 (angle_rows=1) | 39.0 | 1.00000 | 0.00 | 0.0% |
| `llama_decode_k_8kvgroups` | 8x64 (angle_rows=1) | 37.5 | 1.00000 | 0.00 | 0.0% |

### RMSNorm (decode, single token)

| Test | Shape | Latency (us) | Bandwidth (GB/s) | Corr | Max Err | 4%-fail |
|---|---|---|---|---|---|---|
| `llama_decode_rms_norm` | 2048 elems, 1 col, weighted | 41.6 | 0.20 | 0.99999 | 0.06 | 0.0% |

### ElementwiseAdd (decode, single token)

| Test | Shape | Latency (us) | Bandwidth (GB/s) | Corr | Max Err | 4%-fail |
|---|---|---|---|---|---|---|
| `llama_decode_add` | 2048 elems, 1 col | 44.3 | 0.28 | 0.99998 | 0.03 | 0.0% |

### SwiGLU Decode (fused FFN)

| Test | Shape | Latency (us) | Corr | 4%-fail |
|---|---|---|---|---|
| `llama_swiglu_decode_1x2048x8192` | emb=2048, hidden=8192 | 4765.6 | 0.99999 | 3.1% |

## 5. Standalone vs In-Model Comparison

| Operator | Standalone (us) | In-model avg (us) | Overhead |
|---|---|---|---|
| GEMV (Q proj, 2048x2048) | 214 | ~1,480 (avg across all GEMV) | ~6.9x |
| SwiGLU Decode | 4,766 | ~7,540 | ~1.6x |
| RMSNorm (decode) | 42 | ~910 | ~22x |
| ElementwiseAdd (decode) | 44 | ~1,030 | ~23x |
| RoPE (decode) | 39 | ~680 | ~17x |

The large overhead for small operators (RMSNorm, Add, RoPE) is expected:
- `sys.setprofile()` adds ~10-20 us per function call/return
- Python `forward()` wraps each NPU dispatch with buffer management, shape checks, numpy conversion
- The standalone test `run_test()` amortizes overhead across 20 timed iterations

For the larger operators (SwiGLU Decode, GEMV), the NPU compute time dominates
and the overhead factor is much smaller.

## 6. Decode Bottleneck Analysis

**Decode is dominated by two operators:**
- **GQA** (30% of token time): 4 GEMV dispatches + 2 RoPE + CPU attention + 1 GEMV out
- **SwiGLU Decode FFN** (30% of token time): fused 5-dispatch runlist (2 GEMV + SiLU + Mul + GEMV)

**The GEMV kernel is the critical path.** At Llama decode shapes:
- FFN fc1/fc2 (8192x2048): 657 us standalone, called 2x per block = 1.3 ms
- FFN fc3 (2048x8192): 660 us standalone, called 1x per block = 0.66 ms
- Q/out proj (2048x2048): 214 us standalone, called 2x per block = 0.43 ms
- K/V proj (512x2048): 98 us standalone, called 2x per block = 0.20 ms
- Final vocab (128256x2048): 9.4 ms standalone, called 1x per token

Total standalone GEMV per token: (1.3 + 0.66 + 0.43 + 0.20) x 16 blocks + 9.4 = 51 ms
Total standalone all kernels per token: ~51 + 16*(0.04+0.04+0.04+0.04+4.8) = ~130 ms
Measured decode per token: 370 ms

The ~240 ms gap is Python/profiler overhead (~55 ms) + CPU attention + buffer sync.
