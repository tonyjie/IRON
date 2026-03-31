#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path


from iron.operators.gemm.op import AIEGEMM
from iron.operators.gemm.reference import generate_golden_reference
from iron.common.test_utils import run_test, report_precision


def generate_test_params(extensive=False):
    # fmt: off
    #   M,     K,     N, num_aie_columns, b_col_maj, c_col_maj,   m,   k,   n, trace_size, partition_N, prio_accuracy, emu_bfp16
    params = [
        (2048,  2048,  2048,               1,     False,     False,  64,  64,  64,          0,           1, True, False),
        (2048,  2048,  2048,               2,      True,     False,  64,  64,  64,          0,           1, True, False),
        (2048,  2048,  2048,               8,      True,      True,  64,  64,  64,          0,           1, True, False),
        ( 384,  1536,  1792,               4,      True,     False,  32,  48,  64,          0,           1, True, False),
        (1792,   896,  1152,               8,     False,      True,  64,  32,  48,          0,           1, True, False),
        ( 896,  1792,   640,               8,     False,      True,  32,  64,  80,          0,           1, True, False),
        ( 192,   384,    64,               4,     False,     False,  48,  96,  16,          0,           1, True, False),
        ( 192,   384,    64,               4,      True,      True,  48,  96,  16,          0,           1, True, False),
    ]
    extensive_params = [
        (2048,  2048,  2048,               8,     False,     False,  32,  32, 128,          0,           1, True, False),
        (2048,  2048,  8192,               2,     False,     False,  64,  64,  64,          0,           1, True, False),
        (2048,  8192,  2048,               2,     False,     False,  64,  64,  64,          0,           1, True, False),
        (2048,    64,  2048,               2,     False,     False,  64,  64,  64,          0,           1, True, False),
        (2048,    64,  8192,               2,     False,     False,  64,  64,  64,          0,           1, True, False),
        (2048,  2048,  2048,               8,      True,     False, 128,  32,  32,          0,           1, True, False),
        (2048,  2048,  8192,               2,      True,     False,  64,  64,  64,          0,           1, True, False),
        (2048,  8192,  2048,               2,      True,     False,  64,  64,  64,          0,           1, True, False),
        (2048,    64,  2048,               2,      True,     False,  64,  64,  64,          0,           1, True, False),
        (2048,    64,  8192,               2,      True,     False,  64,  64,  64,          0,           1, True, False),
        (2048,  2048,  2048,               2,     False,      True,   8,  16,  32,          0,           1, True, False),
        (2048,  2048,  8192,               2,     False,      True,  64,  64,  64,          0,           1, True, False),
        (2048,  8192,  2048,               2,     False,      True,  64,  64,  64,          0,           1, True, False),
        (2048,    64,  2048,               2,     False,      True,  64,  64,  64,          0,           1, True, False),
        (2048,    64,  8192,               2,     False,      True,  64,  64,  64,          0,           1, True, False),
    ]
    # fmt: on

    if extensive:
        params = extensive_params

    names = []
    for (
        M, K, N, num_aie_columns, b_col_maj, c_col_maj,
        m, k, n, trace_size, partition_N, _prio_acc, _emu_bfp,
    ) in params:
        name = f"gemm_{M}x{K}x{N}_{m}x{k}x{n}_{num_aie_columns}cols"
        if b_col_maj:
            name += "_bcolmaj"
        if c_col_maj:
            name += "_ccolmaj"
        if partition_N > 1:
            name += f"_{partition_N}npart"
        if trace_size > 0:
            name += f"_{trace_size}trace"
        names.append(name)

    return params, names


regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

# Llama 3.2 1B prefill configurations
# emb_dim=2048, kv_out_dim=512, hidden_dim=8192, vocab_size=128256
# All use: num_aie_columns=8, tile_m=tile_k=tile_n=64
# Llama model defaults: prio_accuracy=False, emulate_bf16_mmul_with_bfp16=True
# Q/K/V/output projections use b_col_maj=False (AIEGEMM default, matching gqa.py config).
# Final vocab GEMM uses b_col_maj=True, partition_N=4 (matching model_with_json.py config).
# NOTE: b_col_maj=True requires --build-dir to avoid cache conflicts with regular tests.
# fmt: off
#                M,     K,      N, cols, b_col_maj, c_col_maj,  m,  k,  n, trace, partition_N, prio_accuracy, emu_bfp16
llama_params = [
    # K/V projection: emb_dim -> kv_out_dim (short prompt_len=13)
    (   13,  2048,    512,    8,     False,     False, 64, 64, 64,     0,  1, False, True),
    # Q/out projection: emb_dim -> emb_dim (short prompt_len=13)
    (   13,  2048,   2048,    8,     False,     False, 64, 64, 64,     0,  1, False, True),
]
llama_names = [
    "llama_kv_proj_13tok",
    "llama_q_out_proj_13tok",
]
llama_extensive_params = [
    # K/V projection: full prefill prompt_len=2048
    ( 2048,  2048,    512,    8,     False,     False, 64, 64, 64,     0,  1, False, True),
    # Q/out projection: full prefill prompt_len=2048
    ( 2048,  2048,   2048,    8,     False,     False, 64, 64, 64,     0,  1, False, True),
    # FFN gate/up projection: emb_dim -> hidden_dim (standalone, for SwiGLU comparison)
    ( 2048,  2048,   8192,    8,     False,     False, 64, 64, 64,     0,  1, False, True),
    # FFN down projection: hidden_dim -> emb_dim (standalone, for SwiGLU comparison)
    ( 2048,  8192,   2048,    8,     False,     False, 64, 64, 64,     0,  1, False, True),
    # Final vocab GEMM: b_col_maj=True, partition_N=4 (matches model_with_json.py)
    ( 2048,  2048, 128256,    8,      True,     False, 64, 64, 64,     0,  4, False, True),
]
llama_extensive_names = [
    "llama_kv_proj_2048tok",
    "llama_q_out_proj_2048tok",
    "llama_ffn_gate_up_2048tok",
    "llama_ffn_down_2048tok",
    "llama_final_vocab_2048tok",
]
# fmt: on

# Combine params with marks - extensive params get pytest.mark.extensive
all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params, regular_names)
] + [
    pytest.param(*params, marks=pytest.mark.extensive, id=name)
    for params, name in zip(extensive_params, extensive_names)
] + [
    pytest.param(*params, marks=pytest.mark.llama, id=name)
    for params, name in zip(llama_params, llama_names)
] + [
    pytest.param(*params, marks=[pytest.mark.llama, pytest.mark.extensive], id=name)
    for params, name in zip(llama_extensive_params, llama_extensive_names)
]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
    Correlation=r"corr: (?P<value>[\d\.]+)",
    MaxError=r"max_err: (?P<value>[\d\.]+)",
    MeanError=r"mean_err: (?P<value>[\d\.]+)",
    FailRate4Pct=r"4%-fail: (?P<value>[\d\.]+)%",
)
@pytest.mark.parametrize(
    "M,K,N,num_aie_columns,b_col_maj,c_col_maj,m,k,n,trace_size,partition_N,prio_accuracy,emu_bfp16",
    all_params,
)
def test_gemm(
    M,
    K,
    N,
    num_aie_columns,
    b_col_maj,
    c_col_maj,
    m,
    k,
    n,
    trace_size,
    partition_N,
    prio_accuracy,
    emu_bfp16,
    aie_context,
):
    golden_ref = generate_golden_reference(
        M=M,
        K=K,
        N=N,
        partition_N=partition_N,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
    )

    operator = AIEGEMM(
        M=M,
        K=K,
        N=N,
        tile_m=m,
        tile_k=k,
        tile_n=n,
        num_aie_columns=num_aie_columns,
        prio_accuracy=prio_accuracy,
        emulate_bf16_mmul_with_bfp16=emu_bfp16,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
        partition_N=partition_N,
        context=aie_context,
    )

    input_buffers = {
        "A": golden_ref["input"].flatten(),
    }
    output_buffers = {}

    # Use looser tolerances for Llama configs (lower precision mode)
    rel_tol = 0.04 if not prio_accuracy else 0.005
    abs_tol = 1.0 if not prio_accuracy else 0.005

    # Create A, B, C dictionaries from the partitioned buffers
    for i in range(partition_N):
        input_buffers[f"B_{i}"] = golden_ref["input_b"][i].flatten()
        output_buffers[f"C_{i}"] = golden_ref["output"][i].flatten()
    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=rel_tol, abs_tol=abs_tol
    )

    gflops = (2.0 * M * K * N) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s")

    # Print precision metrics for each output partition
    for i in range(partition_N):
        metrics = report_precision(operator, f"C_{i}", golden_ref["output"][i].flatten())

    if prio_accuracy:
        assert not errors, f"Test failed"
    else:
        # Lower-precision Llama mode (bfp16 emulation): element-wise errors are expected.
        # Use correlation as the primary quality check instead.
        assert metrics["corr"] >= 0.999, (
            f"Correlation too low: {metrics['corr']:.5f} (expected >= 0.999)"
        )
