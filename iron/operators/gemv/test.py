#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path


from iron.operators.gemv.op import AIEGEMV
from iron.operators.gemv.reference import generate_golden_reference
from iron.common.test_utils import run_test, report_precision


def generate_test_params(extensive=False):
    # fmt: off
    # (M, K, num_aie_columns, tile_size_input, tile_size_output, is_mv)
    params = [
        (128,    128, 1, 32,   128, True),
        (2048,  8192, 1,  1,  2048, True),
        (8192,  2048, 1,  4,  1024, True),
        (2048,  8192, 2,  1,  1024, True),
        (8192,  2048, 2,  4,  1024, True),
        (2048,  8192, 4,  1,   512, True),
        (8192,  2048, 4,  4,  1024, True),
        (2048,  8192, 8,  1,   256, True),
        (8192,  2048, 8,  4,  1024, True),
    ]
    names = [
        f"matrix_vector_mul_{M}x{K}_{tile_size_input}tsi_{tile_size_output}tso_{num_aie_columns}col"
        for M, K, num_aie_columns, tile_size_input, tile_size_output, is_mv in params
    ]
    # fmt: on
    return params, names


# Llama 3.2 1B decode configurations
# emb_dim=2048, hidden_dim=8192, num_heads=32, num_kv_groups=8, head_dim=64, kv_out_dim=512, vocab_size=128256
llama_params = [
    # (M, K, num_aie_columns, tile_size_input, tile_size_output, is_mv)
    # GQA Q projection: M=emb_dim, K=emb_dim, tile_out=emb_dim//16
    (  2048,   2048, 8,  1,  128, False),
    # GQA K/V projection: M=kv_out_dim, K=emb_dim, tile_out=kv_out_dim//16
    (   512,   2048, 8,  1,   32, False),
    # GQA output projection: M=emb_dim, K=emb_dim, tile_out=emb_dim//16
    (  2048,   2048, 8,  1,  128, False),
    # FFN fc1/fc2 (up/gate): M=hidden_dim, K=emb_dim, tile_out=hidden_dim//16
    (  8192,   2048, 8,  1,  512, False),
    # FFN fc3 (down): M=emb_dim, K=hidden_dim, tile_out=emb_dim//16
    (  2048,   8192, 8,  1,  128, False),
    # Final vocab projection: M=vocab_size, K=emb_dim, is_mv=True
    (128256,   2048, 8,  4,   32, True),
]
llama_names = [
    "llama_gqa_q",
    "llama_gqa_kv",
    "llama_gqa_out",
    "llama_ffn_fc12",
    "llama_ffn_fc3",
    "llama_final_vocab",
]

regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

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
    "M,K,num_aie_columns,tile_size_input,tile_size_output,is_mv", all_params
)
def test_gemv(M, K, num_aie_columns, tile_size_input, tile_size_output, is_mv, aie_context):
    golden_ref = generate_golden_reference(M=M, K=K)

    operator = AIEGEMV(
        M=M,
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size_input=tile_size_input,
        tile_size_output=tile_size_output,
        is_mv=is_mv,
        context=aie_context,
    )

    input_buffers = {"matrix": golden_ref["A"].flatten(), "vector": golden_ref["B"]}
    output_buffers = {"output": golden_ref["C"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-3
    )

    print(f"\nLatency (us): {latency_us:.1f}")

    gflops = (2.0 * M * K) / (latency_us * 1e-6) / 1e9
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    report_precision(operator, "output", golden_ref["C"])

    assert not errors, f"Test failed with errors: {errors}"
