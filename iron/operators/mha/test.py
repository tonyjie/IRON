#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path


from iron.operators.mha.op import AIEMHA
from iron.operators.mha.reference import generate_golden_reference
from iron.common.test_utils import run_test, report_precision


def generate_test_params(extensive=False):
    # (seq_len, head_dim, heads, number_of_pipeline, num_kv_heads)

    names = []

    params = [(16384, 64, 1, 8, 0)]

    if extensive:
        params += [
            (4096, 64, 8, 8, 4),
            (4096, 64, 8, 8, 2),
            (4096, 64, 8, 8, 0),
        ]

    for seq_len, head_dim, heads, number_of_pipeline, num_kv_heads in params:
        names += [
            f"mha_{seq_len}_{head_dim}_{heads}_{number_of_pipeline}_{num_kv_heads}"
        ]

    return params, names


regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

# Llama 3.2 1B MHA prefill configurations
# num_heads=32, head_dim=64, num_kv_heads=0 (treated as regular MHA), num_of_pipelines=8
# (seq_len, dim, num_heads, num_pipelines, num_kv_heads)
llama_params = [
    # seq_len=512 produces correctness failures with num_heads=32 (num_q_block_per_pipeline=1
    # causes multi-head layout issues similar to the original seq_len < B_q*num_pipelines bug).
    # Use seq_len=1024 which gives num_q_block_per_pipeline=2, the minimum valid value.
    (1024, 64, 32, 8, 0),
]
llama_names = ["llama_prefill_1024tok"]
llama_extensive_params = [
    # Full prefill (prompt_len=2048)
    (2048, 64, 32, 8, 0),
]
llama_extensive_names = ["llama_prefill_2048tok"]

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
    Correlation=r"corr: (?P<value>[\d\.]+)",
    MaxError=r"max_err: (?P<value>[\d\.]+)",
    MeanError=r"mean_err: (?P<value>[\d\.]+)",
    FailRate4Pct=r"4%-fail: (?P<value>[\d\.]+)%",
)
@pytest.mark.parametrize("seq_len,dim,num_heads,num_pipelines,num_kv_heads", all_params)
def test_mha(
    seq_len: int,
    dim: int,
    num_heads: int,
    num_pipelines: int,
    num_kv_heads: int,
    aie_context,
):

    print(
        f"\nTest configuration: seq_len={seq_len}, dim={dim}, num_heads={num_heads}, num_pipelines={num_pipelines}, num_kv_heads={num_kv_heads}"
    )

    golden_ref = generate_golden_reference(
        S_q=seq_len,
        S_kv=seq_len,
        d=dim,
        heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_pipeline=num_pipelines,
    )

    operator = AIEMHA(
        num_heads=num_heads,
        seq_len=seq_len,
        d=dim,
        num_KV_heads=num_kv_heads,
        num_of_pipelines=num_pipelines,
        context=aie_context,
    )

    input_buffers = {
        "Q": golden_ref["Q"].flatten(),
        "K": golden_ref["K"].flatten(),
        "V": golden_ref["V"].flatten(),
    }
    output_buffers = {"O": golden_ref["O"].flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=4.0e-2, abs_tol=1.5e-1
    )

    error_threshold = 0.005
    max_acceptable_errors = int(seq_len * dim * num_heads * error_threshold)

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    num_errors = len(errors.get("O", []))
    print(f"({num_errors} errors out of {max_acceptable_errors} max allowable)")

    report_precision(operator, "O", golden_ref["O"].flatten())

    assert (
        num_errors <= max_acceptable_errors
    ), f"Test failed with {num_errors} errors (max allowable: {max_acceptable_errors})"
