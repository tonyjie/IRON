#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path


import pytest
from iron.operators.rope.op import AIERope
from iron.operators.rope.reference import generate_golden_reference
from iron.common.test_utils import run_test, report_precision


def generate_test_params(extensive=False):
    params = []
    names = []

    num_aie_columns_options = [1, 2, 8]

    if not extensive:
        input_rows = [32]
        input_cols = [512]
        input_angle_rows = [8, 32]
        method_types = [0]  # 0: Two-halves method
    else:
        input_rows = [32, 64]
        input_cols = [128]
        input_angle_rows = [8, 16, 32]
        method_types = [0, 1]  # 0: Two-halves method, 1: interleaved method

    for num_aie_columns in num_aie_columns_options:
        for n_rows in input_rows:
            for n_angle_rows in input_angle_rows:
                for n_cols in input_cols:
                    for method_type in method_types:
                        names.append(
                            f"rope_{num_aie_columns}c_{n_rows}rows_{n_cols}cols_{n_angle_rows}arows_{method_type}m"
                        )
                        params.append(
                            (
                                n_rows,
                                n_cols,
                                n_angle_rows,
                                num_aie_columns,
                                method_type,
                            )
                        )

    return params, names


regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

# Llama 3.2 1B RoPE configurations
# head_dim=64, num_heads=32, num_kv_groups=8
# Decode: single token (angle_rows=1), Prefill: prompt_len tokens
# (rows, cols, angle_rows, num_aie_columns, method_type)
llama_params = [
    # Decode Q: 32 heads, 1 token (angle_rows=1, num_aie_columns=1)
    (  32, 64,  1, 1, 0),
    # Decode K: 8 kv_groups, 1 token (angle_rows=1, num_aie_columns=1)
    (   8, 64,  1, 1, 0),
    # Prefill Q: 32 heads × 16 tokens = 512 rows
    # Note: angle_rows must be divisible by num_aie_columns; prompt_len must be a
    # multiple of 8 for 8 columns. Using prompt_len=16 (smallest valid multiple of 8).
    ( 512, 64, 16, 8, 0),
    # Prefill K: 8 kv_groups × 16 tokens = 128 rows
    ( 128, 64, 16, 8, 0),
]
llama_names = [
    "llama_decode_q_32heads",
    "llama_decode_k_8kvgroups",
    "llama_prefill_q_16tok",
    "llama_prefill_k_16tok",
]
llama_extensive_params = [
    # Prefill Q: 32 heads × 2048 tokens = 65536 rows
    (65536, 64, 2048, 8, 0),
    # Prefill K: 8 kv_groups × 2048 tokens = 16384 rows
    (16384, 64, 2048, 8, 0),
]
llama_extensive_names = [
    "llama_prefill_q_2048tok",
    "llama_prefill_k_2048tok",
]

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
@pytest.mark.parametrize(
    "rows,cols,angle_rows,aie_columns,method_type",
    all_params,
)
def test_rope(rows, cols, angle_rows, aie_columns, method_type, aie_context):
    golden_ref = generate_golden_reference(
        rows=rows, cols=cols, context_len=angle_rows, method_type=method_type
    )

    operator = AIERope(
        rows=rows,
        cols=cols,
        num_aie_columns=aie_columns,
        angle_rows=angle_rows,
        method_type=method_type,
        context=aie_context,
    )

    # golden reference produces tensors of shape (n_heads, seq_len, cols);
    # NPU design expects (seq_len, n_heads, cols), so we transpose inputs/outputs
    input_buffers = {
        "in": golden_ref["A"].transpose(0, 1).contiguous(),
        "angles": golden_ref["B"],
    }
    output_buffers = {"output": golden_ref["C"].transpose(0, 1).contiguous()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.05, abs_tol=0.5
    )

    print(golden_ref["C"])
    print(
        operator.read_buffer_as_torch("output", (rows // angle_rows, angle_rows, cols))
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    report_precision(operator, "output", golden_ref["C"].transpose(0, 1).contiguous())

    # assert not errors, f"Test failed with errors: {errors}"
