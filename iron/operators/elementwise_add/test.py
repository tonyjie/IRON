#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path


from iron.operators.elementwise_add.op import AIEElementwiseAdd
from iron.operators.elementwise_add.reference import generate_golden_reference
from iron.common.test_utils import run_test, report_precision


def generate_test_params(extensive=False):
    max_aie_columns = 8
    num_channels = 2
    input_lengths = [2048] if not extensive else [1024, 4096, 8192]

    params = []
    names = []
    for input_length in input_lengths:
        for num_aie_columns in range(1, max_aie_columns + 1):
            tile_size = input_length // num_aie_columns
            if tile_size * num_aie_columns != input_length:
                continue
            names.append(
                f"eltwise_add_{num_aie_columns}_cols_{num_channels}_channels_{input_length}_tile_{tile_size}"
            )
            params.append((input_length, num_aie_columns, num_channels, tile_size))
    return params, names


regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

# Llama 3.2 1B residual add configurations
# Decode: size=emb_dim=2048, num_aie_columns=1, num_channels=2, tile_size=2048
# Prefill: size=prompt_len*emb_dim, num_aie_columns=8, num_channels=2, tile_size=2048
# (input_length, num_aie_columns, num_channels, tile_size)
llama_params = [
    # Decode: single token × emb_dim=2048 (1 AIE column, tile_size=2048)
    (2048, 1, 2, 2048),
    # Prefill: 13 tokens × emb_dim=2048 (size=26624, tile_size=emb_dim=2048)
    (26624, 8, 2, 2048),
]
llama_names = ["llama_decode_add", "llama_prefill_add_13tok"]
llama_extensive_params = [
    # Prefill: 2048 tokens × emb_dim=2048 (size=4194304, tile_size=emb_dim=2048)
    (4194304, 8, 2, 2048),
]
llama_extensive_names = ["llama_prefill_add_2048tok"]

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
    "input_length,num_aie_columns,num_channels,tile_size",
    all_params,
)
def test_elementwise_add(
    input_length, num_aie_columns, num_channels, tile_size, aie_context
):
    golden_ref = generate_golden_reference(input_length=input_length)

    operator = AIEElementwiseAdd(
        size=input_length,
        num_aie_columns=num_aie_columns,
        num_channels=num_channels,
        tile_size=tile_size,
        context=aie_context,
    )

    input_buffers = {"input1": golden_ref["A"], "input2": golden_ref["B"]}
    output_buffers = {"output": golden_ref["C"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    report_precision(operator, "output", golden_ref["C"])

    assert not errors, f"Test failed with errors: {errors}"
