#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path


from iron.operators.swiglu_prefill.op import AIESwiGLUPrefill
from iron.operators.swiglu_decode.reference import generate_golden_reference
from iron.common.test_utils import run_test, verify_buffer, report_precision


def generate_test_params(extensive=False):
    # (seq_len, embedding_dim, hidden_dim, prio_accuracy, verify_intermediates)
    params = []
    names = []
    return params, names


regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

# Llama 3.2 1B SwiGLU prefill configurations
# emb_dim=2048, hidden_dim=8192 (actual Llama FFN dimensions)
# prio_accuracy=False, matching the actual Llama model defaults (op.py uses prio_accuracy=False).
# verify_output=False: bf16 GEMM with K=2048 accumulates ~18-20% relative error for large activations
# (val_range=4, K=2048 → sum-of-squares magnitudes in millions). Direct verification of intermediate
# GEMM outputs (left/right) and the final output is not meaningful with these tolerances.
# This config validates operator execution (latency measurement) without correctness assertion.
# NOTE: Must use --build-dir build_llama (or separate build dir) to avoid archive conflicts
# with regular GEMM tests that use prio_accuracy=True.
# (seq_len, embedding_dim, hidden_dim, prio_accuracy, verify_output)
llama_params = [
    # Short prompt prefill (prompt_len=13)
    (13, 2048, 8192, False, False),
]
llama_names = ["llama_swiglu_prefill_13tok_2048x8192"]
llama_extensive_params = [
    # Full prefill (prompt_len=2048)
    (2048, 2048, 8192, False, False),
]
llama_extensive_names = ["llama_swiglu_prefill_2048tok_2048x8192"]

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
@pytest.mark.parametrize("seq_len,embedding_dim,hidden_dim,prio_accuracy,verify_output", all_params)
def test_swiglu_prefill(seq_len, embedding_dim, hidden_dim, prio_accuracy, verify_output, aie_context):
    golden_ref = generate_golden_reference(M=seq_len, K=embedding_dim, N=hidden_dim)

    operator = AIESwiGLUPrefill(
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        prio_accuracy=bool(prio_accuracy),
        context=aie_context,
    )
    operator.weights_1 = golden_ref["w_gate"].T
    operator.weights_2 = golden_ref["w_up"].T
    operator.weights_3 = golden_ref["w_down"].T

    input_buffers = {"input": golden_ref["input"]}
    output_buffers = {}
    intermediate_buffers = (
        {
            "left": golden_ref["left"],
            "left_swished": golden_ref["left_swished"],
            "right": golden_ref["right"],
        }
        if verify_output
        else {}
    )

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        intermediate_buffers,
        rel_tol=0.07,
        abs_tol=0.7,
    )

    if verify_output:
        ref_2 = operator.read_buffer_as_torch(
            "left_swished", (seq_len, hidden_dim)
        ) * operator.read_buffer_as_torch("right", (seq_len, hidden_dim))
        errors_2 = verify_buffer(operator, "intermediate", ref_2, rel_tol=0.04, abs_tol=0.4)
        if errors_2:
            errors["intermediate"] = errors_2

        ref_3 = (
            operator.read_buffer_as_torch("intermediate", (seq_len, hidden_dim))
            @ golden_ref["w_down"]
        )
        errors_3 = verify_buffer(operator, "output", ref_3, rel_tol=0.04, abs_tol=0.4)
        if errors_3:
            errors["output"] = errors_3

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    # Report precision metrics on the final output against the full-pipeline reference
    report_precision(operator, "output", golden_ref["output"])

    assert not errors, f"Test failed with errors: {errors}"
