# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
from ml_dtypes import bfloat16
from .utils import torch_to_numpy
import logging


def nearly_equal(
    a, b, rel_tol=128 * np.finfo(np.float32).eps, abs_tol=np.finfo(np.float32).tiny
):
    """
    Compare two floating point numbers for approximate equality.

    Adapted from Stack Overflow, License CC BY-SA 4.0
    Original author: P-Gn
    Source: https://stackoverflow.com/a/32334103
    """
    assert np.finfo(np.float32).eps <= rel_tol
    assert rel_tol < 1.0

    if a == b:
        return True

    diff = abs(float(a) - float(b))
    norm = min(abs(float(a)) + abs(float(b)), np.finfo(np.float32).max)
    return diff < max(abs_tol, rel_tol * norm)


def verify_buffer(operator, buf_name, reference, rel_tol=0.04, abs_tol=1e-6):
    errors = []
    expected_np = torch_to_numpy(reference).reshape((-1,))
    buf_size = operator.buffers[buf_name] // 2
    output = operator.read_buffer(buf_name, (buf_size,))
    if len(output) < len(expected_np):
        # Allow larger buffers - binning may have allocated more space than needed
        print(
            f"Buffer size mismatch for {buf_name}: expected {len(expected_np)}, got {len(output)}"
        )
        errors.extend(i for i in range(abs(len(output) - len(expected_np))))
    compare_len = min(len(output), len(expected_np))
    for i in range(compare_len):
        if not nearly_equal(float(output[i]), float(expected_np[i]), rel_tol, abs_tol):
            errors.append(i)
            if len(errors) <= 10:
                print(
                    f"Mismatch in {buf_name}[{i}]: expected {float(expected_np[i]):.6f}, got {float(output[i]):.6f}"
                )
    return errors


def compute_precision_metrics(npu_output, reference):
    """Compute precision metrics between NPU output and CPU reference.

    Args:
        npu_output: numpy array from NPU (bfloat16 or float)
        reference: numpy array or torch tensor of expected values

    Returns:
        dict with keys: corr, max_err, mean_err, fail_rate_4pct
    """
    ref_np = torch_to_numpy(reference).reshape((-1,)).astype(np.float32)
    out_np = np.array(npu_output, dtype=np.float32).reshape((-1,))

    compare_len = min(len(out_np), len(ref_np))
    out_np = out_np[:compare_len]
    ref_np = ref_np[:compare_len]

    abs_err = np.abs(out_np - ref_np)
    max_err = float(np.max(abs_err))
    mean_err = float(np.mean(abs_err))

    # Pearson correlation
    if np.std(out_np) > 0 and np.std(ref_np) > 0:
        corr = float(np.corrcoef(out_np, ref_np)[0, 1])
    else:
        corr = 1.0 if np.allclose(out_np, ref_np) else 0.0

    # Percentage of elements failing at 4% relative tolerance
    ref_abs = np.abs(ref_np)
    threshold = np.maximum(0.04 * (np.abs(out_np) + ref_abs), np.finfo(np.float32).tiny)
    fail_mask = abs_err >= threshold
    fail_rate_4pct = float(np.sum(fail_mask)) / max(compare_len, 1) * 100.0

    return {
        "corr": corr,
        "max_err": max_err,
        "mean_err": mean_err,
        "fail_rate_4pct": fail_rate_4pct,
    }


def print_precision_metrics(metrics):
    """Print precision metrics in a standardized format for CSV reporter parsing.

    Args:
        metrics: dict returned by compute_precision_metrics()
    """
    print(
        f"Precision — corr: {metrics['corr']:.5f}, "
        f"max_err: {metrics['max_err']:.2f}, "
        f"mean_err: {metrics['mean_err']:.4f}, "
        f"4%-fail: {metrics['fail_rate_4pct']:.1f}%"
    )


def report_precision(operator, buf_name, reference):
    """Read an operator buffer, compute precision metrics, and print them.

    Convenience wrapper combining read_buffer + compute_precision_metrics +
    print_precision_metrics. Suitable for calling after run_test().

    Args:
        operator: AIE operator instance with populated buffers
        buf_name: Name of the output buffer to read
        reference: Expected output (numpy array or torch tensor)

    Returns:
        dict with precision metrics (corr, max_err, mean_err, fail_rate_4pct)
    """
    buf_size = operator.buffers[buf_name] // 2
    npu_output = operator.read_buffer(buf_name, (buf_size,))
    ref_np = torch_to_numpy(reference).reshape((-1,))
    metrics = compute_precision_metrics(npu_output[:len(ref_np)], reference)
    print_precision_metrics(metrics)
    return metrics


def run_test(
    operator,
    input_buffers,
    output_buffers,
    intermediate_buffers=None,
    rel_tol=0.04,
    abs_tol=1e-6,
    warmup_iters=5,
    timed_iters=20,
):
    """
    Run operator test with specified input/output/intermediate buffers.

    Args:
        operator: AIE operator instance with registered buffers
        input_buffers: Dict mapping buffer names to input data arrays
        output_buffers: Dict mapping buffer names to reference output arrays
        intermediate_buffers: Optional dict mapping buffer names to reference arrays for validation
        rel_tol: Relative tolerance for comparison of output and intermediate buffers
        abs_tol: Absolute tolerance for comparison of output and intermediate buffers

    Returns:
        (errors: list, latency_us: float, bandwidth_gbps: float)
    """
    if intermediate_buffers is None:
        intermediate_buffers = {}

    # Build operator and prepare runtime
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    operator.context.compile_all()
    operator.context.prepare_runtime()

    # Run warmup iterations before writing to buffers (warmup iters might corrupt the buffers)
    for _ in range(warmup_iters):
        operator.run_runlist()  # warmup run to configure

    # Pre-convert input buffers to numpy once
    input_np = {buf_name: torch_to_numpy(data) for buf_name, data in input_buffers.items()}

    # Run timed iterations, re-writing inputs before each run to avoid
    # corrupted intermediate buffers in multi-stage operators (e.g., SwiGLU)
    elapsed_total = 0
    for _ in range(timed_iters):
        for buf_name in output_buffers:
            buf_size = operator.buffers[buf_name]
            operator.write_buffer(buf_name, np.zeros(buf_size, dtype=np.uint8))
        for buf_name, data_np in input_np.items():
            operator.write_buffer(buf_name, data_np)
        elapsed_total += operator.run_runlist()
    elapsed = elapsed_total / timed_iters
    latency_us = elapsed * 1e6

    # Verify outputs
    errors = {}
    for buf_name, expected in output_buffers.items():
        buf_errors = verify_buffer(operator, buf_name, expected, rel_tol, abs_tol)
        if buf_errors:
            errors[buf_name] = buf_errors

    for buf_name, expected in intermediate_buffers.items():
        buf_errors = verify_buffer(operator, buf_name, expected, rel_tol, abs_tol)
        if buf_errors:
            errors[buf_name] = buf_errors

    # Calculate bandwidth
    input_bytes = sum(operator.buffers[buf_name] for buf_name in input_buffers)
    output_bytes = sum(operator.buffers[buf_name] for buf_name in output_buffers)
    total_bytes = input_bytes + output_bytes
    bandwidth_gbps = total_bytes / (latency_us * 1e-6) / 1e9

    return errors, latency_us, bandwidth_gbps
