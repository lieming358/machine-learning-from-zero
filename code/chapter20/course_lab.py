"""Chapter 20 lab: measure, estimate memory, quantize, and simulate a queue."""

from __future__ import annotations

from time import perf_counter

import numpy as np


def amdahl_speedup(accelerated_fraction: float, local_speedup: float) -> float:
    return 1.0 / ((1.0 - accelerated_fraction) + accelerated_fraction / local_speedup)


def symmetric_quantize(x: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    maximum_integer = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(x)) / maximum_integer
    integer = np.clip(np.round(x / scale), -maximum_integer, maximum_integer)
    return integer * scale, float(scale)


def transformer_memory_gib(
    parameter_count: int,
    bytes_per_weight: int,
    layers: int,
    kv_heads: int,
    head_dimension: int,
    cached_tokens: int,
    concurrency: int = 1,
) -> tuple[float, float]:
    weight = parameter_count * bytes_per_weight / 1024**3
    kv = (
        2 * layers * kv_heads * head_dimension * cached_tokens * concurrency * bytes_per_weight
        / 1024**3
    )
    return weight, kv


def benchmark_matmul(size: int = 256, repeats: int = 5) -> tuple[float, float]:
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(size, size)), rng.normal(size=(size, size))
    _ = a @ b  # warm-up
    durations = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = a @ b
        durations.append(perf_counter() - start)
    operations = 2 * size**3
    gflops = operations / np.median(durations) / 1e9
    return float(gflops), float(np.linalg.norm(result))


def simulate_single_server(arrival_rate: float, service_rate: float, count: int, seed: int):
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1.0 / arrival_rate, count))
    service = rng.exponential(1.0 / service_rate, count)
    finish = 0.0
    latency = np.empty(count)
    for index in range(count):
        start = max(arrivals[index], finish)
        finish = start + service[index]
        latency[index] = finish - arrivals[index]
    return latency


def main() -> None:
    rng = np.random.default_rng(41)
    weights = rng.normal(size=100_000)
    int8, scale8 = symmetric_quantize(weights, 8)
    int4, scale4 = symmetric_quantize(weights, 4)
    error8 = np.mean((weights - int8) ** 2)
    error4 = np.mean((weights - int4) ** 2)

    weight_gib, kv_gib = transformer_memory_gib(
        parameter_count=1_000_000_000,
        bytes_per_weight=2,
        layers=32,
        kv_heads=8,
        head_dimension=128,
        cached_tokens=4096,
        concurrency=8,
    )
    measured_gflops, checksum = benchmark_matmul()
    low_load = simulate_single_server(40, 100, 8000, seed=3)
    high_load = simulate_single_server(90, 100, 8000, seed=3)

    print(f"amdahl_speedup={amdahl_speedup(0.6, 4):.3f}")
    print(f"int8_scale={scale8:.5f}, int4_scale={scale4:.5f}")
    print(f"quantization_mse_int8={error8:.6f}, int4={error4:.6f}")
    print(f"weight_gib={weight_gib:.3f}, kv_gib={kv_gib:.3f}")
    print(f"measured_matmul_gflops={measured_gflops:.2f}, checksum={checksum:.2f}")
    print(f"queue_p99_low={np.quantile(low_load, .99):.4f}, high={np.quantile(high_load, .99):.4f}")

    assert np.isclose(amdahl_speedup(0.6, 4), 1.818181818, rtol=1e-6)
    assert error8 < error4
    assert weight_gib > 1.8 and kv_gib > 0
    assert np.quantile(high_load, 0.99) > np.quantile(low_load, 0.99)
    assert np.isfinite(checksum)


if __name__ == "__main__":
    main()
