"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""

# Imports here:
import numpy as np
import matplotlib.pyplot as plt
import time, os, statistics, psutil
from multiprocessing import Pool
from numba import njit
from pathlib import Path

# print(os.cpu_count()) = 8
# print(psutil.cpu_count(logical=False)) = 8

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_itter):
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_itter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4:
            return i
        else:
            z_imag = 2 * z_real * z_imag + c_imag
            z_real = zr2 - zi2 + c_real
    return max_itter

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_itter):
    out = np.empty((row_end - row_start, N), dtype = np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for row in range(row_end - row_start):
        c_image = y_min + (row + row_start) * dy
        for col in range(N):
            out[row, col] = mandelbrot_pixel(x_min + col * dx, c_image, max_itter=100)
    return out

def mandebrot_serial(N, x_min, x_max, y_min, y_max, max_itter = 100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_itter)

################################################################
##              PARALLEL IMPLEMENTATION                       ##
################################################################
def worker(args):
    return mandelbrot_chunk(*args)

def build_chunks(N, x_min, x_max, y_min, y_max, max_itter, n_chunks):
    rows_per_chunk = max(1,N // n_chunks)
    chunks = []

    for i in range(n_chunks):
        row_start = i * rows_per_chunk

        # last worker takes remainder
        if i == n_chunks - 1:
            row_end = N
        else:
            row_end = (i + 1) * rows_per_chunk

        chunks.append((row_start, row_end, N, x_min, x_max, y_min, y_max, max_itter))

    return chunks


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_itter, workers, n_chunks=None, pool=None):
    n_workers = workers
    if n_chunks is None:
        n_chunks = workers

    chunks = build_chunks(N, x_min, x_max, y_min, y_max, max_itter, n_chunks)
    
    if pool is None:
        with Pool(processes=n_workers) as pool:
            pool.map(worker, chunks) # un-timed warm-up: Numba JIT in workers
            parts = pool.map(worker, chunks)
    else:
        parts = pool.map(worker, chunks)

    return np.vstack(parts)


def benchmark_parallel(N, x_min, x_max, y_min, y_max, max_itter, t_serial):
    max_workers = os.cpu_count()
    results = []
    
    print(f"Serial time (median): {t_serial:.4f} s\n")

    for p in range(1, max_workers + 1):
        chunks = build_chunks(N, x_min, x_max, y_min, y_max, max_itter, p)
        with Pool(processes=p) as pool:
            pool.map(worker, chunks)
            times = []
            for _ in range(3):
                t0 = time.time()
                parts = pool.map(worker, chunks)
                result = np.vstack(parts)
                t1 = time.time()
                times.append(t1 - t0)

        tp = statistics.median(times)

        # metrics
        speedup = t_serial / tp
        LIF = p * tp / t_serial - 1

        results.append((p, tp, speedup, LIF))

        print(
            f"Workers: {p:3d} | Time: {tp:.4f} s | "
            f"Speedup: {speedup:.2f} | LIF: {LIF:.2f}"
        )
    return results

def benchmark_parallel_chunks(N, x_min, x_max, y_min, y_max, max_itter, t_serial):
    p = 8  # fixed best worker count
    results = []

    chunk_sweep = [p, 2*p, 4*p, 8*p, 16*p, 32*p, 64*p]

    with Pool(processes=p) as pool:
        for n_chunks in chunk_sweep:
            chunks = build_chunks(N, x_min, x_max, y_min, y_max, max_itter, n_chunks)
            # warm-up
            pool.map(worker, chunks)
            times = []
            for _ in range(10):
                t0 = time.perf_counter()

                parts = pool.map(worker, chunks)
                result = np.vstack(parts)

                tp = time.perf_counter() - t0
                times.append(tp)

            tp = statistics.median(times)

            # metrics
            speedup = t_serial / tp
            LIF = p * tp / t_serial - 1

            results.append((n_chunks, tp, speedup, LIF))

            print(
                f"Chunks: {n_chunks:3d} | Time: {tp:.4f} s | "
                f"Speedup: {speedup:.2f} | LIF: {LIF:.2f}"
            )

    return results

# Needed for multiprocessing
if __name__ == "__main__":
    N = 1024
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_itter = 100
    max_workers = os.cpu_count()

    mandebrot_serial(N, x_min, x_max, y_min, y_max, max_itter) # warmup

    serial_times = []
    for i in range(3):
        t0 = time.time()
        serial_result = mandebrot_serial(N, x_min, x_max, y_min, y_max, max_itter)
        t1 = time.time()
        serial_times.append(t1 - t0)


    t_serial = statistics.median(serial_times)
    print("Median time serial:", t_serial)

    # Compare parellel array to serial
    test_parallel = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_itter, workers=4, n_chunks=16)

    print(np.array_equal(serial_result,test_parallel))
    result_workers = benchmark_parallel(N, x_min, x_max, y_min, y_max, max_itter, t_serial)
    print("\n")
    results_chunks = benchmark_parallel_chunks(N, x_min, x_max, y_min, y_max, max_itter, t_serial)

    