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
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster


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
            out[row, col] = mandelbrot_pixel(x_min + col * dx, c_image, max_itter)
    return out

def mandebrot_serial(N, x_min, x_max, y_min, y_max, max_itter = 100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_itter)


def build_chunks(N, n_chunks):
    rows_per_chunk = max(1,N // n_chunks)
    chunks = []

    row = 0
    for _ in range(n_chunks):
        row_end = min(row + rows_per_chunk, N)
        chunks.append((row, row_end))
        row = row_end
        if row >= N:
            break

    return chunks


def compute_mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter, n_chunks):
    chunks = build_chunks(N, n_chunks)

    tasks = [
        delayed(mandelbrot_chunk)(
            r0, r1,
            N, x_min, x_max, y_min, y_max, max_iter
        )
        for (r0, r1) in chunks
        ]

    results = dask.compute(*tasks)
    return np.vstack(results)


if __name__ == "__main__":

    N = 1024
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 100
    n_chunks = 8
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    client.run(lambda: mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 100))
    
    # baseline run
    t0 = time.perf_counter()
    mandebrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    T1 = time.perf_counter() - t0

    chunk_sweep = list(range(2, 64, 2))
    results = []
    p = 8 # worker count
    for n_chunks in chunk_sweep:

        times = []
        for i in range(3):
            t0 = time.perf_counter()
            serial_result = compute_mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter, n_chunks)
            times.append(time.perf_counter()- t0)
        t_median = statistics.median(times)
        speedup = T1 / t_median
        LIF = p * t_median / T1 - 1
        results.append((n_chunks, t_median, speedup, LIF))

    print(f"Dask local {n_chunks} chunks :{statistics.median(times):.3f} s")
    client.close() 
    cluster.close()
    print("n_chunks | time (s) | vs 1x | speedup | LIF")
    print("----------------------------------------------")

    for n, t, s, l in results:
        print(f"{n:8d} | {t:7.3f} | {T1/t:5.2f}x | {s:7.2f} | {l:7.2f}")
