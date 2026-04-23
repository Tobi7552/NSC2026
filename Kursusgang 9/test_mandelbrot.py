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
import pytest


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

#########
# Test belows - above is code from earlier lectures
########


@pytest.mark.parametrize("c_real, c_imag, expected", [
    (0.0, 0.0, 100),     # stays bounded
    (2.0, 0.0, 1),       # escapes immediately
    (-1.0, 0.0, 100),    # inside set
])
# test case 1
def test_mandelbrot_pixel_known_values(c_real, c_imag, expected):
    result = mandelbrot_pixel(c_real, c_imag, 100)
    assert result == expected

# test case 2
def test_chunk_shape():
    N = 32
    out = mandelbrot_chunk(0, 8, N, -2, 1, -1.5, 1.5, 50)

    assert out.shape == (8, N)
    assert out.dtype == np.int32

# test 3
def test_serial_equals_chunk():
    N = 32
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 50

    full = mandebrot_serial(N, x_min, x_max, y_min, y_max, max_iter)

    # rebuild using chunks manually
    parts = []
    for r0 in range(0, N, 8):
        r1 = min(r0 + 8, N)
        parts.append(
            mandelbrot_chunk(r0, r1, N, x_min, x_max, y_min, y_max, max_iter)
        )

    rebuilt = np.vstack(parts)

    assert np.allclose(full, rebuilt)

# test 4
def test_dask_matches_serial():
    N = 32
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 50

    serial = mandebrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    dask_result = compute_mandelbrot_dask(
        N, x_min, x_max, y_min, y_max, max_iter, n_chunks=4
    )

    assert np.allclose(serial, dask_result)
