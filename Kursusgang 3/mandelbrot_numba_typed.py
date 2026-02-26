"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""
# Imports here:
import numpy as np
import matplotlib.pyplot as plt
import time , statistics
import cProfile, pstats
from numba import njit, prange


@njit(fastmath=True)
def compute_mandelbrot_numba_typed(size, itterations, dtype=np.float64):
    xDomain = np.linspace(-2.0, 1.0, size)
    yDomain = np.linspace(-1.5, 1.5, size)
    bound = 2
    max_iterations = itterations
    c = np.empty((size,size), dtype=np.complex128)

    for y in range(size):
        for x in range(size):
            c[y, x] = complex(xDomain[x], yDomain[y])

    z = np.zeros_like(c)
    m = np.zeros((size, size), dtype=np.int32)
    mask = np.ones((size, size),dtype=np.int32)
        
    for i in range(max_iterations):
        """
        We have to change this loop compared to the vectorized version
        as numba doesnt like indexing such as z[mask], cause of boleans(Thanks to ChatGPT for the explanation)
        This fixed my problem of "object mode" where it ran slower than the vectorized version
        I have included the slow version as well, so the difference can be seen.
        """
        for y in range(size):
            for x in range(size):
                if mask[y,x]:
                    z[y, x] = z[y, x]*z[y, x] + c[y, x]
                    if z[y, x].real**2 + z[y, x].imag**2 > bound**2:
                        m[y, x] = i
                        mask[y, x] = 0
        if np.all(mask == 0):
            break
    return m
compute_mandelbrot_numba_typed(1024,100, np.float64)

for dtype in [np.float16, np.float32, np.float64]:
    t0 = time.perf_counter()
    compute_mandelbrot_numba_typed(1024,100, dtype)
    print(f"{dtype.__name__}: {time.perf_counter()-t0:.3f}s")


r16 = compute_mandelbrot_numba_typed(1024,100, np.float16)
r32 = compute_mandelbrot_numba_typed(1024,100, np.float32)
r64 = compute_mandelbrot_numba_typed(1024,100, np.float64)

fig, axes = plt.subplots(1,3,figsize=(12, 4))

for ax, result, title in zip(axes, [r16,r32,r64],["Float16","Float32","Float64 (ref)"]):
    ax.imshow(result,cmap="magma")
    ax.set_title(title); ax.axis=("off")

plt.savefig("Precision_comparisson.png",dpi=150)

print(f"Max diff float32 vs float64: {np.abs(r32-r64).max()}")
print(f"Max diff float16 vs float64: {np.abs(r16-r64).max()}")
