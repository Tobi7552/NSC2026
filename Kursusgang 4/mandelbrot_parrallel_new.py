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

# Variable list
size = 1024 # Size of image
max_itter = 100

# print(os.cpu_count()) = 8
# print(psutil.cpu_count(logical=False)) = 8

@njit
def mandelbrot_pixel(c_real, c_imag, max_itter):
    z_real = z_imag = 0.0
    pass


@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_itter):
    pass




@njit(fastmath=True)
def compute_mandelbrot_numba(size, itterations):
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








# Needed for multiprocessing
if __name__ == "__main__":
    pass