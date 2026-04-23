"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""

# Imports here:
import numpy as np
from numba import njit



@njit(cache=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_itter: int) -> int:
    """
    Computes escape count for a single complex point

    Parameters
    ------------------
    c_real: float
        real part of the complex number c
    c_imag: float
        imaginary part of the complex number c
    max_iter: int
        maximum itteration count
    ------------------
    
    Returns
    ---------
    int
        number of itterations before divergene or the max itter if bounder
    """
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
def mandelbrot_chunk(row_start: int, row_end: int, N: int, x_min: float, x_max: float, y_min: float, y_max: float, max_itter: int) -> np.ndarray:
    """
    Computes a chunk of the mandelbrot set

    Parameters
    ----------
    row_start: int
        Starting row index
    row_end: int
        Ending row index
    N: int
        Total grid size
    x_min: float
        Minimum real value
    x_max: float
        Maximum real value
    y_min: float
        Minimum imaginary value
    y_max: float
        Maximum imaginary value
    max_iter: int
        Maximum number of iterations
    -----------
    
    Returns
    ---------
    np.ndarray
        2D array of the shape (row_end - row_start, N) containing the iteration counts
    """
    out = np.empty((row_end - row_start, N), dtype = np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for row in range(row_end - row_start):
        c_image = y_min + (row + row_start) * dy
        for col in range(N):
            out[row, col] = mandelbrot_pixel(x_min + col * dx, c_image, max_itter)
    return out

def mandebrot_serial(N: int, x_min: float, x_max: float, y_min: float, y_max: float, max_itter: int = 100) -> np.ndarray:
    """
    Computes the full mandelbrot set using the serial approach

    Parameters
    -----------
    N: int
        Total grid size
    x_min: float
        Minimum real value
    x_max: float
        Maximum real value
    y_min: float
        Minimum imaginary value
    y_max: float
        Maximum imaginary value
    max_iter: int
        Maximum number of iterations
    -------------

    Returns
    --------
    np.ndarray
        2D array of the shape (N,N) containing the iteration count
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_itter)