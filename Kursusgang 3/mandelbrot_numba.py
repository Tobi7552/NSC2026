"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""
# Imports here:
import numpy as np
import matplotlib.pyplot as plt
import time , statistics
from numba import njit, prange

def compute_mandelbrot_vectorized(size, itterations, plot=False):
    start = time.time()
    xDomain, yDomain = np.linspace(-2, 1,size), np.linspace(-1.5, 1.5,size)
    max_iterations = itterations
    bound = 2
    colormap = "magma"

    x, y = np.meshgrid(xDomain, yDomain)
    c = x + 1j * y
    z = np.zeros_like(c)
    m = np.zeros(c.shape, dtype=int)
    mask = np.ones(c.shape, dtype=bool)
    # Implement Mandelbrot logic here
    for i in range(max_iterations):
            z[mask] = z[mask]**2 + c[mask]
            mask = (np.abs(z) <= bound)
            m[~mask & (m == 0)] = i # Le chat helped me with this
                                    # for the mask update
            if not np.any(mask):
                break

    elapsed_vectorized = time.time() - start
    #print(f"Time elapsed using the vectorized imementation is:{elapsed_vectorized} seconds")
    if plot:
        ax = plt.axes()
        ax.set_aspect('equal')
        graph = ax.pcolormesh(xDomain, yDomain, m, cmap = colormap)
        plt.colorbar(graph)
        plt.xlabel("Real-Axis")
        plt.ylabel("Imaginary-Axis")
        plt.title('Multibrot set for $z_{{new}} = z^{{{}}} + c$'.format(2))
        plt.gcf().set_size_inches(5,4)
        plt.show()
    return elapsed_vectorized

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

@njit
def compute_mandelbrot_numba_slow(size, itterations):

    xDomain = np.linspace(-2.0, 1.0, size)
    yDomain = np.linspace(-1.5, 1.5, size)
    bound = 2
    max_iterations = itterations

    
    c = np.empty((1024,1024), dtype=np.complex128)

    for y in range(size):
        for x in range(size):
            c[y, x] = complex(xDomain[x], yDomain[y])

    z = np.zeros_like(c)
    m = np.zeros((size, size), dtype=np.int32)
    mask = np.ones((size, size), dtype=np.bool_)
        
    for i in np.arange(max_iterations, dtype=np.int32):
        z = np.where(mask, z*z + c, z)
        mask = (z.real*z.real + z.imag*z.imag) <= bound**2
        m = np.where((~mask) & (m == 0), i, m)
        if not np.any(mask):
            break

    return m

def compute_mandelbrot(size, itterations):
    xDomain, yDomain = np.linspace(-2, 1, size), np.linspace(-1.5, 1.5, size)
    bound = 10
    max_iterations = itterations   # any positive integer value
    colormap = 'magma'    # set to any matplotlib valid colormap
    # computing 2-d array to represent the mandelbrot-set
    iterationArray = []
    for y in yDomain:
        row = []
        for x in xDomain:
            c = complex(x,y)
            z = 0
            for iterationNumber in range(max_iterations):
                if(abs(z) >= bound):
                    row.append(iterationNumber)
                    break
                else: z = z**2 + c
            else:
                row.append(0)

        iterationArray.append(row)
    return iterationArray

def benchmark ( func , * args , n_runs =3) :
    """ Time func , return median of n_runs . """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(* args)
        times.append ( time.perf_counter() - t0)
    median_t = statistics.median(times)
    print(f" Median : {median_t:.4f}s ", f"( min ={ min(times):.4f}, max ={max(times):.4f})")
    return median_t, result


compute_mandelbrot_numba(2048, 100)
compute_mandelbrot_numba_slow(2048, 100)

naive , M = benchmark(compute_mandelbrot, 2048, 100)
numbaed , M = benchmark(compute_mandelbrot_numba, 2048, 100)
numbaed_slow , M = benchmark(compute_mandelbrot_numba_slow, 2048, 100)
vector , M = benchmark(compute_mandelbrot_vectorized, 2048, 100)
print(f"Time elapsed Naive {naive}\n")
print(f"Time elapsed vectorized {vector}")
print(f"Time elapsed numba {numbaed}\n")
print(f"Time elapsed numba slow {numbaed_slow}\n")
