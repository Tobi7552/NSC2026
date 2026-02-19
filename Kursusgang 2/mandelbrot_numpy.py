"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""
# Imports here:
import numpy as np
import matplotlib.pyplot as plt
import time , statistics


def compute_mandelbrot_naive(plot = False):
    start = time.time()
    xDomain, yDomain = np.linspace(-2, 1,1024), np.linspace(-1.5, 1.5,1024)
    bound = 2
    max_iterations = 1000  
    colormap = 'magma'    # Set to any matplotlib valid colormap
    
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
    elapsed = time.time() - start
    print (f"Time elapsed using the naive imementation is:{elapsed} seconds")
    # plotting the data
    if plot:
        ax = plt.axes()
        ax.set_aspect('equal')
        graph = ax.pcolormesh(xDomain, yDomain, iterationArray, cmap = colormap)
        plt.colorbar(graph)
        plt.xlabel("Real-Axis")
        plt.ylabel("Imaginary-Axis")
        plt.title('Multibrot set for $z_{{new}} = z^{{{}}} + c$'.format(2))
        plt.gcf().set_size_inches(5,4)
        plt.show()
    return elapsed


def compute_mandelbrot_vectorized(plot = False):
    start = time.time()
    xDomain, yDomain = np.linspace(-2, 1,1024), np.linspace(-1.5, 1.5,1024)
    max_iterations = 1000
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
    print(f"Time elapsed using the vectorized imementation is:{elapsed_vectorized} seconds")
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


#time_naive = compute_mandelbrot_naive()
#time_vectorized = compute_mandelbrot_vectorized()
t , M = benchmark(compute_mandelbrot_vectorized)
t1 , M1 = benchmark(compute_mandelbrot_naive)
print(f"Median time for Naive is {t1}, and for vectorized {t}")