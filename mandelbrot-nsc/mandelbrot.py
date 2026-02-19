"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""

# Imports here:
import numpy as np
import matplotlib.pyplot as plt
import time


def compute_mandelbrot():
    start = time.time()
    xDomain, yDomain = np.linspace(-2, 1,1000), np.linspace(-1.5, 1.5,1000)
    bound = 10
    max_iterations = 1000   # any positive integer value
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
    elapsed = time.time() - start
    print (f" Computation took {elapsed} seconds ")
    # plotting the data
    ax = plt.axes()
    ax.set_aspect('equal')
    graph = ax.pcolormesh(xDomain, yDomain, iterationArray, cmap = colormap)
    plt.colorbar(graph)
    plt.xlabel("Real-Axis")
    plt.ylabel("Imaginary-Axis")
    plt.title('Multibrot set for $z_{{new}} = z^{{{}}} + c$'.format(2))
    plt.gcf().set_size_inches(5,4)
    plt.show()

compute_mandelbrot()
