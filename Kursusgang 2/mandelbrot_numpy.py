"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""
# Imports here:
import numpy as np
import matplotlib.pyplot as plt
import time


def compute_mandelbrot_naive():
    start = time.time()
    xDomain, yDomain = np.linspace(-2, 1,1000), np.linspace(-1.5, 1.5,1000)
    bound = 10
    max_iterations = 100  
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
    ax = plt.axes()
    ax.set_aspect('equal')
    graph = ax.pcolormesh(xDomain, yDomain, iterationArray, cmap = colormap)
    plt.colorbar(graph)
    plt.xlabel("Real-Axis")
    plt.ylabel("Imaginary-Axis")
    plt.title('Multibrot set for $z_{{new}} = z^{{{}}} + c$'.format(2))
    plt.gcf().set_size_inches(5,4)
    plt.show()



def compute_mandelbrot_vectorized():
    start = time.time()
    xDomain, yDomain = np.linspace(-2, 1,1000), np.linspace(-1.5, 1.5,1000)
    max_iterations = 1000
    bound = 2
    colormap = "magma"

    x, y = np.meshgrid(xDomain,yDomain)
    c = complex(x,y)
    z = np.zeros_like(c)
    array = np.zeros(c.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)
    
    # Implement Mandelbrot logic here
    for i in range(max_iterations):
        pass

    elapsed = time.time() - start
    print(f"Time elapsed using the vectorized imementation is:{elapsed} seconds")



compute_mandelbrot_naive()
compute_mandelbrot_vectorized()
