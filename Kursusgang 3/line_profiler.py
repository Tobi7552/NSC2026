"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""

# Imports here:
import numpy as np
import matplotlib.pyplot as plt
import cProfile, pstats
from mandelbrot_numba import compute_mandelbrot, compute_mandelbrot_vectorized

cProfile.run("compute_mandelbrot(1024,100)","Naive_profile.prof")
cProfile.run("compute_mandelbrot_vectorized(1024,100)","Numpy_profile.prof")


for name in ("Naive_profile.prof","Numpy_profile.prof"):
    stats = pstats.Stats(name)
    stats.sort_stats("cumulative")
    stats.print_stats(10)