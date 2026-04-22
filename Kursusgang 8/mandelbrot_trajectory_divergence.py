"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""

# Imports here:
import numpy as np
import matplotlib.pyplot as plt
import time


# Parameters from the slides(37)
N = 1024
x_min, x_max = -0.7530, -0.7490
y_min, y_max = 0.0990, 0.1030
max_iter = 1000 
taus = [0.01, 0.001, 0.0001]

# Grid parameters
x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(x, y)
C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
C32 = C64.astype(np.complex64)

Z64 = np.zeros_like(C64)
Z32 = np.zeros_like(C32)

divergence_iter = np.full(C64.shape, max_iter, dtype=np.int32)
mask = np.ones(C64.shape, dtype=bool)

print("Computing divergence map...")
for tau in taus:

    print(f"\nComputing divergence map for tau = {tau}...")
    t0 = time.time()

    # reset state EACH tau
    Z64 = np.zeros_like(C64)
    Z32 = np.zeros_like(C32)

    divergence_iter = np.full(C64.shape, max_iter, dtype=np.int32)
    mask = np.ones(C64.shape, dtype=bool)

    for i in range(max_iter):
        Z64[mask] = Z64[mask] * Z64[mask] + C64[mask]
        Z32[mask] = Z32[mask] * Z32[mask] + C32[mask]

        diff = np.abs(Z64 - Z32)

        diverged = (diff > tau) & mask

        divergence_iter[diverged] = i
        mask[diverged] = False
        #print(i, np.sum(mask))
        
        if not mask.any():
            break

    print(f"Time: {time.time() - t0:.2f} s")

    fraction = np.mean(divergence_iter < max_iter)
    print("Fractions of pixels diverging:", fraction)

    # -----------------------------
    # Save divergence plot
    # -----------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(divergence_iter, cmap="inferno",
               extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label="Divergence iteration")
    plt.title(f"Divergence (tau = {tau})")
    plt.xlabel("Re")
    plt.ylabel("Im")

    filename = f"divergence_map_tau_{tau}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")

print("Computing escape-time map...")
Z = np.zeros_like(C64)
escape = np.full(C64.shape, max_iter, dtype=np.int32)
mask = np.ones(C64.shape, dtype=bool)

for k in range(max_iter):
    Z[mask] = Z[mask]**2 + C64[mask]

    escaped = (np.abs(Z) > 2) & mask
    escape[escaped] = k
    mask[escaped] = False

    if not mask.any():
        break

plt.figure(figsize=(6, 6))
plt.imshow(escape, cmap="viridis",
           extent=[x_min, x_max, y_min, y_max])
plt.colorbar(label="Escape iteration")
plt.title("Mandelbrot Escape-Time Map (float64)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.savefig("escape_map.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: escape_map.png")