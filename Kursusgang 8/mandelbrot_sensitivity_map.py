"""
Name: Tobias C. C. Lundgaard
Course: Numerical Scientific Computing 2026
"""

# Imports here:
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm

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

C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
eps32 = float(np.finfo(np.float32).eps)
delta = np.maximum(eps32 * np.abs(C), 1e-10)


def escape_count(C, max_iter):
    z = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)
    for i in range(max_iter):
        with np.errstate(over='ignore', invalid='ignore'):
            mask = ~esc
            z[mask] = z[mask]**2 + C[mask]
            
            newly = (~esc) & (np.abs(z) > 2.0)
            cnt[newly] = i
            esc[newly] = True
    return cnt

n_base = escape_count(C, max_iter).astype(float)
n_perturb = escape_count(C+delta, max_iter).astype(float)
dn = np.abs(n_base - n_perturb)

########
# Below made witg gemini to fix error
########
# Avoid division by zero and handle interior points (where n_base is max_iter)
with np.errstate(divide='ignore', invalid='ignore'):
    kappa = dn / (eps32 * n_base)
    # n_base == max_iter are interior points, theory says n is not well-defined there
    # Let's set those and any 0 values to NaN so they show as grey
    kappa[n_base == max_iter] = np.nan
    kappa[kappa <= 0] = np.nan 

# 2. Setup Plotting
cmap_k = plt.cm.hot.copy()
cmap_k.set_bad("0.25") # Grey for NaNs

# Safely calculate vmax, ensuring it's at least greater than vmin
valid_kappa = kappa[np.isfinite(kappa)]
if valid_kappa.size > 0:
    vmax = np.nanpercentile(valid_kappa, 99)
    vmax = max(vmax, 10) # Ensure vmax is significantly above vmin=1
else:
    vmax = 100

# 3. Plot
plt.figure(figsize=(10, 8))
# Use LogNorm safely now that kappa has no 0s
plt.imshow(kappa, cmap=cmap_k, origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    norm=LogNorm(vmin=1, vmax=vmax))

plt.colorbar(label=r"$\kappa(c)$ (log scale, $\kappa \geq 1$)")
plt.title(r"Condition number approx $\kappa(c)=|\Delta n|\,/\,(\varepsilon_{32}\,n(c))$")
plt.savefig("sensitivity_map.png", dpi=300, bbox_inches="tight")
plt.close()
