import numpy as np
import matplotlib.pyplot as plt
from math import erf


def eff_tof_kernel(s, sigma, Delta):
    """
    Compute the effective TOF kernel w_eff(s) for a given distance s from the TOF bin center,
    TOF standard deviation sigma, and TOF bin width Delta.
    """
    sqrt2 = np.sqrt(2.0)
    return 0.5 * (
        erf((s + Delta / 2) / (sqrt2 * sigma)) - erf((s - Delta / 2) / (sqrt2 * sigma))
    )


# --- Parameters (edit as needed) ---
sigma = 15.0  # sigma_TOF (distance units along LOR)
Delta = 20.0  # TOF bin width
# ----------------------------------

# here s equal s - s-c (distance from the TOF bin center), so that the kernel is centered at 0
s, ds = np.linspace(-60, 60, 1000, retstep=True)

# Continuous Gaussian g(s; mu, sigma)
g = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (s / sigma) ** 2)

# Effective kernel w_eff(s) (difference of erf terms), matching the doc text
sqrt2 = np.sqrt(2.0)

v_eff_tof_kernel = np.vectorize(eff_tof_kernel, excluded={1, 2}, otypes=[float])
w_eff = v_eff_tof_kernel(s, sigma, Delta)

# %%

s_cs = [-40, 0, 25]

fig, ax = plt.subplots(
    1, 2, figsize=(10, 5), layout="constrained", sharex=True, sharey=False
)
ax[0].plot(s, g / (ds * g.sum()), label="Gaussian kernel")

for i, s_c in enumerate(s_cs):
    mask = np.abs(s - s_c) <= Delta / 2
    ax[0].fill_between(s[mask], 0, g[mask], alpha=0.25, color=plt.cm.tab10(i + 2))
    ax[0].axvline(s_c, linestyle="--", linewidth=1.0, color="k")
    ax[0].axvline(s_c - Delta / 2, linestyle="--", linewidth=0.5, color="k")
    ax[0].axvline(s_c + Delta / 2, linestyle="--", linewidth=0.5, color="k")


ax[1].plot(s, g / (ds * g.sum()), label="Gaussian kernel", ls="--")
ax[1].plot(s, w_eff / (ds * w_eff.sum()), label="effective kernel")

for i, s_c in enumerate(s_cs):
    ax[1].axvline(s_c, linestyle="--", linewidth=1.0, color="k")
    ax[1].plot(
        [s_c],
        [eff_tof_kernel(s_c, sigma, Delta) / (ds * w_eff.sum())],
        "o",
        color=plt.cm.tab10(i + 2),
    )

for axx in ax:
    axx.set_xlabel("distance from TOF bin center (s - s_c) [mm]")
    axx.grid(ls=":")

ax[0].set_ylabel("kernel value")
ax[1].set_ylabel("(normalized ) kernel value")
ax[0].legend()
ax[1].legend()

fig.suptitle(
    f"Comparison of Gaussian and effective TOF kernels (sigma={sigma} mm, Delta={Delta} mm)"
)

fig.savefig("gaussian_tof_kernel.svg", format="svg", bbox_inches="tight")
fig.show()
