import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def effective_tof_kernel(dx: float, sigma_t: float, tbin_width: float) -> float:
    """Gaussian integrated over a tof bin width."""
    sqrt2 = math.sqrt(2.0)
    return 0.5 * (
        erf((dx + 0.5 * tbin_width) / (sqrt2 * sigma_t))
        - erf((dx - 0.5 * tbin_width) / (sqrt2 * sigma_t))
    )


img_dim = (280, 270, 280)
voxsize = (1, 1, 1)

xstart = (300, 0, 0)
xend = (-300, 0, 0)

sigma_tof: float = 24.0
tofbin_width: float = 0.25*sigma_tof
num_sigmas: float = 3.0
tof_center_offset: float = 0.0

show_fig = True

num_tofbins: int | None = None

###########################################

if num_tofbins is None:
    ray_length = math.sqrt(
        (xend[0] - xstart[0]) ** 2
        + (xend[1] - xstart[1]) ** 2
        + (xend[2] - xstart[2]) ** 2
    )
    num_tofbins = math.ceil(ray_length / tofbin_width)

it = int(0.3*num_tofbins)

n0, n1, n2 = img_dim

img_origin = (
    (-(n0 / 2) + 0.5) * voxsize[0],
    (-(n1 / 2) + 0.5) * voxsize[1],
    (-(n2 / 2) + 0.5) * voxsize[2],
)


d0 = xend[0] - xstart[0]
d1 = xend[1] - xstart[1]
d2 = xend[2] - xstart[2]


# %%
# get the correction factor cf

dr_sq = (d0**2, d1**2, d2**2)
direction = max(range(3), key=lambda i: dr_sq[i])

sum_sq = sum(dr_sq)
cos_sq = dr_sq[direction] / sum_sq

cf = voxsize[direction] / (cos_sq**0.5)
# %%


# %%
# step through the volume plane by plane

assert direction == 0

dr = dr_sq[direction] ** 0.5

#### ONLY VALID FOR direction == 0 ####
a1 = (d1 * voxsize[direction]) / (voxsize[1] * dr)
b1 = (
    xstart[1] - img_origin[1] + d1 * (img_origin[direction] - xstart[direction]) / dr
) / voxsize[1]

a2 = (d2 * voxsize[direction]) / (voxsize[2] * dr)
b2 = (
    xstart[2] - img_origin[2] + d2 * (img_origin[direction] - xstart[direction]) / dr
) / voxsize[2]
#### ONLY VALID FOR direction == 0 ####

#####################################
#####################################
#####################################

# TOF related calculations

# max tof bin diffference where kernel is effectively non-zero
# max_tof_bin_diff = num_sigmas * max(sigma_tof, tofbin_width) / tofbin_width
costheta: float = voxsize[direction] / cf
max_tof_bin_diff: float = num_sigmas * sigma_tof / (tofbin_width)

# calculate the where the TOF bins are located along the projected line
# in world coordinates
sign: int = 1 if xend[direction] >= xstart[direction] else -1
# the tof bin centers (in world coordinates projected to the axis along which we step through the volume)
# are at it*a_tof + b_tof for it in range(num_tofbins)
tof_origin: float = (
    0.5 * (xstart[direction] + xend[direction])
    - sign * (num_tofbins / 2 - 0.5) * (tofbin_width * costheta)
    + tof_center_offset * costheta
)
tof_slope: float = sign * (tofbin_width * costheta)

### TOF offset and increment per voxel step in direction
at: float = sign * cf / tofbin_width
bt: float = (img_origin[direction] - tof_origin) / tof_slope

## calculate start and stop plane according to TOF

istart = math.floor(((it - sign*num_sigmas*sigma_tof/tofbin_width) - bt) / at)
iend = math.floor(((it + sign*num_sigmas*sigma_tof/tofbin_width) - bt) / at)

print(f"istart: {istart}, iend: {iend}")

i1_f = istart * a1 + b1
i2_f = istart * a2 + b2


# it_f is the index of the TOF bin at the current plane
it_f: float = istart * at + bt

#####################################
#####################################
#####################################

if show_fig:
    fig = plt.figure(figsize=(8, 8), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")

    ax.plot([xstart[0], xend[0]], [xstart[1], xend[1]], [xstart[2], xend[2]], "r-")
    ax.plot([xstart[0], xend[0]], [xstart[1], xstart[1]], [xstart[2], xstart[2]], "b-")

    ax.scatter(xstart[0], xstart[1], xstart[2], c="r", marker="x")
    ax.scatter(xend[0], xend[1], xend[2], c="r")
    ax.scatter(xend[0], xstart[1], xstart[2], c="b")

    ax.scatter(img_origin[0], xstart[1], xstart[2], marker=".", c="k")
    ax.scatter(
        img_origin[0] + voxsize[0] * (img_dim[0] - 1),
        xstart[1],
        xstart[2],
        marker=".",
        c="k",
    )

    ax.scatter(
        [i * tof_slope + tof_origin for i in range(num_tofbins)],
        num_tofbins * [xstart[1]],
        num_tofbins * [xstart[2]],
        marker="x",
    )

    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_zlim(-400, 400)

#####################################
#####################################
#####################################

tof_weights = np.zeros(img_dim[direction])
tof_sum = 0.0
tof_integral = 0.0

for i in range(istart, iend + 1):
    # we sum over all planes that are within num_sigmas*sigma_tof of the current TOF bin center
    # it can be that part of those planes are outside the image
    # we have to calculate the TOF weight for all planes, to get the normalization right,
    # but we only add the contribution to the integral for planes that are inside the image

    dist = abs(it_f - it) * tofbin_width
    tof_weight = effective_tof_kernel(dist, sigma_tof, tofbin_width)
    tof_sum += tof_weight

    if i >= 0 and i < img_dim[direction]:
        #interp_img_val = bilinear_interp_fixed0(img, n0, n1, n2, i0, i1_f, i2_f);
        interp_img_val = 2.3
        tof_integral += tof_weight * interp_img_val
        tof_weights[i] = tof_weight

    it_f += at
    i1_f += a1
    i2_f += a2

expected_tof_sum = (tofbin_width / (cf*voxsize[direction]))
tof_weight_sum_corr_factor = expected_tof_sum  / tof_sum
tof_integral_corr = tof_integral * tof_weight_sum_corr_factor

print("Sum of unnorm. TOF weights:", tof_sum)
print("Expected approx. sum:", tofbin_width / (cf*voxsize[direction]))
print(f"TOF weight sum corr factor:", tof_weight_sum_corr_factor)
print(f"TOF integral before corr: {tof_integral}, after corr: {tof_integral_corr}")



if show_fig:
    fig2, ax2 = plt.subplots(1,2,figsize=(10, 5), layout="constrained")
    ax2[0].plot(tof_weights, "o-")
    ax2[0].plot(tof_weights * tof_weight_sum_corr_factor, ".-")
    ax2[0].axvline(istart, color="g", linestyle="--")
    ax2[0].axvline(iend, color="r", linestyle="--")
    ax2[0].set_xlabel("Plane index")
    ax2[0].set_ylabel(f"TOF weight w.r.t TOF bin {it}/{num_tofbins}")

    # plot the TOF kernel
    dist = tofbin_width*(np.arange(num_tofbins) - num_tofbins//2 + 0.5)
    y = effective_tof_kernel(dist, sigma_tof, tofbin_width)
    ax2[1].plot(dist / tofbin_width, y, "o-")
    ax2[1].set_xlabel("TOF bin")
    ax2[1].set_ylabel("Effective TOF kernel value")

    plt.show()
