"""float64 reference of the TOF sinogram forward projection of single LORs (tests only).

Same conventions as joseph3d_tof_sino_fwd: Joseph plane range (computed in float32 as in
the library, since an edge plane can carry a large part of a TOF bin for narrow kernels),
bilinear interpolation, TOF bin positions incl. center offset, truncated TOF kernel
(src/tof_kernel.h) with windows wider than max_window_bins truncated and renormalized.
"""

import math

import numpy as np


def _erf(x):
    return np.vectorize(math.erf)(x)


def _bilinear(img, direction, ip, fa, fb):
    plane = np.take(img, ip, axis=direction)  # 2D array of the two other axes (in order)
    na, nb = plane.shape
    ia, ib = math.floor(fa), math.floor(fb)
    wa, wb = fa - ia, fb - ib
    val = 0.0
    for ja, jb, wgt in [(ia, ib, (1 - wa) * (1 - wb)), (ia + 1, ib, wa * (1 - wb)),
                        (ia, ib + 1, (1 - wa) * wb), (ia + 1, ib + 1, wa * wb)]:
        if 0 <= ja < na and 0 <= jb < nb:
            val += wgt * float(plane[ja, jb])
    return val


def tof_sino_fwd_reference(xs, xe, img, origin, vs, tof_bin_width, sigma, offset, num_tof_bins,
                           num_sigmas=3.0, max_window_bins=64):
    """TOF sinogram (num_tof_bins,) of one LOR from xs to xe (3-vectors), float64"""
    xs, xe = np.asarray(xs, np.float64), np.asarray(xe, np.float64)
    origin, vs = np.asarray(origin, np.float64), np.asarray(vs, np.float64)
    img = np.asarray(img, np.float64)
    dim = np.array(img.shape)
    out = np.zeros(num_tof_bins)
    d = xe - xs
    if not d.any():
        return out

    # --- plane range in float32, as ray_cube_intersection_joseph
    f32 = np.float32
    xs32, d32 = xs.astype(f32), xe.astype(f32) - xs.astype(f32)
    bmin32 = origin.astype(f32) - f32(0.5) * vs.astype(f32)
    bmax32 = bmin32 + dim.astype(f32) * vs.astype(f32)
    t0, t1 = f32(0), f32(1)
    for k in range(3):
        if d32[k] != 0:
            inv = f32(1) / d32[k]
            ta, tb = (bmin32[k] - xs32[k]) * inv, (bmax32[k] - xs32[k]) * inv
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        elif xs32[k] < bmin32[k] or xs32[k] > bmax32[k]:
            return out
    if t1 < t0:
        return out
    cos_sq = d**2 / (d**2).sum()
    direction = 0
    if cos_sq[1] >= cos_sq[0] and cos_sq[1] >= cos_sq[2]:
        direction = 1
    elif cos_sq[2] >= cos_sq[0] and cos_sq[2] >= cos_sq[1]:
        direction = 2
    o32, v32 = origin.astype(f32), vs.astype(f32)
    f1 = float((xs32[direction] + t0 * d32[direction] - o32[direction]) / v32[direction])
    f2 = float((xs32[direction] + t1 * d32[direction] - o32[direction]) / v32[direction])
    if math.floor(f1) == math.floor(f2):
        return out
    f1, f2 = min(f1, f2), max(f1, f2)
    istart, iend = math.floor(f1) + 1, math.floor(f2)

    # --- TOF
    cf = vs[direction] / math.sqrt(cos_sq[direction])
    ax, bx = [k for k in range(3) if k != direction]
    costheta = vs[direction] / cf
    sign = 1.0 if xe[direction] >= xs[direction] else -1.0
    tof_origin = (0.5 * (xs[direction] + xe[direction]) - sign * (0.5 * num_tof_bins - 0.5) * tof_bin_width * costheta
                  + sign * offset * costheta)
    m = num_sigmas * sigma / tof_bin_width  # half width of the support in TOF bins
    scale = tof_bin_width / (math.sqrt(2) * sigma)
    for ip in range(istart, iend + 1):
        x_dir = origin[direction] + ip * vs[direction]
        t = (x_dir - xs[direction]) / d[direction]
        fa = (xs[ax] + t * d[ax] - origin[ax]) / vs[ax]
        fb = (xs[bx] + t * d[bx] - origin[bx]) / vs[bx]
        val = cf * _bilinear(img, direction, ip, fa, fb)
        it_f = (x_dir - tof_origin) / (sign * tof_bin_width * costheta)
        it_min = math.floor(it_f - m + 0.5)
        n = math.floor(it_f + m + 0.5) - it_min + 1
        truncated = n > max_window_bins
        if truncated:
            it_min, n = math.floor(it_f + 0.5) - max_window_bins // 2, max_window_bins
        bins = it_min + np.arange(n)
        u = np.clip(np.arange(n + 1) + it_min - 0.5 - it_f, -m, m)
        e = _erf(u * scale)
        wgt = 0.5 * (e[1:] - e[:-1])
        wgt /= (0.5 * (e[-1] - e[0])) if truncated else math.erf(m * scale)
        ok = (bins >= 0) & (bins < num_tof_bins)
        out[bins[ok]] += val * wgt[ok]
    return out
