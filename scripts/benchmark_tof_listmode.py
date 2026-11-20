"""Experiments to find the bottlenecks of the TOF listmode projectors (issue #54).

No change of the library is needed: all experiments use the existing Python bindings.
The script times the TOF listmode forward and back projection of the same events

- in different EVENT ORDERS (hypotheses H1 / H2: locality of the image accesses of
  neighbouring GPU threads, and threads of a warp waiting for the slowest one):
    random        events in random order (like time-ordered listmode data)
    lor           sorted by LOR (sinogram index), then TOF bin
    axis_tof      grouped by the principal axis of the LOR, then sorted by the coordinate
                  of the TOF center along that axis (Cui et al., Med Phys 2011)
    morton<c>     sorted along a 3D Morton (Z-order) curve of the TOF center, on a grid of
                  c voxels (neighbouring events are close in 3D)
    axis_morton<c>  grouped by the principal axis of the LOR, then as morton<c>
    axis_dir<n>_morton<c>  additionally grouped by n bins of the transaxial direction
- with different THREADS PER BLOCK (hypothesis H4: occupancy; with 64 threads per block,
  at most 16 blocks = 1024 threads are resident per SM on many GPUs),
- with the Gaussian TOF kernel (erf) and the same Gaussian as a table (hypothesis H5:
  arithmetic per plane; if both are equally fast, erf is not the limit).

Finally, it times "sorting inside every call" (sort key + sort + reordering + projection
in one call) against events that are sorted once (e.g. once per subset, reused in every
iteration), to decide whether the projectors should sort automatically.

It also estimates the number of image planes visited per event with the current plane
range (margin of 1 TOF bin) and with the minimal range (margin of 0.5 TOF bin)
(hypothesis H3), and checks that the projections do not depend on the event order.

The TOF center of an event (the position of the center of its TOF bin on the LOR) is
computed from the LOR end points and the TOF bin, as in the projectors.

usage (run from the repository root, with the build on PYTHONPATH):
  PYTHONPATH=build-cuda13 pixi run -e cuda13 python scripts/benchmark_tof_listmode.py
  PYTHONPATH=build-cuda13 pixi run -e cuda13 python scripts/benchmark_tof_listmode.py --preset signa_fine
  PYTHONPATH=build pixi run python scripts/benchmark_tof_listmode.py --device cpu --num_events 200000 --tpb 64

Needs PyTorch. The geometry and TOF presets are those of scripts/benchmark_tof_sinogram.py.
"""

import argparse
import math
import re
import time

import torch
import parallelproj_core as pc

from benchmark_tof_sinogram import FWHM_TO_SIGMA, PRESETS, image_geometry, sinogram_lors

ORDERS = ["random", "lor", "axis_tof", "morton2", "axis_morton1", "axis_morton2", "axis_morton4", "axis_morton8",
          "axis_dir4_morton2", "axis_dir8_morton2", "axis_dir8_morton4", "axis_dir16_morton4"]


def voxel_grid(shape, vs, origin):
    """coordinates (X, Y, Z) of the voxel centers (in the coordinates of the LORs)"""
    axes = [origin[k] + torch.arange(shape[k]) * vs[k] for k in range(3)]
    X, Y, Z = torch.meshgrid(*axes, indexing="ij")
    return X, Y, Z


def make_events(xs, xe, axial_axis, nb, w, sigma, num_events, seed=1):
    """Listmode events of a cylinder phantom (radius 200 mm): random LORs, emission point
    uniform along the part of the LOR inside the cylinder (rejection sampling), TOF bin
    of the emission point plus Gaussian TOF noise. Returns (event LOR index, TOF bin)."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, xs.shape[0], (num_events,), generator=gen)
    a, b = xs[idx].double(), xe[idx].double()
    ia, ib = (1, 2) if axial_axis == 0 else (0, 1)  # transaxial components
    t = torch.rand(num_events, generator=gen, dtype=torch.float64)
    for _ in range(10):
        p = a + t[:, None] * (b - a)
        redo = (p[:, ia] ** 2 + p[:, ib] ** 2) >= 200.0**2
        if not redo.any():
            break
        t[redo] = torch.rand(int(redo.sum()), generator=gen, dtype=torch.float64)
    length = torch.linalg.norm(b - a, dim=1)
    # distance of the emission point from the LOR center (towards lor_end) + TOF noise
    u = (t - 0.5) * length + torch.randn(num_events, generator=gen, dtype=torch.float64) * sigma
    it = torch.round(u / w + 0.5 * (nb - 1)).clamp(0, nb - 1).to(torch.int16)
    return idx, it


def tof_centers(a, b, it, nb, w):
    """position of the center of TOF bin it on the LOR from a to b (zero center offset)"""
    d = b - a
    u = (it.double() - 0.5 * (nb - 1)) * w  # distance from the LOR center towards b
    return 0.5 * (a + b) + u[:, None] * d / torch.linalg.norm(d, dim=1, keepdim=True)


def morton3(q):
    """3D Morton (Z-order) code of non-negative integer coordinates q (N, 3), 10 bits each"""
    def spread(x):  # insert two zero bits between the bits of x (10 bits -> 30 bits)
        x = x & 0x3FF
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        x = (x | (x << 2)) & 0x09249249
        return x
    return spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)


def sort_key(s, e, it, nb, w, vs, origin, cell, ndir, use_axis, axial_axis):
    """64-bit sort key per event, computed from the float32 event data (as a library
    function would do it):

      bits 40-41: principal axis of the LOR (if use_axis)
      bits 32-39: coarse transaxial direction bin (ndir bins of the azimuthal angle in [0, pi))
      bits  0-29: 3D Morton (Z-order) code of the TOF center, on a grid of `cell` voxels

    Events with close keys have TOF centers close in 3D (and similar directions)."""
    d = e - s
    axis = torch.argmax(d.abs(), dim=1)
    u = (it.float() - 0.5 * (nb - 1)) * w
    c = 0.5 * (s + e) + u[:, None] * d / torch.linalg.norm(d, dim=1, keepdim=True)
    q = torch.floor((c - origin) / (cell * vs)).long().clamp(0, 1023)
    key = morton3(q)
    if ndir > 1:
        ta, tb = (1, 2) if axial_axis == 0 else (0, 1)  # transaxial components
        phi = torch.remainder(torch.atan2(d[:, tb], d[:, ta]), math.pi)  # direction of the line, in [0, pi)
        dbin = torch.clamp((phi * (ndir / math.pi)).long(), 0, ndir - 1)
        key = key | (dbin << 32)
    if use_axis:
        key = key | (axis.long() << 40)
    return key


def parse_order(name):
    """(cell, ndir, use_axis) of a Morton order name: morton<c>, axis_morton<c>,
    axis_dir<n>_morton<c>; None for the other orders"""
    m = re.fullmatch(r"(axis_)?(?:dir(\d+)_)?morton(\d+)", name)
    if m is None:
        return None
    return int(m.group(3)), int(m.group(2) or 1), m.group(1) is not None


def event_order(name, lor_idx, it, s, e, a, b, nb, w, vs, origin, gen, axial_axis):
    """permutation of the events for the given order (computed on the device of the inputs)"""
    n = lor_idx.shape[0]
    if name == "random":
        return torch.randperm(n, generator=gen).to(lor_idx.device)
    if name == "lor":
        key = lor_idx.long() * nb + it.long()
        return torch.argsort(key)
    if name == "axis_tof":
        axis = torch.argmax((b - a).abs(), dim=1)
        c = tof_centers(a, b, it, nb, w)
        pos = c.gather(1, axis[:, None])[:, 0]
        order = torch.argsort(pos, stable=True)
        return order[torch.argsort(axis[order], stable=True)]
    spec = parse_order(name)
    if spec is None:
        raise ValueError(f"unknown order {name}")
    cell, ndir, use_axis = spec
    return torch.argsort(sort_key(s, e, it, nb, w, vs, origin, cell, ndir, use_axis, axial_axis))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preset", default="signa", choices=sorted(PRESETS))
    p.add_argument("--num_events", type=int, default=10_000_000)
    p.add_argument("--num_sigmas", type=float, default=3.0)
    p.add_argument("--orders", nargs="+", default=ORDERS,
                   help="random, lor, axis_tof, or Morton orders: morton<c>, axis_morton<c>, axis_dir<n>_morton<c> "
                        "(c: grid cell in voxels, n: number of direction bins)")
    p.add_argument("--tpb", nargs="+", type=int, default=[64, 256], help="threads per block to test")
    p.add_argument("--kernels", nargs="+", default=["gauss"], choices=["gauss", "table"])
    p.add_argument("--auto_order", default="axis_morton2",
                   help="order used to time 'sorting inside every call' (key + sort + reorder + projection)")
    p.add_argument("--axial_axis", type=int, default=2, choices=[0, 2])
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    dev = torch.device(args.device)
    on_gpu = dev.type == "cuda"
    cfg = PRESETS[args.preset]
    nb, w = cfg["num_tof_bins"], cfg["tof_bin_width"]
    sig = cfg["fwhm"] * FWHM_TO_SIGMA
    img_shape, vox = image_geometry(cfg, args.axial_axis)
    vs = torch.tensor(vox, dtype=torch.float32)
    origin = -0.5 * (torch.tensor(img_shape, dtype=torch.float32) - 1) * vs
    xs, xe = sinogram_lors(cfg, "RVP", args.axial_axis)

    # phantom image: cylinder (radius 200 mm, value 1) with three hot spheres
    X, Y, Z = voxel_grid(img_shape, vs, origin)
    tr = (Y, Z) if args.axial_axis == 0 else (X, Y)
    img = (tr[0] ** 2 + tr[1] ** 2 < 200.0**2).float()
    ax = X if args.axial_axis == 0 else Z
    for cx, cy, cz, r in [(60, 0, 0, 30), (-80, 50, 40, 20), (0, -100, -60, 25)]:
        img += 3 * ((tr[0] - cx) ** 2 + (tr[1] - cy) ** 2 + (ax - cz) ** 2 < r**2).float()

    lor_idx, it = make_events(xs, xe, args.axial_axis, nb, w, sig, args.num_events)
    a, b = xs[lor_idx].double(), xe[lor_idx].double()

    # --- H3: planes per event (estimate, ignoring clipping at the image boundary):
    # |at| = cf / w TOF bins per plane, cf = voxel size along the principal axis / cos
    d = b - a
    axis = torch.argmax(d.abs(), dim=1)
    cosv = d.abs().gather(1, axis[:, None])[:, 0] / torch.linalg.norm(d, dim=1)
    cf = vs.double()[axis] / cosv
    m = args.num_sigmas * sig / w
    at = cf / w
    planes_now = (2 * m + 2) / at
    planes_min = (2 * m + 1) / at
    print(f"parallelproj_core {pc.__version__}, device {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if on_gpu else ""))
    print(f"preset {args.preset}: {nb} TOF bins of {w:.2f} mm, sigma {sig:.2f} mm ({sig / w:.2f} bins), "
          f"{args.num_sigmas:g} sigma, image {img_shape} (axial axis {args.axial_axis}), {args.num_events / 1e6:.1f} M events")
    print(f"H3 planes per event (estimate): current range {planes_now.mean():.1f}, minimal range {planes_min.mean():.1f} "
          f"(-{100 * (1 - planes_min.mean() / planes_now.mean()):.0f} %); principal axes: "
          + ", ".join(f"{k}: {100 * float((axis == k).double().mean()):.0f} %" for k in range(3)))

    # --- arrays on the device
    f32 = torch.float32
    ev_s, ev_e = xs[lor_idx].to(dev), xe[lor_idx].to(dev)
    it_d = it.to(dev)
    img_d, origin_d, vs_d = img.to(dev), origin.to(dev), vs.to(dev)
    sigma_d, off_d = torch.tensor([sig], dtype=f32, device=dev), torch.zeros(1, dtype=f32, device=dev)
    u = torch.linspace(-args.num_sigmas * sig, args.num_sigmas * sig, 2001, dtype=torch.float64)
    cdf_d = (0.5 * (1 + torch.erf(u / (math.sqrt(2) * sig)))).to(f32).to(dev)
    u_min, u_max = float(u[0]), float(u[-1])
    scale_d = torch.ones(1, dtype=f32, device=dev)

    def sync():
        if on_gpu:
            torch.cuda.synchronize(dev)

    def best_time(fn):
        fn()
        times = []
        for _ in range(args.repeats):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            times.append(time.perf_counter() - t0)
        return min(times)

    def fwd(s, e, tb, out, kernel, tpb):
        if kernel == "gauss":
            pc.joseph3d_tof_lm_fwd(s, e, img_d, origin_d, vs_d, out, w, sigma_d, off_d, tb, nb,
                                   num_sigmas=args.num_sigmas, threads_per_block=tpb)
        else:
            pc.joseph3d_tof_lm_fwd_tab(s, e, img_d, origin_d, vs_d, out, w, off_d, tb, nb, cdf_d, u_min, u_max,
                                       scale_d, threads_per_block=tpb)

    def back(s, e, tb, vals, x, kernel, tpb):
        x.zero_()
        if kernel == "gauss":
            pc.joseph3d_tof_lm_back(s, e, x, origin_d, vs_d, vals, w, sigma_d, off_d, tb, nb,
                                    num_sigmas=args.num_sigmas, threads_per_block=tpb)
        else:
            pc.joseph3d_tof_lm_back_tab(s, e, x, origin_d, vs_d, vals, w, off_d, tb, nb, cdf_d, u_min, u_max,
                                        scale_d, threads_per_block=tpb)

    # reference: forward / back projection in the original (random) event order
    y_ref = torch.zeros(args.num_events, dtype=f32, device=dev)
    fwd(ev_s, ev_e, it_d, y_ref, "gauss", 64)
    ones = torch.ones(args.num_events, dtype=f32, device=dev)
    x_ref = torch.zeros(img_shape, dtype=f32, device=dev)
    back(ev_s, ev_e, it_d, ones, x_ref, "gauss", 64)

    gen = torch.Generator().manual_seed(2)
    a_d, b_d, lor_d, origin64, vs64 = a.to(dev), b.to(dev), lor_idx.to(dev), origin.to(dev), vs.to(dev)
    results = []
    x = torch.zeros(img_shape, dtype=f32, device=dev)
    y = torch.zeros(args.num_events, dtype=f32, device=dev)
    for order in args.orders:
        sync()
        t0 = time.perf_counter()
        perm = event_order(order, lor_d, it_d, ev_s, ev_e, a_d, b_d, nb, w, vs_d, origin_d, gen, args.axial_axis)
        s, e, tb = ev_s[perm].contiguous(), ev_e[perm].contiguous(), it_d[perm].contiguous()
        sync()
        t_sort = time.perf_counter() - t0
        # results must not depend on the order: forward bit-identical, back up to rounding
        fwd(s, e, tb, y, "gauss", 64)
        err_f = float((y - y_ref[perm]).abs().max() / y_ref.abs().max())
        back(s, e, tb, ones, x, "gauss", 64)
        err_b = float((x - x_ref).abs().max() / x_ref.abs().max())
        print(f"\norder {order:20s} sort + permute {t_sort:7.3f} s   check: fwd max rel diff {err_f:.1e}, "
              f"back max rel diff {err_b:.1e}")
        for kernel in args.kernels:
            for tpb in args.tpb:
                t_f = best_time(lambda: fwd(s, e, tb, y, kernel, tpb))
                t_b = best_time(lambda: back(s, e, tb, ones, x, kernel, tpb))
                results.append((order, kernel, tpb, t_f, t_b))
                print(f"  kernel {kernel:5s} threads/block {tpb:4d}:  fwd {t_f:8.4f} s ({args.num_events / t_f / 1e6:6.1f} M events/s)"
                      f"   back {t_b:8.4f} s ({args.num_events / t_b / 1e6:6.1f} M events/s)")

    # summary relative to random order, Gaussian kernel, 64 threads per block (the default)
    base = {(r[1], r[2]): r for r in results if r[0] == "random"}
    ref = base.get(("gauss", 64))
    if ref is not None:
        print("\nspeed-up vs. random order, Gaussian kernel, 64 threads per block (default call):")
        for order, kernel, tpb, t_f, t_b in results:
            print(f"  {order:20s} {kernel:5s} tpb {tpb:4d}:  fwd {ref[3] / t_f:5.2f}x   back {ref[4] / t_b:5.2f}x")

    # --- "sorting inside every call": what an automatic sort in each projector call would
    # cost. Timed: sort key + argsort + reordering of the inputs + projection (+ putting the
    # forward results back into the original order), all in one call, compared with the
    # projection of pre-sorted events (sorted once, e.g. once per subset for all iterations).
    spec = parse_order(args.auto_order)
    if spec is not None:
        cell, ndir, use_axis = spec
        y_out = torch.zeros(args.num_events, dtype=f32, device=dev)
        vals = torch.ones(args.num_events, dtype=f32, device=dev)

        def key_and_sort():
            return torch.argsort(sort_key(ev_s, ev_e, it_d, nb, w, vs_d, origin_d, cell, ndir, use_axis, args.axial_axis))

        def auto_fwd():
            perm = key_and_sort()
            fwd(ev_s[perm], ev_e[perm], it_d[perm], y, "gauss", 64)
            y_out[perm] = y

        def auto_back():
            perm = key_and_sort()
            back(ev_s[perm], ev_e[perm], it_d[perm], vals[perm], x, "gauss", 64)

        perm = key_and_sort()
        s, e, tb = ev_s[perm].contiguous(), ev_e[perm].contiguous(), it_d[perm].contiguous()
        t_key = best_time(key_and_sort)
        t_reorder = best_time(lambda: (ev_s[perm], ev_e[perm], it_d[perm]))
        t_pre_f = best_time(lambda: fwd(s, e, tb, y, "gauss", 64))
        t_pre_b = best_time(lambda: back(s, e, tb, ones, x, "gauss", 64))
        t_auto_f = best_time(auto_fwd)
        t_auto_b = best_time(auto_back)
        t_rand_f = best_time(lambda: fwd(ev_s, ev_e, it_d, y, "gauss", 64))
        t_rand_b = best_time(lambda: back(ev_s, ev_e, it_d, ones, x, "gauss", 64))
        print(f"\nsorting inside every call ({args.auto_order}, Gaussian kernel, 64 threads per block):")
        print(f"  sort key + argsort {t_key:7.4f} s, reordering the 3 event arrays {t_reorder:7.4f} s")
        print(f"  fwd:  random order {t_rand_f:7.4f} s   sorted once {t_pre_f:7.4f} s   sorted in every call {t_auto_f:7.4f} s")
        print(f"  back: random order {t_rand_b:7.4f} s   sorted once {t_pre_b:7.4f} s   sorted in every call {t_auto_b:7.4f} s")


if __name__ == "__main__":
    main()
