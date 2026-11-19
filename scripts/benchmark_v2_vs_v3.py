"""Compare speed and results of two libparallelproj versions (e.g. 2.0.7 and 3.0).

The script works with the Python bindings of both versions (the positional arguments of
the projectors are the same). Run it once per version, each time with the build of that
version on PYTHONPATH, then compare the two result files:

  # 1. results of the old version (e.g. a build of the v2.0.7 tag in ../pp207/build-cuda13)
  PYTHONPATH=../pp207/build-cuda13 pixi run -e cuda13 python scripts/benchmark_v2_vs_v3.py run --out v2.npz
  # 2. results of the current version
  PYTHONPATH=build-cuda13 pixi run -e cuda13 python scripts/benchmark_v2_vs_v3.py run --out v3.npz
  # 3. table of timings (speed-up v3 vs v2) and differences of the results
  pixi run -e cuda13 python scripts/benchmark_v2_vs_v3.py compare v2.npz v3.npz

What is measured
----------------
TOF sinogram forward / back projection and TOF listmode forward / back projection
(best of --repeats calls, after a warm-up call) for

- TOF settings ("tof" presets, all with 5.5 M sinogram LORs of a SIGNA-like geometry):
    signa   27 TOF bins of 169 ps, CTR 400 ps  (TOF bin ~ sigma)
    vision  33 TOF bins of 143 ps, CTR 214 ps  (Biograph-Vision-like, TOF bin ~ 1.6 sigma)
    fine    134 TOF bins of sigma / 5, CTR 400 ps (narrow TOF bins)
- num_sigmas 3 (Python default) and 4 (used by STIR),
- two memory layouts ("layout"):
    fast    sinogram order RVP (planes fastest), axial image axis last (2)
    stir    sinogram order PVR (radial bins fastest), axial image axis first (0) - the layout
            STIR passes to parallelproj (segment / axial position / view / tangential
            position, image (z, y, x))
- two array locations ("arrays", GPU runs only):
    device  all arrays on the GPU (PyTorch tensors), no copies
    host    all arrays in CPU memory, as STIR passes them: every call copies the inputs to
            the GPU and the result back (included in the time)

The listmode projectors are timed with --num_events events sampled from a cylindrical
phantom (random LORs of the sinogram geometry, TOF bin from the emission point).

For the comparison of the results, the projections of the first 200 000 LORs / events and
the back projections are stored (for the first layout and array location given). Differences
between 2.x and 3.0 are expected at the level of ~1e-4 - 1e-3 (truncated TOF kernel, see
the changelog); larger differences in the listmode projections of 2.x for narrow kernels
are bug fixes.

Needs PyTorch. On a CPU-only build, use --device cpu (and a small --lor_fraction).
"""

import argparse
import json
import math
import time

import numpy as np

C_MM_PER_PS = 0.299792458 / 2  # mm along the LOR per ps of arrival time difference
FWHM_TO_SIGMA = 1 / (2 * math.sqrt(2 * math.log(2)))

GEOMETRY = dict(radius=311.8, n_views=224, n_rad=351, rad_half=300.0, n_rings=45, ring_spacing=5.56,
                n_ring_pairs=70, img=(256, 256, 89), vox=(2.34, 2.34, 2.78))
_TOF_RANGE = 169 * C_MM_PER_PS * 27  # covered length along the LOR (SIGNA)
_SIG400 = 400 * C_MM_PER_PS * FWHM_TO_SIGMA
TOF_PRESETS = {
    "signa": dict(tof_bin_width=169 * C_MM_PER_PS, num_tof_bins=27, sigma=_SIG400),
    "vision": dict(tof_bin_width=143 * C_MM_PER_PS, num_tof_bins=33, sigma=214 * C_MM_PER_PS * FWHM_TO_SIGMA),
    "fine": dict(tof_bin_width=_SIG400 / 5, num_tof_bins=int(round(_TOF_RANGE / (_SIG400 / 5))), sigma=_SIG400),
}
NUM_SAVE = 200_000  # LORs / events whose projections are stored for the comparison


def sinogram_lors(torch, lor_order, axial_axis, lor_fraction):
    """LOR end points (views x radial bins x ring pairs) in the given memory order;
    lor_fraction < 1 keeps only every k-th view (smaller problem for CPU runs)"""
    g = GEOMETRY
    n_views = max(1, int(round(g["n_views"] * lor_fraction)))
    phi = torch.arange(n_views, dtype=torch.float64) * math.pi / n_views
    rad = torch.linspace(-g["rad_half"], g["rad_half"], g["n_rad"], dtype=torch.float64)
    zr = (torch.arange(g["n_rings"], dtype=torch.float64) - 0.5 * (g["n_rings"] - 1)) * g["ring_spacing"]
    pairs = sorted(((a, b) for a in range(g["n_rings"]) for b in range(g["n_rings"])),
                   key=lambda q: (abs(q[0] - q[1]), q[0]))
    step = len(pairs) / g["n_ring_pairs"]
    pairs = [pairs[int(k * step)] for k in range(g["n_ring_pairs"])]
    z0 = torch.tensor([zr[a] for a, _ in pairs], dtype=torch.float64)
    z1 = torch.tensor([zr[b] for _, b in pairs], dtype=torch.float64)
    P, V, R = len(pairs), n_views, g["n_rad"]
    ph = phi[None, :, None].expand(P, V, R)
    r = rad[None, None, :].expand(P, V, R)
    h = torch.sqrt(g["radius"] ** 2 - r**2)
    c, s = torch.cos(ph), torch.sin(ph)
    za, zb = z0[:, None, None].expand(P, V, R), z1[:, None, None].expand(P, V, R)
    start = torch.stack([r * c - h * s, r * s + h * c, za], -1)
    end = torch.stack([r * c + h * s, r * s - h * c, zb], -1)
    perm = ["PVR".index(a) for a in lor_order] + [3]
    start, end = start.permute(*perm).reshape(-1, 3), end.permute(*perm).reshape(-1, 3)
    if axial_axis == 0:
        start, end = start[:, [2, 0, 1]], end[:, [2, 0, 1]]
    return start.float().contiguous(), end.float().contiguous()


def image_geometry(torch, axial_axis):
    shape, vox = tuple(GEOMETRY["img"]), tuple(GEOMETRY["vox"])
    if axial_axis == 0:
        shape, vox = (shape[2], shape[0], shape[1]), (vox[2], vox[0], vox[1])
    vs = torch.tensor(vox, dtype=torch.float32)
    origin = -0.5 * (torch.tensor(shape, dtype=torch.float32) - 1) * vs
    return shape, vs, origin


def phantom(torch, shape, axial_axis):
    """cylinder (radius 200 mm) with a few hot spheres, deterministic"""
    vs = GEOMETRY["vox"]
    n0, n1, n2 = GEOMETRY["img"]
    x = (torch.arange(n0) - 0.5 * (n0 - 1)) * vs[0]
    y = (torch.arange(n1) - 0.5 * (n1 - 1)) * vs[1]
    z = (torch.arange(n2) - 0.5 * (n2 - 1)) * vs[2]
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    img = ((X**2 + Y**2) < 200**2).float()
    for cx, cy, cz, rr in [(60, 0, 0, 30), (-80, 50, 40, 20), (0, -100, -60, 25)]:
        img += 3 * (((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) < rr**2).float()
    if axial_axis == 0:
        img = img.permute(2, 0, 1)
    return img.contiguous()


def listmode_events(torch, xs, xe, axial_axis, tof, num_events, seed=1):
    """events: random LORs, emission point sampled along the LOR inside the phantom
    (weighted by the phantom), TOF bin from the emission point + Gaussian TOF noise"""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, xs.shape[0], (num_events,), generator=gen)
    a, b = xs[idx].double(), xe[idx].double()
    # rejection sampling of the emission point along the LOR (phantom value as weight)
    t = torch.rand(num_events, generator=gen, dtype=torch.float64)
    ia, ib = (1, 2) if axial_axis == 0 else (0, 1)  # transaxial components
    for _ in range(8):
        p = a + t[:, None] * (b - a)
        inside = (p[:, ia] ** 2 + p[:, ib] ** 2) < 200**2
        redo = ~inside
        t[redo] = torch.rand(int(redo.sum()), generator=gen, dtype=torch.float64)
    length = torch.linalg.norm(b - a, dim=1)
    # distance of the emission point from the LOR center towards the end point + TOF noise
    u = (t - 0.5) * length + torch.randn(num_events, generator=gen, dtype=torch.float64) * tof["sigma"]
    nb = tof["num_tof_bins"]
    it = torch.round(u / tof["tof_bin_width"] + 0.5 * (nb - 1)).clamp(0, nb - 1).short()
    return xs[idx].contiguous(), xe[idx].contiguous(), it.contiguous()


def run(args):
    import torch
    import parallelproj_core as pc

    dev = torch.device(args.device)
    on_gpu = dev.type == "cuda"
    version = pc.__version__
    print(f"parallelproj_core {version}, device {dev}")
    results = {"version": version, "device": str(dev), "timings": {}}
    if on_gpu:
        results["gpu"] = torch.cuda.get_device_name(dev)
    saved = {}

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

    for layout in args.layouts:
        lor_order, axial_axis = {"fast": ("RVP", 2), "stir": ("PVR", 0)}[layout]
        xs_c, xe_c = sinogram_lors(torch, lor_order, axial_axis, args.lor_fraction)
        shape, vs_c, origin_c = image_geometry(torch, axial_axis)
        img_c = phantom(torch, shape, axial_axis)
        num_lors = xs_c.shape[0]
        for arrays in (args.arrays if on_gpu else ["host"]):
            tdev = dev if arrays == "device" else torch.device("cpu")
            xs, xe, vs, origin, img = (t.to(tdev) for t in (xs_c, xe_c, vs_c, origin_c, img_c))
            for tof_name in args.tof:
                tof = TOF_PRESETS[tof_name]
                nb, w = tof["num_tof_bins"], tof["tof_bin_width"]
                sigma = torch.tensor([tof["sigma"]], device=tdev)
                offset = torch.zeros(1, device=tdev)
                y = torch.zeros((num_lors, nb), device=tdev)
                x = torch.zeros(shape, device=tdev)
                ev_s, ev_e, ev_it = listmode_events(torch, xs_c, xe_c, axial_axis, tof, args.num_events)
                ev_s, ev_e, ev_it = ev_s.to(tdev), ev_e.to(tdev), ev_it.to(tdev)
                y_lm = torch.zeros(args.num_events, device=tdev)
                ones_lm = torch.ones(args.num_events, device=tdev)
                for ns in args.num_sigmas:
                    key = f"{layout}/{arrays}/{tof_name}/ns{ns:g}"

                    def sino_fwd():
                        pc.joseph3d_tof_sino_fwd(xs, xe, img, origin, vs, y, w, sigma, offset, nb, ns)

                    def sino_back():
                        x.zero_()
                        pc.joseph3d_tof_sino_back(xs, xe, x, origin, vs, y, w, sigma, offset, nb, ns)

                    def lm_fwd():
                        pc.joseph3d_tof_lm_fwd(ev_s, ev_e, img, origin, vs, y_lm, w, sigma, offset, ev_it, nb, ns)

                    def lm_back():
                        x.zero_()
                        pc.joseph3d_tof_lm_back(ev_s, ev_e, x, origin, vs, ones_lm, w, sigma, offset, ev_it, nb, ns)

                    # results are stored for one layout / array location per TOF setting
                    save = layout == args.layouts[0] and arrays == (args.arrays[0] if on_gpu else "host")
                    t = {}
                    t["sino_fwd"] = best_time(sino_fwd)
                    if save:
                        saved[f"{key}/sino_fwd"] = y[:NUM_SAVE].cpu().numpy().copy()
                    t["sino_back"] = best_time(sino_back)  # back projects y = forward projection
                    if save:
                        saved[f"{key}/sino_back"] = x.cpu().numpy().copy()
                    t["lm_fwd"] = best_time(lm_fwd)
                    if save:
                        saved[f"{key}/lm_fwd"] = y_lm[:NUM_SAVE].cpu().numpy().copy()
                    t["lm_back"] = best_time(lm_back)
                    if save:
                        saved[f"{key}/lm_back"] = x.cpu().numpy().copy()
                    results["timings"][key] = t
                    print(f"  {key:28s} sino fwd {t['sino_fwd']:8.4f} s  back {t['sino_back']:8.4f} s   "
                          f"lm fwd {t['lm_fwd']:8.4f} s  back {t['lm_back']:8.4f} s   "
                          f"({num_lors / 1e6:.2f} M LORs, {nb} TOF bins, {args.num_events / 1e6:.1f} M events)")
                del y, x
                if on_gpu:
                    torch.cuda.empty_cache()

    np.savez_compressed(args.out, results=json.dumps(results), **{k.replace("/", "|"): v for k, v in saved.items()})
    print(f"saved {args.out}")


def compare(args):
    a, b = np.load(args.old), np.load(args.new)
    ra, rb = json.loads(str(a["results"])), json.loads(str(b["results"]))
    print(f"old: {ra['version']} ({ra['device']}, {ra.get('gpu', '')})")
    print(f"new: {rb['version']} ({rb['device']}, {rb.get('gpu', '')})")
    ops = ["sino_fwd", "sino_back", "lm_fwd", "lm_back"]
    print("\ntimings in s (old -> new, speed-up old / new)")
    print(f"{'case':28s}" + "".join(f"{op:>27s}" for op in ops))
    for key, ta in ra["timings"].items():
        tb = rb["timings"].get(key)
        if tb is None:
            continue
        cols = "".join(f"{ta[op]:9.4f} ->{tb[op]:8.4f} {ta[op] / tb[op]:5.1f}x" for op in ops)
        print(f"{key:28s}{cols}")
    print("\nmax |new - old| / max |old| of the results")
    for k in a.files:
        if k == "results" or k not in b.files:
            continue
        va, vb = a[k].astype(np.float64), b[k].astype(np.float64)
        scale = np.abs(va).max()
        rel = np.abs(vb - va).max() / scale if scale > 0 else float("nan")
        nan = "  NaN in old" if np.isnan(va).any() else ""
        nan += "  NaN in new" if np.isnan(vb).any() else ""
        print(f"  {k.replace('|', '/'):40s} {rel:9.2e}{nan}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="time and store results of the parallelproj_core on PYTHONPATH")
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--tof", nargs="+", default=["signa", "vision", "fine"], choices=sorted(TOF_PRESETS))
    r.add_argument("--num_sigmas", nargs="+", type=float, default=[3.0, 4.0])
    r.add_argument("--layouts", nargs="+", default=["fast", "stir"], choices=["fast", "stir"])
    r.add_argument("--arrays", nargs="+", default=["device", "host"], choices=["device", "host"])
    r.add_argument("--num_events", type=int, default=10_000_000)
    r.add_argument("--lor_fraction", type=float, default=1.0,
                   help="fraction of the views to use (e.g. 0.05 for quick CPU runs)")
    c = sub.add_parser("compare", help="compare two result files")
    c.add_argument("old")
    c.add_argument("new")
    args = p.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
