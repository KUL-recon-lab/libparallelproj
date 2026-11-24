"""Batch benchmark (#47): B data sets that share the same projector (e.g. a mini-batch in a
PyTorch network that computes Poisson log-likelihood gradients and their Hessian products).

Setup (default preset "signa_dl"): one subset (1 of 28, every 28th view) of the full SIGNA
PET/MR sinogram (span 1: all 45^2 = 2025 ring pairs, 351 radial bins, 8 views = 5.7 M LORs),
27 TOF bins of 169 ps, CTR 400 ps, image 192 x 192 x 250 with voxels of 3.125 x 3.125 x 1 mm.

Timed for every batch size B (best of --repeats):
  - TOF sinogram forward / back projection and non-TOF forward / back projection of B images,
  - a complete Poisson log-likelihood "gradient layer" per batch (TOF):
        expected = m * A x + s
        grad     = A^T (m * y / expected) - sens                        (forward pass; the
                                                                        sensitivity image
                                                                        sens = A^T m is
                                                                        precomputed)
        hvp      = A^T (m^2 * y / expected^2 * A v)                    (backward pass:
                                                                        Hessian times v)
    i.e. 2 forward and 2 back projections per data set,
each as
  - "loop": one library call per data set (what is possible today),
  - "batch": one batched call for all B data sets, if the library provides batched
    projectors (joseph3d_*_batch, prototype of #47; skipped otherwise).

usage (from the repository root):
  PYTHONPATH=build-cuda13 pixi run -e cuda13 python scripts/benchmark_batch.py
  PYTHONPATH=build-cuda13 pixi run -e cuda13 python scripts/benchmark_batch.py --batch_sizes 1 2 3 5 --voxel_size 3.125 3.125 1.0
"""

import argparse
import math
import time

import torch
import parallelproj_core as pc

from benchmark_tof_sinogram import FWHM_TO_SIGMA, PRESETS, sinogram_lors

PRESETS["signa_dl"] = dict(PRESETS["signa_subset"], img=(192, 192, 250), vox=(3.125, 3.125, 1.0))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preset", default="signa_dl", choices=sorted(PRESETS))
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 3, 5])
    p.add_argument("--image_shape", nargs=3, type=int, default=None)
    p.add_argument("--voxel_size", nargs=3, type=float, default=None)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    dev = torch.device(args.device)
    cfg = dict(PRESETS[args.preset])
    if args.image_shape:
        cfg["img"] = tuple(args.image_shape)
    if args.voxel_size:
        cfg["vox"] = tuple(args.voxel_size)
    shape = tuple(cfg["img"])
    vs = torch.tensor(cfg["vox"], dtype=torch.float32, device=dev)
    origin = -0.5 * (torch.tensor(shape, dtype=torch.float32, device=dev) - 1) * vs
    xs, xe = (t.to(dev) for t in sinogram_lors(cfg, "RVP", 2))
    nl = xs.shape[0]
    nb, w = cfg["num_tof_bins"], cfg["tof_bin_width"]
    sigma = torch.tensor([cfg["fwhm"] * FWHM_TO_SIGMA], device=dev)
    off = torch.zeros(1, device=dev)
    has_batch = hasattr(pc, "joseph3d_tof_sino_fwd_batch")

    def sync():
        if dev.type == "cuda":
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

    # projectors for a batch of images x [B, n0, n1, n2] -> data [B, nl, nb] (TOF) / [B, nl]
    def fwd_tof_loop(x, out):
        for b in range(x.shape[0]):
            pc.joseph3d_tof_sino_fwd(xs, xe, x[b], origin, vs, out[b], w, sigma, off, nb)

    def back_tof_loop(y, out):
        out.zero_()
        for b in range(y.shape[0]):
            pc.joseph3d_tof_sino_back(xs, xe, out[b], origin, vs, y[b], w, sigma, off, nb)

    def fwd_nt_loop(x, out):
        for b in range(x.shape[0]):
            pc.joseph3d_fwd(xs, xe, x[b], origin, vs, out[b])

    def back_nt_loop(y, out):
        out.zero_()
        for b in range(y.shape[0]):
            pc.joseph3d_back(xs, xe, out[b], origin, vs, y[b])

    def fwd_tof_batch(x, out):
        pc.joseph3d_tof_sino_fwd_batch(xs, xe, x, origin, vs, out, w, sigma, off, nb)

    def back_tof_batch(y, out):
        out.zero_()
        pc.joseph3d_tof_sino_back_batch(xs, xe, out, origin, vs, y, w, sigma, off, nb)

    def fwd_nt_batch(x, out):
        pc.joseph3d_fwd_batch(xs, xe, x, origin, vs, out)

    def back_nt_batch(y, out):
        out.zero_()
        pc.joseph3d_back_batch(xs, xe, out, origin, vs, y)

    print(f"parallelproj_core {pc.__version__}, " + (torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"))
    print(f"preset {args.preset}: {nl / 1e6:.2f} M LORs, {nb} TOF bins, image {shape}, voxel {cfg['vox']} mm; "
          f"batched projectors available: {has_batch}")
    gen = torch.Generator(device=dev).manual_seed(0)
    rows = []

    def run(B):
        """time all variants for batch size B"""
        x = torch.rand((B, *shape), generator=gen, device=dev)
        v = torch.rand((B, *shape), generator=gen, device=dev)
        y = torch.poisson(torch.full((B, nl, nb), 0.5, device=dev), generator=gen)
        m = torch.rand((B, nl, 1), generator=gen, device=dev) * 0.5 + 0.5  # attenuation x normalization per LOR
        s = 0.05  # scatter + randoms (a constant here, saves memory)
        p = torch.empty((B, nl, nb), device=dev)
        p2 = torch.empty((B, nl, nb), device=dev)
        p_nt = torch.empty((B, nl), device=dev)
        img = torch.empty((B, *shape), device=dev)
        img2 = torch.empty((B, *shape), device=dev)
        sens = torch.rand((B, *shape), generator=gen, device=dev)  # precomputed (read from disk)

        def layer(fwd, back):
            fwd(x, p)  # A x
            expected = m * p + s
            back(m * y / expected, img)
            grad = img - sens  # gradient (sensitivity image precomputed)
            fwd(v, p2)  # A v
            back(m * m * y / (expected * expected) * p2, img2)  # Hessian times v

        variants = [("loop", fwd_tof_loop, back_tof_loop, fwd_nt_loop, back_nt_loop)]
        if has_batch:
            variants.append(("batch", fwd_tof_batch, back_tof_batch, fwd_nt_batch, back_nt_batch))
        for name, f_t, b_t, f_n, b_n in variants:
            t = dict(B=B, variant=name)
            t["tof_fwd"] = best_time(lambda: f_t(x, p))
            t["tof_back"] = best_time(lambda: b_t(p, img))
            t["nt_fwd"] = best_time(lambda: f_n(x, p_nt))
            t["nt_back"] = best_time(lambda: b_n(p_nt, img))
            t["layer"] = best_time(lambda: layer(f_t, b_t))
            rows.append(t)
            print(f"B={B} {name:5s}  TOF fwd {t['tof_fwd']:.4f} s  TOF back {t['tof_back']:.4f} s   "
                  f"non-TOF fwd {t['nt_fwd']:.4f} s  back {t['nt_back']:.4f} s   "
                  f"Poisson grad + Hessian layer (TOF) {t['layer']:.4f} s  ({t['layer'] / B:.4f} s per data set)")

    for B in args.batch_sizes:
        try:
            run(B)
        except torch.OutOfMemoryError:
            print(f"B={B}: out of memory - skipped")
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if has_batch:
        print("\nspeed-up batch vs. loop:")
        for B in args.batch_sizes:
            lo = next(r for r in rows if r["B"] == B and r["variant"] == "loop")
            ba = next(r for r in rows if r["B"] == B and r["variant"] == "batch")
            print(f"B={B}: " + "  ".join(f"{k} {lo[k] / ba[k]:.2f}x" for k in ("tof_fwd", "tof_back", "nt_fwd", "nt_back",
                                                                             "layer")))


if __name__ == "__main__":
    main()
