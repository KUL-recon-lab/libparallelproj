"""Scaling of the TOF listmode projectors with the number of events (sorted vs. random).

For every number of events (e.g. 1e6, 1e7, 1e8, 3e8), the script times
  - a TOF listmode forward and back projection of all events in random (time) order,
  - computing the sort keys (tof_lm_sort_key) and sorting the events,
  - the forward and back projection of the sorted events.

Large numbers of events are processed in chunks of at most --chunk events (like the subsets
of a listmode OSEM reconstruction): every chunk is sorted on its own and projected with one
call, and the back projections of all chunks are added into one image. This keeps the GPU
memory needed for sorting small (keys, permutation, sort buffers and the reordered copy of
one chunk). The events are generated directly on the device (float32), so that host memory
is not a limit either.

Events: random LORs of the SIGNA-like sinogram geometry of benchmark_tof_sinogram.py,
emission point uniform along the part of the LOR inside a cylinder of radius 200 mm, TOF bin
of the emission point plus Gaussian TOF noise.

Memory (GPU): ~26 bytes per event for the events (start, end: 2 x 12 bytes, TOF bin: 2
bytes) plus ~4 bytes per event for the forward projection, plus ~40 bytes per event of one
chunk for sorting; 3e8 events need ~9 GB + ~2 GB per 5e7 events of chunk.

usage (from the repository root):
  PYTHONPATH=build-cuda13 pixi run -e cuda13 python scripts/benchmark_tof_listmode_scaling.py
  PYTHONPATH=build-cuda13 pixi run -e cuda13 python scripts/benchmark_tof_listmode_scaling.py --num_events 1e6 1e7 1e8 3e8 --chunk 5e7
"""

import argparse
import math
import time

import torch
import parallelproj_core as pc

from benchmark_tof_sinogram import FWHM_TO_SIGMA, PRESETS, image_geometry, sinogram_lors


def make_events(xs, xe, n, nb, w, sigma, gen, dev):
    """n events on the device: (start, end, TOF bin); see the module docstring"""
    idx = torch.randint(0, xs.shape[0], (n,), generator=gen, device=dev)
    a, b = xs[idx], xe[idx]
    t = torch.rand(n, generator=gen, device=dev)
    for _ in range(10):  # rejection sampling of the emission point inside the cylinder
        p = a + t[:, None] * (b - a)
        redo = (p[:, 0] ** 2 + p[:, 1] ** 2) >= 200.0**2
        if not bool(redo.any()):
            break
        t = torch.where(redo, torch.rand(n, generator=gen, device=dev), t)
    length = torch.linalg.norm(b - a, dim=1)
    u = (t - 0.5) * length + torch.randn(n, generator=gen, device=dev) * sigma
    it = torch.round(u / w + 0.5 * (nb - 1)).clamp(0, nb - 1).to(torch.int16)
    return a.contiguous(), b.contiguous(), it


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preset", default="signa", choices=sorted(PRESETS))
    p.add_argument("--num_events", nargs="+", type=float, default=[1e6, 1e7, 1e8, 3e8])
    p.add_argument("--chunk", type=float, default=5e7, help="max. number of events per chunk (sorting and projection)")
    p.add_argument("--num_sigmas", type=float, default=3.0)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--skip_random", action="store_true", help="do not time the projections in random order")
    args = p.parse_args()

    dev = torch.device(args.device)
    on_gpu = dev.type == "cuda"
    cfg = PRESETS[args.preset]
    nb, w = cfg["num_tof_bins"], cfg["tof_bin_width"]
    sig = cfg["fwhm"] * FWHM_TO_SIGMA
    img_shape, vox = image_geometry(cfg, 2)
    vs = torch.tensor(vox, dtype=torch.float32, device=dev)
    origin = -0.5 * (torch.tensor(img_shape, dtype=torch.float32, device=dev) - 1) * vs
    xs, xe = (t.to(dev) for t in sinogram_lors(cfg, "RVP", 2))
    sigma_d = torch.tensor([sig], dtype=torch.float32, device=dev)
    off_d = torch.zeros(1, dtype=torch.float32, device=dev)

    # phantom image: cylinder of radius 200 mm
    X = origin[0] + torch.arange(img_shape[0], device=dev) * vs[0]
    Y = origin[1] + torch.arange(img_shape[1], device=dev) * vs[1]
    img = ((X[:, None] ** 2 + Y[None, :] ** 2) < 200.0**2).float()[:, :, None].expand(img_shape).contiguous()

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

    print(f"parallelproj_core {pc.__version__}, device {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if on_gpu else ""))
    print(f"preset {args.preset}: {nb} TOF bins of {w:.2f} mm, sigma {sig:.2f} mm, {args.num_sigmas:g} sigma, "
          f"image {img_shape}, chunks of at most {args.chunk:.0e} events")
    header = (f"{'events':>8s} {'chunks':>6s} | {'random fwd':>10s} {'random back':>11s} | {'key+sort':>9s} | "
              f"{'sorted fwd':>10s} {'sorted back':>11s} | {'ns/event fwd':>12s} {'back':>6s} | {'speed-up fwd':>12s} {'back':>5s}")
    print(header)
    print("-" * len(header))

    gen = torch.Generator(device=dev).manual_seed(1)
    x = torch.zeros(img_shape, dtype=torch.float32, device=dev)
    for n_float in args.num_events:
        n = int(n_float)
        chunk = int(args.chunk)
        sizes = [min(chunk, n - k) for k in range(0, n, chunk)]
        try:
            chunks = [make_events(xs, xe, m, nb, w, sig, gen, dev) for m in sizes]
            ys = [torch.zeros(m, dtype=torch.float32, device=dev) for m in sizes]
        except torch.OutOfMemoryError:
            print(f"{n:8.0e}  out of GPU memory when creating the events - skipped")
            continue

        def fwd(cs):
            for (s, e, tb), y in zip(cs, ys):
                pc.joseph3d_tof_lm_fwd(s, e, img, origin, vs, y, w, sigma_d, off_d, tb, nb, num_sigmas=args.num_sigmas)

        def back(cs):
            x.zero_()
            for (s, e, tb), y in zip(cs, ys):
                pc.joseph3d_tof_lm_back(s, e, x, origin, vs, y, w, sigma_d, off_d, tb, nb, num_sigmas=args.num_sigmas)

        if args.skip_random:
            t_rf = t_rb = float("nan")
        else:
            t_rf, t_rb = best_time(lambda: fwd(chunks)), best_time(lambda: back(chunks))

        # sort every chunk once: key, argsort, reorder (the original chunk is replaced)
        sync()
        t0 = time.perf_counter()
        for k, (s, e, tb) in enumerate(chunks):
            key = torch.empty(s.shape[0], dtype=torch.int64, device=dev)
            pc.tof_lm_sort_key(s, e, origin, vs, key, w, off_d, tb, nb)
            order = torch.argsort(key)
            del key
            chunks[k] = (s[order], e[order], tb[order])
            del s, e, tb, order
        sync()
        t_sort = time.perf_counter() - t0

        t_sf, t_sb = best_time(lambda: fwd(chunks)), best_time(lambda: back(chunks))
        print(f"{n:8.0e} {len(sizes):6d} | {t_rf:10.4f} {t_rb:11.4f} | {t_sort:9.4f} | {t_sf:10.4f} {t_sb:11.4f} | "
              f"{1e9 * t_sf / n:12.2f} {1e9 * t_sb / n:6.2f} | {t_rf / t_sf:11.1f}x {t_rb / t_sb:4.1f}x")
        del chunks, ys
        if on_gpu:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
