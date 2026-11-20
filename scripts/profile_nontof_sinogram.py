"""Minimal driver for profiling the non-TOF sinogram projectors with Nsight Compute (#53).

Runs one warm-up and one profiled call each of joseph3d_fwd and joseph3d_back for the LORs
of a preset of benchmark_tof_sinogram.py (default: signa, 5.5 M LORs, image 256x256x89),
in a chosen LOR order / image layout (default: the fast layout, planes fastest and axial
image axis last). Prints the wall-clock times, so it can also be run without the profiler.

usage (repository root):
  # without profiler (timing only)
  PYTHONPATH=build-cuda13 pixi run -e cuda13 python scripts/profile_nontof_sinogram.py
  # full Nsight Compute report of the 2nd (non-warm-up) call of each kernel
  PYTHONPATH=build-cuda13 pixi run -e cuda13 ncu --set full -k regex:"joseph3d_(fwd|back)_kernel" \\
      --launch-skip 2 --launch-count 2 -o nontof_sino -f \\
      python scripts/profile_nontof_sinogram.py
"""

import argparse
import time

import torch
import parallelproj_core as pc

from benchmark_tof_sinogram import PRESETS, image_geometry, sinogram_lors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preset", default="signa", choices=sorted(PRESETS))
    p.add_argument("--lor_order", default="RVP")
    p.add_argument("--axial_axis", type=int, default=2, choices=[0, 2])
    p.add_argument("--threads_per_block", type=int, default=64)
    args = p.parse_args()

    dev = torch.device(args.device)
    cfg = PRESETS[args.preset]
    img_shape, vox = image_geometry(cfg, args.axial_axis)
    vs = torch.tensor(vox, dtype=torch.float32, device=dev)
    origin = -0.5 * (torch.tensor(img_shape, dtype=torch.float32, device=dev) - 1) * vs
    xs, xe = (t.to(dev) for t in sinogram_lors(cfg, args.lor_order, args.axial_axis))
    img = torch.rand(img_shape, generator=torch.Generator().manual_seed(0)).to(dev)
    y = torch.zeros(xs.shape[0], device=dev)
    x = torch.zeros(img_shape, device=dev)

    print(f"parallelproj_core {pc.__version__}, {torch.cuda.get_device_name(dev)}, {xs.shape[0] / 1e6:.2f} M LORs "
          f"(order {args.lor_order}), image {img_shape} (axial axis {args.axial_axis})")
    tpb = args.threads_per_block
    # order of the kernel launches: fwd (warm-up), back (warm-up), fwd, back
    fns = {"fwd": lambda: pc.joseph3d_fwd(xs, xe, img, origin, vs, y, threads_per_block=tpb),
           "back": lambda: pc.joseph3d_back(xs, xe, x, origin, vs, y, threads_per_block=tpb)}
    for k in range(2):
        for name, fn in fns.items():
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(dev)
            if k == 1:
                print(f"joseph3d_{name}: {time.perf_counter() - t0:.4f} s")


if __name__ == "__main__":
    main()
