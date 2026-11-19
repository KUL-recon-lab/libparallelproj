"""Benchmark of the TOF sinogram projectors for realistic scanner settings.

Times the TOF sinogram forward and back projection of one image and compares them with
the non-TOF projection along the same LORs (the cost of the image access, a lower bound).

Presets (--preset):
  signa       ~1/28 of the sinogram LORs of a SIGNA PET/MR-like scanner (5.5 M LORs),
              27 TOF bins of 169 ps, CTR 400 ps (TOF bin width ~ sigma)
  signa_fine  as signa with TOF bins of sigma / 5 (134 bins)
  ctr50       CTR 50 ps, TOF bin width = sigma (215 bins)
  small       a small geometry for quick CPU runs

Sinogram order (--lor_order, default RVP: planes fastest) and image layout
(--axial_axis, default 2: axial direction fastest in memory) have a large effect on GPUs,
see the "Performance" section of the documentation.

usage:
  python scripts/benchmark_tof_sinogram.py --device cuda:0 --preset signa
  python scripts/benchmark_tof_sinogram.py --device cpu --preset small

Needs PyTorch (array backend and GPU synchronization).
"""

import argparse
import math
import time

import torch
import parallelproj_core as pc

C_MM_PER_PS = 0.299792458 / 2  # mm along the LOR per ps of arrival time difference
FWHM_TO_SIGMA = 1 / (2 * math.sqrt(2 * math.log(2)))

_SIGNA = dict(radius=311.8, n_views=224, n_rad=351, rad_half=300.0, n_rings=45, ring_spacing=5.56,
              n_ring_pairs=70, img=(256, 256, 89), vox=(2.34, 2.34, 2.78),
              tof_bin_width=169 * C_MM_PER_PS, num_tof_bins=27, fwhm=400 * C_MM_PER_PS)
_sigma = _SIGNA["fwhm"] * FWHM_TO_SIGMA
_tof_range = _SIGNA["tof_bin_width"] * _SIGNA["num_tof_bins"]
_fwhm50 = 50 * C_MM_PER_PS
PRESETS = {
    "signa": _SIGNA,
    "signa_fine": dict(_SIGNA, tof_bin_width=_sigma / 5, num_tof_bins=int(round(_tof_range / (_sigma / 5)))),
    "ctr50": dict(_SIGNA, fwhm=_fwhm50, tof_bin_width=_fwhm50 * FWHM_TO_SIGMA,
                  num_tof_bins=int(math.ceil(_tof_range / (_fwhm50 * FWHM_TO_SIGMA)))),
    "small": dict(radius=300.0, n_views=96, n_rad=129, rad_half=150.0, n_rings=8, ring_spacing=4.0,
                  n_ring_pairs=16, img=(96, 96, 32), vox=(2.5, 2.5, 2.5),
                  tof_bin_width=169 * C_MM_PER_PS, num_tof_bins=27, fwhm=400 * C_MM_PER_PS),
}


def sinogram_lors(cfg, lor_order="RVP", axial_axis=2):
    """LOR end points of a cylindrical scanner (views x radial bins x ring pairs)

    lor_order: memory order of the LORs, axes from slowest to fastest varying:
               P = planes (ring pairs), V = views, R = radial bins (e.g. "RVP": planes
               fastest, the fastest order on GPUs with axial_axis = 2; "PVR": radial bins
               fastest)
    axial_axis: image axis (0 or 2) along the scanner axis; the axial coordinate is
                stored in this component of the LOR end points
    """
    radius = cfg["radius"]
    phi = torch.arange(cfg["n_views"], dtype=torch.float64) * math.pi / cfg["n_views"]
    rad = torch.linspace(-cfg["rad_half"], cfg["rad_half"], cfg["n_rad"], dtype=torch.float64)
    zr = (torch.arange(cfg["n_rings"], dtype=torch.float64) - 0.5 * (cfg["n_rings"] - 1)) * cfg["ring_spacing"]
    pairs = sorted(((a, b) for a in range(cfg["n_rings"]) for b in range(cfg["n_rings"])),
                   key=lambda q: (abs(q[0] - q[1]), q[0]))  # direct planes first
    step = len(pairs) / cfg["n_ring_pairs"]
    pairs = [pairs[int(k * step)] for k in range(cfg["n_ring_pairs"])]
    z0 = torch.tensor([zr[a] for a, _ in pairs], dtype=torch.float64)
    z1 = torch.tensor([zr[b] for _, b in pairs], dtype=torch.float64)

    # arrays of shape (planes, views, radial bins)
    P, V, R = len(pairs), phi.numel(), rad.numel()
    ph = phi[None, :, None].expand(P, V, R)
    r = rad[None, None, :].expand(P, V, R)
    h = torch.sqrt(radius**2 - r**2)
    c, s = torch.cos(ph), torch.sin(ph)
    za = z0[:, None, None].expand(P, V, R)
    zb = z1[:, None, None].expand(P, V, R)
    start = torch.stack([r * c - h * s, r * s + h * c, za], -1)
    end = torch.stack([r * c + h * s, r * s - h * c, zb], -1)

    perm = ["PVR".index(a) for a in lor_order.upper()] + [3]
    start = start.permute(*perm).reshape(-1, 3)
    end = end.permute(*perm).reshape(-1, 3)
    if axial_axis == 0:  # (x, y, z) -> (z, x, y)
        start, end = start[:, [2, 0, 1]], end[:, [2, 0, 1]]
    return start.float().contiguous(), end.float().contiguous()


def image_geometry(cfg, axial_axis=2):
    """image shape and voxel size (preset: axial axis last) for the given axial axis"""
    shape, vox = tuple(cfg["img"]), tuple(cfg["vox"])
    if axial_axis == 0:
        shape, vox = (shape[2], shape[0], shape[1]), (vox[2], vox[0], vox[1])
    return shape, vox


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preset", default="signa", choices=sorted(PRESETS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--num_sigmas", type=float, default=3.0)
    p.add_argument("--lor_order", default="RVP",
                   help="memory order of the sinogram LORs, slowest to fastest: P planes, V views, R radial (PVR, RVP, VRP, ...)")
    p.add_argument("--axial_axis", type=int, default=2, choices=[0, 2],
                   help="image axis along the scanner axis (2: axial fastest in memory, 0: slowest)")
    args = p.parse_args()

    cfg = PRESETS[args.preset]
    dev = torch.device(args.device)
    img_shape, vox = image_geometry(cfg, args.axial_axis)
    vs = torch.tensor(vox, dtype=torch.float32)
    origin = -0.5 * (torch.tensor(img_shape, dtype=torch.float32) - 1) * vs
    ls, le = sinogram_lors(cfg, args.lor_order, args.axial_axis)
    num_lors = ls.shape[0]
    nb, w = cfg["num_tof_bins"], cfg["tof_bin_width"]
    sig = cfg["fwhm"] * FWHM_TO_SIGMA
    sigma, offset = torch.tensor([sig]), torch.tensor([0.0])
    ls, le, origin, vs, sigma, offset = (t.to(dev) for t in (ls, le, origin, vs, sigma, offset))
    img = torch.rand(img_shape, generator=torch.Generator().manual_seed(0)).to(dev)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def best_time(fn):
        fn()  # warm-up
        times = []
        for _ in range(args.repeats):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            times.append(time.perf_counter() - t0)
        return min(times)

    y_nt = torch.zeros(num_lors, device=dev)
    y = torch.zeros((num_lors, nb), device=dev)
    x = torch.zeros(img_shape, device=dev)

    def nontof_fwd():
        pc.joseph3d_fwd(ls, le, img, origin, vs, y_nt)

    def nontof_back():
        x.zero_()
        pc.joseph3d_back(ls, le, x, origin, vs, y_nt)

    def tof_fwd():
        pc.joseph3d_tof_sino_fwd(ls, le, img, origin, vs, y, w, sigma, offset, nb, num_sigmas=args.num_sigmas)

    def tof_back():
        x.zero_()
        pc.joseph3d_tof_sino_back(ls, le, x, origin, vs, y, w, sigma, offset, nb, num_sigmas=args.num_sigmas)

    print(f"parallelproj_core {pc.__version__}, device {dev}")
    print(f"preset {args.preset}: {num_lors / 1e6:.2f} M LORs (order {args.lor_order}), image {img_shape} "
          f"(axial axis {args.axial_axis}), voxel {vox} mm")
    print(f"TOF: {nb} bins of {w:.2f} mm, FWHM {cfg['fwhm']:.2f} mm (sigma {sig:.2f} mm = {sig / w:.2f} bins), "
          f"{args.num_sigmas:g} sigma, TOF sinogram {num_lors * nb * 4 / 1e9:.2f} GB")
    t_nf, t_tf = best_time(nontof_fwd), best_time(tof_fwd)
    t_nb, t_tb = best_time(nontof_back), best_time(tof_back)
    print(f"  forward:  TOF {t_tf:8.4f} s   non-TOF {t_nf:8.4f} s   TOF / non-TOF {t_tf / t_nf:6.2f}")
    print(f"  back:     TOF {t_tb:8.4f} s   non-TOF {t_nb:8.4f} s   TOF / non-TOF {t_tb / t_nb:6.2f}")


if __name__ == "__main__":
    main()
