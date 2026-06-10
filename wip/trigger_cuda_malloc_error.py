#!/usr/bin/env python3
"""Manually verify that a failed device allocation inside the CUDA lib
surfaces as a Python RuntimeError (and not as a silent wrong result).

Strategy:
  Call pp.joseph3d_fwd() with HOST (numpy) xstart/xend arrays sized such
  that the device copy of xstart (made first inside the lib) is bigger than
  the total VRAM, while all host arrays combined stay below 75% of host RAM.
  handle_cuda_input_array() then fails at cudaMalloc
  -> std::runtime_error -> nanobind -> Python RuntimeError.

Host memory per LOR: xstart 12 B + xend 12 B + img_fwd 4 B = 28 B.
The device copy of xstart needs 12 B per LOR, so the test is feasible when
0.75 * RAM / 28 * 12 > VRAM (e.g. 64 GB RAM / 20 GB VRAM: ~20.6 GB > 20 GB).

NOTE: on Windows / WSL2 the WDDM driver can page CUDA allocations into
system RAM ("CUDA - Sysmem Fallback Policy"), so a marginally oversized
allocation may still succeed there. Run on native Linux for a guaranteed
failure, or set the NVIDIA Control Panel policy to "Prefer No Sysmem
Fallback".

Run manually with:  python wip/trigger_cuda_malloc_error.py
"""

import os
import numpy as np
import cupy as cp
import parallelproj_core as pp

GiB = 1024**3
RAM_FRACTION = 0.75
HOST_BYTES_PER_LOR = 28  # xstart (12) + xend (12) + img_fwd (4)
DEV_BYTES_PER_LOR = 12   # device copy of xstart, allocated first in the lib

print(f"parallelproj_core {pp.__version__}, cuda_enabled={pp.cuda_enabled}")
assert pp.cuda_enabled, "this script needs a CUDA build of parallelproj_core"

free_vram, total_vram = cp.cuda.runtime.memGetInfo()
total_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
print(f"VRAM: free {free_vram / GiB:.2f} / total {total_vram / GiB:.2f} GiB, "
      f"host RAM: {total_ram / GiB:.2f} GiB")

# ---------------------------------------------------------------------------
# 1. choose the number of LORs: max 75% of RAM for all host arrays combined,
#    and the 12-byte-per-LOR device copy of xstart must exceed total VRAM
# ---------------------------------------------------------------------------

# factor 2 needed to trigger error on WSL
num_lors = int(2* RAM_FRACTION * total_ram) // HOST_BYTES_PER_LOR
dev_request = num_lors * DEV_BYTES_PER_LOR

print(f"using {num_lors:,} LORs:")
print(f"  host arrays combined: {num_lors * HOST_BYTES_PER_LOR / GiB:.2f} GiB "
      f"(<= {RAM_FRACTION:.0%} of RAM)")
print(f"  device request for xstart: {dev_request / GiB:.2f} GiB "
      f"(total VRAM: {total_vram / GiB:.2f} GiB)")

if dev_request <= total_vram:
    raise SystemExit(
        "FAIL: cannot trigger the error on this machine: "
        f"{RAM_FRACTION:.0%} of RAM does not allow a device request larger "
        "than total VRAM. Needs RAM > VRAM * 28/12 / "
        f"{RAM_FRACTION} = {total_vram * 28 / 12 / RAM_FRACTION / GiB:.1f} GiB."
    )
if dev_request < 1.2 * total_vram:
    print("  WARNING: device request is less than 1.2x total VRAM - "
          "on WSL/Windows, sysmem fallback may let it succeed")

# ---------------------------------------------------------------------------
# 2. mini random image + oversized host LOR arrays (values irrelevant: the
#    cudaMalloc fails before the kernel ever runs)
# ---------------------------------------------------------------------------
img_dim = (8, 8, 8)
rng = np.random.default_rng(0)
img = rng.random(img_dim, dtype=np.float32)
voxsize = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
img_origin = (-0.5 * np.asarray(img_dim, dtype=np.float32) + 0.5) * voxsize

print("allocating host arrays ...")
xstart = np.empty((num_lors, 3), dtype=np.float32)
xend = np.empty((num_lors, 3), dtype=np.float32)
img_fwd = np.zeros(num_lors, dtype=np.float32)

# ---------------------------------------------------------------------------
# 3. the call must raise RuntimeError mentioning cudaMalloc
# ---------------------------------------------------------------------------
try:
    pp.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd)
except RuntimeError as e:
    print(f"\nOK: got expected RuntimeError:\n    {e}")
    assert "cudaMalloc" in str(e), "exception raised, but not from cudaMalloc?"
else:
    raise SystemExit("FAIL: joseph3d_fwd did not raise - error not propagated! "
                     "(on WSL/Windows this can be due to sysmem fallback)")

del xstart, xend, img_fwd

# ---------------------------------------------------------------------------
# 4. GPU must still be usable afterwards (no leaked allocation, no corrupted
#    context): run a real projection
# ---------------------------------------------------------------------------
free_vram, _ = cp.cuda.runtime.memGetInfo()
print(f"\nVRAM after failed call: free {free_vram / GiB:.2f} GiB")

n_ok = int(1e6)
xstart_ok = rng.random((n_ok, 3), dtype=np.float32) * 16.0 - 8.0
xend_ok = rng.random((n_ok, 3), dtype=np.float32) * 16.0 - 8.0
img_fwd_ok = np.zeros(n_ok, dtype=np.float32)
pp.joseph3d_fwd(xstart_ok, xend_ok, img, img_origin, voxsize, img_fwd_ok)
print(f"OK: follow-up projection ran fine "
      f"(img_fwd mean = {img_fwd_ok.mean():.4f}, nonzero = {(img_fwd_ok != 0).sum()})")

print("\nall checks passed")
