#!/usr/bin/env python3
"""Manually verify that a failed device allocation inside the CUDA lib
surfaces as a Python RuntimeError (and not as a silent wrong result).

Strategy (WSL / Linux, GPU with enough VRAM):
  1. Use cupy to hog almost all free VRAM (allocated in chunks to avoid
     fragmentation issues), leaving only a small amount free.
  2. Call pp.joseph3d_fwd() with HOST (numpy) xstart/xend arrays that are
     bigger than the remaining free VRAM. handle_cuda_input_array() then
     fails at cudaMalloc when trying to create the device copy of lor_start
     -> std::runtime_error -> nanobind -> Python RuntimeError.
  3. Verify the GPU is still usable afterwards.

Host RAM usage stays small (~2-3 GB) because the host arrays only need to be
slightly larger than the VRAM we left free - not larger than total VRAM.

Run manually with:  python wip/trigger_cuda_malloc_error.py
"""

import numpy as np
import cupy as cp
import parallelproj_core as pp

GiB = 1024**3
LEAVE_FREE = 512 * 1024**2  # VRAM to leave unallocated (512 MiB)
HOG_CHUNK = 1 * GiB         # hog VRAM in 1 GiB chunks

print(f"parallelproj_core {pp.__version__}, cuda_enabled={pp.cuda_enabled}")
assert pp.cuda_enabled, "this script needs a CUDA build of parallelproj_core"

# ---------------------------------------------------------------------------
# 1. hog almost all free VRAM with cupy
# ---------------------------------------------------------------------------
free0, total = cp.cuda.runtime.memGetInfo()
print(f"VRAM before hogging: free {free0 / GiB:.2f} GiB / total {total / GiB:.2f} GiB")

hogs = []  # keep references so the allocations stay alive
while True:
    free, _ = cp.cuda.runtime.memGetInfo()
    chunk = min(HOG_CHUNK, free - LEAVE_FREE)
    if chunk <= 0:
        break
    try:
        # raw allocation, bypasses the cupy memory pool
        hogs.append(cp.cuda.alloc(chunk))
    except cp.cuda.memory.OutOfMemoryError:
        break

free, _ = cp.cuda.runtime.memGetInfo()
print(f"VRAM after hogging:  free {free / GiB:.2f} GiB ({len(hogs)} chunks held)")

# ---------------------------------------------------------------------------
# 2. host arrays: mini random image, but more LOR endpoints than fit in the
#    remaining VRAM. joseph3d_fwd device-copies lor_start first -> cudaMalloc
#    must fail there. Coordinate values are irrelevant (the kernel never
#    runs), so use np.empty to keep this fast.
# ---------------------------------------------------------------------------
img_dim = (8, 8, 8)
rng = np.random.default_rng(0)
img = rng.random(img_dim, dtype=np.float32)
voxsize = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
img_origin = (-0.5 * np.asarray(img_dim, dtype=np.float32) + 0.5) * voxsize

# one endpoint array should need ~2x the remaining free VRAM
target_bytes = 2 * free
num_lors = int(target_bytes) // (3 * 4)  # 3 float32 per endpoint
print(f"using {num_lors:,} LORs -> xstart/xend each "
      f"{num_lors * 12 / GiB:.2f} GiB (host RAM)")

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
    raise SystemExit("FAIL: joseph3d_fwd did not raise - error not propagated!")

# ---------------------------------------------------------------------------
# 4. GPU must still be usable afterwards (no leaked allocation, no corrupted
#    context): release the hog and do a real projection
# ---------------------------------------------------------------------------
hogs.clear()
# cp.cuda.alloc goes through cupy's memory pool, which keeps freed blocks
# cached - explicitly return them to the driver so memGetInfo shows reality
cp.get_default_memory_pool().free_all_blocks()
free, _ = cp.cuda.runtime.memGetInfo()
print(f"\nVRAM after releasing hog: free {free / GiB:.2f} GiB")

n_ok = 1000
xstart_ok = rng.random((n_ok, 3), dtype=np.float32) * 16.0 - 8.0
xend_ok = rng.random((n_ok, 3), dtype=np.float32) * 16.0 - 8.0
img_fwd_ok = np.zeros(n_ok, dtype=np.float32)
pp.joseph3d_fwd(xstart_ok, xend_ok, img, img_origin, voxsize, img_fwd_ok)
print(f"OK: follow-up projection ran fine "
      f"(img_fwd mean = {img_fwd_ok.mean():.4f}, nonzero = {(img_fwd_ok != 0).sum()})")

print("\nall checks passed")
