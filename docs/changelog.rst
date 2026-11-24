Changelog
=========

v3.0.0 (unreleased)
-------------------

3.0 bundles the changes planned for 2.1 (never released) with CUDA stream
support, which changes the C API signatures (ABI). Code that includes
``parallelproj.h`` compiles unchanged, but has to be rebuilt.

Breaking changes
^^^^^^^^^^^^^^^^

* **C API: new trailing parameter ``void *stream = nullptr`` of all projector
  functions** (see "Added"). Existing calls compile unchanged and keep their
  synchronous behavior, but the ABI changed: the shared library version (SONAME)
  is now ``libparallelproj.so.3``, and binaries built against 2.x must be rebuilt.
  Code that stores typed function pointers to the projector functions, or
  declares them itself, needs to add the parameter.
  (`#41 <https://github.com/KUL-recon-lab/libparallelproj/issues/41>`_)
* **Changed numerical results** (bug fixes, see "Fixed"): non-zero TOF center
  offsets for LORs with ``lor_end[dir] < lor_start[dir]``; TOF listmode
  projections (~0.1 % for typical TOF kernels, more for kernels that are narrow
  compared to the TOF bin width); TOF sinogram back projection of signed data.
* **Changed numerical results: truncated TOF kernel in all TOF projectors** (see
  "Changed"). Differences to 2.x are up to ~0.3 % of the maximum of a TOF
  projection, at the edges of the TOF window; TOF listmode projections of kernels
  that are narrow compared to the voxels change more (they now equal the TOF
  sinogram bins).
  (`#43 <https://github.com/KUL-recon-lab/libparallelproj/issues/43>`_)
* **Removed the compile-time macro ``MAX_NUM_TOF_WEIGHTS``** (``utils.h``). The
  maximum TOF window of 64 bins is now fixed.
  (`#43 <https://github.com/KUL-recon-lab/libparallelproj/issues/43>`_)
* **Python bindings:** ``device_id`` defaults to -1 (automatic) and must match
  the GPU of GPU input arrays if given (see "Changed").

Added
^^^^^

* **TOF projectors with a tabulated (non-Gaussian, possibly asymmetric) TOF kernel:**
  ``joseph3d_tof_sino_fwd_tab``, ``joseph3d_tof_sino_back_tab``,
  ``joseph3d_tof_lm_fwd_tab`` and ``joseph3d_tof_lm_back_tab`` (C API and Python). The
  kernel is given as a table of its cumulative distribution F(u) on a uniform grid from
  ``u_min`` to ``u_max`` in spatial units (u positive towards the LOR end point; from a
  measured arrival time difference dt = t(start) - t(end): u = c dt / 2), and can be
  stretched per LOR / event with ``tof_kernel_scale``. The TOF bin width is not part of
  the table (the projectors integrate the kernel over each TOF bin). Speed: on CPUs
  ~1.6x faster than the Gaussian kernel (table lookup instead of erf), on GPUs about the
  same. See the theory section and the new gallery example.
  (`#46 <https://github.com/KUL-recon-lab/libparallelproj/issues/46>`_)

* **Much faster TOF listmode projections with sorted events:** new function
  ``tof_lm_sort_key`` (C API and Python, CPU and CUDA). It computes a 64-bit sort key per
  event: the principal axis of its LOR and the 3D Morton (Z-order) code of the grid cell
  (2 x 2 x 2 voxels) that contains its TOF center. Sorting the events by this key once
  (e.g. per subset, reused in all iterations) brings events with nearby TOF centers
  together, so that consecutive GPU threads access the same image region. Measured on an
  RTX A4500 (10 M events, SIGNA-like geometry): TOF listmode forward projection 8.5x and
  back projection 8.8x faster than with time-ordered (random) events; on a 4-core CPU the
  back projection was ~1.5x faster. The order does not change the projections. New
  documentation section "Listmode: the order of the events matters" (performance page),
  gallery example ``09_tof_listmode_event_order.py`` and
  ``scripts/benchmark_tof_listmode.py``.
  (`#54 <https://github.com/KUL-recon-lab/libparallelproj/issues/54>`_)

* **TOF listmode projectors visit fewer image planes:** the planes of an event are now
  selected with a margin of half a TOF bin (plus 0.001) around the kernel support instead
  of one TOF bin; the extra planes always had weight 0. ~12 % fewer planes for TOF bins of
  ~sigma (3 sigma), ~3 % for TOF bins of sigma / 5; results are unchanged.
  (`#54 <https://github.com/KUL-recon-lab/libparallelproj/issues/54>`_)

* **TOF window limit of 64 bins only applies with more than 64 TOF bins; Python
  warning.** The TOF window of an image plane now contains only existing TOF bins, so it
  has at most ``min(floor(support) + 2, num_tof_bins)`` bins: with at most 64 TOF bins,
  all TOF projections are exact, however wide the kernel (previously, kernels wider than
  ~62 TOF bins were always truncated). The accumulator size of the TOF sinogram
  projectors is chosen from the same bound (smaller for few TOF bins). If a call needs
  windows of more than 64 bins, the Python functions issue a ``UserWarning`` (checked if
  sigma / the kernel scale are CPU arrays). New: ``PARALLELPROJ_TOF_MAX_WINDOW_BINS``
  (C) and ``parallelproj_core.tof_max_window_bins`` (Python).
  (`#46 <https://github.com/KUL-recon-lab/libparallelproj/issues/46>`_)

* **CUDA stream support.** All projector functions take an optional CUDA stream
  (C API: ``void *stream = nullptr``; Python: ``stream=None``, an integer handle
  such as ``torch.cuda.current_stream(dev).cuda_stream`` or
  ``cupy.cuda.get_current_stream().ptr``, where ``0`` is the default stream).

  - Without a stream, a call is synchronous as before: results are complete on
    return.
  - With a stream, all copies and the kernel are queued on that stream and, for
    GPU arrays, the call returns immediately, like a PyTorch / CuPy operation on
    that stream. This allows inputs produced on side streams, several GPUs
    running concurrently from one Python thread, and recording projector calls
    in CUDA graphs.
  - Calls with CPU arrays always return complete results.
  - CUDA graph capture works with PyTorch tensors directly. CuPy arrays have to
    be passed as ``x.__dlpack__(stream=handle)`` during capture: the bindings
    import arrays via DLPack without a stream, and CuPy then orders the legacy
    default stream after its current stream, which is not allowed during capture.
  - Temporary device buffers are allocated, copied and freed stream-ordered
    (``cudaMallocAsync`` / ``cudaFreeAsync``), and the image dimensions are
    passed to the kernels by value, so asynchronous calls do not synchronize
    internally.

  (`#41 <https://github.com/KUL-recon-lab/libparallelproj/issues/41>`_)

* **New gallery examples for multiple GPUs** (PyTorch):

  - ``05_multi_gpu_streams.py``: synchronous vs. asynchronous calls, a mini
    batch split across several GPUs, an autograd layer that works with one or
    several GPUs, and how to time GPU code (non-TOF sinogram projectors),
  - ``06_multi_gpu_tof_sinogram.py``: the same with the TOF sinogram projectors,
  - ``07_multi_gpu_tof_listmode.py``: TOF listmode projectors with a different
    number of events per sample, load balancing across GPUs, and a training step
    with the Poisson listmode log-likelihood.

  Shared helpers are in ``docs/examples/gpu_utils.py``. Examples with ``gpu`` in
  their file name are not executed when the docs are built.

* **Python bindings release the GIL** while a projector runs, so that other
  Python threads (e.g. one thread per GPU) can run in the meantime.
  (`#41 <https://github.com/KUL-recon-lab/libparallelproj/issues/41>`_)

* **TOF listmode "point" projector:** new ``tof_lm_point_fwd`` /
  ``tof_lm_point_back`` (C API and Python bindings, CPU/OpenMP and CUDA).
  Every event is reduced to a single point, the center of its TOF bin (same
  bin placement and center-offset convention as the TOF listmode Joseph
  projectors). The forward projector trilinearly interpolates the image at
  that point, the back projector is its exact adjoint (trilinear splatting
  with atomic adds). Back projecting events with value 1 gives a TOF
  "histo-image". No ray tracing and no TOF kernel are involved, which makes
  this pair much faster than ``joseph3d_tof_lm_*`` -- intended for quick
  looks, motion detection on short frames, inputs for neural networks, and as
  an approximate TOF projector in iterative methods.

* **pixi environment ``docs-gpu``** (docs + CUDA 13) to build the
  documentation on a GPU system with a CUDA build of ``parallelproj_core``
  (``pixi run docs-gpu``, CMake preset ``docs-cuda13``), executing all gallery
  examples including the multi-GPU ones, and ``scripts/run_examples.py``
  (``pixi run run-examples-gpu`` / ``pixi run -e docs run-examples``) to run
  every example as a standalone script.
  (`#59 <https://github.com/KUL-recon-lab/libparallelproj/issues/59>`_)

Changed
^^^^^^^

* **Much faster TOF sinogram forward projection, especially on GPUs.** The TOF
  sums of a LOR are kept in a sliding window in registers and every TOF bin is
  written exactly once, instead of adding into all bins of the TOF window in
  global memory for every image plane. The accumulator size (8 to 64 TOF bins) is
  chosen per LOR on the device (no host synchronization, LOR-dependent TOF
  resolution supported); TOF bins narrower than ~2.5 voxels are written in aligned
  groups of 4 (16-byte stores). Measured on an RTX A6000 (5.5 M LORs, image 256x256x89): 8.3x
  faster for TOF bins of ~sigma (169 ps bins, 400 ps CTR), 21x for bins of
  sigma / 5, 8.7x for a 50 ps CTR; about 2x on CPUs.
  (`#43 <https://github.com/KUL-recon-lab/libparallelproj/issues/43>`_)

* **Faster TOF sinogram back projection.** The TOF values of a LOR are read once
  into a sliding window in registers (the mirror of the forward projection), instead
  of reading all bins of the TOF window for every image plane; buffer size and load
  width are chosen as for the forward projection. Measured on a desktop GPU (5.5 M
  LORs, image 256x256x89, planes fastest / axial image axis last): 1.26x faster for
  TOF bins of ~sigma (169 ps bins, 400 ps CTR), 4.6x for bins of sigma / 5, 5.3x for
  a 50 ps CTR; for wide TOF bins and for the 50 ps CTR, the TOF back projection then
  takes about as long as the non-TOF back projection. (With the radial bins fastest and the axial
  axis last, wide TOF bins are ~20 % slower, narrow bins 1.5-5x faster.)
  (`#44 <https://github.com/KUL-recon-lab/libparallelproj/issues/44>`_)

* **New documentation section "Performance: sinogram order and image layout"** and
  options ``--lor_order`` / ``--axial_axis`` of ``scripts/benchmark_tof_sinogram.py``:
  on GPUs, a sinogram order with the planes fastest together with the axial direction
  as the last image axis is up to ~20x faster than unfavorable combinations.
  (`#44 <https://github.com/KUL-recon-lab/libparallelproj/issues/44>`_)

* **TOF projectors: one code base for all kernels.** The four TOF workers (sinogram /
  listmode, forward / back) are templates in the TOF kernel type and share the LOR geometry
  (``src/joseph3d_lor_geometry.h``); the listmode workers were rewritten on it. TOF
  sinogram and listmode projectors with degenerate TOF parameters (e.g. sigma <= 0) now
  give zero TOF contributions (previously NaN for the sinogram projectors).
  (`#46 <https://github.com/KUL-recon-lab/libparallelproj/issues/46>`_)

* **Truncated TOF kernel in all TOF projectors.** The weight of a TOF bin is the
  integral of the Gaussian TOF kernel truncated at ``num_sigmas`` sigma over the
  bin, normalized such that the weights of all bins overlapping the kernel support
  sum to 1. Previously the full Gaussian bin integrals of all bins within
  ``floor``/``ceil`` of the support were used, so a bin entering or leaving the
  window changed the weights by a jump, and a rounding difference of the TOF
  position could flip it (differences up to ~3e-4 between CPU and GPU or between
  forward and back projection). The weights are now continuous in the TOF
  position, the sum over the TOF bins of a sinogram projection equals the non-TOF
  projection (if the TOF bins cover the support), and the TOF listmode projectors
  use the same weights as the sinogram projectors: a listmode event gets exactly
  the value of its TOF sinogram bin (previously the listmode weights of an event
  were normalized over the image planes).
  (`#43 <https://github.com/KUL-recon-lab/libparallelproj/issues/43>`_)

* **Synchronous CUDA calls no longer wait for the whole GPU.** Without a
  stream, a projector call now waits only for its own (legacy default) stream
  instead of calling ``cudaDeviceSynchronize()``. Results are still complete on
  return, but the call no longer waits for unrelated work on other
  (non-blocking) streams, e.g. the gradient all-reduce of PyTorch DDP.
  (`#41 <https://github.com/KUL-recon-lab/libparallelproj/issues/41>`_)

* **Python bindings: ``device_id`` defaults to -1 (automatic).** For GPU input
  arrays the projectors now run on the GPU of the arrays; an explicit
  ``device_id`` that does not match it raises a ``ValueError``. For CPU input
  arrays, -1 means the current CUDA device (previously always device 0). The C
  API is unchanged.
  (`#39 <https://github.com/KUL-recon-lab/libparallelproj/issues/39>`_)

Fixed
^^^^^

* **Python bindings: complete device checks.** A CPU-only build accepted GPU
  arrays (as long as all of them were on the same GPU) and dereferenced device
  pointers on the host; it now raises a ``ValueError``. With several GPUs,
  arrays on a GPU other than ``device_id`` (default 0) were processed by a
  kernel on the wrong GPU. ``tof_sigma`` and ``tof_center_offset`` were not
  part of the device checks; they can now be CPU arrays or arrays on the GPU of
  the other input arrays. All projector bindings use one shared check.
  The CUDA projectors now also restore the caller's current CUDA device after
  switching to ``device_id`` (previously the current device of the process was
  left changed, so e.g. new CuPy arrays were created on that device).
  (`#39 <https://github.com/KUL-recon-lab/libparallelproj/issues/39>`_)

* **TOF listmode projectors: the TOF window now includes the TOF bin width.**
  The listmode projectors only used image planes within
  ``num_sigmas * tof_sigma`` of the center of the event's TOF bin, ignoring that
  the effective TOF kernel (the Gaussian integrated over the bin) extends half a
  bin further. For TOF kernels that are narrow compared to the bin width, the
  listmode projectors therefore sampled the image near the bin center instead of
  integrating over the bin (e.g. 26 % difference to the TOF sinogram projector
  for sigma = 0.1 mm and 4 mm bins). The listmode projectors now use the same
  (plane, TOF bin) pairs as the TOF sinogram projectors
  (``|it_f - it| < num_sigmas * tof_sigma / tof_bin_width + 1``). Results change
  slightly (~0.1 %) also for wider kernels.
  (`#37 <https://github.com/KUL-recon-lab/libparallelproj/issues/37>`_)

* **TOF listmode projectors: no more NaN when no image plane has a non-zero TOF
  weight.** The event normalization divided by the sum of the TOF weights of
  the image planes without checking that it is non-zero. With voxels larger
  than the TOF bins and a very narrow TOF kernel, some TOF bins contain no
  voxel-center plane: the forward projection returned NaN and the back
  projection added NaN to every voxel along the LOR. Such events now have no
  contribution (forward: 0, back: skipped), as in the TOF sinogram projectors.
  (`#35 <https://github.com/KUL-recon-lab/libparallelproj/issues/35>`_)

* **TOF sinogram back projector: LORs whose TOF bins sum to zero are no longer
  skipped.** The early exit for LORs without data tested the sum over the TOF
  bins instead of whether all bins are zero. With signed input (e.g. gradients,
  difference sinograms) a LOR such as ``[..., +1, ..., -1, ...]`` was silently
  not back projected, breaking the adjointness of the TOF sinogram projector
  pair. Non-negative data was not affected. (`#33 <https://github.com/KUL-recon-lab/libparallelproj/issues/33>`_)

* **TOF projectors: the TOF center offset now always shifts towards the LOR end
  point.** As documented, a positive ``tof_center_offset`` shifts the central
  TOF bin from the LOR midpoint towards ``lor_end``. The four TOF workers
  (sinogram and listmode, forward and back) applied the offset along the
  principal axis direction instead, so for LORs with
  ``lor_end[dir] < lor_start[dir]`` (``dir`` = principal axis) the shift went
  towards ``lor_start``. **Results change for non-zero offsets on those LORs**;
  zero offsets are unaffected. If you calibrated offsets against the previous
  behavior, negate them for LORs with ``lor_end[dir] < lor_start[dir]``.
  Swapping start and end point of a LOR now requires mirroring the TOF bin
  index (``i -> N - 1 - i``) *and* negating its offset to obtain the same
  projection. New tests: ``tests/test_tof_center_offset.py``.

v2.0.7 (2026-07-08)
-------------------

Fixed
^^^^^

* **Joseph projectors: eliminate single-precision drift along the ray.** The
  forward/back workers now recompute the ray/voxel-plane intersection
  coordinates directly per plane (``i_f = i*a + b``) instead of accumulating
  them incrementally (``i_f += a``). The incremental update accumulated float32
  round-off that grew from the entry toward the exit of the image (up to
  ~1e-2 voxel on a large field of view). Projected values can therefore shift
  very slightly (~1e-3 relative) versus previous versions. Applied to all six
  workers (non-TOF and TOF, sinogram and listmode, forward and back), preserving
  the forward/back adjoint relationship.

* **TOF sinogram projectors: fixed a possible buffer overflow.** The per-plane
  TOF-weight loop could write past the fixed ``tof_weights[MAX_NUM_TOF_WEIGHTS]``
  buffer when the requested TOF window exceeded ``MAX_NUM_TOF_WEIGHTS`` bins
  (e.g. very fine TOF binning) -- undefined behavior. The window is now clamped
  to at most ``MAX_NUM_TOF_WEIGHTS`` bins centered on the ray's TOF position.

* **Forward projectors now always initialize their output.** For LORs/events
  that miss the image volume the output element is now set to 0 (non-TOF, TOF
  sinogram and TOF listmode). Previously a missed LOR left the output element
  unwritten, so it retained whatever was in the (possibly uninitialized) output
  buffer unless the caller pre-zeroed it.

* ``ray_cube_intersection_joseph`` hardening:

  - the single-plane guard uses ``floorf`` consistently (was ``(int)``
    truncation), so grazing rays entering at the image boundary face are no
    longer dropped;

  - rays parallel to an axis (``dr[k] == 0``, e.g. in-plane / direct-plane LORs)
    are handled explicitly instead of via ``1/0`` -> inf/NaN arithmetic, making
    the test well-defined also under ``-ffast-math`` / ``-ffinite-math-only``;

  - zero-length LORs (``xstart == xend``) are detected and treated as
    "no intersection" instead of producing ``0/0 = NaN``.

* **TOF listmode projectors: guard against degenerate TOF parameters.**
  Non-positive or non-finite ``tof_bin_width`` / ``sigma`` / ``num_sigmas`` are
  now rejected per event (they previously could divide by zero and yield
  out-of-range plane indices).

* The adjoint bilinear interpolation helpers no longer ``reinterpret_cast`` the
  image pointer to ``float *`` (which silently assumed ``T == float``);
  ``atomic_sum`` is now a template, so the helpers are correct for any
  floating-point element type.

v2.0.6 (2026-06-10)
-------------------

Changed
^^^^^^^

* The project version is now defined in a single ``VERSION`` file at the repo
  root (instead of being derived from ``git describe``); release tags must
  match it, enforced via the ``tag-release`` pixi task and a CI check.

Fixed
^^^^^

* Better checking of CUDA errors: failures of ``cudaMalloc``, ``cudaMemcpy``,
  kernel launches and ``cudaDeviceSynchronize`` now raise exceptions (visible
  as ``RuntimeError`` in Python), device memory is freed via RAII on all error
  paths, and stale CUDA error states can no longer be misattributed to later
  calls.

v2.0.5 (2026-03-25)
-------------------

Fixed
^^^^^

* Fix bug in TOF listmode forward and backprojector such that only image planes are taken that where the ray intersects the image volume. 
  Now the behavior of sinogram and LM TOF projectors should be very close for all LORs and the sum over TOF should be very close to non-TOF.

v2.0.4 (2026-03-13)
-------------------

Fixed
^^^^^

* remove cuda arch auto detection magic (conda-forge sets CUDAARCHS env variable)

* improve install docs

v2.0.3 (2026-03-05)
-------------------

Fixed
^^^^^

* bugfix to support all cuda 12.x versions (cudaMemLocation)

* improved auto version detection from git in build process


v2.0.2 (2026-03-04)
-------------------

Fixed
^^^^^

* better default install directories of python bindings compatible with conda-forge


v2.0.1 (2026-03-04)
-------------------

Fixed
^^^^^

* add VERSION and SOVERSION number to build of shared library libparallelproj

v2.0.0 (2026-03-03)
-------------------

Added
^^^^^

* direct support for host arrays, cuda managed arrays and cuda device arrays in the C API

* support for even number of TOF bins in TOF projectors

* nanobind Python API using python abi3 (python 3.12+ required) for better compatibility and easier installation

Breaking Changes
^^^^^^^^^^^^^^^^

* **listmode TOF** forward and backward projectors now require the **extra argument** ``num_tof_bins`` to specify the number of TOF bins, which can be even or odd.

* renaming: ``joseph3d_fwd_tof_sino`` to ``joseph3d_tof_sino_fwd``

* renaming: ``joseph3d_back_tof_sino`` to ``joseph3d_tof_sino_back``

* renaming: ``joseph3d_fwd_tof_lm`` to ``joseph3d_tof_lm_fwd``

* renaming: ``joseph3d_back_tof_lm`` to ``joseph3d_tof_lm_back``

* In the **Python API** to all **TOF projectors**, the **order of the TOF related arguments** has changed. ``num_sigmas`` is now last and has a default value of ``3.0``
