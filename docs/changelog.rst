Changelog
=========

v2.1.0 (unreleased)
-------------------

Added
^^^^^

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

Fixed
^^^^^

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
