Performance: sinogram order and image layout
============================================

On GPUs, the speed of the sinogram projectors depends strongly on the order of the LORs in
memory and on the memory layout of the image. Both are chosen by the caller: the projectors
accept LORs in any order and C-ordered images with any assignment of the scanner axes to the
image axes. With an unfavorable combination, projections can be several times slower.

Why the order matters
---------------------

Every GPU thread projects one LOR, and 32 consecutive LORs (a *warp*) are processed together.
Image memory accesses of neighboring threads that hit nearby addresses are served by the same
cache lines - for the reads of the forward projection as well as for the atomic additions of
the back projection.

If the **planes (ring pairs) are the fastest varying axis of the sinogram**, consecutive LORs
share the same view and radial position: they follow the same path through the transaxial
image plane and differ only in the axial direction. If, in addition, the **axial direction is
the fastest varying (last) axis of the image** (shape ``(nx, ny, nz)``), these threads access
neighboring voxels in memory at the same time.

Measurements
------------

TOF sinogram of ~1/28 of the LORs of a SIGNA PET/MR-like scanner (5.5 M LORs, 27 TOF bins of
169 ps, CTR 400 ps), image 256 x 256 x 89 with voxels of 2.34 x 2.34 x 2.78 mm, measured on a
desktop NVIDIA GPU with ``scripts/benchmark_tof_sinogram.py`` (times in seconds, best of 3):

.. csv-table::
   :header: "LOR order (slowest to fastest)", "axial image axis", "non-TOF fwd", "non-TOF back", "TOF fwd"
   :widths: 30, 20, 15, 15, 15

   "RVP (radial, view, plane)", "last (2)", "**0.010**", "**0.083**", "**0.057**"
   "VRP (view, radial, plane)", "last (2)", "0.011", "0.098", "0.059"
   "PVR (plane, view, radial)", "first (0)", "0.021", "0.131", "0.065"
   "PVR (plane, view, radial)", "last (2)", "0.052", "0.557", "0.080"
   "RVP (radial, view, plane)", "first (0)", "0.149", "0.527", "0.134"
   "VRP (view, radial, plane)", "first (0)", "0.220", "0.717", "0.130"

The TOF back projection shows the same pattern. Between the fastest and the slowest
combination there is a factor of ~20 (non-TOF forward) and ~9 (non-TOF back projection).
On CPUs, the effect is much smaller. The same effect was reported for parallelproj v1 in
`Schramm & Thielemans (2024), Fig. 3 <https://arxiv.org/abs/2212.12519>`_.

Recommendations
---------------

* Use a sinogram order with the **planes fastest** (e.g. RVP or VRP) together with the
  **axial direction as the last image axis**. This is the default of `parallelproj`.
* With data in an order where the radial bins are fastest (e.g. PVR, used by STIR), either
  reorder the sinogram once (a transpose costs much less than a projection), or use the
  axial direction as the *first* image axis. PVR with the axial direction as the last image
  axis is the slowest of the common combinations.
* For listmode data, the event order matters in the same way (events that are close in
  memory should be close in space), but it is usually given by the acquisition.
* To check a setup, run the benchmark with ``--lor_order`` and ``--axial_axis``, e.g.

  .. code-block:: bash

     python scripts/benchmark_tof_sinogram.py --device cuda:0 --preset signa --lor_order PVR --axial_axis 0
