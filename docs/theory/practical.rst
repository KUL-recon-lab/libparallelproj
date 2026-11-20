Practical guide
===============

Recipes
-------

The snippets use the Python interface with an array API compatible library ``xp`` (NumPy,
CuPy, PyTorch) on device ``dev``; ``pc`` is ``parallelproj_core``. All arrays are ``float32``
(TOF bins: ``int16``) and C-contiguous.

**Back projectors add to the image.** All back projectors *add* their result to the image
passed to them; set it to zero first.

Sensitivity image
^^^^^^^^^^^^^^^^^

The sensitivity image :math:`A^T \mathbf{m}` (with the multiplicative corrections
:math:`\mathbf{m}` = attenuation × normalization per LOR) is needed by (OS)EM. **For TOF
reconstructions, a non-TOF back projection is enough:** the TOF weights of every sample sum to
1, so summing a TOF back projection over all TOF bins gives the non-TOF back projection (as
long as the TOF bins cover the kernel support of all samples). For listmode reconstructions,
the sensitivity image is computed from *all possible* LORs of the scanner (not from the
events).

.. code-block:: python

   # attenuation factors per LOR: exp(-forward projection of the attenuation image)
   att_proj = xp.zeros(num_lors, dtype=xp.float32, device=dev)
   pc.joseph3d_fwd(lor_start, lor_end, mu_image, image_origin, voxel_size, att_proj)
   mult = xp.exp(-att_proj) * norm  # multiplicative corrections per LOR

   sens = xp.zeros(image_shape, dtype=xp.float32, device=dev)  # back projectors add!
   pc.joseph3d_back(lor_start, lor_end, sens, image_origin, voxel_size, mult)

For large scanners, process the LORs in chunks (e.g. per ring difference): the LOR end points
of all LORs of a large scanner can need several GB.

TOF listmode OSEM with sorted events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # once: split the events into subsets and sort the events of every subset
   subsets = []
   for k in range(num_subsets):
       idx = xp.arange(k, num_events, num_subsets, device=dev)
       s, e, tb = event_start[idx, :], event_end[idx, :], tof_bin[idx]
       m, c = mult_events[idx], contamination_events[idx]  # per-event corrections
       key = xp.zeros(s.shape[0], dtype=xp.int64, device=dev)
       pc.tof_lm_sort_key(s, e, image_origin, voxel_size, key, tof_bin_width, tof_center_offset, tb, num_tof_bins)
       o = xp.argsort(key)
       subsets.append(tuple(a[o, ...] for a in (s, e, tb, m, c)))

   # iterations
   for it in range(num_iterations):
       for s, e, tb, m, c in subsets:
           expected = xp.zeros(s.shape[0], dtype=xp.float32, device=dev)
           pc.joseph3d_tof_lm_fwd(s, e, x, image_origin, voxel_size, expected, tof_bin_width, tof_sigma,
                                  tof_center_offset, tb, num_tof_bins)
           ratio = m / (m * expected + c)  # every event has count 1
           update = xp.zeros_like(x)
           pc.joseph3d_tof_lm_back(s, e, update, image_origin, voxel_size, ratio, tof_bin_width, tof_sigma,
                                   tof_center_offset, tb, num_tof_bins)
           x = x * update / (sens / num_subsets)

Testing adjointness
^^^^^^^^^^^^^^^^^^^

For your own geometry and parameters, check :math:`\langle A x, y \rangle = \langle x, A^T y \rangle`
with random :math:`x, y`:

.. code-block:: python

   x = xp.asarray(rng.uniform(size=image_shape), dtype=xp.float32, device=dev)
   y = xp.asarray(rng.uniform(size=(num_lors, num_tof_bins)), dtype=xp.float32, device=dev)
   ax = xp.zeros((num_lors, num_tof_bins), dtype=xp.float32, device=dev)
   pc.joseph3d_tof_sino_fwd(lor_start, lor_end, x, image_origin, voxel_size, ax, tof_bin_width, tof_sigma,
                            tof_center_offset, num_tof_bins)
   aty = xp.zeros(image_shape, dtype=xp.float32, device=dev)
   pc.joseph3d_tof_sino_back(lor_start, lor_end, aty, image_origin, voxel_size, y, tof_bin_width, tof_sigma,
                             tof_center_offset, num_tof_bins)
   print(float(xp.sum(ax * y)), float(xp.sum(x * aty)))  # equal up to float rounding (~1e-6 relative)

Choosing parameters
^^^^^^^^^^^^^^^^^^^

* **TOF bin width:** about σ is usually enough to sample the TOF kernel; finer TOF bins make
  TOF sinogram projections slower (more TOF bins per sample) and listmode projections barely.
* ``num_sigmas``: 3 (default) is sufficient for most applications.
* **Data order and image layout:** see :doc:`/performance` (sinograms: planes fastest and the
  axial direction as the last image axis; listmode: sort the events).
* **Memory:** keep the data on the GPU between calls; CPU arrays are copied to the GPU in every
  call. Estimates: listmode ~30-40 bytes per event (LOR end points 24, TOF bin 2, corrections,
  one buffer); TOF sinograms 4 bytes per TOF bin and array.

Pitfalls and FAQ
----------------

**My back projection keeps growing.** Back projectors add to the image; zero it before every
back projection.

**Start and end points of my LORs are swapped.** Non-TOF projections are not affected. For
TOF, TOF bin :math:`t` of a LOR becomes TOF bin :math:`N - 1 - t` of the swapped LOR, and the
TOF center offset changes its sign (and a tabulated kernel has to be mirrored).

**The sum over the TOF bins is smaller than the non-TOF projection.** The TOF bins do not
cover the TOF kernels of all samples (e.g. the image extends beyond the TOF range of the LOR);
the part of the kernel outside the TOF bins is lost, as in a real scanner.

**I get a UserWarning about 64 TOF bins.** The TOF kernel is wider than ~62 TOF bins and
there are more than 64 TOF bins, see :ref:`tof-window-limit`.

**The back projection differs slightly between runs.** The order of the atomic additions is
not fixed (~10⁻⁷ relative differences); see :doc:`gpu_implementation`.

**LORs that miss the image, or with start = end,** give 0 (and add nothing in the back
projection).

**Arrays are rejected (TypeError).** The Python interface expects ``float32`` arrays
(``int16`` for TOF bins, ``int64`` for sort keys) that are C-contiguous; it does not convert
arrays.

Glossary
--------

.. glossary::

   LOR
      Line of response: the line between the two detectors of a coincidence, given by its end
      points ``lor_start`` and ``lor_end``.

   TOF bin
      Interval of time differences (converted to positions along the LOR) into which TOF
      data are binned; width ``tof_bin_width`` (spatial units).

   CTR
      Coincidence time resolution, usually given as FWHM in ps;
      :math:`\sigma_\text{TOF} = c/2 \cdot \text{CTR} / 2.355`.

   TOF window
      The TOF bins that get a non-zero weight from one ray sample (those overlapping the
      truncated TOF kernel).

   TOF center (of an event)
      The center of the event's TOF bin on its LOR.

   principal axis
      The image axis along which a LOR changes the most; Joseph's method samples the LOR in
      the image planes perpendicular to it.

   warp
      32 GPU threads that run in lock step.

   atomic addition
      An addition to memory that cannot be interrupted by other threads; needed when many
      threads add to the same voxels.

Notation
--------

.. csv-table::
   :header: "symbol", "meaning", "API name"
   :widths: 25, 45, 30

   ":math:`\mathbf{d}_0, \mathbf{d}_1`", "LOR start and end point", "``lor_start``, ``lor_end``"
   ":math:`f`, :math:`x`", "image", "``image``"
   ":math:`c_f`", "length of the LOR between two image planes", "(internal)"
   ":math:`\Delta`", "TOF bin width", "``tof_bin_width``"
   ":math:`N`", "number of TOF bins", "``num_tof_bins``"
   ":math:`o`", "TOF center offset", "``tof_center_offset``"
   ":math:`\sigma_{\mathrm{TOF}}`", "TOF standard deviation (spatial units)", "``tof_sigma``"
   ":math:`n_\sigma`", "truncation of the Gaussian in units of :math:`\sigma_{\mathrm{TOF}}`", "``num_sigmas``"
   ":math:`F(u)`", "cumulative TOF kernel (table)", "``tof_kernel_cdf``"
   ":math:`u_\text{min}, u_\text{max}`", "support of the tabulated kernel", "``tof_kernel_u_min``, ``tof_kernel_u_max``"
   ":math:`s`", "scale of the tabulated kernel", "``tof_kernel_scale``"
