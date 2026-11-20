Python API Reference
====================

Module: ``parallelproj_core``
---------------------------------

.. automodule:: parallelproj_core
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

Non-TOF Projections
^^^^^^^^^^^^^^^^^^^

.. autofunction:: parallelproj_core.joseph3d_fwd

.. autofunction:: parallelproj_core.joseph3d_back

TOF Sinogram Projections
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: parallelproj_core.joseph3d_tof_sino_fwd

.. autofunction:: parallelproj_core.joseph3d_tof_sino_back

TOF Listmode Projections
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: parallelproj_core.joseph3d_tof_lm_fwd

.. autofunction:: parallelproj_core.joseph3d_tof_lm_back

TOF Projections with a Tabulated TOF Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: parallelproj_core.joseph3d_tof_sino_fwd_tab

.. autofunction:: parallelproj_core.joseph3d_tof_sino_back_tab

.. autofunction:: parallelproj_core.joseph3d_tof_lm_fwd_tab

.. autofunction:: parallelproj_core.joseph3d_tof_lm_back_tab

TOF Listmode Point Projections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: parallelproj_core.tof_lm_point_fwd

.. autofunction:: parallelproj_core.tof_lm_point_back

Sorting of TOF Listmode Events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sorting the events by this key (once, e.g. per subset) makes the TOF listmode projectors
much faster on GPUs, see :ref:`listmode-event-order`.

.. autofunction:: parallelproj_core.tof_lm_sort_key
