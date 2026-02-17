Python API Reference
====================

This page documents the Python interface for parallelproj-backend, generated from nanobind bindings.

Module: ``parallelproj_backend``
---------------------------------

.. automodule:: parallelproj_backend
   :members:
   :undoc-members:
   :show-inheritance:

Module Attributes
~~~~~~~~~~~~~~~~~

.. autodata:: parallelproj_backend.__version__
   :annotation: = version string

.. autodata:: parallelproj_backend.PARALLELPROJ_CUDA
   :annotation: = 0 or 1 (whether CUDA support is enabled)

Functions
~~~~~~~~~

Non-TOF Projections
^^^^^^^^^^^^^^^^^^^

.. autofunction:: parallelproj_backend.joseph3d_fwd

.. autofunction:: parallelproj_backend.joseph3d_back

TOF Sinogram Projections
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: parallelproj_backend.joseph3d_tof_sino_fwd

.. autofunction:: parallelproj_backend.joseph3d_tof_sino_back

TOF Listmode Projections
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: parallelproj_backend.joseph3d_tof_lm_fwd
