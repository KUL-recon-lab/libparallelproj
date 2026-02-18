.. parallelproj-backend documentation master file

parallelproj-backend docs
=========================

parallelproj-backend is a high-performance library for 3D forward and backward projection,
supporting both CUDA and non-CUDA builds with a minimal Python interface.

The library implements the Joseph 3D ray-driven projection algorithm with support for:

- **Non-TOF projections**: Standard forward and back projection
- **TOF sinogram projections**: Time-Of-Flight sinogram-based projections
- **TOF listmode projections**: Time-Of-Flight listmode (event-by-event) projections

API Documentation
=================

.. toctree::
   :maxdepth: 2

   python_api
   c_api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
