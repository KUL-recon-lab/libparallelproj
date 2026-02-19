.. libparallelproj documentation master file

libparallelproj docs
====================

libparallelproj is a high-performance library for 3D forward and backward projection,
supporting both CUDA and non-CUDA builds with a minimal Python interface.

The library implements the Joseph 3D ray-driven projection algorithm with support for:

- **Non-TOF projections**: Standard forward and back projection
- **TOF sinogram projections**: Time-Of-Flight sinogram-based projections
- **TOF listmode projections**: Time-Of-Flight listmode (event-by-event) projections

**github repository**: `<https://github.com/KUL-recon-lab/libparallelproj>`_

.. hint::
  *If you are using parallelproj, we highly recommend to read and cite our publication* :cite:`Schramm2023`

  * G. Schramm, K. Thielemans: "**PARALLELPROJ - An open-source framework for fast calculation of projections in tomography**", Front. Nucl. Med., Volume 3 - 2023, doi: 10.3389/fnume.2023.1324562, `link to paper <https://www.frontiersin.org/articles/10.3389/fnume.2023.1324562/abstract>`_, `link to arxiv version <https://arxiv.org/abs/2212.12519>`_

API Documentation
=================

.. toctree::
   :maxdepth: 2

   c_api
   python_api

Indices and tables
==================

* :ref:`genindex`

References
==========

.. bibliography::
