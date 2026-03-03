.. libparallelproj documentation master file

libparallelproj docs
====================

libparallelproj is a high-performance library for 3D forward and backward projection,
supporting both CUDA and non-CUDA builds with a minimal Python interface.

The library implements the Joseph 3D ray-driven projection algorithm :cite:`Joseph1982` with support for:

- **Non-TOF projections**: Standard forward and back projection
- **TOF sinogram projections**: Time-Of-Flight sinogram-based projections
- **TOF listmode projections**: Time-Of-Flight listmode (event-by-event) projections

**github repository**: `<https://github.com/KUL-recon-lab/libparallelproj>`_

.. important::
  **Key Features:**

  - **C API**: Direct support for host arrays, CUDA managed arrays, and device arrays, enabling seamless integration with different memory models
  - **Python API**: Direct support for Python Array API compliant frameworks (e.g., NumPy, CuPy, PyTorch in CPU or GPU mode), providing flexibility and easy adoption

.. important::
  *If you are using libparallelproj, we highly recommend reading and appreciate citing our publication* :cite:`Schramm2023`

  * G. Schramm, K. Thielemans: "**PARALLELPROJ - An open-source framework for fast calculation of projections in tomography**", Front. Nucl. Med., Volume 3 - 2023, doi: 10.3389/fnume.2023.1324562, `link to paper <https://www.frontiersin.org/articles/10.3389/fnume.2023.1324562/abstract>`_, `link to arxiv version <https://arxiv.org/abs/2212.12519>`_

.. note::
  `libparallelproj` and `parallelproj-core`` are **minimal APIs for the core projection operations**, and do not include higher-level reconstruction algorithms or utilities. However, they can be easily integrated into existing reconstruction frameworks.

Content
=======

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Theory and Background <theory_background>
   Installation <installation>
   C API Reference <c_api>
   Python API Reference <python_api>
   Examples <auto_examples/index>
   Changelog <changelog>

References
----------

.. bibliography::
