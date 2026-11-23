Examples
========

These examples show how to call the projector functions of ``libparallelproj`` from
Python (:mod:`parallelproj_core`): what they compute, what their arguments mean, and how
to use them efficiently on GPUs. The same arguments are used by the C API.

The examples use a simple 2D geometry (one image plane and LORs in that plane), which keeps
the figures easy to read; the projectors themselves are always 3D. They run on the CPU
(NumPy) or, with a CUDA build and CuPy, on the GPU. The background is explained in
:doc:`/theory/index`.

Complete PET reconstructions (scanner geometries, sinograms, resolution and attenuation
models, MLEM / OSEM and other algorithms, listmode reconstruction, PyTorch layers) are shown
in the examples of `parallelproj <https://parallelproj.readthedocs.io>`_, which builds on
``libparallelproj``.
