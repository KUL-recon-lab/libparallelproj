.. _theory:

Theory and implementation
=========================

These pages explain what the projectors of ``libparallelproj`` compute and how they do it,
for users of the library who know PET image reconstruction but not necessarily the inner
workings of projectors. They start with the role of a projector in a reconstruction, explain
Joseph's ray tracing method, what changes with time-of-flight (TOF), how TOF sinogram and TOF
listmode projections differ, and end with the implementation techniques that make the
projectors fast on GPUs and a practical guide.

.. toctree::
   :maxdepth: 2

   projectors
   joseph
   tof
   sino_vs_lm
   gpu_implementation
   practical

**Where to start?**

* New to projectors: read the pages in order.
* Choosing between TOF sinogram and TOF listmode reconstruction: :doc:`sino_vs_lm`.
* Speed: :doc:`gpu_implementation` and the :doc:`/performance` page (data order and image
  layout).
* Recipes (sensitivity image, (OS)EM, adjointness test), pitfalls and a glossary:
  :doc:`practical`.
