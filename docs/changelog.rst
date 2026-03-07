Changelog
=========

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

* nanobind python API using python abi3 (python 3.12+ required) for better compatibility and easier installation

Breaking Changes
^^^^^^^^^^^^^^^^

* TOF forward and backward projectors now require the extra argument ``num_tof_bins`` to specify the number of TOF bins, which can be even or odd.
