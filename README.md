<p align="center">
  <img src="docs/_static/logo.png" alt="libparallelproj logo" width="220">
</p>

# libparallelproj

<p align="center">
  <a href="https://github.com/KUL-recon-lab/libparallelproj/actions/workflows/build_and_test.yml">
    <img src="https://github.com/KUL-recon-lab/libparallelproj/actions/workflows/build_and_test.yml/badge.svg" alt="Build and Test">
  </a>
  <a href="https://github.com/KUL-recon-lab/libparallelproj/actions/workflows/build_cuda.yml">
    <img src="https://github.com/KUL-recon-lab/libparallelproj/actions/workflows/build_cuda.yml/badge.svg" alt="Build CUDA">
  </a>
  <a href="https://github.com/KUL-recon-lab/libparallelproj/actions/workflows/build_docs.yml">
    <img src="https://github.com/KUL-recon-lab/libparallelproj/actions/workflows/build_docs.yml/badge.svg" alt="Build Documentation">
  </a>
  <br>
  <a href="https://libparallelproj.readthedocs.io">
    <img src="https://readthedocs.org/projects/libparallelproj/badge/?version=latest" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/KUL-recon-lab/libparallelproj">
    <img src="https://codecov.io/gh/KUL-recon-lab/libparallelproj/branch/main/graph/badge.svg" alt="Code Coverage">
  </a>
  <a href="https://github.com/KUL-recon-lab/libparallelproj/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/KUL-recon-lab/libparallelproj/tags">
    <img src="https://img.shields.io/github/v/tag/KUL-recon-lab/libparallelproj" alt="Latest Tag">
  </a>
  <a href="https://anaconda.org/conda-forge/libparallelproj">
    <img src="https://img.shields.io/conda/vn/conda-forge/libparallelproj.svg" alt="conda-forge version">
  </a>
</p>

**libparallelproj** is a high-performance library for **forward and back projections
(line integrals) in 3D tomography**, in particular TOF PET, with a C/C++ API, a minimal Python
API (`parallelproj_core`) and CPU (OpenMP) and CUDA builds.

- **Projectors:** Joseph's method for **non-TOF**, **TOF sinogram** and **TOF listmode**
  projections (Gaussian or tabulated TOF kernels), and a fast TOF listmode **point**
  projector. Forward and back projections are exact adjoints.
- **Python:** works directly on NumPy, CuPy and PyTorch arrays (CPU or GPU, no copies), via
  the Python array API.
- **GPUs:** CUDA streams (asynchronous calls), several GPUs in parallel, and fast TOF listmode
  projections of sorted events.

Documentation (theory, examples, API): **[libparallelproj.readthedocs.io](https://libparallelproj.readthedocs.io)**.
For complete reconstructions (scanner geometries, sinograms, OSEM, PyTorch layers), see
[parallelproj](https://parallelproj.readthedocs.io), which builds on `parallelproj_core`.

> **Upgrading from 2.x:** version 3 adds a trailing `stream` parameter to all projector
> functions of the C API (the library is now `libparallelproj.so.3`), so binaries built
> against 2.x have to be rebuilt; C++ and Python code works unchanged. Some numerical
> results change slightly. See the [changelog](https://libparallelproj.readthedocs.io/en/latest/changelog.html).

## Table of Contents
- [Installation](#installation)
- [Python interface](#python-interface)
- [Building from source](#building-from-source)
- [Building the docs](#building-the-docs)
- [Using libparallelproj from another CMake project](#using-libparallelproj-from-another-cmake-project)
- [Scripts](#scripts)
- [Versioning and release procedure](#versioning-and-release-procedure)
- [Citation](#citation)

---

## Installation

Pre-compiled packages (CPU and CUDA 12 / 13 builds) are available on conda-forge:

```bash
conda install -c conda-forge parallelproj-core   # Python API (installs libparallelproj)
conda install -c conda-forge libparallelproj     # C/C++ library only
```

On systems with an NVIDIA GPU, a CUDA build is selected automatically. See the
[installation guide](https://libparallelproj.readthedocs.io/en/latest/installation.html) for
choosing the CUDA version or a CPU build explicitly.

## Python interface

All arrays are `float32` (TOF bin indices `int16`) and are passed as they are: the projectors
write into existing output arrays, and GPU arrays (CuPy, PyTorch) are projected on their GPU.

```python
import numpy as np
import parallelproj_core as pp

# image: 64 x 64 x 64 voxels of 2 mm, value 1; image_origin = center of voxel [0, 0, 0]
image = np.ones((64, 64, 64), dtype=np.float32)
voxel_size = np.array([2.0, 2.0, 2.0], dtype=np.float32)
image_origin = (-31.5 * voxel_size).astype(np.float32)

# two LORs (start and end points in mm)
lor_start = np.array([[-100.0, 0.0, 0.0], [0.0, -100.0, 10.0]], dtype=np.float32)
lor_end = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 10.0]], dtype=np.float32)

proj = np.zeros(2, dtype=np.float32)
pp.joseph3d_fwd(lor_start, lor_end, image, image_origin, voxel_size, proj)  # forward projection
print(proj)  # line integrals: [128. 128.] (128 mm inside the image)

back = np.zeros_like(image)  # the back projection adds to this array
pp.joseph3d_back(lor_start, lor_end, back, image_origin, voxel_size, np.ones(2, dtype=np.float32))
```

TOF sinogram / listmode projections, the conventions of the arguments, PyTorch layers and
multiple GPUs are shown in the
[example gallery](https://libparallelproj.readthedocs.io/en/latest/auto_examples/index.html).

## Building from source

### With pixi (recommended)

[pixi](https://pixi.sh) (≥ 0.84.2) installs all dependencies (compilers, CMake, Ninja,
nanobind, CUDA toolkit, ...) into isolated environments:

```bash
pixi run -e default configure   # CPU / OpenMP build: cmake --preset non-cuda
pixi run -e default build       # cmake --build build
pixi run -e default test        # ctest + pytest
```

```bash
pixi run -e cuda13 configure    # CUDA 13 build (cmake --preset cuda13; same for cuda12)
pixi run -e cuda13 build
pixi run -e cuda13 test
```

The CUDA environments are only installed on machines that report a CUDA 12 / 13 driver (or
with `CONDA_OVERRIDE_CUDA=12` / `13`, as in CI). Without a GPU, the build works, but the tests
are skipped.

| Task | Environments | Description |
|------|--------------|-------------|
| `configure`, `build` | `default`, `cuda12`, `cuda13` | CMake configure (the environment's preset) and build |
| `test-cpp`, `test-python`, `test` | `default`, `cuda12`, `cuda13` | C++ tests (ctest), Python tests (pytest), both |
| `install` | `default`, `cuda12`, `cuda13` | `cmake --install` |
| `uninstall` | `default` | remove the installed files |
| `configure-coverage`, `test-python-cov`, `collect-coverage-cpp` | `default` | coverage reports (CI, Linux) |
| `build-docs`, `run-examples`, `clean-gallery` | `docs` | see [Building the docs](#building-the-docs) |
| `docs-gpu`, `run-examples-gpu` | `docs-gpu` | docs / examples with a CUDA build |

### With CMake directly

Requirements: CMake ≥ 3.31, Ninja, a C++17 compiler, OpenMP (CPU builds), CUDA toolkit 12 or
13 (CUDA builds), Python ≥ 3.12 and nanobind (Python extension); for the tests pytest, numpy,
array-api-compat, array-api-strict and pytorch.

```bash
cmake --preset non-cuda && cmake --build build && ctest --output-on-failure --test-dir build
```

```bash
cmake --preset cuda13 && cmake --build build-cuda13   # or cuda12 / build-cuda12
```

See [CMakePresets.json](CMakePresets.json) for all presets. Notes:

- The CUDA presets use `CMAKE_CUDA_ARCHITECTURES=native` (the GPU of the build machine). For
  other or several GPU generations, use e.g. `all-major` (see the
  [CMake docs](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)).
- `-DBUILD_PYTHON=OFF` disables the Python extension.

## Building the docs

```bash
pixi run -e docs build-docs
```

This runs CMake configure, Doxygen, the library build and `sphinx-build`; the output is in
`docs/_build/html`. This is a CPU build (as on Read the Docs): the examples with `gpu` in their
file name are not executed.

On a system with an NVIDIA GPU (CUDA 13 driver; one GPU is enough):

```bash
pixi run docs-gpu          # docs with a CUDA build, executing ALL examples (docs/_build/html-gpu)
pixi run run-examples-gpu  # every example as a standalone script: pass / fail + run time
```

`docs-gpu` first removes the generated gallery pages (`clean-gallery`), so that all examples
run again. Without pixi: install the documentation dependencies and run `cd docs && make html`.

## Using libparallelproj from another CMake project

After installing `libparallelproj`, downstream CMake projects can locate it via:

```cmake
find_package(parallelproj CONFIG REQUIRED)
```

This provides the imported CMake target:

```cmake
parallelproj::parallelproj
```

which should be linked to your executable or library.

### Minimal example

```cmake
cmake_minimum_required(VERSION 3.18)  # of your project
project(my_project LANGUAGES C CXX)

find_package(parallelproj CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE parallelproj::parallelproj)
```

### If CMake cannot find the package

If `libparallelproj` was installed into a non-standard location, point CMake to the installation prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libparallelproj/install
```

CMake will then look for the installed package configuration files in the corresponding install tree.

### Package version selection

You can request a minimum version in `find_package`, for example:

```cmake
find_package(parallelproj 3.0 CONFIG REQUIRED)
```

The exported package version is numeric (e.g. `3.0.0`). Any newer version is accepted
(`AnyNewerVersion`): e.g. `find_package(parallelproj 2.0 ...)` also finds 3.0, because C++
code written for 2.x compiles unchanged with 3.x.

### Available CMake variables

After calling

```cmake
find_package(parallelproj CONFIG REQUIRED)
```

the following variables are available:

- `PARALLELPROJ_CUDA`
  `1` if `libparallelproj` was built with CUDA support, otherwise `0`

- `PARALLELPROJ_VERSION`
  Numeric package version, for example `3.0.0`

- `PARALLELPROJ_INCLUDE_DIRS`
  Install include directory

- `PARALLELPROJ_LIBRARY_DIRS`
  Install library directory

- `PARALLELPROJ_VERSION_STRING`
  full version string (the content of the `VERSION` file), for example `3.0.0` or `3.0.1.dev0`

In most cases, it is best to link against the imported target `parallelproj::parallelproj` rather than manually using include and library directory variables.

### Checking whether the installed library was built with CUDA

At CMake configure time:

```cmake
find_package(parallelproj CONFIG REQUIRED)

if(PARALLELPROJ_CUDA)
  message(STATUS "parallelproj was built WITH CUDA support")
else()
  message(STATUS "parallelproj was built WITHOUT CUDA support")
endif()
```

At runtime, the linked library can be queried via the C API (`parallelproj.h` is a C++
header with C linkage, `extern "C"`):

```cpp
#include "parallelproj.h"

if (parallelproj_cuda_enabled()) {
    /* built with CUDA support */
} else {
    /* built without CUDA support */
}

// full version string (the content of the VERSION file, e.g. "3.0.0" or "3.0.1.dev0")
const char* version = parallelproj_version();

// the numeric version components can be checked via
int pp_major_version = parallelproj_version_major();
int pp_minor_version = parallelproj_version_minor();
int pp_patch_version = parallelproj_version_patch();
```

**Note**:
- `parallelproj_version()` returns the full version string, which for dev builds
  includes a `.devN` suffix (e.g. `3.0.1.dev0`).

## Scripts

`scripts/` contains benchmarks (TOF sinogram / listmode projectors, scaling with the number of
events, mini batches, comparison of two library versions), a profiling driver and maintenance
scripts; see [scripts/README.md](scripts/README.md).

## Versioning and release procedure

The **`VERSION` file at the repo root is the single source of truth** for the
project version (PEP 440 style: `X.Y.Z` for releases, `X.Y.Z.devN` between
releases). CMake, the Python extension, the docs, and the conda recipes all
derive their version from it.

To make a release:

1. open a release PR that bumps `VERSION` to the release version (e.g.
   `3.0.0`) and adds a `v3.0.0` entry to `docs/changelog.rst`, then merge it
2. on the up-to-date `main` branch run `pixi run tag-release` — this creates
   the annotated tag `v3.0.0` after validating that the working tree is
   clean, `VERSION` is a non-dev version, and the changelog entry exists
3. publish the tag with `git push origin v3.0.0` — the
   `check_version` workflow re-verifies that the tag matches `VERSION`
   (mismatched tags fail CI before any release artifacts exist)
4. open a follow-up PR bumping `VERSION` to the next dev version
   (e.g. `3.0.1.dev0`)

For a new **major** version whose C++ API is not source compatible with the previous one,
also switch the CMake package version compatibility from `AnyNewerVersion` to
`SameMajorVersion` (`write_basic_package_version_file` in `CMakeLists.txt`), so that
`find_package(parallelproj <old major>)` does not accept it.

The conda-forge feedstock derives its version from the release tag, which the
guards above guarantee to be identical to the `VERSION` file.

## Citation

If you use libparallelproj, please cite

G. Schramm, K. Thielemans: "PARALLELPROJ - An open-source framework for fast calculation of
projections in tomography", Front. Nucl. Med., Volume 3 - 2023,
[doi: 10.3389/fnume.2023.1324562](https://doi.org/10.3389/fnume.2023.1324562)
([arXiv](https://arxiv.org/abs/2212.12519)).
