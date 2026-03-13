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

libparallelproj is a high-performance library for 3D forward and backward projection,
supporting both CUDA and non-CUDA builds and a minimal python interface.

Official documentation: [https://libparallelproj.readthedocs.io](https://libparallelproj.readthedocs.io)

## Table of Contents
- [Installation](#installation)
- [Build Requirements](#requirements)
- [Building and Testing](#building-the-project)
- [Python Interface](#python-interface)
- [Using libparallelproj from another CMake project](#using-libparallelproj-from-another-cmake-project)

---

## Installation

We recommend to install pre-compiled versions of `libparallelproj` from conda-forge
where cuda 12/13 and non-cuda builds are available (see official docs).

## Build requirements

### General Requirements
- **CMake** (version 3.18 or higher)
- **C++17** compatible compiler
- **OpenMP** (for non-CUDA builds)

### Optional Requirements for python API
- **python** (version >= 3.12)
- **nanobind**

### CUDA-Specific Requirements
- **CUDA Toolkit** (if building with CUDA support)

---

## Building the Project

### Configure the project for cuda / vs non-cuda build

```bash
# enter the root directory of this repository
cd libparallelproj
```

```bash
# cuda build
cmake --preset cuda
```

or

```bash
# non-cuda build
cmake --preset non-cuda
```

### Build the project

```bash
cmake --build build
```

### Run tests

```bash
ctest --test-dir build
```

## Notes

- Have a look into [CMakePresets.json](CMakePresets.json) to better understand the `cuda` / `non-cuda` cmake presets and options
- For CUDA builds, ensure that the CUDA Toolkit is installed and properly configured.
- For non-CUDA builds, OpenMP is required for parallelization.
- You can use [environment.yml](environment.yml) or [environment_cuda.yml](environment_cuda.yml) to create the respective build environments
- Both presets also build the python interface. This can be disabled by using `-DBUILD_PYTHON=OFF`
- The `cuda` preset uses `"CMAKE_CUDA_ARCHITECTURES": "native"` for local builds. You might want to change that for local builds that are supposed to run on several architectures (e.g. `all` or `all-major`) - see [here](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html).

## Building the docs with Sphinx

```
cd docs
make html
```

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
cmake_minimum_required(VERSION 3.18)
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
find_package(parallelproj 2.0 CONFIG REQUIRED)
```

The exported package version is numeric, for example `2.0.2`, so standard CMake version matching works as expected.

### Available CMake variables

After calling

```cmake
find_package(parallelproj CONFIG REQUIRED)
```

the following variables are available:

- `PARALLELPROJ_CUDA`
  `1` if `libparallelproj` was built with CUDA support, otherwise `0`

- `PARALLELPROJ_VERSION`
  Numeric package version, for example `2.0.2`

- `PARALLELPROJ_INCLUDE_DIRS`
  Install include directory

- `PARALLELPROJ_LIBRARY_DIRS`
  Install library directory

- `PARALLELPROJ_VERSION_STRING`
  full (more descriptive) package version containing labels for "dirty" versions, for example `2.0.2-dirty-0-12ab3`

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

At runtime, the linked library can be queried via the C API:

```c
#include "parallelproj.h"

if (parallelproj_cuda_enabled()) {
    /* built with CUDA support */
} else {
    /* built without CUDA support */
}

// full version string, with potential "dirty" suffix
const char* version = parallelproj_version();

// assuming you are building a clean version, the numeric version can be checked via
int pp_major_version = parallelproj_version_major();
int pp_minor_version = parallelproj_version_minor();
int pp_patch_version = parallelproj_version_patch();
```

**Note**:
- `parallelproj_version()` returns the full library version string potentially including "dirty" suffixes.
