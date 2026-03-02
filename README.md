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
- [Linking against libparallelproj](#linking)

---

## Installation

We recommend to install pre-compiled versions of `libparallelproj` from conda forge.
(**to come in the near future**).

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

```
# cuda build
cmake --preset cuda
```

or

```
# non-cuda build
cmake --preset default
```

### Build the project

```
cmake --build build
```

### Run tests

```
ctest --test-dir build
```

## Notes

- Have a look into [CMakePresets.json](CMakePresets.json) to better understand the `cuda` / `default` cmake presets and options
- For CUDA builds, ensure that the CUDA Toolkit is installed and properly configured.
- For non-CUDA builds, OpenMP is required for parallelization.
- You can use [environment.yml](environment.yml) or [environment_cuda.yml](environment_cuda.yml) to create the respective build environments
- Both presets also build the python interface. This can be disabled by using `-DBUILD_PYTHON=OFF`

## Building the docs with Sphinx

```
cd docs
make html
```

## Linking against libparallelproj

When building a project with cmake and linking against `libparallelproj`, the following
lines can be used to see whether it was built with or without CUDA.

```
find_package(parallelproj CONFIG REQUIRED)

if(PARALLELPROJ_CUDA)
  message(STATUS "parallelproj was built WITH CUDA")
else()
  message(STATUS "parallelproj was built WITHOUT CUDA")
endif()
```

At runtime, call the C API helper:

```c
#include "parallelproj.h"

if (parallelproj_cuda_enabled()) {
   /* built with CUDA support */
} else {
   /* built without CUDA support */
}

const char* version = parallelproj_version();
/* e.g. "2.0.0-alpha..." */
```
