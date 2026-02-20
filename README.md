# libparallelproj

libparallelproj is a high-performance library for 3D forward and backward projection,
supporting both CUDA and non-CUDA builds and a minimal python interface.

Official documentation: [link to readthedocs](https://libparallelproj.readthedocs.io)

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

### CUDA-Specific Requirements
- **CUDA Toolkit** (if building with CUDA support)

All build and test requirements can be installed from `conda-forge` using
`environment.yaml` or `environment_cuda.yaml`.
---

## Building the Project

To build the project with CUDA support:

1. Create a build directory:
   ```bash
   mkdir -p build && cd build
   ```

2. Configure the project with CMake:
   ```bash
   cmake ..
   ```

   The following CMake option can be used to cofigure the build:
   `-DUSE_CUDA`, `-DBUILD_PYTHON`, `-DBUILD_TESTS`, `-DBUILD_DOCS`


3. Build the project:
   ```bash
   cmake --build .
   ```

4. In case the tests where build, they can be executed via:
   ```bash
   ctest --output-on-failure
   ```

---

## Notes

- for CUDA builds, ensure that the CUDA Toolkit is installed and properly configured.
- for non-CUDA builds, OpenMP is required for parallelization.
- many important tests are written in python and requires the python interface


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
```
