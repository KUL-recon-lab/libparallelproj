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
- **python** (optional, version >= 3.12, for the python API)
- **nanobind** (optional, for the python API)

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
- You can use [environment.yaml](environment.yaml) or [environment_cuda.yaml](environment_cuda.yaml) to create the respective build environments
- Both presets also build the python interface. This can be disabled by using `-DBUILD_PYTHON=OFF`

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
