cmake_minimum_required(VERSION 3.18)

# Policy used in newer CMake; keep compatible with older CMake by guarding.
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()

# ------------------------------------------------------------------------------
# Project
# ------------------------------------------------------------------------------
project(PARALLELPROJ LANGUAGES CXX)

include(CheckLanguage)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# Options (conda-forge friendly “split” builds)
# ------------------------------------------------------------------------------
option(USE_CUDA       "Build with CUDA support" OFF)
option(BUILD_PYTHON   "Build the Python interface using nanobind" OFF)
option(BUILD_TESTS    "Build the tests" OFF)

# Split packaging support:
# - BUILD_LIBRARY : build the C++ shared library target (parallelproj)
# - INSTALL_DEVEL : install headers + CMake package config files (devel bits)
option(BUILD_LIBRARY  "Build the C++ shared library (parallelproj)" ON)
option(INSTALL_DEVEL  "Install headers + CMake package config files" ON)

# Optional debug prints
option(DEBUG "Enable debug printing" OFF)
if(DEBUG)
  add_compile_definitions(DEBUG)
endif()

# ------------------------------------------------------------------------------
# Versioning (conda-forge override friendly)
# ------------------------------------------------------------------------------
# Use project version override if provided by packager; otherwise best effort.
if(DEFINED ENV{READTHEDOCS_VERSION} AND NOT "$ENV{READTHEDOCS_VERSION}" STREQUAL "")
  set(PROJECT_VERSION "$ENV{READTHEDOCS_VERSION}")
  message(STATUS "Using version from READTHEDOCS_VERSION environment variable: ${PROJECT_VERSION}")
elseif(DEFINED PROJECT_VERSION_OVERRIDE AND NOT PROJECT_VERSION_OVERRIDE STREQUAL "")
  set(PROJECT_VERSION "${PROJECT_VERSION_OVERRIDE}")
  message(STATUS "Using version from PROJECT_VERSION_OVERRIDE CMake variable: ${PROJECT_VERSION}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND git describe --tags --long --dirty
    OUTPUT_VARIABLE GIT_DESCRIBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  execute_process(
    COMMAND git describe --tags --exact-match
    OUTPUT_VARIABLE GIT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(GIT_TAG AND NOT GIT_DESCRIBE MATCHES "-dirty$")
    set(PROJECT_VERSION "${GIT_TAG}")
  elseif(GIT_DESCRIBE)
    set(PROJECT_VERSION "${GIT_DESCRIBE}")
  else()
    set(PROJECT_VERSION "0.0.0-unknown")
  endif()
else()
  get_filename_component(_FOLDER_NAME "${CMAKE_SOURCE_DIR}" NAME)
  string(REGEX MATCH "v?([0-9]+\\.[0-9]+\\.[0-9]+([-.][A-Za-z0-9]+)?)" _m "${_FOLDER_NAME}")
  if(_m)
    set(PROJECT_VERSION "${CMAKE_MATCH_1}")
  else()
    set(PROJECT_VERSION "0.0.0-unknown")
  endif()
endif()

# Extract numeric major/minor/patch safely
string(REGEX MATCHALL "[0-9]+" _VERSION_PARTS "${PROJECT_VERSION}")
list(LENGTH _VERSION_PARTS _n)
if(_n GREATER_EQUAL 3)
  list(GET _VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
  list(GET _VERSION_PARTS 1 PROJECT_VERSION_MINOR)
  list(GET _VERSION_PARTS 2 PROJECT_VERSION_PATCH)
else()
  set(PROJECT_VERSION_MAJOR 0)
  set(PROJECT_VERSION_MINOR 0)
  set(PROJECT_VERSION_PATCH 0)
endif()

message(STATUS "Building version ${PROJECT_VERSION}")
message(STATUS "Version components: major=${PROJECT_VERSION_MAJOR}, minor=${PROJECT_VERSION_MINOR}, patch=${PROJECT_VERSION_PATCH}")

# ------------------------------------------------------------------------------
# CUDA toolchain (only when needed)
# ------------------------------------------------------------------------------
set(PP_NEED_CUDA_TOOLCHAIN OFF)
if(USE_CUDA AND (BUILD_LIBRARY OR BUILD_TESTS))
  set(PP_NEED_CUDA_TOOLCHAIN ON)
endif()

if(PP_NEED_CUDA_TOOLCHAIN)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "USE_CUDA=ON but no CUDA compiler was found by CMake. "
      "Install a CUDA toolkit / nvcc, or configure with -DUSE_CUDA=OFF."
    )
  endif()

  # Curated list (filtered against what this nvcc supports)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(_CANDIDATE_ARCHS 70 75 80 86 89 90 120 121)

    execute_process(
      COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-arch
      OUTPUT_VARIABLE _NVCC_ARCHS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )

    string(REGEX MATCHALL "compute_[0-9]+" _TOKENS "${_NVCC_ARCHS}")
    message(STATUS "nvcc supports the following architectures: ${_TOKENS}")

    set(_SUPPORTED_SMS "")
    foreach(t IN LISTS _TOKENS)
      string(REPLACE "compute_" "" sm "${t}")
      list(APPEND _SUPPORTED_SMS "${sm}")
    endforeach()
    list(REMOVE_DUPLICATES _SUPPORTED_SMS)

    set(_FINAL_ARCHS "")
    foreach(sm IN LISTS _CANDIDATE_ARCHS)
      list(FIND _SUPPORTED_SMS "${sm}" _idx)
      if(NOT _idx EQUAL -1)
        list(APPEND _FINAL_ARCHS "${sm}")
      endif()
    endforeach()

    if(_FINAL_ARCHS STREQUAL "")
      message(WARNING "No overlap between curated arch list and nvcc-supported list; falling back to nvcc list.")
      set(_FINAL_ARCHS "${_SUPPORTED_SMS}")
    endif()

    if(_FINAL_ARCHS STREQUAL "")
      message(WARNING "Could not determine architectures from nvcc; using default fallback list.")
      set(_FINAL_ARCHS "75;80;86;89;90")
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "${_FINAL_ARCHS}" CACHE STRING "CUDA SM architectures" FORCE)
  endif()

  message(STATUS "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ------------------------------------------------------------------------------
# Toolchain / platform tweaks
# ------------------------------------------------------------------------------
# Auto-export symbols helps when you don't explicitly annotate exports on Windows.
# Only affects SHARED libs.
if(WIN32 AND BUILD_LIBRARY)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

# Prefer position-independent code for libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# LTO (keep optional; conda-forge toolchains may not always like forced IPO)
# ------------------------------------------------------------------------------
include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg)
if(_ipo_ok AND CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  # You can omit JOBS; keep it conservative if you want.
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_JOBS 4)
else()
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "IPO/LTO not enabled: ${_ipo_msg}")
  endif()
endif()

# ------------------------------------------------------------------------------
# C++ library target: parallelproj
# ------------------------------------------------------------------------------
set(PARALLELPROJ_CUDA 0)

if(BUILD_LIBRARY AND USE_CUDA)
  message(STATUS "CUDA library build enabled.")
  set(PARALLELPROJ_CUDA 1)

  set(LIB_SOURCES
    src/joseph3d_fwd.cu
    src/joseph3d_back.cu
    src/joseph3d_tof_sino_fwd.cu
    src/joseph3d_tof_sino_back.cu
    src/joseph3d_tof_lm_fwd.cu
    src/joseph3d_tof_lm_back.cu
    src/cuda_utils.cu
    src/parallelproj_info.cpp
  )

  add_library(parallelproj SHARED ${LIB_SOURCES})
  target_link_libraries(parallelproj PRIVATE CUDA::cudart)
  target_compile_definitions(parallelproj
    PUBLIC  PARALLELPROJ_CUDA=1
    PRIVATE PARALLELPROJ_EXPORTS
            PROJECT_VERSION="${PROJECT_VERSION}"
            PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  )

elseif(BUILD_LIBRARY)
  message(STATUS "CUDA library build disabled.")
  set(PARALLELPROJ_CUDA 0)

  set(LIB_SOURCES
    src/joseph3d_fwd.cpp
    src/joseph3d_back.cpp
    src/joseph3d_tof_sino_fwd.cpp
    src/joseph3d_tof_sino_back.cpp
    src/joseph3d_tof_lm_fwd.cpp
    src/joseph3d_tof_lm_back.cpp
    src/parallelproj_info.cpp
  )

  find_package(OpenMP REQUIRED)

  add_library(parallelproj SHARED ${LIB_SOURCES})
  target_link_libraries(parallelproj PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(parallelproj
    PUBLIC  PARALLELPROJ_CUDA=0
    PRIVATE PARALLELPROJ_EXPORTS
            PROJECT_VERSION="${PROJECT_VERSION}"
            PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  )
endif()

if(TARGET parallelproj)
  target_include_directories(parallelproj
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  # Versioned shared library + SONAME (Linux/macOS). Windows ignores VERSION/SOVERSION.
  set_target_properties(parallelproj PROPERTIES
    VERSION   "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    OUTPUT_NAME "parallelproj"
  )
endif()

# ------------------------------------------------------------------------------
# Python extension: parallelproj_core (nanobind)
# ------------------------------------------------------------------------------
if(BUILD_PYTHON)
  # Require interpreter + module development; request stable-ABI support when available.
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
               OPTIONAL_COMPONENTS Development.SABIModule)

  # nanobind
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
  )
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(parallelproj_core
    STABLE_ABI
    python/bindings.cpp
  )

  # Link to in-tree library if it exists, otherwise to installed package config.
  if(TARGET parallelproj)
    target_link_libraries(parallelproj_core PRIVATE parallelproj)
    add_dependencies(parallelproj_core parallelproj)
  else()
    find_package(parallelproj CONFIG REQUIRED)
    target_link_libraries(parallelproj_core PRIVATE parallelproj::parallelproj)
  endif()

  target_compile_definitions(parallelproj_core
    PRIVATE PARALLELPROJ_CUDA=${PARALLELPROJ_CUDA}
            PROJECT_VERSION="${PROJECT_VERSION}"
  )

  message(STATUS "Installing Python extension to ${Python_SITEARCH}")

  install(TARGETS parallelproj_core
  LIBRARY DESTINATION "${Python_SITEARCH}"   # Unix/macOS
  RUNTIME DESTINATION "${Python_SITEARCH}"   # Windows (.pyd)
  ARCHIVE DESTINATION "${Python_SITEARCH}"   # harmless
  )
endif()

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
if(BUILD_TESTS)
  enable_testing()

  set(TEST_SOURCES
    tests/test_ray_cube_intersection.cpp
    tests/test_bilinear_interp.cpp
    tests/test_bilinear_interp_adj.cpp
    tests/test_nontof_fwd_back.cpp
  )

  if(USE_CUDA)
    foreach(TEST_SOURCE IN LISTS TEST_SOURCES)
      get_filename_component(TEST_NAME "${TEST_SOURCE}" NAME_WE)
      set(CUDA_TEST_SOURCE "${CMAKE_SOURCE_DIR}/tests/${TEST_NAME}_cuda.cu")
      if(EXISTS "${CUDA_TEST_SOURCE}")
        list(APPEND TEST_SOURCES "${CUDA_TEST_SOURCE}")
      else()
        message(WARNING "CUDA test source not found: ${CUDA_TEST_SOURCE}")
      endif()
    endforeach()
  endif()

  set(TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/tests")
  set(TEST_DATA_FILES
    img.txt
    vstart.txt
    vend.txt
    expected_fwd_vals.txt
  )

  set(TEST_DATA_OUTPUTS "")
  foreach(TEST_FILE IN LISTS TEST_DATA_FILES)
    add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/${TEST_FILE}"
      COMMAND "${CMAKE_COMMAND}" -E copy
              "${TEST_DATA_DIR}/${TEST_FILE}"
              "${CMAKE_BINARY_DIR}/${TEST_FILE}"
      DEPENDS "${TEST_DATA_DIR}/${TEST_FILE}"
    )
    list(APPEND TEST_DATA_OUTPUTS "${CMAKE_BINARY_DIR}/${TEST_FILE}")
  endforeach()

  add_custom_target(copy_test_data ALL DEPENDS ${TEST_DATA_OUTPUTS})

  foreach(TEST_SOURCE IN LISTS TEST_SOURCES)
    get_filename_component(TEST_NAME "${TEST_SOURCE}" NAME_WE)
    add_executable("${TEST_NAME}" "${TEST_SOURCE}")
    target_include_directories("${TEST_NAME}" PRIVATE
      "${CMAKE_SOURCE_DIR}/include"
      "${CMAKE_SOURCE_DIR}/tests"
    )
    add_dependencies("${TEST_NAME}" copy_test_data)

    if(USE_CUDA)
      target_link_libraries("${TEST_NAME}" PRIVATE parallelproj CUDA::cudart)
    else()
      target_link_libraries("${TEST_NAME}" PRIVATE parallelproj OpenMP::OpenMP_CXX)
    endif()

    add_test(NAME "${TEST_NAME}" COMMAND "${TEST_NAME}")
  endforeach()
endif()

# ------------------------------------------------------------------------------
# Install: runtime library
# ------------------------------------------------------------------------------
if(TARGET parallelproj)
  install(TARGETS parallelproj
    EXPORT parallelprojTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
endif()

# ------------------------------------------------------------------------------
# Install: devel (headers + CMake package config)
# ------------------------------------------------------------------------------
if(INSTALL_DEVEL AND TARGET parallelproj)
  # Headers
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

  # Export targets
  install(EXPORT parallelprojTargets
    FILE parallelprojTargets.cmake
    NAMESPACE parallelproj::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj"
  )

  # Config + version files
  write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/parallelprojConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
  )

  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/parallelprojConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/parallelprojConfig.cmake"
    @ONLY
  )

  install(FILES
    "${CMAKE_BINARY_DIR}/parallelprojConfig.cmake"
    "${CMAKE_BINARY_DIR}/parallelprojConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj"
  )
endif()

# ------------------------------------------------------------------------------
# Uninstall target
# ------------------------------------------------------------------------------
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
  )
  add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_BINARY_DIR}/cmake_uninstall.cmake"
  )
endif()

# ------------------------------------------------------------------------------
# Doxygen
# ------------------------------------------------------------------------------
option(BUILD_DOCS "Build Doxygen documentation" OFF)
if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)

  set(DOXYGEN_INPUT_FILE  "${CMAKE_SOURCE_DIR}/Doxyfile.in")
  set(DOXYGEN_OUTPUT_FILE "${CMAKE_BINARY_DIR}/Doxyfile")
  set(DOXYGEN_OUTPUT_DIR  "${CMAKE_BINARY_DIR}/docs")

  configure_file("${DOXYGEN_INPUT_FILE}" "${DOXYGEN_OUTPUT_FILE}" @ONLY)

  add_custom_target(doxygen-docs
    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYGEN_OUTPUT_FILE}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Generating Doxygen documentation..."
    VERBATIM
  )

  add_custom_target(doxygen-clean
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "${DOXYGEN_OUTPUT_DIR}"
    COMMENT "Cleaning Doxygen documentation..."
    VERBATIM
  )
endif()
