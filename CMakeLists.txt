cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0177)
    cmake_policy(SET CMP0177 NEW)
endif()

project(PARALLELPROJ LANGUAGES CXX)

include(CheckLanguage)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    option(USE_CUDA "Enable CUDA build" ON)
else()
    option(USE_CUDA "Enable CUDA build" OFF)
endif()

if(USE_CUDA)
  # If user/toolchain did not provide architectures, choose a default based on nvcc version.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    execute_process(
      COMMAND "${CMAKE_CUDA_COMPILER}" --version
      OUTPUT_VARIABLE _nvcc_version_out
      ERROR_VARIABLE  _nvcc_version_err
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
    )

    # Extract the major version from nvcc output (works with typical "release 12.9" / "release 13.0" strings)
    string(REGEX MATCH "release[ \t]+([0-9]+)\\." _ "${_nvcc_version_out}")
    set(_nvcc_major "${CMAKE_MATCH_1}")

    if(_nvcc_major STREQUAL "")
      message(WARNING "Could not parse nvcc version; defaulting to Turing+ arch set.")
      set(_nvcc_major 13)
    endif()

    if(_nvcc_major GREATER_EQUAL 13)
      # CUDA 13+: cannot compile sm_70 offline anymore -> start at 75
      message(STATUS "Detected nvcc >= 13, targeting Turning-Blackwell")
      set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;120" CACHE STRING "CUDA arch list")
    else()
      message(STATUS "Detected nvcc < 13, targeting Volta+")
      set(CMAKE_CUDA_ARCHITECTURES "60;61;62;70;75;80;86;89;90" CACHE STRING "CUDA arch list")
    endif()
    message(STATUS "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# Add an option to build the Python interface
option(BUILD_PYTHON "Build the Python interface using nanobind" OFF)

# Add an option to build the tests
option(BUILD_TESTS "Build the tests" OFF)

# Split packaging support:
# - BUILD_LIBRARY: build the C++ shared library target (parallelproj)
# - INSTALL_DEVEL: install headers + CMake package config files (devel bits)
option(BUILD_LIBRARY "Build the C++ shared library (parallelproj)" ON)
option(INSTALL_DEVEL "Install headers + CMake package config files" ON)

# Enable automatic symbol export on Windows for shared libraries
if(WIN32 AND BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

################################################################################
### DERIVE VERSION FROM GIT ####################################################
################################################################################

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    # Get the full output of git describe
    execute_process(
        COMMAND git describe --tags --long --dirty
        OUTPUT_VARIABLE GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Check if we are on a clean, tagged version
    execute_process(
        COMMAND git describe --tags --exact-match
        OUTPUT_VARIABLE GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(GIT_TAG AND NOT GIT_DESCRIBE MATCHES "-dirty$")
        # Use only the tag if we are on a clean, tagged version
        set(PROJECT_VERSION "${GIT_TAG}")
    elseif(GIT_DESCRIBE)
        # Use the full git describe output otherwise
        set(PROJECT_VERSION "${GIT_DESCRIBE}")
    else()
        # Fallback version if git describe fails
        set(PROJECT_VERSION "0.0.0-unknown")
    endif()
else()
    # Fallback for non-Git environments
    get_filename_component(FOLDER_NAME ${CMAKE_SOURCE_DIR} NAME)
    string(REGEX MATCH "([0-9]+\\.[0-9]+\\.[0-9]+(-[a-zA-Z0-9]+)?)" PROJECT_VERSION "${FOLDER_NAME}")
    if(NOT PROJECT_VERSION)
        set(PROJECT_VERSION "0.0.0-unknown")
    endif()
endif()

# Extract major, minor, and patch versions
string(REGEX MATCHALL "[0-9]+" VERSION_PARTS "${PROJECT_VERSION}")
list(GET VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
list(GET VERSION_PARTS 1 PROJECT_VERSION_MINOR)
list(GET VERSION_PARTS 2 PROJECT_VERSION_PATCH)

message(STATUS "Building version ${PROJECT_VERSION}")

option(DEBUG "Enable debug printing" OFF)
if(DEBUG)
    add_compile_definitions(DEBUG)
endif()

################################################################################
### DEFINE TARGETS #############################################################
################################################################################

# Set the number of parallel jobs for Link Time Optimization (LTO)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_JOBS 4) # Adjust the number of jobs as needed
endif()

if(BUILD_LIBRARY AND USE_CUDA)
    message(STATUS "CUDA build enabled.")

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

    set(PARALLELPROJ_CUDA 1)

    add_library(parallelproj SHARED ${LIB_SOURCES})
    target_link_libraries(parallelproj PRIVATE CUDA::cudart)
    target_compile_definitions(parallelproj PUBLIC PARALLELPROJ_CUDA=1)
    target_compile_definitions(parallelproj PRIVATE PARALLELPROJ_EXPORTS PROJECT_VERSION="${PROJECT_VERSION}" PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR} PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR} PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH})
elseif(BUILD_LIBRARY)
    message(STATUS "CUDA build disabled.")

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
    set(PARALLELPROJ_CUDA 0)

    add_library(parallelproj SHARED ${LIB_SOURCES})
    target_link_libraries(parallelproj PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(parallelproj PUBLIC PARALLELPROJ_CUDA=0)
    target_compile_definitions(parallelproj PRIVATE PARALLELPROJ_EXPORTS PROJECT_VERSION="${PROJECT_VERSION}" PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR} PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR} PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH})

endif()

if(TARGET parallelproj)
    target_include_directories(parallelproj
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
endif()

################################################################################
### BUILD THE PYTHON INTERFACE #################################################
################################################################################

# Build the Python interface if enabled
if(BUILD_PYTHON)
    if (CMAKE_VERSION VERSION_LESS 3.18)
      set(DEV_MODULE Development)
    else()
      set(DEV_MODULE Development.Module)
    endif()

    find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

    # Add an option for the Python interface install destination
    set(PYTHON_INTERFACE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Directory to install the Python interface")
    message(STATUS "Python interface will be installed to: ${PYTHON_INTERFACE_INSTALL_DIR}")

    # Detect the installed nanobind package and import it into CMake
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(
      parallelproj_core
      python/bindings.cpp
    )

    # Prefer linking against the in-tree target if we built it,
    # otherwise link against an installed libparallelproj via its CMake package.
    if(TARGET parallelproj)
        target_link_libraries(parallelproj_core PRIVATE parallelproj)
        add_dependencies(parallelproj_core parallelproj)
    else()
        find_package(parallelproj CONFIG REQUIRED)
        target_link_libraries(parallelproj_core PRIVATE parallelproj::parallelproj)
    endif()

    # Pass the PARALLELPROJ_CUDA definition to the Python module
    target_compile_definitions(parallelproj_core PRIVATE PARALLELPROJ_CUDA=${PARALLELPROJ_CUDA} PROJECT_VERSION="${PROJECT_VERSION}")

    # Install parallelproj_core to the specified destination
    install(TARGETS parallelproj_core
        EXPORT parallelprojTargets
        LIBRARY DESTINATION ${PYTHON_INTERFACE_INSTALL_DIR}
        RUNTIME DESTINATION ${PYTHON_INTERFACE_INSTALL_DIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

################################################################################
### BUILD TESTS ################################################################
################################################################################

if(BUILD_TESTS)
    enable_testing()

    set(TEST_SOURCES
        tests/test_ray_cube_intersection.cpp
        tests/test_bilinear_interp.cpp
        tests/test_bilinear_interp_adj.cpp
        tests/test_nontof_fwd_back.cpp
    )

    if(USE_CUDA)
        # for each test in TEST_SOURCES append the cuda one by replacing .cpp with _cuda.cu
        foreach(TEST_SOURCE ${TEST_SOURCES})
            get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
            set(CUDA_TEST_SOURCE ${CMAKE_SOURCE_DIR}/tests/${TEST_NAME}_cuda.cu)
            if(EXISTS ${CUDA_TEST_SOURCE})
                list(APPEND TEST_SOURCES ${CUDA_TEST_SOURCE})
            else()
                message(WARNING "CUDA test source not found: ${CUDA_TEST_SOURCE}")
            endif()
        endforeach()
    endif()

    set(TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/tests)

    # List all the .txt files that need to be copied
    set(TEST_DATA_FILES
        img.txt
        vstart.txt
        vend.txt
        expected_fwd_vals.txt
    )

    # Copy each file to the build directory
    foreach(TEST_FILE ${TEST_DATA_FILES})
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/${TEST_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_DATA_DIR}/${TEST_FILE} ${CMAKE_BINARY_DIR}/${TEST_FILE}
            DEPENDS ${TEST_DATA_DIR}/${TEST_FILE}
        )
        list(APPEND TEST_DATA_OUTPUTS ${CMAKE_BINARY_DIR}/${TEST_FILE})
    endforeach()

    # Add a custom target to ensure the files are copied before running tests
    add_custom_target(copy_test_data ALL DEPENDS ${TEST_DATA_OUTPUTS})

    # Add the directory containing file_utils.h to the include paths
    include_directories(${CMAKE_SOURCE_DIR}/tests)

    foreach(TEST_SOURCE ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests)
        add_dependencies(${TEST_NAME} copy_test_data) # Ensure test data is copied before running tests
        if(USE_CUDA)
            target_link_libraries(${TEST_NAME} PRIVATE parallelproj CUDA::cudart)
        else()
            target_link_libraries(${TEST_NAME} PRIVATE parallelproj OpenMP::OpenMP_CXX)
        endif()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()

    if(BUILD_PYTHON)
        ### ADD THE PYTHON INTERFACE TESTS ###
        find_program(PYTEST_EXECUTABLE pytest REQUIRED)

        add_test(NAME python_interface_test COMMAND ${Python_EXECUTABLE} -m  pytest -v ${CMAKE_SOURCE_DIR}/tests)

        set_tests_properties(python_interface_test PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:parallelproj_core>$<SEMICOLON>$ENV{PYTHONPATH}")
    endif()
endif()

################################################################################
### INSTALL AND CONFIGS ########################################################
################################################################################

if(TARGET parallelproj)
    # Install the library runtime
    install(TARGETS parallelproj
        EXPORT parallelprojTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

if(INSTALL_DEVEL AND TARGET parallelproj)
    # Headers
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Export the build tree for other projects (build-tree usage)
    export(EXPORT parallelprojTargets
        FILE ${CMAKE_BINARY_DIR}/parallelprojTargets.cmake
        NAMESPACE parallelproj::
    )

    # Install the export configuration
    install(EXPORT parallelprojTargets
        FILE parallelprojTargets.cmake
        NAMESPACE parallelproj::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj
    )

    # Create and install a package configuration file
    write_basic_package_version_file(
        ${CMAKE_BINARY_DIR}/parallelprojConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    configure_file(${CMAKE_SOURCE_DIR}/cmake/parallelprojConfig.cmake.in
        ${CMAKE_BINARY_DIR}/parallelprojConfig.cmake
        @ONLY
    )

    install(FILES
        ${CMAKE_BINARY_DIR}/parallelprojConfig.cmake
        ${CMAKE_BINARY_DIR}/parallelprojConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj
    )
endif()

################################################################################
### UNINSTALL SCRIPT ###########################################################
################################################################################

# Create an uninstall script
if(NOT TARGET uninstall)
    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
        ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()

################################################################################
### DOXYGEN DOCUMENTATION ######################################################
################################################################################

# Add an option to enable or disable building the documentation
option(BUILD_DOCS "Build Doxygen documentation" OFF)

if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    # Set the Doxygen configuration file
    set(DOXYGEN_INPUT_FILE ${CMAKE_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUTPUT_FILE ${CMAKE_BINARY_DIR}/Doxyfile)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs)

    # Configure the Doxyfile with project-specific variables
    configure_file(${DOXYGEN_INPUT_FILE} ${DOXYGEN_OUTPUT_FILE} @ONLY)

    # Add a custom target to generate the documentation
    add_custom_target(doxygen-docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUTPUT_FILE}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating Doxygen documentation..."
        VERBATIM
    )

    # Add a clean target to remove the generated documentation
    add_custom_target(doxygen-clean
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOXYGEN_OUTPUT_DIR}
        COMMENT "Cleaning Doxygen documentation..."
        VERBATIM
    )
endif()
