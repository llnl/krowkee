cmake_minimum_required(VERSION 3.14)

# ---------------------------------------------------------------------------- #
# CMake policy
# ---------------------------------------------------------------------------- #
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.30")
    cmake_policy(SET CMP0167 NEW)
endif ()

# ---------------------------------------------------------------------------- #
# krowkee configuration
# ---------------------------------------------------------------------------- #
project(
    krowkee
    VERSION 0.2
    DESCRIPTION "HPC multi-stream sketching library"
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#
# Configuration options for krowkee.
#
option(KROWKEE_BUILD_TESTS "Build tests" OFF)
option(KROWKEE_BUILD_EXAMPLES "Build examples" OFF)
option(KROWKEE_USE_YGM "Compile using YGM features (only for testing)" OFF)
option(KROWKEE_DOXYGEN "Adds targets for generating Doxygen documentation" OFF)
option(KROWKEE_RTD_ONLY
       "Run Cmake for only generating documentation for Read the Docs" OFF
)

#
# FetchContent dependencies that can be overridden on the command line.
#
set(YGM_FETCH_GIT
    "https://github.com/LLNL/ygm.git"
    CACHE STRING "GIT to fetch ygm")
set(YGM_FETCH_TAG
    "v0.10-dev"
    CACHE STRING "Tag to fetch ygm")
set(EIGEN3_FETCH_GIT
    "https://github.com/PX4/eigen.git"
    CACHE STRING "GIT to fetch Eigen3")
set(EIGEN3_FETCH_TAG
    "3.4"
    CACHE STRING "Tag to fetch Eigen3")

#
# Doxygen and Read the Docs (RTD) Documentation
#
if (KROWKEE_DOXYGEN OR KROWKEE_RTD_ONLY)
    add_subdirectory(docs)
    if (KROWKEE_RTD_ONLY)
        return()
    endif ()
endif ()

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if (EXISTS "${LOC_PATH}")
    message(
        FATAL_ERROR "Cannot build in a directory with a CMakeLists.txt file. "
                    "Please make a build subdirectory."
    )
endif ()

# Prepare to invoke FetchContent
include(FetchContent)


add_library(krowkee INTERFACE)
add_library(krowkee::krowkee ALIAS krowkee)
target_include_directories(
    krowkee INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include>
)
target_compile_features(krowkee INTERFACE cxx_std_20)

#
# Optionally link YGM (mostly for testing)
#
if (KROWKEE_USE_YGM)
    if (NOT ygm_POPULATED)
        find_package(ygm CONFIG QUIET)
        if (NOT ygm_FOUND)
            set(JUST_INSTALL_YGM TRUE)
            FetchContent_Declare(
                ygm
                GIT_REPOSITORY ${YGM_FETCH_GIT}
                GIT_TAG ${YGM_FETCH_TAG}
            )
            FetchContent_MakeAvailable(ygm)
        endif ()
    endif ()
    target_link_libraries(krowkee INTERFACE ygm::ygm)
endif ()

#
# Find-else-fetch Eigen for linear algebra
#
if (NOT Eigen3_POPULATED)
    find_package(Eigen3 CONFIG QUIET)
    if (NOT Eigen3_FOUND)
        # Turn off build tests for Eigen3
        option(BUILD_TESTING OFF)
        FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY ${EIGEN3_FETCH_GIT}
            GIT_TAG ${EIGEN3_FETCH_TAG}
        )
        FetchContent_MakeAvailable(Eigen3)
    endif()
    target_link_libraries(krowkee INTERFACE Eigen3::Eigen)
endif ()

if (KROWKEE_BUILD_TESTS)
    option(TEST_WITH_SLURM "Run tests with Slurm" OFF)
    enable_testing()
    add_subdirectory(test)
endif ()

if (KROWKEE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (KROWKEE_BUILD_PERFORMANCE)
    add_subdirectory(performance)
endif ()
