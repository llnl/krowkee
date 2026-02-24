# KROWKEE: A Distributed Multi-Stream Data Sketching Toolkit.

This repository implements `krowkee`, a toolkit for scalably and efficiently 
summarizing many data streams in distributed memory.
`krowkee` is intended for applications involving where one needs to summarize 
huge loosely structured data, such as matrices or graphs, where individual 
components such as rows/columns or vertex adjacency information are impractical 
to store and directly inspect.
`krowkee` ingests these objects as data streams - unstructured, arbitrarily 
ordered lists of updates - and accumulates summaries thereof in the form of data 
sketches.

Although there are many types of data sketches, the practical varieties encompassed by
krowkee have many advantages compared to directly observing data:
* approximation guarantees on some stream statistic
* merge operator
* worst-case logarithmic memory usage
* ... and many even support sparse storage

`krowkee` currently supports only 
[sparse subspace embeddings](https://arxiv.org/abs/1207.6365)
to perform randomized and fast dimensionality reduction. 
Future sketch support is planned, principally including cardinality sketches 
such as the
[HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog).

# Getting started

## Reqirements
* C++17 - GCC versions 10-11 are tested. 
Your mileage may vary with other compilers.
* [Eigen](https://github.com/PX4/eigen) v3.4 or greater. If package `Eigen3` is
  not installed, `krowkee` will attempt to clone and install via
  [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html).
* Optional dependencies:
    - [YGM](https://github.com/LLNL/ygm) 0.10 or greater for distributed memory
      communication.
      Toggle with CMake option `KROWKEE_USE_YGM`. 
      Default `OFF`.
      YGM is not used within the library, but is used in some tests and examples
      that are only compiled if `KROWKEE_USE_YGM` is `ON`.
      `KROWKEE_USE_YGM` is NOT required if your downstream project uses both
      `krowkee` and `ygm`.
      If toggled on and package `ygm` is not installed, `krowkee` will attempt
      to clone and install via 
      [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html). Includes additional dependencies:
        * [Cereal](https://github.com/USCiLab/cereal) - C++ serialization 
          library. 
          If package `cereal` is not installed, `krowkee` will attempt
          [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html).
        * MPI         
    - [Boost](https://www.boost.org/) 1.75 or greater for 
      `boost::container::flatmap`.
      Toggle with CMake option `KROWKEE_USE_BOOST`.
      Default `ON`.

## Using `krowkee` with CMake

`krowkee` is a header-only library that is simple to incorporate into dependent
projects using CMake.
Add the following to your `CMakeLists.txt` to "find-else-fetch" `krowkee`, 
cloning it and its dependencies and preparing their headers for installation as
a part of your project.
```
set(DESIRED_KROWKEE_VERSION 0.3)
find_package(krowkee ${DESIRED_KROWKEE_VERSION} CONFIG)
if (NOT krowkee_FOUND)
    FetchContent_Declare(
        krowkee
        GIT_REPOSITORY https://github.com/LLNL/krowkee
        GIT_TAG v${DESIRED_KROWKEE_VERSION}
    )
    FetchContent_MakeAvailable(krowkee)
endif ()
```


## Building
These instructions assume that you have a relatively modern C++ compiler 
(C++20 required, only tested using GCC).
If included, `krowkee`'s CMake build makes use of find-else-fetch semantics for 
`Eigen` and its optional `ygm` and `cereal` dependencies.
`krowkee` will try to find local installations of the libraries, and will clone
and link the repositories internally if none are found.

`spack` is a convenient means to include `cereal` and manage compilers, but 
is not required to build `krowkee`. 

### Build steps
Clone the project and make the build directory
``` bash
$ git clone ssh://git@czgitlab.llnl.gov:7999/krowkee/krowkee.git
$ mkdir krowkee/build
$ cd krowkee/build
```

Option 1: use spack
``` bash
$ spack load gcc     # tested at >=8.3.1.
$ spack load boost   # optional, at least version 1.75
$ spack load cereal  # optional, at least version 1.3.0
```

Option 2: use module
``` bash
$ module load gcc/8.3.1  # or desired gcc version
```

Build `krowkee`. `krowkee` is a library, so it has no default compile targets.
Toggle on different compile targets with the following arguments
`KROWKEE_BUILD_TESTS`, `KROWKEE_BUILD_EXAMPLES`, and
`KROWKEE_BUILD_PERFORMANCE`. So, to build the tests and examples (including the optional YGM exes) but NOT the performance benchmarks, use
``` bash
$ cmake .. \
  -DKROWKEE_USE_YGM=ON \
  -DKROWKEE_BUILD_TESTS=ON \
  -DKROWKEE_BUILD_EXAMPLES=ON \
  -DKROWKEE_BUILD_PERFORMANCE=OFF \
  -DCMAKE_BUILD_TYPE=Release
$ make
```

## Testing

It is easy to run all test cases once krowkee is built as above by running
``` bash
make test
```

Alternately, one can directly run individual test cases with more options and 
verbose outputs, e.g.
``` bash
$ ./test/linearsketch_test
```
or
``` bash
$ mpirun -n 4 ./test/ygm/power_iteration_test
```

All tests support an `-h` flag listing options.

## Building docs

`krowkee`'s documentation is produced using
[doxygen](https://www.doxygen.nl/index.html) and
[sphinx](https://www.sphinx-doc.org/en/master/index.html).
Documentation is hosted on
[readthedocs](https://krowkee.readthedocs.io/en/latest/index.html).

To locally build the documentation, use the cmake argument
`-DKROKEE_DOXYGEN=ON` during the build and then compile all targets.
Instal the sphinx dependencies in a python environment via
```bash
$ pip install -r docs/rtd/requirements.txt
```
The documentation can then be locally constructed from the build directory via
```
$ make sphinx
```
Finally, open the file `build/docs/rtd/sphinx/index.html` in your browser of
choice.

# About

## Authors

* Min W. Priest (priest2 at llnl dot gov)
* Alec Dunton (dunton1 at llnl dot gov)

# License

`krowkee` is distributed under the MIT license.

All new contributions must be made under the MIT license.

See [LICENSE-MIT](LICENSE-MIT), [NOTICE](NOTICE), and [COPYRIGHT](COPYRIGHT) 
for details.

SPDX-License-Identifier: MIT

# Release

LLNL-CODE-827987