// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

// Klugy, but includes need to be in this order.

#include <krowkee/sketch.hpp>

#include <type_traits>

// Krowkee provides the relatively simple sketch interfaces available in the
// header `krowkee/sketch.hpp`. These include the `krowkee::sketch::SparseJLT`
// used in `examples/sparse_jlt.cpp` and `krowkee::sketch::FWHT`. These types
// consolidate the component parts (the transform, the container, and the
// sketch chassis) and reduce them to more easily addressable aliases that
// reflect the most common sketch decisions. Most applications need only use
// these simple APIs.
using register_type                           = float;
constexpr const std::size_t range_size        = 8;
constexpr const std::size_t replication_count = 3;
using simple_sketch_type =
    krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                               std::shared_ptr>;

// However, these simpler type interfaces are provided merely for the
// convenience of the user. It is possible to directly template
// `krowkee::sketch::Sketch` to specify more granular behavior. This class
// requires a concrete sketch transform type, a template container type, a
// template merge operator type, and a template pointer type.

// Transform types are found in `krowkee/transform and are in the
// `krowkee::transform` namespace. For example, the sparse
// Johnson-Lindenstrauss transform functor implemented with tiled CountSketch
// is used within the `krowkee::sketch::SparseJLT` convenience type. In
// addition to register type and sketch size parameters, the sparse jlt
// requires a template type defining the behavior of the CountSketch hash
// functor. For example, `krowkee::hash::CountSketchHash` is the default
// implementation of this hash.

// `krowkee::hash::CountSketchHash` can internally use many different hash
// functions. The current default is `krowkee::hash::MulAddShift`, a
// 2-universal hash that satisfies the requirements for several theorems
// involving CountSketch while remaining efficient. We use it here, although
// in principle one could replace it with a different hash type.
template <std::size_t RangeSize>
using count_sketch_type =
    krowkee::hash::CountSketchHash<RangeSize, krowkee::hash::MulAddShift>;

// This `count_sketch_type` then allows us to define a concrete transform type.
// It is possible to replace this template argument with a template type that
// providing a different implementation of the CountSketch hash behavior.
using transform_type =
    krowkee::transform::SparseJLT<register_type, count_sketch_type, range_size,
                                  replication_count>;

// We then require a container type. The simplest container type is
// `krowkee::sketch::Dense`, which stores the sketch as a vector of length
// `range_size * replication_count`. In addition to the register type, it
// requires a merge operator that governs how two sketches of the same type are
// merged.
using dense_type = krowkee::sketch::Dense<register_type, std::plus,
                                          range_size * replication_count>;

// Finally, the `krowkee::sketch::Sketch` class requires a shared pointer type
// that determines how the transform pointer sharing is implemented. In shared
// memory this should always be `std::shared_ptr`.

// We can alternately define our sketch type by putting all of this together.
using verbose_sketch_type = krowkee::sketch::Sketch<
    krowkee::transform::SparseJLT<register_type, krowkee::hash::CountSketchHash,
                                  range_size, replication_count>,
    krowkee::sketch::Dense<register_type, std::plus,
                           range_size * replication_count>,
    std::shared_ptr>;

// If this exe compiles, then the two types are identical.
static_assert(std::is_same<simple_sketch_type, verbose_sketch_type>::value);

int main(int argc, char **argv) { return 0; }