// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <krowkee/sketch.hpp>

#include <iostream>
#include <random>

int main(int argc, char **argv) {
  uint64_t      stream_size(20000);
  std::uint64_t domain_size(16384);
  std::uint64_t seed(krowkee::hash::default_seed);

  // ---------------------------------------------------------------------------
  // data preparation
  // ---------------------------------------------------------------------------

  // In this example we demonstrate how to merge krowkee sketches, and verify
  // that it is equivalent to concatenating streams. In the case of linear
  // sketches like SparseJLT and FWHT, it is further equivalent to adding
  // together their implicit frequency vectors. First we define our sketch type
  // and create a shared transform as in the other examples.
  constexpr const std::size_t range_size        = 8;
  constexpr const std::size_t replication_count = 3;
  using register_type                           = float;
  using sketch_type =
      krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                                 std::shared_ptr>;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  transform_ptr_type transform_ptr(std::make_shared<transform_type>(seed));

  // Now we create three sketches using the same transform. We will sample two
  // streams, A and B. These correspond to updates to two frequency vectors of
  // the same size. We will read stream A into one, stream B into the second,
  // and the concatenation of A and B into the third, which corresponds to
  // adding the frequency vectors of A and B together.
  sketch_type sketch_A{transform_ptr};
  sketch_type sketch_B{transform_ptr};
  sketch_type sketch_AB{transform_ptr};

  std::mt19937                                 gen(seed);
  std::uniform_int_distribution<std::uint64_t> dist(0, domain_size - 1);
  for (std::int64_t j(0); j < stream_size; ++j) {
    std::uint64_t item_A = dist(gen);
    sketch_A.insert(item_A);
    sketch_AB.insert(item_A);
  }
  for (std::int64_t j(0); j < stream_size; ++j) {
    std::uint64_t item_B = dist(gen);
    sketch_B.insert(item_B);
    sketch_AB.insert(item_B);
  }

  // Now we will merge the sketches using the `+` operator and verify that the
  // merged sketch is equivalent to the sketch of both streams.
  sketch_type sketch_merged = sketch_A + sketch_B;
  if (sketch_merged == sketch_AB) {
    std::cout << "(+) merge check passed!" << std::endl;
  }

  // In the last merge, we created a new sketch without modifying `sketch_A` or
  // `sketch_B`. However, it is also possible to merge a sketch into another
  // in-place using the `+=` operator, modifying the first sketch. We do this
  // here and verify that the result is equivalent to the sketch of both streams
  // as expected.
  sketch_A += sketch_B;
  if (sketch_A == sketch_AB) {
    std::cout << "(+=) merge check passed!" << std::endl;
  }
  return 0;
}