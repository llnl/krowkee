// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// write all need helper functions
#pragma once

#include <krowkee/transform/fwht/utils_temp.hpp>

namespace krowkee {
namespace transform {
namespace fwht {

template <typename RegType>
constexpr RegType rademacher_flip(RegType val, const std::uint64_t col_index,
                                  const std::uint64_t seed) {
  const std::uint64_t col_seed = seed + col_index;
  std::srand(col_seed);
  return ((std::rand() % 2 == 0)
              ? RegType(1) * val
              : RegType(-1) * val);  // assign random numbers, same random
                                     // number used for each row
}

std::vector<uint64_t> uniform_sample_vec(
    const std::uint64_t input_size, const std::uint64_t sketch_size,
    const std::uint64_t row_index,
    const std::uint64_t seed = krowkee::hash::default_seed) {
  // Initialize a default_random_engine with the seed
  const std::uint64_t        new_seed = seed + row_index;
  std::default_random_engine myRandomEngine(new_seed);
  std::vector<std::uint64_t> sketch_vec(sketch_size);
  // Initialize a uniform_int_distribution to produce values between 0 and the
  // input size
  std::uniform_int_distribution<uint64_t> myUnifIntDist(0, input_size - 1);
  // Fill the vector with uniform random variables
  for (int i = 0; i < sketch_size; i++) {
    sketch_vec[i] = myUnifIntDist(myRandomEngine);
  }
  return sketch_vec;
}

uint64_t count_set_bits(std::uint64_t num) {
  std::uint64_t count = 0;
  while (num) {
    count += num & 1;
    num >>= 1;
  }
  return count;
}

constexpr bool get_parity(std::uint64_t num) { return (num & 1); }

template <typename RegType>
constexpr RegType get_hadamard_element(const std::uint64_t row_index,
                                       const std::uint64_t col_index) {
  const std::uint64_t ele_sign = count_set_bits(row_index & col_index);
  const bool          is_odd   = get_parity(ele_sign);
  return (is_odd) ? RegType(-1) : RegType(1);
}

template <typename RegType>
constexpr std::vector<RegType> get_sketch_vector(
    const RegType val, const std::uint64_t row_index,
    const std::uint64_t col_index, const std::uint64_t num_vertices,
    const std::uint64_t sketch_size,
    const std::uint64_t seed = krowkee::hash::default_seed) {
  const RegType signed_multiplicity = rademacher_flip(val, row_index, seed);
  const std::vector<uint64_t> sample_indices =
      uniform_sample_vec(num_vertices, sketch_size, col_index, seed);
  std::vector<RegType> out_sketch(sketch_size);
  for (int i = 0; i < sketch_size; i++) {
    out_sketch[i] = signed_multiplicity *
                    get_hadamard_element<RegType>(col_index, sample_indices[i]);
  }
  return out_sketch;
}
}  // namespace fwht
}  // namespace transform
}  // namespace krowkee
