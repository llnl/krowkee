// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// write all need helper functions
#pragma once

#include <krowkee/hash/util.hpp>

#include <random>

namespace krowkee {
namespace transform {
namespace fwht {

struct FWHT_no_hash {
  template <typename RegType>
  constexpr RegType rademacher_flip(
      RegType &vec, const std::uint64_t col_index,
      const std::uint64_t seed = krowkee::hash::default_seed) {
    const std::uint64_t col_seed = seed + col_index;
    std::srand(col_seed);
    // assign random numbers, same random number used for each row
    int flip = std::rand() % 2;
    if (flip == 0) {
      return RegType(-1) * &vec;
    } else {
      return RegType(1) * &vec;
    }
  }
};
}  // namespace fwht
}  // namespace transform
}  // namespace krowkee
