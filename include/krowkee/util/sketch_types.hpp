// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>

namespace krowkee {
namespace util {
enum class sketch_impl_type : std::uint8_t {
  cst,
  fwht,
  sparse_cst,
  promotable_cst
};

sketch_impl_type get_sketch_impl_type(const char *arg) {
  if (strcmp(arg, "cst") == 0) {
    return sketch_impl_type::cst;
  } else if (strcmp(arg, "sparse_cst") == 0) {
    return sketch_impl_type::sparse_cst;
  } else if (strcmp(arg, "fwht") == 0) {
    return sketch_impl_type::fwht;
  } else if (strcmp(arg, "promotable_cst") == 0) {
    return sketch_impl_type::promotable_cst;
  } else {
    std::stringstream ss;
    ss << "error: requested sketch type " << arg << " is not supported!";
    throw std::invalid_argument(ss.str());
  }
}
}  // namespace util
}  // namespace krowkee
