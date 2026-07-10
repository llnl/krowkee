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
enum class cmap_impl_type : std::uint8_t { std, boost };

cmap_impl_type get_cmap_impl_type(char *arg) {
  if (strcmp(arg, "std") == 0) {
    return cmap_impl_type::std;
#if __has_include(<boost/container/flat_map.hpp>)
  } else if (strcmp(arg, "boost") == 0) {
    return cmap_impl_type::boost;
#endif
  } else {
    std::stringstream ss;
    ss << "error: requested map type " << arg << " is not supported !";
    throw std::invalid_argument(ss.str());
  }
}
}  // namespace util
}  // namespace krowkee
