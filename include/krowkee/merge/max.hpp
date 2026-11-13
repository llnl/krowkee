// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>

namespace krowkee::merge {
template <class T>
struct max {
  constexpr T operator()(const T &lhs, const T &rhs) const {
    return std::max(lhs, rhs);
  }

  template <typename ContainerType>
  static constexpr void eigen_inplace_merge(ContainerType       &lhs,
                                            const ContainerType &rhs) {
    lhs = lhs.cwiseMax(rhs);
  }
};
}  // namespace krowkee::merge
