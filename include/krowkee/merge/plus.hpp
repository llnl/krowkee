// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>

namespace krowkee::merge {
template <class T>
struct plus : public std::plus<T> {
  template <typename ContainerType>
  static constexpr void eigen_inplace_merge(ContainerType       &lhs,
                                            const ContainerType &rhs) {
    lhs += rhs;
  }
};
}  // namespace krowkee::merge
