// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <sstream>
#include <vector>
// // #include <memory>

namespace krowkee {
namespace stream {

template <typename RegType>
struct Element {
  using register_type = RegType;

  std::uint64_t item;
  std::uint64_t identifier;
  register_type multiplicity;

  // Constructors from raw values, indices, and multiplicities
  Element(const std::uint64_t x) : item(x), identifier(0), multiplicity(1) {}
  Element(const std::pair<std::uint64_t, std::uint64_t> indices)
      : item(indices.first), identifier(indices.second), multiplicity(1) {}
  Element(const std::uint64_t x, register_type mult)
      : item(x), identifier(0), multiplicity(mult) {}
  Element(const std::uint64_t x, const std::uint64_t idx, register_type mult)
      : item(x), identifier(idx), multiplicity(mult) {}
  Element(const std::pair<std::uint64_t, std::uint64_t> indices,
          register_type                                 mult)
      : item(indices.first), identifier(indices.second), multiplicity(mult) {}

  // Copy constructor
  Element(const Element<register_type> &rhs)
      : item(rhs.item),
        identifier(rhs.identifier),
        multiplicity(rhs.multiplicity) {}
};

template <typename RegType>
std::ostream &operator<<(std::ostream &os, const Element<RegType> &item) {
  os << item.item << " " << item.identifier << " "
     << std::int64_t(item.multiplicity);
  return os;
}

}  // namespace stream
}  // namespace krowkee
