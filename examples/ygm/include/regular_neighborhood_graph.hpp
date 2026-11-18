// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>

using graph_type = std::map<unsigned int, std::set<unsigned int>>;

constexpr unsigned int k_hop_neighborhood_size(const unsigned int vertex_count,
                                               const unsigned int coeff,
                                               const unsigned int k) {
  return std::min(vertex_count, (k + 1) * (coeff - 1) + 1);
}

constexpr unsigned int max_hops(const unsigned int vertex_count,
                                const unsigned int coeff) {
  return std::ceil(((vertex_count - 1) / (coeff - 1)));
}

graph_type build_regular_graph(unsigned int coeff, unsigned int k) {
  if (coeff <= 2 || k <= 2 || k >= 8 || (coeff % 2 == 0)) {
    throw std::invalid_argument("Require odd coeff > 2 and 2 < k < 8");
  }

  constexpr unsigned int max_uint = std::numeric_limits<unsigned int>::max();

  if (coeff > (max_uint >> k)) {
    throw std::overflow_error("c * (1u << k) overflows unsigned int");
  }

  const unsigned int n       = coeff * (1u << k);
  const std::int64_t radius  = (coeff - 1) / 2;
  const std::int64_t modulus = n;

  graph_type graph;

  for (unsigned int i = 0; i < n; ++i) {
    auto &neighbors = graph[i];

    for (std::int64_t delta = -radius; delta <= radius; ++delta) {
      std::int64_t value = (static_cast<std::int64_t>(i) + delta) % modulus;
      if (value < 0) {
        value += modulus;
      }

      neighbors.insert(static_cast<unsigned int>(value));
    }
  }

  return graph;
}
