// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <graph.hpp>

#include <random>

template <typename ValueType>
std::vector<ValueType> make_samples(const Parameters &params) {
  std::mt19937                             gen(params.seed);
  std::uniform_int_distribution<ValueType> dist(0, params.domain_size - 1);

  std::vector<ValueType> samples(params.count);

  for (std::uint64_t i(0); i < params.count; ++i) {
    samples[i] = dist(gen);
  }
  return samples;
}

template <typename VertexType>
std::vector<edge_type<VertexType>> make_edge_samples(const Parameters &params) {
  using vertex_t = VertexType;
  using edge_t   = edge_type<VertexType>;

  std::mt19937                            gen(params.seed);
  std::uniform_int_distribution<vertex_t> dist(0, params.domain_size - 1);

  std::vector<edge_t> samples(params.count);

  for (std::uint64_t i(0); i < params.count; ++i) {
    samples[i] = {dist(gen), dist(gen)};
  }
  return samples;
}
