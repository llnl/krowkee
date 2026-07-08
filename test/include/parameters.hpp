// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/util/parameters.hpp>

namespace krowkee::test {

namespace hash {
/**
 * Struct bundling the experiment parameters.
 */
struct parameters : public krowkee::parameters {
  using base_type = krowkee::parameters;

  krowkee::parameter<std::uint64_t> count;
  krowkee::parameter<std::uint64_t> seed;
  krowkee::parameter<std::uint64_t> range_size;
  krowkee::parameter<bool>          verbose;

  parameters()
      : base_type(),
        count("count", "Number of vertices in the input graph", 'c', true,
              10000),
        seed("seed", "Random seed", 's', true, krowkee::hash::default_seed),
        range_size("range", "Range of hash functors", 'r', true, 32),
        verbose("verbose?", "Whether to print verbose output", 'V', false,
                false) {
    _params.push_back(&count);
    _params.push_back(&seed);
    _params.push_back(&range_size);
    _params.push_back(&verbose);
  }

  bool _help_needed() const override {
    bool ret = base_type::_help_needed();
    if (count() == 0) {
      std::cout << "Must indicate a positive number of matrix rows."
                << std::endl;
      return true;
    }
    return ret;
  }
};
}  // namespace hash

namespace sketch {
struct parameters : public krowkee::test::hash::parameters {
  using base_type = krowkee::test::hash::parameters;

  krowkee::parameter<std::uint64_t> replication_count;
  krowkee::parameter<std::uint64_t> domain_size;
  krowkee::parameter<std::uint64_t> observation_count;

  parameters()
      : base_type(),
        replication_count("replication_count",
                          "Number of tiled sketch transforms", 'R', true, 4),
        domain_size("domain_size", "Size of the domain of the hash function",
                    'd', true, 4096),
        observation_count("observation_count", "Number of elements to insert",
                          'b', true, 16) {
    _params.push_back(&replication_count);
    _params.push_back(&domain_size);
    _params.push_back(&observation_count);
  }
};
}  // namespace sketch

namespace matrix {
struct parameters : public krowkee::test::sketch::parameters {
  using base_type = krowkee::test::sketch::parameters;

  krowkee::parameter<std::uint64_t> internal_range_size;
  krowkee::parameter<std::uint64_t> internal_replication_count;

  parameters()
      : base_type(),
        internal_range_size("internal_range_size",
                            "Range of (internal) sketch transform", 'i', true,
                            128),
        internal_replication_count(
            "internal_replication_count",
            "Number of (internal) tiled sketch replications", 'I', true, 4) {
    _params.push_back(&internal_range_size);
    _params.push_back(&internal_replication_count);
  }
};
}  // namespace matrix

template <typename Parameters>
Parameters chirp_parameters(int argc, char **argv) {
  Parameters params = krowkee::parse_cmd_line<Parameters>(argc, argv);

  if (params.verbose()) {
    std::cout << params;
  }
  return params;
}
}  // namespace krowkee::test
