// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/hash.hpp>
#include <krowkee/util/cmap_types.hpp>
#include <krowkee/util/parameters.hpp>
#include <krowkee/util/sketch_types.hpp>

using sketch_impl_type = krowkee::util::sketch_impl_type;
using cmap_impl_type   = krowkee::util::cmap_impl_type;

struct parameters : public krowkee::parameters {
  using base_type = krowkee::parameters;

  krowkee::parameter<std::uint64_t> count;
  krowkee::parameter<std::uint64_t> seed;
  krowkee::parameter<std::uint64_t> range_size;
  krowkee::parameter<std::uint64_t> domain_size;
  krowkee::parameter<std::uint64_t> observation_count;
  krowkee::parameter<std::uint64_t> iterations;
  krowkee::parameter<std::string>   sketch_impl_type;
  krowkee::parameter<std::string>   cmap_impl_type;
  krowkee::parameter<bool>          verbose;

  parameters()
      : base_type(),
        count("count", "Number of vertices in the input graph", 'c', true,
              10000),
        seed("seed", "Random seed", 's', true, krowkee::hash::default_seed),
        range_size("range", "Range of hash functors", 'r', true, 128),
        domain_size("domain_size", "Size of the domain of the hash function",
                    'd', true, 4096),
        observation_count("observation_count", "Number of elements to insert",
                          'b', true, 16),
        iterations("iterations", "Number of timing iterations", 'i', true, 10),
        sketch_impl_type("sketch_impl_type", "Type of sketch to use", 't', true,
                         "cst"),
        cmap_impl_type("cmap_impl_type", "Type of commpacting map to use", 'm',
                       true, "std"),
        verbose("verbose?", "Whether to print verbose output", 'V', false,
                false) {
    _params.push_back(&count);
    _params.push_back(&seed);
    _params.push_back(&range_size);
    _params.push_back(&domain_size);
    _params.push_back(&observation_count);
    _params.push_back(&iterations);
    _params.push_back(&sketch_impl_type);
    _params.push_back(&cmap_impl_type);
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
