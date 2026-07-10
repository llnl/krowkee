// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <parameters.hpp>
#include <sketch_interface.hpp>
#include <timing.hpp>
#include <types.hpp>
#include <utils.hpp>

template <typename ValueType>
struct vector_init : Vector<ValueType> {
  vector_init(const Parameters &params)
      : Vector<ValueType>(params.range_size) {}
};

template <typename KeyType, typename ValueType>
struct map_init : Map<KeyType, ValueType> {
  map_init(const Parameters &params)
      : Map<KeyType, ValueType>(params.range_size) {}
};

template <typename KeyType>
struct set_init : Set<KeyType> {
  set_init(const Parameters &params) : Set<KeyType>(params.range_size) {}
};

void benchmark(const Parameters &params) {
  using sample_type = std::uint64_t;
#if __has_include(<boost/container/flat_map.hpp>)
  auto sk_profiles =
      profile_sketch_init<dense32_cs_hist<sample_type>,
                          map_sparse32_cs_hist<sample_type>,
                          map_promotable32_cs_hist<sample_type>,
                          flatmap_sparse32_cs_hist<sample_type>,
                          flatmap_promotable32_cs_hist<sample_type>>(params);
#else
  auto sk_profiles =
      profile_sketch_init<dense32_cs_hist<sample_type>,
                          map_sparse32_cs_hist<sample_type>,
                          map_promotable32_cs_hist<sample_type>>(params);
#endif
  auto container_profiles =
      profile_container_init<vector_init<sample_type>, set_init<sample_type>,
                             map_init<sample_type, sample_type>>(params);
  print_profiles(sk_profiles, container_profiles);
}

int main(int argc, char **argv) {
  Parameters params = parse_args(argc, argv);

  std::cout << params << std::endl;

  benchmark(params);
  return 0;
}