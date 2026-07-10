// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <parameters.hpp>
#include <sampling.hpp>
#include <sketch_interface.hpp>
#include <timing.hpp>
#include <types.hpp>
#include <utils.hpp>

template <typename ValueType>
struct vector_hist : public Vector<ValueType> {
  vector_hist(const Parameters &params)
      : Vector<ValueType>(params.domain_size) {}

  template <typename KeyType>
  void insert(const KeyType idx) {
    this->check_bounds(idx);
    this->_vec[idx]++;
  }
};

template <typename KeyType, typename ValueType>
struct map_hist : Map<KeyType, ValueType> {
  map_hist(const Parameters &params)
      : Map<KeyType, ValueType>(params.domain_size) {}

  void insert(const KeyType idx) {
    this->check_bounds(idx);
    this->_map[idx]++;
  }
};

template <typename KeyType>
struct set_hist : Set<KeyType> {
  set_hist(const Parameters &params) : Set<KeyType>(params.domain_size) {}

  void insert(const KeyType idx) {
    this->check_bounds(idx);
    this->_set.insert(idx);
  }
};

void benchmark(const Parameters &params) {
  using sample_type = std::uint64_t;
  std::vector<sample_type> samples(make_samples<sample_type>(params));

#if __has_include(<boost/container/flat_map.hpp>)
  auto sk_profiles =
      profile_sketch_histogram<sample_type, dense32_cs_hist<sample_type>,
                               map_sparse32_cs_hist<sample_type>,
                               map_promotable32_cs_hist<sample_type>,
                               flatmap_sparse32_cs_hist<sample_type>,
                               flatmap_promotable32_cs_hist<sample_type>>(
          samples, params);
#else
  auto sk_profiles =
      profile_sketch_histogram<sample_type, dense32_cs_hist<sample_type>,
                               map_sparse32_cs_hist<sample_type>,
                               map_promotable32_cs_hist<sample_type>>(samples,
                                                                      params);
#endif

  auto hist_profiles =
      profile_container_histogram<sample_type, vector_hist<sample_type>,
                                  set_hist<sample_type>,
                                  map_hist<sample_type, sample_type>>(samples,
                                                                      params);
  print_profiles(sk_profiles, hist_profiles);
}

int main(int argc, char **argv) {
  Parameters params = parse_args(argc, argv);

  std::cout << params << std::endl;

  benchmark(params);
  return 0;
}