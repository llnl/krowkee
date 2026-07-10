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
struct vector_vector_graph : public Vector<std::vector<ValueType>> {
  vector_vector_graph(const Parameters &params)
      : Vector<std::vector<ValueType>>(
            params.domain_size, std::vector<ValueType>(params.domain_size)) {}

  void insert(const edge_type<ValueType> edge) {
    this->check_bounds(edge.first);
    this->check_bounds(edge.second);
    this->_vec[edge.first][edge.second]++;
  }

  static std::string name() { return "std::vector<std::vector>"; }

  std::vector<ValueType> &operator[](const int i) { return this->_vec[i]; }
};

// key type must equal value type because of how edge_type is constructed
template <typename ValueType>
struct map_vector_graph : public Map<ValueType, std::vector<ValueType>> {
  map_vector_graph(const Parameters &params)
      : Map<ValueType, std::vector<ValueType>>(params.domain_size),
        _default_value(params.domain_size) {}

  void insert(const edge_type<ValueType> edge) {
    const auto [src, dst] = edge;
    this->check_bounds(src);
    this->check_bounds(dst);
    auto range = this->_map.equal_range(src);
    if (range.first == range.second) {
      this->_map.insert(std::make_pair(src, _default_value));
    }
    this->_map[src][dst]++;
  }

  static std::string name() { return "std::map<std::vector>"; }

 protected:
  std::vector<ValueType> _default_value;
};

// key type must equal value type because of how edge_type is constructed
template <typename ValueType>
struct map_set_graph : public Map<ValueType, std::set<ValueType>> {
  map_set_graph(const Parameters &params)
      : Map<ValueType, std::set<ValueType>>(params.domain_size) {}

  void insert(const edge_type<ValueType> edge) {
    const auto [src, dst] = edge;
    this->check_bounds(src);
    this->check_bounds(dst);
    this->_map[src].insert(dst);
  }

  static std::string name() { return "std::map<std::set>"; }
};

void benchmark(const Parameters &params) {
  using vertex_type = std::uint64_t;
  using edge_type   = edge_type<vertex_type>;
  std::vector<edge_type> samples(make_edge_samples<vertex_type>(params));

#if __has_include(<boost/container/flat_map.hpp>)
  auto sk_profiles = profile_sketch_histogram<
      edge_type, dense32_cs_vector_graph<vertex_type>,
      map_sparse32_cs_vector_graph<vertex_type>,
      map_promotable32_cs_vector_graph<vertex_type>,
      flatmap_sparse32_cs_vector_graph<vertex_type>,
      flatmap_promotable32_cs_vector_graph<vertex_type>,
      dense32_cs_map_graph<vertex_type>, map_sparse32_cs_map_graph<vertex_type>,
      map_promotable32_cs_map_graph<vertex_type>,
      flatmap_sparse32_cs_map_graph<vertex_type>,
      flatmap_promotable32_cs_map_graph<vertex_type>>(samples, params);
#else
  auto sk_profiles =
      profile_sketch_histogram<edge_type, dense32_cs_vector_graph<vertex_type>,
                               map_sparse32_cs_vector_graph<vertex_type>,
                               map_promotable32_cs_map_graph<vertex_type>,
                               dense32_cs_map_graph<vertex_type>,
                               map_sparse32_cs_map_graph<vertex_type>,
                               map_promotable32_cs_map_graph<vertex_type>>(
          samples, params);
#endif

  auto container_profiles =
      profile_container_histogram<edge_type, vector_vector_graph<vertex_type>,
                                  map_vector_graph<vertex_type>,
                                  map_set_graph<vertex_type>>(samples, params);
  print_profiles(sk_profiles, container_profiles);
}

int main(int argc, char **argv) {
  Parameters params = parse_args(argc, argv);

  std::cout << params << std::endl;

  benchmark(params);
  return 0;
}