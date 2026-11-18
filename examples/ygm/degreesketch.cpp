// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <regular_neighborhood_graph.hpp>

#include <krowkee/HyperLogLog.hpp>
#include <krowkee/util/runtime.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>

/**
 * @brief Check how well all k-hop neighborhoods estimate the true constant
 * neighborhood size
 *
 * @tparam ContainerType ygm container type.
 * @param sketches the k-hop sketch container.
 * @param target the true k-hop neighborhood size for each vertex.
 * @param level the current k-hop level.
 */
template <typename ContainerType>
void test_level_guarantee(const ContainerType &sketches, const double target,
                          const int level) {
  using index_type  = typename ContainerType::key_type;
  using sketch_type = typename ContainerType::mapped_type;

  // Set some parameters
  ygm::comm        &comm       = sketches.comm();
  const int         world_size = comm.size();
  const std::size_t range_size = sketch_type::range_size();
  const std::string full_name  = sketch_type::full_name();
  const double      thresh     = 1.5 * (1.04 / std::sqrt(range_size));

  // Create metrics
  double mean_err     = 0.0;
  double success_rate = 0.0;

  // Compare each k-hop sketch's estimate to the target and record error
  sketches.for_all([&target, &world_size, &thresh, &mean_err, &success_rate](
                       const index_type &idx, const sketch_type &sketch) {
    // Invoke the transform's static estimator to the sketch container.
    double est = sketch_type::transform_type::estimate(sketch.container());
    double err = std::abs(target - est) / target;

    bool guarantee_success = (err < thresh);

    mean_err += err;
    success_rate += guarantee_success;
  });
  comm.barrier();

  // Accounting and reporting.
  mean_err     = ygm::sum(mean_err, comm) / sketches.size();
  success_rate = ygm::sum(success_rate, comm) / sketches.size();
  comm.cout0(full_name, " ", level,
             "-hop neighborhood estimates (true size:", target,
             "):\n\tmean estimated error: ", mean_err,
             "\n\tstandard error threshold: ", thresh,
             "\n\tguarantee success rate: ", success_rate);
}

/**
 * @brief Create the (k+1)-hop sketches given the k-hop sketches and the exact
 * graph and recurse until the neighborhood function has saturated.
 *
 * @tparam ContainerType The ygm::container to be used.
 * @param last_sketches the k-hop sketch container.
 * @param graph the exact graph with adjacency lists.
 * @param coeff the underlying graph construction coefficient.
 * @param level The current k-hop layer of the experiment minus one.
 * @param exponent the underlying graph construction exponent.
 */
template <typename ContainerType>
void create_and_test_next_sketches(ContainerType     &last_sketches,
                                   const graph_type  &graph,
                                   const unsigned int coeff,
                                   const unsigned int level,
                                   const unsigned int exponent) {
  using index_type  = typename ContainerType::key_type;
  using sketch_type = typename ContainerType::mapped_type;

  // Utilize graph construction information
  const unsigned int vertex_count = coeff * (1u << exponent);
  const unsigned int level_max    = max_hops(vertex_count, coeff);
  const unsigned int target =
      k_hop_neighborhood_size(vertex_count, coeff, level);

  // terminate if the neighborhood function has saturated, which we know a prior
  // by construction.
  if (level > level_max) {
    return;
  }

  // Create the (k+1)-hop sketches, initially by copying the k-hop sketches.
  ContainerType next_sketches(last_sketches);

  // For each vertex idx, forward the k-hop sketch to each of idx's neighbors in
  // the graph and merge it into their (k+1)-hop sketch.
  last_sketches.for_all([&next_sketches, &graph](
                            const index_type  &idx,
                            const sketch_type &last_sketch) {
    for (const index_type &nbr : graph.at(idx)) {
      next_sketches.async_visit(
          nbr,
          [](auto ptr, const index_type &idx, sketch_type &next_sketch,
             const sketch_type &last_sketch) { next_sketch += last_sketch; },
          last_sketch);
    }
  });
  last_sketches.comm().barrier();

  // Check the accuracy of neighborhood size estimates for each (k+1)-hop
  // sketch.
  test_level_guarantee(next_sketches, target, level + 1);
  // Recurse, incrementing the hop level.
  create_and_test_next_sketches(next_sketches, graph, coeff, level + 1,
                                exponent);
  return;
}

/**
 * @brief Executable illustrating the DegreeSketch algorithm [1], a distributed
 * generalization of HyperANF [2].
 *
 * This example constructs a simple regular graph in memory whose k-hop
 * neighborhood function is constant for each k. It then iteratively embeds
 * k-hop cardinality sketches estimating each vertex's neighborhood function.
 * Finally, it prints performance metrics to illustrate the accuracy of the
 * sketches.
 *
 * [1] https://arxiv.org/abs/2004.04289
 * [2] https://dl.acm.org/doi/abs/10.1145/1963405.1963493
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    // set important parameters
    constexpr const std::uint32_t seed         = 1;
    constexpr const std::uint32_t coeff        = 51;
    constexpr const std::uint32_t exponent     = 3;
    constexpr const std::uint32_t vertex_count = coeff * (1u << exponent);

    const int world_size = world.size();
    const int rank       = world.rank();

    // We define our sketch type and create a transform pointer.
    constexpr const std::size_t range_size = 64;
    using register_type                    = std::uint8_t;
    using sketch_type =
        krowkee::sketch::HyperLogLog32<std::uint8_t, range_size, ygm::ygm_ptr>;
    using transform_type     = typename sketch_type::transform_type;
    using transform_ptr_type = typename sketch_type::transform_ptr_type;
    transform_type     transform{seed};
    transform_ptr_type transform_ptr(world.make_ygm_ptr(transform));

    // First we create a simple graph representation. This graph is regular and
    // its k-hop neighborhood function is also regular for each k. The details
    // of the graph creation are abstracted, as they are not necessary to
    // understand the example.
    graph_type graph = build_regular_graph(coeff, exponent);

    // Now we populate a `ygm::container::map` with two sketches that we will
    // then merge.
    ygm::container::map<const int, sketch_type> sketches(world);

    for (const auto &[idx, nbhd] : graph) {
      if (idx % world_size == rank) {
        sketch_type sketch(transform_ptr);
        for (const unsigned int &j : nbhd) {
          sketch.insert(j);
        }
        sketches.async_insert(idx, sketch);
      }
    }
    world.barrier();  // universe is now seeded

    // We now evaluate the approximation guarantees for the 1-hop neighborhoods
    // (degrees).
    test_level_guarantee(sketches, coeff, 1);

    // We iteratively evaluate the approximation guarantees for the neighborhood
    // functions until the graph saturates into a complete graph. Inspect the
    // function for detail on how the next k-hop sketch level is computed.
    create_and_test_next_sketches(sketches, graph, coeff, 1, exponent);
  }
  return 0;
}