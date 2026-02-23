// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <krowkee/hash/hash.hpp>
#include <krowkee/sketch.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>

#include <iostream>
#include <random>

int main(int argc, char **argv) {
  // We create the YGM communicator to be used.
  ygm::comm world(&argc, &argv);
  {
    uint64_t      stream_size(20000);
    std::uint64_t domain_size(16384);
    std::uint64_t seed(krowkee::hash::default_seed);
    bool          verbose(true);

    // We define our sketch type and create a transform pointer.
    constexpr const std::size_t range_size        = 8;
    constexpr const std::size_t replication_count = 2;
    using register_type                           = float;
    using sketch_type =
        krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                                   ygm::ygm_ptr>;
    using transform_type     = typename sketch_type::transform_type;
    using transform_ptr_type = typename sketch_type::transform_ptr_type;
    transform_type     transform{seed};
    transform_ptr_type transform_ptr(world.make_ygm_ptr(transform));

    // Now we populate a `ygm::container::map` with two sketches that we will
    // then merge.
    ygm::container::map<std::string, sketch_type> sketches(world);
    sketch_type                                   sketch_A{transform_ptr};
    sketch_type                                   sketch_B{transform_ptr};
    sketch_type                                   sketch_AB{transform_ptr};
    if (world.rank0()) {
      std::mt19937                                 gen(seed);
      std::uniform_int_distribution<std::uint64_t> dist(0, domain_size - 1);
      for (std::int64_t j(0); j < stream_size; ++j) {
        std::uint64_t item_A = dist(gen);
        sketch_A.insert(item_A);
        sketch_AB.insert(item_A);
      }
      for (std::int64_t j(0); j < stream_size; ++j) {
        std::uint64_t item_B = dist(gen);
        sketch_B.insert(item_B);
        sketch_AB.insert(item_B);
      }
      sketches.async_insert("A", sketch_A);
    }
    world.barrier();

    // Now we create a YGM task to add a sketch that merges the sketch
    // associated with key "A" with `sketch_B` and checks for correctness.
    if (world.rank0()) {
      sketches.async_visit(
          "A",
          [](const std::string &name, const sketch_type &sketch_A,
             const sketch_type &sketch_B, const sketch_type &sketch_AB) {
            sketch_type sketch_merge = sketch_A + sketch_B;
            if (sketch_merge == sketch_AB) {
              std::cout << "(+) merge check passed!" << std::endl;
            }
          },
          sketch_B, sketch_AB);
    }

    // Now we create a YGM task to add a sketch that merges `sketch_B` into the
    // sketch associated with key "A" and checks for correctness.
    if (world.rank0()) {
      sketches.async_visit(
          "A",
          [](const std::string &name, sketch_type &sketch_A,
             const sketch_type &sketch_B, const sketch_type &sketch_AB) {
            sketch_A += sketch_B;
            if (sketch_A == sketch_AB) {
              std::cout << "(+=) merge check passed!" << std::endl;
            }
          },
          sketch_B, sketch_AB);
    }
  }
  return 0;
}