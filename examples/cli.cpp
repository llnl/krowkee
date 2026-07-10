// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

// Klugy, but includes need to be in this order.

#include <krowkee/sketch.hpp>
#include <krowkee/util/parameters.hpp>
#include <krowkee/util/runtime.hpp>

#include <cstring>
#include <iostream>

// Many Krowkee transforms depend on compile-time numeric parameters. For
// instance, the Johnson-Lindenstrauss implementations
// `krowkee::sketch::SparseJLT` and `krowkee::sketch::FWHT` depend on two size
// parameters `RangeSize` and `ReplicationCount`. Krowkee includes the
// convenience function `krowkee::dispatch` for translating runtime CLI
// parameters to dispatch these workflows using the corresponding compile-time
// parameters. Note that the use of this function can drastically increase
// compile time.

// // Here we make a functor defining the runtime behavior.
template <std::size_t RangeSize, std::size_t ReplicationCount>
struct runtime_functor {
  // this runtime simply prints the name of the sketch type.
  void operator()() const {
    using sketch_type =
        krowkee::sketch::SparseJLT<float, RangeSize, ReplicationCount,
                                   std::shared_ptr>;
    std::cout << "sketch type: " << sketch_type::full_name() << std::endl;
  }
};

// Here we create a simple struct to hold CLI parameters using
// krowkee::parameters.
struct parameters : public krowkee::parameters {
  using base_type = krowkee::parameters;

  krowkee::parameter<std::uint64_t> range_size;
  krowkee::parameter<std::uint64_t> replication_count;

  parameters()
      : base_type(),
        range_size("range", "Range of hash functors", 'r', true, 32),
        replication_count("replication_count",
                          "Number of tiled sketch transforms", 'R', true, 4) {
    _params.push_back(&range_size);
    _params.push_back(&replication_count);
  }
};

int main(int argc, char **argv) {
  // Parse the parameters.
  parameters params = krowkee::parse_cmd_line<parameters>(argc, argv);

  // This code executes the workflow with hard-coded parameters.
  // runtime_functor<8, 2>{}();

  // To utilize the CLI parameters, use the following pattern.
  // `krowkee:dispatch` can also take additional parameters
  // and even supports returning a template type (`void` in this example). It
  // is also possible to define your own similar functions supporting
  // different options.
  krowkee::dispatch<runtime_functor, void>{params.range_size(),
                                           params.replication_count()}();

  return 0;
}