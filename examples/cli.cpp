// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

// Klugy, but includes need to be in this order.

#include <krowkee/sketch.hpp>
#include <krowkee/util/runtime.hpp>

#include <getopt.h>
#include <stdio.h>
#include <unistd.h>

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

// Here we create a simple struct to hold CLI parameters.
struct Parameters {
  std::size_t range_size;
  std::size_t replication_count;

  Parameters(std::size_t range_size, std::size_t replication_count)
      : range_size(range_size), replication_count(replication_count) {}
};

// Here is a simple parser that checks the CLI for numeric parameters
// following
// `-r` and `-R` and populates the parameters accordingly.
Parameters parse_args(int argc, char **argv) {
  Parameters params{16, 4};
  int        c;

  while (1) {
    int                  option_index(0);
    static struct option long_options[] = {
        {"range_size", required_argument, NULL, 'r'},
        {"replication_count", required_argument, NULL, 'R'},
        {NULL, 0, NULL, 0}};

    int curind = optind;
    c          = getopt_long(argc, argv, "r:R:", long_options, &option_index);
    if (c == -1) {
      break;
    }

    switch (c) {
      case 'r':
        params.range_size = std::atoll(optarg);
        break;
      case 'R':
        params.replication_count = std::atoll(optarg);
        break;
      default:
        printf("?? getopt returned character code 0%o ??\n", c);
        break;
    }
  }
  return params;
}

int main(int argc, char **argv) {
  // Parse the parameters.
  Parameters params = parse_args(argc, argv);

  // This code executes the workflow with hard-coded parameters.
  runtime_functor<8, 2>{}();

  // To utilize the CLI parameters, use the following pattern.
  // `krowkee:dispatch` can also take additional parameters
  // and even supports returning a template type (`void` in this example). It
  // is also possible to define your own similar functions supporting
  // different options.
  krowkee::dispatch<runtime_functor, void>{params.range_size,
                                           params.replication_count}();

  return 0;
}