// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#undef NDEBUG

#include <krowkee/hash/countsketch.hpp>

#include <krowkee/util/runtime.hpp>

#if __has_include(<cereal/cereal.hpp>)
#include <check_archive.hpp>
#endif

#include <getopt.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <iostream>

using krowkee::chirp;
using krowkee::dispatch_with_sketch_sizes;
using krowkee::do_test;
using krowkee::make_shared_functor;
using krowkee::online_statistics;
using krowkee::print_line;

template <std::size_t RangeSize>
using countsketch_hash_type =
    krowkee::hash::CountSketchHash<RangeSize, krowkee::hash::MulAddShift>;

using Clock   = std::chrono::system_clock;
using ns_type = std::chrono::nanoseconds;

/**
 * Struct bundling the experiment parameters.
 */
struct Parameters {
  std::uint64_t count;
  std::uint64_t range;
  std::uint64_t seed;
  bool          verbose;
};

template <typename HashType>
void cs_init(std::uint64_t i) {
  HashType hash{i};
}

template <std::size_t RangeSize>
struct init_check {
  const char *name() const { return "hash initialization check"; }

  void operator()() const {
    for (std::uint64_t i(0); i < 63; ++i) {
      CHECK_DOES_NOT_THROW<std::exception>(
          cs_init<countsketch_hash_type<RangeSize>>,
          "good value initialization", i);
    }
    CHECK_CONDITION(true, "good value initialization");
  }
};

template <std::size_t RangeSize>
struct empirical_histograms {
  const char *name() const { return "empirical histograms"; }

  template <typename HashType, typename... ARGS>
  void empirical_histogram(const Parameters &params, const double std_dev_range,
                           ARGS &&...args) const {
    auto                       start = Clock::now();
    HashType                   hash{params.seed, args...};
    std::size_t                range_size(hash.size());
    std::vector<std::uint64_t> register_hist(range_size);
    std::vector<std::int32_t>  polarity_hist(2);
    for (std::uint64_t i(0); i < params.count; ++i) {
      auto [register_hash, polarity_hash] = hash(i);
      ++register_hist[register_hash];
      ++polarity_hist[(polarity_hash == 1) ? 1 : 0];
    }
    {
      online_statistics os{};
      os.push(register_hist);
      double mean(os.mean()), var(os.variance()), std_dev(os.std_dev()),
          target(std_dev_range * mean);
      if (params.verbose == true) {
        std::cout << "Empirical histogram of " << params.count
                  << " elements hashed to " << range_size << " bins using "
                  << HashType::name() << " with state:" << std::endl;
        std::cout
            << "[" << hash.state() << "], ("
            << std::chrono::duration_cast<ns_type>(Clock::now() - start).count()
            << " ns):";

        for (int i(0); i < range_size; ++i) {
          if (i % 20 == 0) {
            std::cout << "\n\t";
          }
          std::cout << register_hist[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "\tmean: " << mean << ", variance: " << var
                  << ", std dev: " << std_dev
                  << ", max target std dev: " << target << std::endl;
      }
      std::stringstream ss;
      ss << HashType::name() << " register std dev";
      CHECK_CONDITION(std_dev < target, ss.str());
    }
    {
      online_statistics os{};
      os.push(polarity_hist);
      double mean(os.mean()), var(os.variance()), std_dev(os.std_dev()),
          target(std_dev_range * mean);
      if (params.verbose == true) {
        std::cout << "Empirical histogram of " << params.count
                  << " elements hashed to 2 bins using " << HashType::name()
                  << " with state:" << std::endl;
        std::cout
            << "[" << hash.state() << "], ("
            << std::chrono::duration_cast<ns_type>(Clock::now() - start).count()
            << " ns):";

        for (int i(0); i < 2; ++i) {
          if (i % 20 == 0) {
            std::cout << "\n\t";
          }
          std::cout << polarity_hist[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "\tmean: " << mean << ", variance: " << var
                  << ", std dev: " << std_dev
                  << ", max target std dev: " << target << std::endl;
      }
      std::stringstream ss;
      ss << HashType::name() << " polarity std dev";
      CHECK_CONDITION(std_dev < target, ss.str());
    }
  }

  void operator()(const Parameters &params) const {
    empirical_histogram<countsketch_hash_type<RangeSize>>(params, 0.01);
  }
};

#if __has_include(<cereal/cereal.hpp>)
template <std::size_t RangeSize>
struct serialize_check {
  const char *name() { return "serialize check"; }

  template <typename HashType, typename... ARGS>
  void serialize(const Parameters &params, ARGS &&...args) const {
    HashType hash{params.seed, args...};
    CHECK_ALL_ARCHIVES(hash, HashType::name());
  }

  void operator()(const Parameters params) const {
    serialize<countsketch_hash_type<RangeSize>>(params);
  }
};
#endif

template <std::size_t RangeSize>
struct do_experiment {
  void operator()(const Parameters &params) {
    print_line();
    print_line();
    std::cout << " Experimenting with " << params.count
              << " insertions into a range of " << params.range
              << " using random seed " << params.seed << std::endl;
    print_line();
    print_line();
    std::cout << std::endl;
    do_test<init_check<RangeSize>>();
    do_test<empirical_histograms<RangeSize>>(params);
#if __has_include(<cereal/cereal.hpp>)
    do_test<serialize_check<RangeSize>>(params);
#endif
    std::cout << std::endl;
  }
};

void print_help(char *exe_name) {
  std::cout << "\nusage:  " << exe_name << "\n"
            << "\t-c, --count <int>  - number of insertions\n"
            << "\t-s, --seed <int>   - random seed\n"
            << "\t-r, --range <int>  - range of hash functors\n"
            << "\t-v, --verbose      - print additional debug information.\n"
            << "\t-h, --help         - print this line and exit\n"
            << std::endl;
}

void parse_args(int argc, char **argv, Parameters &params) {
  int c;

  while (1) {
    int                  option_index(0);
    static struct option long_options[] = {
        {"count", required_argument, NULL, 'c'},
        {"range", required_argument, NULL, 'r'},
        {"seed", required_argument, NULL, 's'},
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}};

    int curind = optind;
    c = getopt_long(argc, argv, "-:c:r:s:vh", long_options, &option_index);
    if (c == -1) {
      break;
    }

    switch (c) {
      case 'h':
        print_help(argv[0]);
        exit(-1);
        break;
      case 0:
        printf("long option %s", long_options[option_index].name);
        if (optarg) {
          printf(" with arg %s", optarg);
        }
        printf("\n");
        break;
      case 1:
        printf("unused regular argument ignored %s\n", optarg);
        break;
      case 'c':
        params.count = std::atoll(optarg);
        break;
      case 'r':
        params.range = std::atoll(optarg);
        break;
      case 's':
        params.seed = std::atoll(optarg);
        break;
      case 'v':
        params.verbose = true;
        break;
      case '?':
        if (optopt == 0) {
          printf("Unknown long option \"%s\",", argv[curind]);
        } else {
          printf("Unknown option %c,", optopt);
        }
        printf(" consult %s --help\n", argv[0]);
        break;
      case ':':
        printf("Missing argument for option -%c/--%s\n", optopt,
               long_options[option_index].name);
        break;
      default:
        printf("?? getopt returned character code 0%o ??\n", c);
    }
  }
}

int main(int argc, char **argv) {
  std::uint64_t count(10000);
  std::uint64_t range(16);
  std::uint64_t seed(krowkee::hash::default_seed);
  bool          verbose(false);

  Parameters params{count, range, seed, verbose};

  parse_args(argc, argv, params);

  dispatch_with_sketch_sizes<do_experiment, void>(params.range, params);

  return 0;
}
