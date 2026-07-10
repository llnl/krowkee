// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#undef NDEBUG

#include <krowkee/hash/hash.hpp>
#include <krowkee/util/runtime.hpp>

#if __has_include(<cereal/cereal.hpp>)
#include <check_archive.hpp>
#endif
#include <parameters.hpp>

using krowkee::chirp;
using krowkee::dispatch_with_sketch_sizes;
using krowkee::do_test;
using krowkee::make_shared_functor;
using krowkee::online_statistics;
using krowkee::print_line;

template <std::size_t RangeSize>
using wang_hash_type = krowkee::hash::WangHash<RangeSize>;
template <std::size_t RangeSize>
using mul_shift_type = krowkee::hash::MulShift<RangeSize>;
template <std::size_t RangeSize>
using mul_add_shift_type = krowkee::hash::MulAddShift<RangeSize>;

using Clock   = std::chrono::system_clock;
using ns_type = std::chrono::nanoseconds;

template <std::size_t RangeSize>
void wh_init(std::uint64_t i) {
  wang_hash_type<RangeSize> hash{i};
}

template <std::size_t RangeSize>
struct init_check {
  const char *name() const { return "hash initialization check"; }

  void operator()() const {
    for (std::uint64_t i(0); i < 63; ++i) {
      CHECK_DOES_NOT_THROW<std::exception>(wh_init<RangeSize>,
                                           "good value initialization", i);
    }
    CHECK_CONDITION(true, "good value initialization");
  }
};

template <std::size_t RangeSize>
struct zero_check {
  const char *name() const { return "zero check"; }

  template <typename HashType, typename... ARGS>
  void zero_handling(const auto params, const double std_dev_range,
                     ARGS &&...args) const {
    auto                       start      = Clock::now();
    std::size_t                range_size = HashType::range_size();
    std::vector<std::uint64_t> hist(range_size);
    auto                       seed = params.seed();
    for (std::uint64_t i(0); i < params.count(); ++i) {
      seed = krowkee::hash::wang64(seed);
      HashType hash{seed, args...};
      ++hist[hash(0)];
    }

    online_statistics os{};
    os.push(hist);
    double mean(os.mean()), var(os.variance()), std_dev(os.std_dev()),
        target(std_dev_range * mean);
    if (params.verbose() == true) {
      std::cout
          << "Empirical histogram of zeros hashed to " << range_size
          << " bins using " << params.count() << " instances of "
          << HashType::name() << " ("
          << std::chrono::duration_cast<ns_type>(Clock::now() - start).count()
          << " ns):";

      for (int i(0); i < range_size; ++i) {
        if (i % 20 == 0) {
          std::cout << "\n\t";
        }
        std::cout << hist[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tmean: " << mean << ", variance: " << var
                << ", std dev: " << std_dev
                << ", max target std dev: " << target << std::endl;
    }
    std::stringstream ss;
    ss << HashType::name() << " std dev";
    CHECK_CONDITION(std_dev < target, ss.str());
  }

  void operator()(const auto &params) const {
    zero_handling<mul_shift_type<RangeSize>>(params, 0.07);
    zero_handling<mul_add_shift_type<RangeSize>>(params, 0.07);
  }
};

template <std::size_t RangeSize>
struct empirical_histograms {
  const char *name() const { return "empirical histograms"; }

  template <typename HashType, typename... ARGS>
  void empirical_histogram(const auto &params, const double std_dev_range,
                           ARGS &&...args) const {
    auto                       start = Clock::now();
    HashType                   hash{params.seed(), args...};
    std::size_t                range_size(hash.size());
    std::vector<std::uint64_t> hist(range_size);
    for (std::uint64_t i(0); i < params.count(); ++i) {
      ++hist[hash(i)];
    }
    online_statistics os{};
    os.push(hist);
    double mean(os.mean()), var(os.variance()), std_dev(os.std_dev()),
        target(std_dev_range * mean);
    if (params.verbose() == true) {
      std::cout << "Empirical histogram of " << params.count()
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
        std::cout << hist[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tmean: " << mean << ", variance: " << var
                << ", std dev: " << std_dev
                << ", max target std dev: " << target << std::endl;
    }
    std::stringstream ss;
    ss << HashType::name() << " std dev";
    CHECK_CONDITION(std_dev < target, ss.str());
  }

  void operator()(const auto &params) const {
    empirical_histogram<wang_hash_type<RangeSize>>(params, 0.05);
    empirical_histogram<mul_shift_type<RangeSize>>(params, 0.01);
    empirical_histogram<mul_add_shift_type<RangeSize>>(params, 0.01);
  }
};

#if __has_include(<cereal/cereal.hpp>)
template <std::size_t RangeSize>
struct serialize_check {
  const char *name() { return "serialize check"; }

  template <typename HashType, typename... ARGS>
  void serialize(const auto &params, ARGS &&...args) const {
    HashType hash{params.seed(), args...};
    CHECK_ALL_ARCHIVES(hash, HashType::name());
  }

  void operator()(const auto params) const {
    serialize<wang_hash_type<RangeSize>>(params);
    serialize<mul_shift_type<RangeSize>>(params);
    serialize<mul_add_shift_type<RangeSize>>(params);
  }
};
#endif

template <std::size_t RangeSize>
struct do_experiment {
  void operator()(const auto params) {
    print_line();
    print_line();
    std::cout << " Experimenting with " << params.count()
              << " insertions into a range of " << params.range_size()
              << " using random seed " << params.seed() << std::endl;
    print_line();
    print_line();
    std::cout << std::endl;
    do_test<init_check<RangeSize>>();
    do_test<zero_check<RangeSize>>(params);
    do_test<empirical_histograms<RangeSize>>(params);
#if __has_include(<cereal/cereal.hpp>)
    do_test<serialize_check<RangeSize>>(params);
#endif
    std::cout << std::endl;
  }
};

int main(int argc, char **argv) {
  using parameters = krowkee::test::hash::parameters;

  parameters params = krowkee::test::chirp_parameters<parameters>(argc, argv);

  dispatch_with_sketch_sizes<do_experiment, void>(params.range_size(), params);

  return 0;
}
