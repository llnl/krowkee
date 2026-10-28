// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#undef NDEBUG

#include <krowkee/hash/hyperloglog.hpp>

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
using hyperloglog_hash_type =
    krowkee::hash::HyperLogLogHash<RangeSize, krowkee::hash::MulAddShift>;

using Clock   = std::chrono::system_clock;
using ns_type = std::chrono::nanoseconds;

template <typename HashType>
void hll_init(std::uint64_t i) {
  HashType hash{i};
}

template <std::size_t RangeSize>
struct init_check {
  const char *name() const { return "hash initialization check"; }

  void operator()() const {
    for (std::uint64_t i(0); i < 63; ++i) {
      CHECK_DOES_NOT_THROW<std::exception>(
          hll_init<hyperloglog_hash_type<RangeSize>>,
          "good value initialization", i);
    }
    CHECK_CONDITION(true, "good value initialization");
  }
};

template <typename ContainerType>
double constexpr statistical_distance(const ContainerType &lhs,
                                      const ContainerType &rhs) {
  if (lhs.size() != rhs.size()) {
    std::stringstream ss;
    ss << "cannot compare discrete distributions with unequal sizes "
       << lhs.size() << " and " << rhs.size();
    throw std::logic_error(ss.str());
  }
  double distance(0.0);
  for (int i(0); i < lhs.size(); ++i) {
    distance += std::abs(lhs[i] - rhs[i]);
  }
  return distance / 2;
}

template <typename HashType, typename ContainerType>
double constexpr estimate_hll(const ContainerType &registers) {
  if (registers.size() != HashType::size()) {
    std::stringstream ss;
    ss << "cannot evaluate mismatched registers (size " << registers.size()
       << ") and expected size " << HashType::size();
    throw std::logic_error(ss.str());
  }
  // double alpha = 0.7213 / (1.0 + 1.079 / HashType::size());
  // alpha *= HashType::size() * HashType::size();
  double        alpha;
  std::uint32_t m_ = HashType::size();
  switch (m_) {
    case 16:
      alpha = 0.673;
      break;
    case 32:
      alpha = 0.697;
      break;
    case 64:
      alpha = 0.709;
      break;
    default:
      alpha = 0.7213 / (1.0 + 1.079 / m_);
      break;
  }
  double alphaMM_ = alpha * m_ * m_;

  constexpr const double pow_2_32     = 4294967296.0;   ///< 2^32
  constexpr const double neg_pow_2_32 = -4294967296.0;  ///< -(2^32)

  // copied from old impl
  double estimate;
  double sum = 0.0;
  for (uint32_t i = 0; i < m_; ++i) {
    sum += 1.0 / (1 << registers[i]);
  }
  estimate = alphaMM_ / sum;  // E in the original paper
  if (estimate <= 2.5 * m_) {
    uint32_t zeros = 0;
    for (uint32_t i = 0; i < m_; i++) {
      if (registers[i] == 0) {
        zeros++;
      }
    }
    if (zeros != 0) {
      estimate = m_ * std::log(static_cast<double>(m_) / zeros);
    }
  } else if (estimate > (1.0 / 30.0) * pow_2_32) {
    estimate = neg_pow_2_32 * log(1.0 - (estimate / pow_2_32));
  }
  return estimate;
}

template <std::size_t RangeSize>
struct empirical_histograms {
  const char *name() const { return "empirical counts"; }

  template <typename HashType, typename... ARGS>
  void empirical_histogram(const auto &params, const double std_dev_range,
                           ARGS &&...args) const {
    auto                       start = Clock::now();
    HashType                   hash{params.seed(), args...};
    std::size_t                range_size(hash.size());
    std::vector<std::uint64_t> index_hist(range_size);
    std::vector<std::uint64_t> zero_count_hist(HashType::max_zero_count());
    std::vector<std::uint8_t>  registers(range_size);
    for (std::uint64_t i(0); i < params.count(); ++i) {
      auto [index, zero_count] = hash(i);
      ++index_hist[index];
      ++zero_count_hist[zero_count - 1];
      registers[index] = std::max(registers[index], zero_count);
    }
    if (params.count() > range_size * 500) {
      {
        online_statistics os{};
        os.push(index_hist);
        double mean(os.mean()), var(os.variance()), std_dev(os.std_dev()),
            target(std_dev_range * mean);
        if (params.verbose() == true) {
          std::cout << "Empirical histogram of " << params.count()
                    << " elements hashed to " << range_size << " bins using "
                    << HashType::name() << " with state:" << std::endl;
          std::cout << "[" << hash.state() << "], ("
                    << std::chrono::duration_cast<ns_type>(Clock::now() - start)
                           .count()
                    << " ns):";

          for (int i(0); i < range_size; ++i) {
            if (i % 20 == 0) {
              std::cout << "\n\t";
            }
            std::cout << index_hist[i] << " ";
          }
          std::cout << std::endl;
          std::cout << "\tmean: " << mean << ", variance: " << var
                    << ", std dev: " << std_dev
                    << ", max target std dev: " << target << std::endl;
        }
        std::stringstream ss;
        ss << HashType::name() << " index std dev";
        CHECK_CONDITION(std_dev < target, ss.str());
      }
      {
        std::vector<double> zc_empirical_distribution;
        std::vector<double> zc_target_distribution;
        double              current_target = 1;
        for (int i(0); i < zero_count_hist.size(); ++i) {
          current_target /= 2;
          zc_target_distribution.push_back(current_target);
          zc_empirical_distribution.push_back((double)zero_count_hist[i] /
                                              params.count());
        }
        double distance_target(std_dev_range);
        double stat_dist = statistical_distance(zc_empirical_distribution,
                                                zc_target_distribution);
        if (params.verbose() == true) {
          std::cout << "Empirical count histogram of " << params.count()
                    << " leading zero hashes with max zero count "
                    << HashType::max_zero_count() << std::endl;
          std::cout << "\t";
          for (int i(0); i < zero_count_hist.size(); ++i) {
            std::cout << zero_count_hist[i] << " ";
          }
          std::cout << std::endl;
          std::cout << "Empirical distribution " << std::endl;
          std::cout << "\t";
          for (int i(0); i < zc_empirical_distribution.size(); ++i) {
            std::cout << zc_empirical_distribution[i] << " ";
          }
          std::cout << std::endl;
          std::cout << "Target distribution " << std::endl;
          std::cout << "\t";
          for (int i(0); i < zc_target_distribution.size(); ++i) {
            std::cout << zc_target_distribution[i] << " ";
          }
          std::cout << std::endl;
          std::cout << "Statistical distance: " << stat_dist << std::endl;
          std::cout << std::endl << std::endl;
        }
        std::stringstream ss;
        ss << HashType::name()
           << " zero count statistical distance from target ";
        CHECK_CONDITION(stat_dist < distance_target, ss.str());
      }
    }
    {
      double estimate = estimate_hll<HashType>(registers);
      double rel_err  = std::abs(estimate - params.count()) / params.count();
      if (params.verbose() == true) {
        std::cout << "Estimated cardinality of " << estimate << " given "
                  << params.count() << " inserts, relative error = " << rel_err
                  << std::endl;
        std::cout << "Registers:" << std::endl;
        for (int i(0); i < registers.size(); ++i) {
          std::cout << (int)registers[i] << " ";
        }
        std::cout << std::endl << std::endl;
      }
    }
    // {
    //   online_statistics os{};
    //   os.push(polarity_hist);
    //   double mean(os.mean()), var(os.variance()), std_dev(os.std_dev()),
    //       target(std_dev_range * mean);
    //   if (params.verbose() == true) {
    //     std::cout << "Empirical histogram of " << params.count()
    //               << " elements hashed to 2 bins using " << HashType::name()
    //               << " with state:" << std::endl;
    //     std::cout
    //         << "[" << hash.state() << "], ("
    //         << std::chrono::duration_cast<ns_type>(Clock::now() -
    //         start).count()
    //         << " ns):";

    //     for (int i(0); i < 2; ++i) {
    //       if (i % 20 == 0) {
    //         std::cout << "\n\t";
    //       }
    //       std::cout << polarity_hist[i] << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "\tmean: " << mean << ", variance: " << var
    //               << ", std dev: " << std_dev
    //               << ", max target std dev: " << target << std::endl;
    //   }
    //   std::stringstream ss;
    //   ss << HashType::name() << " polarity std dev";
    //   CHECK_CONDITION(std_dev < target, ss.str());
    // }
  }

  void operator()(const auto &params) const {
    empirical_histogram<hyperloglog_hash_type<RangeSize>>(params, 0.02);
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
    serialize<hyperloglog_hash_type<RangeSize>>(params);
  }
};
#endif

template <std::size_t RangeSize>
struct do_experiment {
  void operator()(const auto &params) {
    print_line();
    print_line();
    std::cout << " Experimenting with " << params.count()
              << " insertions into a range of " << params.range_size()
              << " using random seed " << params.seed() << std::endl;
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

int main(int argc, char **argv) {
  using parameters = krowkee::test::hash::parameters;

  parameters params = krowkee::test::chirp_parameters<parameters>(argc, argv);

  dispatch_with_sketch_sizes<do_experiment, void>(params.range_size(), params);

  return 0;
}
