// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <parameters.hpp>
#include <sketch_test_templates.hpp>
#include <sketch_types.hpp>

#include <krowkee/hash/hll.hpp>
#include <krowkee/sketch/Dense.hpp>
#include <krowkee/sketch/Sketch.hpp>
#include <krowkee/transform/hll.hpp>

#include <krowkee/util/runtime.hpp>

#if __has_include(<cereal/cereal.hpp>)
#include <cereal/types/base_class.hpp>
#include <check_archive.hpp>
#endif

#include <cstdint>

using krowkee::online_statistics;
using krowkee::print_line;

using Clock   = std::chrono::system_clock;
using ns_type = std::chrono::nanoseconds;

template <template <std::size_t> class Func, typename ReturnType,
          typename... Args>
ReturnType dispatch_hll(const std::size_t &range_size, Args &...args) {
  switch (range_size) {
    case 16:
      return Func<16>{}(args...);
      break;
    case 32:
      return Func<32>{}(args...);
      break;
    case 64:
      return Func<64>{}(args...);
      break;
    case 128:
      return Func<128>{}(args...);
      break;
    case 256:
      return Func<256>{}(args...);
      break;
    case 512:
      return Func<512>{}(args...);
      break;
    default:
      throw std::logic_error(
          "no-replication dispatch_with_sketch_sizes() convenience function "
          "only accepts power-of-2 range size from 16-512. Hard-code or create "
          "a new dispatch function if you need an unsupported range size.");
  }
}

template <std::size_t RangeSize>
using hll_hash32_type = krowkee::hash::HyperLogLogHash32<RangeSize>;

template <typename HashType>
void hll_init(std::uint32_t i) {
  HashType hash{i};
}

template <std::size_t RangeSize>
struct init_hash_check {
  const char *name() const { return "hash initialization check"; }

  void operator()() const {
    for (std::uint64_t i(0); i < 63; ++i) {
      CHECK_DOES_NOT_THROW<std::exception>(hll_init<hll_hash32_type<RangeSize>>,
                                           "good value initialization", i);
    }
    CHECK_CONDITION(true, "good value initialization");
  }
};

#if __has_include(<cereal/cereal.hpp>)
template <std::size_t RangeSize>
struct serialize_hash_check {
  const char *name() { return "serialize hash check"; }

  template <typename HashType, typename... ARGS>
  void serialize(const auto &params, ARGS &&...args) const {
    HashType hash{(std::uint32_t)params.seed(), args...};
    CHECK_ALL_ARCHIVES(hash, HashType::name());
  }

  void operator()(const auto params) const {
    serialize<hll_hash32_type<RangeSize>>(params);
  }
};
#endif

/// here begins sketch data structure-aware tests.

namespace krowkee {
template <class T>
struct max {
  constexpr T operator()(const T &lhs, const T &rhs) const {
    return std::max(lhs, rhs);
  }
};
}  // namespace krowkee

template <std::size_t RangeSize>
using hll_sketch_type = krowkee::sketch::Sketch<
    krowkee::transform::HyperLogLog<std::uint8_t, hll_hash32_type, RangeSize>,
    krowkee::sketch::Dense<std::uint8_t, krowkee::max, RangeSize>, ptr_type>;

template <typename SketchType, template <typename> class MakePtrFunc>
struct init_check : krowkee::test::init_check<SketchType> {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  void operator()(const auto &params) const {
    make_ptr_type      _make_ptr = make_ptr_type();
    transform_ptr_type lhs_ptr(_make_ptr(0));
    transform_ptr_type rhs_ptr(_make_ptr(0));
    sketch_type        lhs(lhs_ptr);
    sketch_type        rhs(rhs_ptr);
    sketch_type        empty_sketch(lhs_ptr);
    sketch_type        not_empty_sketch(lhs_ptr);
    not_empty_sketch.insert(1);
    for (int i(0); i < 1000; i++) {
      lhs.insert(i);
      rhs.insert(i);
    }
    this->run_tests(lhs, rhs, empty_sketch, not_empty_sketch);
  }
};

template <typename SketchType, template <typename> class MakePtrFunc>
struct bad_merge_check : public krowkee::test::bad_merge_check<SketchType> {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  void operator()(const auto &params) const {
    make_ptr_type      _make_ptr = make_ptr_type();
    transform_ptr_type lhs_ptr(_make_ptr(32));
    transform_ptr_type rhs_ptr(_make_ptr(22));
    sketch_type        lhs(lhs_ptr);
    sketch_type        rhs(rhs_ptr);
    this->run_tests(lhs, rhs);
  }
};

/**
 * Verify that merge (+/+=) operators work as expected.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct good_merge_check : krowkee::test::good_merge_check<SketchType> {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  void operator()(const auto &params) const {
    make_ptr_type      _make_ptr = make_ptr_type();
    transform_ptr_type transform_ptr(_make_ptr(8));
    sketch_type        sketch_A(transform_ptr);
    sketch_type        sketch_B(transform_ptr);
    sketch_type        sketch_C(transform_ptr);
    sketch_type        sketch_AB(transform_ptr);
    sketch_type        sketch_ABC(transform_ptr);
    for (std::uint64_t i(0); i < 1000; ++i) {
      sketch_A.insert(i);
      sketch_AB.insert(i);
      sketch_ABC.insert(i);
    }
    for (std::uint64_t i(1000); i < 2000; ++i) {
      sketch_B.insert(i);
      sketch_AB.insert(i);
      sketch_ABC.insert(i);
    }
    for (std::uint64_t i(1000); i < 2000; ++i) {
      sketch_C.insert(i);
      sketch_ABC.insert(i);
    }
    this->run_tests(sketch_A, sketch_B, sketch_C, sketch_AB, sketch_ABC);
  }
};

template <template <typename> class MakePtrFunc>
struct guarantee_check {
  const char *name() { return "HLL guarantee check"; }

  void operator()(const auto params) {
    check_guarantee<16>(params);
    check_guarantee<64>(params);
    check_guarantee<512>(params);
  }

  template <std::size_t RangeSize>
  void check_guarantee(const auto params) {
    using sketch_type        = hll_sketch_type<RangeSize>;
    using transform_type     = typename sketch_type::transform_type;
    using transform_ptr_type = typename sketch_type::transform_ptr_type;
    using make_ptr_type      = MakePtrFunc<transform_type>;

    const int     iteration_count(10);
    const double  target_rate(0.5);
    double        mean_err(0.0);
    double        success_rate(0.0);
    double        thresh = 1.5 * (1.04 / std::sqrt(RangeSize));
    make_ptr_type _make_ptr{};
    for (int i(0); i < iteration_count; ++i) {
      transform_ptr_type transform_ptr(_make_ptr(params.seed() + i));
      sketch_type        sketch(transform_ptr);
      for (std::uint64_t j(0); j < params.count(); ++j) {
        sketch.insert(j);
      }

      double est     = transform_type::estimate(sketch.container());
      double err     = std::abs(params.count() - est) / params.count();
      bool   success = err < thresh;
      mean_err += err;
      success_rate += success;
      if (i == 0 && params.verbose()) {
        std::cout << "\t(tru, est, err): (" << params.count() << ", " << est
                  << ", " << err << ")" << std::endl;
      }
    }
    mean_err /= iteration_count;
    success_rate /= iteration_count;
    std::cout << "\t(range_size, mean err, threshold, success_rate) ("
              << RangeSize << ", " << mean_err << ", " << thresh << ", "
              << success_rate << ")" << std::endl;
    {
      bool success = (mean_err < thresh);
      if (success == false) {
        std::cout << "Mean estimated error is " << mean_err
                  << ", which exceeds expected threshold of " << thresh
                  << std::endl;
      }
      std::stringstream ss;
      ss << sketch_type::full_name() << " mean estimated error threshold";
      CHECK_CONDITION(success == true, ss.str());
    }
    {
      bool success = (success_rate >= target_rate);
      if (success == false) {
        std::cout << "Estimates achieve guarantee with rate " << success_rate
                  << ", which is below expected threshold of " << target_rate
                  << std::endl;
      }
      std::stringstream ss;
      ss << sketch_type::full_name() << " guarantee success rate";
      CHECK_CONDITION(success == true, ss.str());
    }

    return;
  }
};

template <typename SketchType, template <typename> class MakePtrFunc>
struct empirical_histograms {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  const char *name() { return "HLL empirical histograms check"; }

  void operator()(const auto &params) const { foo(params, 0.3); }

  void foo(const auto &params, const double std_dev_range) const {
    make_ptr_type      _make_ptr{};
    transform_ptr_type transform_ptr(_make_ptr(params.seed()));
    sketch_type        sketch(transform_ptr);
    for (std::uint64_t j(0); j < params.count(); ++j) {
      sketch.insert(j);
    }

    const auto &registers = sketch.container().registers();

    const double sum = std::accumulate(
        registers.begin(), registers.end(), 0.0,
        [](const auto &sum, const auto &reg) { return sum + reg; });

    const double mean = sum / sketch_type::range_size();

    const double var =
        std::accumulate(registers.begin(), registers.end(), 0.0,
                        [&mean](const auto &var, const auto &reg) {
                          double diff = (double)reg - mean;
                          return var + diff * diff;
                        }) /
        sketch_type::range_size();
    const double std_dev = std::sqrt(var);
    const double target  = (std_dev_range * mean);

    {
      bool success = (std_dev < target);
      // if (params.verbose()) {
      //   std::cout << "\tmean: " << mean << std::endl;
      //   std::cout << "\tvar: " << var << std::endl;
      //   std::cout << "\tstd_dev: " << std_dev << std::endl;
      //   std::cout << "\ttarget: " << target << std::endl;
      // }
      if (success == false) {
        std::cout << "empirical register histogram is:" << std::endl;
        for (const auto reg : registers) {
          std::cout << (int)reg << " ";
        }
        std::cout << std::endl;
        std::cout << "\tmean: " << mean << std::endl;
        std::cout << "\tvar: " << var << std::endl;
        std::cout << "\tstd_dev: " << std_dev << std::endl;
        std::cout << "\ttarget: " << target << std::endl;
      }
      std::stringstream ss;
      ss << sketch_type::full_name() << " register histogram check";
      CHECK_CONDITION(success == true, ss.str());
    }

    return;
  }
};

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
    krowkee::do_test<init_hash_check<RangeSize>>();
    do_test<init_check<hll_sketch_type<RangeSize>, make_ptr_functor>>(params);
    do_test<bad_merge_check<hll_sketch_type<RangeSize>, make_ptr_functor>>(
        params);
    do_test<good_merge_check<hll_sketch_type<RangeSize>, make_ptr_functor>>(
        params);
    do_test<empirical_histograms<hll_sketch_type<RangeSize>, make_ptr_functor>>(
        params);
#if __has_include(<cereal/cereal.hpp>)
    do_test<serialize_hash_check<RangeSize>>(params);
#endif
    krowkee::do_test<guarantee_check<make_ptr_functor>>(params);
    std::cout << std::endl;
  }
};

int main(int argc, char **argv) {
  using parameters = krowkee::test::hash::parameters;

  parameters params = krowkee::test::chirp_parameters<parameters>(argc, argv);

  do_experiment<32>{}(params);
  // dispatch_hll<do_experiment, void>(params.range_size(), params);

  return 0;
}
