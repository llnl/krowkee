// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

// Klugy, but includes need to be in this order.

#include <check_archive.hpp>
#include <sketch_test_templates.hpp>
#include <sketch_types.hpp>

#include <krowkee/hash/hash.hpp>
#include <krowkee/transform/HyperLogLog.hpp>
#include <krowkee/util/cmap_types.hpp>
#include <krowkee/util/runtime.hpp>
#include <krowkee/util/sketch_types.hpp>

#include <parameters.hpp>

using krowkee::chirp;
using krowkee::do_test;
using krowkee::make_shared_functor;
using krowkee::print_line;

using sketch_impl_type = krowkee::util::sketch_impl_type;
using cmap_impl_type   = krowkee::util::cmap_impl_type;

template <typename SketchType>
struct init_check : krowkee::test::init_check<SketchType> {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  void operator()(const auto &params) const {
    transform_ptr_type lhs_ptr(std::make_shared<transform_type>(0));
    transform_ptr_type rhs_ptr(std::make_shared<transform_type>(0));
    sketch_type        lhs(lhs_ptr);
    sketch_type        rhs(rhs_ptr);
    sketch_type        empty_sketch(lhs_ptr);
    sketch_type        not_empty_sketch(lhs_ptr);
    not_empty_sketch.insert(1);
    std::cout << "not empty sketch empty: " << not_empty_sketch.empty()
              << std::endl;
    std::cout << not_empty_sketch << std::endl;
    for (int i(0); i < 1000; i++) {
      lhs.insert(i);
      rhs.insert(i);
    }
    this->run_tests(lhs, rhs, empty_sketch, not_empty_sketch);
  }
};

template <typename SketchType>
struct bad_merge_check : public krowkee::test::bad_merge_check<SketchType> {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  void operator()(const auto &params) const {
    transform_ptr_type lhs_ptr(std::make_shared<transform_type>(32));
    transform_ptr_type rhs_ptr(std::make_shared<transform_type>(22));
    sketch_type        lhs(lhs_ptr);
    sketch_type        rhs(rhs_ptr);
    this->run_tests(lhs, rhs);
  }
};

/**
 * Verify that merge (+/+=) operators work as expected.
 */
template <typename SketchType>
struct good_merge_check : krowkee::test::good_merge_check<SketchType> {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  void operator()(const auto &params) const {
    transform_ptr_type transform_ptr(std::make_shared<transform_type>(8));
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

#if __has_include(<cereal/cereal.hpp>)
template <typename SketchType>
struct serialize_check : krowkee::test::serialize_check<SketchType> {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  void operator()(const auto &params) const {
    transform_ptr_type transform_ptr(
        std::make_shared<transform_type>(params.seed()));

    sketch_type sketch(transform_ptr);
    for (std::uint64_t i(0); i < params.count(); sketch.insert(i++)) {
    }
    this->run_tests(*transform_ptr, sketch);
  }
};
#endif

/**
 * Execute the battery of tests for the given sketch functor.
 */
template <typename SketchType>
void perform_tests(const auto &params) {
  using sketch_type    = SketchType;
  using transform_type = typename sketch_type::transform_type;

  print_line();
  print_line();
  std::cout << "Testing " << sketch_type::full_name() << std::endl;
  print_line();
  print_line();

  std::cout << std::endl << std::endl;

  do_test<init_check<sketch_type>>(params);
  do_test<bad_merge_check<sketch_type>>(params);
  do_test<good_merge_check<sketch_type>>(params);
#if __has_include(<cereal/cereal.hpp>)
  do_test<serialize_check<sketch_type>>(params);
#endif
}

namespace krowkee {
template <class T>
struct max {
  constexpr T operator()(const T &lhs, const T &rhs) const {
    return std::max(lhs, rhs);
  }
};
}  // namespace krowkee

template <typename RegType, std::size_t RangeSize,
          template <typename> class PtrType = std::shared_ptr>
using HLL = krowkee::sketch::Sketch<
    krowkee::transform::HyperLogLog<RegType, krowkee::hash::HyperLogLogHash,
                                    RangeSize>,
    krowkee::sketch::Dense<RegType, krowkee::max, RangeSize>, PtrType>;

template <std::size_t RangeSize>
struct do_all_tests {
  void operator()(const auto &params) {
    perform_tests<HLL<std::uint8_t, RangeSize>>(params);
  }
};

struct parameters : public krowkee::test::sketch::parameters {
  using base_type = krowkee::test::sketch::parameters;

  parameters() : base_type() {}
};

int main(int argc, char **argv) {
  parameters params = krowkee::test::chirp_parameters<parameters>(argc, argv);

  // do_all_tests<32>{}(params);
  krowkee::dispatch_with_sketch_sizes<do_all_tests, void>(params.range_size(),
                                                          params);
  return 0;
}
