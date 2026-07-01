// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

// Klugy, but includes need to be in this order.
#pragma once

#include <check_archive.hpp>
#include <sketch_types.hpp>

#include <krowkee/hash/hash.hpp>
#include <krowkee/util/cmap_types.hpp>
#include <krowkee/util/runtime.hpp>
#include <krowkee/util/sketch_types.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

namespace krowkee::test {
using krowkee::chirp;
using krowkee::do_test;
using krowkee::make_shared_functor;
using krowkee::print_line;

/**
 * Verify that initialization and assignment (=) operators work as expected.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct init_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " constructors";
    return ss.str();
  }

  template <typename ParametersType>
  void operator()(const ParametersType &params) const {
    make_ptr_type _make_ptr = make_ptr_type();
    {
      transform_ptr_type transform_ptr_1(_make_ptr(0));
      transform_ptr_type transform_ptr_2(_make_ptr(0));
      sketch_type        sketch_1(transform_ptr_1);
      sketch_type        sketch_2(transform_ptr_2);
      for (int i(0); i < 1000; i++) {
        sketch_1.insert(i);
        sketch_2.insert(i);
      }
      bool constructors_match = sketch_1 == sketch_2;
      if (constructors_match == false) {
        std::cout << "sketch_1 : " << sketch_1 << std::endl;
        std::cout << "sketch_2 : " << sketch_2 << std::endl;
      }
      CHECK_CONDITION(constructors_match == true,
                      "constructor/insert consistency");
    }
    {
      transform_ptr_type transform_ptr(_make_ptr(0));
      sketch_type        sketch(transform_ptr);
      for (int i(0); i < 1000; sketch.insert(i++)) {
      }
      sketch_type sketch2(sketch);
      bool        copy_matches = sketch == sketch2;
      if (copy_matches == false) {
        std::cout << "sketch : " << sketch << std::endl;
        std::cout << "sketch2 : " << sketch2 << std::endl;
      }
      CHECK_CONDITION(copy_matches == true, "copy constructor");
    }
    {
      transform_ptr_type transform_ptr(_make_ptr(0));
      sketch_type        sketch(transform_ptr);
      for (int i(0); i < 1000; sketch.insert(i++)) {
      }
      sketch_type sketch2      = sketch;
      bool        swap_matches = sketch == sketch2;
      if (swap_matches == false) {
        std::cout << "sketch : " << sketch << std::endl;
        std::cout << "sketch2 : " << sketch2 << std::endl;
      }
      CHECK_CONDITION(swap_matches, "copy-and-swap assignment");
    }
    {
      transform_ptr_type transform_ptr(_make_ptr(0));
      sketch_type        sketch(transform_ptr);

      bool init_empty = sketch.empty();

      CHECK_CONDITION(init_empty == true, "initial empty");

      sketch.insert(1);
      bool not_empty = sketch.empty();
      CHECK_CONDITION(not_empty == false, "post-insert not empty");

      sketch.clear();
      bool clear_empty = sketch.empty();
      CHECK_CONDITION(clear_empty == true, "post-clear empty");
    }
  }
};

template <typename SketchType>
void check_throws_bad_plus_equals(SketchType &lhs, const SketchType &rhs) {
  lhs += rhs;
}

template <typename SketchType>
void check_throws_bad_plus(SketchType &lhs, const SketchType &rhs) {
  SketchType sketch = lhs += rhs;
}

/**
 * Verify that merge (+/+=) operators catch bad merges.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct bad_merge_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " bad merges";
    return ss.str();
  }

  void operator()(const auto &params) const {
    make_ptr_type      _make_ptr = make_ptr_type();
    transform_ptr_type transform_ptr_1(_make_ptr(32));
    transform_ptr_type transform_ptr_2(_make_ptr(22));
    sketch_type        sketch_1(transform_ptr_1);
    sketch_type        sketch_2(transform_ptr_2);
    CHECK_THROWS<std::invalid_argument>(
        check_throws_bad_plus_equals<SketchType>,
        "bad merge (+=) with different functor seeds", sketch_1, sketch_2);
    CHECK_THROWS<std::invalid_argument>(
        check_throws_bad_plus<SketchType>,
        "bad merge (+) with different functor seeds", sketch_1, sketch_2);
  }
};

/**
 * Verify that merge (+/+=) operators work as expected.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct good_merge_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " good merges";
    return ss.str();
  }

  void operator()(const auto &params) const {
    make_ptr_type      _make_ptr = make_ptr_type();
    transform_ptr_type transform_ptr(_make_ptr(8));
    sketch_type        first(transform_ptr);
    sketch_type        middle(transform_ptr);
    sketch_type        last(transform_ptr);
    sketch_type        both(transform_ptr);
    sketch_type        all(transform_ptr);
    for (std::uint64_t i(0); i < 1000; ++i) {
      first.insert(i);
      both.insert(i);
      all.insert(i);
    }
    for (std::uint64_t i(1000); i < 2000; ++i) {
      middle.insert(i);
      both.insert(i);
      all.insert(i);
    }
    for (std::uint64_t i(1000); i < 2000; ++i) {
      last.insert(i);
      all.insert(i);
    }
    first.compactify();
    middle.compactify();
    last.compactify();
    both.compactify();
    all.compactify();
    sketch_type bb = first + middle;
    bb.compactify();
    {
      bool merge_success = both == bb;
      CHECK_CONDITION(merge_success == true, "merge (+)");
    }
    {
      sketch_type aa = first + middle + last;
      aa.compactify();
      bool multimerge_success = all == aa;
      CHECK_CONDITION(multimerge_success == true, "multi-merge (+, +)");
    }
    {
      first += middle;
      bool inplace_merge_success = both == first;
      CHECK_CONDITION(inplace_merge_success == true, "merge (+=)");
    }
  }
};

#if __has_include(<cereal/cereal.hpp>)
template <typename SketchType, template <typename> class MakePtrFunc>
struct serialize_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " serialize";
    return ss.str();
  }

  void operator()(const auto &params) const {
    make_ptr_type      _make_ptr{};
    transform_ptr_type transform_ptr(_make_ptr(params.seed));

    CHECK_ALL_ARCHIVES(*transform_ptr, "sketch functor");

    sketch_type sketch(transform_ptr);
    for (std::uint64_t i(0); i < params.count; sketch.insert(i++)) {
    }
    sketch.compactify();

    CHECK_ALL_ARCHIVES(sketch.container(), "sketch container");
    CHECK_ALL_ARCHIVES(sketch, "whole sketch object");
  }
};
#endif

/**
 * Verify that promotion works as expected. Can only be compiled if
 * sketch_type::container_type is krowkee::sketch::Promotable.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct promotion_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " promotion check";
    return ss.str();
  }

  void operator()(const auto &params) const {
    make_ptr_type      _make_ptr = make_ptr_type();
    transform_ptr_type transform_ptr(_make_ptr(params.seed));
    sketch_type        s1(transform_ptr);
    sketch_type        s2(transform_ptr);
    sketch_type        d1(transform_ptr);
    sketch_type        d2(transform_ptr);
    sketch_type        d12(transform_ptr);
    sketch_type        dall(transform_ptr);
    int                pt(sketch_type::container_type::promotion_threshold() /
                          sketch_type::transform_type::replication_count());
    for (std::uint64_t i(0); i < pt - 1; ++i) {
      s1.insert(i);
      d12.insert(i);
      d1.insert(i);
      d1.insert(i);
      d2.insert(i);
      dall.insert(i);
    }
    for (std::uint64_t i(pt); i < 2 * pt - 1; ++i) {
      s2.insert(i);
      d12.insert(i);
      d1.insert(i);
      d2.insert(i);
      d2.insert(i);
      dall.insert(i);
    }
    for (std::uint64_t i(2 * pt); i < params.count; ++i) {
      d1.insert(i);
      d2.insert(i);
      dall.insert(i);
    }
    s1.compactify();
    s2.compactify();
    d1.compactify();
    d2.compactify();
    d12.compactify();
    dall.compactify();
    {
      sketch_type d1_ = s1 + dall;
      d1_.compactify();
      bool sp_merge_success = d1_ == d1;
      CHECK_CONDITION(sp_merge_success == true, "merge (+) sparse/dense");
    }
    {
      sketch_type d1_ = dall + s1;
      d1_.compactify();
      bool sp_merge_success = d1_ == d1;
      CHECK_CONDITION(sp_merge_success == true, "merge (+) dense/sparse");
    }
    {
      sketch_type dall12 = dall + s1 + s2;
      sketch_type d12_   = dall + d12;
      dall12.compactify();
      d12_.compactify();
      bool sp_multimerge_success = dall12 == d12_;
      CHECK_CONDITION(sp_multimerge_success == true,
                      "multi-merge (+) dense/sparse/sparse");
    }
    {
      sketch_type s11 = s1 + s1;
      s11.compactify();
      bool sss_merge_success =
          accumulate(s11, 0.0) == 2 * accumulate(s1, 0.0) &&
          s11.is_sparse() == true;
      CHECK_CONDITION(sss_merge_success == true,
                      "merge (+) sparse/sparse (sparse)");
    }
    {
      sketch_type s12 = s1 + s2;
      s12.compactify();
      bool ssd_merge_success = s12 == d12;
      CHECK_CONDITION(ssd_merge_success == true,
                      "merge (+) sparse/sparse (dense)");
    }
    {
      s1 += dall;
      s1.compactify();
      bool ssd_inplace_merge_success = s1 == d1;
      CHECK_CONDITION(ssd_inplace_merge_success == true,
                      "merge (+=) sparse/dense (dense)");
    }
    {
      dall += s2;
      dall.compactify();
      bool dsd_inplace_merge_success = dall == d2;
      CHECK_CONDITION(dsd_inplace_merge_success == true,
                      "merge (+=) dense/sparse (dense)");
    }
  }
};
}  // namespace krowkee::test
