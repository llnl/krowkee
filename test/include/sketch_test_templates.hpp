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
template <typename SketchType>
struct init_check {
  using sketch_type    = SketchType;
  using transform_type = typename sketch_type::transform_type;

  std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " constructors";
    return ss.str();
  }

  void chirp_pair(bool success, std::string &&lhs_name, sketch_type &lhs,
                  std::string &&rhs_name, sketch_type &rhs,
                  std::size_t size) const {
    if (success == false && size <= 256) {
      std::cout << lhs_name << ": " << lhs << std::endl;
      std::cout << rhs_name << ": " << rhs << std::endl;
    }
  }

  void chirp(bool success, std::string &&name, sketch_type &sketch,
             std::size_t size) const {
    if (success == false && size <= 256) {
      std::cout << name << ": " << sketch << std::endl;
    }
  }

  void run_tests(sketch_type &lhs, sketch_type &rhs, sketch_type &empty_sketch,
                 sketch_type &not_empty_sketch) const {
    std::size_t size = lhs.size();
    {
      bool constructors_match = lhs == rhs;
      chirp_pair(constructors_match == true, "lhs", lhs, "rhs", rhs, size);
      CHECK_CONDITION(constructors_match == true,
                      "constructor/insert consistency");
    }
    {
      bool init_empty = empty_sketch.empty();
      chirp(init_empty == true, "empty_sketch", empty_sketch, size);
      CHECK_CONDITION(init_empty == true, "initial empty");
    }
    {
      bool not_empty = not_empty_sketch.empty();
      chirp(not_empty == false, "not_empty_sketch", not_empty_sketch, size);
      CHECK_CONDITION(not_empty == false, "post-insert not empty");
    }
    {
      not_empty_sketch.clear();
      bool clear_empty = not_empty_sketch.empty();
      chirp(clear_empty == true, "cleared_sketch", not_empty_sketch, size);
      CHECK_CONDITION(clear_empty == true, "post-clear empty");
    }

    {
      sketch_type copy(lhs);
      bool        copy_matches = lhs == copy;
      chirp_pair(copy_matches == true, "sketch", lhs, "copy", copy, size);
      CHECK_CONDITION(copy_matches == true, "copy constructor");
    }
    {
      sketch_type swap         = lhs;
      bool        swap_matches = lhs == swap;
      chirp_pair(swap_matches == true, "sketch", lhs, "swap", swap, size);
      CHECK_CONDITION(swap_matches == true, "copy-and-swap assignment");
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
template <typename SketchType>
struct bad_merge_check {
  using sketch_type    = SketchType;
  using transform_type = typename sketch_type::transform_type;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " bad merges";
    return ss.str();
  }

  void run_tests(sketch_type &lhs, sketch_type &rhs) const {
    CHECK_THROWS<std::invalid_argument>(
        check_throws_bad_plus_equals<sketch_type>,
        "bad merge (+=) with different functor seeds", lhs, rhs);
    CHECK_THROWS<std::invalid_argument>(
        check_throws_bad_plus<sketch_type>,
        "bad merge (+) with different functor seeds", lhs, rhs);
  }
};

/**
 * Verify that merge (+/+=) operators work as expected.
 */
template <typename SketchType>
struct good_merge_check {
  using sketch_type    = SketchType;
  using transform_type = typename sketch_type::transform_type;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " good merges";
    return ss.str();
  }

  void chirp(bool success, std::string &&lhs_name, sketch_type &lhs,
             std::string &&rhs_name, sketch_type &rhs,
             std::size_t &size) const {
    if (success == false) {
      std::cout << "fails with mean absolute error "
                << krowkee::sketch::detail::mean_absolute_error(
                       lhs.container().registers(), rhs.container().registers())
                << std::endl;
      if (size <= 256) {
        std::cout << lhs_name << ": " << std::endl
                  << lhs << std::endl
                  << std::endl;
        std::cout << rhs_name << ": " << std::endl
                  << rhs << std::endl
                  << std::endl;
      }
    }
  }

  void run_tests(sketch_type &sketch_A, sketch_type &sketch_B,
                 sketch_type &sketch_C, sketch_type &sketch_AB,
                 sketch_type &sketch_ABC) const {
    sketch_A.compactify();
    sketch_B.compactify();
    sketch_C.compactify();
    sketch_AB.compactify();
    sketch_ABC.compactify();
    sketch_type merge_AB = sketch_A + sketch_B;
    merge_AB.compactify();

    std::size_t size = sketch_A.size();
    {
      bool merge_success = sketch_AB == merge_AB;
      chirp(merge_success, "merge_AB", merge_AB, "sketch_AB", sketch_AB, size);
      CHECK_CONDITION(merge_success == true, "merge (+)");
    }
    {
      sketch_type merge_ABC = sketch_A + sketch_B + sketch_C;
      merge_ABC.compactify();
      bool multimerge_success = sketch_ABC == merge_ABC;
      chirp(multimerge_success, "merge_ABC", merge_ABC, "sketch_ABC",
            sketch_ABC, size);
      CHECK_CONDITION(multimerge_success == true, "multi-merge (+, +)");
    }
    {
      sketch_A += sketch_B;
      bool inplace_merge_success = sketch_AB == sketch_A;
      chirp(inplace_merge_success, "sketch_A", sketch_A, "sketch_AB", sketch_AB,
            size);
      CHECK_CONDITION(inplace_merge_success == true, "merge (+=)");
    }
  }
};

#if __has_include(<cereal/cereal.hpp>)
template <typename SketchType>
struct serialize_check {
  using sketch_type    = SketchType;
  using transform_type = typename sketch_type::transform_type;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " serialize";
    return ss.str();
  }

  void run_tests(transform_type &transform, sketch_type &sketch) const {
    sketch.compactify();

    CHECK_ALL_ARCHIVES(transform, "sketch functor");
    CHECK_ALL_ARCHIVES(sketch.container(), "sketch container");
    CHECK_ALL_ARCHIVES(sketch, "whole sketch object");
  }
};
#endif

/**
 * Verify that promotion works as expected. Can only be compiled if
 * sketch_type::container_type is krowkee::sketch::Promotable.
 */
template <typename SketchType>
struct promotion_check {
  using sketch_type    = SketchType;
  using transform_type = typename sketch_type::transform_type;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " promotion check";
    return ss.str();
  }

  void run_tests(sketch_type &sparse_A, sketch_type &sparse_B,
                 sketch_type &dense_AABC, sketch_type &dense_ABBC,
                 sketch_type &dense_AB, sketch_type &dense_ABC) const {
    sparse_A.compactify();
    sparse_B.compactify();
    dense_AABC.compactify();
    dense_ABBC.compactify();
    dense_AB.compactify();
    dense_ABC.compactify();
    {
      sketch_type dense_AABC_ = sparse_A + dense_ABC;
      dense_AABC_.compactify();
      bool sp_merge_success = dense_AABC_ == dense_AABC;
      CHECK_CONDITION(sp_merge_success == true, "merge (+) sparse/dense");
    }
    {
      sketch_type dense_AABC_ = dense_ABC + sparse_A;
      dense_AABC_.compactify();
      bool sp_merge_success = dense_AABC_ == dense_AABC;
      CHECK_CONDITION(sp_merge_success == true, "merge (+) dense/sparse");
    }
    {
      sketch_type dense_AABBC  = dense_ABC + sparse_A + sparse_B;
      sketch_type dense_AABBC_ = dense_ABC + dense_AB;
      dense_AABBC.compactify();
      dense_AABBC_.compactify();
      bool sp_multimerge_success = dense_AABBC == dense_AABBC_;
      CHECK_CONDITION(sp_multimerge_success == true,
                      "multi-merge (+) dense/sparse/sparse");
    }
    {
      sketch_type sparse_AA = sparse_A + sparse_A;
      sparse_AA.compactify();
      bool sparse_AA_merge_success =
          accumulate(sparse_AA, 0.0) == 2 * accumulate(sparse_A, 0.0) &&
          sparse_AA.is_sparse() == true;
      CHECK_CONDITION(sparse_AA_merge_success == true,
                      "merge (+) sparse/sparse (sparse)");
    }
    {
      sketch_type sparse_AB = sparse_A + sparse_B;
      sparse_AB.compactify();
      bool sparse_AB_merge_success =
          sparse_AB == dense_AB && sparse_AB.is_sparse() == false;
      CHECK_CONDITION(sparse_AB_merge_success == true,
                      "merge (+) sparse/sparse (dense)");
    }
    {
      sparse_A += dense_ABC;
      sparse_A.compactify();
      bool ssd_inplace_merge_success = sparse_A == dense_AABC;
      CHECK_CONDITION(ssd_inplace_merge_success == true,
                      "merge (+=) sparse/dense (dense)");
    }
    {
      dense_ABC += sparse_B;
      dense_ABC.compactify();
      bool dsd_inplace_merge_success = dense_ABC == dense_ABBC;
      CHECK_CONDITION(dsd_inplace_merge_success == true,
                      "merge (+=) dense/sparse (dense)");
    }
  }
};
}  // namespace krowkee::test
