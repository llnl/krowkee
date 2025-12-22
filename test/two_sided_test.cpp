// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

// Klugy, but includes need to be in this order.

#include <check_archive.hpp>
#include <sketch_types.hpp>

#include <krowkee/hash/hash.hpp>
#include <krowkee/util/runtime.hpp>

#include <getopt.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

using krowkee::chirp;
using krowkee::dispatch_with_sketch_sizes;
using krowkee::do_test;
using krowkee::make_shared_functor;
using krowkee::print_line;

/**
 * Struct bundling the experiment parameters.
 */
struct Parameters {
  std::uint64_t count;
  std::uint64_t range_size;
  std::uint64_t replication_count;
  std::uint64_t domain_size;
  std::uint64_t observation_count;
  std::uint64_t seed;
  bool          verbose;
};

template <typename... Sketches>
void sketch_both(Eigen::MatrixXf matrix, Sketches &...sketches) {
  int i(0);
  for (const auto &row : matrix.rowwise()) {
    int j(0);
    for (const auto &element : row) {
      (..., (sketches.insert({i, j}, element)));
      ++j;
    }
    ++i;
  }
}

template <typename MatrixType, typename TransformPtrType>
constexpr MatrixType sketch_cols(const MatrixType       &matrix,
                                 const TransformPtrType &transform_ptr) {
  MatrixType sketch_matrix(
      static_cast<std::size_t>(matrix.rows()),
      transform_ptr->range_size() * transform_ptr->replication_count());
  // required to initialize matrix coeffs
  sketch_matrix.setZero(
      static_cast<std::size_t>(matrix.rows()),
      transform_ptr->range_size() * transform_ptr->replication_count());

  int i(0);
  for (const auto &row : matrix.rowwise()) {
    int j(0);
    for (const auto &element : row) {
      auto hashes = transform_ptr->apply(j);
      for (int k(0); k < hashes.first.size(); ++k) {
        auto idx      = hashes.first[k];
        auto polarity = hashes.second[k];
        // if (matrix(i, j) != 0.0) {
        //   std::cout << "row sketch applying " << polarity << " to (" << i <<
        //   ","
        //             << idx << ") with insert " << j << std::endl;
        // }
        sketch_matrix(i, idx) += polarity * matrix(i, j);
      }
      ++j;
    }
    ++i;
  }
  return sketch_matrix;
}

template <typename MatrixType, typename TransformPtrType>
constexpr MatrixType sketch_rows(const MatrixType       &matrix,
                                 const TransformPtrType &transform_ptr) {
  MatrixType sketch_matrix(
      transform_ptr->range_size() * transform_ptr->replication_count(),
      static_cast<std::size_t>(matrix.cols()));
  // required to initialize matrix coeffs
  sketch_matrix.setZero(
      transform_ptr->range_size() * transform_ptr->replication_count(),
      static_cast<std::size_t>(matrix.cols()));

  int j(0);
  for (const auto &col : matrix.colwise()) {
    int i(0);
    for (const auto &element : col) {
      auto hashes = transform_ptr->apply(i);
      for (int k(0); k < hashes.first.size(); ++k) {
        auto idx      = hashes.first[k];
        auto polarity = hashes.second[k];
        // if (matrix(i, j) != 0.0) {
        //   std::cout << "col sketch applying " << polarity << " to (" << idx
        //             << "," << j << ") with insert " << i << std::endl;
        // }
        sketch_matrix(idx, j) += polarity * matrix(i, j);
      }
      ++i;
    }
    ++j;
  }
  return sketch_matrix;
}

/**
 * Verify that initialization and assignment (=) operators work as expected.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct init_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;
  using row_transform_type = typename transform_type::row_transform_type;
  using row_transform_ptr_type =
      typename transform_type::row_transform_ptr_type;
  using make_row_ptr_type  = MakePtrFunc<row_transform_type>;
  using col_transform_type = typename transform_type::col_transform_type;
  using col_transform_ptr_type =
      typename transform_type::col_transform_ptr_type;
  using make_col_ptr_type = MakePtrFunc<col_transform_type>;

  std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " constructors";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type     _make_ptr     = make_ptr_type();
    make_row_ptr_type _make_row_ptr = make_row_ptr_type();
    make_col_ptr_type _make_col_ptr = make_col_ptr_type();

    row_transform_ptr_type row_transform_ptr(_make_row_ptr(0));
    col_transform_ptr_type col_transform_ptr(_make_col_ptr(1));

    Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(16, 16);
    {
      transform_ptr_type transform_ptr(
          _make_ptr(row_transform_ptr, col_transform_ptr));
      sketch_type sketch(transform_ptr);

      bool init_empty = sketch.empty();

      CHECK_CONDITION(init_empty == true, "initial empty");

      sketch.insert({1, 1});
      bool not_empty = sketch.empty();
      if (not_empty == true) {
        std::cout << "sketch: \n" << sketch << std::endl;
      }
      CHECK_CONDITION(not_empty == false, "post-insert not empty");

      sketch.clear();
      bool clear_empty = sketch.empty();
      CHECK_CONDITION(clear_empty == true, "post-clear empty");
    }
    {
      transform_ptr_type transform_ptr_1(
          _make_ptr(row_transform_ptr, col_transform_ptr));
      transform_ptr_type transform_ptr_2(
          _make_ptr(row_transform_ptr, col_transform_ptr));
      sketch_type sketch_1(transform_ptr_1);
      sketch_type sketch_2(transform_ptr_2);

      sketch_both(matrix, sketch_1, sketch_2);

      bool constructors_match = sketch_1 == sketch_2;
      if (constructors_match == false) {
        std::cout << "sketch_1:" << std::endl
                  << sketch_1 << std::endl
                  << std::endl;
        std::cout << "sketch_2:" << std::endl << sketch_2 << std::endl;
      }
      CHECK_CONDITION(constructors_match == true,
                      "constructor/insert consistency");

      sketch_type sketch_3(sketch_1);
      bool        copy_matches = sketch_1 == sketch_3;
      if (copy_matches == false) {
        std::cout << "sketch_1:" << std::endl
                  << sketch_1 << std::endl
                  << std::endl;
        std::cout << "sketch_3:" << std::endl << sketch_3 << std::endl;
      }
      CHECK_CONDITION(copy_matches == true, "copy constructor");

      sketch_type sketch_4     = sketch_1;
      bool        swap_matches = sketch_1 == sketch_4;
      if (swap_matches == false) {
        std::cout << "sketch1:" << std::endl
                  << sketch_1 << std::endl
                  << std::endl;
        std::cout << "sketch3:" << std::endl << sketch_4 << std::endl;
      }
      CHECK_CONDITION(swap_matches, "copy-and-swap assignment");
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
  using row_transform_type = typename transform_type::row_transform_type;
  using row_transform_ptr_type =
      typename transform_type::row_transform_ptr_type;
  using make_row_ptr_type  = MakePtrFunc<row_transform_type>;
  using col_transform_type = typename transform_type::col_transform_type;
  using col_transform_ptr_type =
      typename transform_type::col_transform_ptr_type;
  using make_col_ptr_type = MakePtrFunc<col_transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " bad merges";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type     _make_ptr     = make_ptr_type();
    make_row_ptr_type _make_row_ptr = make_row_ptr_type();
    make_col_ptr_type _make_col_ptr = make_col_ptr_type();

    row_transform_ptr_type row_transform_ptr_1(_make_row_ptr(32));
    col_transform_ptr_type col_transform_ptr_1(_make_col_ptr(1));
    row_transform_ptr_type row_transform_ptr_2(_make_row_ptr(22));
    col_transform_ptr_type col_transform_ptr_2(_make_col_ptr(2));

    transform_ptr_type transform_ptr_1(
        _make_ptr(row_transform_ptr_1, col_transform_ptr_1));
    transform_ptr_type transform_ptr_2(
        _make_ptr(row_transform_ptr_2, col_transform_ptr_2));
    sketch_type sketch_1(transform_ptr_1);
    sketch_type sketch_2(transform_ptr_2);
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
  using row_transform_type = typename transform_type::row_transform_type;
  using row_transform_ptr_type =
      typename transform_type::row_transform_ptr_type;
  using make_row_ptr_type  = MakePtrFunc<row_transform_type>;
  using col_transform_type = typename transform_type::col_transform_type;
  using col_transform_ptr_type =
      typename transform_type::col_transform_ptr_type;
  using make_col_ptr_type = MakePtrFunc<col_transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " good merges";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type     _make_ptr     = make_ptr_type();
    make_row_ptr_type _make_row_ptr = make_row_ptr_type();
    make_col_ptr_type _make_col_ptr = make_col_ptr_type();

    row_transform_ptr_type row_transform_ptr(_make_row_ptr(params.seed));
    col_transform_ptr_type col_transform_ptr(_make_col_ptr(params.seed + 1));
    transform_ptr_type     transform_ptr(
        _make_ptr(row_transform_ptr, col_transform_ptr));

    sketch_type sketch_A(transform_ptr);
    sketch_type sketch_B(transform_ptr);
    sketch_type sketch_C(transform_ptr);
    sketch_type sketch_AB(transform_ptr);
    sketch_type sketch_ABC(transform_ptr);

    Eigen::MatrixXf matrix_A = Eigen::MatrixXf::Random(128, 128);
    Eigen::MatrixXf matrix_B = Eigen::MatrixXf::Random(128, 128);
    Eigen::MatrixXf matrix_C = Eigen::MatrixXf::Random(128, 128);

    sketch_both(matrix_A, sketch_A, sketch_AB, sketch_ABC);
    sketch_both(matrix_B, sketch_B, sketch_AB, sketch_ABC);
    sketch_both(matrix_C, sketch_C, sketch_ABC);

    std::size_t size = sketch_A.container().get_registers().size();
    {
      sketch_type merge_AB      = (sketch_A + sketch_B);
      bool        merge_success = merge_AB == sketch_AB;
      if (merge_success == false) {
        std::cout << "fails with mean absolute error "
                  << krowkee::sketch::detail::mean_absolute_error(
                         merge_AB.container().get_registers(),
                         sketch_AB.container().get_registers())
                  << std::endl;
        if (size <= 256) {
          std::cout << "merge_AB:" << std::endl
                    << merge_AB << std::endl
                    << std::endl;
          std::cout << "sketch_AB:" << std::endl
                    << sketch_AB << std::endl
                    << std::endl;
        }
      }
      CHECK_CONDITION(merge_success == true, "merge (+)");
    }
    {
      sketch_type merge_ABC          = (sketch_A + sketch_B + sketch_C);
      bool        multimerge_success = merge_ABC == sketch_ABC;
      if (multimerge_success == false) {
        std::cout << "fails with mean absolute error "
                  << krowkee::sketch::detail::mean_absolute_error(
                         merge_ABC.container().get_registers(),
                         sketch_ABC.container().get_registers())
                  << std::endl;
        if (size <= 256) {
          std::cout << "merge_ABC:" << std::endl
                    << merge_ABC << std::endl
                    << std::endl;
          std::cout << "sketch_ABC:" << std::endl
                    << sketch_ABC << std::endl
                    << std::endl;
        }
      }
      CHECK_CONDITION(multimerge_success == true, "multi-merge (+, +)");
    }
    {
      sketch_A += sketch_B;
      bool inplace_merge_success = sketch_A == sketch_AB;
      if (inplace_merge_success == false) {
        std::cout << "fails with mean absolute error "
                  << krowkee::sketch::detail::mean_absolute_error(
                         sketch_A.container().get_registers(),
                         sketch_AB.container().get_registers())
                  << std::endl;
        if (size <= 256) {
          std::cout << "sketch_A:" << std::endl
                    << sketch_A << std::endl
                    << std::endl;
          std::cout << "sketch_AB:" << std::endl
                    << sketch_AB << std::endl
                    << std::endl;
        }
      }
      CHECK_CONDITION(inplace_merge_success == true, "merge (+)");
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
  using row_transform_type = typename transform_type::row_transform_type;
  using row_transform_ptr_type =
      typename transform_type::row_transform_ptr_type;
  using make_row_ptr_type  = MakePtrFunc<row_transform_type>;
  using col_transform_type = typename transform_type::col_transform_type;
  using col_transform_ptr_type =
      typename transform_type::col_transform_ptr_type;
  using make_col_ptr_type = MakePtrFunc<col_transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " serialize";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type     _make_ptr     = make_ptr_type();
    make_row_ptr_type _make_row_ptr = make_row_ptr_type();
    make_col_ptr_type _make_col_ptr = make_col_ptr_type();

    row_transform_ptr_type row_transform_ptr(_make_row_ptr(params.seed));
    col_transform_ptr_type col_transform_ptr(_make_col_ptr(params.seed + 1));
    transform_ptr_type     transform_ptr(
        _make_ptr(row_transform_ptr, col_transform_ptr));

    CHECK_ALL_ARCHIVES(*transform_ptr, "sketch functor");

    Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(128, 128);
    sketch_type     sketch(transform_ptr);
    sketch_both(matrix, sketch);

    CHECK_ALL_ARCHIVES(sketch.container(), "sketch container");
    CHECK_ALL_ARCHIVES(sketch, "whole sketch object");
  }
};
#endif

template <typename SketchType, template <typename> class MakePtrFunc>
struct ingest_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;
  using row_transform_type = typename transform_type::row_transform_type;
  using row_transform_ptr_type =
      typename transform_type::row_transform_ptr_type;
  using make_row_ptr_type  = MakePtrFunc<row_transform_type>;
  using col_transform_type = typename transform_type::col_transform_type;
  using col_transform_ptr_type =
      typename transform_type::col_transform_ptr_type;
  using make_col_ptr_type = MakePtrFunc<col_transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " ingest";
    return ss.str();
  }

  void rel_mag_test(const transform_ptr_type &transform_ptr,
                    const Parameters         &params) const {
    sketch_type     sketch(transform_ptr);
    Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(128, 128);

    sketch_both(matrix, sketch);

    int    sum(accumulate(sketch, 0.0));
    double rel_mag((double)sum / (1000 * 1000 * params.range_size *
                                  params.replication_count));
    if (params.verbose == true) {
      std::cout << "\t" << sketch << std::endl;
      std::cout << "\tregister sum (should be near zero): " << sum
                << ", relative magnitude: " << rel_mag << std::endl;
    }
    CHECK_CONDITION(rel_mag < 0.1, "register sum relative magnitude near zero");
  }

  void amm_test(const row_transform_ptr_type &transform_ptr,
                const Parameters             &params) const {
    const int row_count = 16;
    const int col_count = 512;

    double success_rate(0.0);
    double empirical_epsilon(0.0);
    double expected_epsilon =
        std::sqrt(16 * std::log(params.observation_count) /
                  (params.range_size * params.range_size));
    int trials(10);
    for (int i(0); i < trials; ++i) {
      Eigen::MatrixXf lhs_dense = Eigen::MatrixXf::Random(row_count, col_count);
      Eigen::MatrixXf rhs_dense = Eigen::MatrixXf::Random(col_count, row_count);

      double lhs_norm = lhs_dense.squaredNorm();
      double rhs_norm = rhs_dense.squaredNorm();

      Eigen::MatrixXf product_dense = lhs_dense * rhs_dense;

      Eigen::MatrixXf lhs_sketch     = sketch_cols(lhs_dense, transform_ptr);
      Eigen::MatrixXf rhs_sketch     = sketch_rows(rhs_dense, transform_ptr);
      Eigen::MatrixXf product_sketch = lhs_sketch * rhs_sketch;

      double sketch_error = (product_dense - product_sketch).squaredNorm();

      double bound      = expected_epsilon * lhs_norm * rhs_norm;
      double this_error = sketch_error / (lhs_norm * rhs_norm);
      empirical_epsilon += this_error;
      if (sketch_error <= bound) {
        success_rate += 1.0;
      }
      if (params.verbose) {
        std::cout << "\tbound " << bound << ", squared error " << sketch_error
                  << " (multiplicative error: " << this_error
                  << ") (in bounds: " << (sketch_error <= bound) << ")"
                  << std::endl;
      }
    }
    success_rate /= trials;
    empirical_epsilon /= trials;
    bool amm_guarantee_success = success_rate > 0.5;
    CHECK_CONDITION(amm_guarantee_success == true, "AMM guarantee (", trials,
                    " trials, ", success_rate,
                    " success rate, expected epsilon=", expected_epsilon,
                    ", mean empirical epsilon=", empirical_epsilon, ")");
  }

  void ammm_test(const row_transform_ptr_type &row_transform_ptr,
                 const col_transform_ptr_type &col_transform_ptr,
                 const transform_ptr_type     &transform_ptr,
                 const Parameters             &params) const {
    const int row_count = 16;
    const int col_count = 512;

    double success_rate(0.0);
    double empirical_epsilon(0.0);
    double expected_epsilon =
        std::sqrt(16 * std::log(params.observation_count) /
                  (params.range_size * params.range_size));
    int trials(10);
    for (int i(0); i < trials; ++i) {
      Eigen::MatrixXf A_dense = Eigen::MatrixXf::Random(row_count, col_count);
      Eigen::MatrixXf B_dense = Eigen::MatrixXf::Random(col_count, col_count);
      Eigen::MatrixXf C_dense = Eigen::MatrixXf::Random(col_count, row_count);

      double A_norm = A_dense.squaredNorm();
      double B_norm = B_dense.squaredNorm();
      double C_norm = C_dense.squaredNorm();

      Eigen::MatrixXf product_dense = A_dense * B_dense * C_dense;

      Eigen::MatrixXf A_sketch = sketch_cols(A_dense, col_transform_ptr);
      sketch_type     sketch(transform_ptr);
      sketch_both(B_dense, sketch);
      Eigen::MatrixXf B_sketch       = sketch.container().get_registers();
      Eigen::MatrixXf C_sketch       = sketch_rows(C_dense, row_transform_ptr);
      Eigen::MatrixXf product_sketch = A_sketch * B_sketch * C_sketch;

      double sketch_error = (product_dense - product_sketch).squaredNorm();

      double bound      = expected_epsilon * A_norm * B_norm * C_norm;
      double this_error = sketch_error / (A_norm * B_norm * C_norm);
      empirical_epsilon += this_error;
      if (sketch_error <= bound) {
        success_rate += 1.0;
      }
      if (params.verbose) {
        std::cout << "\tbound " << bound << ", squared error " << sketch_error
                  << " (multiplicative error: " << this_error
                  << ") (in bounds: " << (sketch_error <= bound) << ")"
                  << std::endl;
      }
    }
    success_rate /= trials;
    empirical_epsilon /= trials;
    bool ammm_guarantee_success = success_rate > 0.5;
    CHECK_CONDITION(ammm_guarantee_success == true, "AMM guarantee (", trials,
                    " trials, ", success_rate,
                    " success rate, expected epsilon=", expected_epsilon,
                    ", mean empirical epsilon=", empirical_epsilon, ")");
  }

  void operator()(const Parameters &params) const {
    make_ptr_type     _make_ptr     = make_ptr_type();
    make_row_ptr_type _make_row_ptr = make_row_ptr_type();
    make_col_ptr_type _make_col_ptr = make_col_ptr_type();

    row_transform_ptr_type row_transform_ptr(_make_row_ptr(params.seed));
    col_transform_ptr_type col_transform_ptr(_make_col_ptr(params.seed + 1));
    transform_ptr_type     transform_ptr(
        _make_ptr(row_transform_ptr, col_transform_ptr));

    rel_mag_test(transform_ptr, params);
    amm_test(row_transform_ptr, params);
    ammm_test(row_transform_ptr, col_transform_ptr, transform_ptr, params);
  }
};

template <typename SketchType, template <typename> class MakePtrFunc>
struct spot_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;
  using row_transform_type = typename transform_type::row_transform_type;
  using row_transform_ptr_type =
      typename transform_type::row_transform_ptr_type;
  using make_row_ptr_type  = MakePtrFunc<row_transform_type>;
  using col_transform_type = typename transform_type::col_transform_type;
  using col_transform_ptr_type =
      typename transform_type::col_transform_ptr_type;
  using make_col_ptr_type = MakePtrFunc<col_transform_type>;
  using update_type       = typename row_transform_type::update_type;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " spot check";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type     _make_ptr     = make_ptr_type();
    make_row_ptr_type _make_row_ptr = make_row_ptr_type();
    make_col_ptr_type _make_col_ptr = make_col_ptr_type();

    row_transform_ptr_type row_transform_ptr(_make_row_ptr(params.seed));
    col_transform_ptr_type col_transform_ptr(_make_col_ptr(params.seed + 1));
    transform_ptr_type     transform_ptr(
        _make_ptr(row_transform_ptr, col_transform_ptr));

    {
      std::pair<std::uint64_t, std::uint64_t> indices{40, 22};
      krowkee::stream::Element<register_type> element(indices);
      CHECK_CONDITION(indices.first == element.item,
                      "stream element gets row idx");
      CHECK_CONDITION(indices.second == element.identifier,
                      "stream element gets col idx");

      sketch_type sketch(transform_ptr);
      bool        is_empty = sketch.empty();
      if (is_empty == false) {
        std::cout << "sketch:" << std::endl
                  << sketch.container() << std::endl
                  << std::endl;
      }
      CHECK_CONDITION(is_empty == true, "sketch is initialized empty");

      sketch.insert(indices);
      update_type       row_hashes = row_transform_ptr->apply(indices.first);
      update_type       col_hashes = col_transform_ptr->apply(indices.second);
      bool              correct_updates(true);
      std::stringstream ss;
      ss << "updates: ";
      std::stringstream fail_ss;
      for (int i(0); i < row_hashes.first.size(); ++i) {
        ss << "(" << row_hashes.first[i] << "," << col_hashes.first[i]
           << ") <- " << "(" << row_hashes.second[i] << ","
           << col_hashes.second[i] << "), ";
        for (int j(0); j < col_hashes.first.size(); ++j) {
          const auto row_idx      = row_hashes.first[i];
          const auto col_idx      = col_hashes.first[j];
          const auto row_polarity = row_hashes.second[i];
          const auto col_polarity = col_hashes.second[j];
          auto       sketch_val(sketch.container()(row_idx, col_idx));
          auto       explicit_val(row_polarity * col_polarity);
          if (sketch_val != explicit_val) {
            correct_updates = false;
            fail_ss << "failed on (" << row_idx << "," << col_idx << "), "
                    << sketch_val << " != " << explicit_val << std::endl;
          };
        }
      }
      ss << std::endl;
      if (correct_updates == false) {
        std::cout << fail_ss.str();
        std::cout << ss.str() << std::endl;
        if (sketch.container().get_registers().size() <= 256) {
          std::cout << "sketch:" << std::endl
                    << sketch.container() << std::endl
                    << std::endl;
        }
      }
      CHECK_CONDITION(correct_updates == true, "single update matches");
      {
        Eigen::MatrixXf matrix_A                = Eigen::MatrixXf::Zero(64, 64);
        matrix_A(indices.first, indices.second) = 1;
        Eigen::MatrixXf matrix_AR   = sketch_cols(matrix_A, col_transform_ptr);
        Eigen::MatrixXf matrix_STAR = sketch_rows(matrix_AR, row_transform_ptr);

        bool allclose = krowkee::sketch::detail::allclose(
            matrix_STAR, sketch.container().get_registers());
        std::size_t size = matrix_STAR.size();
        if (allclose == false) {
          std::cout << "fails with mean absolute error "
                    << krowkee::sketch::detail::mean_absolute_error(
                           matrix_STAR, sketch.container().get_registers())
                    << std::endl;
          std::cout << ss.str() << std::endl;
          if (size <= 256) {
            std::cout << "matrix_STAR:" << std::endl
                      << matrix_STAR << std::endl
                      << std::endl;
            std::cout << "sketch:" << std::endl
                      << sketch << std::endl
                      << std::endl;
          }
        }
        CHECK_CONDITION(allclose == true,
                        "single update matches dense version");
      }
    }
    {
      Eigen::MatrixXf matrix_A  = Eigen::MatrixXf::Random(64, 64);
      Eigen::MatrixXf matrix_AR = sketch_cols(matrix_A, col_transform_ptr);
      // std::cout << "matrix_AR" << std::endl
      //           << matrix_AR << std::endl
      //           << std::endl;
      Eigen::MatrixXf matrix_STAR = sketch_rows(matrix_AR, row_transform_ptr);
      Eigen::MatrixXf matrix_STA  = sketch_rows(matrix_A, row_transform_ptr);
      // std::cout << "matrix_STA" << std::endl
      //           << matrix_STA << std::endl
      //           << std::endl;
      Eigen::MatrixXf matrix_STAR2 = sketch_cols(matrix_STA, col_transform_ptr);
      // std::cout << "matrix_STAR2" << std::endl
      //           << matrix_STAR2 << std::endl
      //           << std::endl;

      sketch_type sketch(transform_ptr);
      sketch_both(matrix_A, sketch);

      bool dense_allclose =
          krowkee::sketch::detail::allclose(matrix_STAR, matrix_STAR2);
      CHECK_CONDITION(dense_allclose == true, "both dense version match");

      bool allclose = krowkee::sketch::detail::allclose(
          matrix_STAR, sketch.container().get_registers());
      std::size_t size = matrix_STAR.size();
      if (allclose == false) {
        std::cout << "fails with mean absolute error "
                  << krowkee::sketch::detail::mean_absolute_error(
                         matrix_STAR, sketch.container().get_registers())
                  << std::endl;
        if (size <= 256) {
          std::cout << "matrix_STAR:" << std::endl
                    << matrix_STAR << std::endl
                    << std::endl;
          std::cout << "sketch:" << std::endl
                    << sketch << std::endl
                    << std::endl;
        }
      }
      CHECK_CONDITION(allclose == true, "macro transform composition matches");
    }
  }
};

/**
 * Execute the batter of tests for the given sketch functor.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
void perform_tests(const Parameters &params) {
  using sketch_type = SketchType;

  MakePtrFunc<std::int32_t> mpf;

  print_line();
  print_line();
  std::cout << "Testing " << sketch_type::full_name() << std::endl;
  std::cout << "\tUsing " << mpf.name() << std::endl;
  print_line();
  print_line();

  std::cout << std::endl << std::endl;

  do_test<init_check<sketch_type, MakePtrFunc>>(params);
  do_test<spot_check<sketch_type, MakePtrFunc>>(params);
  do_test<bad_merge_check<sketch_type, MakePtrFunc>>(params);
  do_test<good_merge_check<sketch_type, MakePtrFunc>>(params);
  do_test<ingest_check<sketch_type, MakePtrFunc>>(params);
#if __has_include(<cereal/cereal.hpp>)
  do_test<serialize_check<sketch_type, MakePtrFunc>>(params);
#endif
}

void print_help(char *exe_name) {
  std::cout << "\nusage:  " << exe_name << "\n"
            << "\t-c, --count <int>              - number of insertions\n"
            << "\t-r, --range <int>              - range of sketch transform\n"
            << "\t-R, --replication <int>        - number of tiled sketch "
               "transforms\n"
            << "\t-d, --domain <int>             - domain of sketch transform\n"
            << "\t-b, --observation_count <int>  - number of sketches to test\n"
            << "\t-s, --seed <int>               - random seed\n"
            << "\t-v, --verbose                  - print additional debug "
               "information.\n"
            << "\t-h, --help                     - print this line and exit\n"
            << std::endl;
}

void parse_args(int argc, char **argv, Parameters &params) {
  int c;

  while (1) {
    int                  option_index(0);
    static struct option long_options[] = {
        {"count", required_argument, NULL, 'c'},
        {"range", required_argument, NULL, 'r'},
        {"replication", required_argument, NULL, 'R'},
        {"domain", required_argument, NULL, 'd'},
        {"observation-count", required_argument, NULL, 'b'},
        {"seed", required_argument, NULL, 's'},
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}};

    int curind = optind;
    c          = getopt_long(argc, argv, "-:c:r:R:d:b:s:vh", long_options,
                             &option_index);
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
        params.count = std::atol(optarg);
        break;
      case 'r':
        params.range_size = std::atoll(optarg);
        break;
      case 'R':
        params.replication_count = std::atoll(optarg);
        break;
      case 'd':
        params.domain_size = std::atoll(optarg);
        break;
      case 'b':
        params.observation_count = std::atoll(optarg);
        break;
      case 's':
        params.seed = std::atol(optarg);
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
        break;
    }
  }
}

template <std::size_t RangeSize, std::size_t ReplicationCount>
struct do_all_tests {
  void operator()(const Parameters &params) {
    perform_tests<matrix::DoubleSparseJLT<RangeSize, ReplicationCount>,
                  make_ptr_functor>(params);
  }
};

int main(int argc, char **argv) {
  uint64_t      count(1000);
  std::uint64_t range_size(32);
  std::uint64_t replication_count(4);
  std::uint64_t domain_size(4096);
  std::uint64_t observation_count(16);
  std::uint64_t seed(krowkee::hash::default_seed);
  bool          verbose(false);

  Parameters params{count,       range_size,        replication_count,
                    domain_size, observation_count, seed,
                    verbose};

  parse_args(argc, argv, params);

  dispatch_with_sketch_sizes<do_all_tests, void>(
      params.range_size, params.replication_count, params);
  return 0;
}