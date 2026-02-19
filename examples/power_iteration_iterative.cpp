// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <krowkee/hash/hash.hpp>
#include <krowkee/sketch.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <random>
#include <type_traits>

double spectral_norm(const Eigen::MatrixXd &matrix) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.singularValues()(0);
}

double stable_rank(const Eigen::MatrixXd &matrix) {
  return matrix.squaredNorm() / std::pow(spectral_norm(matrix), 2);
}

bool in_bounds(const double tru, const double est, const double eps) {
  return (est < (1 + eps) * tru) && (est > (1 - eps) * tru);
}

int main(int argc, char **argv) {
  const std::size_t row_count = 256;
  const std::size_t col_count = row_count;
  const std::size_t exponent  = 4;
  std::uint64_t     seed      = 4;
  bool              verbose   = true;

  // Using krowkee requires the selection of a sketch type, here encapsulated as
  // `sketch_type`. We use the `SparseJLT` type defined in the simple API in
  // `krowkee/sketch.hpp`. This type has four template parameters:
  //   1. the numeric type to be used by each register (here `double`),
  //   2. a compile-time `std::size_t` parameter `range_size` indicating the
  //      number of registers used by each instance of the internal transform,
  //   3. a compile-time `std::size_t` parameter `replication_count` indicating
  //      the number of instances of the transform to be used, and
  //   4. a shared pointer type to be used by the shared transform object
  //      (`std::shared_ptr` for shared memory implementations).
  constexpr const std::size_t range_size        = 128;
  constexpr const std::size_t replication_count = 4;
  constexpr const std::size_t embedding_size = range_size * replication_count;
  using register_type                        = double;
  using sketch_type =
      krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                                 std::shared_ptr>;

  // Having established our sketch type, we must now create a share pointer
  // to the associated sketch transform. The sketch types includes typedefs of
  // the transform and pointer types. This is where the random seed is used.
  // Transforms of the same type sharing the same seed will behave identically.
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  // We create a shared pointer for the sketch transform.
  transform_ptr_type transform_ptr = std::make_shared<transform_type>(seed);

  // We create a `std::vector` that will hold the AS embeddings for each row of
  // A.
  std::vector<sketch_type> row_sketches;
  for (int i(0); i < row_count; ++i) {
    row_sketches.emplace_back(transform_ptr);
  }

  // We sample a random matrix to embed. Note that this is not implemented
  // efficiently, as this is a toy example and we will compute a ground
  // truth solution that is intractable for large matrices.
  srand(seed);
  static Eigen::MatrixXd matrix_A =
      Eigen::MatrixXd::Random(row_count, col_count);

  // We apply the sketch to each element of `matrix_A` in a single pass.
  int i(0);
  for (const auto &row : matrix_A.rowwise()) {
    int j(0);
    for (const auto &element : row) {
      // insert `(j, element)` into the `i`th row sketch of `AS`
      row_sketches[i].insert(j, element);
      ++j;
    }
    ++i;
  }

  // We dump the contents of the AS embedding to an Eigen matrix.
  Eigen::MatrixXd matrix_AS = Eigen::MatrixXd::Zero(row_count, embedding_size);
  for (int i(0); i < row_count; ++i) {
    std::vector<register_type> embedding = row_sketches[i].scaled_registers();
    for (int j(0); j < embedding_size; ++j) {
      matrix_AS(i, j) = embedding[j];
    }
  }

  // We compute the iterative power iteration product.
  Eigen::MatrixXd product_iterative = matrix_A;
  for (int i(2); i < exponent; ++i) {
    product_iterative *= matrix_A;
  }
  product_iterative *= matrix_AS;

  // We compute the exact power iteration product.
  Eigen::MatrixXd product_exact = matrix_A;
  for (int i(1); i < exponent; ++i) {
    product_exact *= matrix_A;
  }

  if (verbose) {
    std::cout << "A(5,7) = " << matrix_A(5, 7) << std::endl;
    std::cout << "A^" << exponent << "(5,7) = " << product_exact(5, 7)
              << std::endl;
  }

  // We now compare the embedding vectors. In practice this could be done more
  // efficiently, but this implementation suffices for illustration.
  double success_rate_iterative(0.0);
  double epsilon_iterative(0.0);
  // We compute the expected value of the approximation bound epsilon given the
  // size of the problem and the embedding size.
  double epsilon_expected_iterative =
      std::sqrt(16 * std::log(row_count) /
                (transform_type::range_size() * transform_type::range_size()));

  int trials(0);

  for (int i(0); i < row_count; ++i) {
    for (int j(i + 1); j < row_count; ++j) {
      ++trials;
      // compute exact distance between power iteration rows
      double dist_exact =
          (product_exact.row(i) - product_exact.row(j)).lpNorm<2>();
      // compute distance between iterative embedding rows
      double dist_iterative =
          (product_iterative.row(i) - product_iterative.row(j)).lpNorm<2>();
      double error_iterative = std::abs(1.0 - dist_iterative / dist_exact);
      epsilon_iterative += error_iterative;
      if (in_bounds(dist_exact, dist_iterative, epsilon_expected_iterative)) {
        success_rate_iterative += 1.0;
      }
      if (verbose && i == 199 && j == 230) {
        std::cout << "\t(" << i << "," << j << ") exact " << dist_exact
                  << ")\n\t\titerative (dist/error/success): ("
                  << dist_iterative << ", 1 +/- " << error_iterative << ", "
                  << in_bounds(dist_exact, dist_iterative,
                               epsilon_expected_iterative)
                  << ")" << std::endl;
      }
    }
  }

  success_rate_iterative /= trials;
  epsilon_iterative /= trials;

  std::cout << "\npower iteration approximate row distances guarantee ("
            << trials << " trials)" << std::endl;
  std::cout << "\titerative success rate / epsilon / expected = ("
            << success_rate_iterative << ", " << epsilon_iterative << ","
            << epsilon_expected_iterative << ")" << std::endl
            << std::endl;

  return 0;
}