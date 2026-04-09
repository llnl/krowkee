// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
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
  constexpr const std::size_t row_count(256);
  constexpr const std::size_t col_count(row_count);
  constexpr const std::size_t transform_count(4);
  std::uint64_t               seed(4);
  bool                        verbose(true);

  // Using krowkee requires the selection of a sketch type for both a single
  // and double-sided sketch, here encapsulated as `sketch_type` and
  // `double_sketch_type`, respectively. We use the `sketch::SparseJLT` and
  // `sketch::DoubleSparseJLT` types defined in the simple API in
  // `krowkee/sketch.hpp`. These types have four shared template parameters:
  //   1. the numeric type to be used by each register (here `double`),
  //   2. a `std::size_t` parameter `range_size` indicating the number of
  //      registers used by each instance of the internal row transform,
  //   3. a `std::size_t` parameter `replication_count` indicating the number
  //      of instances of the row transforms to be used, and
  //   4. a shared pointer type to be used by the shared transform object
  //      (`std::shared_ptr` for shared memory implementations).
  // `sketch::DoubleSparseJLT` has an additional two template parameters
  // controlling the column transform. If not provided these default to the
  // values provided for the row transform.
  //   5. a `std::size_t` parameter `range_size` indicating the number of
  //      registers used by each instance of the internal column transform,
  //   6. a `std::size_t` parameter `replication_count` indicating the number
  //      of instances of the column transforms to be used, and

  constexpr const std::size_t range_size        = 128;
  constexpr const std::size_t replication_count = 4;
  constexpr const std::size_t embedding_size = range_size * replication_count;
  using register_type                        = double;
  using sketch_type =
      krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                                 std::shared_ptr>;
  using double_sketch_type =
      krowkee::sketch::DoubleSparseJLT<register_type, range_size,
                                       replication_count, std::shared_ptr>;

  // Here we create the type of the final two-sided transform, which, unlike
  // `final_double_sketch_type`, is not square. This projects the final
  // embedding down into a smaller dimension, which allows the storage of
  // higher-precision inter-matrix sketches while producing a smaller final
  // embedding.
  constexpr const std::size_t final_range_size        = 128;
  constexpr const std::size_t final_replication_count = 4;
  using final_double_sketch_type = krowkee::sketch::DoubleSparseJLT<
      register_type, range_size, replication_count, std::shared_ptr,
      final_range_size, final_replication_count>;

  // Having established our sketch types, we must now create shared pointers
  // to all of the associated sketch transforms. Each doubled transform is
  // multiplied together with its neighbor in the form $AS S^TARR^TAQ$, for
  // sketch transforms `S`, `R`, and `Q` and input matrix `A`. The sketch
  // types includes typedefs of the transform and pointer types. This is
  // where the random seed is used. Transforms of the same type sharing the
  // same seed will behave identically. As this is a distributed memory
  // code, we create a `std::shared_ptr` of the transform to be used to
  // define the sketch data structures on each rank, ensuring that each uses
  // the same transform.
  using single_transform_type     = typename sketch_type::transform_type;
  using single_transform_ptr_type = typename sketch_type::transform_ptr_type;
  using double_transform_type     = typename double_sketch_type::transform_type;
  using double_transform_ptr_type =
      typename double_sketch_type::transform_ptr_type;
  using final_double_transform_type =
      typename final_double_sketch_type::transform_type;
  using final_double_transform_ptr_type =
      typename final_double_sketch_type::transform_ptr_type;
  //  We also get the type `final_single_sketch_type` of the corresponding final
  //  column sketch.
  using final_single_transform_type =
      typename final_double_transform_type::col_transform_type;
  using final_single_transform_ptr_type =
      typename final_double_transform_type::col_transform_ptr_type;
  // We verify that we did not make a mistake above, and both sketch types use
  // the same transform type.
  static_assert(
      std::is_same<single_transform_type,
                   typename double_transform_type::row_transform_type>::value);
  static_assert(
      std::is_same<single_transform_type,
                   typename double_transform_type::col_transform_type>::value);
  static_assert(
      std::is_same<
          single_transform_type,
          typename final_double_transform_type::row_transform_type>::value);
  // Note that we do not try to match the final column transform, as it is
  // allowed to have different parameters.
  static_assert(final_double_transform_type::col_transform_type::range_size() ==
                final_range_size);
  static_assert(
      final_double_transform_type::col_transform_type::replication_count() ==
      final_replication_count);

  // We create these counters for convenience. Note that `transform_count` is
  // the total number of single sketch transforms involved in computing the
  // product, matching the exponent approximately applied to $A$. So, there are
  // `transform_count - 1` single transforms of the inserted type and one of the
  // final type, and `transform_count - 2` double transforms of the intermediate
  // type and one of the final, possibly rectangular, type.
  constexpr const std::size_t single_transform_count = transform_count - 1;
  constexpr const std::size_t double_transform_count = transform_count - 2;

  // We create a vector of shared pointers for each of the individual
  // sketch transforms.
  std::vector<single_transform_ptr_type> single_transform_ptrs;
  for (int i(0); i < single_transform_count; ++i) {
    single_transform_ptrs.push_back(
        std::make_shared<single_transform_type>(seed + i));
  }
  // Using these shared pointers, we now create a vector of pointers to all of
  // the two-sided sketch transforms.
  std::vector<double_transform_ptr_type> double_transform_ptrs;
  for (int i(0); i < double_transform_count; ++i) {
    double_transform_ptrs.push_back(std::make_shared<double_transform_type>(
        single_transform_ptrs[i], single_transform_ptrs[i + 1]));
  }

  // We create pointers to the final column transform and double transform
  // pointers.
  final_single_transform_ptr_type final_single_transform(
      std::make_shared<final_single_transform_type>(seed + transform_count));
  final_double_transform_ptr_type final_double_transform(
      std::make_shared<final_double_transform_type>(
          single_transform_ptrs.back(), final_single_transform));

  // We create a `std::vector` that will hold the AS embeddings for each row of
  // A. Note that the transform is conforming with the left-hand transform of
  // the first double transform, so resulting matrices can be multiplied
  // together.
  std::vector<sketch_type> row_sketches;
  for (int i(0); i < row_count; ++i) {
    row_sketches.emplace_back(single_transform_ptrs[0]);
  }

  // We also create a vector of local double-sided sketches that will hold the
  // double sided embeddings.
  std::vector<double_sketch_type> double_sketches;
  for (const double_transform_ptr_type &double_transform_ptr :
       double_transform_ptrs) {
    double_sketches.emplace_back(double_transform_ptr);
  }

  // We also create the final double-sided sketch.
  final_double_sketch_type final_double_sketch(final_double_transform);

  // We sample a random matrix to embed. Note that this is not implemented
  // efficiently, as this is a toy example and we will compute a ground
  // truth solution that is intractable for large matrices.
  srand(seed);
  static Eigen::MatrixXd matrix_A =
      Eigen::MatrixXd::Random(row_count, col_count);

  // We apply both the single and double sketches to each element of `matrix_A`
  // in a single pass.
  int i(0);
  for (const auto &row : matrix_A.rowwise()) {
    int j(0);
    for (const auto &element : row) {
      // insert `(j, element)` into the `i`th row sketch of `AS`
      row_sketches[i].insert(j, element);
      for (double_sketch_type &double_sketch : double_sketches) {
        // insert `((i, j), element)` into each double sketch of the form
        // `S^tAR`
        double_sketch.insert({i, j}, element);
      }
      // insert `((i, j), element)` into the final double sketch
      final_double_sketch.insert({i, j}, element);
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

  // We dump the contents of the S^tAR embeddings to Eigen matrices.
  std::vector<Eigen::MatrixXd> double_matrices;
  for (const double_sketch_type &double_sketch : double_sketches) {
    double_matrices.push_back(double_sketch.scaled_registers());
  }

  // We dump the contents of the final S^tAR embedding to an Eigen Matrix.
  Eigen::MatrixXd final_double_matrix = final_double_sketch.scaled_registers();

  // We compute the streaming power iteration product.
  Eigen::MatrixXd product_streaming = matrix_AS;
  for (int i(0); i < double_matrices.size(); ++i) {
    product_streaming *= double_matrices[i];
  }
  product_streaming *= final_double_matrix;

  // We compute the exact power iteration product.
  Eigen::MatrixXd product_exact = matrix_A;
  for (int i(0); i < transform_count - 1; ++i) {
    product_exact *= matrix_A;
  }
  if (verbose) {
    std::cout << "A(5,7) = " << matrix_A(5, 7) << std::endl;
    std::cout << "A^" << transform_count << "(5,7) = " << product_exact(5, 7)
              << std::endl;
  }

  // We now compare the embedding vectors. In practice this could be done more
  // efficiently, but this implementation suffices for illustration.
  double success_rate_streaming(0.0);
  double epsilon_streaming(0.0);
  // We use the stable rank to compute the expected approximation bound epsilon
  // from the embedding size
  // :math:`\sqrt{2} \left ( (1 + 2\varepsilon)^{r - 1} \right ) \|A\|^r_{op}`.
  const double srank                      = stable_rank(matrix_A);
  const double epsilon_expected_streaming = std::sqrt(
      16 *
      (srank + std::log((transform_count - 1) *
                        sketch_type::transform_type::replication_count())) /
      range_size);
  int trials(0);

  for (int i(0); i < row_count; ++i) {
    for (int j(i + 1); j < row_count; ++j) {
      ++trials;
      // compute exact distance between power iteration rows
      double dist_exact =
          (product_exact.row(i) - product_exact.row(j)).lpNorm<2>();
      // compute distance between streaming embedding rows
      double dist_streaming =
          (product_streaming.row(i) - product_streaming.row(j)).lpNorm<2>();
      double error_streaming = std::abs(1.0 - dist_streaming / dist_exact);
      epsilon_streaming += error_streaming;
      if (in_bounds(dist_exact, dist_streaming, epsilon_expected_streaming)) {
        success_rate_streaming += 1.0;
      }
      if (i == 199 && j == 230) {
        std::cout << "\t(" << i << "," << j << ") exact " << dist_exact
                  << "\n\t\tstreaming (dist/error/success): (" << dist_streaming
                  << ", 1 +/- " << error_streaming << ", "
                  << in_bounds(dist_exact, dist_streaming,
                               epsilon_expected_streaming)
                  << std::endl;
      }
    }
  }

  success_rate_streaming /= trials;
  epsilon_streaming /= trials;

  std::cout << "\npower iteration approximate row distances guarantee ("
            << trials << " trials)" << std::endl;
  std::cout << "\tstreaming success rate / epsilon / expected = ("
            << success_rate_streaming << ", " << epsilon_streaming << ","
            << epsilon_expected_streaming << ")" << std::endl;

  return 0;
}