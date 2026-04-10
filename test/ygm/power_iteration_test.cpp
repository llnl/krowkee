// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <krowkee/hash/hash.hpp>
#include <krowkee/sketch.hpp>
#include <krowkee/util/runtime.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>

#include <Eigen/Dense>

#include <getopt.h>
#include <stdio.h>
#include <unistd.h>

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

template <typename T>
bool values_close(const T &lhs, const T &rhs, const double rtol = 1e-5,
                  const double atol = 1e-8) {
  return std::abs(lhs - rhs) <=
         (atol + rtol * std::max(std::abs(lhs), std::abs(rhs)));
}

template <typename T>
bool ranks_close(const T &value, ygm::comm &comm, const double rtol = 1e-5,
                 const double atol = 1e-8) {
  T pos_min = ygm::min(value, comm);
  T neg_min = ygm::min(-value, comm);
  return std::abs(pos_min - neg_min) <=
         (atol + rtol * std::max(std::abs(pos_min), std::abs(neg_min)));
}

/**
 * Struct bundling the experiment parameters.
 */
struct Parameters {
  std::uint64_t count;
  std::uint64_t range_size;
  std::uint64_t replication_count;
  std::uint64_t transform_count;
  std::uint64_t seed;
  bool          verbose;
};

template <typename SingleSketchType>
Eigen::MatrixXd serial_accumulate_AS(
    const Eigen::MatrixXd                         &matrix_A,
    typename SingleSketchType::transform_ptr_type &single_transform_ptr) {
  // Initialize a zeros matrix
  Eigen::MatrixXd serial_matrix_AS = Eigen::MatrixXd::Zero(
      matrix_A.rows(), SingleSketchType::transform_type::size());

  // We create a `std::map` that will hold the AS embeddings for each row of
  // A. Initialize each such row to be an empty sketch using the zeroth
  // transform.
  std::vector<SingleSketchType> serial_single_sketches;
  for (int i(0); i < matrix_A.rows(); ++i) {
    serial_single_sketches.emplace_back(single_transform_ptr);
  }

  // We apply the single-sided sketches to each element of `matrix_A` in a
  // single pass.
  int i(0);
  for (const auto &row : matrix_A.rowwise()) {
    int j(0);
    for (const auto &element : row) {
      // insert `(j, element)` into the `i`th row sketch of `AS`
      serial_single_sketches[i].insert(j, element);
      ++j;
    }
    ++i;
  }

  // We dump the contents of the AS embedding to an Eigen matrix.
  for (int i(0); i < serial_single_sketches.size(); ++i) {
    auto embedding = serial_single_sketches[i].scaled_registers();
    for (int j(0); j < embedding.size(); ++j) {
      serial_matrix_AS(i, j) = embedding[j];
    }
  }

  return serial_matrix_AS;
}

template <typename SingleSketchType, typename DoubleSketchType>
ygm::container::map<int, Eigen::VectorXd> parallel_accumulate_AS(
    ygm::comm &comm, const Eigen::MatrixXd &matrix_A,
    typename SingleSketchType::transform_ptr_type &single_transform_ptr) {
  using single_sketch_type = SingleSketchType;

  // Eigen::VectorXd default_embedding = Eigen::VectorXd::Zero(matrix_A.cols());
  ygm::container::map<int, Eigen::VectorXd> parallel_matrix_AS(comm);

  // We create a ygm::map that will hold the sketches for each row of A.
  single_sketch_type default_sketch(single_transform_ptr);
  ygm::container::map<int, single_sketch_type> single_sketches(comm,
                                                               default_sketch);

  for (int col_idx(0); col_idx < matrix_A.cols(); ++col_idx) {
    if (comm.rank() == (col_idx % comm.size())) {
      for (int row_idx(0); row_idx < matrix_A.rows(); ++row_idx) {
        auto insert_lambda = [](const int row_idx, single_sketch_type &sketch,
                                const int col_idx, const double update) {
          sketch.insert(col_idx, update);
        };
        single_sketches.async_visit(row_idx, insert_lambda, col_idx,
                                    matrix_A(row_idx, col_idx));
      }
    }
  }
  comm.barrier();

  single_sketches.for_all(
      [&parallel_matrix_AS](const int idx, const single_sketch_type &sketch) {
        Eigen::VectorXd embedding(sketch.size());
        auto            scaled_registers = sketch.scaled_registers();
        for (int i(0); i < scaled_registers.size(); ++i) {
          embedding(static_cast<Eigen::Index>(i)) =
              static_cast<double>(scaled_registers[i]);
        }
        parallel_matrix_AS.async_insert(idx, embedding);
      });
  comm.barrier();

  return parallel_matrix_AS;
}

void agreement_parallel_matrix(
    const std::string &&name, const Eigen::MatrixXd &serial_matrix_AS,
    const ygm::container::map<int, Eigen::VectorXd> &parallel_matrix_AS,
    const Parameters                                &params) {
  ygm::comm &comm  = parallel_matrix_AS.comm();
  bool       match = true;
  double     mae(0.0);
  double     max_error(0.0);
  parallel_matrix_AS.for_all([&serial_matrix_AS, &match, &mae, &max_error,
                              &params](const int              idx,
                                       const Eigen::VectorXd &lhs) {
    const auto &rhs = serial_matrix_AS(idx, Eigen::all);
    if (params.verbose && lhs.size() <= 32 && idx == 199) {
      std::cout << "\tparallel embedding: " << std::endl;
      std::cout << lhs << std::endl;
      std::cout << "\tserial embedding: " << std::endl;
      std::cout << rhs << std::endl;
    }
    double this_mae = krowkee::sketch::detail::mean_absolute_error(lhs, rhs);
    double this_max_error =
        krowkee::sketch::detail::max_absolute_error(lhs, rhs);
    if (this_mae >= 1.e-5) {
      match = false;
    }
    mae += this_mae;
    max_error = std::max(this_max_error, max_error);
  });
  comm.barrier();

  match     = ygm::min(match, comm);
  mae       = ygm::sum(mae, comm) / serial_matrix_AS.rows();
  max_error = ygm::max(max_error, comm);

  bool mae_thresh       = values_close(mae, 0.0);
  bool max_error_thresh = values_close(max_error, 0.0);
  if (match == false || mae_thresh == false || max_error_thresh == false) {
    comm.cout0("\t", name, " match? ", match, ", mae: ", mae,
               ", max_error: ", max_error);
    comm.cout0("");
  }
  CHECK_CONDITION(comm, match == true, name + " ranks agree");
  CHECK_CONDITION(
      comm, mae_thresh == true,
      name + " low mean absolute error (" + std::to_string(mae) + ")");
  CHECK_CONDITION(
      comm, max_error_thresh == true,
      name + " low maximum absolute error (" + std::to_string(max_error) + ")");
}

template <typename DoubleSketchType, typename FinalSketchType>
std::pair<std::vector<Eigen::MatrixXd>, Eigen::MatrixXd>
serial_accumulate_double_matrices(
    const Eigen::MatrixXd &matrix_A,
    std::vector<typename DoubleSketchType::transform_ptr_type>
                                                 &double_transform_ptrs,
    typename FinalSketchType::transform_ptr_type &final_transform_ptr) {
  // We create the double matrices array
  std::vector<Eigen::MatrixXd> serial_double_matrices;

  // We create a vector of local double-sided sketches that will hold the double
  // sided embeddings.
  std::vector<DoubleSketchType> serial_double_sketches;
  for (const auto &double_transform_ptr : double_transform_ptrs) {
    serial_double_sketches.emplace_back(double_transform_ptr);
  }
  FinalSketchType serial_final_sketch(final_transform_ptr);

  // We apply the double sketches to each element of `matrix_A` in a single
  // pass.
  int i(0);
  for (const auto &row : matrix_A.rowwise()) {
    int j(0);
    for (const auto &element : row) {
      for (DoubleSketchType &double_sketch : serial_double_sketches) {
        // insert `((i, j), element)` into each double sketch of the form
        // `S^tAR`
        double_sketch.insert({i, j}, element);
      }
      serial_final_sketch.insert({i, j}, element);
      ++j;
    }
    ++i;
  }

  // We dump the contents of the S^tAR embeddings to Eigen matrices.
  for (const DoubleSketchType &double_sketch : serial_double_sketches) {
    serial_double_matrices.push_back(double_sketch.scaled_registers());
  }
  Eigen::MatrixXd serial_final_matrix = serial_final_sketch.scaled_registers();

  return {serial_double_matrices, serial_final_matrix};
}

template <typename DoubleSketchType, typename FinalSketchType>
std::pair<std::vector<Eigen::MatrixXd>, Eigen::MatrixXd>
parallel_accumulate_double_matrices(
    const Eigen::MatrixXd                     &matrix_A,
    ygm::container::map<int, Eigen::VectorXd> &parallel_matrix_AS,
    std::vector<typename DoubleSketchType::transform_ptr_type>
                                                 &double_transform_ptrs,
    typename FinalSketchType::transform_ptr_type &final_transform_ptr) {
  using double_sketch_type = DoubleSketchType;
  using double_transform_ptr_type =
      typename DoubleSketchType::transform_ptr_type;
  using final_sketch_type    = FinalSketchType;
  using final_transform_type = typename final_sketch_type::transform_type;
  using final_transform_ptr_type =
      typename final_sketch_type::transform_ptr_type;

  ygm::comm &comm = parallel_matrix_AS.comm();

  // We also create local double-sided sketches that will hold the double
  // sided embeddings.
  static std::vector<double_sketch_type> parallel_double_sketches;
  for (const double_transform_ptr_type &double_transform_ptr :
       double_transform_ptrs) {
    parallel_double_sketches.emplace_back(double_transform_ptr);
  }
  static final_sketch_type parallel_final_sketch(final_transform_ptr);

  // We apply the double sketches to each element of `matrix_A` in a single
  // pass. In practice this could be interleaved with the accumulation of
  // AS.
  for (int col_idx(0); col_idx < matrix_A.cols(); ++col_idx) {
    if (comm.rank() == (col_idx % comm.size())) {
      for (int row_idx(0); row_idx < matrix_A.rows(); ++row_idx) {
        auto insert_lambda = [](const int              row_idx,
                                const Eigen::VectorXd &payload,
                                const int col_idx, const double update) {
          // insert `(row_idx, col_idx) <- update' into all matrix
          // sketches.
          for (double_sketch_type &double_sketch : parallel_double_sketches) {
            double_sketch.insert({row_idx, col_idx}, update);
          }
          parallel_final_sketch.insert({row_idx, col_idx}, update);
        };
        parallel_matrix_AS.async_visit(row_idx, insert_lambda, col_idx,
                                       matrix_A(row_idx, col_idx));
      }
    }
  }
  comm.barrier();

  // We dump the contents of the S^tAR embeddings to Eigen matrices.
  std::vector<Eigen::MatrixXd> parallel_double_matrices;
  for (int i(0); i < parallel_double_sketches.size(); ++i) {
    // for (const DoubleSketchType &double_sketch : parallel_double_sketches) {
    // parallel_double_matrices.push_back(double_sketch.scaled_registers());
    // using a const reference to avoid an extra copy
    const Eigen::MatrixXd &double_matrix =
        parallel_double_sketches[i].container().registers();
    // it is very important that the dummy matrix have the the correct shapes!
    parallel_double_matrices.push_back(
        Eigen::MatrixXd::Zero(double_matrix.rows(), double_matrix.cols()));
    YGM_ASSERT_MPI(MPI_Allreduce(
        double_matrix.data(), parallel_double_matrices[i].data(),
        double_matrix.rows() * double_matrix.cols(),
        ygm::detail::mpi_typeof(double()), MPI_SUM, comm.get_mpi_comm()));
    comm.barrier();
    // apply scaling factor
    parallel_double_matrices[i] /=
        DoubleSketchType::transform_type::scaling_factor;
  }
  Eigen::MatrixXd parallel_final_matrix =
      Eigen::MatrixXd::Zero(final_transform_type::row_transform_type::size(),
                            final_transform_type::col_transform_type::size());
  {
    const Eigen::MatrixXd &local_final_matrix =
        parallel_final_sketch.container().registers();
    YGM_ASSERT_MPI(MPI_Allreduce(
        local_final_matrix.data(), parallel_final_matrix.data(),
        local_final_matrix.rows() * local_final_matrix.cols(),
        ygm::detail::mpi_typeof(double()), MPI_SUM, comm.get_mpi_comm()));
    comm.barrier();
    parallel_final_matrix /= final_transform_type::scaling_factor;
  }

  return {parallel_double_matrices, parallel_final_matrix};
}

void agreement_double_matrices(
    ygm::comm &comm, const std::vector<Eigen::MatrixXd> &serial_double_matrices,
    const std::vector<Eigen::MatrixXd> &parallel_double_matrices) {
  bool   match = true;
  double mae(0.0);
  double max_error(0.0);

  for (int i(0); i < serial_double_matrices.size(); ++i) {
    const auto &lhs = parallel_double_matrices[i];
    const auto &rhs = serial_double_matrices[i];
    double this_mae = krowkee::sketch::detail::mean_absolute_error(lhs, rhs);
    double this_max_error =
        krowkee::sketch::detail::max_absolute_error(lhs, rhs);
    if (this_mae >= 1e-5) {
      match = false;
    }
    mae += this_mae;
    max_error = std::max(max_error, this_max_error);
  }
  mae /= serial_double_matrices.size();

  bool mae_thresh       = values_close(mae, 0.0);
  bool max_error_thresh = values_close(max_error, 0.0);
  bool match_match      = ranks_close(match, comm) && ranks_close(mae, comm) &&
                     ranks_close(max_error, comm);

  std::string name = "two-sided sketches";

  if (match_match == false || match == false || mae_thresh == false ||
      max_error_thresh == false) {
    comm.cout0("\t", name, " ranks agree? ", match_match, ", match? ", match,
               ", mae: ", mae, ", max_error: ", max_error);
    comm.cout0("");
  }
  CHECK_CONDITION(comm, match_match == true, name + " ranks agree");
  CHECK_CONDITION(comm, match == true, name + " no failures");
  CHECK_CONDITION(
      comm, mae_thresh == true,
      name + " low mean absolute error (" + std::to_string(mae) + ")");
  CHECK_CONDITION(
      comm, max_error_thresh == true,
      name + " low maximum absolute error (" + std::to_string(max_error) + ")");
}

void agreement_matrices(ygm::comm &comm, const std::string &&name,
                        const Eigen::MatrixXd &lhs,
                        const Eigen::MatrixXd &rhs) {
  bool   match     = true;
  double mae       = krowkee::sketch::detail::mean_absolute_error(lhs, rhs);
  double max_error = krowkee::sketch::detail::max_absolute_error(lhs, rhs);
  if (mae >= 1e-5) {
    match = false;
  }

  bool match_success     = ranks_close(match, comm);
  bool mae_success       = ranks_close(mae, comm);
  bool max_error_success = ranks_close(max_error, comm);
  bool success           = match_success && mae_success && max_error_success;

  bool mae_thresh       = values_close(mae, 0.0);
  bool max_error_thresh = values_close(max_error, 0.0);

  if (success == false || match == false || mae_thresh == false ||
      max_error_thresh == false) {
    comm.cout0("\t", name, ", ranks agree (match, mae, max error)? (",
               match_success, ", ", mae_success, ", ", max_error_success,
               "), partial products match? ", match, ", mae: ", mae,
               ", max_error: ", max_error);
    comm.cout0("");
  }
  CHECK_CONDITION(comm, success == true, name + " ranks agree");
  CHECK_CONDITION(comm, match == true, name + " no failures");
  CHECK_CONDITION(
      comm, mae_thresh == true,
      name + " low mean absolute error (" + std::to_string(mae) + ")");
  CHECK_CONDITION(
      comm, max_error_thresh == true,
      name + " low maximum absolute error (" + std::to_string(max_error) + ")");
}

Eigen::MatrixXd serial_multiplication_exact(const Eigen::MatrixXd &matrix_A,
                                            int double_matrix_count) {
  // We compute the exact power iteration product.
  Eigen::MatrixXd serial_product_exact = matrix_A;
  for (int i(0); i < double_matrix_count; ++i) {
    serial_product_exact *= matrix_A;
  }
  return serial_product_exact;
}

Eigen::MatrixXd serial_multiplication_iterative(
    const Eigen::MatrixXd &serial_matrix_AS, const Eigen::MatrixXd &matrix_A,
    int double_matrix_count) {
  // We compute the iterative power iteration product.
  Eigen::MatrixXd serial_product_iterative = matrix_A;
  for (int i(1); i < double_matrix_count; ++i) {
    serial_product_iterative *= matrix_A;
  }
  serial_product_iterative *= serial_matrix_AS;
  return serial_product_iterative;
}

Eigen::MatrixXd localize_AS(
    ygm::container::map<int, Eigen::VectorXd> &parallel_matrix_AS,
    int                                        embedding_size) {
  ygm::comm &comm = parallel_matrix_AS.comm();
  // We compute the iterative power iteration product. If matrix_A were large,
  // it would be necessary to implement this product differently.
  Eigen::MatrixXd localized_AS_piece =
      Eigen::MatrixXd::Zero(parallel_matrix_AS.size(), embedding_size);
  Eigen::MatrixXd localized_AS =
      Eigen::MatrixXd::Zero(parallel_matrix_AS.size(), embedding_size);

  parallel_matrix_AS.for_all(
      [&localized_AS_piece](const int &idx, const Eigen::VectorXd &row) {
        localized_AS_piece.row(idx) = row;
      });
  comm.barrier();

  YGM_ASSERT_MPI(MPI_Allreduce(
      localized_AS_piece.data(), localized_AS.data(),
      localized_AS_piece.rows() * localized_AS_piece.cols(),
      ygm::detail::mpi_typeof(double()), MPI_SUM, comm.get_mpi_comm()));
  comm.barrier();

  // comm.cout0("localized_AS dimensions: (", localized_AS.rows(), ", ",
  //            localized_AS.cols(), ")");

  return localized_AS;
}

Eigen::MatrixXd parallel_multiplication_iterative(
    Eigen::MatrixXd &localized_AS, const Eigen::MatrixXd &matrix_A,
    int double_matrix_count) {
  Eigen::MatrixXd parallel_product_iterative = localized_AS;
  for (int i(0); i < double_matrix_count; ++i) {
    parallel_product_iterative.transpose() *= matrix_A.transpose();
  }

  return parallel_product_iterative;
}

struct lemma_results {
  double success_rate_streaming;
  double success_rate_iterative;
  double epsilon_streaming;
  double epsilon_iterative;

  lemma_results()
      : success_rate_streaming(0.0),
        success_rate_iterative(0.0),
        epsilon_streaming(0.0),
        epsilon_iterative(0.0) {}
};

lemma_results lemma_check(ygm::comm &comm, const std::string &&name,
                          const Eigen::MatrixXd &product_exact,
                          const Eigen::MatrixXd &product_iterative,
                          const Eigen::MatrixXd &product_streaming,
                          const double           epsilon_expected_streaming,
                          const double           epsilon_expected_iterative,
                          const Parameters      &params) {
  lemma_results results;
  int           trials(0);

  for (int i(0); i < product_exact.rows(); ++i) {
    for (int j(i + 1); j < product_exact.rows(); ++j) {
      ++trials;
      // compute exact distance between power iteration rows
      double dist_exact =
          (product_exact.row(i) - product_exact.row(j)).lpNorm<2>();
      // compute distance between iterative embedding rows
      double dist_iterative =
          (product_iterative.row(i) - product_iterative.row(j)).lpNorm<2>();
      double error_iterative = std::abs(1.0 - dist_iterative / dist_exact);
      results.epsilon_iterative += error_iterative;
      if (in_bounds(dist_exact, dist_iterative, epsilon_expected_iterative)) {
        results.success_rate_iterative += 1.0;
      }
      // compute distance between streaming embedding rows
      double dist_streaming =
          (product_streaming.row(i) - product_streaming.row(j)).lpNorm<2>();
      double error_streaming = std::abs(1.0 - dist_streaming / dist_exact);
      results.epsilon_streaming += error_streaming;
      if (in_bounds(dist_exact, dist_streaming, epsilon_expected_streaming)) {
        results.success_rate_streaming += 1.0;
      }
      if (params.verbose && i == 199 && j == 230) {
        if (product_iterative.row(i).size() <= 32) {
          comm.cout0("\tlhs_embedding:\n", product_iterative.row(i), "\n",
                     "\trhs_embedding:\n", product_iterative.row(j));
        }
        comm.cout0(
            "\t(", i, ",", j, ") exact ", dist_exact,
            ")\n\t\titerative (dist/error/success): (", dist_iterative,
            ", 1 +/- ", error_iterative, ", ",
            in_bounds(dist_exact, dist_iterative, epsilon_expected_iterative),
            ")", "\n\t\tstreaming (dist/error/success): (", dist_streaming,
            ", 1 +/- ", error_streaming, ", ",
            in_bounds(dist_exact, dist_streaming, epsilon_expected_streaming));
      }
    }
  }

  results.success_rate_streaming /= trials;
  results.success_rate_iterative /= trials;
  results.epsilon_streaming /= trials;
  results.epsilon_iterative /= trials;

  if (params.verbose) {
    comm.cout0("\n", name,
               " power iteration approximate row distances guarantee (", trials,
               " trials)");
    comm.cout0("\titerative success rate / epsilon / expected = (",
               results.success_rate_iterative, ", ", results.epsilon_iterative,
               ", ", epsilon_expected_iterative, ")");
    comm.cout0("\tstreaming success rate / epsilon / expected = (",
               results.success_rate_streaming, ", ", results.epsilon_streaming,
               ", ", epsilon_expected_streaming, ")\n");
  }
  return results;
}

template <typename SingleSketchType, typename DoubleSketchType,
          typename FinalSketchType>
struct power_iteration_check {
  using single_sketch_type    = SingleSketchType;
  using single_transform_type = typename single_sketch_type::transform_type;
  using single_transform_ptr_type =
      typename single_sketch_type::transform_ptr_type;
  using double_sketch_type    = DoubleSketchType;
  using double_transform_type = typename double_sketch_type::transform_type;
  using double_transform_ptr_type =
      typename double_sketch_type::transform_ptr_type;
  using final_sketch_type    = FinalSketchType;
  using final_transform_type = typename final_sketch_type::transform_type;
  using final_transform_ptr_type =
      typename final_sketch_type::transform_ptr_type;
  using final_col_transform_type =
      typename final_transform_type::col_transform_type;
  using final_col_transform_ptr_type =
      typename final_transform_type::col_transform_ptr_type;

  static_assert(
      std::is_same<single_transform_type,
                   typename double_transform_type::row_transform_type>::value);
  static_assert(
      std::is_same<single_transform_type,
                   typename double_transform_type::col_transform_type>::value);
  static_assert(
      std::is_same<single_transform_type,
                   typename final_transform_type::row_transform_type>::value);

  constexpr std::string name() const {
    std::stringstream ss;
    ss << double_transform_type::name() << " ygm power iteration check";
    return ss.str();
  }

  void operator()(ygm::comm &world, const Parameters &params) const {
    const std::size_t row_count(params.count);
    const std::size_t col_count(row_count);
    const int         single_transform_count = params.transform_count - 1;
    const int         double_transform_count = single_transform_count - 1;

    // We create a vector of shared pointers for each of the individual
    // sketch transforms.
    std::vector<single_transform_ptr_type> single_transform_ptrs;
    for (int i(0); i < single_transform_count; ++i) {
      single_transform_ptrs.push_back(
          std::make_shared<single_transform_type>(params.seed + i));
    }
    final_col_transform_ptr_type final_col_transform_ptr(
        std::make_shared<final_col_transform_type>(params.seed +
                                                   params.transform_count));

    // Using these shared pointers, we now create a vector of pointers to all
    // of the two-sided sketch transforms.
    std::vector<double_transform_ptr_type> double_transform_ptrs;
    for (int i(0); i < double_transform_count; ++i) {
      double_transform_ptrs.push_back(std::make_shared<double_transform_type>(
          single_transform_ptrs[i], single_transform_ptrs[i + 1]));
    }
    final_transform_ptr_type final_transform_ptr(
        std::make_shared<double_transform_type>(single_transform_ptrs.back(),
                                                final_col_transform_ptr));

    // We sample a random matrix to embed. Note that this is not implemented
    // efficiently, as this is a toy example and we will compute a ground
    // truth solution that is intractable for large matrices.
    srand(params.seed);
    static Eigen::MatrixXd matrix_A =
        Eigen::MatrixXd::Random(row_count, col_count);

    // We create the serial one-sided sketch object
    Eigen::MatrixXd serial_matrix_AS = serial_accumulate_AS<single_sketch_type>(
        matrix_A, single_transform_ptrs[0]);
    // We create the parallel one-sided sketch object
    ygm::container::map<int, Eigen::VectorXd> parallel_matrix_AS =
        parallel_accumulate_AS<single_sketch_type, double_sketch_type>(
            world, matrix_A, single_transform_ptrs[0]);
    // We confirm that both implementations arrive at the same AS.
    agreement_parallel_matrix("one-sided sketches", serial_matrix_AS,
                              parallel_matrix_AS, params);

    // We create the serial two-sided matrices
    auto [serial_double_matrices, serial_final_matrix] =
        serial_accumulate_double_matrices<double_sketch_type,
                                          final_sketch_type>(
            matrix_A, double_transform_ptrs, final_transform_ptr);
    // We create the parallel two-sided matrices
    auto [parallel_double_matrices, parallel_final_matrix] =
        parallel_accumulate_double_matrices<double_sketch_type,
                                            final_sketch_type>(
            matrix_A, parallel_matrix_AS, double_transform_ptrs,
            final_transform_ptr);
    // We confirm that both implementations arrive at close double matrices.
    agreement_double_matrices(world, serial_double_matrices,
                              parallel_double_matrices);
    {
      // this is really wasteful but fine for small tests
      std::vector<Eigen::MatrixXd> serial_dummy;
      serial_dummy.push_back(serial_final_matrix);
      std::vector<Eigen::MatrixXd> parallel_dummy;
      parallel_dummy.push_back(parallel_final_matrix);
      agreement_double_matrices(world, serial_dummy, parallel_dummy);
    }

    // We compute the partial products for both serial and parallel
    // implementations.
    Eigen::MatrixXd serial_product_partial = serial_double_matrices[0];
    for (int i(1); i < serial_double_matrices.size(); ++i) {
      serial_product_partial *= serial_double_matrices[i];
    }
    serial_product_partial *= serial_final_matrix;
    Eigen::MatrixXd parallel_product_partial = parallel_double_matrices[0];
    for (int i(1); i < parallel_double_matrices.size(); ++i) {
      parallel_product_partial *= parallel_double_matrices[i];
    }
    parallel_product_partial *= parallel_final_matrix;
    // We confirm that both implementations arrive at close partial products
    agreement_matrices(world, "partial products", serial_product_partial,
                       parallel_product_partial);

    // We compute the serial power iteration product.
    Eigen::MatrixXd serial_product_exact =
        serial_multiplication_exact(matrix_A, params.transform_count - 1);

    // We confirm that serial and parallel iterative products are close.
    Eigen::MatrixXd serial_product_iterative = serial_multiplication_iterative(
        serial_matrix_AS, matrix_A, serial_double_matrices.size());
    Eigen::MatrixXd localized_AS =
        localize_AS(parallel_matrix_AS, serial_matrix_AS.cols());
    Eigen::MatrixXd parallel_product_iterative =
        parallel_multiplication_iterative(localized_AS, matrix_A,
                                          serial_double_matrices.size());
    agreement_matrices(world, "iterative products", serial_product_iterative,
                       parallel_product_iterative);

    // We confirm that the serial and parallel streaming products are close.
    Eigen::MatrixXd serial_product_streaming =
        serial_matrix_AS * serial_product_partial;
    Eigen::MatrixXd parallel_product_streaming =
        localized_AS * parallel_product_partial;
    agreement_matrices(world, "streaming products", serial_product_streaming,
                       parallel_product_streaming);

    if (params.verbose) {
      world.cout0("A(5,7) = ", matrix_A(5, 7));
      world.cout0("A^", params.transform_count,
                  "(5,7) = ", serial_product_exact(5, 7));
    }

    // We now compare the embedding vectors. In practice this could be done
    // more efficiently, but this implementation suffices for illustration.
    const double srank = stable_rank(matrix_A);
    // :math:`\sqrt{2} \left ( (1 + 2\varepsilon)^{r - 1} \right )
    // \|A\|^r_{op}`.
    const double epsilon_expected_streaming = std::sqrt(
        16 *
        (srank + std::log((params.transform_count - 1) *
                          single_transform_type::replication_count())) /
        single_transform_type::range_size());
    double expected_epsilon_iterative =
        std::sqrt(16 * std::log(params.count) /
                  (single_transform_type::range_size() *
                   single_transform_type::range_size()));

    lemma_results lemma_results_serial = lemma_check(
        world, "serial", serial_product_exact, serial_product_iterative,
        serial_product_streaming, epsilon_expected_streaming,
        expected_epsilon_iterative, params);
    lemma_results lemma_results_parallel = lemma_check(
        world, "parallel", serial_product_exact, parallel_product_iterative,
        parallel_product_streaming, epsilon_expected_streaming,
        expected_epsilon_iterative, params);
    bool success_rate_agreement_iterative =
        lemma_results_serial.success_rate_iterative ==
        lemma_results_parallel.success_rate_iterative;
    bool success_rate_agreement_streaming =
        lemma_results_serial.success_rate_streaming ==
        lemma_results_parallel.success_rate_streaming;
    bool epsilon_agreement_iterative =
        values_close(lemma_results_serial.epsilon_iterative,
                     lemma_results_parallel.epsilon_iterative);
    bool epsilon_agreement_streaming =
        values_close(lemma_results_serial.epsilon_streaming,
                     lemma_results_parallel.epsilon_streaming);
    CHECK_CONDITION(world, success_rate_agreement_iterative == true,
                    "equal iterative success rates");
    CHECK_CONDITION(world, success_rate_agreement_streaming == true,
                    "equal streaming success rates");
    CHECK_CONDITION(world, epsilon_agreement_iterative == true,
                    "close iterative empirical epsilon rates");
    CHECK_CONDITION(world, epsilon_agreement_streaming == true,
                    "close streaming empirical epsilon rates");
    bool epsilon_iterative_succeeds =
        lemma_results_serial.epsilon_iterative <= expected_epsilon_iterative;
    bool epsilon_streaming_succeeds =
        lemma_results_serial.epsilon_streaming <= epsilon_expected_streaming;
    CHECK_CONDITION(world, epsilon_agreement_iterative == true,
                    "iterative empirical epsilon (" +
                        std::to_string(lemma_results_serial.epsilon_iterative) +
                        ") below threshold (" +
                        std::to_string(expected_epsilon_iterative) + ")");
    CHECK_CONDITION(world, epsilon_agreement_iterative == true,
                    "streaming empirical epsilon (" +
                        std::to_string(lemma_results_serial.epsilon_streaming) +
                        ") below threshold (" +
                        std::to_string(epsilon_expected_streaming) + ")");
  }
};

void print_help(char *exe_name) {
  std::cout
      << "\nusage:  " << exe_name << "\n"
      << "\t-c, --count <int>              - number of rows/cols in matrix\n"
      << "\t-r, --range <int>              - range of sketch transform\n"
      << "\t-R, --replication <int>        - number of tiled sketch "
         "transforms\n"
      << "\t-t, --transforms <int>         - number of transforms (i.e., power "
         "of matrix)\n"
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
        {"transforms", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}};

    int curind = optind;
    c = getopt_long(argc, argv, "-:c:r:R:t:s:vh", long_options, &option_index);
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
        std::cout << "Warning: setting range size on the command line is not "
                     "currently supported for this test"
                  << std::endl;
        params.range_size = std::atoll(optarg);
        break;
      case 'R':
        std::cout << "Warning: setting replication count on the command line "
                     "is not currently supported for this test"
                  << std::endl;
        params.replication_count = std::atoll(optarg);
        break;
      case 't':
        params.transform_count = std::atol(optarg);
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
        break;
    }
  }
}

using register_type = double;

template <std::size_t RangeSize, std::size_t ReplicationCount>
void perform_tests(ygm::comm &world, const Parameters &params) {
  using single_sketch_type =
      krowkee::sketch::SparseJLT<register_type, RangeSize, ReplicationCount,
                                 std::shared_ptr>;
  using double_sketch_type =
      krowkee::sketch::DoubleSparseJLT<register_type, RangeSize,
                                       ReplicationCount, std::shared_ptr>;
  using final_sketch_type =
      krowkee::sketch::DoubleSparseJLT<register_type, RangeSize,
                                       ReplicationCount, std::shared_ptr,
                                       RangeSize, ReplicationCount>;
  krowkee::print_line(world);
  krowkee::print_line(world);
  world.cout0("Testing ", double_sketch_type::full_name());
  world.cout0("\tUsing std::shared_ptr pointers");
  krowkee::print_line(world);
  krowkee::print_line(world);

  world.cout0("\n");

  krowkee::do_ygm_test<power_iteration_check<
      single_sketch_type, double_sketch_type, final_sketch_type>>(world, world,
                                                                  params);
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    // Using krowkee requires the selection of a sketch type for both a single
    // and double-sided sketch, here encapsulated as `single_sketch_type` and
    // `double_sketch_type`, respectively. We use the `SparseJLT` and
    // `DoubleSparseJLT` types defined in the simple API in
    // `krowkee/sketch.hpp`. These types have four template parameters:
    //   1. the numeric type to be used by each register (here `double`),
    //   2. a `std::size_t` parameter `range_size` indicating the number of
    //      registers used by each instance of the internal transform,
    //   3. a `std::size_t` parameter `replication_count` indicating the number
    //      of instances of the transform to be used, and
    //   4. a shared pointer type to be used by the shared transform object
    //      (`std::shared_ptr` for shared memory implementations).

    uint64_t                    count             = 256;
    constexpr const std::size_t range_size        = 128;
    constexpr const std::size_t replication_count = 4;
    uint64_t                    transform_count   = 4;
    std::uint64_t               seed              = 4;
    bool                        verbose           = false;
    bool                        do_all(argc == 1);

    Parameters params{count,           range_size, replication_count,
                      transform_count, seed,       verbose};
    parse_args(argc, argv, params);

    perform_tests<range_size, replication_count>(world, params);
  }
  return 0;
}