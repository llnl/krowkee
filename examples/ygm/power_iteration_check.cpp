// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <krowkee/hash/hash.hpp>
#include <krowkee/sketch.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <random>
#include <type_traits>

static const bool verbose = false;

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
    const ygm::container::map<int, Eigen::VectorXd> &parallel_matrix_AS) {
  ygm::comm &comm  = parallel_matrix_AS.comm();
  bool       match = true;
  double     mae(0.0);
  double     max_error(0.0);
  parallel_matrix_AS.for_all([&serial_matrix_AS, &match, &mae, &max_error](
                                 const int idx, const Eigen::VectorXd &lhs) {
    const auto &rhs = serial_matrix_AS(idx, Eigen::all);
    if (verbose && idx == 199) {
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

  comm.cout0("\t", name, " match? ", match, ", mae: ", mae,
             ", max_error: ", max_error);
  comm.cout0("");
}

template <typename DoubleSketchType>
std::vector<Eigen::MatrixXd> serial_accumulate_double_matrices(
    const Eigen::MatrixXd &matrix_A,
    std::vector<typename DoubleSketchType::transform_ptr_type>
        &double_transform_ptrs) {
  // We create the double matrices array
  std::vector<Eigen::MatrixXd> serial_double_matrices;

  // We create a vector of local double-sided sketches that will hold the double
  // sided embeddings.
  std::vector<DoubleSketchType> serial_double_sketches;
  for (const auto &double_transform_ptr : double_transform_ptrs) {
    serial_double_sketches.emplace_back(double_transform_ptr);
  }

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
      ++j;
    }
    ++i;
  }

  // We dump the contents of the S^tAR embeddings to Eigen matrices.
  for (const DoubleSketchType &double_sketch : serial_double_sketches) {
    serial_double_matrices.push_back(double_sketch.scaled_registers());
  }

  return serial_double_matrices;
}

template <typename DoubleSketchType>
std::vector<Eigen::MatrixXd> parallel_accumulate_double_matrices(
    const Eigen::MatrixXd                     &matrix_A,
    ygm::container::map<int, Eigen::VectorXd> &parallel_matrix_AS,
    std::vector<typename DoubleSketchType::transform_ptr_type>
        &double_transform_ptrs) {
  using double_sketch_type = DoubleSketchType;
  using double_transform_ptr_type =
      typename DoubleSketchType::transform_ptr_type;

  ygm::comm &comm = parallel_matrix_AS.comm();

  // We also create local double-sided sketches that will hold the double
  // sided embeddings.
  static std::vector<double_sketch_type> parallel_double_sketches;
  for (const double_transform_ptr_type &double_transform_ptr :
       double_transform_ptrs) {
    parallel_double_sketches.emplace_back(double_transform_ptr);
  }

  // We apply the double sketches to each element of `matrix_A` in a single
  // pass. In practice this could be interleaved with the accumulation of AS.
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

  return parallel_double_matrices;
}

template <typename T>
bool ranks_close(T &value, ygm::comm &comm, double rtol = 1e-5,
                 double atol = 1e-8) {
  T pos_min = ygm::min(value, comm);
  T neg_min = ygm::min(-value, comm);
  return std::abs(pos_min - neg_min) <=
         (atol + rtol * std::max(std::abs(pos_min), std::abs(neg_min)));
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

  bool success = ranks_close(match, comm) && ranks_close(mae, comm) &&
                 ranks_close(max_error, comm);

  comm.cout0("\tRanks agree? ", success, ", two-sided sketches match? ", match,
             ", mae: ", mae, ", max_error: ", max_error);
  comm.cout0("");
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

  bool success = ranks_close(match, comm) && ranks_close(mae, comm) &&
                 ranks_close(max_error, comm);

  comm.cout0("\t", name, ", ranks agree? ", success,
             ", partial products match? ", match, ", mae: ", mae,
             ", max_error: ", max_error);
  comm.cout0("");
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

Eigen::MatrixXd parallel_multiplication_iterative(
    ygm::container::map<int, Eigen::VectorXd> &parallel_matrix_AS,
    const Eigen::MatrixXd &matrix_A, int double_matrix_count,
    int embedding_size) {
  ygm::comm &comm = parallel_matrix_AS.comm();
  // We compute the iterative power iteration product. If matrix_A were large,
  // it would be necessary to implement this product differently.
  Eigen::MatrixXd localized_AS_piece =
      Eigen::MatrixXd::Zero(matrix_A.rows(), embedding_size);
  Eigen::MatrixXd localized_AS =
      Eigen::MatrixXd::Zero(matrix_A.rows(), embedding_size);

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

  Eigen::MatrixXd parallel_product_iterative = matrix_A;
  for (int i(1); i < double_matrix_count; ++i) {
    parallel_product_iterative *= matrix_A;
  }
  parallel_product_iterative *= localized_AS;

  return parallel_product_iterative;
}

void lemma_check(ygm::comm &comm, const std::string &&name,
                 const Eigen::MatrixXd &product_exact,
                 const Eigen::MatrixXd &product_streaming,
                 const Eigen::MatrixXd &product_iterative,
                 const double           epsilon_expected) {
  double success_rate_streaming(0.0);
  double success_rate_iterative(0.0);
  double epsilon_streaming(0.0);
  double epsilon_iterative(0.0);
  int    trials(0);

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
      epsilon_iterative += error_iterative;
      if (in_bounds(dist_exact, dist_iterative, epsilon_expected)) {
        success_rate_iterative += 1.0;
      }
      // compute distance between streaming embedding rows
      double dist_streaming =
          (product_streaming.row(i) - product_streaming.row(j)).lpNorm<2>();
      double error_streaming = std::abs(1.0 - dist_streaming / dist_exact);
      epsilon_streaming += error_streaming;
      if (in_bounds(dist_exact, dist_streaming, epsilon_expected)) {
        success_rate_streaming += 1.0;
      }
      if (verbose && i == 199 && j == 230) {
        comm.cout0("\tlhs_embedding:\n", product_iterative.row(i), "\n",
                   "\trhs_embedding:\n", product_iterative.row(j));

        comm.cout0("\t(", i, ",", j, ") exact ", dist_exact,
                   ")\n\t\titerative (dist/error/success): (", dist_iterative,
                   ", 1 +/- ", error_iterative, ", ",
                   in_bounds(dist_exact, dist_iterative, epsilon_expected), ")",
                   "\n\t\tstreaming (dist/error/success): (", dist_streaming,
                   ", 1 +/- ", error_streaming, ", ",
                   in_bounds(dist_exact, dist_streaming, epsilon_expected));
      }
    }
  }

  success_rate_streaming /= trials;
  success_rate_iterative /= trials;
  epsilon_streaming /= trials;
  epsilon_iterative /= trials;

  comm.cout0("\n", name,
             " power iteration approximate row distances guarantee (", trials,
             " trials)");
  comm.cout0("\titerative success rate / epsilon / expected = (",
             success_rate_iterative, ", ", epsilon_iterative, ", ",
             epsilon_expected, ")");
  comm.cout0("\tstreaming success rate / epsilon / expected = (",
             success_rate_streaming, ", ", epsilon_streaming, ", ",
             epsilon_expected, ")\n");
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    const std::size_t row_count(256);
    const std::size_t col_count(row_count);
    const std::size_t transform_count(4);
    std::uint64_t     seed(4);
    bool              verbose(true);

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
    constexpr const std::size_t range_size        = 128;
    constexpr const std::size_t replication_count = 4;
    using register_type                           = double;
    using single_sketch_type =
        krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                                   std::shared_ptr>;
    using double_sketch_type =
        krowkee::sketch::DoubleSparseJLT<register_type, range_size,
                                         replication_count, std::shared_ptr>;

    // Having established our sketch types, we must now create shared pointers
    // to all of the associated sketch transforms. Each doubled transform is
    // multiplied together with its neighbor in the form $AS S^TARR^TAQ$, for
    // sketch transforms `S`, `R`, and `Q` and input matrix `A`. The sketch
    // types includes typedefs of the transform and pointer types. This is where
    // the random seed is used. Transforms of the same type sharing the same
    // seed will behave identically. As this is a distributed memory code, we
    // create a `std::shared_ptr` of the transform to be used to define the
    // sketch data structures on each rank, ensuring that each uses the same
    // transform.
    using single_transform_type = typename single_sketch_type::transform_type;
    using single_transform_ptr_type =
        typename single_sketch_type::transform_ptr_type;
    using double_transform_type = typename double_sketch_type::transform_type;
    using double_transform_ptr_type =
        typename double_sketch_type::transform_ptr_type;
    // We verify that we did not make a mistake above, and both sketch types use
    // the same transform type.
    static_assert(std::is_same<
                  single_transform_type,
                  typename double_transform_type::row_transform_type>::value);
    static_assert(std::is_same<
                  single_transform_type,
                  typename double_transform_type::col_transform_type>::value);

    // We create a vector of shared pointers for each of the individual
    // sketch transforms.
    std::vector<single_transform_ptr_type> single_transform_ptrs;
    for (int i(0); i < transform_count; ++i) {
      single_transform_ptrs.push_back(
          std::make_shared<single_transform_type>(seed + i));
    }
    // Using these shared pointers, we now create a vector of pointers to all of
    // the two-sided sketch transforms.
    std::vector<double_transform_ptr_type> double_transform_ptrs;
    for (int i(0); i < transform_count - 1; ++i) {
      double_transform_ptrs.push_back(std::make_shared<double_transform_type>(
          single_transform_ptrs[i], single_transform_ptrs[i + 1]));
    }

    // We sample a random matrix to embed. Note that this is not implemented
    // efficiently, as this is a toy example and we will compute a ground
    // truth solution that is intractable for large matrices.
    srand(seed);
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
    agreement_parallel_matrix("One-sided sketches", serial_matrix_AS,
                              parallel_matrix_AS);

    // We create the serial two-sided matrices
    std::vector<Eigen::MatrixXd> serial_double_matrices =
        serial_accumulate_double_matrices<double_sketch_type>(
            matrix_A, double_transform_ptrs);
    // We create the parallel two-sided matrices
    std::vector<Eigen::MatrixXd> parallel_double_matrices =
        parallel_accumulate_double_matrices<double_sketch_type>(
            matrix_A, parallel_matrix_AS, double_transform_ptrs);
    // We confirm that both implementations arrive at close double matrices.
    agreement_double_matrices(world, serial_double_matrices,
                              parallel_double_matrices);

    // We compute the partial products for both serial and parallel
    // implementations.
    Eigen::MatrixXd serial_product_partial = serial_double_matrices[0];
    for (int i(1); i < serial_double_matrices.size(); ++i) {
      serial_product_partial *= serial_double_matrices[i];
    }
    Eigen::MatrixXd parallel_product_partial = parallel_double_matrices[0];
    for (int i(1); i < parallel_double_matrices.size(); ++i) {
      parallel_product_partial *= parallel_double_matrices[i];
    }
    // We confirm that both implementations arrive at close partial products
    agreement_matrices(world, "Partial products", serial_product_partial,
                       parallel_product_partial);

    // We compute the serial power iteration product.
    Eigen::MatrixXd serial_product_exact =
        serial_multiplication_exact(matrix_A, serial_double_matrices.size());

    // We confirm that serial and parallel iterative products are close.
    Eigen::MatrixXd serial_product_iterative = serial_multiplication_iterative(
        serial_matrix_AS, matrix_A, serial_double_matrices.size());
    Eigen::MatrixXd parallel_product_iterative =
        parallel_multiplication_iterative(parallel_matrix_AS, matrix_A,
                                          serial_double_matrices.size(),
                                          serial_matrix_AS.cols());
    agreement_matrices(world, "Iterative products", serial_product_iterative,
                       parallel_product_iterative);

    // We confirm that the serial and parallel streaming products are close.
    Eigen::MatrixXd serial_product_streaming =
        serial_product_partial * serial_matrix_AS;
    // Eigen::MatrixXd parallel_product_streaming =
    //     parallel_multiplication_streaming(parallel_matrix_AS,
    //                                       parallel_product_partial);
    // agreement_matrices(world, "Streaming products", serial_product_streaming,
    //                    parallel_product_streaming);

    world.cout0("A(5,7) = ", matrix_A(5, 7));
    world.cout0("A^", transform_count, "(5,7) = ", serial_product_exact(5, 7));

    // We now compare the embedding vectors. In practice this could be done more
    // efficiently, but this implementation suffices for illustration.
    const double srank = stable_rank(matrix_A);
    // :math:`\sqrt{2} \left ( (1 + 2\varepsilon)^{r - 1} \right )
    // \|A\|^r_{op}`.
    const double epsilon_expected = std::sqrt(
        16 *
        (srank +
         std::log((transform_count - 1) *
                  single_sketch_type::transform_type::replication_count())) /
        range_size);

    lemma_check(world, "serial", serial_product_exact, serial_product_streaming,
                serial_product_iterative, epsilon_expected);
    lemma_check(world, "parallel", serial_product_exact,
                parallel_product_iterative, parallel_product_iterative,
                epsilon_expected);
  }
  return 0;
}