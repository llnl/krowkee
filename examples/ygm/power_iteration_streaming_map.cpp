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

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    constexpr const std::size_t row_count(256);
    constexpr const std::size_t col_count(row_count);
    constexpr const std::size_t transform_count(4);
    std::uint64_t               seed(4);
    static bool                 verbose(true);

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
    using single_sketch_type =
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
    constexpr const std::size_t final_range_size        = 8;
    constexpr const std::size_t final_replication_count = 4;
    using final_double_sketch_type = krowkee::sketch::DoubleSparseJLT<
        register_type, range_size, replication_count, std::shared_ptr,
        final_range_size, final_replication_count>;

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
    using final_double_transform_type =
        typename final_double_sketch_type::transform_type;
    using final_double_transform_ptr_type =
        typename final_double_sketch_type::transform_ptr_type;
    //  We also get the type `final_single_sketch_type` of the corresponding
    //  final column sketch.
    using final_single_transform_type =
        typename final_double_transform_type::col_transform_type;
    using final_single_transform_ptr_type =
        typename final_double_transform_type::col_transform_ptr_type;
    // We verify that we did not make a mistake above, and both sketch types use
    // the same transform type.
    static_assert(std::is_same<
                  single_transform_type,
                  typename double_transform_type::row_transform_type>::value);
    static_assert(std::is_same<
                  single_transform_type,
                  typename double_transform_type::col_transform_type>::value);
    static_assert(
        std::is_same<
            single_transform_type,
            typename final_double_transform_type::row_transform_type>::value);
    // Note that we do not try to match the final column transform, as it is
    // allowed to have different parameters.
    static_assert(
        final_double_transform_type::col_transform_type::range_size() ==
        final_range_size);
    static_assert(
        final_double_transform_type::col_transform_type::replication_count() ==
        final_replication_count);

    // We create these counters for convenience. Note that `transform_count` is
    // the total number of single sketch transforms involved in computing the
    // product, matching the exponent approximately applied to $A$. So, there
    // are `transform_count - 1` single transforms of the inserted type and one
    // of the final type, and `transform_count - 2` double transforms of the
    // intermediate type and one of the final, possibly rectangular, type.
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

    // We sample a random matrix to embed. Note that this is not implemented
    // efficiently, as this is a toy example and we will compute a ground
    // truth solution that is intractable for large matrices.
    srand(seed);
    static Eigen::MatrixXd matrix_A =
        Eigen::MatrixXd::Random(row_count, col_count);

    // We create a ygm::map that will hold the sketches for each row of A. Note
    // that we use the transform pointer that conforms with the left-hand
    // transform of the zeroth two-sided sketch.
    single_sketch_type default_sketch(single_transform_ptrs[0]);
    ygm::container::map<int, single_sketch_type> single_sketches(
        world, default_sketch);

    // We create the parallel two-sided matrices
    std::vector<Eigen::MatrixXd> parallel_double_matrices;
    // We also create local double-sided sketches that will hold the double
    // sided embeddings.
    static std::vector<double_sketch_type> parallel_double_sketches;
    for (const double_transform_ptr_type &double_transform_ptr :
         double_transform_ptrs) {
      parallel_double_sketches.emplace_back(double_transform_ptr);
    }

    // We also create the parallel final double-sided sketch.
    static final_double_sketch_type parallel_final_double_sketch(
        final_double_transform);

    // here we simulate streaming over `matrix_A` where each rank gets a subset
    // of the columns, even those in this example `matrix_A` is small enough
    // that it is replicated to each rank. We apply the both the single and
    // double sketches to each element of `matrix_A` in a single pass.
    for (int col_idx(0); col_idx < matrix_A.cols(); ++col_idx) {
      if (world.rank() == (col_idx % world.size())) {
        for (int row_idx(0); row_idx < matrix_A.rows(); ++row_idx) {
          auto insert_lambda = [](const int row_idx, single_sketch_type &sketch,
                                  const int col_idx, const double update) {
            // insert `(col_idx) <- update` into `row_idx`'s vector sketch
            sketch.insert(col_idx, update);
            // insert `(row_idx, col_idx) <- update' into all matrix
            // sketches.
            for (double_sketch_type &double_sketch : parallel_double_sketches) {
              double_sketch.insert({row_idx, col_idx}, update);
            }
            // insert `(row_idx, col_idx) <- update' into the final matrix
            // sketch.
            parallel_final_double_sketch.insert({row_idx, col_idx}, update);
          };
          single_sketches.async_visit(row_idx, insert_lambda, col_idx,
                                      matrix_A(row_idx, col_idx));
        }
      }
    }
    world.barrier();

    // We dump the contents of the S^tAR embeddings to Eigen matrices. Each rank
    // will hold their local updates, and then we allreduce the matrices so that
    // each rank holds the fully sketched matrices.
    for (int i(0); i < parallel_double_sketches.size(); ++i) {
      // using a const reference to avoid an extra copy
      const Eigen::MatrixXd &double_matrix =
          parallel_double_sketches[i].container().registers();
      // it is very important that the dummy matrix have the the correct shapes!
      parallel_double_matrices.push_back(
          Eigen::MatrixXd::Zero(double_matrix.rows(), double_matrix.cols()));
      YGM_ASSERT_MPI(MPI_Allreduce(
          double_matrix.data(), parallel_double_matrices[i].data(),
          double_matrix.rows() * double_matrix.cols(),
          ygm::detail::mpi_typeof(double()), MPI_SUM, world.get_mpi_comm()));
      world.barrier();
      // apply scaling factor
      parallel_double_matrices[i] /= double_transform_type::scaling_factor;
    }

    // We repeat this process for the final matrix sketch.
    const Eigen::MatrixXd &local_final_double_matrix =
        parallel_final_double_sketch.container().registers();
    Eigen::MatrixXd parallel_final_double_matrix = Eigen::MatrixXd::Zero(
        local_final_double_matrix.rows(), local_final_double_matrix.cols());
    YGM_ASSERT_MPI(MPI_Allreduce(
        local_final_double_matrix.data(), parallel_final_double_matrix.data(),
        local_final_double_matrix.rows() * local_final_double_matrix.cols(),
        ygm::detail::mpi_typeof(double()), MPI_SUM, world.get_mpi_comm()));
    world.barrier();
    parallel_final_double_matrix /= final_double_transform_type::scaling_factor;

    // We compute the partial products of the two-sided matrices.
    Eigen::MatrixXd parallel_product_partial = parallel_double_matrices[0];
    for (int i(1); i < parallel_double_matrices.size(); ++i) {
      parallel_product_partial *= parallel_double_matrices[i];
    }
    parallel_product_partial *= parallel_final_double_matrix;

    // We compute the parallel streaming product.
    ygm::container::map<int, Eigen::VectorXd> parallel_product_streaming(world);
    single_sketches.for_all(
        [&parallel_product_streaming, &parallel_product_partial](
            const int idx, const single_sketch_type &sketch) {
          // dump the scaled sketch to an Eigen vector.
          Eigen::VectorXd embedding(sketch.size());
          auto            scaled_registers = sketch.scaled_registers();
          for (int i(0); i < scaled_registers.size(); ++i) {
            embedding(static_cast<Eigen::Index>(i)) =
                static_cast<double>(scaled_registers[i]);
          }
          // multiply by the partial product and store the row product. Note
          // that Eigen assumes all vectors are Nx1 matrices, so the transpose
          // is required for correctness.
          parallel_product_streaming.async_insert(
              idx, embedding.transpose() * parallel_product_partial);
        });
    world.barrier();

    // In practice the embedding would be done here, and we could use
    // `parallel_product_streaming` as an embedding of the rows for downstream
    // applications. In this example, we compute the exact product in serial and
    // show that the row distances are approximately, if coarsely, preserved.

    // We compute the serial exact power iteration product. We will check our
    // embedded results against this.
    static Eigen::MatrixXd serial_product_exact = matrix_A;
    for (int i(0); i < transform_count - 1; ++i) {
      serial_product_exact *= matrix_A;
    }

    if (verbose) {
      world.cout0("A(5,7) = ", matrix_A(5, 7));
      world.cout0("A^", transform_count,
                  "(5,7) = ", serial_product_exact(5, 7));
    }

    // We use the stable rank to compute the expected approximation bound
    // epsilon from the embedding size
    // :math:`\sqrt{2} \left ( (1 + 2\varepsilon)^{r - 1} \right )
    // \|A\|^r_{op}`.
    static const double srank                      = stable_rank(matrix_A);
    static const double epsilon_expected_streaming = std::sqrt(
        16 *
        (srank + std::log((transform_count - 1) *
                          single_transform_type::replication_count())) /
        single_transform_type::range_size());

    // We now compare the pairwise distance between embedding vectors to their
    // true distances. One would never do this in practice, but this
    // implementation suffices for illustration.

    static double success_rate_streaming = 0.0;
    static double epsilon_streaming      = 0.0;
    static int    trials                 = 0;

    parallel_product_streaming.for_all([&parallel_product_streaming](
                                           const int              row_idx,
                                           const Eigen::VectorXd &row_product) {
      for (int col_idx(row_idx + 1); col_idx < serial_product_exact.rows();
           ++col_idx) {
        parallel_product_streaming.async_visit(
            col_idx,
            [](const int col_idx, const Eigen::VectorXd &col_product,
               const int row_idx, const Eigen::VectorXd &row_product) {
              ++trials;
              // compute exact distance between power iteration rows
              double dist_exact = (serial_product_exact.row(row_idx) -
                                   serial_product_exact.row(col_idx))
                                      .lpNorm<2>();
              // compute distance between streaming embedding rows
              double dist_streaming = (row_product - col_product).lpNorm<2>();
              double error_streaming =
                  std::abs(1.0 - dist_streaming / dist_exact);
              epsilon_streaming += error_streaming;
              if (in_bounds(dist_exact, dist_streaming,
                            epsilon_expected_streaming)) {
                success_rate_streaming += 1.0;
              }
              if (verbose && row_idx == 199 && col_idx == 230) {
                std::cout << "\t(" << row_idx << "," << col_idx << ") exact "
                          << dist_exact
                          << ")\n\t\tstreaming (dist/error/success): ("
                          << dist_streaming << ", 1 +/- " << error_streaming
                          << ", "
                          << in_bounds(dist_exact, dist_streaming,
                                       epsilon_expected_streaming);
              }
            },
            row_idx, row_product);
      }
    });
    world.barrier();

    trials                 = ygm::sum(trials, world);
    success_rate_streaming = ygm::sum(success_rate_streaming, world) / trials;
    epsilon_streaming      = ygm::sum(epsilon_streaming, world) / trials;

    if (verbose) {
      world.cout0(
          "\nparallel power iteration approximate row distances guarantee (",
          trials, " trials)");
      world.cout0("\tstreaming success rate / epsilon / expected = (",
                  success_rate_streaming, ", ", epsilon_streaming, ", ",
                  epsilon_expected_streaming, ")\n");
    }
  }
}