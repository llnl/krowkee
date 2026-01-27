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
bool allclose(const std::vector<T> &lhs, const std::vector<T> &rhs,
              const double rtol = 1e-7, const double atol = 1e-5) {
  for (int i(0); i < lhs.size(); ++i) {
    if (std::abs(lhs[i] - rhs[i]) <= atol + lhs[i] ||
        std::abs(lhs[i] - rhs[i]) <= atol + rhs[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
T mean_absolute_error(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  T ret(0);
  for (int i(0); i < lhs.size(); ++i) {
    ret += std::abs(lhs[i] - rhs[i]);
  }
  return ret /= lhs.size();
}

int main(int argc, char **argv) {
  // We create the YGM communicator to be used. The example proceeds similarly
  // to `examples/sparse_jlt.cpp`, where we sample some large-dimensional data
  // on each rank, accumulate sketches, and insert those sketches into a
  // ygm::container::map.
  ygm::comm world(&argc, &argv);
  {
    const std::size_t row_count(256);
    const std::size_t col_count(row_count);
    const std::size_t transform_count(4);
    std::uint64_t     seed(4);
    srand(seed);
    bool verbose(true);

    // Using krowkee requires the selection of a sketch type for both a single
    // and double-sided sketch, here encapsulated as `sketch_type` and
    // `double_sketch_type`, respectively. We use the `SparseJLT` and
    // `DoubleSparseJLT` types defined in the simple API in
    // `krowkee/sketch.hpp`. These types have four template parameters:
    //   1. the numeric type to be used by each register (here `double`),
    //   2. a `std::size_t` parameter `range_size` indicating the number of
    //      registers used by each instance of the internal transform,
    //   3. a `std::size_t` parameter `replication_count` indicating the number
    //      of instances of the transform to be used, and
    //   4. a shared pointer type to be used by the shared transform object
    //      (`ygm::ygm_ptr` for shared memory implementations).
    constexpr const std::size_t range_size        = 128;
    constexpr const std::size_t replication_count = 4;
    constexpr const std::size_t embedding_size = range_size * replication_count;
    using register_type                        = double;
    using sketch_type =
        krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                                   ygm::ygm_ptr>;
    using double_sketch_type =
        krowkee::sketch::DoubleSparseJLT<register_type, range_size,
                                         replication_count, ygm::ygm_ptr>;

    // Having established our sketch types, we must now create shared pointers
    // to all of the associated sketch transforms. Each doubled transform is
    // multiplied together with its neighbor in the form $AS S^TARR^TAQ$, for
    // sketch transforms `S`, `R`, and `Q` and input matrix `A`. The sketch
    // types includes typedefs of the transform and pointer types. This is where
    // the random seed is used. Transforms of the same type sharing the same
    // seed will behave identically. As this is a distributed memory code, we
    // create a ygm::ygm_ptr of the transform to be used to define the sketch
    // data structures on each rank, ensuring that each uses the same transform.
    using single_transform_type     = typename sketch_type::transform_type;
    using single_transform_ptr_type = typename sketch_type::transform_ptr_type;
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
    std::vector<single_transform_type> single_transforms;
    for (int i(0); i < transform_count; ++i) {
      single_transforms.emplace_back(seed + i);
    }
    std::vector<single_transform_ptr_type> single_transform_ptrs;
    for (single_transform_type &transform : single_transforms) {
      single_transform_ptrs.push_back(world.make_ygm_ptr(transform));
    }
    std::vector<double_transform_type> double_transforms;
    for (int i(0); i < transform_count - 1; ++i) {
      double_transforms.emplace_back(single_transform_ptrs[i],
                                     single_transform_ptrs[i + 1]);
    }
    std::vector<double_transform_ptr_type> double_transform_ptrs;
    for (double_transform_type &double_transform : double_transforms) {
      double_transform_ptrs.push_back(world.make_ygm_ptr(double_transform));
    }

    for (const double_transform_ptr_type &transform : double_transform_ptrs) {
      world.cout0(transform->full_name(), ", seed: ", transform->seed());
    }
    world.cout0("");

    // We create a `ygm::container::map` that will hold the AS embeddings for
    // each of our sketches. We also create an empty sketch using our transform
    // as the default value.
    sketch_type default_sketch(single_transform_ptrs[0]);
    ygm::container::map<int, sketch_type> sketch_map(world, default_sketch);

    // We also create local double-sided sketches that will hold the double
    // sided embeddings.
    static std::vector<double_sketch_type> double_sketches;
    for (const double_transform_ptr_type &double_transform_ptr :
         double_transform_ptrs) {
      double_sketches.emplace_back(double_transform_ptr);
    }

    // We sample a random matrix to embed. Note that this is not implemented
    // efficiently, as this is a toy example and we will compute a ground
    // truth solution that is intractable for large matrices.
    static Eigen::MatrixXd matrix_A =
        Eigen::MatrixXd::Random(row_count, col_count);

    // We compute the ground truth power iteration of the matrix.
    static Eigen::MatrixXd product_exact = matrix_A;
    for (int i(1); i < transform_count; ++i) {
      product_exact *= matrix_A;
    }
    double srank = stable_rank(product_exact);
    world.cout0("A(5,7) = ", matrix_A(5, 7));
    world.cout0("A^", transform_count, "(5,7) = ", product_exact(5, 7));
    world.cout0("");

    // We will now simulate asynchronously updates to the associated sketch by
    // communicating elements of the matrix using YGM. In this example each
    // stream is generated by a particular rank's slicing from a shared matrix,
    // but in practice we assume that updates to any element could appear on any
    // rank.
    for (int col_idx(0); col_idx < col_count; ++col_idx) {
      if (world.rank() == (col_idx % world.size())) {
        for (int row_idx(0); row_idx < row_count; ++row_idx) {
          // This remote lambda will be executed for each stream update and will
          // go to the appropriate single sketch on the appropriate rank,
          // simultaneously updating all two-sided sketches.
          auto insert_lambda = [](const int &row_idx, sketch_type &sketch,
                                  const int &col_idx, const double update) {
            // insert `(col_idx) <- update` into a sketch vector associated
            // with row_idx.
            sketch.insert(col_idx, update);
            // insert `(row_idx, col_idx) <- update' into all matrix
            // sketches.
            for (double_sketch_type &sketch : double_sketches) {
              sketch.insert({row_idx, col_idx}, update);
            }
          };
          sketch_map.async_visit(row_idx, insert_lambda, col_idx,
                                 matrix_A(row_idx, col_idx));
        }
      }
    }
    world.barrier();

    // We produce scaled embeddings for the matrix sketches and allreduce them
    // to put all updates in one place.
    std::vector<Eigen::MatrixXd> double_matrices;
    for (int i(0); i < double_sketches.size(); ++i) {
      // using a const reference to avoid an extra copy
      const Eigen::MatrixXd &double_matrix =
          double_sketches[i].container().registers();
      // it is very important that the dummy matrix have the the correct shapes!
      double_matrices.push_back(
          Eigen::MatrixXd::Zero(embedding_size, embedding_size));
      YGM_ASSERT_MPI(MPI_Allreduce(
          double_matrix.data(), double_matrices[i].data(),
          embedding_size * embedding_size, ygm::detail::mpi_typeof(double()),
          MPI_SUM, world.get_mpi_comm()));
      world.barrier();
    }

    // apply scaling factor
    for (Eigen::MatrixXd &double_matrix : double_matrices) {
      double_matrix /= double_transform_type::scaling_factor;
    }

    // Replicate workflow in serial to consistency check
    std::vector<double_sketch_type> double_sketches_check;
    for (const double_transform_ptr_type &double_transform_ptr :
         double_transform_ptrs) {
      double_sketches_check.emplace_back(double_transform_ptr);
    }
    for (int i(0); i < row_count; ++i) {
      for (int j(0); j < col_count; ++j) {
        for (double_sketch_type &double_sketch : double_sketches_check) {
          double_sketch.insert({i, j}, matrix_A(i, j));
        }
      }
    }
    std::vector<Eigen::MatrixXd> double_matrices_check;
    for (const double_sketch_type &double_sketch : double_sketches_check) {
      double_matrices_check.push_back(double_sketch.scaled_registers());
    }

    for (int i(0); i < double_matrices.size(); ++i) {
      world.cout0("transform ", i, " matches? ",
                  krowkee::sketch::detail::allclose(double_matrices[i],
                                                    double_matrices_check[i]),
                  ", max_error: ",
                  krowkee::sketch::detail::max_absolute_error(
                      double_matrices[i], double_matrices_check[i]),
                  ", mae: ",
                  krowkee::sketch::detail::mean_absolute_error(
                      double_matrices[i], double_matrices_check[i]));
    }
    world.cout0("");

    // multiply the (small) matrix sketches in-place so that everyone can access
    // the power iteration linear operator.
    Eigen::MatrixXd partial_product_sketch = double_matrices[0];
    for (int i(1); i < double_matrices.size(); ++i) {
      partial_product_sketch *= double_matrices[i];
    }

    Eigen::MatrixXd partial_product_sketch_check = double_matrices_check[0];
    for (int i(1); i < double_matrices_check.size(); ++i) {
      partial_product_sketch_check *= double_matrices_check[i];
    }

    world.cout0("partial product matches? ",
                krowkee::sketch::detail::allclose(partial_product_sketch,
                                                  partial_product_sketch_check),
                ", max_error: ",
                krowkee::sketch::detail::max_absolute_error(
                    partial_product_sketch, partial_product_sketch_check),
                ", mae: ",
                krowkee::sketch::detail::mean_absolute_error(
                    partial_product_sketch, partial_product_sketch_check));
    world.cout0("");

    // We prepare a datastructure to hold the scaled embeddings once all
    // sketches have accumulated. This is currently required, but the
    // sketches may perform scaling on insertion in a future version, in
    // which case this step is no longer necessary.
    ygm::container::map<int, Eigen::VectorXd> embeddings(world);

    sketch_map.for_all([&embeddings, &partial_product_sketch](
                           const int idx, const sketch_type &sketch) {
      Eigen::VectorXd embedding(sketch.size());
      auto            scaled_registers = sketch.scaled_registers();
      for (std::size_t i(0); i < scaled_registers.size(); ++i) {
        embedding(static_cast<Eigen::Index>(i)) =
            static_cast<double>(scaled_registers[i]);
      }
      // if (idx == 0) {
      //   std::cout << "embedding before: " << std::endl
      //             << embedding << std::endl;
      // }
      embedding *= partial_product_sketch;
      // if (idx == 0) {
      //   std::cout << "embedding after: " << std::endl << embedding <<
      //   std::endl;
      // }
      embeddings.async_insert(idx, embedding);
    });
    world.barrier();

    // Replicate one sided sketch in serial for consistency.
    std::vector<sketch_type> sketch_check(row_count, default_sketch);
    for (int i(0); i < row_count; ++i) {
      for (int j(0); j < col_count; ++j) {
        sketch_check[i].insert(j, matrix_A(i, j));
      }
    }

    bool   one_sided_match = true;
    double one_sided_mae(0.0);
    double one_sided_max_error(0.0);
    sketch_map.for_all(
        [&world, &sketch_check, &one_sided_match, &one_sided_mae,
         &one_sided_max_error](const int idx, sketch_type &sketch) {
          double this_mae =
              mean_absolute_error(sketch.container().registers(),
                                  sketch_check[idx].container().registers());
          if (this_mae >= 1e-5) {
            one_sided_match = false;
          }
          one_sided_mae += this_mae;
          one_sided_max_error = std::max(this_mae, one_sided_max_error);
        });
    world.barrier();

    one_sided_match     = ygm::min(one_sided_match, world);
    one_sided_mae       = ygm::sum(one_sided_mae, world) / sketch_map.size();
    one_sided_max_error = ygm::max(one_sided_max_error, world);

    world.cout0("One sided sketches match? ", one_sided_match,
                ", mae: ", one_sided_mae, ", max_error: ", one_sided_max_error);
    world.cout0("");

    // replicate embedding production for consistency.
    std::vector<Eigen::VectorXd> embeddings_check;
    for (int i(0); i < row_count; ++i) {
      std::vector<register_type> sketch = sketch_check[i].scaled_registers();
      Eigen::VectorXd embedding         = Eigen::VectorXd::Zero(sketch.size());
      for (int j(0); j < sketch.size(); ++j) {
        embedding[j] = sketch[j];
      }
      embedding *= partial_product_sketch_check;
      embeddings_check.push_back(embedding);
    }

    bool   embeddings_match = true;
    double embeddings_mae(0.0);
    double embeddings_max_error(0.0);
    embeddings.for_all(
        [&world, &embeddings_check, &embeddings_match, &embeddings_mae,
         &embeddings_max_error](const int idx, Eigen::VectorXd &embedding) {
          // if (idx == 0) {
          //   std::cout << "embedding: " << std::endl << embedding <<
          //   std::endl; std::cout << "check: " << std::endl
          //             << embeddings_check[idx] << std::endl;
          // }
          double this_mae = (embedding - embeddings_check[idx]).lpNorm<1>();
          if (this_mae >= 1e-5) {
            embeddings_match = false;
          }
          embeddings_mae += this_mae;
          embeddings_max_error = std::max(this_mae, embeddings_max_error);
        });
    world.barrier();

    embeddings_match     = ygm::min(embeddings_match, world);
    embeddings_mae       = ygm::sum(embeddings_mae, world) / embeddings.size();
    embeddings_max_error = ygm::max(embeddings_max_error, world);

    world.cout0("Embedding sketches match? ", embeddings_match,
                ", mae: ", embeddings_mae,
                ", max_error: ", embeddings_max_error);
    world.cout0("");

    // We now verify that the embedding vectors satisfy the guarantees
    double              success_rate_streaming_check(0.0);
    double              epsilon_streaming_check(0.0);
    const static double epsilon_expected =
        std::sqrt(16 * std::log(col_count) * srank / (range_size * range_size));
    int trials_check(0);

    for (int i(0); i < row_count; ++i) {
      for (int j(0); j < i; ++j) {
        ++trials_check;
        double dist_exact =
            (product_exact.row(i) - product_exact.row(j)).lpNorm<2>();
        double dist_streaming =
            (embeddings_check[i] - embeddings_check[j]).lpNorm<2>();
        double error_streaming = std::abs(1.0 - dist_streaming / dist_exact);
        epsilon_streaming_check += error_streaming;
        if (in_bounds(dist_exact, dist_streaming, epsilon_expected)) {
          success_rate_streaming_check += 1.0;
        }
      }
    }

    success_rate_streaming_check /= trials_check;
    epsilon_streaming_check /= trials_check;

    world.cout0("local power iteration approximate row distances guarantee (",
                trials_check, " trials)");
    world.cout0("\tstreaming success rate / epsilon / expected = (",
                success_rate_streaming_check, ", ", epsilon_streaming_check,
                ", ", epsilon_expected, ")");
    world.cout0("");

    // We now compare the embedding vectors. In practice this could be done more
    // efficiently, but this implementation suffices for illustration.
    static double success_rate_streaming(0.0);
    static double epsilon_streaming(0.0);
    // const static double epsilon_expected =
    //     std::sqrt(16 * std::log(col_count) * srank / (range_size *
    //     range_size));
    static int trials(0);

    embeddings.for_all([&embeddings](const int              lhs_idx,
                                     const Eigen::VectorXd &lhs_embedding) {
      for (int rhs_idx(lhs_idx + 1); rhs_idx < row_count; ++rhs_idx) {
        embeddings.async_visit(
            rhs_idx,
            [](const int rhs_idx, const Eigen::VectorXd &rhs_embedding,
               const int lhs_idx, const Eigen::VectorXd &lhs_embedding) {
              ++trials;
              double dist_exact =
                  (product_exact.row(lhs_idx) - product_exact.row(rhs_idx))
                      .lpNorm<2>();
              double dist_streaming =
                  (lhs_embedding - rhs_embedding).lpNorm<2>();
              double error_streaming =
                  std::abs(1.0 - dist_streaming / dist_exact);
              epsilon_streaming += error_streaming;
              if (in_bounds(dist_exact, dist_streaming, epsilon_expected)) {
                success_rate_streaming += 1.0;
              }
              if (lhs_idx == 199 && rhs_idx == 230) {
                // std::cout << "\tlhs_embedding:" << std::endl;
                // std::cout << lhs_embedding << std::endl;
                // std::cout << "\trhs_embedding:" << std::endl;
                // std::cout << rhs_embedding << std::endl;
                std::cout << "\t(" << lhs_idx << "," << rhs_idx << ") exact "
                          << dist_exact
                          << "\n\t\tstreaming (dist/error/success): ("
                          << dist_streaming << ", 1 +/- " << error_streaming
                          << ", "
                          << in_bounds(dist_exact, dist_streaming,
                                       epsilon_expected)
                          // << ")\n\t\titerative (dist/error/success): ("
                          // << dist_iterative << ", 1 +/- " << error_iterative
                          // << ", "
                          // << in_bounds(dist_exact, dist_iterative,
                          //              epsilon_expected)
                          << ")" << std::endl;
              }
            },
            lhs_idx, lhs_embedding);
      }
    });
    world.barrier();

    success_rate_streaming = ygm::sum(success_rate_streaming, world);
    epsilon_streaming      = ygm::sum(epsilon_streaming, world);
    trials                 = ygm::sum(trials, world);
    success_rate_streaming /= trials;
    epsilon_streaming /= trials;

    // world.cout0("\npower iteration approximate row distances guarantee (",
    //             trials, " trials), success rate = ", success_rate_streaming,
    //             ", expected epsilon = ", epsilon_expected,
    //             ", mean empirical epsilon = ", epsilon_streaming);
    world.cout0(
        "\ndistributed power iteration approximate row distances guarantee (",
        trials, " trials)");
    world.cout0("\tstreaming success rate / epsilon / expected = (",
                success_rate_streaming, ", ", epsilon_streaming, ", ",
                epsilon_expected, ")");
  }

  return 0;
}