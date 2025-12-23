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

float spectral_norm(const Eigen::MatrixXf &matrix) {
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(
      matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.singularValues()(0);
}

float stable_rank(const Eigen::MatrixXf &matrix) {
  return matrix.squaredNorm() / std::pow(spectral_norm(matrix), 2);
}

bool in_bounds(const double tru, const double est, const double eps) {
  return (est < (1 + eps) * tru) && (est > (1 - eps) * tru);
}

int main(int argc, char **argv) {
  // We create the YGM communicator to be used. The example proceeds similarly
  // to `examples/sparse_jlt.cpp`, where we sample some large-dimensional data
  // on each rank, accumulate sketches, and insert those sketches into a
  // ygm::container::map.
  ygm::comm world(&argc, &argv);
  {
    const std::size_t row_count = 256;
    const std::size_t col_count = row_count;
    std::uint64_t     seed(4);
    bool              verbose(true);

    // Using krowkee requires the selection of a sketch type for both a single
    // and double-sided sketch, here encapsulated as `sketch_type` and
    // `double_sketch_type`, respectively. We use the `SparseJLT` and
    // `DoubleSparseJLT` types defined in the simple API in
    // `krowkee/sketch.hpp`. These types have four template parameters:
    //   1. the numeric type to be used by each register (here `float`),
    //   2. a `std::size_t` parameter `range_size` indicating the number of
    //      registers used by each instance of the internal transform,
    //   3. a `std::size_t` parameter `replication_count` indicating the number
    //      of instances of the transform to be used, and
    //   4. a shared pointer type to be used by the shared transform object
    //      (`ygm::ygm_ptr` for shared memory implementations).
    constexpr const std::size_t range_size        = 32;
    constexpr const std::size_t replication_count = 4;
    using register_type                           = float;
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
    single_transform_type     transform_S{seed};
    single_transform_type     transform_R{seed};
    single_transform_type     transform_Q{seed};
    single_transform_ptr_type transform_S_ptr(world.make_ygm_ptr(transform_S));
    single_transform_ptr_type transform_R_ptr(world.make_ygm_ptr(transform_R));
    single_transform_ptr_type transform_Q_ptr(world.make_ygm_ptr(transform_Q));
    double_transform_type     transform_SR(transform_S_ptr, transform_R_ptr);
    double_transform_type     transform_RQ(transform_S_ptr, transform_R_ptr);
    double_transform_ptr_type transform_SR_ptr(
        world.make_ygm_ptr(transform_SR));
    double_transform_ptr_type transform_RQ_ptr(
        world.make_ygm_ptr(transform_RQ));

    // We create a `ygm::container::map` that will hold the AS embeddings for
    // each of our sketches. We also create an empty sketch using our transform
    // as the default value.
    sketch_type                           default_sketch(transform_S_ptr);
    ygm::container::map<int, sketch_type> sketch_map_AS(world, default_sketch);

    // We also create local double-sided sketches that will hold the double
    // sided embeddings.
    double_sketch_type sketch_StAR(transform_SR_ptr);
    double_sketch_type sketch_RtAQ(transform_RQ_ptr);
    using double_sketch_ptr_type           = ygm::ygm_ptr<double_sketch_type>;
    double_sketch_ptr_type sketch_StAR_ptr = world.make_ygm_ptr(sketch_StAR);
    double_sketch_ptr_type sketch_RtAQ_ptr = world.make_ygm_ptr(sketch_RtAQ);

    // We sample a random matrix to embed. Note that this is not implemented
    // efficiently, as this is a toy example and we will compute a ground
    // truth solution that is intractable for large matrices.
    static Eigen::MatrixXf matrix_A =
        Eigen::MatrixXf::Random(row_count, col_count);

    // We will now simulate asynchronously updates to the associated sketch by
    // communicating elements of the matrix using YGM. In this example each
    // stream is generated by a particular rank's slicing from a shared matrix,
    // but in practice we assume that updates to any element could appear on any
    // rank.
    for (int col_idx(0); col_idx < col_count; ++col_idx) {
      if (world.rank() == col_idx % world.size()) {
        for (int row_idx(0); row_idx < row_count; ++row_idx) {
          // This remote lambda will be executed for each stream update and will
          // go to the appropriate single sketch on the appropriate rank,
          // simultaneously updating the two-sided sketches.
          auto insert_lambda = [](const int &row_idx, sketch_type &sketch,
                                  const int &col_idx, const float update,
                                  auto SR_ptr, auto RQ_ptr) {
            // insert `(col_idx) <- update` into a sketch vector associated
            // with row_idx.
            sketch.insert(col_idx, update);
            // insert `(row_idx, col_idx) <- update' into both matrix
            // sketches.
            SR_ptr->insert({row_idx, col_idx}, update);
            RQ_ptr->insert({row_idx, col_idx}, update);
          };
          sketch_map_AS.async_visit(row_idx, insert_lambda, col_idx,
                                    matrix_A(row_idx, col_idx), sketch_StAR_ptr,
                                    sketch_RtAQ_ptr);
        }
      }
    }
    world.barrier();

    // We produce scaled embeddings for the matrix sketches and allreduce them
    // to put all updates in one place.
    Eigen::MatrixXf matrix_StAR_pre = sketch_StAR.scaled_registers();
    Eigen::MatrixXf matrix_RtAQ_pre = sketch_RtAQ.scaled_registers();

    Eigen::MatrixXf matrix_StAR = Eigen::MatrixXf::Zero(row_count, col_count);
    Eigen::MatrixXf matrix_RtAQ = Eigen::MatrixXf::Zero(row_count, col_count);

    YGM_ASSERT_MPI(MPI_Allreduce(
        matrix_StAR_pre.data(), matrix_StAR.data(), row_count * col_count,
        ygm::detail::mpi_typeof(float()), MPI_SUM, world.get_mpi_comm()));
    world.barrier();
    YGM_ASSERT_MPI(MPI_Allreduce(
        matrix_RtAQ_pre.data(), matrix_RtAQ.data(), row_count * col_count,
        ygm::detail::mpi_typeof(float()), MPI_SUM, world.get_mpi_comm()));
    world.barrier();

    // multiply the (small) matrix sketches in-place so that everyone can access
    // the power iteration linear operator.
    Eigen::MatrixXf matrix_StARRtAQ = matrix_StAR * matrix_RtAQ;

    // We prepare a datastructure to hold the scaled embeddings once all
    // sketches have accumulated. This is currently required, but the sketches
    // may perform scaling on insertion in a future version, in which case this
    // step is no longer necessary.
    ygm::container::map<int, Eigen::VectorXf> embeddings(world);

    sketch_map_AS.for_all([&embeddings, &matrix_StARRtAQ, &matrix_StAR](
                              const int idx, const sketch_type &sketch) {
      Eigen::VectorXf embedding(sketch.size());
      auto            scaled_registers = sketch.scaled_registers();
      for (std::size_t i(0); i < scaled_registers.size(); ++i) {
        embedding(static_cast<Eigen::Index>(i)) =
            static_cast<float>(scaled_registers[i]);
      }
      embedding *= matrix_StARRtAQ;
      embeddings.async_insert(idx, embedding);
    });
    world.barrier();

    // We compute the ground truth power iteration of the matrix.

    static Eigen::MatrixXf matrix_AAA = matrix_A * matrix_A * matrix_A;
    float                  srank      = stable_rank(matrix_AAA);

    // We now compare the embedding vectors. In practice this could be done more
    // efficiently, but this implementation suffices for illustration.
    static double       success_rate(0.0);
    static double       empirical_epsilon(0.0);
    const static double expected_epsilon =
        std::sqrt(16 * std::log(col_count) * srank / (range_size * range_size));
    static int trials(0);

    embeddings.for_all([&embeddings](const int              lhs_idx,
                                     const Eigen::VectorXf &lhs_embedding) {
      for (int rhs_idx(lhs_idx + 1); rhs_idx < row_count; ++rhs_idx) {
        embeddings.async_visit(
            rhs_idx,
            [](const int rhs_idx, const Eigen::VectorXf &rhs_embedding,
               const int lhs_idx, const Eigen::VectorXf &lhs_embedding) {
              ++trials;
              double exact_dist =
                  (matrix_AAA.row(lhs_idx) - matrix_AAA.row(rhs_idx))
                      .lpNorm<2>();
              double sketch_dist = (lhs_embedding - rhs_embedding).lpNorm<2>();
              double this_error  = std::abs(1.0 - sketch_dist / exact_dist);
              empirical_epsilon += this_error;
              if (in_bounds(exact_dist, sketch_dist, expected_epsilon)) {
                success_rate += 1.0;
              }
              std::cout << "\t(" << lhs_idx << "," << rhs_idx << ") exact "
                        << exact_dist << ", sketched " << sketch_dist
                        << " (multiplicative error: 1 +/- " << this_error
                        << ") (in bounds: "
                        << in_bounds(exact_dist, sketch_dist, expected_epsilon)
                        << ")" << std::endl;
            },
            lhs_idx, lhs_embedding);
      }
    });
    world.barrier();

    success_rate      = ygm::sum(success_rate, world);
    empirical_epsilon = ygm::sum(empirical_epsilon, world);
    trials            = ygm::sum(trials, world);
    success_rate /= trials;
    empirical_epsilon /= trials;

    world.cout0("\npower iteration approximate row distances guarantee (",
                trials, "trials), success rate = ", success_rate,
                ", expected epsilon = ", expected_epsilon,
                ", mean empirical epsilon = ", empirical_epsilon);
  }

  return 0;
}