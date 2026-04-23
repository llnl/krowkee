// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <krowkee/sketch.hpp>

#include <cassert>
#include <iostream>
#include <random>

// We are using floats as our feature type in this example.
using register_type = float;

// This function computes squared l2 distance, and will be used to verify the
// guarantees of the Johnson-Lindenstrauss lemma.
template <typename T>
double l2_distance_sq(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  assert(lhs.size() == rhs.size());
  double dist_sq(0);
  for (int i(0); i < lhs.size(); ++i) {
    dist_sq += std::pow(lhs[i] - rhs[i], 2);
  }
  return dist_sq;
}

double l2_distance_sq(const Eigen::VectorXf &lhs, const Eigen::VectorXf &rhs) {
  assert(lhs.size() == rhs.size());
  return (lhs - rhs).squaredNorm();
}

int main(int argc, char **argv) {
  uint64_t      stream_size(20000);
  std::uint64_t domain_size(16384);
  std::uint64_t observation_count(8);
  std::uint64_t seed(krowkee::hash::default_seed);
  bool          verbose(true);

  std::cout << std::endl;
  std::cout << "This example realizes a pairwise relative l2 distance error "
               "workflow. "
            << observation_count
            << " data streams defining feature vectors containing "
            << domain_size
            << " dimensions are sampled, and those data streams are used to "
               "accumulate "
               "corresponding sparse Johnson-Lindenstrauss transform sketches. "
               "We then "
               "demonstrate that these sketches approximately preserve the "
               "pairwise l2 "
               "distances between the original feature vectors."
            << std::endl;

  // ---------------------------------------------------------------------------
  // data preparation
  // ---------------------------------------------------------------------------

  // First we sample `observation_count` data streams, each of which contain a
  // stream of +1 updates to the corresponding indices of a `domain_size` vector
  // of observables. In this case, each data stream has `stream_size` updates.
  // In practice one would obtain these updates from a true data stream (file,
  // generator, sensor, or other).");
  std::mt19937                                 gen(seed);
  std::uniform_int_distribution<std::uint64_t> dist(0, domain_size - 1);
  std::vector<std::vector<std::uint64_t>>      streams(
      observation_count, std::vector<std::uint64_t>(stream_size));
  for (std::int64_t i(0); i < observation_count; ++i) {
    for (std::int64_t j(0); j < stream_size; ++j) {
      streams[i][j] = dist(gen);
    }
  }

  // We then consolidate the data streams in `observation_count` observations
  // vectors, each containing `domain_size` dimensions.
  std::vector<std::vector<register_type>> observations(
      observation_count, std::vector<register_type>(domain_size));
  for (int i(0); i < observation_count; ++i) {
    for (int j(0); j < stream_size; ++j) {
      observations[i][streams[i][j]]++;
    }
  }

  // ---------------------------------------------------------------------------
  // sketch accumulation
  // ---------------------------------------------------------------------------

  // Using krowkee requires the selection of a sketch type, here encapsulated as
  // `sketch_type`. We use the `SparseJLT` type defined in the simple API in
  // `krowkee/sketch.hpp`. This type has four template parameters:
  //   1. the numeric type to be used by each register (here `float`),
  //   2. a `std::size_t` parameter `range_size` indicating the number of
  //      registers used by each instance of the internal transform,
  //   3. a `std::size_t` parameter `replication_count` indicating the number of
  //      instances of the transform to be used, and
  //   4. a shared pointer type to be used by the shared transform object
  //      (`std::shared_ptr` for shared memory implementations).
  constexpr const std::size_t range_size        = 8;
  constexpr const std::size_t replication_count = 3;
  using sketch_type =
      krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                                 std::shared_ptr>;
  using registers_type = typename sketch_type::registers_type;

  // Most krowkee classes support `name()` and `full_name()` static functions
  // that print their implementation details specified by template choices.
  std::cout << std::endl;
  std::cout << "This sketch has the following name and full name:" << std::endl;
  std::cout << sketch_type::name() << std::endl;
  std::cout << sketch_type::full_name() << std::endl;

  // Having established a sketch type, the first step to use a krowkee sketch is
  // to create a shared pointer to a transform functor. The sketch type includes
  // typedefs of the transform and pointer types. This is where the random seed
  // is used. Transforms of the same type sharing the same seed will behave
  // identically. As this is a shared-memory code, we create a std::shared_ptr
  // of the transform to be used to define the sketch data structures.
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  transform_ptr_type transform_ptr(std::make_shared<transform_type>(seed));

  // Once we have a pointer to a sketch transform, we use it in the constructor
  // for every sketch that needs to share its random projection behavior. If we
  // aim to embed a collection of vectors, this means that we use the same
  // transform for each corresponding sketch object. We create a sketch for each
  // feature vector and insert all elements of the corresponding data stream.
  std::vector<sketch_type> sketches(observation_count,
                                    sketch_type(transform_ptr));
  for (int i(0); i < observation_count; ++i) {
    // for each synthetic data stream, we insert all elements into the
    // appropriate sketch.
    for (int j(0); j < stream_size; ++j) {
      sketches[i].insert(streams[i][j]);
    }
  }

  std::cout << std::endl;
  std::cout
      << "These are the (index, register) pairs resulting from each sketch:"
      << std::endl;
  for (int i(0); i < sketches.size(); ++i) {
    std::cout << "(" << i << ")\t" << sketches[i] << std::endl;
  }

  // Once the sketch accumulation is finished, we apply a scaling factor to each
  // register. In principle, this step could be avoided by interally dividing
  // each update by the scaling factor, which is known a priori. However, this
  // can cause floating point discrepancies between sketches obtained by merges
  // versus those obtained by concatenating the corresponding streams together,
  // especially for dense transforms such as the FWHT. As these two cases should
  // be mathematically identical, this step has been separated out in the
  // current version. This means that the embedding vectors must be recomputed
  // if the sketches receive further updates. As the sketch dimensionality is
  // assumed to be low, this is likely not a large computational problem. It
  // does, however, add a step to any implementation.
  std::vector<registers_type> embeddings;
  for (int i(0); i < observation_count; ++i) {
    embeddings.push_back(sketches[i].scaled_registers());
  }

  std::cout << std::endl;
  std::cout << "We print the scaled embedding vectors resulting from each "
               "sketch using the (inverse) scaling factor "
            << transform_type::scaling_factor << ":" << std::endl;
  for (int i(0); i < observation_count; ++i) {
    std::cout << "(" << i << ")\t" << embeddings[i] << std::endl;
  }

  // ---------------------------------------------------------------------------
  // embedding evaluation
  // ---------------------------------------------------------------------------

  // In practice, we would now use the embeddings for the application of
  // interest, such as the feature vectors for a metric or kernel algorithm. In
  // this case, we will verify that the pairwise l2 distances are approximately
  // preserved up to the Monte Carlo guarantees of the Johnson-Lindenstrauss
  // lemma.

  // First, we create a target approximation factor. A function of the
  // `range_size` parameter, we expect that most embedded distances will be
  // within this multiplicative factor of their corresponding distances in the
  // original feature space.
  double target_approximation_factor =
      std::sqrt(8 * std::log(observation_count) / (range_size * range_size));
  std::cout << std::endl;
  std::cout << "Our desired multiplicative approximation factor is 1 +/- "
            << target_approximation_factor << "." << std::endl;

  // Here we loop over each observation, record its empirical multiplicative
  // error, and compare it to the target. We record the mean empirical error and
  // the rate at which the empirical error is smaller than the target. The
  // success rate for a fixed `range_size` can be improved by increasing the
  // `replication_count` at the expense of multiplicatively increasing the
  // memory required by each sketch.
  std::cout << std::endl;
  std::cout << "Empirical approximation measurements:" << std::endl;
  double success_rate(0.0);
  double mean_approximation_factor{0.0};
  int    trials(0);
  for (int i(0); i < observation_count; ++i) {
    for (int j(0); j < i; ++j) {
      ++trials;
      double observed_distance =
          l2_distance_sq(observations[i], observations[j]);
      double embedded_distance = l2_distance_sq(embeddings[i], embeddings[j]);
      double error             = 1 - embedded_distance / observed_distance;
      double abs_error         = std::abs(error);
      mean_approximation_factor += abs_error;
      bool in_bounds = abs_error < target_approximation_factor;
      bool positive  = error > 0;

      if (in_bounds) {
        success_rate += 1.0;
      }
      if (verbose) {
        std::cout << "\t(" << i << "," << j << ") distance: observed "
                  << observed_distance << " versus embedded "
                  << embedded_distance << " (multiplicative error factor: 1 "
                  << ((positive) ? "+ " : "- ") << abs_error
                  << ") (success: " << ((in_bounds) ? "true" : "false") << ")"
                  << std::endl;
      }
    }
  }
  success_rate /= trials;
  mean_approximation_factor /= trials;
  bool lemma_guarantee_success = success_rate > 0.5;

  std::cout << std::endl;
  std::cout << "We find an empirical success rate of " << success_rate
            << " and a mean emprical approximation factor of 1 +/- "
            << mean_approximation_factor << std::endl;

  return 0;
}