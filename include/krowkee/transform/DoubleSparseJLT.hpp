// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/hash.hpp>
#include <krowkee/transform/Element.hpp>

#include <sstream>
#include <vector>

namespace krowkee {
namespace transform {

using krowkee::stream::Element;

/**
 * @brief A functor implementing a CountSketch-based sparse JLT on each side of
 * a matrix of registers.
 *
 * Implements CountSketch using a `ReplicationCount` number of pairs of hash
 * functions.
 *
 * [0] M. Charikar, K. Chen, M. Farach-Colton. Finding frequent items in data
 * streams. Theoretical Computer Science. 2004.
 * https://edoliberty.github.io/datamining2011aFiles/FindingFrequentItemsInDataStreams.pdf
 *
 * [1] K. Clarkson, D. Woodruff. Low-rank approximation in input sparsity time.
 * Journal of the ACM. 2017.
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.746.6409&rep=rep1&type=pdf
 *
 * @tparam RegType The type of register over which the functor operates
 * @tparam HashType The hash functor type to use to define CountSketch random
 * mappings.
 * @tparam RangeSize The power-of-two embedding dimension.
 * @tparam ReplicationCount The number of replicated CountSketch transforms to
 * use.
 */
template <typename RegType, template <std::size_t> class HashType,
          std::size_t RangeSize, std::size_t ReplicationCount>
class DoubleSparseJLT {
 public:
  using register_type = RegType;
  using hash_type     = HashType<RangeSize>;
  using self_type =
      DoubleSparseJLT<register_type, HashType, RangeSize, ReplicationCount>;

 private:
  std::vector<hash_type> _row_hashes;
  std::vector<hash_type> _col_hashes;

 public:
  /**
   * @brief Construct a new DoubleSparseJLT Functor object by initializing hash
   * functors.
   *
   * Depending on the hash functor to be used, the effective embedding dimension
   * (returned by `this->size()`) may be rounded up to the next power of two.
   * Further, we are assuming no redundancy. Each insert is only hashed to one
   * register location.
   *
   * @note This behavior may change in the future.
   *
   * @tparam Args type(s) of additional hash parameters
   * @param seed The random seed.
   * @param args Any additional parameters required by the hash functions.
   */
  template <typename... Args>
  DoubleSparseJLT(std::uint64_t seed, const Args &...args) {
    _row_hashes.reserve(ReplicationCount);
    _col_hashes.reserve(ReplicationCount);
    for (int i(0); i < ReplicationCount; ++i) {
      _row_hashes.emplace_back(seed, args...);
      seed = krowkee::hash::wang64(seed);
      _col_hashes.emplace_back(seed, args...);
      seed = krowkee::hash::wang64(seed);
    }
  }

  DoubleSparseJLT() {}

  //////////////////////////////////////////////////////////////////////////////
  // Cereal Archives
  //////////////////////////////////////////////////////////////////////////////

#if __has_include(<cereal/cereal.hpp>)
  /**
   * @brief Serialize DoubleSparseJLT object to/from `cereal` archive.
   *
   * @tparam Archive `cereal` archive type.
   * @param archive The `cereal` archive to which to serialize the transform.
   */
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_row_hashes);
    archive(_col_hashes);
  }
#endif

  //////////////////////////////////////////////////////////////////////////////
  // Function: Apply to Container
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Update a register container with an observation.
   *
   * @tparam ContainerType The type of the register data structure.
   * @tparam ItemArgs Types of stream item observables.
   * @param registers The register container.
   * @param item_args Stream item observables.
   */
  template <typename ContainerType, typename... ItemArgs>
  constexpr void operator()(ContainerType &registers,
                            const ItemArgs &...item_args) const {
    static_assert(std::is_same<register_type,
                               typename ContainerType::register_type>::value);
    using merge_type = typename ContainerType::merge_type;
    const Element<register_type> stream_element(item_args...);
    std::vector<std::size_t>     row_indices(ReplicationCount);
    std::vector<std::size_t>     col_indices(ReplicationCount);
    std::vector<int>             row_polarities(ReplicationCount);
    std::vector<int>             col_polarities(ReplicationCount);
    for (int i(0); i < ReplicationCount; ++i) {
      auto [row_index, row_polarity] = _row_hashes[i](stream_element.item);
      auto [col_index, col_polarity] =
          _col_hashes[i](stream_element.identifier);
      row_index += i * range_size();
      col_index += i * range_size();
      row_indices[i]    = row_index;
      col_indices[i]    = col_index;
      row_polarities[i] = row_polarity;
      col_polarities[i] = col_polarity;
    }
    for (int i(0); i < ReplicationCount; ++i) {
      for (int j(0); j < ReplicationCount; ++j) {
        register_type &reg      = registers[row_indices[i]][col_indices[j]];
        auto           polarity = row_polarities[i] * col_polarities[j];
        merge_type()(reg, polarity * stream_element.multiplicity);
        if (reg == 0) {
          registers.erase(row_indices[i], col_indices[j]);
        }
      }
    }
  }

 public:
  //////////////////////////////////////////////////////////////////////////////
  // Getters
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Get the maximum number of range values returnable by the register
   * hash function.
   *
   * This is equivalent to the number of registers in each replicated tile in
   * passed containers.
   *
   * @return constexpr std::size_t The range size.
   */
  static constexpr std::size_t range_size() { return hash_type::size(); }

  /**
   * @brief Get the number of replicated CountSketch transforms.
   *
   * @return constexpr std::size_t The replication count.
   */
  static constexpr std::size_t replication_count() { return ReplicationCount; }

  /**
   * @brief Get the scaling factor to be used for projections.
   *
   * @return constexpr std::size_t The replication count.
   */
  static constexpr RegType scaling_factor =
      std::sqrt((RegType)replication_count());

  /**
   * @brief Get the total number of addressable registers across all hash
   * functions.
   *
   * This is equivalent to the range size times the number of replicated tiles.
   *
   * @return constexpr std::size_t The range size.
   */
  static constexpr std::size_t size() {
    return range_size() * replication_count();
  }

  /** Get the random seed. */
  constexpr std::uint64_t seed() const { return _hashes[0].seed(); }

  /**
   * @brief Return a description of the transform type.
   *
   * @return std::string "DoubleSparseJLT"
   */
  static constexpr std::string name() { return "DoubleSparseJLT"; }

  /**
   * @brief Return a description of the fully-qualified transform type.
   *
   * @return std::string Transform description, e.g. "DoubleSparseJLT using
   * MulAddShift hashes and 4 byte registers"
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << name() << " using two " << ReplicationCount << " replications of "
       << hash_type::full_name() << " and " << sizeof(register_type)
       << "-byte registers";
    return ss.str();
  }

  /**
   * @brief Check for equality between two DoubleSparseJLTs.
   *
   * @param lhs The left-hand functor.
   * @param rhs The right-hand functor.
   * @return true The seeds and range sizes agree.
   * @return false The seeds or range sizes disagree.
   */
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    for (int i(0); i < ReplicationCount; ++i) {
      if (lhs._row_hashes[i] != rhs._row_hashes[i]) {
        return false;
      }
      if (lhs._col_hashes[i] != rhs._col_hashes[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Check for inequality between two DoubleSparseJLTs.
   *
   * @param lhs The left-hand functor.
   * @param rhs The right-hand functor.
   * @return true The seeds or range sizes disagree.
   * @return false The seeds and range sizes agree.
   */
  friend constexpr bool operator!=(const self_type &lhs, const self_type &rhs) {
    return !operator==(lhs, rhs);
  }

  /**
   * @brief Serialize a transform to human-readable output stream.
   *
   * Prints the space-separated range size and seed.
   *
   * @param os The output stream.
   * @param func The functor object.
   * @return std::ostream& The new stream state.
   */
  friend std::ostream &operator<<(std::ostream &os, const self_type &func) {
    os << func.range_size() << " " << func.replication_count() << " "
       << func.seed();
    return os;
  }
};

}  // namespace transform
}  // namespace krowkee
