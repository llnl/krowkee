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

/// good reference for operator declarations:
/// https://stackoverflow.com/questions/4421706/what-are-the-basic-rules-and-idioms-for-operator-overloading

/**
 * @brief A functor implementing a CountSketch-based sparse JLT on a collection
 * of registers.
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
class SparseJLT {
 public:
  using register_type = RegType;
  using hash_type     = HashType<RangeSize>;
  using self_type =
      SparseJLT<register_type, HashType, RangeSize, ReplicationCount>;

 private:
  std::vector<hash_type> _hashes;

 public:
  /**
   * @brief Construct a new SparseJLT Functor object by initializing hash
   * functors.
   *
   * Depending on the hash functor to be used, the effective embedding dimension
   * (returned by `this->size()`) may be rounded up to the next power of two.
   * Each insert is hashed to one register location per ReplicationCount
   * register replica.
   *
   * @tparam Args type(s) of additional hash parameters
   * @param seed The random seed.
   * @param args Any additional parameters required by the hash functions.
   */
  template <typename... Args>
  SparseJLT(std::uint64_t seed, const Args &...args) {
    _hashes.reserve(ReplicationCount);
    for (int i(0); i < ReplicationCount; ++i) {
      _hashes.emplace_back(seed, args...);
      seed = krowkee::hash::wang64(seed);
    }
  }

  SparseJLT() {}

  //////////////////////////////////////////////////////////////////////////////
  // Cereal Archives
  //////////////////////////////////////////////////////////////////////////////

#if __has_include(<cereal/cereal.hpp>)
  /**
   * @brief Serialize SparseJLT object to/from `cereal` archive.
   *
   * @tparam Archive `cereal` archive type.
   * @param archive The `cereal` archive to which to serialize the transform.
   */
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_hashes);
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
    for (int i(0); i < ReplicationCount; ++i) {
      auto [index, polarity] = _hashes[i](stream_element.item);
      index += i * range_size();
      register_type &reg = registers[index];
      reg = merge_type()(reg, polarity * stream_element.multiplicity);
      if (reg == 0) {
        registers.erase(index);
      }
    }
  }

  constexpr std::pair<std::vector<std::size_t>, std::vector<std::size_t>> apply(
      const Element<register_type> &element) const {
    std::pair<std::vector<std::size_t>, std::vector<std::size_t>> hashes = {
        std::vector<std::size_t>(ReplicationCount),
        std::vector<std::size_t>(ReplicationCount)};
    for (int i(0); i < ReplicationCount; ++i) {
      auto [index, polarity] = _hashes[i](element.item);
      index += i * range_size();
      hashes.first[i]  = index;
      hashes.second[i] = polarity;
    }
    return hashes;
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
   * @return std::string "SparseJLT"
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "SparseJLT<" << RangeSize << ", " << ReplicationCount << ", "
       << hash_type::name() << ">";
    return ss.str();
  }

  /**
   * @brief Return a description of the fully-qualified transform type.
   *
   * @return std::string Transform description, e.g. "SparseJLT using
   * MulAddShift hashes and 4 byte registers"
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << "SparseJLT<" << RangeSize << ", " << ReplicationCount << ", "
       << hash_type::full_name() << ", " << sizeof(register_type) << ">";
    return ss.str();
  }

  /**
   * @brief Check for equality between two SparseJLTs.
   *
   * @param lhs The left-hand functor.
   * @param rhs The right-hand functor.
   * @return true The seeds and range sizes agree.
   * @return false The seeds or range sizes disagree.
   */
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    for (int i(0); i < ReplicationCount; ++i) {
      if (lhs._hashes[i] != rhs._hashes[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Check for inequality between two SparseJLTs.
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
    os << func.range_size() << " " << func.seed();
    return os;
  }
};

}  // namespace transform
}  // namespace krowkee
