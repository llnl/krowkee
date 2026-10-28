// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/hyperloglog.hpp>
#include <krowkee/transform/Element.hpp>

#include <sstream>
#include <vector>

namespace krowkee {
namespace transform {

using krowkee::stream::Element;

/// good reference for operator declarations:
/// https://stackoverflow.com/questions/4421706/what-are-the-basic-rules-and-idioms-for-operator-overloading

/**
 * @brief A functor implementing a HyperLogLog-based cardinality sketch on a
 * collection of registers.
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
 * @tparam HashType The hash functor type to use to define HyperLogLog initial
 * zero counts.
 * @tparam RangeSize The power-of-two embedding dimension.
 * use.
 */
template <typename RegType, template <std::size_t> class HashType,
          std::size_t RangeSize>
class HyperLogLog {
 public:
  using register_type   = RegType;
  using hash_type       = HashType<RangeSize>;
  using index_type      = std::uint32_t;
  using zero_count_type = std::uint8_t;
  using update_type     = std::pair<index_type, zero_count_type>;
  using self_type       = HyperLogLog<register_type, HashType, RangeSize>;

 private:
  hash_type _hash;

 public:
  /**
   * @brief Construct a new HyperLogLog Functor object by initializing hash
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
  HyperLogLog(std::uint64_t seed, const Args &...args) : _hash(seed, args...) {
    // _hashes.reserve(ReplicationCount);
    // for (int i(0); i < ReplicationCount; ++i) {
    //   _hashes.emplace_back(seed, args...);
    //   seed = krowkee::hash::wang64(seed);
    // }
  }

  HyperLogLog() {}

  //////////////////////////////////////////////////////////////////////////////
  // Cereal Archives
  //////////////////////////////////////////////////////////////////////////////

#if __has_include(<cereal/cereal.hpp>)
  /**
   * @brief Serialize HyperLogLog object to/from `cereal` archive.
   *
   * @tparam Archive `cereal` archive type.
   * @param archive The `cereal` archive to which to serialize the transform.
   */
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_hash);
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
    auto [index, zero_count] = _hash(stream_element.item);
    register_type &reg       = registers[index];
    reg                      = merge_type()(reg, zero_count);
  }

  constexpr update_type apply(const std::uint64_t &idx) const {
    auto [index, zero_count] = _hash(idx);
    return {index, zero_count};
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
   * @brief Get the scaling factor to be used for projections.
   *
   * @return constexpr std::size_t The replication count.
   */
  static constexpr RegType scaling_factor = 1;

  /**
   * @brief Get the total number of addressable registers across all hash
   * functions.
   *
   * This is equivalent to the range size times the number of replicated tiles.
   *
   * @return constexpr std::size_t The range size.
   */
  static constexpr std::size_t size() { return range_size(); }

  /** Get the random seed. */
  constexpr std::uint64_t seed() const { return _hash.seed(); }

  /**
   * @brief Return a description of the transform type.
   *
   * @return std::string "HyperLogLog"
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "HyperLogLog<" << RangeSize << ", " << hash_type::name() << ">";
    return ss.str();
  }

  /**
   * @brief Return a description of the fully-qualified transform type.
   *
   * @return std::string Transform description, e.g. "HyperLogLog using
   * MulAddShift hashes and 4 byte registers"
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << "HyperLogLog<" << RangeSize << ", " << hash_type::full_name() << ", "
       << sizeof(register_type) << ">";
    return ss.str();
  }

  /**
   * @brief Check for equality between two HyperLogLogs.
   *
   * @param lhs The left-hand functor.
   * @param rhs The right-hand functor.
   * @return true The seeds and range sizes agree.
   * @return false The seeds or range sizes disagree.
   */
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    return lhs._hash == rhs._hash;
  }

  /**
   * @brief Check for inequality between two HyperLogLogs.
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
