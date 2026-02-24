// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/hash.hpp>
#include <krowkee/transform/Element.hpp>
#include <krowkee/transform/SparseJLT.hpp>

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
 * functions. Currently assumes that both sides of the matrix are to be
 * projected to the same dimension.
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
 * @tparam PtrType The type of shared pointer used to wrap individual sketch
 * functors.
 * @tparam RangeSize The power-of-two embedding dimension.
 * @tparam ReplicationCount The number of replicated CountSketch transforms to
 * use.
 */
template <typename RegType, template <std::size_t> class HashType,
          template <typename> class PtrType, std::size_t RangeSize,
          std::size_t ReplicationCount>
class DoubleSparseJLT {
 public:
  using register_type = RegType;
  using hash_type     = HashType<RangeSize>;
  using row_transform_type =
      SparseJLT<RegType, HashType, RangeSize, ReplicationCount>;
  using row_transform_ptr_type = PtrType<row_transform_type>;
  using col_transform_type =
      SparseJLT<RegType, HashType, RangeSize, ReplicationCount>;
  using col_transform_ptr_type = PtrType<col_transform_type>;
  using self_type = DoubleSparseJLT<register_type, HashType, PtrType, RangeSize,
                                    ReplicationCount>;
  using indices_type    = typename row_transform_type::indices_type;
  using polarities_type = typename row_transform_type::polarities_type;
  using update_type     = typename row_transform_type::update_type;

 private:
  row_transform_ptr_type _row_transform_ptr;
  col_transform_ptr_type _col_transform_ptr;

 public:
  /**
   * @brief Construct a new DoubleSparseJLT Functor object by initializing hash
   * functors.
   *
   * Depending on the hash functor to be used, the effective embedding dimension
   * (returned by `this->size()`) may be rounded up to the next power of two.
   * Each insert is hashed to one row and one column location for each
   * combination of ReplicationCount register replicas in the rows and columns
   * of the two-sided sketch. E.g., an insert to a sketch with whose row and
   * columns feature 4 replications will result in updating 16 total indices in
   * the matrix data structure.
   *
   * The implementation currently assumes that the row and column hashes use the
   * same functional form, meaning that the underlying Matrix data structure
   * will always be square.
   *
   * @note This behavior may change in the future.
   *
   * @tparam Args type(s) of additional hash parameters
   * @param row_transform_ptr shared pointer to the row transform.
   * @param col_transform_ptr shared pointer to the column transform.
   */
  DoubleSparseJLT(row_transform_ptr_type row_transform_ptr,
                  col_transform_ptr_type col_transform_ptr)
      : _row_transform_ptr(row_transform_ptr),
        _col_transform_ptr(col_transform_ptr) {}

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
    archive(_row_transform_ptr);
    archive(_col_transform_ptr);
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
    update_type row_hashes = _row_transform_ptr->apply(stream_element.item);
    update_type col_hashes =
        _col_transform_ptr->apply(stream_element.identifier);
    for (int i(0); i < ReplicationCount; ++i) {
      for (int j(0); j < ReplicationCount; ++j) {
        const std::pair<std::uint64_t, std::uint64_t> indices = {
            row_hashes.first[i], col_hashes.first[j]};
        register_type &reg      = registers[indices];
        auto           polarity = row_hashes.second[i] * col_hashes.second[j];
        reg = merge_type()(reg, polarity * stream_element.multiplicity);
        if (reg == 0) {
          registers.erase(indices);
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
      row_transform_type::scaling_factor * col_transform_type::scaling_factor;

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
  constexpr std::uint64_t seed() const { return _row_transform_ptr->seed(); }

  /**
   * @brief Return a description of the transform type.
   *
   * @return std::string "DoubleSparseJLT"
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "DoubleSparseJLT<" << RangeSize << ", " << ReplicationCount << ", "
       << hash_type::name() << ">";
    return ss.str();
  }

  /**
   * @brief Return a description of the fully-qualified transform type.
   *
   * @return std::string Transform description, e.g. "DoubleSparseJLT using
   * MulAddShift hashes and 4 byte registers"
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << "DoubleSparseJLT<" << RangeSize << ", " << ReplicationCount << ", "
       << hash_type::full_name() << ", " << sizeof(register_type) << ">";
    return ss.str();
  }

  constexpr bool same_transforms(const self_type &rhs) const {
    return *_row_transform_ptr == *(rhs._row_transform_ptr) &&
           *_col_transform_ptr == *(rhs._col_transform_ptr);
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
    return lhs.same_transforms(rhs);
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
