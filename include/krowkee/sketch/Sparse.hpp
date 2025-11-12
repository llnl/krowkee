// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/container/compacting_map.hpp>

#include <algorithm>
#include <sstream>
#include <vector>

namespace krowkee {
namespace sketch {

/**
 * @brief General Sparse Sketch container.
 *
 * Implements the container handling infrastructure of any sketch whose atomic
 * elements can be stored as sorted sparse index-value pairs into a notional
 * register list of fixed size and supporting a merge operation consisting of
 * the union of their index-value pair lists  with the application of an
 * element-wise operator on pairs with matching indices.
 *
 * @tparam RegType The type held by each register.
 * @tparam MergeOp An template merge operator to combine two sketches.
 * @tparam MapType The underlying map class to use for buffered updates to the
 * underlying compacting_map.
 * @tparam KeyType The key type to use for this underlying compacting map.
 * @tparam Size The maximum number of registers. Must equal the embedding
 * dimension of the transform to be used.
 */
template <typename RegType, template <typename> class MergeOp,
          template <typename, typename> class MapType, typename KeyType,
          std::size_t Size, std::size_t CompactionThreshold>
class Sparse {
 public:
  /** Alias for the fully-templated register compacting map type. */
  using register_type = RegType;
  using merge_type    = MergeOp<register_type>;
  using registers_type =
      krowkee::container::compacting_map<KeyType, register_type, MapType>;
  using vec_iter_type  = typename registers_type::vec_iter_type;
  using vec_citer_type = typename registers_type::vec_citer_type;
  using pair_type      = typename registers_type::pair_type;
  using map_type       = typename registers_type::map_type;
  using self_type      = Sparse<register_type, MergeOp, MapType, KeyType, Size,
                                CompactionThreshold>;

 protected:
  registers_type _registers;

 public:
  /**
   * @brief Construct a new Sparse container object
   */
  Sparse() : _registers(compaction_threshold()) {}

  /**
   * @brief Copy constructor.
   *
   * @param rhs The base Sparse container to copy.
   */
  Sparse(const self_type &rhs) : _registers(rhs._registers) {}

  //////////////////////////////////////////////////////////////////////////////
  // Swaps
  //////////////////////////////////////////////////////////////////////////////
  /**
   * Swap boilerplate.
   *
   */
  /**
   * @brief Swap two Sparse containers.
   *
   * @note For some reason, calling std::swap on the registers here causes a
   * segfault in Distributed::data_t::swap, but not Sketch::swap. Peculiar.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   */
  friend void swap(self_type &lhs, self_type &rhs) {
    swap(lhs._registers, rhs._registers);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Cereal Archives
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Serialize Sparse object to/from `cereal` archive.
   *
   * @tparam Archive `cereal` archive type.
   * @param archive The `cereal` archive to which to serialize the sketch.
   */
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_registers);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Compaction
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Checks whether the compacting sketch is compact.
   *
   * @return true The update buffer is empty.
   * @return false The update buffer is not empty.
   */
  constexpr bool is_compact() const { return _registers.is_compact(); }

  /** Flush the update buffer. */
  void compactify() { _registers.compactify(); }

  //////////////////////////////////////////////////////////////////////////////
  // Clear & Empty
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Remove all information from sketch.
   */
  void clear() { _registers.clear(); }

  /**
   * @brief Check if registers contain any information.
   */
  bool empty() const { return _registers.empty(); }

  //////////////////////////////////////////////////////////////////////////////
  // Erase
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Erase a map index.
   *
   * @param index The map index to erase.
   */
  constexpr void erase(const std::uint64_t index) { _registers.erase(index); }

  //////////////////////////////////////////////////////////////////////////////
  // Merge operators
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Merge other Sparse registers into `this`.
   *
   * `merge_type` determines how register lists are combined. For linear
   * sketches, merge amounts to the element-wise addition of register
   * arrays.
   *
   *
   * @param rhs the other Sparse. Care must be taken to ensure that one does not
   * merge sketches of different types.
   * @throw std::logic_error if invoked on uncompacted sketches.
   */
  constexpr void merge(const self_type &rhs) {
    _registers.merge(rhs._registers, merge_type());
  }

  /**
   *
   *
   */

  /**
   * @brief Operator overload for convenience for embeddings without additional
   * consistency checks.
   *
   * @param rhs the other Sparse. Care must be taken to ensure that one does not
   * merge subspace embeddings of different types.
   * @return self_type& `this` Sparse, having been merged with `rhs`.
   */
  self_type &operator+=(const self_type &rhs) {
    merge(rhs);
    return *this;
  }

  /**
   * @brief Merge two Sparse containers.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   * @return self_type The merge of the two containers.
   */
  constexpr friend self_type operator+(self_type lhs, const self_type &rhs) {
    lhs += rhs;
    return lhs;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Register iterators
  //////////////////////////////////////////////////////////////////////////////

  /** Mutable begin iterator. */
  constexpr auto begin() { return std::begin(_registers); }
  /** Const begin iterator. */
  constexpr auto begin() const { return std::cbegin(_registers); }
  /** Const begin iterator. */
  constexpr auto cbegin() const { return std::cbegin(_registers); }
  /** Mutable end iterator. */
  constexpr auto end() { return std::end(_registers); }
  /** Const end iterator. */
  constexpr auto end() const { return std::cend(_registers); }
  /** Const end iterator. */
  constexpr auto cend() { return std::cend(_registers); }

  //////////////////////////////////////////////////////////////////////////////
  // Accessors
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Const access Sparse at `index`.
   *
   * @param index The index of the underlying vector to index. Must be less than
   * `size`.
   * @return constexpr const register_type& A const refernce to
   * `_registers[index]`.
   */
  constexpr const register_type &operator[](const std::uint64_t index) const {
    return _registers.get(index);
  }

  /**
   * @brief Access Sparse at `index`.
   *
   * @param index The index of the underlying vector to index. Must be less than
   * `size`.
   * @return register_type& A refernce to `_registers[index]`.
   */
  register_type &operator[](const std::uint64_t index) {
    return _registers[index];
  }

  /**
   * @brief Access Sparse at `index`.
   *
   * @param index The index of the underlying vector to index. Must be less than
   * `size`.
   * @return register_type& A refernce to `_registers.at(index)`.
   */
  register_type &at(const std::uint64_t index) { return _registers.at(index); }

  /**
   * @brief Access Sparse at `index`.
   *
   * @param index The index of the underlying vector to index. Must be less than
   * `size`.
   * @param def The default value to return if the index does not exist.
   * @return register_type& A refernce to `_registers.at(index)`.
   */
  register_type &at(const std::uint64_t index, const register_type def) {
    return _registers.at(index, def);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Returns a description of the type of container.
   *
   * @return std::string "Sparse"
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "Sparse<" << Size << ", " << CompactionThreshold << ", "
       << typeid(map_type).name() << ">";
    return ss.str();
  }

  /**
   * @brief Returns a description of the fully-qualified type of container
   *
   * @return std::string "Sparse" plus the full name of the fully-templated
   * compacting_map type.
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << "Sparse<" << Size << ", " << CompactionThreshold << ", "
       << krowkee::hash::type_name<map_type>() << ">";
    return ss.str();
  }

  /** Sparse is always Sparse. */
  constexpr bool is_sparse() const { return true; }

  /** The current size of the compacting_map. */
  constexpr std::size_t size() const { return _registers.size(); }

  /** The number of bytes used by each register. */
  static constexpr std::size_t reg_size() { return sizeof(register_type); }

  /** The maximum possible size of the compacting_map. */
  static constexpr std::size_t max_size() { return Size; }

  /** The size of the embedding. Equal to max_size() */
  static constexpr std::size_t embedding_size() { return Size; }

  /** The size at which the compaction buffer will flush. */
  static constexpr std::size_t compaction_threshold() {
    return CompactionThreshold;
  }

  /**
   * @brief Get a copy of the raw (sparse) register vector
   *
   * @return std::vector<register_type> The register vector, with zeros for
   * unset indices.
   * @throws std::logic_error if the object is not compact.
   */
  std::vector<register_type> register_vector() const {
    if (is_compact() == false) {
      throw std::logic_error("Bad attempt to export uncompacted map!");
    }
    std::vector<register_type> ret(Size);
    std::for_each(std::cbegin(_registers), std::cend(_registers),
                  [&ret](const std::pair<KeyType, register_type> &elem) {
                    ret[elem.first] = elem.second;
                  });
    return ret;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Equality operators
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Checks whether another Sparse container has the same register state.
   *
   * @param rhs The other container.
   * @return true The registers agree.
   * @return false At least one register disagrees.
   */
  constexpr bool same_registers(const self_type &rhs) const {
    return _registers == rhs._registers;
  }

  /**
   * @brief Checks equality between two Sparse containers.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   * @return true The registers agree.
   * @return false At least one register disagrees.
   */
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    return lhs.same_registers(rhs);
  }

  /**
   * @brief Checks inequality between two Sparse containers.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   * @return true At least one register disagrees.
   * @return false The registers agree.
   */
  friend constexpr bool operator!=(const self_type &lhs, const self_type &rhs) {
    return !operator==(lhs, rhs);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Assignment
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Copy-and-swap assignment operator
   *
   * @note
   * https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
   *
   * @param rhs The other container.
   * @return self_type& `this`, having been swapped with `rhs`.
   */
  self_type &operator=(self_type rhs) {
    std::swap(*this, rhs);
    return *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  // I/O Operators
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Serialize a Sparse container to human-readable output stream.
   *
   * Output format is a list of (key, value) pairs, skipping empty registers.
   *
   * @note Intended for debugging only.
   *
   * @param os The output stream.
   * @param sk The Sparse object.
   * @return std::ostream& The new stream state.
   * @throw std::logic_error if invoked on uncompacted sketch.
   */
  friend std::ostream &operator<<(std::ostream &os, const self_type &sk) {
    if (sk.is_compact() == false) {
      throw std::logic_error("Bad attempt to write uncompacted map!");
    }
    // os << sk._registers.print_state();
    bool first(true);
    for_each(sk, [&](const auto &p) {
      if (first == false) {
        os << " ";
      } else {
        first == false;
      }
      os << "(" << std::int64_t(p.first) << "," << int(p.second) << ") ";
    });
    return os;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Accumulation
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Accumulate sum of register values.
   *
   * @tparam RetType The value type to return.
   * @param sk The Sparse to accumulate.
   * @param init Initial value of return.
   * @return RetType The sum over all register values + `init`.
   */
  template <typename RetType>
  friend RetType accumulate(const self_type &sk, const RetType init) {
    return std::accumulate(std::cbegin(sk._registers), std::cend(sk._registers),
                           init, [](const RetType val, const pair_type &itr) {
                             return RetType(val + itr.second);
                           });
  }

  /**
   * @brief Apply a given function to all register values.
   *
   * @tparam Func The (probably lambda) function type.
   * @param sk The Sparse object.
   * @param func The particular function to apply.
   */
  template <typename Func>
  friend void for_each(const self_type &sk, const Func &func) {
    std::for_each(std::cbegin(sk._registers), std::cend(sk._registers), func);
  }
};

}  // namespace sketch
}  // namespace krowkee
