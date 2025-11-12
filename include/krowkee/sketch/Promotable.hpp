// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/sketch/Dense.hpp>
#include <krowkee/sketch/Sparse.hpp>

#if __has_include(<cereal/types/memory.hpp>)
#include <cereal/types/memory.hpp>
#endif

#include <memory>

namespace krowkee {
namespace sketch {

enum class promotable_mode_type : std::uint8_t { sparse, dense };

/**
 * @brief Sparse to Dense data structure for Sketch objects.
 *
 * Provides a unified interface to the Sparse and Dense containers, allowing for
 * seamless "promotion" of a Sparse representation to a Dense one.
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
          std::size_t Size, std::size_t CompactionThreshold,
          std::size_t PromotionThreshold>
class Promotable {
  static_assert(CompactionThreshold < PromotionThreshold);

 public:
  /** Alias for the fully-templated Dense container. */
  using register_type  = RegType;
  using merge_type     = MergeOp<register_type>;
  using dense_type     = krowkee::sketch::Dense<register_type, MergeOp, Size>;
  using dense_ptr_type = std::unique_ptr<dense_type>;
  /** Alias for the fully-templated Sparse container. */
  using sparse_type =
      krowkee::sketch::Sparse<register_type, MergeOp, MapType, KeyType, Size,
                              CompactionThreshold>;
  using map_type        = typename sparse_type::map_type;
  using sparse_ptr_type = std::unique_ptr<sparse_type>;
  /** Alias for the fully-templated Promotable container. */
  using self_type = Promotable<register_type, MergeOp, MapType, KeyType, Size,
                               CompactionThreshold, PromotionThreshold>;

 private:
  dense_ptr_type       _dense_ptr;
  sparse_ptr_type      _sparse_ptr;
  promotable_mode_type _mode;

 public:
  /**
   * @brief Initialize the size parameters and create the base container.
   */
  Promotable()
      : _mode(promotable_mode_type::sparse),
        _sparse_ptr(std::make_unique<sparse_type>()) {}

  /**
   * @brief Copy constructor.
   *
   * @note Must explicitly copy contents of rhs's container, not just copy the
   * pointer.
   *
   * @param rhs container to be copied.
   */
  Promotable(const self_type &rhs) : _mode(rhs._mode) {
    if (_mode == promotable_mode_type::sparse) {
      sparse_ptr_type sparse_ptr(
          std::make_unique<sparse_type>(*rhs._sparse_ptr));
      std::swap(sparse_ptr, _sparse_ptr);
    } else {
      dense_ptr_type dense_ptr(std::make_unique<dense_type>(*rhs._dense_ptr));
      std::swap(dense_ptr, _dense_ptr);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Swaps
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Swap two Promotable objects.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand containr.
   */
  friend void swap(self_type &lhs, self_type &rhs) {
    std::swap(lhs._mode, rhs._mode);
    if (lhs._mode == promotable_mode_type::sparse) {
      std::swap(lhs._sparse_ptr, rhs._sparse_ptr);
    } else {
      std::swap(lhs._dense_ptr, rhs._dense_ptr);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Cereal Archives
  //////////////////////////////////////////////////////////////////////////////

#if __has_include(<cereal/types/memory.hpp>)
  /**
   * @brief Save Promotable object to `cereal` archive.
   *
   * @tparam Archive `cereal` archive type.
   * @param archive The `cereal` archive to which to serialize the sketch.
   */
  template <class Archive>
  void save(Archive &oarchive) const {
    oarchive(_mode);
    if (_mode == promotable_mode_type::sparse) {
      oarchive(_sparse_ptr);
    } else {
      oarchive(_dense_ptr);
    }
  }

  /**
   * @brief Load Promotable object to `cereal` archive.
   *
   * @tparam Archive `cereal` archive type.
   * @param archive The `cereal` archive from which to serialize the sketch.
   */
  template <class Archive>
  void load(Archive &iarchive) {
    iarchive(_mode);
    if (_mode == promotable_mode_type::sparse) {
      iarchive(_sparse_ptr);
    } else {
      iarchive(_dense_ptr);
    }
  }
#endif

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
   * @return sparse_type& `this`, having been swapped with `rhs`.
   */
  self_type &operator=(self_type rhs) {
    std::swap(*this, rhs);
    return *this;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Returns a description of the type of container.
   *
   * @return std::string "Promotable"
   */
  static constexpr std::string name() { return "Promotable"; }

  /**
   * @brief Returns a description of the fully-qualified type of container
   *
   * @return std::string "Promotable" plus the full name of the fully-templated
   * compacting_map type.
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << name() << " using " << krowkee::hash::type_name<map_type>();
    return ss.str();
  }

  /** The size of the underlying Sparse or Dense container, depending on the
   * mode. */
  constexpr std::size_t size() const {
    return (_mode == promotable_mode_type::sparse) ? _sparse_ptr->size()
                                                   : _dense_ptr->size();
  }

  /** The maximum possible size of the sketch, disregarding mode. */
  static constexpr std::size_t max_size() { return Size; }

  /** The size of the embedding. Equal to max_size() */
  static constexpr std::size_t embedding_size() { return Size; }

  /** True unless we are in Sparse mode and uncompacted. */
  constexpr bool is_compact() const {
    return (_mode == promotable_mode_type::sparse) ? _sparse_ptr->is_compact()
                                                   : true;
  }

  /** Return `true` if in sparse mode. */
  constexpr bool is_sparse() const {
    return (_mode == promotable_mode_type::sparse) ? true : false;
  }

  /** The number of bytes used by each register. */
  static constexpr std::size_t reg_size() { return sizeof(register_type); }

  /** The size at which we switch from Sparse mode to Dense mode. */
  static constexpr std::size_t promotion_threshold() {
    return PromotionThreshold;
  }

  /** Get the current mode. */
  constexpr promotable_mode_type mode() const { return _mode; }

  /** The size at which the compaction buffer will flush. */
  static constexpr std::size_t compaction_threshold() {
    return CompactionThreshold;
  }

  /**
   * @brief Get a copy of the raw register vector.
   *
   * The vector will be sparse, with zeros for unset indices, if we are in
   * Sparse mode.
   *
   * @return std::vector<register_type> Copy of the register vector.
   */
  std::vector<register_type> register_vector() const {
    return (_mode == promotable_mode_type::sparse)
               ? _sparse_ptr->register_vector()
               : _dense_ptr->register_vector();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Equality operators
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Checks equality between two Promotable containers
   *
   * @note[bwp] do we want to be able to compare sparse to dense containers
   * directly? no for now but will want to revisit.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   * @return true The registers agree and promotion thresholds are the same.
   * @return false At least one register disagrees or the promotion thresholds
   * disagree.
   */
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    if (lhs._mode != rhs._mode) {
      return false;
    }
    if (lhs._mode == promotable_mode_type::sparse) {
      return *lhs._sparse_ptr == *rhs._sparse_ptr;
    } else {
      return *lhs._dense_ptr == *rhs._dense_ptr;
    }
  }

  /**
   * @brief Checks inequality between two Promotable containers
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   * @return true At least one register disagrees or the promotion thresholds
   * disagree.
   * @return false The registers agree and promotion thresholds are the same.
   */
  friend constexpr bool operator!=(const self_type &lhs, const self_type &rhs) {
    return !operator==(lhs, rhs);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Compaction
  //////////////////////////////////////////////////////////////////////////////

  /** If in Sparse mode, flush the update buffer. */
  void compactify() {
    if (_mode == promotable_mode_type::sparse) {
      _sparse_ptr->compactify();
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Clear & Empty
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Remove all state and reset to sparse mode.
   */
  void clear() {
    if (_mode == promotable_mode_type::sparse) {
      _sparse_ptr->clear();
    } else {
      _dense_ptr.reset(nullptr);
      _mode = promotable_mode_type::sparse;
      _sparse_ptr.reset(nullptr);
      _sparse_ptr = std::make_unique<sparse_type>();
    }
  }

  /**
   * @brief Remove all state and reset to sparse mode.
   */
  bool empty() const {
    if (_mode == promotable_mode_type::sparse) {
      return _sparse_ptr->empty();
    } else {
      return _dense_ptr->empty();
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Erase
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Trigger the underlying container to erase an index.
   *
   * @param index The index to erase.
   */
  constexpr void erase(const std::uint64_t index) {
    if (_mode == promotable_mode_type::sparse) {
      _sparse_ptr->erase(index);
    } else {
      _dense_ptr->erase(index);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Register iterators
  //////////////////////////////////////////////////////////////////////////////

  /** Mutable begin iterator. */
  constexpr auto begin() {
    if (_mode == promotable_mode_type::sparse) {
      return std::begin(*_sparse_ptr);
    } else {
      return std::begin(*_dense_ptr);
    }
  }
  /** Const begin iterator. */
  constexpr auto begin() const {
    if (_mode == promotable_mode_type::sparse) {
      return std::cbegin(*_sparse_ptr);
    } else {
      return std::cbegin(*_dense_ptr);
    }
  }
  /** Const begin iterator. */
  constexpr auto cbegin() const {
    if (_mode == promotable_mode_type::sparse) {
      return std::cbegin(*_sparse_ptr);
    } else {
      return std::cbegin(*_dense_ptr);
    }
  }
  /** Mutable end iterator. */
  constexpr auto end() {
    if (_mode == promotable_mode_type::sparse) {
      return std::end(*_sparse_ptr);
    } else {
      return std::end(*_dense_ptr);
    }
  }
  /** Const end iterator. */
  constexpr auto end() const {
    if (_mode == promotable_mode_type::sparse) {
      return std::cend(*_sparse_ptr);
    } else {
      return std::cend(*_dense_ptr);
    }
  }
  /** Const end iterator. */
  constexpr auto cend() const {
    if (_mode == promotable_mode_type::sparse) {
      return std::cend(*_sparse_ptr);
    } else {
      return std::cend(*_dense_ptr);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Accessors
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Access reference to the specified element.
   *
   * @param index The index to access.
   * @return register_type& The accessed register.
   */
  register_type &operator[](const std::uint64_t index) {
    if (_mode == promotable_mode_type::sparse) {
      if (size() == promotion_threshold()) {
        compactify();
        promote();
        return (*_dense_ptr)[index];
      }
      return (*_sparse_ptr)[index];
    } else {
      return (*_dense_ptr)[index];
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Merge Operators
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Merge other Promotable into `this`.
   *
   * @param rhs The other promotable. Must have the same parameters.
   * @return self_type& `this` after merging rhs.
   */
  self_type &operator+=(const self_type &rhs) {
    if (_mode == rhs._mode) {
      if (_mode == promotable_mode_type::sparse) {
        // sparse_type tmp(*rhs._sparse_ptr);
        // *_sparse_ptr += tmp;
        *_sparse_ptr += *rhs._sparse_ptr;
        if (size() >= promotion_threshold()) {
          promote();
        }
      } else {
        *_dense_ptr += *rhs._dense_ptr;
      }
    } else {
      if (_mode == promotable_mode_type::sparse) {
        // we are sparse; promote to dense and add rhs._dense_ptr
        // This will be slow; try to avoid it.
        promote();
        *_dense_ptr += *rhs._dense_ptr;
      } else {
        // we are dense; add rhs._sparse_ptr
        merge_from_sparse(rhs);
      }
    }
    return *this;
  }

  /**
   * @brief Merget two Promotable objects together.
   *
   * lhs and rhs must have the same parameters.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   * @return self_type The merge of the two containers.
   */
  constexpr friend self_type operator+(const self_type &lhs,
                                       const self_type &rhs) {
    if (rhs._mode == promotable_mode_type::dense) {
      self_type ret(rhs);
      ret += lhs;
      return ret;
    }
    self_type ret(lhs);
    ret += rhs;
    return ret;
  }

  /**
   * Promote sparse container into a dense container.
   *
   * Initializes dense pointer while releasing sparse pointer.
   *
   * @throws std::logic_error if the container is not in sparse mode.
   * @throws std::logic_error if the sparse container is not compacted.
   */

  /**
   * @brief Promote sparse container into a dense container.
   *
   * Initializes dense pointer while releasing sparse pointer.
   *
   * @throws std::logic_error if the container is not in sparse mode.
   * @throws std::logic_error if the sparse container is not compacted.
   */
  void promote() {
    if (_mode != promotable_mode_type::sparse) {
      throw std::logic_error("Attempt to promote non-sparse container!");
    }
    if (_sparse_ptr->is_compact() == false) {
      throw std::logic_error("Attempt to promote uncompacted container!");
    }

    _dense_ptr = std::make_unique<dense_type>();
    merge_from_sparse(*this);
    _sparse_ptr.reset(nullptr);
    _mode = promotable_mode_type::dense;
  }

  /**
   * @brief Incorporate the information in a sparse container into a dense
   * container.
   *
   * Initializes dense pointer while releasing sparse pointer.
   *
   * @param rhs The sparse container.
   * @throws std::logic_error if the rhs container is not in sparse mode.
   */
  void merge_from_sparse(const self_type &rhs) {
    if (rhs._mode != promotable_mode_type::sparse) {
      throw std::logic_error("Attempt to dense merge a non-sparse rhs!");
    }
    if (rhs.is_compact() == false) {
      throw std::logic_error("Attempt to dense merge a non-compact rhs!");
    }
    for_each(*rhs._sparse_ptr, [&](const auto &p) {
      register_type &val((*_dense_ptr)[p.first]);
      val = merge_type()(val, p.second);
    });
  }

  //////////////////////////////////////////////////////////////////////////////
  // I/O Operators
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Serialize a Sparse container to human-readable output stream.
   *
   * Output format is a list of (key, value) pairs, skipping empty registers if
   * in Sparse mode.
   *
   * @note Intended for debugging only.
   *
   * @param os The output stream.
   * @param con The Promotable to print.
   * @return std::ostream& The new stream state.
   * @throw std::logic_error if invoked on uncompacted sketch.
   */
  friend std::ostream &operator<<(std::ostream &os, const self_type &con) {
    if (con._mode == promotable_mode_type::sparse) {
      os << *con._sparse_ptr;
    } else {
      os << *con._dense_ptr;
    }
    return os;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Accumulation
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Accumulate sum of register values.
   *
   * @tparam RetType The value type to return.
   * @param con The Promotable to accumulate.
   * @param init Initial value of return.
   * @return RetType The sum over all register values + `init`.
   */
  template <typename RetType>
  friend RetType accumulate(const self_type &con, const RetType init) {
    if (con._mode == promotable_mode_type::sparse) {
      return accumulate(*con._sparse_ptr, init);
    } else {
      return accumulate(*con._dense_ptr, init);
    }
  }
};

}  // namespace sketch
}  // namespace krowkee
