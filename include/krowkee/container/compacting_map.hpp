// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/util.hpp>

#if __has_include(<cereal/types/map.hpp>)
#include <cereal/types/map.hpp>
#endif

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace krowkee {
namespace container {

/**
 * Merge sorted key-value data structures, applying a binary operator to the
 * values of ties.
 *
 * @note[bwp] This is fucking awful. Need to move this logic inside of
 * compacting_map and clean it up. Invocations of std::copy are slow; see if
 * there is a better way to do this.
 */
template <typename MergeOp, typename LhsIt, typename RhsIt, typename OutIt>
void merge_and_compact(LhsIt lhs_itr, const LhsIt lhs_end, RhsIt rhs_itr,
                       const RhsIt rhs_end, OutIt new_itr, MergeOp merge_op) {
  for (; lhs_itr != lhs_end; ++new_itr) {
    if (rhs_itr == rhs_end) {
      std::copy(lhs_itr, lhs_end, new_itr);
      break;
    }
    if (rhs_itr->first < lhs_itr->first) {
      std::copy(rhs_itr, rhs_itr + 1, new_itr);
      ++rhs_itr;
    } else if (rhs_itr->first > lhs_itr->first) {
      std::copy(lhs_itr, lhs_itr + 1, new_itr);
      ++lhs_itr;
    } else {  // if the indices are equal, perform the merge!
              /**
               * @note[bwp] wow this is wildly unsafe. Need a better
               * implementation.
               */
      auto ret = merge_op(lhs_itr->second, rhs_itr->second);
      if (ret != 0) {
        lhs_itr->second = ret;
        std::copy(lhs_itr, lhs_itr + 1, new_itr);
      }
      ++lhs_itr;
      ++rhs_itr;
    }
  }
  std::copy(rhs_itr, rhs_end, new_itr);
}

/**
 * Space-efficient map, supporting lazy insertion.
 *
 * Core data structures are an archive std::vector and a small,
 * templated MapType instance. Code is based in part on an archived C++
 * User's Group article [0], whose source code appears to be lost to
 * time. There appear to be similar implementations, e.g. from Apple
 * [1], a hashmap implementation [2], and boost::container::flat_map,
 * but there was no standalone, simple, open-source solution to be had
 * that satisfied the requirements of being
 *   (1) fast,
 *   (2) scalable, and
 *   (3) truly compact.
 *
 * This implementation only supports iteration once the data structure
 * is *compacted*, which occurs naturally as elements are inserted and
 * when member function `compactify` is called. Care must be taken to avoid
 * interacting with the map when in an uncompacted state.
 *
 * References:
 *
 * [0]
 * https://www.drdobbs.com/space-efficient-sets-and-maps/184401668?queryText=carrato
 * [1]
 * https://developer.apple.com/documentation/swift/sequence/2950916-compactmap
 * [2] https://github.com/greg7mdp/sparsepp
 * [4] https://gist.github.com/jeetsukumaran/307264
 *
 * @tparam KeyType Hashable type of the index elements of the map.
 * @tparam ValueType Value type of the map.
 * @tparam MapType Underlying map implementation. Defaults to std::map but
 * boost::container::flat_map is better for performance.
 */
template <typename KeyType, typename ValueType,
          template <typename, typename> class MapType = std::map>
class compacting_map {
 public:
  // Key needs to be const to agree with runtime map iterator typing.
  using pair_type      = std::pair<KeyType, ValueType>;
  using vec_type       = std::vector<std::pair<KeyType, ValueType>>;
  using vec_iter_type  = typename vec_type::iterator;
  using vec_citer_type = typename vec_type::const_iterator;
  using map_type       = MapType<KeyType, ValueType>;
  using map_iter_type  = typename map_type::iterator;
  using map_citer_type = typename map_type::const_iterator;
  using iterator       = typename vec_type::iterator;
  using const_iterator = typename vec_type::const_iterator;
  using self_type      = compacting_map<KeyType, ValueType, MapType>;

 protected:
  std::vector<bool> _erased;
  vec_type          _archive_map;
  map_type          _dynamic_map;
  std::size_t       _compaction_threshold;
  int               _erased_count;

  struct compare_first_f {
    bool operator()(const pair_type &lhs, const pair_type &rhs) const {
      return lhs.first < rhs.first;
    }
    bool operator()(const pair_type &lhs, const KeyType &key) const {
      return lhs.first < key;
    }
    bool operator()(const KeyType &key, const pair_type &rhs) const {
      return key < rhs.first;
    }
  };

  compare_first_f compare_first;

 public:
  //////////////////////////////////////////////////////////////////////////////
  // Constructors
  //////////////////////////////////////////////////////////////////////////////

  compacting_map(std::size_t compaction_threshold)
      : _compaction_threshold(compaction_threshold), _erased_count(0) {
    _archive_map.reserve(_compaction_threshold);
    _erased.reserve(_compaction_threshold);
  }

  compacting_map(const self_type &rhs)
      : _compaction_threshold(rhs._compaction_threshold),
        _erased_count(rhs._erased_count),
        _erased(rhs._erased),
        _archive_map(rhs._archive_map),
        _dynamic_map(rhs._dynamic_map) {}

  compacting_map() {}

  // compacting_map(self_type &&rhs) : self_type() { std::swap(*this, rhs); }

  //////////////////////////////////////////////////////////////////////////////
  // Cereal Archives
  //////////////////////////////////////////////////////////////////////////////

#if __has_include(<cereal/types/map.hpp>)
  template <class Archive>
  void save(Archive &archive) const {
    if (is_compact() != true) {
      throw std::logic_error(
          "Incorrectly trying to archive a non-compact map!");
    }
    archive(_compaction_threshold, _archive_map);
  }

  template <class Archive>
  void load(Archive &archive) {
    archive(_compaction_threshold, _archive_map);
    _erased.resize(_archive_map.size());
    std::fill(std::begin(_erased), std::end(_erased), false);
    _erased_count = 0;
  }
#endif

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  //////////////////////////////////////////////////////////////////////////////

  constexpr std::size_t size() const {
    return _archive_map.size() + _dynamic_map.size();
  }

  constexpr bool is_compact() const {
    return _dynamic_map.size() == 0 && _erased_count == 0;
  }

  constexpr std::size_t compaction_threshold() const {
    return _compaction_threshold;
  }

  constexpr std::size_t erased_count() const { return _erased_count; }

  constexpr std::size_t erased_count_manual() const {
    std::size_t erased_count(
        std::count(std::begin(_erased), std::end(_erased), true));
    return erased_count;
  }

  static constexpr std::string name() { return "compacting_map"; }

  constexpr std::string full_name() const {
    std::stringstream ss;
    ss << name() << " using " << krowkee::hash::type_name<map_type>()
       << " with threshold " << _compaction_threshold;
    return ss.str();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Compaction
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Sort _dynamic_map into _archive_map, clearing deleted items along the way.
   *
   * @note[bwp] right now this does two copies, which is suboptimal. Should look
   * into ways to improve performance.
   */
  void compactify() {
    if (is_compact()) {
      return;
    }
    // copy nondeleted elements
    vec_type tmp1;
    tmp1.reserve(_archive_map.size());
    for (std::size_t i(0); i < _archive_map.size(); ++i) {
      if (_erased[i] == false) {
        tmp1.push_back(_archive_map[i]);
      }
    }
    // perform union
    vec_type tmp2;
    tmp2.reserve(tmp2.size() + _dynamic_map.size());
    std::set_union(std::begin(tmp1), std::end(tmp1), std::begin(_dynamic_map),
                   std::end(_dynamic_map), std::back_inserter(tmp2),
                   compare_first);
    _dynamic_map.clear();
    std::swap(_archive_map, tmp2);
    _erased.resize(_archive_map.size());
    std::fill(std::begin(_erased), std::end(_erased), false);
    _erased_count = 0;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Clear & Empty
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Clear all state.
   */
  void clear() {
    _dynamic_map.clear();
    _archive_map.clear();
    _erased.clear();
  }

  /**
   * @brief Check if there is any state.
   */
  bool empty() const {
    return _dynamic_map.empty() && _archive_map.empty() && _erased.empty();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Insert
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Insert a key-value pair.
   *
   * Checks for the existence of the key in the dynamic map. Failing to find
   * that, calls `try_insert_archive` in an attempt to insert the pair into
   * the archive (or, failing that, the dynamic set).
   *
   * @note[bwp]: currently does not support iterator return, breaking the
   * conventional std::map API. This is probably necessary, as we do not want
   * to return iterators to the dynamic set or archive set can get immediately
   * invalidated. Moreover, we do not want a user to attempt to increment or
   * decrement such an iterator, as its meaning is heavily dependent upon the
   * data structure's state.
   */
  bool insert(const pair_type &pair) {
    auto it_dyn(_dynamic_map.find(pair.first));
    if (it_dyn != std::end(_dynamic_map)) {
      return false;
    }
    return try_insert_archive(pair);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Accessors
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Access an element.
   *
   * Mimicks the behavior of the std::map accessor operator.
   */
  ValueType &operator[](const KeyType &key) {
    auto dyn_iter(_dynamic_map.find(key));
    if (dyn_iter != std::end(_dynamic_map)) {
      return dyn_iter->second;
    }

    auto [axv_iter, axv_code] = archive_find(key);
    if (axv_code == archive_code_t::present) {
      return axv_iter->second;
    } else if (axv_code == archive_code_t::deleted) {
      axv_iter->second                             = 0;
      _erased[axv_iter - std::begin(_archive_map)] = false;
      --_erased_count;
      return axv_iter->second;
    } else if (axv_code == archive_code_t::absent) {
      auto [dyn_iter, dyn_code] = dynamic_insert(pair_type{key, 0});
      if (dyn_code == dynamic_code_t::compaction) {
        auto [axv_iter2, axv_code2] = archive_find(key);
        if (axv_code2 != archive_code_t::present) {
          throw std::logic_error(
              "Compaction failed to move dynamic element to archive map!");
        }
        return axv_iter2->second;
      }
      return dyn_iter->second;
    } else {
      throw std::invalid_argument("archive code is invalid");
    }
  }

  /**
   * Access an element.
   *
   * Mimicks the behavior of std::map::at. Throws an error if the supplied key
   * is not already stored or is erased.
   */
  ValueType &at(const KeyType &key) {
    auto dyn_iter(_dynamic_map.find(key));
    if (dyn_iter != std::end(_dynamic_map)) {
      return dyn_iter->second;
    }
    auto [axv_iter, axv_code] = archive_find(key);
    if (axv_code == archive_code_t::present) {
      return axv_iter->second;
    } else {
      std::stringstream ss;
      ss << "Key name " << key << " does not exist!";
      throw std::out_of_range(ss.str());
    }
  }

  /**
   * Access an element.
   *
   * Mimicks the behavior of std::map::at. Throws an error if the supplied key
   * is not already stored or is erased.
   */
  const ValueType &at(const KeyType &key) const {
    auto dyn_iter(_dynamic_map.find(key));
    if (dyn_iter != std::end(_dynamic_map)) {
      return dyn_iter->second;
    }
    auto [axv_iter, axv_code] = archive_find(key);
    if (axv_code == archive_code_t::present) {
      return axv_iter->second;
    } else {
      std::stringstream ss;
      ss << "Key name " << key << " does not exist!";
      throw std::out_of_range(ss.str());
    }
  }

  /**
   * Access an element, returning a default if the desired key is not found.
   *
   * Mimicks the behavior of std::map::at. Throws an error if the supplied key
   * is not already stored or is erased.
   */
  const ValueType &at(const KeyType &key, const ValueType &val) const {
    auto dyn_iter(_dynamic_map.find(key));
    if (dyn_iter != std::end(_dynamic_map)) {
      return dyn_iter->second;
    }
    auto [axv_iter, axv_code] = archive_find(key);
    if (axv_code == archive_code_t::present) {
      return axv_iter->second;
    } else {
      return val;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Erase
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Erases a given key, if found.
   *
   * Erases they key directly if found in the dynamic map. Instead sets the
   * corresponding element of the _erased vector to true if found in the
   * archive map.
   */
  std::size_t erase(const KeyType &key) {
    std::size_t erase_count(_dynamic_map.erase(key));

    if (erase_count > 0) {
      return erase_count;
    }

    auto [axv_lb, axv_code] = archive_find(key);
    if (axv_code == archive_code_t::present) {
      _erased[axv_lb - std::begin(_archive_map)] = true;
      ++_erased_count;
      return 1;
    }

    return 0;
  }

  /**
   * Erases a given key given its iterator.
   *
   * @note[bwp] Not compact safe! Obtained iterators are invalidated once
   * compactify() occurs.
   */
  std::size_t erase(const vec_iter_type &iter) {
    if (iter >= std::begin(_archive_map) && iter < std::end(_archive_map)) {
      _erased[iter - std::begin(_archive_map)] = true;
      ++_erased_count;
      return 1;
    } else {
      return 0;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Find
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Return an interator to the given key-value pair.
   *
   * @throw std::logic_error if invoked on an uncompacted map.
   */
  vec_iter_type find(const KeyType &key) {
    if (_dynamic_map.size() > 0) {
      throw std::logic_error("Bad invocation of `find` on uncompacted map!");
    }

    auto [axv_lb, axv_code] = archive_find(key);
    if (axv_code == archive_code_t::present) {
      return axv_lb;
    }

    return end();
  }

  /**
   * Return a const interator to the given key-value pair.
   *
   * Throws an std::logic_error if invoked on an uncompacted map.
   */
  vec_citer_type find(const KeyType &key) const {
    if (_dynamic_map.size() > 0) {
      throw std::logic_error("Bad invocation of `find` on uncompacted map!");
    }

    auto [axv_lb, axv_code] = archive_find(key);
    if (axv_code == archive_code_t::present) {
      return axv_lb;
    }

    return cend();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Iterators
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Return an interator to the beginning of the archive vector.
   *
   * @note[bwp]: currently not compact safe!
   */
  constexpr vec_iter_type  begin() { return std::begin(_archive_map); }
  constexpr vec_citer_type begin() const { return std::cbegin(_archive_map); }
  constexpr vec_citer_type cbegin() const { return std::cbegin(_archive_map); }

  /**
   * Return an interator to the end of the archive vector.
   *
   * @note[bwp]: currently not compact safe!
   */
  constexpr vec_iter_type  end() { return std::end(_archive_map); }
  constexpr vec_citer_type end() const { return std::cend(_archive_map); }
  constexpr vec_citer_type cend() const { return std::cend(_archive_map); }

  //////////////////////////////////////////////////////////////////////////////
  // Equality operators
  //////////////////////////////////////////////////////////////////////////////
  constexpr bool same_maps(const self_type &rhs) const {
    return _compaction_threshold == rhs._compaction_threshold &&
           _dynamic_map.size() == rhs._dynamic_map.size() &&
           _erased_count == rhs._erased_count &&
           _erased.size() == rhs._erased.size() &&
           _archive_map.size() == rhs._archive_map.size() &&
           std::equal(std::cbegin(_dynamic_map), std::cend(_dynamic_map),
                      std::cbegin(rhs._dynamic_map)) &&
           std::equal(std::cbegin(_erased), std::cend(_erased),
                      std::cbegin(rhs._erased)) &&
           std::equal(std::cbegin(_archive_map), std::cend(_archive_map),
                      std::cbegin(rhs._archive_map));
  }

  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    return lhs.same_maps(rhs);
  }
  friend constexpr bool operator!=(const self_type &lhs, const self_type &rhs) {
    return !operator==(lhs, rhs);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Swaps
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Swaps
   *
   * @note[bwp]: Is this safe? Should we be checking for compatibility?
   */
  friend void swap(self_type &lhs, self_type &rhs) {
    std::swap(lhs._compaction_threshold, rhs._compaction_threshold);
    std::swap(lhs._erased_count, rhs._erased_count);
    std::swap(lhs._dynamic_map, rhs._dynamic_map);
    std::swap(lhs._erased, rhs._erased);
    std::swap(lhs._archive_map, rhs._archive_map);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Assignment
  //////////////////////////////////////////////////////////////////////////////
  /**
   * copy-and-swap assignment operator
   *
   * https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
   */
  self_type &operator=(self_type rhs) {
    std::swap(*this, rhs);
    return *this;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Merge
  //////////////////////////////////////////////////////////////////////////////
  /**
   * Merge other compacting map into `this` using supplied MergeOp to break
   * ties.
   *
   * @param rhs the other compacting_map.
   *
   * @throw std::logic_error if invoked on uncompacted maps.
   */
  template <typename MergeOp>
  constexpr void merge(const self_type &rhs, MergeOp merge_op) {
    if (is_compact() == false) {
      throw std::logic_error(
          "Bad attempt to merge on uncompacted left hand side!");
    }
    if (rhs.is_compact() == false) {
      throw std::logic_error(
          "Bad attempt to merge on uncompacted right hand side!");
    }
    vec_type tmp;
    tmp.reserve(_archive_map.size() + rhs._archive_map.size());
    merge_and_compact(std::begin(_archive_map), std::end(_archive_map),
                      std::cbegin(rhs._archive_map),
                      std::cend(rhs._archive_map), std::back_inserter(tmp),
                      merge_op);
    std::swap(_archive_map, tmp);
    _erased.resize(_archive_map.size());
  }

  //////////////////////////////////////////////////////////////////////////////
  // I/O Operators
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Returns the current state of the data structure to a string.
   *
   * @note[bwp]: For debugging only.
   */
  std::string print_state() const {
    std::stringstream ss{};
    ss << "axv (" << _archive_map.size() << "): ";
    for (auto iter(std::begin(_archive_map)); iter != std::end(_archive_map);
         ++iter) {
      ss << "(" << iter->first << "," << iter->second << ") ";
    }
    ss << '\n';
    ss << "dyn (" << _dynamic_map.size() << "): ";
    for (auto iter(std::begin(_dynamic_map)); iter != std::end(_dynamic_map);
         ++iter) {
      ss << "(" << iter->first << "," << iter->second << ") ";
    }
    ss << '\n';
    ss << "cmp: ";
    ss << _compaction_threshold;
    return ss.str();
  }

 private:
  enum class archive_code_t : std::uint8_t { present, deleted, absent };

  enum class dynamic_code_t : std::uint8_t { success, failure, compaction };

  /**
   * Attempts to find an iterator to the desired key-value pair in the archive
   * vector.
   *
   * Searches through the archive vector, returning the specified iterator if
   * found else the end iterator. Returns a status code indicating whether the
   * desired key is absent, present, or present but deleted.
   */
  constexpr std::pair<vec_iter_type, archive_code_t> archive_find(
      const KeyType &key) {
    auto axv_lb = std::lower_bound(std::begin(_archive_map),
                                   std::end(_archive_map), key, compare_first);
    bool is_present =
        (axv_lb != std::end(_archive_map) && axv_lb->first == key);
    if (is_present == false) {
      return {axv_lb, archive_code_t::absent};
    } else {
      int axv_offset = axv_lb - std::begin(_archive_map);
      if (_erased[axv_offset] == true) {
        return {axv_lb, archive_code_t::deleted};
      } else {
        return {axv_lb, archive_code_t::present};
      }
    }
  }

  /**
   * Const version of `archive_find`.
   */
  constexpr std::pair<vec_citer_type, archive_code_t> archive_find(
      const KeyType &key) const {
    auto axv_lb = std::lower_bound(std::cbegin(_archive_map),
                                   std::cend(_archive_map), key, compare_first);
    bool is_present =
        (axv_lb != std::cend(_archive_map) && axv_lb->first == key);
    if (is_present == false) {
      return {axv_lb, archive_code_t::absent};
    } else {
      int axv_offset = axv_lb - std::begin(_archive_map);
      if (_erased[axv_offset] == true) {
        return {axv_lb, archive_code_t::deleted};
      } else {
        return {axv_lb, archive_code_t::present};
      }
    }
  }

  /**
   * Attempts to insert a key-value pair in the dynamic map.
   *
   * Calls the dynamic map's `insert` function. If this causes the size to
   * reach the compaction threshold, the compacts.
   */
  constexpr std::pair<map_iter_type, dynamic_code_t> dynamic_insert(
      const pair_type &pair) {
    // this call to std::make_pair introduces an unnecessary copy, but seems
    // to be needed for compilation when `_dynamic_map` is a
    // boost::container::flat_map...
    auto [dyn_iter, success] =
        _dynamic_map.insert(std::make_pair(pair.first, pair.second));
    if (_dynamic_map.size() == _compaction_threshold) {
      compactify();
      return {dyn_iter, dynamic_code_t::compaction};
      // return true;
    } else {
      return {dyn_iter, (success == true) ? dynamic_code_t::success
                                          : dynamic_code_t::failure};
      // return success;
    }
  }

  /**
   * Attempts to insert a key-value pair in the archive vector.
   *
   * Calls `archive_find`. If the key is already present, the insert fails. If
   * the key is present but deleted, the insert occurs in-place. If the key is
   * absent, call `dynamic_insert` and insert into the dynamic buffer.
   */
  constexpr bool try_insert_archive(const pair_type &pair) {
    auto [axv_lb, axv_code] = archive_find(pair.first);

    if (axv_code == archive_code_t::present) {
      return false;
    } else if (axv_code == archive_code_t::deleted) {
      axv_lb->second      = pair.second;
      int axv_offset      = axv_lb - std::begin(_archive_map);
      _erased[axv_offset] = false;
      --_erased_count;
      return true;
    } else {  // axv_code == archive_code_t::absent
      auto [dyn_iter, dyn_code] = dynamic_insert(pair);
      return (dyn_code == dynamic_code_t::failure) ? false : true;
      // return dynamic_insert(pair);
    }
  }
};

}  // namespace container
}  // namespace krowkee
