// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file hyperloglog.hpp
 * @brief HyperLogLog cardinality estimator
 * @date Created 2013/3/20
 * @author Hideaki Ohno
 *
 * (The MIT License)
 *
 * Copyright (c) 2013 Hideaki Ohno <hide.o.j55{at}gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the 'Software'), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <krowkee/transform/Element.hpp>

#if __has_include(<cereal/types/base_class.hpp>)
#include <cereal/types/base_class.hpp>
#endif

#include <cstdint>
#include <vector>

namespace krowkee::transform {

using krowkee::stream::Element;

static const double pow_2_32     = 4294967296.0;   ///< 2^32
static const double neg_pow_2_32 = -4294967296.0;  ///< -(2^32)

/** @class HyperLogLog
 *  @brief Implement of 'HyperLogLog' estimate cardinality algorithm
 */
template <typename RegType, template <std::size_t> class HashType,
          std::size_t RangeSize>
class HyperLogLog {
 public:
  using hash_type       = HashType<RangeSize>;
  using register_type   = RegType;
  using index_type      = typename hash_type::index_type;
  using zero_count_type = typename hash_type::zero_count_type;
  using update_type     = typename hash_type::update_type;
  using seed_type       = std::uint32_t;
  using self_type       = HyperLogLog<register_type, HashType, RangeSize>;

  // this might need to be reengineered
  static_assert(std::is_same<register_type, zero_count_type>::value,
                "HLL transform and hash types must both use std::uint8_t "
                "register types!");

 protected:
  hash_type                    _hash;
  static constexpr std::size_t _range_size = RangeSize;  ///< register size
  static constexpr double _alpha = (_range_size == 16)   ? 0.673
                                   : (_range_size == 32) ? 0.697
                                   : (_range_size == 64)
                                       ? 0.709
                                       : 0.7213 / (1.0 + 1.079 / _range_size);
  static constexpr double _estimator_scaling_factor =
      _alpha * _range_size * _range_size;

 public:
  /**
   * Constructor
   *
   * @param seed random seed for the hash function.
   *
   * @exception std::invalid_argument the argument is out of range.
   */
  HyperLogLog(seed_type seed = 1) : _hash(seed) {}

  // Assignment operators
  HyperLogLog(const HyperLogLog &)            = default;
  HyperLogLog(HyperLogLog &&)                 = default;
  HyperLogLog &operator=(HyperLogLog &)       = default;
  HyperLogLog &operator=(const HyperLogLog &) = default;

  /**
   * @brief Print functor name
   *
   * @return constexpr std::string
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "HyperLogLog";
    return ss.str();
  }

  /**
   * @brief Print fully-qualified functor name
   *
   * @return constexpr std::string
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << name() << "<" << hash_type::full_name() << ">";
    return ss.str();
  }

  static constexpr std::size_t   range_size() { return _range_size; }
  static constexpr std::size_t   size() { return range_size(); }
  static constexpr register_type scaling_factor = 1;
  static constexpr double        estimator_scaling_factor() {
    return _estimator_scaling_factor;
  }
  constexpr seed_type seed() const { return _hash.seed(); }

  template <typename ObjType>
  constexpr update_type apply(const ObjType &obj) const {
    return _hash(obj);
  }

  template <typename ContainerType, typename... ItemArgs>
  constexpr void operator()(ContainerType &registers,
                            const ItemArgs &...item_args) const {
    static_assert(std::is_same<register_type,
                               typename ContainerType::register_type>::value);
    using merge_type = typename ContainerType::merge_type;
    const Element<register_type> stream_element(item_args...);
    const auto [index, zero_count] = apply(stream_element.item);
    register_type &reg             = registers[index];
    reg                            = merge_type()(reg, zero_count);
    if (reg == 0) {
      registers.erase(index);
    }
  }

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

  /**
   * Estimates cardinality value.
   *
   * @return Estimated cardinality value.
   */
  template <typename ContainerType>
  static constexpr double estimate(const ContainerType &container) {
    static_assert(std::is_same<register_type,
                               typename ContainerType::register_type>::value);
    static_assert(range_size() == ContainerType::size());

    double estimate;
    double sum = 0.0;
    // this can be reengineered to be much more efficient
    for (uint32_t i = 0; i < range_size(); i++) {
      sum += 1.0 / (1 << container[i]);
    }
    estimate = estimator_scaling_factor() / sum;  // E in the original paper
    if (estimate <= 2.5 * range_size()) {
      uint32_t zeros = 0;
      for (uint32_t i = 0; i < range_size(); i++) {
        if (container[i] == 0) {
          zeros++;
        }
      }
      if (zeros != 0) {
        estimate =
            range_size() * std::log(static_cast<double>(range_size()) / zeros);
      }
    } else if (estimate > (1.0 / 30.0) * pow_2_32) {
      estimate = neg_pow_2_32 * log(1.0 - (estimate / pow_2_32));
    }
    return estimate;
  }

  /**
   * Returns size of register.
   *
   * @return Register size
   */
  uint32_t registerSize() const { return _range_size; }

  /**
   * Exchanges the content of the instance
   *
   * @param[in,out] rhs Another HyperLogLog instance
   */
  void swap(HyperLogLog &rhs) { _hash.swap(rhs._hash); }

  /**
   * @brief Check for equality between two SparseJLTs.
   *
   * @param lhs The left-hand functor.
   * @param rhs The right-hand functor.
   * @return true The seeds and range sizes agree.
   * @return false The seeds or range sizes disagree.
   */
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    if (lhs._hash != rhs._hash) {
      return false;
    }
    return true;
  }

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

}  // namespace krowkee::transform
