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

#if __has_include(<cereal/types/base_class.hpp>)
#include <cereal/types/base_class.hpp>
#endif

#include <cstdint>
#include <vector>

namespace krowkee::transform {

static const double pow_2_32     = 4294967296.0;   ///< 2^32
static const double neg_pow_2_32 = -4294967296.0;  ///< -(2^32)

/** @class HyperLogLog
 *  @brief Implement of 'HyperLogLog' estimate cardinality algorithm
 */
template <typename HashType>
class HyperLogLog {
 public:
  using hash_type       = HashType;
  using index_type      = typename hash_type::index_type;
  using zero_count_type = typename hash_type::zero_count_type;
  using update_type     = typename hash_type::update_type;
  using self_type       = HyperLogLog;

 protected:
  hash_type                    _hash;
  static constexpr std::size_t _range_size =
      hash_type::range_size();         ///< register size
  std::vector<uint8_t>    _registers;  ///< registers
  static constexpr double _alpha =
      (hash_type::range_size() == 16)   ? 0.673
      : (hash_type::range_size() == 32) ? 0.697
      : (hash_type::range_size() == 64)
          ? 0.709
          : 0.7213 / (1.0 + 1.079 / hash_type::range_size());
  static constexpr double _scaling_factor =
      _alpha * hash_type::range_size() * hash_type::range_size();

 public:
  /**
   * Constructor
   *
   * @param seed random seed for the hash function.
   *
   * @exception std::invalid_argument the argument is out of range.
   */
  HyperLogLog(std::uint32_t seed = 1)
      : _hash(seed), _registers(_range_size, 0) {}

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

  static constexpr std::size_t range_size() { return _range_size; }

  template <typename ObjType>
  constexpr update_type apply(const ObjType &obj) const {
    return _hash(obj);
  }
  /**
   * Inserts element to the data structure
   *
   * @param[in] str string to insert
   * @param[in] len length of string
   */
  void insert(uint32_t obj) {
    const auto [index, zero_count] = apply(obj);
    if (zero_count > _registers[index]) {
      _registers[index] = zero_count;
    }
    // std::cout << std::bitset<32>(hash) << std::endl;
    // std::cout << std::bitset<4>(index) << std::endl;
    // std::cout << (int)rank << std::endl;
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
  double estimate() {
    double estimate;
    double sum = 0.0;
    for (uint32_t i = 0; i < _range_size; i++) {
      sum += 1.0 / (1 << _registers[i]);
    }
    estimate = _scaling_factor / sum;  // E in the original paper
    if (estimate <= 2.5 * _range_size) {
      uint32_t zeros = 0;
      for (uint32_t i = 0; i < _range_size; i++) {
        if (_registers[i] == 0) {
          zeros++;
        }
      }
      if (zeros != 0) {
        estimate =
            _range_size * std::log(static_cast<double>(_range_size) / zeros);
      }
    } else if (estimate > (1.0 / 30.0) * pow_2_32) {
      estimate = neg_pow_2_32 * log(1.0 - (estimate / pow_2_32));
    }
    return estimate;
  }

  /**
   * Merges the estimate from 'other' into this object, returning the estimate
   * of their union. The number of registers in each must be the same.
   *
   * @param[in] other HyperLogLog instance to be merged
   *
   * @exception std::invalid_argument number of registers doesn't match.
   */
  void merge(const HyperLogLog &other) {
    if (_range_size != other._range_size) {
      std::stringstream ss;
      ss << "number of registers doesn't match: " << _range_size
         << " != " << other._range_size;
      throw std::invalid_argument(ss.str().c_str());
    }
    for (uint32_t r = 0; r < _range_size; ++r) {
      if (_registers[r] < other._registers[r]) {
        _registers[r] |= other._registers[r];
      }
    }
  }

  HyperLogLog &operator+=(const HyperLogLog &other) {
    merge(other);
    return *this;
  }

  /**
   * Clears all internal registers.
   */
  void clear() { std::fill(_registers.begin(), _registers.end(), 0); }

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
  void swap(HyperLogLog &rhs) { _registers.swap(rhs._registers); }

  constexpr const std::vector<uint8_t> &registers() const { return _registers; }
};

}  // namespace krowkee::transform
