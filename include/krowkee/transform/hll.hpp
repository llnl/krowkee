#if !defined(HYPERLOGLOG_HPP)
#define HYPERLOGLOG_HPP

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

#include "krowkee/hash/murmur3.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace krowkee::hash {

template <std::size_t RangeSize>
struct HyperLogLogHash32 {
  using index_type      = std::uint32_t;
  using zero_count_type = std::uint8_t;
  using update_type     = std::pair<index_type, zero_count_type>;
  using self_type       = HyperLogLogHash32<RangeSize>;

 protected:
  std::uint32_t                  _seed;
  static constexpr std::size_t   _range_size     = RangeSize;
  static constexpr std::uint32_t _range_width    = std::log2(_range_size);
  static constexpr int           _max_zero_count = 32 - _range_width;

  // check that _range_size is a power of 2
  static_assert((_range_size > 0 && (_range_size & (_range_size - 1)) == 0),
                "RangeSize must be a power of 2 for HyperLogLog hash");
  static_assert(_range_width > 3 && _range_width < 31,
                "bit width must be in the range [4,30]");

 public:
  /**
   * @brief Constructor for the HyperLogLogHash functor
   *
   * @param seed the random seed controlling the randomness.
   */
  template <typename... Args>
  HyperLogLogHash32(const std::uint32_t seed, const Args &...args)
      : _seed(seed) {}

  HyperLogLogHash32() {}

  /**
   * Compute the register hash and leading zero count of the input.
   *
   * @tparam OBJ the object to be hashed. Presently must fit into a 64-bit
   *     register.
   *
   * @param x the obejct to be hashed.
   */
  template <typename OBJ>
  constexpr update_type operator()(const OBJ &x) const {
    // std::uint64_t hash64[2];
    // MurmurHash3_x86_128((void *)&x, sizeof(std::uint64_t), _seed,
    //                     (void *)&hash64);
    // return hash64[0];
    index_type hash32;
    MurmurHash3_x86_32((void *)&x, sizeof(index_type), _seed, (void *)&hash32);
    index_type      index32 = hash32 >> _max_zero_count;
    zero_count_type zero_count32 =
        (uint8_t)std::min(_max_zero_count,
                          std::countl_zero(hash32 << _range_width)) +
        1;
    return {index32, zero_count32};
  }

  /**
   * @brief Print functor name
   *
   * @return constexpr std::string
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "HyperLogLogHash";
    return ss.str();
  }

  /**
   * @brief Print fully-qualified functor name
   *
   * @return constexpr std::string
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << name() << "<" << _range_size << ">";
    return ss.str();
  }

  static constexpr std::size_t size() { return _range_size; }
  static constexpr std::size_t range_size() { return _range_size; }
  static constexpr std::size_t max_zero_count() { return _max_zero_count; }
  static constexpr std::size_t range_width() { return _range_width; }
  constexpr std::size_t        seed() const { return _seed; }

  constexpr std::string state() const {
    std::stringstream ss;
    ss << "size: " << size() << ", seed: " << seed();
    return ss.str();
  }

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_seed);
  }
#endif

  friend void swap(self_type &lhs, self_type &rhs) {
    std::swap(lhs._seed, rhs._seed);
  }
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    return lhs._seed == rhs._seed;
  }

  friend constexpr bool operator!=(const self_type &lhs, const self_type &rhs) {
    return !operator==(lhs, rhs);
  }

  friend std::ostream &operator<<(std::ostream &os, const self_type &func) {
    os << func.state();
    return os;
  }
};

}  // namespace krowkee::hash

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

#endif  // !defined(HYPERLOGLOG_HPP)