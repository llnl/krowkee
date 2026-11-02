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
struct HyperLogLogHash {
  using index_type      = std::uint32_t;
  using zero_count_type = std::uint8_t;
  using update_type     = std::pair<index_type, zero_count_type>;
  using self_type       = HyperLogLogHash<RangeSize>;

  // check that RangeSize is a power of 2
  static_assert((RangeSize > 0 && (RangeSize & (RangeSize - 1)) == 0),
                "RangeSize must be a power of 2 for HyperLogLog hash");

 protected:
  std::uint32_t                _seed;
  static constexpr std::size_t _range_width    = std::log2(RangeSize);
  static constexpr std::size_t _max_zero_count = 32 - _range_width;

 public:
  /**
   * @brief Constructor for the HyperLogLogHash functor
   *
   * @param seed the random seed controlling the randomness.
   */
  template <typename... Args>
  HyperLogLogHash(const std::uint32_t seed, const Args &...args)
      : _seed(seed) {}

  HyperLogLogHash() {}

  /**
   * Compute the register hash and leading zero count of the input.
   *
   * @tparam OBJ the object to be hashed. Presently must fit into a 64-bit
   *     register.
   *
   * @param x the obejct to be hashed.
   */
  template <typename OBJ>
  constexpr std::uint32_t operator()(const OBJ &x) const {
    // std::uint64_t hash64[2];
    // MurmurHash3_x86_128((void *)&x, sizeof(std::uint64_t), _seed,
    //                     (void *)&hash64);
    // return hash64[0];
    std::uint32_t hash32;
    MurmurHash3_x86_32((void *)&x, sizeof(std::uint32_t), _seed,
                       (void *)&hash32);
    return hash32;
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
    ss << name() << "<" << RangeSize << ">";
    return ss.str();
  }

  static constexpr std::size_t size() { return RangeSize; }
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
template <typename HashType = krowkee::hash::HyperLogLogHash<16>>
class HyperLogLog {
 public:
  using hash_type       = HashType;
  using index_type      = typename hash_type::index_type;
  using zero_count_type = typename hash_type::zero_count_type;
  using update_type     = typename hash_type::update_type;
  using self_type       = HyperLogLog;

 protected:
  krowkee::hash::HyperLogLogHash<16> _hash;
  uint32_t                           m_;        ///< register size
  uint8_t                            b_;        ///< register bit width
  double                             alphaMM_;  ///< alpha * m^2
  std::vector<uint8_t>               M_;        ///< registers

 public:
  /**
   * Constructor
   *
   * @param[in] b bit width (register size will be 2 to the b power).
   *            This value must be in the range[4,30].Default value is 4.
   *
   * @exception std::invalid_argument the argument is out of range.
   */
  HyperLogLog(uint32_t m = 16, std::uint32_t seed = 1)
      : _hash(seed), m_(m), b_(std::log2(m_)), M_(m_, 0) {
    // std::cout << "(b, m) = (" << int(b_) << ", " << m_ << ")" << std::endl;
    if (b_ < 4 || 30 < b_) {
      throw std::invalid_argument("bit width must be in the range [4,30]");
    }

    double alpha;
    switch (m_) {
      case 16:
        alpha = 0.673;
        break;
      case 32:
        alpha = 0.697;
        break;
      case 64:
        alpha = 0.709;
        break;
      default:
        alpha = 0.7213 / (1.0 + 1.079 / m_);
        break;
    }
    alphaMM_ = alpha * m_ * m_;
  }

  // Assignment operators
  HyperLogLog(const HyperLogLog &)            = default;
  HyperLogLog(HyperLogLog &&)                 = default;
  HyperLogLog &operator=(HyperLogLog &)       = default;
  HyperLogLog &operator=(const HyperLogLog &) = default;

  template <typename ObjType>
  constexpr update_type apply(const ObjType &obj) const {
    index_type      hash  = _hash(obj);
    index_type      index = hash >> (32 - b_);
    zero_count_type zero_count =
        (uint8_t)std::min(32 - b_, std::countl_zero(hash << b_)) + 1;
    return {index, zero_count};
  }
  /**
   * Inserts element to the data structure
   *
   * @param[in] str string to insert
   * @param[in] len length of string
   */
  void insert(uint32_t obj) {
    const auto [index, zero_count] = apply(obj);
    if (zero_count > M_[index]) {
      M_[index] = zero_count;
    }
    // std::cout << std::bitset<32>(hash) << std::endl;
    // std::cout << std::bitset<4>(index) << std::endl;
    // std::cout << std::bitset<32>(hash << b_) << std::endl;
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
    for (uint32_t i = 0; i < m_; i++) {
      sum += 1.0 / (1 << M_[i]);
    }
    estimate = alphaMM_ / sum;  // E in the original paper
    if (estimate <= 2.5 * m_) {
      uint32_t zeros = 0;
      for (uint32_t i = 0; i < m_; i++) {
        if (M_[i] == 0) {
          zeros++;
        }
      }
      if (zeros != 0) {
        estimate = m_ * std::log(static_cast<double>(m_) / zeros);
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
    if (m_ != other.m_) {
      std::stringstream ss;
      ss << "number of registers doesn't match: " << m_ << " != " << other.m_;
      throw std::invalid_argument(ss.str().c_str());
    }
    for (uint32_t r = 0; r < m_; ++r) {
      if (M_[r] < other.M_[r]) {
        M_[r] |= other.M_[r];
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
  void clear() { std::fill(M_.begin(), M_.end(), 0); }

  /**
   * Returns size of register.
   *
   * @return Register size
   */
  uint32_t registerSize() const { return m_; }

  /**
   * Exchanges the content of the instance
   *
   * @param[in,out] rhs Another HyperLogLog instance
   */
  void swap(HyperLogLog &rhs) {
    std::swap(m_, rhs.m_);
    std::swap(alphaMM_, rhs.alphaMM_);
    M_.swap(rhs.M_);
  }

  constexpr const std::vector<uint8_t> &registers() const { return M_; }

  /**
   * Restore the status from a stream
   *
   * @param[in] is The input stream where the status is saved
   *
   * @exception std::runtime_error When failed to restore.
   */
  void restore(std::istream &is) {
    uint8_t b = 0;
    is.read((char *)&b, sizeof(b));
    HyperLogLog tempHLL(b);
    is.read((char *)&(tempHLL.M_[0]), sizeof(M_[0]) * tempHLL.m_);
    if (is.fail()) {
      throw std::runtime_error("Failed to restore");
    }
    swap(tempHLL);
  }
};

}  // namespace krowkee::transform

#endif  // !defined(HYPERLOGLOG_HPP)