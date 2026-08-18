// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/hash.hpp>
// #include "xxhash.h"

#if __has_include(<cereal/types/base_class.hpp>)
#include <cereal/types/base_class.hpp>
#endif

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <random>

namespace krowkee {
namespace hash {

#if defined(__has_builtin) && (defined(__GNUC__) || defined(__clang__))

#define _GET_CLZ(x, max_zero_count) \
  (uint8_t)std::min((int)max_zero_count, ::__builtin_clz(x)) + 1

#else

inline uint8_t _get_leading_zero_count(uint32_t x, uint8_t max_zero_count) {

#if defined(_MSC_VER)
  uint32_t leading_zero_len = 32;
  ::_BitScanReverse(&leading_zero_len, x);
  --leading_zero_len;
  return std::min(max_zero_count, (uint8_t)leading_zero_len);
#else
  uint8_t v = 1;
  while (v <= max_zero_count && !(x & 0x80000000)) {
    v++;
    x <<= 1;
  }
  return v;
#endif
}
#define _GET_CLZ(x, max_zero_count) _get_leading_zero_count(x, max_zero_count)
#endif /* defined(__GNUC__) */

/**
 * HyperLogLog hash base class
 *
 * A fast implementation of the HyperLogLog hash operators. Returns a pair
 * of hash values into the register space [0, R] and [0, M], where R is the
 * number of registers - 1 and M is the maximum number of allowed leading zeros,
 * respectively.
 *
 * @tparam RangeSize The power-of-two register count.
 * @tparam HashType The hash functor.
 */
template <std::size_t RangeSize,
          template <std::size_t> class HashType = MulAddShift>
struct HyperLogLogHash {
  // This is currently hard coded for 32bit hashes. Will need to be redone for
  // 64 bit. TODO: replace with murmurhash.
  using hash_type = HashType<1uLL << 32>;
  using self_type = HyperLogLogHash<RangeSize, HashType>;

  // check that RangeSize is a power of 2
  static_assert((RangeSize > 0 && (RangeSize & (RangeSize - 1)) == 0),
                "RangeSize must be a power of 2 for HyperLogLog hash");

 protected:
  hash_type                    _hash;
  static constexpr std::size_t _range_width    = std::log2(RangeSize);
  static constexpr std::size_t _max_zero_count = 32 - _range_width;

 public:
  /**
   * @param seed the random seed controlling any randomness.
   */
  template <typename... Args>
  HyperLogLogHash(const std::uint64_t seed, const Args &...args)
      : _hash(seed, args...) {}

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
  constexpr std::pair<std::uint32_t, std::uint8_t> operator()(
      const OBJ &x) const {
    std::uint32_t hash  = _hash(x);
    std::uint32_t index = hash >> _max_zero_count;
    std::uint8_t  zero_count =
        _GET_CLZ((hash << _range_width), _max_zero_count) - 1;
    // std::cout << std::bitset<32>(hash) << std::endl;
    // std::cout << std::bitset<_range_width>(index) << std::endl;
    // std::cout << std::bitset<_max_zero_count>(zero_count) << std::endl;
    // exit(0);
    return {index, zero_count};
  }

  /**
   * Print functor name.
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "HyperLogLogHash";
    return ss.str();
  }

  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << "HyperLogLogHash<" << hash_type::full_name() << ">";
    return ss.str();
  }

  static constexpr std::size_t size() { return RangeSize; }
  static constexpr std::size_t max_zero_count() { return _max_zero_count; }
  static constexpr std::size_t range_width() { return _range_width; }
  constexpr std::size_t        seed() const { return _hash.seed(); }

  constexpr std::string state() const {
    std::stringstream ss;
    ss << "size: " << size() << ", seed: " << seed();
    return ss.str();
  }

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_hash);
  }
#endif

  friend void swap(self_type &lhs, self_type &rhs) {
    std::swap(lhs._hash, rhs._hash);
  }
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    return lhs._hash == rhs._hash;
  }

  friend constexpr bool operator!=(const self_type &lhs, const self_type &rhs) {
    return !operator==(lhs, rhs);
  }

  friend std::ostream &operator<<(std::ostream &os, const self_type &func) {
    os << func.state();
    return os;
  }
};

}  // namespace hash
}  // namespace krowkee
