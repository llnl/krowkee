// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "krowkee/hash/murmur3.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <utility>

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
