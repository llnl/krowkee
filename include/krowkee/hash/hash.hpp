// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/util.hpp>
// #include "xxhash.h"

#if __has_include(<cereal/types/base_class.hpp>)
#include <cereal/types/base_class.hpp>
#endif

#include <cstdint>
#include <random>

namespace krowkee {
namespace hash {
/**
 * Hash functor base class
 *
 * Right now only supports hashing to a power of 2 range.
 *
 * @tparam RangeSize The desired range for the hash function. Must be a power
 * of 2.
 */
template <std::size_t RangeSize>
struct Base {
  using self_type = Base<RangeSize>;

  static_assert(is_power_of_2<RangeSize>::value,
                "RangeSize must be a power of 2!");

  /**
   * @param seed the random seed controlling any randomness. Might be ignored.
   */
  Base(const std::uint64_t seed) : _seed(seed) {}

  Base() {}

  /**
   * Truncate a hashed 64-bit value to the desired range specified by
   * `_log2_kernel_range_size`.
   *
   * @param val the hashed value to be truncated.
   */
  constexpr std::uint64_t truncate(const std::uint64_t val) const {
    return val >> _log2_kernel_range_size;
  }

  static constexpr std::uint64_t get_range() { return _log2_kernel_range_size; }
  static constexpr std::size_t   range_size() { return RangeSize; }
  static constexpr std::size_t   size() {
    return _1u64 << (64 - _log2_kernel_range_size);
  }

  constexpr std::size_t seed() const { return _seed; }
  constexpr std::string state() const {
    std::stringstream ss;
    ss << "seed: " << seed();
    return ss.str();
  }

  void swap(self_type &rhs) { std::swap(_seed, rhs._seed); }

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_seed);
  }
#endif

  friend void swap(self_type &lhs, self_type &rhs) { lhs.swap(rhs); }

  friend std::ostream &operator<<(std::ostream &os, const self_type &func) {
    os << func.state();
    return os;
  }

 protected:
  static constexpr std::uint64_t _log2_kernel_range_size =
      64 - log2_64<RangeSize>::value;
  static constexpr std::uint64_t _1u64 = 1;
  std::uint64_t                  _seed;
};

/**
 * Thomas Wang hash functor
 *
 * https://naml.us/blog/tag/thomas-wang
 *
 * @tparam RangeSize The desired range for the hash function. Must be a power
 * of 2.
 */
template <std::size_t RangeSize>
struct WangHash : public Base<RangeSize> {
  using self_type = WangHash<RangeSize>;
  using base_type = Base<RangeSize>;

  /**
   * @tparam ARGS types of additional (ignored) hash functor arguments.
   * @param range the power of 2 range for the hash function.
   */
  template <typename... ARGS>
  WangHash(ARGS &...) : base_type() {}

  WangHash() : base_type() {}

  template <typename OBJ>
  constexpr std::uint64_t operator()(const OBJ &x) const {
    return this->truncate(wang64(std::uint64_t(x)));
  }

  /**
   * Print functor name.
   */
  static constexpr std::string name() { return "WangHash"; }

  /**
   * Print full functor name.
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << name() << "<" << self_type::size() << ">";
    return ss.str();
  }

  constexpr std::string state() const { return full_name(); }

  friend constexpr bool operator==(const WangHash &lhs, const WangHash &rhs) {
    return lhs._log2_kernel_range_size == rhs._log2_kernel_range_size &&
           lhs._seed == rhs._seed;
  }

  friend constexpr bool operator!=(const WangHash &lhs, const WangHash &rhs) {
    return !operator==(lhs, rhs);
  }
};

// /**
//  * XXHash functor
//  *
//  * https://github.com/Cyan4973/xxHash
//  */
// struct XXHash : public Base {
//   template <typename... ARGS>
//   XXHash(const std::uint64_t m, const std::uint64_t seed = default_seed,
//          const ARGS &&...)
//       : Base(m, std::uint32_t(seed)) {}

//   template <typename OBJ>
//   constexpr std::uint64_t operator()(const OBJ &x) const {
//     std::cout << "gets here" << std::endl;
//     return xxh::xxhash<64>(&x, 1, _seed);
//     // return std::uint64_t(XXH64(&x, sizeof(x), _seed));
//   }

//   inline std::string name() { return "XXHash"; }
// };

/**
 * multiply-shift hash
 *
 * https://en.wikipedia.org/wiki/Universal_hashing
 *
 * @tparam RangeSize The desired range for the hash function. Must be a power
 * of 2.
 */
template <std::size_t RangeSize>
struct MulShift : public Base<RangeSize> {
  using self_type = MulShift<RangeSize>;
  using base_type = Base<RangeSize>;

  /**
   * Initialize `_multiplicand` given the specified random `seed`.
   *
   * `_multiplicand` is randomly sampled from [0, 2^64 - 1].
   *
   * @tparam ARGS types of additional (ignored) hash functor arguments.
   *
   * @param seed the random seed.
   */
  template <typename... ARGS>
  MulShift(const std::uint64_t seed, const ARGS &...) : base_type(seed) {
    std::mt19937_64                              rnd_gen(wang64(this->_seed));
    std::uniform_int_distribution<std::uint64_t> udist(
        1, std::numeric_limits<std::uint64_t>::max());
    std::uint64_t multiplicand;
    do {
      multiplicand = udist(rnd_gen);
    } while (is_even(multiplicand));
    _multiplicand = multiplicand;
  }

  MulShift() : base_type() {}

  /**
   * Compute `_multiplicand * x mod _log2_kernel_range_size`.
   *
   * +1 is added to x to handle settings where x could be zero. This behavior
   * can be removed by defining the `KROWKEE_NOZERO` precompiler flag.
   *
   * @tparam OBJ the object to be hashed. Presently must fit into a 64-bit
   *     register.
   *
   * @param x the obejct to be hashed.
   */
  template <typename OBJ>
  constexpr std::uint64_t operator()(const OBJ &x) const {
#ifdef KROWKEE_NOZERO
    return this->truncate(_multiplicand * x);
#else
    return this->truncate(_multiplicand * (x + 1));
#endif
  }

  /**
   * Print functor name.
   */
  static constexpr std::string name() { return "MulShift"; }

  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << name() << "<" << self_type::size() << ">";
    return ss.str();
  }

  constexpr std::string state() const {
    std::stringstream ss;
    ss << full_name() << ", " << base_type::state()
       << ", multiplicand: " << _multiplicand;
    return ss.str();
  }

  friend void swap(MulShift &lhs, MulShift &rhs) {
    std::swap(lhs._multiplicand, rhs._multiplicand);
    lhs.swap(rhs);
  }

  friend constexpr bool operator==(const MulShift &lhs, const MulShift &rhs) {
    return lhs._log2_kernel_range_size == rhs._log2_kernel_range_size &&
           lhs._seed == rhs._seed && lhs._multiplicand == rhs._multiplicand;
  }

  friend constexpr bool operator!=(const MulShift &lhs, const MulShift &rhs) {
    return !operator==(lhs, rhs);
  }

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(cereal::base_class<base_type>(this), _multiplicand);
  }
#endif

 private:
  std::uint64_t _multiplicand;
};

/**
 * multiply-add-shift hash
 *
 * https://en.wikipedia.org/wiki/Universal_hashing
 *
 * @tparam RangeSize The desired range for the hash function. Must be a power
 * of 2.
 */
template <std::size_t RangeSize>
struct MulAddShift : public Base<RangeSize> {
  using self_type = MulShift<RangeSize>;
  using base_type = Base<RangeSize>;
  /**
   * Initialize `_multiplicand` and `_summand` given the specified random
   * `seed`.
   *
   * `_multiplicand` and `_summand` are randomly sampled from [0, 2^64 - 1] and
   * [0, range_size() - 1], respectively.
   *
   * @tparam ARGS types of additional (ignored) hash functor arguments.
   *
   * @param seed the random seed.
   */
  template <typename... ARGS>
  MulAddShift(const std::uint64_t seed, const ARGS &...) : base_type(seed) {
    std::mt19937_64                              rnd_gen(wang64(this->_seed));
    std::uniform_int_distribution<std::uint64_t> udist_multiplicand(
        0, std::numeric_limits<std::uint64_t>::max());
    std::uniform_int_distribution<std::uint64_t> udist_summand(0, this->size());
    std::uint64_t                                multiplicand;
    do {
      multiplicand = udist_multiplicand(rnd_gen);
    } while (is_even(multiplicand));
    _multiplicand = multiplicand;
    _summand      = udist_summand(rnd_gen);
  }

  MulAddShift() : base_type() {}

  /**
   * Compute `_multiplicand * x + _summand mod _log2_kernel_range_size`.
   *
   * +1 is added to x to handle settings where x could be zero. This behavior
   * can be removed by defining the `KROWKEE_NOZERO` precompiler flag.
   *
   * @tparam OBJ the object to be hashed. Presently must fit into a 64-bit
   *     register.
   *
   * @param x the obejct to be hashed.
   */
  template <typename OBJ>
  constexpr std::uint64_t operator()(const OBJ &x) const {
#ifdef KROWKEE_NOZERO
    return this->truncate(_multiplicand * x + _summand);
#else
    return this->truncate(_multiplicand * (x + 1) + _summand);
#endif
  }

  /**
   * Print functor name.
   */
  static constexpr std::string name() { return "MulAddShift"; }

  /**
   * Print full functor name.
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << name() << "<" << self_type::size() << ">";
    return ss.str();
  }

  constexpr std::string state() const {
    std::stringstream ss;
    ss << full_name() << ", " << base_type::state()
       << ", multiplicand: " << _multiplicand << ",  summand: " << _summand;
    return ss.str();
  }

  friend void swap(MulAddShift &lhs, MulAddShift &rhs) {
    std::swap(lhs._multiplicand, rhs._multiplicand);
    std::swap(lhs._summand, rhs._summand);
    lhs.swap(rhs);
  }

  friend constexpr bool operator==(const MulAddShift &lhs,
                                   const MulAddShift &rhs) {
    return lhs._log2_kernel_range_size == rhs._log2_kernel_range_size &&
           lhs._seed == rhs._seed && lhs._multiplicand == rhs._multiplicand &&
           lhs._summand == rhs._summand;
  }

  friend constexpr bool operator!=(const MulAddShift &lhs,
                                   const MulAddShift &rhs) {
    return !operator==(lhs, rhs);
  }

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(cereal::base_class<base_type>(this), _multiplicand, _summand);
  }
#endif

 private:
  std::uint64_t _multiplicand;
  std::uint64_t _summand;
};

}  // namespace hash
}  // namespace krowkee
