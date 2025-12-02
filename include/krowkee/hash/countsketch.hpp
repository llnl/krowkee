// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/hash.hpp>
// #include "xxhash.h"

#if __has_include(<cereal/types/base_class.hpp>)
#include <cereal/types/base_class.hpp>
#endif

#include <cstdint>
#include <random>

namespace krowkee {
namespace hash {

/**
 * 2-universal CountSketch hash functor base class
 *
 * A fast implementation of CountSketch hash operators. Returns a pair
 * of hash values into the register space and {+1, -1}, respectively, each drawn
 * from a separate (ideally) 2-universal hash functor.
 *
 * @tparam RangeSize The power-of-two embedding dimension.
 * @tparam HashType The hash functor, ideally a 2-universal hash family.
 */
template <std::size_t RangeSize,
          template <std::size_t> class HashType = MulAddShift>
struct CountSketchHash {
  using register_hash_type = HashType<RangeSize>;
  using polarity_hash_type = HashType<2>;
  using self_type          = CountSketchHash<RangeSize, HashType>;

 protected:
  register_hash_type _register_hash;
  polarity_hash_type _polarity_hash;

 public:
  /**
   * @param seed the random seed controlling any randomness.
   */
  template <typename... Args>
  CountSketchHash(const std::uint64_t seed, const Args &...args)
      : _register_hash(seed, args...), _polarity_hash(wang64(seed), args...) {}

  CountSketchHash() {}

  /**
   * Compute the register and polarity hash of an input.
   *
   * @tparam OBJ the object to be hashed. Presently must fit into a 64-bit
   *     register.
   *
   * @param x the obejct to be hashed.
   */
  template <typename OBJ>
  constexpr std::pair<std::uint64_t, std::int32_t> operator()(
      const OBJ &x) const {
    return {_register_hash(x), _polarity_hash(x) == 1 ? 1 : -1};
  }

  /**
   * Print functor name.
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "CountSketchHash";
    return ss.str();
  }

  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << "CountSketchHash<" << register_hash_type::full_name() << ", "
       << polarity_hash_type::full_name() << ">";
    return ss.str();
  }

  static constexpr std::size_t size() { return register_hash_type::size(); }
  constexpr std::size_t        seed() const { return _register_hash.seed(); }

  constexpr std::string state() const {
    std::stringstream ss;
    ss << "size: " << size() << ", seed: " << seed();
    return ss.str();
  }

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_register_hash);
    archive(_polarity_hash);
  }
#endif

  friend void swap(self_type &lhs, self_type &rhs) {
    std::swap(lhs._register_hash, rhs._register_hash);
    std::swap(lhs._polarity_hash, rhs._polarity_hash);
    lhs.swap(rhs);
  }
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    return lhs._register_hash == rhs._register_hash &&
           lhs._polarity_hash == lhs._polarity_hash;
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
