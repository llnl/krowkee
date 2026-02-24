// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <cstdint>
#include <memory>
#include <sstream>

namespace krowkee {
namespace hash {

const std::uint64_t default_seed = 1082087245;

/**
 * Thomas Wang hash function (base 64).
 *
 * @relatesalso WangHash, MulShift, MulAddShift
 */
constexpr std::uint64_t wang64(const std::uint64_t &x) {
  std::uint64_t y(x);
  y = (~y) + (y << 21);  // y = (y << 21) - y - 1;
  y = y ^ (y >> 24);
  y = (y + (y << 3)) + (y << 8);  // y * 265
  y = y ^ (y >> 14);
  y = (y + (y << 2)) + (y << 4);  // y * 21
  y = y ^ (y >> 28);
  y = y + (y << 31);
  return y;
}

/**
 * Check if `val` is even.
 *
 * @tparam UIntType the unsigned integer type of the input
 *
 * @param val the unsigned int to be evaluated.
 */
template <typename UIntType>
constexpr bool is_even(const UIntType val) {
  return val & 1;
}

/**
 * Compute log2 of `RangeSize` at compile time. Assumes `RangeSize` is a power
 * of 2.
 *
 * CURRENTLY NOT PORTABLE. Uses `__builtin_clzll`.
 *
 * @tparam RangeSize The int to be evaluated. Assumed to be a power of 2.
 */
template <std::size_t RangeSize>
struct log2_64 {
  static constexpr std::size_t value = 63 - __builtin_clzll(RangeSize);
};

/**
 * Checks if `RangeSize` is a power of 2 at compile time.
 *
 * @tparam RangeSize The int to be evaluated.
 */
template <std::size_t RangeSize>
struct is_power_of_2 {
  static constexpr bool value =
      (RangeSize > 0) && !(RangeSize & (RangeSize - 1)) == 1;
};

/**
 * Print name of type
 *
 * Usage: `os << type_name<decltype(obj)>()`
 *
 * https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
 *
 * @tparam T the type to be interpreted/printed (via `decltype(T obj)`).
 */
template <class T>
std::string type_name() {
  using TR = typename std::remove_reference<T>::type;
  std::unique_ptr<char, void (*)(void *)> own(
#ifndef _MSC_VER
      abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
#else
      nullptr,
#endif
      std::free);
  std::string r = own != nullptr ? own.get() : typeid(TR).name();
  if (std::is_const<TR>::value) r += " const";
  if (std::is_volatile<TR>::value) r += " volatile";
  if (std::is_lvalue_reference<T>::value)
    r += "&";
  else if (std::is_rvalue_reference<T>::value)
    r += "&&";
  return r;
}
}  // namespace hash
}  // namespace krowkee
