// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/util/check.hpp>

#if __has_include(<ygm/comm.hpp>)
#include <ygm/comm.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

namespace krowkee {

using Clock   = std::chrono::system_clock;
using ns_type = std::chrono::nanoseconds;

inline void print_line() {
  std::cout << "-----------------------------------------------------"
            << std::endl;
}

#if __has_include(<ygm/comm.hpp>)
inline void print_line(ygm::comm &comm) {
  comm.cout0("-----------------------------------------------------");
}
#endif

inline void chirp() { std::cout << "gets here" << std::endl; }

template <template <std::size_t> class Func, typename ReturnType,
          typename... Args>
ReturnType dispatch_with_sketch_sizes(const std::size_t &range_size,
                                      Args &...args) {
  switch (range_size) {
    case 4:
      return Func<4>{}(args...);
      break;
    case 8:
      return Func<8>{}(args...);
      break;
    case 16:
      return Func<16>{}(args...);
      break;
    case 32:
      return Func<32>{}(args...);
      break;
    case 64:
      return Func<64>{}(args...);
      break;
    case 128:
      return Func<128>{}(args...);
      break;
    case 256:
      return Func<256>{}(args...);
      break;
    case 512:
      return Func<512>{}(args...);
      break;
    default:
      throw std::logic_error(
          "no-replication dispatch_with_sketch_sizes() convenience function "
          "only accepts power-of-2 range size from 4-512. Hard-code or create "
          "a new dispatch function if you need an unsupported range size.");
  }
}

#define DISPATCH_LEVEL0_CASE(SIZE0, SIZE1, ...) \
  case SIZE0:                                   \
    return Func<SIZE0, SIZE1>{}(__VA_ARGS__);
#define DISPATCH_LEVEL1_CASE(SIZE1, SUB_FN, ...) \
  case SIZE1:                                    \
    _sizes.pop_back();                           \
    return SUB_FN<SIZE1>(__VA_ARGS__);
#define DISPATCH_ERROR_CASE(LEVEL, LOW, HIGH)                       \
  default:                                                          \
    std::stringstream ss;                                           \
    ss << "dispatch() convenience functor only accepts power-of-2 " \
          "sizes at level "                                         \
       << LEVEL << " from " << LOW << "-" << HIGH                   \
       << ". Hard-code or create a new dispatch functor "           \
          "if you need an unsupported range size.";                 \
    throw std::logic_error(ss.str());
#define DISPATCH_SMALL_CASES(CASE_MACRO, LEVEL, ...) \
  switch (_sizes.back()) {                           \
    CASE_MACRO(1, __VA_ARGS__);                      \
    CASE_MACRO(2, __VA_ARGS__);                      \
    CASE_MACRO(4, __VA_ARGS__);                      \
    CASE_MACRO(8, __VA_ARGS__);                      \
    DISPATCH_ERROR_CASE(LEVEL, 1, 8);                \
  }
#define DISPATCH_MEDIUM_CASES(CASE_MACRO, LEVEL, ...) \
  switch (_sizes.back()) {                            \
    CASE_MACRO(4, __VA_ARGS__);                       \
    CASE_MACRO(8, __VA_ARGS__);                       \
    CASE_MACRO(16, __VA_ARGS__);                      \
    CASE_MACRO(32, __VA_ARGS__);                      \
    DISPATCH_ERROR_CASE(LEVEL, 4, 32);                \
  }

template <template <std::size_t, std::size_t> class Func, typename ReturnType>
struct dispatch {
 private:
  std::vector<std::size_t> _sizes;

  template <std::size_t Size, typename... Args>
  ReturnType subdispatch_0(Args &...args) {
    DISPATCH_MEDIUM_CASES(DISPATCH_LEVEL0_CASE, 0, Size, args...)
  }

 public:
  template <typename... Sizes>
  dispatch(const Sizes &...sizes) : _sizes{sizes...} {}

  template <typename... Args>
  ReturnType operator()(Args &...args) {
    DISPATCH_SMALL_CASES(DISPATCH_LEVEL1_CASE, 1, subdispatch_0, args...);
  }
};

#undef DISPATCH_LEVEL0_CASE
#undef DISPATCH_LEVEL1_CASE
#undef DISPATCH_ERROR_CASE
#undef DISPATCH_SMALL_CASES
#undef DISPATCH_MEDIUM_CASES

template <typename FuncType, typename... Args>
void do_test(Args &&...args) {
  FuncType func;
  print_line();
  std::cout << func.name() << ":" << std::endl;
  print_line();
  auto start(Clock::now());
  func(args...);
  auto end(Clock::now());
  auto ns(std::chrono::duration_cast<ns_type>(end - start).count());
  print_line();
  std::cout << "\tTest time: " << ((double)ns / 1e9) << "s" << std::endl;
  std::cout << std::endl << std::endl;
}

#if __has_include(<ygm/comm.hpp>)
template <typename FuncType, typename... Args>
void do_ygm_test(ygm::comm &comm, Args &&...args) {
  FuncType func;
  print_line(comm);
  comm.cout0(func.name(), ":");
  print_line(comm);
  auto start(Clock::now());
  func(args...);
  comm.barrier();
  auto end(Clock::now());
  auto ns(std::chrono::duration_cast<ns_type>(end - start).count());
  print_line(comm);
  comm.cout0("\tTest time: ", ((double)ns / 1e9), "s\n\n");
}
#endif

/**
 * Functor wrapping std::make_shared
 */
template <typename T>
struct make_shared_functor {
  using ptr_type = std::shared_ptr<T>;

  make_shared_functor() {}

  template <typename... Args>
  ptr_type operator()(Args... args) {
    return std::make_shared<T>(args...);
  }

  static constexpr std::string name() { return "std::shared_ptr"; }
};

// See Knuth TAOCP vol 2, 3rd edition, page 232
class online_statistics {
 public:
  online_statistics() : _count(0) {}

  void clear() { _count = 0; }

  void push(double x) {
    ++_count;

    if (_count == 1) {
      _oldM = _newM = x;
      _oldS         = 0.0;
    } else {
      _newM = _oldM + (x - _oldM) / _count;
      _newS = _oldS + (x - _oldM) * (x - _newM);

      // set up for next iteration
      _oldM = _newM;
      _oldS = _newS;
    }
  }

  template <typename T>
  void push(std::vector<T> &arr) {
    std::for_each(std::begin(arr), std::end(arr),
                  [&](const T val) { push(val); });
  }

  constexpr int count() const { return _count; }

  constexpr double mean() const { return (_count > 0) ? _newM : 0.0; }

  constexpr double variance() const {
    return ((_count > 1) ? _newS / (_count - 1) : 0.0);
  }

  constexpr double M2() const { return ((_count > 1) ? _newS : 0.0); }

  constexpr double std_dev() const { return std::sqrt(variance()); }

 private:
  std::uint64_t _count;
  double        _oldM, _newM, _oldS, _newS;
};

}  // namespace krowkee