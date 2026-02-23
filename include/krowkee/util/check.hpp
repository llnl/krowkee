// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/util/assert.hpp>

#if __has_include(<ygm/comm.hpp>)
#include <ygm/comm.hpp>
#endif

template <typename... Args>
inline void CHECK_CONDITION(const bool success, const Args &...args) {
  std::cout << ((success == true) ? "passed" : "failed") << " ";
  (std::cout << ... << args);
  std::cout << " test" << std::endl;
  KROWKEE_ASSERT_RELEASE(success);
}

template <typename ExceptType, typename FuncType, typename... Args>
inline void CHECK_THROWS(const FuncType &func, const std::string &msg,
                         Args &...args) {
  bool        caught = false;
  std::string str;
  try {
    func(args...);
  } catch (ExceptType &e) {
    str    = e.what();
    caught = true;
  }
  if (caught == true) {
    std::cout << "caught expected " << msg << " exception:\n\t" << str
              << std::endl;
  } else {
    std::cout << "failed to catch expected " << msg << " exception"
              << std::endl;
  }
  KROWKEE_ASSERT_RELEASE(caught);
}

template <typename ExceptType, typename FuncType, typename... Args>
inline void CHECK_DOES_NOT_THROW(const FuncType &func, const std::string &msg,
                                 const Args &...args) {
  try {
    func(args...);
  } catch (ExceptType &e) {
    std::cout << msg << " incorrectly threw exception \"" << e.what() << "\""
              << std::endl;
    KROWKEE_ASSERT_RELEASE(false);
  }
}

#if __has_include(<ygm/comm.hpp>)
template <typename... Args>
inline void CHECK_CONDITION(ygm::comm &comm, const bool success,
                            const Args &...args) {
  std::stringstream ss;
  (ss << ... << args);
  comm.cout0(((success == true) ? "passed" : "failed"), " ", ss.str(), " test");
  KROWKEE_ASSERT_RELEASE(success);
}
#endif
