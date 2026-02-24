// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>

// work  on this:  https://github.com/lattera/glibc/blob/master/assert/assert.c
void krowkee_release_assert_fail(const char *assertion, const char *file,
                                 unsigned int line, const char *function) {
  std::stringstream ss;
  ss << " " << assertion << " " << file << ":" << line << " " << function
     << std::endl;
  throw std::runtime_error(ss.str());
}

#define KROWKEE_ASSERT_DEBUG(expr) assert(expr)

#define KROWKEE_ASSERT_RELEASE(expr) \
  (static_cast<bool>(expr)           \
       ? void(0)                     \
       : krowkee_release_assert_fail(#expr, __FILE__, __LINE__, ""))
