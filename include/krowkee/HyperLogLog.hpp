// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/countsketch.hpp>
#include <krowkee/hash/hash.hpp>

#include <krowkee/transform/SparseJLT.hpp>

#include <krowkee/merge/plus.hpp>

#include <krowkee/sketch/Dense.hpp>

#include <krowkee/sketch/Sketch.hpp>

#include <map>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
// Sketch Type Presets
////////////////////////////////////////////////////////////////////////////////

namespace krowkee::sketch {

template <typename RegType, std::size_t RangeSize,
          template <typename> class PtrType = std::shared_ptr>
using HyperLogLog32 = krowkee::sketch::Sketch<
    krowkee::transform::HyperLogLog<RegType, krowkee::hash::HyperLogLogHash32,
                                    RangeSize>,
    krowkee::sketch::Dense<RegType, krowkee::merge::max, RangeSize>, PtrType>;

}  // namespace krowkee::sketch
