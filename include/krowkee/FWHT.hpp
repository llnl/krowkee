// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/transform/FWHT.hpp>

#include <krowkee/merge/plus.hpp>

#include <krowkee/sketch/Dense.hpp>
#include <krowkee/sketch/Sketch.hpp>

#include <map>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
// Sketch Type Presets
////////////////////////////////////////////////////////////////////////////////

namespace krowkee::sketch {

template <typename RegType, std::size_t RangeSize, std::size_t ReplicationCount,
          template <typename> class PtrType = std::shared_ptr>
using FWHT = krowkee::sketch::Sketch<
    krowkee::transform::FWHT<RegType, RangeSize, ReplicationCount>,
    krowkee::sketch::Dense<RegType, krowkee::merge::plus,
                           RangeSize * ReplicationCount>,
    PtrType>;

}  // namespace krowkee::sketch
