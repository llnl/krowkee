// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/countsketch.hpp>
#include <krowkee/hash/hash.hpp>

#include <krowkee/transform/SparseJLT.hpp>

#include <krowkee/merge/plus.hpp>

#include <krowkee/sketch/Promotable.hpp>

#include <krowkee/sketch/Sketch.hpp>

#if __has_include(<boost/container/flat_map.hpp>)
#include <boost/container/flat_map.hpp>
#endif

#include <map>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
// Sketch Type Presets
////////////////////////////////////////////////////////////////////////////////

namespace krowkee::sketch::promotable {
template <typename RegType, std::size_t RangeSize, std::size_t ReplicationCount,
          std::size_t CompactionThreshold, std::size_t PromotionThreshold,
          template <typename, typename> class MapType,
          template <typename> class PtrType = std::shared_ptr>
using SparseJLT = krowkee::sketch::Sketch<
    krowkee::transform::SparseJLT<RegType, krowkee::hash::CountSketchHash,
                                  RangeSize, ReplicationCount>,
    krowkee::sketch::Promotable<RegType, krowkee::merge::plus, MapType,
                                std::uint32_t, RangeSize * ReplicationCount,
                                CompactionThreshold, PromotionThreshold>,
    PtrType>;

}  // namespace krowkee::sketch::promotable
