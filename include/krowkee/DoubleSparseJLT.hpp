// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/countsketch.hpp>
#include <krowkee/hash/hash.hpp>

#include <krowkee/transform/DoubleSparseJLT.hpp>

#include <krowkee/merge/plus.hpp>

#include <krowkee/sketch/Matrix.hpp>

#include <krowkee/sketch/Sketch.hpp>

#include <map>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
// Sketch Type Presets
////////////////////////////////////////////////////////////////////////////////

namespace krowkee::sketch {

template <typename RegType, std::size_t RowRangeSize,
          std::size_t RowReplicationCount,
          template <typename> class PtrType = std::shared_ptr,
          std::size_t ColRangeSize          = RowRangeSize,
          std::size_t ColReplicationCount   = RowReplicationCount>
using DoubleSparseJLT = krowkee::sketch::Sketch<
    krowkee::transform::DoubleSparseJLT<
        PtrType,
        krowkee::transform::SparseJLT<RegType, krowkee::hash::CountSketchHash,
                                      RowRangeSize, RowReplicationCount>,
        krowkee::transform::SparseJLT<RegType, krowkee::hash::CountSketchHash,
                                      ColRangeSize, ColReplicationCount>>,
    krowkee::sketch::Matrix<RegType, krowkee::merge::plus,
                            RowRangeSize * RowReplicationCount,
                            ColRangeSize * ColReplicationCount>,
    PtrType>;

}  // namespace krowkee::sketch
