// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/hash/countsketch.hpp>
#include <krowkee/hash/hash.hpp>

#include <krowkee/transform/DoubleSparseJLT.hpp>
#include <krowkee/transform/FWHT.hpp>
#include <krowkee/transform/SparseJLT.hpp>

#include <krowkee/sketch/Dense.hpp>
#include <krowkee/sketch/Matrix.hpp>
#include <krowkee/sketch/Promotable.hpp>
#include <krowkee/sketch/Sparse.hpp>

#include <krowkee/sketch/Sketch.hpp>

#if __has_include(<boost/container/flat_map.hpp>)
#include <boost/container/flat_map.hpp>
#endif

#include <map>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
// Sketch Type Presets
////////////////////////////////////////////////////////////////////////////////

namespace krowkee {
namespace sketch {

template <typename RegType, std::size_t RangeSize, std::size_t ReplicationCount,
          template <typename> class PtrType = std::shared_ptr>
using SparseJLT = krowkee::sketch::Sketch<
    krowkee::transform::SparseJLT<RegType, krowkee::hash::CountSketchHash,
                                  RangeSize, ReplicationCount>,
    krowkee::sketch::Dense<RegType, std::plus, RangeSize * ReplicationCount>,
    PtrType>;

template <typename RegType, std::size_t RangeSize, std::size_t ReplicationCount,
          template <typename> class PtrType = std::shared_ptr>
using DoubleSparseJLT = krowkee::sketch::Sketch<
    krowkee::transform::DoubleSparseJLT<RegType, krowkee::hash::CountSketchHash,
                                        RangeSize, ReplicationCount>,
    krowkee::sketch::Matrix<RegType, std::plus, RangeSize * ReplicationCount,
                            RangeSize * ReplicationCount>,
    PtrType>;

template <typename RegType, std::size_t RangeSize, std::size_t ReplicationCount,
          template <typename> class PtrType = std::shared_ptr>
using FWHT = krowkee::sketch::Sketch<
    krowkee::transform::FWHT<RegType, RangeSize, ReplicationCount>,
    krowkee::sketch::Dense<RegType, std::plus, RangeSize * ReplicationCount>,
    PtrType>;

namespace sparse {
template <typename RegType, std::size_t RangeSize, std::size_t ReplicationCount,
          template <typename, typename> class MapType,
          template <typename> class PtrType = std::shared_ptr>
using SparseJLT = krowkee::sketch::Sketch<
    krowkee::transform::SparseJLT<RegType, krowkee::hash::CountSketchHash,
                                  RangeSize, ReplicationCount>,
    krowkee::sketch::Sparse<RegType, std::plus, MapType, std::uint32_t,
                            RangeSize * ReplicationCount>,
    PtrType>;

}  // namespace sparse

namespace promotable {
template <typename RegType, std::size_t RangeSize, std::size_t ReplicationCount,
          std::size_t PromotionThreshold,
          template <typename, typename> class MapType,
          template <typename> class PtrType = std::shared_ptr>
using SparseJLT = krowkee::sketch::Sketch<
    krowkee::transform::SparseJLT<RegType, krowkee::hash::CountSketchHash,
                                  RangeSize, ReplicationCount>,
    krowkee::sketch::Promotable<RegType, std::plus, MapType, std::uint32_t,
                                RangeSize * ReplicationCount,
                                PromotionThreshold>,
    PtrType>;

}  // namespace promotable

}  // namespace sketch
}  // namespace krowkee
