// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <graph.hpp>

#include <krowkee/hash/hash.hpp>
#include <krowkee/sketch.hpp>

template <typename SketchType, typename ValueType>
struct HistSketch {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  HistSketch(const transform_ptr_type &transform_ptr, const auto &params)
      : _sketch(transform_ptr) {}

  void insert(const ValueType &idx) { _sketch.insert(idx); }

  std::string static full_name() { return sketch_type::full_name(); }

 private:
  sketch_type _sketch;
};

template <typename SketchType, typename ValueType>
struct VectorGraphSketch {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  VectorGraphSketch(const transform_ptr_type &transform_ptr, const auto &params)
      : _sketch_vec() {
    for (int i(0); i < params.domain_size(); ++i) {
      _sketch_vec.push_back(std::make_unique<sketch_type>(transform_ptr));
    }
  }

  void insert(const edge_type<ValueType> &edge) {
    _sketch_vec[edge.first]->insert(edge.second);
  }

  std::string static full_name() {
    std::stringstream ss;
    ss << "std::vector graph of " << sketch_type::full_name();
    return ss.str();
  }

 private:
  std::vector<std::unique_ptr<SketchType>> _sketch_vec;
};

template <typename SketchType, typename ValueType>
struct MapGraphSketch {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  MapGraphSketch(const transform_ptr_type &transform_ptr, const auto &params)
      : _sketch_map(), _transform_ptr(transform_ptr) {}

  void insert(const edge_type<ValueType> &edge) {
    auto [src, dst] = edge;
    auto range      = _sketch_map.equal_range(src);
    if (range.first == range.second) {
      _sketch_map.insert(
          std::make_pair(src, std::make_unique<sketch_type>(_transform_ptr)));
    }
    _sketch_map[src]->insert(dst);
  }

  std::string static full_name() {
    std::stringstream ss;
    ss << "std::map graph of " << sketch_type::full_name();
    return ss.str();
  }

 private:
  std::map<ValueType, std::unique_ptr<sketch_type>> _sketch_map;
  transform_ptr_type                                _transform_ptr;
};

// histogram sketch types

const static std::size_t RANGE_SIZE(32);
const static std::size_t REPLICATION_COUNT(1);
const static std::size_t COMPACTION_THRESHOLD(8);
const static std::size_t PROMOTION_THRESHOLD(32);

using dense_type =
    krowkee::sketch::SparseJLT<std::int32_t, RANGE_SIZE, REPLICATION_COUNT>;
using sparse_map_type =
    krowkee::sketch::sparse::SparseJLT<std::int32_t, RANGE_SIZE,
                                       REPLICATION_COUNT, COMPACTION_THRESHOLD,
                                       std::map>;
using promotable_map_type = krowkee::sketch::promotable::SparseJLT<
    std::int32_t, RANGE_SIZE, REPLICATION_COUNT, COMPACTION_THRESHOLD,
    PROMOTION_THRESHOLD, std::map>;

template <typename ValueType>
using dense32_cs_hist = HistSketch<dense_type, ValueType>;
template <typename ValueType>
using map_sparse32_cs_hist = HistSketch<sparse_map_type, ValueType>;
template <typename ValueType>
using map_promotable32_cs_hist = HistSketch<promotable_map_type, ValueType>;

// vector graph sketch types

template <typename ValueType>
using dense32_cs_vector_graph = VectorGraphSketch<dense_type, ValueType>;
template <typename ValueType>
using map_sparse32_cs_vector_graph =
    VectorGraphSketch<sparse_map_type, ValueType>;
template <typename ValueType>
using map_promotable32_cs_vector_graph =
    VectorGraphSketch<promotable_map_type, ValueType>;

// map graph sketch types

template <typename ValueType>
using dense32_cs_map_graph = MapGraphSketch<dense_type, ValueType>;
template <typename ValueType>
using map_sparse32_cs_map_graph = MapGraphSketch<sparse_map_type, ValueType>;
template <typename ValueType>
using map_promotable32_cs_map_graph =
    MapGraphSketch<promotable_map_type, ValueType>;

#if __has_include(<boost/container/flat_map.hpp>)

using sparse_flatmap_type =
    krowkee::sketch::sparse::SparseJLT<std::int32_t, RANGE_SIZE,
                                       REPLICATION_COUNT, COMPACTION_THRESHOLD,
                                       boost::container::flat_map>;
using promotable_flatmap_type = krowkee::sketch::promotable::SparseJLT<
    std::int32_t, RANGE_SIZE, REPLICATION_COUNT, COMPACTION_THRESHOLD,
    PROMOTION_THRESHOLD, boost::container::flat_map>;

template <typename ValueType>
using flatmap_sparse32_cs_hist = HistSketch<sparse_flatmap_type, ValueType>;

template <typename ValueType>
using flatmap_promotable32_cs_hist =
    HistSketch<promotable_flatmap_type, ValueType>;
template <typename ValueType>
using flatmap_sparse32_cs_vector_graph =
    VectorGraphSketch<sparse_flatmap_type, ValueType>;

template <typename ValueType>
using flatmap_promotable32_cs_vector_graph =
    VectorGraphSketch<promotable_flatmap_type, ValueType>;

template <typename ValueType>
using flatmap_sparse32_cs_map_graph =
    MapGraphSketch<sparse_flatmap_type, ValueType>;

template <typename ValueType>
using flatmap_promotable32_cs_map_graph =
    MapGraphSketch<promotable_flatmap_type, ValueType>;
#endif
