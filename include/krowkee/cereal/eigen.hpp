// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <ygm/detail/ygm_cereal_archive.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <sstream>
#include <vector>

#if __has_include(<cereal/cereal.hpp>)
// Based upon
// https://stackoverflow.com/questions/22884216/serializing-eigenmatrix-using-cereal-library
namespace cereal {
// cereal for Eigen::PlainObject binary archive
template <class Archive, class Derived>
inline typename std::enable_if<
    traits::is_output_serializable<BinaryData<typename Derived::Scalar>,
                                   Archive>::value,
    void>::type
CEREAL_SAVE_FUNCTION_NAME(Archive                               &archive,
                          Eigen::PlainObjectBase<Derived> const &array) {
  using array_type = Eigen::PlainObjectBase<Derived>;
  if (array_type::RowsAtCompileTime == Eigen::Dynamic) {
    archive(array.rows());
  }
  if (array_type::ColsAtCompileTime == Eigen::Dynamic) {
    archive(array.cols());
  }
  archive(binary_data(array.data(),
                      array.size() * sizeof(typename Derived::Scalar)));
}

template <class Archive, class Derived>
inline typename std::enable_if<
    traits::is_input_serializable<BinaryData<typename Derived::Scalar>,
                                  Archive>::value,
    void>::type
CEREAL_LOAD_FUNCTION_NAME(Archive                         &archive,
                          Eigen::PlainObjectBase<Derived> &array) {
  using array_type  = Eigen::PlainObjectBase<Derived>;
  Eigen::Index rows = array_type::RowsAtCompileTime;
  Eigen::Index cols = array_type::ColsAtCompileTime;
  if (rows == Eigen::Dynamic) {
    archive(rows);
  }
  if (cols == Eigen::Dynamic) {
    archive(cols);
  }
  array.resize(rows, cols);
  archive(binary_data(array.data(),
                      static_cast<std::size_t>(
                          rows * cols * sizeof(typename Derived::Scalar))));
}
}  // namespace cereal
#endif
