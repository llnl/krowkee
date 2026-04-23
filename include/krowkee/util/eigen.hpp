// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <sstream>
#include <vector>

namespace krowkee::sketch::detail {
// Eigen-compatible implementation of numpy.allclose() derived from
// https://stackoverflow.com/questions/15051367/how-to-compare-vectors-approximately-in-eigen
template <typename DerivedLHS, typename DerivedRHS>
bool allclose(
    const Eigen::DenseBase<DerivedLHS>    &lhs,
    const Eigen::DenseBase<DerivedRHS>    &rhs,
    const typename DerivedLHS::RealScalar &rtol =
        Eigen::NumTraits<typename DerivedLHS::RealScalar>::dummy_precision() *
        10,
    const typename DerivedLHS::RealScalar &atol =
        Eigen::NumTraits<typename DerivedLHS::RealScalar>::epsilon() * 100) {
  // std::cout << "atol: " << atol << std::endl;
  // std::cout << "rtol: " << rtol << std::endl;
  // std::cout << "mean rhs: "
  //           << ((rtol *
  //                lhs.derived().array().abs().max(rhs.derived().array().abs()))
  //                   .sum() /
  //               lhs.size())
  //           << std::endl;
  // std::cout << "max error: "
  //           << (lhs.derived() - rhs.derived()).array().abs().maxCoeff()
  //           << std::endl;
  return ((lhs.derived() - rhs.derived()).array().abs() <=
          (atol +
           rtol * lhs.derived().array().abs().max(rhs.derived().array().abs())))
      .all();
}

template <typename DerivedLHS, typename DerivedRHS>
double mean_absolute_error(const Eigen::DenseBase<DerivedLHS> &lhs,
                           const Eigen::DenseBase<DerivedRHS> &rhs) {
  return (lhs.derived() - rhs.derived()).array().abs().sum() / lhs.size();
}

template <typename DerivedLHS, typename DerivedRHS>
double max_absolute_error(const Eigen::DenseBase<DerivedLHS> &lhs,
                          const Eigen::DenseBase<DerivedRHS> &rhs) {
  return (lhs.derived() - rhs.derived()).cwiseAbs().maxCoeff();
}

template <typename DerivedLHS, typename DerivedRHS>
double min_absolute_error(const Eigen::DenseBase<DerivedLHS> &lhs,
                          const Eigen::DenseBase<DerivedRHS> &rhs) {
  return (lhs.derived() - rhs.derived()).cwiseAbs().minCoeff();
}
}  // namespace krowkee::sketch::detail
