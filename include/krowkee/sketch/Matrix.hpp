// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#if __has_include(<cereal/cereal.hpp>)
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <ygm/detail/ygm_cereal_archive.hpp>
#endif

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
}  // namespace krowkee::sketch::detail
namespace krowkee {
namespace sketch {

/// good reference for operator declarations:
/// https://stackoverflow.com/questions/4421706/what-are-the-basic-rules-and-idioms-for-operator-overloading

/**
 * @brief General Matrix Sketch Container
 *
 * Implements the container handling infrastructure of any sketch whose atomic
 * elements include a fixed size set of register values organized into a matrix,
 * and supporting a merge operation consisting of the application of an
 * element-wise operator on pairs of register matrices.
 *
 * @note MergeOp template parameter is currently unused and may be removed in a
 * future update.
 *
 * @tparam RegType The type held by each register.
 * @tparam MergeOp An template merge operator to combine two sketches.
 * @tparam RowCount The maximum number of rows. Assumed equal to ColCount.
 * @tparam ColCount The maximum number of columns. Assumed equal to RowCount.
 */
template <typename RegType, template <typename> class MergeOp,
          std::size_t RowCount, std::size_t ColCount>
class Matrix {
 public:
  using register_type = RegType;
  using registers_type =
      Eigen::Matrix<register_type, Eigen::Dynamic, Eigen::Dynamic>;
  using dense_registers_type =
      Eigen::Matrix<register_type, Eigen::Dynamic, Eigen::Dynamic>;
  using merge_type = MergeOp<register_type>;
  using self_type  = Matrix<register_type, MergeOp, RowCount, ColCount>;

  /**
   * Currently assuming that Matrix Objects are always square, but preparing for
   * a world where they aren't.
   */
  static_assert(RowCount == ColCount);

 protected:
  static const std::size_t Size = RowCount * ColCount;
  registers_type           _registers;

 public:
  /**
   * @brief Construct a new Matrix container object. Currently assuming that
   * Matrix objects are always square.
   */
  Matrix() : _registers(registers_type::Zero(RowCount, ColCount)) {}

  /**
   * @brief Copy constructor.
   *
   * @param rhs The base Matrix container to copy.
   */
  Matrix(const self_type &rhs) : _registers(rhs._registers) {}

  //////////////////////////////////////////////////////////////////////////////
  // Swaps
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Swap two Matrix containers.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   */
  friend void swap(self_type &lhs, self_type &rhs) {
    std::swap(lhs._registers, rhs._registers);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Cereal Archives
  //////////////////////////////////////////////////////////////////////////////

#if __has_include(<cereal/types/vector.hpp>)
  /**
   * @brief Serialize Matrix object to/from `cereal` archive.
   *
   * @tparam Archive `cereal` archive type.
   * @param archive The `cereal` archive to which to serialize the sketch.
   */
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_registers);
  }
#endif

  //////////////////////////////////////////////////////////////////////////////
  // Compactify
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief A no-op for dense containers.
   */
  void compactify() {}

  //////////////////////////////////////////////////////////////////////////////
  // Clear & Empty
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Set all registers to 0.
   */
  void clear() { _registers.setZero(RowCount, ColCount); }

  /**
   * @brief Check if all registers are 0.
   */
  bool empty() const { return _registers.isZero(0); }

  //////////////////////////////////////////////////////////////////////////////
  // Erase
  //////////////////////////////////////////////////////////////////////////////

  constexpr void erase(const std::pair<std::uint64_t, std::uint64_t> indices) {}

  //////////////////////////////////////////////////////////////////////////////
  // Merge operators
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Merge other Matrix registers into `this`.
   *
   * @param rhs the other Matrix.
   */
  constexpr void merge(const self_type &rhs) { _registers += rhs._registers; }

  /**
   * @brief Operator overload for convenience for embeddings without additional
   * consistency checks.
   *
   * @param rhs the other Matrix.
   * @return self_type& `this` Matrix, having been merged with `rhs`.
   */
  self_type &operator+=(const self_type &rhs) {
    merge(rhs);
    return *this;
  }

  /**
   * @brief Merge two Matrix containers.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   * @return self_type The merge of the two container objects.
   */
  constexpr friend self_type operator+(self_type lhs, const self_type &rhs) {
    lhs += rhs;
    return lhs;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Register iterators
  //////////////////////////////////////////////////////////////////////////////

  /** Mutable begin iterator. */
  constexpr typename registers_type::iterator begin() {
    return std::begin(_registers.reshaped());
  }
  /** Const begin iterator. */
  constexpr typename registers_type::const_iterator begin() const {
    return std::cbegin(_registers.reshaped());
  }
  /** Const begin iterator. */
  constexpr typename registers_type::const_iterator cbegin() const {
    return std::cbegin(_registers.reshaped());
  }
  /** Mutable end iterator. */
  constexpr typename registers_type::iterator end() {
    return std::end(_registers.reshaped());
  }
  /** Const end iterator. */
  constexpr typename registers_type::const_iterator end() const {
    return std::cend(_registers.reshaped());
  }
  /** Const end iterator. */
  constexpr typename registers_type::const_iterator cend() {
    return std::cend(_registers.reshaped());
  }

  /**
   * @brief Const access Matrix at `indices` pair.
   *
   * @param indices The row and column indices of the underlying matrix to
   * index. Must be less than `row_count` and `col_count`, respectively.
   * @return constexpr const register_type& A const reference to the object at
   * the indicated register.
   */
  constexpr const register_type &operator[](
      const std::pair<std::uint64_t, std::uint64_t> &indices) const {
    return _registers(indices.first, indices.second);
  }
  /**
   * @brief Access Matrix at `indices` pair.
   *
   * @param indices The row and column indices of the underlying matrix to
   * index. Must be less than `row_count` and `col_count`, respectively.
   * @return register_type& A reference to the object at the indicated register.
   */
  register_type &operator[](
      const std::pair<std::uint64_t, std::uint64_t> &indices) {
    return _registers(indices.first, indices.second);
  }

  /**
   * @brief Const access Matrix at `(row_idx, col_idx)`.
   *
   * @param row_idx The row index of the underlying matrix. Must be less than
   * `row_count`.
   * @param col_idx The column index of the underlying matrix. Must be less
   * than `col_count`.
   * @return constexpr const register_type& A const reference to the object at
   * the indicated register.
   */
  constexpr const register_type &operator()(
      const std::uint64_t &row_idx, const std::uint64_t &col_idx) const {
    return _registers(row_idx, col_idx);
  }
  /**
   * @brief Access Matrix at `(row_idx, col_idx)`.
   *
   * @param row_idx The row index of the underlying matrix. Must be less than
   * `row_count`.
   * @param col_idx The column index of the underlying matrix. Must be less
   * than `col_count`.
   * @return register_type& A reference to the object at the indicated register.
   */
  register_type &operator()(const std::uint64_t &row_idx,
                            const std::uint64_t &col_idx) {
    return _registers(row_idx, col_idx);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Returns a description of the type of container.
   *
   * @return std::string "Matrix"
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "Matrix<" << RowCount << ", " << ColCount << ">";
    return ss.str();
  }

  /**
   * @brief Returns a description of the fully-qualified type of container.
   *
   * @return std::string "Matrix"
   */
  static constexpr std::string full_name() { return name(); }

  /** Matrix is also not Sparse. */
  constexpr bool is_sparse() const { return false; }

  /** The size of the registers vector. */
  static constexpr std::size_t size() { return Size; }

  /** The number of rows in the registers matrix. */
  static constexpr std::size_t row_count() { return RowCount; }

  /** The number of columns in the registers matrix. */
  static constexpr std::size_t col_count() { return ColCount; }

  /** The size of the registers vector. */
  static constexpr std::size_t max_size() { return Size; }

  /** The size of the embedding. Equal to RowCount and ColCount. Assuming that
   * the matrix is square.
   */
  static constexpr std::size_t embedding_size() { return RowCount; }

  /** The number of bytes used by each register. */
  constexpr std::size_t reg_size() const { return sizeof(register_type); }

  constexpr std::size_t compaction_threshold() const { return 0; }

  /**
   * @brief Get a copy of the raw registers vector.
   *
   * @return const registers_type The register vector.
   */
  const registers_type &registers() const { return _registers; }

  /**
   * @brief Get a copy of the scaled registers vector.
   *
   * @param scaling_factor the scalar scaling factor.
   * @return const registers_type The register vector.
   */
  registers_type scaled_registers(double scaling_factor) const {
    return _registers / scaling_factor;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Equality operators
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Checks whether another Dense container has the same register state.
   *
   * @note allows for small disagreements in floating point representations that
   * may arise from merging, so will return true even in cases where register
   * sets are not exactly equivalent.
   *
   * @param rhs The other container.
   * @return true The registers agree.
   * @return false At least one register disagrees.
   */
  constexpr bool same_registers(const self_type &rhs) const {
    return detail::allclose(_registers, rhs._registers);
  }

  /**
   * @brief Checks equality between two Dense containers.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   * @return true The registers agree.
   * @return false At least one register disagrees.
   */
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    return lhs.same_registers(rhs);
  }

  /**
   * @brief CHecks inequality between two Dense containers.
   *
   * @param lhs The left-hand container.
   * @param rhs The right-hand container.
   * @return true At least one register disagrees.
   * @return false The registers agree.
   */
  friend constexpr bool operator!=(const self_type &lhs, const self_type &rhs) {
    return !operator==(lhs, rhs);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Assignment
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Copy-and-swap assignment operator
   *
   * @note
   * https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
   *
   * @param rhs The other container.
   * @return self_type& `this`, having been swapped with `rhs`.
   */
  self_type &operator=(self_type rhs) {
    std::swap(*this, rhs);
    return *this;
  }

  //////////////////////////////////////////////////////////////////////////////
  // I/O Operators
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Serialize a Dense container to human-readable output stream.
   *
   * Output format is a space-delmined list of (key, value) pairs.
   *
   * @note Intended for debugging only. Now using Eigen's default print style.
   *
   * @param os The output stream.
   * @param sk The Dense object.
   * @return std::ostream& The new stream state.
   */
  friend std::ostream &operator<<(std::ostream &os, const self_type &sk) {
    os << sk._registers;
    return os;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Accumulation
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Accumulate sum of register values.
   *
   * @tparam RetType The value type to return.
   * @param sk The Dense to accumulate.
   * @param init Initial value of return.
   * @return RetType The sum over all register values + `init`.
   */
  template <typename RetType>
  friend RetType accumulate(const self_type &sk, const RetType init) {
    return std::accumulate(std::cbegin(sk._registers.reshaped()),
                           std::cend(sk._registers.reshaped()), init);
  }

  /**
   * @brief Apply a given function to all register values.
   *
   * @tparam Func The (probably lambda) function type.
   * @param sk The Dense object.
   * @param func The particular function to apply.
   */
  template <typename Func>
  friend void for_each(const self_type &sk, const Func &func) {
    std::for_each(std::cbegin(sk._registers.reshaped()),
                  std::cend(sk._registers.reshaped()), func);
  }
};

}  // namespace sketch
}  // namespace krowkee
