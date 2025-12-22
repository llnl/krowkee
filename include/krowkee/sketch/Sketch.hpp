// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#if __has_include(<cereal/types/memory.hpp>)
#include <cereal/types/memory.hpp>
#endif

#include <type_traits>

namespace krowkee {
namespace sketch {

/**
 * @brief General Sketch Chassis.
 *
 * Implements the container handling infrastructure of a linear sketch
 * defined by a given sketch functor projecting into a fixed-size set of
 * registers supporting an (element-wise) merge operator. The registers might be
 * Dense, Sparse, or variable, depending on the templated ContainerType.
 *
 * @tparam SketchFunc A sketch functor such as
 * `krowkee::transform::CountSketchFunctor` whose first template parameter is
 * set by `RegType`.
 * @tparam ContainerType A container class such as `krowkee::sketch::Dense`.
 * @tparam PtrType The type of shared pointer used to wrap the sketch functor.
 * Should be `std::shared_ptr`in shared memory and `ygm::ygm_ptr` in distributed
 * memory.
 * @tparam Args Additional template parameters for `SketchFunc`, such as a hash
 * functor.
 */
template <typename SketchFunc, typename ContainerType,
          template <typename> class PtrType>
class Sketch {
 public:
  /** Alias for fully-templated sketch functor type*/
  using transform_type = SketchFunc;
  /** Alias for fully-templated sketch functor pointer type*/
  using transform_ptr_type = PtrType<transform_type>;
  /** Alias for Register type */
  using register_type = typename transform_type::register_type;
  /** Alias for fully-templated container type */
  using container_type = ContainerType;
  /** Alias for fully-templated raw container type */
  using registers_type = typename container_type::dense_registers_type;
  /** Alias for fully-templated self type*/
  using self_type = Sketch<transform_type, ContainerType, PtrType>;

  static_assert(std::is_same<register_type,
                             typename container_type::register_type>::value);
  static_assert(SketchFunc::size() == ContainerType::embedding_size());

 private:
  transform_ptr_type _transform_ptr;
  container_type     _container;

 public:
  /**
   * @brief Construct a new Sketch object
   *
   * @param sf_ptr pointer desired linear sketch functor.
   */
  Sketch(const transform_ptr_type &sf_ptr)
      : _transform_ptr(sf_ptr), _container() {}

  /**
   * @brief Copy constructor
   *
   * @param rhs `krowkee::sketch::Sketch` to be copied
   */
  Sketch(const self_type &rhs)
      : _transform_ptr(rhs._transform_ptr), _container(rhs._container) {}

  /**
   * @brief default constructor
   *
   * @note Only used for move constructor.
   */
  Sketch() {}

  //////////////////////////////////////////////////////////////////////////////
  // Cereal Archives
  //////////////////////////////////////////////////////////////////////////////

#if __has_include(<cereal/types/memory.hpp>)
  /**
   * @brief Serialize Sketch object to/from `cereal` archive.
   *
   * @tparam Archive `cereal` archive type
   * @param archive The `cereal` archive to which to serialize the sketch.
   */
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_transform_ptr, _container);
  }
#endif

  /**
   * @brief Return a reference to the underlying container data structure.
   *
   * @note For testing purposes. Might want to get rid of this.
   *
   * @return const container_type& A reference to the internal register
   * container object.
   */
  const container_type &container() { return _container; }

  registers_type scaled_registers() const {
    return _container.scaled_registers(transform_type::scaling_factor);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Insertion
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Insert item into registers using sketch functor.
   *
   * Example could involve `(x, multiplicity)` where `x` is the object to insert
   * and `multiplicity` is the number of insert repetitions to perform.
   *
   * @tparam ItemArgs types of parameters of the stream object to be inserted.
   * Will be used to construct an krowkee::stream::Element object.
   * @param args Arguments describing the stream object.
   */
  template <typename... ItemArgs>
  constexpr void insert(const ItemArgs &...args) {
    (*_transform_ptr)(_container, args...);
  }

  /**
   * @brief Insert item into registers using sketch functor, assuming a tuple of
   * indices as the first argument.
   *
   * Example could involve `({i, j}, multiplicity)` where `i` and `j` are the
   * row and columnn indices of the matrix insertion, respectively, and
   * `multiplicity` is the number of insert repetitions to perform.
   *
   * @tparam ItemArgs types of parameters of the stream object to be inserted.
   * Will be used to construct an krowkee::stream::Element object.
   * @param indices Describes data structure indices to be inserted.
   * @param args Addtional arguments describing the stream object.
   */
  template <typename... ItemArgs>
  constexpr void insert(const std::pair<std::uint64_t, std::uint64_t> &indices,
                        const ItemArgs &...args) {
    (*_transform_ptr)(_container, indices, args...);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Compactify
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Signal sketch container to compactify, if it supports such
   * functionality.
   *
   */
  void compactify() { _container.compactify(); }

  //////////////////////////////////////////////////////////////////////////////
  // Clear & Empty
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Signal sketch container to remove all information.
   */
  void clear() { _container.clear(); }

  /**
   * @brief Check if container is holding any state.
   */
  bool empty() const { return _container.empty(); }

  //////////////////////////////////////////////////////////////////////////////
  // Merge Operators
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Merge other Sketch registers into `this`.
   *
   * @param rhs the other Sketch embedding. Must use the same random
   *     seed.
   * @return self_type& The merge of the two sketches.
   * @throws std::invalid_argument if the `SketchFunc`s do not agree.
   */
  self_type &operator+=(const self_type &rhs) {
    if (!same_functors(rhs)) {
      std::stringstream ss;
      ss << "error: attempting to merge linear sketch objects with different "
            "hash functors : ("
         << *(_transform_ptr) << ") and (" << *(rhs._transform_ptr) << ")";
      throw std::invalid_argument(ss.str());
    }
    _container += rhs._container;
    return *this;
  }

  /**
   * @brief Merge two sketches
   *
   * @param lhs The left-hand sketch object.
   * @param rhs The right-hand sketch object.
   * @return self_type The merge of the two sketch objects.
   */
  constexpr friend self_type operator+(const self_type &lhs,
                                       const self_type &rhs) {
    self_type ret(lhs);
    ret += rhs;
    return ret;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Register iterators
  //////////////////////////////////////////////////////////////////////////////

  /** Mutable begin iterator. */
  constexpr auto begin() { return std::begin(_container); }
  /** Const begin iterator. */
  constexpr auto begin() const { return std::cbegin(_container); }
  /** Const begin iterator. */
  constexpr auto cbegin() const { return std::cbegin(_container); }
  /** Mutable end iterator. */
  constexpr auto end() { return std::end(_container); }
  /** Const end iterator. */
  constexpr auto end() const { return std::cend(_container); }
  /** Const end iterator. */
  constexpr auto cend() { return std::cend(_container); }

  // constexpr const std::uint8_t &operator[](const std::uint64_t index) const {
  //   return _container[index];
  // }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Return a description of the type of container and transform that
   * this sketch implements.
   *
   * @return std::string Sketch description, e.g. "Dense CountSketch"
   */
  static constexpr std::string name() {
    std::stringstream ss;
    ss << "Sketch<" << transform_type::name() << ", " << container_type::name()
       << ">";
    return ss.str();
  }

  /**
   * @brief Return a description of the fully-qualified type of container and
   * transform that this sketch implements.
   *
   * @return std::string Sketch description, e.g. "Promotable using
   * std::map<unsigned int, int, std::less<unsigned int>,
   * std::allocator<std::pair<unsigned int const, int> > > CountSketch using
   * MulAddShift hashes and 4 byte registers"
   */
  static constexpr std::string full_name() {
    std::stringstream ss;
    ss << "Sketch<" << transform_type::full_name() << ", "
       << container_type::full_name() << ">";
    return ss.str();
  }

  /**
   * @brief Check for sparsity of container.
   *
   * @return true The container is sparse.
   * @return false The container is not sparse.
   */
  constexpr bool is_sparse() const { return _container.is_sparse(); }

  /**
   * @brief Get the number of registers, real or notional, in the container.
   *
   * @return constexpr std::size_t The size of the container.
   */
  constexpr std::size_t size() const { return _container.size(); }

  /** The number of bytes used by each register. */
  constexpr std::size_t reg_size() const { return sizeof(register_type); }

  /**
   * @brief Get the number of possible range elements returned by the sketch
   * functor.
   *
   * @return constexpr std::size_t The size of the sketch functor range.
   */
  static constexpr std::size_t range_size() {
    return transform_type::range_size();
  }

  /**
   * @brief Get the compaction threshold of the container.
   *
   * @return constexpr std::size_t The compaction threshold.
   */
  constexpr std::size_t compaction_threshold() const {
    return _container.compaction_threshold();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Equality operators
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Checks whether another sketch is using the same functor.
   *
   * @param rhs The other sketch.
   * @return true The functors agree.
   * @return false The functors disagree.
   */
  constexpr bool same_functors(const self_type &rhs) const {
    return *_transform_ptr == *rhs._transform_ptr;
  }

  /**
   * @brief Checks equality between sketch objects.
   *
   * Two sketches are equivalent if they are using the same sketch functor and
   * their registers hold the same values.
   *
   * @param lhs The left-hand sketch.
   * @param rhs The right-hand sketch.
   * @return true The sketches are equivalent.
   * @return false The sketches disagree.
   */
  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    if (!lhs.same_functors(rhs)) {
      return false;
    }
    return lhs._container == rhs._container;
  }

  /**
   * @brief Check inequality between two sketch objects.
   *
   * @param lhs The left-hand sketch.
   * @param rhs The right-hand sketch.
   * @return true The sketches disagree.
   * @return false The sketches are equivalent.
   */
  friend constexpr bool operator!=(const self_type &lhs, const self_type &rhs) {
    return !operator==(lhs, rhs);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Swaps
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Swap two Sketch objects.
   *
   * @note For some reason, calling std::swap on the container here causes a "no
   * matching function compiler error". Peculiar.
   *
   * @param lhs The left-hand sketch.
   * @param rhs The right-hand sketch.
   */
  friend void swap(self_type &lhs, self_type &rhs) {
    std::swap(lhs._transform_ptr, rhs._transform_ptr);
    std::swap(lhs._container, rhs._container);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Assignment
  //////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Copy-and-swap assignment operator.
   *
   * @note
   * https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
   *
   * @param rhs The right-hand sketch.
   * @return self_type& The new value of this sketch, swapped with `rhs`.
   */
  self_type &operator=(self_type rhs) {
    std::swap(*this, rhs);
    return *this;
  }

  //////////////////////////////////////////////////////////////////////////////
  // I/O Operators
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Serialize a sketch to human-readable output stream.
   *
   * @note Intended for debugging only.
   *
   * @param os The output stream.
   * @param sk The sketch object.
   * @return std::ostream& The new stream state.
   */
  friend std::ostream &operator<<(std::ostream &os, const self_type &sk) {
    os << sk._container;
    return os;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Accumulation
  //////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Accumulate sum of register values.
   *
   * @tparam RetType The value type to return.
   * @param sk The sketch to accumulate.
   * @param init Initial value of return.
   * @return RetType The sum over all register values + `init`.
   */
  template <typename RetType>
  friend RetType accumulate(const self_type &sk, const RetType init) {
    return accumulate(sk._container, init);
  }
};

}  // namespace sketch
}  // namespace krowkee
