// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/transform/fwht/utils.hpp>

#include <krowkee/transform/Element.hpp>

#include <sstream>
#include <vector>

namespace krowkee {
namespace transform {

using krowkee::stream::Element;

/// good reference for operator declarations:
/// https://stackoverflow.com/questions/4421706/what-are-the-basic-rules-and-idioms-for-operator-overloading

/**
 * Fast Walsh-Hadamard Transform Functor Class
 */
template <typename RegType, std::size_t RangeSize, std::size_t ReplicationCount>
class FWHT {
 public:
  using register_type = RegType;
  using self_type     = FWHT<RegType, RangeSize, ReplicationCount>;

 protected:
  std::uint64_t _range_size = RangeSize;
  std::uint64_t _seed;
  std::uint64_t _domain_size;

 public:
  /**Quick
   * Initialize members.
   *
   * @note[BWP] Do we actually need to pass n?
   *
   * @tparam Args type(s) of additional hash parameters.
   *
   * @param seed the random seed.
   * @param args any additional paramters required by the hash functions.
   */
  template <typename... Args>
  FWHT(const std::uint64_t seed, const std::uint64_t domain_size = 1024,
       const Args &&...args)
      : _seed(seed), _domain_size(domain_size) {}

  FWHT() {}

  //////////////////////////////////////////////////////////////////////////////
  // Cereal Archives
  //////////////////////////////////////////////////////////////////////////////

#if __has_include(<cereal/cereal.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(_range_size, _seed, _domain_size);
  }
#endif

  //////////////////////////////////////////////////////////////////////////////
  // Function: Apply to Container
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Update a vector of registers with an observation.
   *
   * @tparam ContainerType The type of the underlying sketch data structure.
   * @tparam ItemArgs... types of parameters of the stream object to be
   *     inserted. Will be used to construct an krowkee::stream::Element object.
   *
   * @param[out] registers the vector of registers.
   * @param[in] x the object to be inserted.
   * @param[in] multiplicity a multiple to modulate insertion.
   */
  template <typename ContainerType, typename... ItemArgs>
  constexpr void operator()(ContainerType &registers,
                            const ItemArgs &...item_args) const {
    static_assert(std::is_same<register_type,
                               typename ContainerType::register_type>::value);
    const Element<RegType> stream_element(item_args...);
    const std::uint64_t    col_index    = stream_element.item;
    const std::uint64_t    row_index    = stream_element.identifier;
    const RegType          multiplicity = stream_element.multiplicity;
    std::uint64_t          seed(_seed);
    for (int i(0); i < ReplicationCount; ++i) {
      std::vector<RegType> sketch_vec =
          krowkee::transform::fwht::get_sketch_vector(multiplicity, row_index,
                                                      col_index, _domain_size,
                                                      _range_size, seed);
      std::size_t offset = i * range_size();
      auto        begin  = std::begin(registers) + offset;
      auto        end    = begin + range_size();
      std::transform(begin, end, std::begin(sketch_vec), begin,
                     std::plus<RegType>());
      seed = krowkee::hash::wang64(seed);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  //////////////////////////////////////////////////////////////////////////////

  static constexpr std::size_t range_size() { return RangeSize; }

  static constexpr std::size_t replication_count() { return ReplicationCount; }

  static constexpr std::size_t size() {
    return self_type::range_size() * self_type::replication_count();
  }

  static constexpr RegType scaling_factor = std::sqrt((RegType)size());

  constexpr std::uint64_t seed() const { return _seed; }

  constexpr std::uint64_t domain_size() const { return _domain_size; }

  static constexpr std::string name() {
    std::stringstream ss;
    ss << "FWHT<" << RangeSize << ", " << ReplicationCount << ", "
       << sizeof(RegType) << ">";
    return ss.str();
  }

  static constexpr std::string full_name() { return name(); }

  friend constexpr bool operator==(const self_type &lhs, const self_type &rhs) {
    return lhs.seed() == rhs.seed() && lhs.range_size() == rhs.range_size();
  }

  friend constexpr bool operator!=(const self_type &lhs, const self_type &rhs) {
    return !operator==(lhs, rhs);
  }

  friend std::ostream &operator<<(std::ostream &os, const self_type &func) {
    os << func.range_size() << " " << func.domain_size() << " " << func.seed();
    return os;
  }
};

}  // namespace transform
}  // namespace krowkee
