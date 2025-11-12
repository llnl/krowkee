// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <krowkee/sketch.hpp>
#include <krowkee/util/runtime.hpp>

#if __has_include(<ygm/comm.hpp>)
#include <ygm/comm.hpp>

template <typename T>
using ptr_type = ygm::ygm_ptr<T>;

template <typename T>
ygm::ygm_ptr<T> _make_ygm_ptr(T &t) {
  return ygm::ygm_ptr<T>(&t);
}

/**
 * Functor wrapping _make_ygm_ptr
 */
template <typename T>
struct make_ygm_ptr_functor {
  using ptr_type = ygm::ygm_ptr<T>;

  make_ygm_ptr_functor() {}

  template <typename... Args>
  ptr_type operator()(Args... args) {
    sptrs.push_back(std::make_unique<T>(args...));
    return _make_ygm_ptr<T>(*(sptrs.back()));
  }

  static constexpr std::string name() { return "ygm::ygm_ptr"; }

 private:
  std::vector<std::unique_ptr<T>> sptrs;
};

template <typename T>
using make_ptr_functor = make_ygm_ptr_functor<T>;
#else

template <typename T>
using ptr_type = std::shared_ptr<T>;

template <typename T>
using make_ptr_functor = krowkee::make_shared_functor<T>;
#endif

using register_type = float;

namespace dense {

template <std::size_t RangeSize, std::size_t ReplicationCount>
using SparseJLT = krowkee::sketch::SparseJLT<register_type, RangeSize,
                                             ReplicationCount, ptr_type>;

template <std::size_t RangeSize, std::size_t ReplicationCount>
using FWHT =
    krowkee::sketch::FWHT<register_type, RangeSize, ReplicationCount, ptr_type>;
}  // namespace dense

namespace matrix {
template <std::size_t RangeSize, std::size_t ReplicationCount>
using DoubleSparseJLT =
    krowkee::sketch::DoubleSparseJLT<register_type, RangeSize, ReplicationCount,
                                     ptr_type>;
}

namespace sparse {
namespace map {
template <std::size_t RangeSize, std::size_t ReplicationCount>
using SparseJLT = krowkee::sketch::sparse::SparseJLT<
    register_type, RangeSize, ReplicationCount,
    RangeSize * ReplicationCount / 16, std::map, ptr_type>;
}

#if __has_include(<boost/container/flat_map.hpp>)
namespace flatmap {
template <std::size_t RangeSize, std::size_t ReplicationCount>
using SparseJLT = krowkee::sketch::sparse::SparseJLT<
    register_type, RangeSize, ReplicationCount,
    RangeSize * ReplicationCount / 16, boost::container::flat_map, ptr_type>;
}
#endif

}  // namespace sparse

namespace promotable {
namespace map {
template <std::size_t RangeSize, std::size_t ReplicationCount>
using SparseJLT = krowkee::sketch::promotable::SparseJLT<
    register_type, RangeSize, ReplicationCount,
    RangeSize * ReplicationCount / 16, RangeSize * ReplicationCount / 4,
    std::map, ptr_type>;
}
#if __has_include(<boost/container/flat_map.hpp>)
namespace flatmap {
template <std::size_t RangeSize, std::size_t ReplicationCount>
using SparseJLT = krowkee::sketch::promotable::SparseJLT<
    register_type, RangeSize, ReplicationCount,
    RangeSize * ReplicationCount / 16, RangeSize * ReplicationCount / 4,
    boost::container::flat_map, ptr_type>;
}
#endif
}  // namespace promotable
