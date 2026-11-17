// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <parameters.hpp>
#include <sketch_test_templates.hpp>
#include <sketch_types.hpp>

#include <krowkee/hash/hll.hpp>
#include <krowkee/merge/max.hpp>
#include <krowkee/sketch/Dense.hpp>
#include <krowkee/sketch/Sketch.hpp>
#include <krowkee/transform/HyperLogLog.hpp>

#include <krowkee/HyperLogLog.hpp>

#include <krowkee/util/runtime.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>

#if __has_include(<cereal/cereal.hpp>)
#include <cereal/types/base_class.hpp>
#include <check_archive.hpp>
#endif

#include <cstdint>

template <std::size_t RangeSize>
using hll_sketch_type =
    krowkee::sketch::HyperLogLog32<std::uint8_t, RangeSize, ptr_type>;

template <template <typename> class MakePtrFunc>
struct guarantee_check {
  using Universe = std::map<unsigned int, std::set<unsigned int>>;

  const char *name() { return "HLL guarantee check"; }

  void operator()(ygm::comm &comm, const auto &params) const {
    test_universe(comm, params, 5, 3);
    check_guarantee<32>(comm, params);
    check_guarantee<128>(comm, params);
    check_guarantee<512>(comm, params);
  }

  template <typename... Args>
  void chirp(ygm::comm &comm, const auto &params, Args &&...args) const {
    if (params.verbose()) {
      comm.cout0(args...);
    }
  }

  void check_universe_sizes(ygm::comm &comm, const auto &params,
                            const Universe &universe, const int level,
                            const int target) const {
    chirp(comm, params, "universe ", level, " size: ", universe.size());
    bool success = true;
    for (const auto &[idx, set] : universe) {
      if (params.verbose()) {
        std::stringstream ss;
        ss << "set " << idx << " (size " << set.size() << "):\t";
        for (const unsigned int &elt : set) {
          ss << elt << " ";
        }
        comm.cout0(ss.str());
      }
      success = (set.size() == target);
    }
    std::stringstream ss;
    ss << "set universe level " << level << " size";
    CHECK_CONDITION(comm, success == true, ss.str());
  }

  void test_universe(ygm::comm &comm, const auto &params,
                     const unsigned int coeff,
                     const unsigned int exponent) const {
    unsigned int target   = coeff;
    Universe     universe = build_universe(coeff, exponent);
    check_universe_sizes(comm, params, universe, 0, target);
    for (unsigned int level(1); level <= exponent; ++level) {
      universe = get_next_universe(universe);
      target *= 2;
      check_universe_sizes(comm, params, universe, level, target);
    }
  }

  Universe get_next_universe(const Universe &last_universe) const {
    Universe next_universe;

    for (const auto &[i, neighbors] : last_universe) {
      auto &next_neighbors = next_universe[i];
      next_neighbors       = neighbors;

      for (unsigned int j : neighbors) {
        const auto &j_neighbors = last_universe.at(j);
        next_neighbors.insert(j_neighbors.begin(), j_neighbors.end());
      }
    }

    return next_universe;
  }

  unsigned int universe_size(unsigned int c, unsigned int k) const {
    const unsigned int  H   = 1u << k;
    const std::uint64_t n64 = static_cast<std::uint64_t>(c) * H;

    if (n64 > std::numeric_limits<unsigned int>::max()) {
      throw std::overflow_error("c * 2^k does not fit in unsigned int");
    }

    return static_cast<unsigned int>(n64);
  }

  Universe build_universe(unsigned int c, unsigned int k) const {
    if (k >= 16) {
      throw std::invalid_argument("k must be less than 16");
    }
    const unsigned int n = universe_size(c, k);
    const unsigned int H = 1u << k;
    Universe           universe;

    // Label each vertex (a, h) as a * H + h.
    // Its initial set contains every (b, h + 1 mod H).
    for (unsigned int a = 0; a < c; ++a) {
      for (unsigned int h = 0; h < H; ++h) {
        const unsigned int i         = a * H + h;
        auto              &neighbors = universe[i];

        const unsigned int next_h = (h + 1) % H;
        for (unsigned int b = 0; b < c; ++b) {
          neighbors.insert(b * H + next_h);
        }
      }
    }

    return universe;
  }

  template <typename ContainerType>
  void test_level_guarantee(ygm::comm &comm, const auto &params,
                            const ContainerType &universe, const double target,
                            const double target_rate, const int level) const {
    using index_type               = typename ContainerType::key_type;
    using sketch_type              = typename ContainerType::mapped_type;
    const int         world_size   = comm.size();
    const std::size_t range_size   = sketch_type::range_size();
    const std::string full_name    = sketch_type::full_name();
    const double      thresh       = 1.5 * (1.04 / std::sqrt(range_size));
    double            mean_err     = 0.0;
    double            success_rate = 0.0;
    universe.for_all([&comm, &params, &target, &target_rate, &level,
                      &world_size, &thresh, &mean_err, &success_rate](
                         const index_type &idx, const sketch_type &sketch) {
      double est = sketch_type::transform_type::estimate(sketch.container());
      double err = std::abs(target - est) / target;

      bool guarantee_success = (err < thresh);

      mean_err += err;
      success_rate += guarantee_success;
      if (idx == 0 & params.verbose()) {
        comm.cout("\tLevel ", level, " (tru, est, err): (", target, ", ", est,
                  ", ", err, ")");
      }
    });
    comm.barrier();

    mean_err     = ygm::sum(mean_err, comm) / world_size;
    success_rate = ygm::sum(success_rate, comm) / world_size;
    chirp(comm, params, "\t(range_size, mean err, threshold, success_rate) (",
          range_size, ", ", mean_err, ", ", thresh, ", ", success_rate, ")");
    {
      bool success = (mean_err < thresh);
      if (success == false) {
        comm.cout0("Level ", level, " mean estimated error is ", mean_err,
                   ", which exceeds expected threshold of ", thresh);
      }
      std::stringstream ss;
      ss << full_name << " Level " << level
         << " mean estimated error threshold";
      CHECK_CONDITION(comm, success == true, ss.str());
    }
    {
      bool success = (success_rate >= target_rate);
      if (success == false) {
        comm.cout0("Level ", level, " estimates achieve guarantee with rate ",
                   success_rate, ", which is below expected threshold of ",
                   target_rate);
      }
      std::stringstream ss;
      ss << full_name << " Level " << level << " guarantee success rate";
      CHECK_CONDITION(comm, success == true, ss.str());
    }
  }

  template <std::size_t RangeSize>
  void check_guarantee(ygm::comm &comm, const auto &params) const {
    using sketch_type        = hll_sketch_type<RangeSize>;
    using transform_type     = typename sketch_type::transform_type;
    using transform_ptr_type = typename sketch_type::transform_ptr_type;
    using make_ptr_type      = MakePtrFunc<transform_type>;

    const int          world_size = comm.size();
    const int          rank       = comm.rank();
    const double       target_rate(0.5);
    make_ptr_type      _make_ptr{};
    transform_ptr_type transform_ptr(_make_ptr(params.seed() + rank));
    ygm::container::map<const int, sketch_type> universe(comm);
    sketch_type                                 sketch(transform_ptr);
    for (std::uint64_t j(0); j < params.count(); ++j) {
      sketch.insert(j);
    }
    universe.async_insert(rank, sketch);

    test_level_guarantee(comm, params, universe, params.count(), target_rate,
                         1);

    return;
  }
};

void perform_tests(ygm::comm &world, const auto &params) {
  using sketch_type = hll_sketch_type<16>;

  krowkee::print_line(world);
  krowkee::print_line(world);
  world.cout0("Testing ", sketch_type::full_name());
  world.cout0("\tUsing std::shared_ptr pointers");
  krowkee::print_line(world);
  krowkee::print_line(world);

  world.cout0("\n");

  krowkee::do_ygm_test<guarantee_check<make_ptr_functor>>(world, world, params);
}

int main(int argc, char **argv) {
  using parameters = krowkee::test::hash::parameters;

  ygm::comm world(&argc, &argv);
  {
    // Using krowkee requires the selection of a sketch type for both a single
    // and double-sided sketch, here encapsulated as `single_sketch_type` and
    // `double_sketch_type`, respectively. We use the `SparseJLT` and
    // `DoubleSparseJLT` types defined in the simple APIs available in the top
    // level krowkee include directory. These types have four template
    // parameters:
    //   1. the numeric type to be used by each register (here `double`),
    //   2. a `std::size_t` parameter `range_size` indicating the number of
    //      registers used by each instance of the internal transform,
    //   3. a `std::size_t` parameter `replication_count` indicating the number
    //      of instances of the transform to be used, and
    //   4. a shared pointer type to be used by the shared transform object
    //      (`std::shared_ptr` for shared memory implementations).

    parameters params =
        krowkee::test::chirp_parameters<parameters>(world, argc, argv);

    perform_tests(world, params);
  }
  return 0;
}