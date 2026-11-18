// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <parameters.hpp>
#include <sketch_test_templates.hpp>

#include <krowkee/HyperLogLog.hpp>

#include <krowkee/util/runtime.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>

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
    check_dist_guarantee<32>(comm, params);
    check_dist_guarantee<128>(comm, params);
    check_dist_guarantee<512>(comm, params);
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
    ss << "set universe level " << level << " size (" << target << ")";
    CHECK_CONDITION(comm, success == true, ss.str());
  }

  void test_universe(ygm::comm &comm, const auto &params,
                     const unsigned int coeff,
                     const unsigned int exponent) const {
    unsigned int n        = coeff * (1u << exponent);
    unsigned int level    = 0;
    unsigned int target   = level_size(n, coeff, level);
    Universe     universe = build_universe(coeff, exponent);
    Universe     next_universe(universe);
    check_universe_sizes(comm, params, universe, 0, target);
    for (unsigned int level(1); level <= max_level(n, coeff); ++level) {
      next_universe = get_next_universe(next_universe, universe);
      target        = level_size(n, coeff, level);
      check_universe_sizes(comm, params, next_universe, level, target);
    }
  }

  constexpr unsigned int level_size(const unsigned int n, const unsigned int c,
                                    const unsigned int t) const {
    return std::min(n, (t + 1) * (c - 1) + 1);
  }

  constexpr unsigned int max_level(const unsigned int n,
                                   const unsigned int c) const {
    return std::ceil(((n - 1) / (c - 1))) - 1;
  }

  Universe get_next_universe(const Universe &last_universe,
                             const Universe &first_universe) const {
    Universe next_universe(last_universe);
    for (const auto &[idx, neighbors] : first_universe) {
      for (const auto &nbr : neighbors) {
        const auto &last_nbhd = last_universe.at(idx);
        next_universe[nbr].insert(last_nbhd.begin(), last_nbhd.end());
      }
    }
    return next_universe;
  }

  Universe build_universe(unsigned int c, unsigned int k) const {
    if (c <= 2 || k <= 2 || k >= 8 || (c % 2 == 0)) {
      throw std::invalid_argument("Require odd c > 2 and 2 < k < 8");
    }

    constexpr unsigned int max_uint = std::numeric_limits<unsigned int>::max();

    if (c > (max_uint >> k)) {
      throw std::overflow_error("c * (1u << k) overflows unsigned int");
    }

    const unsigned int n       = c * (1u << k);
    const std::int64_t radius  = (c - 1) / 2;
    const std::int64_t modulus = n;

    Universe universe;

    for (unsigned int i = 0; i < n; ++i) {
      auto &neighbors = universe[i];

      for (std::int64_t delta = -radius; delta <= radius; ++delta) {
        std::int64_t value = (static_cast<std::int64_t>(i) + delta) % modulus;
        if (value < 0) {
          value += modulus;
        }

        neighbors.insert(static_cast<unsigned int>(value));
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

    mean_err     = ygm::sum(mean_err, comm) / universe.size();
    success_rate = ygm::sum(success_rate, comm) / universe.size();
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
         << " mean estimated error threshold (target: " << target
         << ", error: " << mean_err << " / " << thresh << ")";
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
      ss << full_name << " Level " << level << " guarantee success rate ("
         << success_rate << ")";
      CHECK_CONDITION(comm, success == true, ss.str());
    }
  }

  template <std::size_t RangeSize>
  void check_guarantee(ygm::comm &comm, const auto &params) const {
    using sketch_type        = hll_sketch_type<RangeSize>;
    using transform_type     = typename sketch_type::transform_type;
    using transform_ptr_type = typename sketch_type::transform_ptr_type;
    using make_ptr_type      = MakePtrFunc<transform_type>;

    const int          world_size  = comm.size();
    const int          rank        = comm.rank();
    const double       target_rate = 0.5;
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

  template <typename ContainerType>
  void create_and_test_next_nbhds(ContainerType  &last_universe,
                                  const Universe &exact_universe,
                                  const auto &params, const unsigned int coeff,
                                  const double       target_rate,
                                  const unsigned int level,
                                  const unsigned int exponent) const {
    using index_type  = typename ContainerType::key_type;
    using sketch_type = typename ContainerType::mapped_type;

    const unsigned int n         = coeff * (1u << exponent);
    const unsigned int level_max = max_level(n, coeff);
    const unsigned int target    = level_size(n, coeff, level);

    if (level > level_max) {
      return;
    }
    ContainerType next_universe(last_universe);

    last_universe.for_all([&next_universe, &exact_universe](
                              const index_type  &idx,
                              const sketch_type &last_sketch) {
      for (const index_type &nbr : exact_universe.at(idx)) {
        next_universe.async_visit(
            nbr,
            [](auto ptr, const index_type &idx, sketch_type &next_sketch,
               const sketch_type &last_sketch) { next_sketch += last_sketch; },
            last_sketch);
      }
    });
    last_universe.comm().barrier();

    test_level_guarantee(last_universe.comm(), params, next_universe, target,
                         target_rate, level);
    create_and_test_next_nbhds(next_universe, exact_universe, params, coeff,
                               target_rate, level + 1, exponent);
    return;
  }

  template <std::size_t RangeSize>
  void check_dist_guarantee(ygm::comm &comm, const auto &params) const {
    using sketch_type        = hll_sketch_type<RangeSize>;
    using transform_type     = typename sketch_type::transform_type;
    using transform_ptr_type = typename sketch_type::transform_ptr_type;
    using make_ptr_type      = MakePtrFunc<transform_type>;

    const int          world_size  = comm.size();
    const int          rank        = comm.rank();
    const double       target_rate = 0.5;
    make_ptr_type      _make_ptr{};
    transform_ptr_type transform_ptr(_make_ptr(params.seed()));
    ygm::container::map<const int, sketch_type> universe(comm);

    const unsigned int coeff          = 51;
    const unsigned int exponent       = 3;
    Universe           exact_universe = build_universe(coeff, exponent);

    for (const auto &[idx, nbhd] : exact_universe) {
      if (idx % world_size == rank) {
        sketch_type sketch(transform_ptr);
        for (const unsigned int &j : nbhd) {
          sketch.insert(j);
        }
        universe.async_insert(idx, sketch);
      }
    }
    comm.barrier();  // universe is now seeded

    test_level_guarantee(comm, params, universe, coeff, target_rate, 0);
    create_and_test_next_nbhds(universe, exact_universe, params, coeff,
                               target_rate, 1, exponent);
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
    parameters params =
        krowkee::test::chirp_parameters<parameters>(world, argc, argv);

    perform_tests(world, params);
  }
  return 0;
}