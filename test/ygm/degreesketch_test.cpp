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
  const char *name() { return "HLL guarantee check"; }

  void operator()(ygm::comm &comm, const auto params) {
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

  template <typename SketchType>
  void test_level_guarantee(ygm::comm &comm, const auto &params,
                            const SketchType &sketch, const double target,
                            const double target_rate, const int level) const {
    const int         world_size = comm.size();
    const std::size_t range_size = SketchType::range_size();
    const std::string full_name  = SketchType::full_name();
    const double      thresh     = 1.5 * (1.04 / std::sqrt(range_size));

    double est = SketchType::transform_type::estimate(sketch.container());
    double err = std::abs(target - est) / target;

    bool guarantee_success = (err < thresh);

    chirp(comm, params, "\tLevel ", level, " (tru, est, err): (",
          params.count(), ", ", est, ", ", err, ")");
    double mean_err = ygm::sum(err, comm) / world_size;
    double success_rate =
        ygm::sum((double)guarantee_success, comm) / world_size;
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
    sketch_type        sketch(transform_ptr);
    for (std::uint64_t j(0); j < params.count(); ++j) {
      sketch.insert(j);
    }

    test_level_guarantee(comm, params, sketch, params.count(), target_rate, 1);

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