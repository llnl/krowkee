// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

// Klugy, but includes need to be in this order.

#include <check_archive.hpp>
#include <sketch_types.hpp>

#include <krowkee/hash/hash.hpp>
#include <krowkee/util/cmap_types.hpp>
#include <krowkee/util/runtime.hpp>
#include <krowkee/util/sketch_types.hpp>

#include <getopt.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

using krowkee::chirp;
using krowkee::do_test;
using krowkee::make_shared_functor;
using krowkee::print_line;

using sketch_impl_type = krowkee::util::sketch_impl_type;
using cmap_impl_type   = krowkee::util::cmap_impl_type;

/**
 * Struct bundling the experiment parameters.
 */
struct Parameters {
  std::uint64_t    count;
  std::uint64_t    range_size;
  std::uint64_t    replication_count;
  std::uint64_t    domain_size;
  std::uint64_t    observation_count;
  std::uint64_t    seed;
  sketch_impl_type sketch_impl;
  cmap_impl_type   cmap_impl;
  bool             verbose;
};

/**
 * Verify that initialization and assignment (=) operators work as expected.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct init_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " constructors";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type _make_ptr = make_ptr_type();
    {
      transform_ptr_type transform_ptr_1(_make_ptr(0));
      transform_ptr_type transform_ptr_2(_make_ptr(0));
      sketch_type        sketch_1(transform_ptr_1);
      sketch_type        sketch_2(transform_ptr_2);
      for (int i(0); i < 1000; i++) {
        sketch_1.insert(i);
        sketch_2.insert(i);
      }
      bool constructors_match = sketch_1 == sketch_2;
      if (constructors_match == false) {
        std::cout << "sketch_1 : " << sketch_1 << std::endl;
        std::cout << "sketch_2 : " << sketch_2 << std::endl;
      }
      CHECK_CONDITION(constructors_match == true,
                      "constructor/insert consistency");
    }
    {
      transform_ptr_type transform_ptr(_make_ptr(0));
      sketch_type        sketch(transform_ptr);
      for (int i(0); i < 1000; sketch.insert(i++)) {
      }
      sketch_type sketch2(sketch);
      bool        copy_matches = sketch == sketch2;
      if (copy_matches == false) {
        std::cout << "sketch : " << sketch << std::endl;
        std::cout << "sketch2 : " << sketch2 << std::endl;
      }
      CHECK_CONDITION(copy_matches == true, "copy constructor");
    }
    {
      transform_ptr_type transform_ptr(_make_ptr(0));
      sketch_type        sketch(transform_ptr);
      for (int i(0); i < 1000; sketch.insert(i++)) {
      }
      sketch_type sketch2      = sketch;
      bool        swap_matches = sketch == sketch2;
      if (swap_matches == false) {
        std::cout << "sketch : " << sketch << std::endl;
        std::cout << "sketch2 : " << sketch2 << std::endl;
      }
      CHECK_CONDITION(swap_matches, "copy-and-swap assignment");
    }
    {
      transform_ptr_type transform_ptr(_make_ptr(0));
      sketch_type        sketch(transform_ptr);

      bool init_empty = sketch.empty();

      CHECK_CONDITION(init_empty == true, "initial empty");

      sketch.insert(1);
      bool not_empty = sketch.empty();
      CHECK_CONDITION(not_empty == false, "post-insert not empty");

      sketch.clear();
      bool clear_empty = sketch.empty();
      CHECK_CONDITION(clear_empty == true, "post-clear empty");
    }
  }
};

/**
 * Verify that the given sketch functor produces a reasonable embedding.
 *
 * @note[BWP] TODO: implement some sort of statistical testing to verify that
 *     we are getting approximately correct shape preservation in embedded
 *     space.
 *
 * @note[BWP] `krowkee::SparseJLTFunctor` templated with `krowkee::WangHash`
 *     produces weird and bad results here. We are unlikely to ever use
 *     `krowkee::WangHash`, but it might be worth figuring out what is wrong.
 *     It probably has something to do with the polarity hash, as it looks like
 *     the entries to the first 1/2 of registers are all -1, while the entries
 *     to the second 1/2 of registers are all +1.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct ingest_check {
  using sketch_type        = SketchType;
  using registers_type     = typename sketch_type::registers_type;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " ingest";
    return ss.str();
  }

  std::vector<std::vector<std::uint64_t>> get_uniform_inserts(
      const Parameters &params) const {
    std::mt19937                                 gen(params.seed);
    std::uniform_int_distribution<std::uint64_t> dist(0,
                                                      params.domain_size - 1);

    std::vector<std::vector<std::uint64_t>> inserts(
        params.observation_count, std::vector<std::uint64_t>(params.count));

    for (std::int64_t i(0); i < params.observation_count; ++i) {
      for (std::int64_t j(0); j < params.count; ++j) {
        inserts[i][j] = dist(gen);
      }
    }
    return inserts;
  }

  template <typename T>
  double _l2_distance_sq(const std::vector<T> &lhs,
                         const std::vector<T> &rhs) const {
    assert(lhs.size() == rhs.size());
    double dist_sq(0);
    for (int i(0); i < lhs.size(); ++i) {
      dist_sq += std::pow(lhs[i] - rhs[i], 2);
    }
    return dist_sq;
  }

  double _l2_distance_sq(const registers_type &lhs,
                         const registers_type &rhs) const {
    assert(lhs.size() == rhs.size());
    return (lhs - rhs).squaredNorm();
  }

  void rel_mag_test(const transform_ptr_type &transform_ptr,
                    const Parameters         &params) const {
    sketch_type sketch(transform_ptr);
    for (std::uint64_t i(0); i < params.count; sketch.insert(i++)) {
    }
    sketch.compactify();
    int    sum(accumulate(sketch, 0.0));
    double rel_mag((double)sum / (params.count * params.range_size *
                                  params.replication_count));
    if (params.verbose == true) {
      std::cout << "\t" << sketch << std::endl;
      std::cout << "\tregister sum (should be near zero): " << sum
                << ", relative magnitude: " << rel_mag << std::endl;
    }
    CHECK_CONDITION(rel_mag < 0.1, "register sum relative magnitude near zero");
  }

  template <typename T>
  void print_mat(const char *name, std::vector<std::vector<T>> &inserts,
                 const int nrows, const int ncosketch) const {
    std::cout << std::endl;
    std::cout << name << ":" << std::endl;
    for (int i(0); i < nrows; ++i) {
      std::cout << "(" << i << ")\t";
      for (int j(0); j < ncosketch; ++j) {
        std::cout << " " << inserts[i][j];
      }
      std::cout << std::endl;
    }
  }

  void print_mat(std::vector<sketch_type> &sketches, const int nrows) const {
    std::cout << std::endl;
    std::cout << "sketches:" << std::endl;
    for (int i(0); i < nrows; ++i) {
      std::cout << "(" << i << ")\t" << sketches[i] << std::endl;
    }
  }

  void print_mat(std::vector<registers_type> &projections,
                 const int                    nrows) const {
    std::cout << std::endl;
    std::cout << "projections:" << std::endl;
    for (int i(0); i < nrows; ++i) {
      std::cout << "(" << i << ")\t" << projections[i] << std::endl;
    }
  }

  std::vector<std::vector<register_type>> fill_observation_vector(
      const std::vector<std::vector<std::uint64_t>> &inserts,
      const Parameters                              &params) const {
    std::vector<std::vector<register_type>> observations(
        params.observation_count,
        std::vector<register_type>(params.domain_size));
    for (int i(0); i < params.observation_count; ++i) {
      for (int j(0); j < params.count; ++j) {
        observations[i][inserts[i][j]]++;
      }
    }
    return observations;
  }

  std::vector<sketch_type> fill_sketch_vector(
      const transform_ptr_type                      &transform_ptr,
      const std::vector<std::vector<std::uint64_t>> &inserts,
      const Parameters                              &params) const {
    std::vector<sketch_type> sketches(params.observation_count,
                                      sketch_type(transform_ptr));
    for (int i(0); i < params.observation_count; ++i) {
      for (int j(0); j < params.count; ++j) {
        sketches[i].insert(inserts[i][j]);
      }
      sketches[i].compactify();
    }
    return sketches;
  }

  std::vector<registers_type> fill_projection_vector(
      const std::vector<sketch_type> &sketches,
      const Parameters               &params) const {
    std::vector<registers_type> projections;
    for (int i(0); i < params.observation_count; ++i) {
      projections.push_back(sketches[i].scaled_registers());
    }
    return projections;
  }

  void lemma_check(const transform_ptr_type &transform_ptr,
                   const Parameters         &params) const {
    std::vector<std::vector<std::uint64_t>> inserts =
        get_uniform_inserts(params);

    std::vector<std::vector<register_type>> observations =
        fill_observation_vector(inserts, params);

    std::vector<sketch_type> sketches =
        fill_sketch_vector(transform_ptr, inserts, params);

    std::vector<registers_type> projections =
        fill_projection_vector(sketches, params);

    double expected_epsilon =
        std::sqrt(16 * std::log(params.observation_count) /
                  (params.range_size * params.range_size));
    // compute distances
    if (params.verbose) {
      print_mat("inserts", inserts, params.observation_count, params.count);
      print_mat("observations", observations, params.observation_count,
                params.domain_size);
      print_mat(sketches, params.observation_count);
      print_mat(projections, params.observation_count);
      std::cout << std::endl;
      std::cout << "projected vectors:" << std::endl;
    }
    double success_rate(0.0);
    double empirical_epsilon;
    int    trials(0);
    for (int i(0); i < params.observation_count; ++i) {
      for (int j(0); j < params.observation_count; ++j) {
        if (i == j) {
          break;
        }
        ++trials;
        double ob_dist    = _l2_distance_sq(observations[i], observations[j]);
        double sk_dist    = _l2_distance_sq(projections[i], projections[j]);
        double this_error = mul_error(ob_dist, sk_dist);
        empirical_epsilon += this_error;
        if (in_bounds(ob_dist, sk_dist, expected_epsilon)) {
          success_rate += 1.0;
        }
        if (params.verbose) {
          std::cout << "\t(" << i << "," << j << ") ob " << ob_dist << ", sk "
                    << sk_dist << " (multiplicative error: 1 +/- " << this_error
                    << ") (in bounds: "
                    << in_bounds(ob_dist, sk_dist, expected_epsilon) << ")"
                    << std::endl;
        }
      }
    }
    success_rate /= trials;
    empirical_epsilon /= trials;
    bool lemma_guarantee_success = success_rate > 0.5;
    CHECK_CONDITION(lemma_guarantee_success == true, "lemma guarantee (",
                    trials, " trials, ", success_rate,
                    " success rate, expected epsilon=", expected_epsilon,
                    ", mean empirical epsilon=", empirical_epsilon, ")");
  }

  void operator()(const Parameters &params) const {
    make_ptr_type      _make_ptr{};
    transform_ptr_type transform_ptr(_make_ptr(params.seed));
    rel_mag_test(transform_ptr, params);
    lemma_check(transform_ptr, params);
  }

  bool in_bounds(const double ob_dist, const double sk_dist,
                 const double epsilon) const {
    return (sk_dist < (1 + epsilon) * ob_dist) &&
           (sk_dist > (1 - epsilon) * ob_dist);
  }

  double mul_error(const double ob_dist, const double sk_dist) const {
    return std::abs(1 - sk_dist / ob_dist);
  }
};

template <typename SketchType>
void check_throws_bad_plus_equals(SketchType &lhs, const SketchType &rhs) {
  lhs += rhs;
}

template <typename SketchType>
void check_throws_bad_plus(SketchType &lhs, const SketchType &rhs) {
  SketchType sketch = lhs += rhs;
}

/**
 * Verify that merge (+/+=) operators catch bad merges.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct bad_merge_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " bad merges";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type      _make_ptr = make_ptr_type();
    transform_ptr_type transform_ptr_1(_make_ptr(32));
    transform_ptr_type transform_ptr_2(_make_ptr(22));
    sketch_type        sketch_1(transform_ptr_1);
    sketch_type        sketch_2(transform_ptr_2);
    CHECK_THROWS<std::invalid_argument>(
        check_throws_bad_plus_equals<SketchType>,
        "bad merge (+=) with different functor seeds", sketch_1, sketch_2);
    CHECK_THROWS<std::invalid_argument>(
        check_throws_bad_plus<SketchType>,
        "bad merge (+) with different functor seeds", sketch_1, sketch_2);
  }
};

/**
 * Verify that merge (+/+=) operators work as expected.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct good_merge_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " good merges";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type      _make_ptr = make_ptr_type();
    transform_ptr_type transform_ptr(_make_ptr(8));
    sketch_type        first(transform_ptr);
    sketch_type        middle(transform_ptr);
    sketch_type        last(transform_ptr);
    sketch_type        both(transform_ptr);
    sketch_type        all(transform_ptr);
    for (std::uint64_t i(0); i < 1000; ++i) {
      first.insert(i);
      both.insert(i);
      all.insert(i);
    }
    for (std::uint64_t i(1000); i < 2000; ++i) {
      middle.insert(i);
      both.insert(i);
      all.insert(i);
    }
    for (std::uint64_t i(1000); i < 2000; ++i) {
      last.insert(i);
      all.insert(i);
    }
    first.compactify();
    middle.compactify();
    last.compactify();
    both.compactify();
    all.compactify();
    sketch_type bb = first + middle;
    bb.compactify();
    {
      bool merge_success = both == bb;
      CHECK_CONDITION(merge_success == true, "merge (+)");
    }
    {
      sketch_type aa = first + middle + last;
      aa.compactify();
      bool multimerge_success = all == aa;
      CHECK_CONDITION(multimerge_success == true, "multi-merge (+, +)");
    }
    {
      first += middle;
      bool inplace_merge_success = both == first;
      CHECK_CONDITION(inplace_merge_success == true, "merge (+=)");
    }
  }
};

#if __has_include(<cereal/cereal.hpp>)
template <typename SketchType, template <typename> class MakePtrFunc>
struct serialize_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " serialize";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type      _make_ptr{};
    transform_ptr_type transform_ptr(_make_ptr(params.seed));

    CHECK_ALL_ARCHIVES(*transform_ptr, "sketch functor");

    sketch_type sketch(transform_ptr);
    for (std::uint64_t i(0); i < params.count; sketch.insert(i++)) {
    }
    sketch.compactify();

    CHECK_ALL_ARCHIVES(sketch.container(), "sketch container");
    CHECK_ALL_ARCHIVES(sketch, "whole sketch object");
  }
};
#endif

/**
 * Verify that promotion works as expected. Can only be compiled if
 * sketch_type::container_type is krowkee::sketch::Promotable.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
struct promotion_check {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;
  using make_ptr_type      = MakePtrFunc<transform_type>;

  constexpr std::string name() const {
    std::stringstream ss;
    ss << transform_type::name() << " promotion check";
    return ss.str();
  }

  void operator()(const Parameters &params) const {
    make_ptr_type      _make_ptr = make_ptr_type();
    transform_ptr_type transform_ptr(_make_ptr(params.seed));
    sketch_type        s1(transform_ptr);
    sketch_type        s2(transform_ptr);
    sketch_type        d1(transform_ptr);
    sketch_type        d2(transform_ptr);
    sketch_type        d12(transform_ptr);
    sketch_type        dall(transform_ptr);
    int                pt(sketch_type::container_type::promotion_threshold() /
                          sketch_type::transform_type::replication_count());
    for (std::uint64_t i(0); i < pt - 1; ++i) {
      s1.insert(i);
      d12.insert(i);
      d1.insert(i);
      d1.insert(i);
      d2.insert(i);
      dall.insert(i);
    }
    for (std::uint64_t i(pt); i < 2 * pt - 1; ++i) {
      s2.insert(i);
      d12.insert(i);
      d1.insert(i);
      d2.insert(i);
      d2.insert(i);
      dall.insert(i);
    }
    for (std::uint64_t i(2 * pt); i < params.count; ++i) {
      d1.insert(i);
      d2.insert(i);
      dall.insert(i);
    }
    s1.compactify();
    s2.compactify();
    d1.compactify();
    d2.compactify();
    d12.compactify();
    dall.compactify();
    {
      sketch_type d1_ = s1 + dall;
      d1_.compactify();
      bool sp_merge_success = d1_ == d1;
      CHECK_CONDITION(sp_merge_success == true, "merge (+) sparse/dense");
    }
    {
      sketch_type d1_ = dall + s1;
      d1_.compactify();
      bool sp_merge_success = d1_ == d1;
      CHECK_CONDITION(sp_merge_success == true, "merge (+) dense/sparse");
    }
    {
      sketch_type dall12 = dall + s1 + s2;
      sketch_type d12_   = dall + d12;
      dall12.compactify();
      d12_.compactify();
      bool sp_multimerge_success = dall12 == d12_;
      CHECK_CONDITION(sp_multimerge_success == true,
                      "multi-merge (+) dense/sparse/sparse");
    }
    {
      sketch_type s11 = s1 + s1;
      s11.compactify();
      bool sss_merge_success =
          accumulate(s11, 0.0) == 2 * accumulate(s1, 0.0) &&
          s11.is_sparse() == true;
      CHECK_CONDITION(sss_merge_success == true,
                      "merge (+) sparse/sparse (sparse)");
    }
    {
      sketch_type s12 = s1 + s2;
      s12.compactify();
      bool ssd_merge_success = s12 == d12;
      CHECK_CONDITION(ssd_merge_success == true,
                      "merge (+) sparse/sparse (dense)");
    }
    {
      s1 += dall;
      s1.compactify();
      bool ssd_inplace_merge_success = s1 == d1;
      CHECK_CONDITION(ssd_inplace_merge_success == true,
                      "merge (+=) sparse/dense (dense)");
    }
    {
      dall += s2;
      dall.compactify();
      bool dsd_inplace_merge_success = dall == d2;
      CHECK_CONDITION(dsd_inplace_merge_success == true,
                      "merge (+=) dense/sparse (dense)");
    }
  }
};

/**
 * Execute the batter of tests for the given sketch functor.
 */
template <typename SketchType, template <typename> class MakePtrFunc>
void perform_tests(const Parameters &params) {
  using sketch_type    = SketchType;
  using transform_type = typename sketch_type::transform_type;

  MakePtrFunc<std::int32_t> mpf;

  print_line();
  print_line();
  std::cout << "Testing " << sketch_type::full_name() << std::endl;
  std::cout << "\tUsing " << mpf.name() << std::endl;
  print_line();
  print_line();

  std::cout << std::endl << std::endl;

  do_test<init_check<sketch_type, MakePtrFunc>>(params);
  do_test<ingest_check<sketch_type, MakePtrFunc>>(params);
  do_test<bad_merge_check<sketch_type, MakePtrFunc>>(params);
  do_test<good_merge_check<sketch_type, MakePtrFunc>>(params);
#if __has_include(<cereal/cereal.hpp>)
  do_test<serialize_check<sketch_type, MakePtrFunc>>(params);
#endif
  // This is a really complex compile-time check indicating whether the sketch
  // being investigated is promotable.
#if __has_include(<boost/container/flat_map.hpp>)
  if constexpr (std::is_same<
                    typename sketch_type::container_type,
                    typename promotable::map::SparseJLT<
                        sketch_type::transform_type::range_size(),
                        sketch_type::transform_type::replication_count()>::
                        container_type>::value ||
                std::is_same<
                    typename sketch_type::container_type,
                    typename promotable::flatmap::SparseJLT<
                        sketch_type::transform_type::range_size(),
                        sketch_type::transform_type::replication_count()>::
                        container_type>::value) {
    do_test<promotion_check<sketch_type, MakePtrFunc>>(params);
  }
#endif
}

void print_help(char *exe_name) {
  std::cout << "\nusage:  " << exe_name << "\n"
            << "\t-c, --count <int>              - number of insertions\n"
            << "\t-r, --range <int>              - range of sketch transform\n"
            << "\t-R, --replication <int>        - number of tiled sketch "
               "transforms\n"
            << "\t-d, --domain <int>             - domain of sketch transform\n"
            << "\t-b, --observation_count <int>  - number of sketches to test\n"
            << "\t-t, --sketch-type <str>        - sketch type "
               "(cst, sparse_cst, promotable_cst, fwht)\n"
            << "\t-m, --map-type <str>           - map type "
#if __has_include(<boost/container/flat_map.hpp>)
               "(std, boost)\n"
#else
               "(std)\n"
#endif
            << "\t-s, --seed <int>               - random seed\n"
            << "\t-v, --verbose                  - print additional debug "
               "information.\n"
            << "\t-h, --help                     - print this line and exit\n"
            << std::endl;
}

void parse_args(int argc, char **argv, Parameters &params) {
  int c;

  while (1) {
    int                  option_index(0);
    static struct option long_options[] = {
        {"count", required_argument, NULL, 'c'},
        {"range", required_argument, NULL, 'r'},
        {"replication", required_argument, NULL, 'R'},
        {"domain", required_argument, NULL, 'd'},
        {"observation-count", required_argument, NULL, 'b'},
        {"sketch-type", required_argument, NULL, 't'},
        {"map-type", required_argument, NULL, 'm'},
        {"seed", required_argument, NULL, 's'},
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}};

    int curind = optind;
    c          = getopt_long(argc, argv, "-:c:r:R:d:b:t:m:s:vh", long_options,
                             &option_index);
    if (c == -1) {
      break;
    }

    switch (c) {
      case 'h':
        print_help(argv[0]);
        exit(-1);
        break;
      case 0:
        printf("long option %s", long_options[option_index].name);
        if (optarg) {
          printf(" with arg %s", optarg);
        }
        printf("\n");
        break;
      case 1:
        printf("unused regular argument ignored %s\n", optarg);
        break;
      case 'c':
        params.count = std::atol(optarg);
        break;
      case 'r':
        params.range_size = std::atoll(optarg);
        break;
      case 'R':
        params.replication_count = std::atoll(optarg);
        break;
      case 'd':
        params.domain_size = std::atoll(optarg);
        break;
      case 'b':
        params.observation_count = std::atoll(optarg);
        break;
      case 't':
        params.sketch_impl = krowkee::util::get_sketch_impl_type(optarg);
        break;
      case 'm':
        params.cmap_impl = krowkee::util::get_cmap_impl_type(optarg);
        break;
      case 's':
        params.seed = std::atol(optarg);
        break;
      case 'v':
        params.verbose = true;
        break;
      case '?':
        if (optopt == 0) {
          printf("Unknown long option \"%s\",", argv[curind]);
        } else {
          printf("Unknown option %c,", optopt);
        }
        printf(" consult %s --help\n", argv[0]);
        break;
      case ':':
        printf("Missing argument for option -%c/--%s\n", optopt,
               long_options[option_index].name);
        break;
      default:
        printf("?? getopt returned character code 0%o ??\n", c);
        break;
    }
  }
}

template <std::size_t RangeSize, std::size_t ReplicationCount>
struct choose_tests {
  void operator()(const Parameters &params) {
    if (params.sketch_impl == sketch_impl_type::cst) {
      perform_tests<dense::SparseJLT<RangeSize, ReplicationCount>,
                    make_ptr_functor>(params);
    } else if (params.sketch_impl == sketch_impl_type::sparse_cst) {
      if (params.cmap_impl == cmap_impl_type::std) {
        perform_tests<sparse::map::SparseJLT<RangeSize, ReplicationCount>,
                      make_ptr_functor>(params);
#if __has_include(<boost/container/flat_map.hpp>)
      } else if (params.cmap_impl == cmap_impl_type::boost) {
        perform_tests<sparse::flatmap::SparseJLT<RangeSize, ReplicationCount>,
                      make_ptr_functor>(params);
#endif
      }
    } else if (params.sketch_impl == sketch_impl_type::promotable_cst) {
      if (params.cmap_impl == cmap_impl_type::std) {
        perform_tests<promotable::map::SparseJLT<RangeSize, ReplicationCount>,
                      make_ptr_functor>(params);
#if __has_include(<boost/container/flat_map.hpp>)
      } else if (params.cmap_impl == cmap_impl_type::boost) {
        perform_tests<
            promotable::flatmap::SparseJLT<RangeSize, ReplicationCount>,
            make_ptr_functor>(params);
#endif
      }
    } else if (params.sketch_impl == sketch_impl_type::fwht) {
      perform_tests<dense::FWHT<RangeSize, ReplicationCount>, make_ptr_functor>(
          params);
    }
  }
};

template <std::size_t RangeSize, std::size_t ReplicationCount>
struct do_all_tests {
  void operator()(const Parameters &params) {
    perform_tests<dense::SparseJLT<RangeSize, ReplicationCount>,
                  make_ptr_functor>(params);
    perform_tests<sparse::map::SparseJLT<RangeSize, ReplicationCount>,
                  make_ptr_functor>(params);
    perform_tests<promotable::map::SparseJLT<RangeSize, ReplicationCount>,
                  make_ptr_functor>(params);
#if __has_include(<boost/container/flat_map.hpp>)
    perform_tests<sparse::flatmap::SparseJLT<RangeSize, ReplicationCount>,
                  make_ptr_functor>(params);
    perform_tests<promotable::flatmap::SparseJLT<RangeSize, ReplicationCount>,
                  make_ptr_functor>(params);
#endif
    perform_tests<dense::FWHT<RangeSize, ReplicationCount>, make_ptr_functor>(
        params);
  }
};

int main(int argc, char **argv) {
  uint64_t         count(10000);
  std::uint64_t    range_size(32);
  std::uint64_t    replication_count(4);
  std::uint64_t    domain_size(4096);
  std::uint64_t    observation_count(16);
  std::uint64_t    seed(krowkee::hash::default_seed);
  sketch_impl_type sketch_impl(sketch_impl_type::cst);
  cmap_impl_type   cmap_impl(cmap_impl_type::std);
  bool             verbose(false);
  bool             do_all(argc == 1);

  Parameters params{count,       range_size,        replication_count,
                    domain_size, observation_count, seed,
                    sketch_impl, cmap_impl,         verbose};

  parse_args(argc, argv, params);

  // do_all_tests<32, 4>{}(params);
  if (do_all == true) {
    krowkee::dispatch<do_all_tests, void>{params.range_size,
                                          params.replication_count}(params);
  } else {
    krowkee::dispatch<choose_tests, void>{params.range_size,
                                          params.replication_count}(params);
  }
  return 0;
}