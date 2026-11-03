#include <krowkee/hash/hll.hpp>
#include <krowkee/transform/hll.hpp>

#include <krowkee/util/runtime.hpp>

#if __has_include(<cereal/cereal.hpp>)
#include <cereal/types/base_class.hpp>
#include <check_archive.hpp>
#endif
#include <parameters.hpp>

#include <cstdint>

using krowkee::print_line;

template <std::size_t RangeSize>
using hll_hash32_type = krowkee::hash::HyperLogLogHash32<RangeSize>;

template <std::size_t RangeSize>
using hll_sketch_type =
    krowkee::transform::HyperLogLog<hll_hash32_type<RangeSize>>;

template <typename HashType>
void hll_init(std::uint32_t i) {
  HashType hash{i};
}

template <std::size_t RangeSize>
struct init_check {
  const char *name() const { return "hash initialization check"; }

  void operator()() const {
    for (std::uint64_t i(0); i < 63; ++i) {
      CHECK_DOES_NOT_THROW<std::exception>(hll_init<hll_hash32_type<RangeSize>>,
                                           "good value initialization", i);
    }
    CHECK_CONDITION(true, "good value initialization");
  }
};

#if __has_include(<cereal/cereal.hpp>)
template <std::size_t RangeSize>
struct serialize_check {
  const char *name() { return "serialize check"; }

  template <typename HashType, typename... ARGS>
  void serialize(const auto &params, ARGS &&...args) const {
    HashType hash{(std::uint32_t)params.seed(), args...};
    CHECK_ALL_ARCHIVES(hash, HashType::name());
  }

  void operator()(const auto params) const {
    serialize<hll_hash32_type<RangeSize>>(params);
  }
};
#endif

struct guarantee_check {
  const char *name() { return "HLL guarantee check"; }

  void operator()(const auto params) {
    check_guarantee<16>(params);
    check_guarantee<64>(params);
    check_guarantee<512>(params);
  }

  template <std::size_t RangeSize>
  void check_guarantee(const auto params) {
    using sketch_type = hll_sketch_type<RangeSize>;
    const int    iteration_count(10);
    const double target_rate(0.5);
    double       mean_err(0.0);
    double       success_rate(0.0);
    double       thresh =
        1.5 * (1.04 / std::sqrt(sketch_type::hash_type::range_size()));
    for (int i(0); i < iteration_count; ++i) {
      sketch_type sketch(params.seed() + i);
      for (std::uint64_t j(0); j < params.count(); ++j) {
        sketch.insert(j);
      }

      double est     = sketch.estimate();
      double err     = std::abs(params.count() - est) / params.count();
      bool   success = err < thresh;
      mean_err += err;
      success_rate += success;
      if (i == 0 && params.verbose()) {
        std::cout << "\t(tru, est, err): (" << params.count() << ", " << est
                  << ", " << err << ")" << std::endl;
      }
    }
    mean_err /= iteration_count;
    success_rate /= iteration_count;
    std::cout << "\t(range_size, mean err, threshold, success_rate) ("
              << sketch_type::hash_type::range_size() << ", " << mean_err
              << ", " << thresh << ", " << success_rate << ")" << std::endl;
    {
      bool success = (mean_err < thresh);
      if (success == false) {
        std::cout << "Mean estimated error is " << mean_err
                  << ", which exceeds expected threshold of " << thresh
                  << std::endl;
      }
      std::stringstream ss;
      ss << sketch_type::full_name() << " mean estimated error threshold";
      CHECK_CONDITION(success == true, ss.str());
    }
    {
      bool success = (success_rate >= target_rate);
      if (success == false) {
        std::cout << "Estimates achieve guarantee with rate " << success_rate
                  << ", which is below expected threshold of " << target_rate
                  << std::endl;
      }
      std::stringstream ss;
      ss << sketch_type::full_name() << " guarantee success rate";
      CHECK_CONDITION(success == true, ss.str());
    }

    return;
  }
};

template <std::size_t RangeSize>
struct do_experiment {
  void operator()(const auto &params) {
    print_line();
    print_line();
    std::cout << " Experimenting with " << params.count()
              << " insertions into a range of " << params.range_size()
              << " using random seed " << params.seed() << std::endl;
    print_line();
    print_line();
    std::cout << std::endl;
    krowkee::do_test<init_check<RangeSize>>();
//     do_test<empirical_histograms<RangeSize>>(params);
#if __has_include(<cereal/cereal.hpp>)
    do_test<serialize_check<RangeSize>>(params);
#endif
    krowkee::do_test<guarantee_check>(params);
    std::cout << std::endl;
  }
};

int main(int argc, char **argv) {
  using parameters = krowkee::test::hash::parameters;

  parameters params = krowkee::test::chirp_parameters<parameters>(argc, argv);

  do_experiment<16>()(params);

  return 0;
}
