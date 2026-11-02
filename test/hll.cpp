#include <krowkee/hash/hll.hpp>
#include <krowkee/transform/hll.hpp>

#if __has_include(<cereal/cereal.hpp>)
#include <cereal/types/base_class.hpp>
#include <check_archive.hpp>
#endif
#include <parameters.hpp>

#include <cstdint>

template <typename C>
void print_container(const C &vec, const char *name) {
  std::cout << name << " : ";
  std::for_each(std::begin(vec), std::end(vec),
                [](auto p) { std::cout << (int)p << " "; });
  std::cout << std::endl;
}

template <typename SketchType>
void do_experiment(auto params) {
  const int iteration_count(10);
  double    mean_err(0.0);
  double    success_rate(0.0);
  double    thresh = (1.04 / std::sqrt(SketchType::hash_type::range_size()));
  for (int i(0); i < iteration_count; ++i) {
    SketchType sketch(params.seed() + i);
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
  std::cout << "(range_size, mean err, threshold, success_rate) ("
            << SketchType::hash_type::range_size() << ", " << mean_err << ", "
            << thresh << ", " << success_rate << ")" << std::endl;
  // print_container(sketch.registers(), "sketch");
  // std::cout << "pre-merge" << std::endl;
  // std::cout << reg;
  // reg.merge();
  // std::cout << "post-merge" << std::endl;
  // std::cout << reg;
  return;
}

template <std::size_t RangeSize>
using sketch_type = krowkee::transform::HyperLogLog<
    krowkee::hash::HyperLogLogHash32<RangeSize>>;

int main(int argc, char **argv) {
  using parameters = krowkee::test::hash::parameters;

  parameters params = krowkee::test::chirp_parameters<parameters>(argc, argv);

  do_experiment<sketch_type<16>>(params);
  do_experiment<sketch_type<64>>(params);
  do_experiment<sketch_type<512>>(params);

  return 0;
}
