#include <krowkee/hash/hash.hpp>
#include <krowkee/transform/hll.hpp>
#include "krowkee/hash/murmur3.h"

#include <krowkee/util/runtime.hpp>

#if __has_include(<cereal/cereal.hpp>)
#include <cereal/types/base_class.hpp>
#include <check_archive.hpp>
#endif
#include <parameters.hpp>

#include <cstdint>

using krowkee::dispatch_with_sketch_sizes;

template <typename C>
void print_container(const C &vec, const char *name) {
  std::cout << name << " : ";
  std::for_each(std::begin(vec), std::end(vec),
                [](auto p) { std::cout << (int)p << " "; });
  std::cout << std::endl;
}

int do_experiment(int argc, char **argv) {
  std::uint32_t seed = 1013;
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage:  " << argv[0]
              << " <prefix_size> <num_inserts> (<seed>)" << std::endl
              << std::flush;
    exit(-1);
  }
  std::uint32_t prefix_size(std::atoi(argv[1]));
  std::uint64_t num_inserts = std::atoll(argv[2]);
  if (argc == 4) {
    seed = std::atoi(argv[3]);
  }

  krowkee::transform::HyperLogLog sketch(prefix_size, seed);
  for (std::uint64_t i(0); i < num_inserts; ++i) {
    sketch.insert(i);
  }

  double est = sketch.estimate();
  double err = std::abs(num_inserts - est) / num_inserts;
  std::cout << "tru: " << num_inserts << std::endl;
  std::cout << "est: " << est << std::endl;
  std::cout << "err: " << err << std::endl;
  // print_container(sketch.registers(), "sketch");
  // std::cout << "pre-merge" << std::endl;
  // std::cout << reg;
  // reg.merge();
  // std::cout << "post-merge" << std::endl;
  // std::cout << reg;
  return 0;
}

int main(int argc, char **argv) {
  using parameters = krowkee::test::hash::parameters;

  parameters params = krowkee::test::chirp_parameters<parameters>(argc, argv);

  return do_experiment(argc, argv);

  // dispatch_with_sketch_sizes<do_experiment, void>(params.range_size(),
  // params);

  return 0;
}
