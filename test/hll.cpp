#include <krowkee/hash/hash.hpp>
#include <krowkee/transform/hll.hpp>
#include "krowkee/hash/murmur3.h"

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

void do_experiment(std::uint32_t prefix_size, auto params) {
  krowkee::transform::HyperLogLog sketch(prefix_size, params.seed());
  for (std::uint64_t i(0); i < params.count(); ++i) {
    sketch.insert(i);
  }

  double est = sketch.estimate();
  double err = std::abs(params.count() - est) / params.count();
  std::cout << "(tru, est, err): (" << params.count() << ", " << est << ", "
            << err << ")" << std::endl;
  // print_container(sketch.registers(), "sketch");
  // std::cout << "pre-merge" << std::endl;
  // std::cout << reg;
  // reg.merge();
  // std::cout << "post-merge" << std::endl;
  // std::cout << reg;
  return;
}

int main(int argc, char **argv) {
  using parameters = krowkee::test::hash::parameters;

  parameters params = krowkee::test::chirp_parameters<parameters>(argc, argv);

  do_experiment(4, params);
  do_experiment(6, params);
  do_experiment(9, params);

  return 0;
}
