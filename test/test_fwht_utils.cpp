// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <krowkee/transform/fwht/utils.hpp>

#include <krowkee/util/runtime.hpp>

#include <getopt.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>

using krowkee::chirp;
using krowkee::do_test;
using krowkee::make_shared_functor;
using krowkee::print_line;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  for (const T& el : vec) {
    os << el << ", ";
  }
  return os;
}

template <typename T>
bool operator==(std::vector<T>& lhs, const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i(0); i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

struct Parameters {
  std::uint64_t count;
  std::uint64_t sketch_size;
  std::uint64_t seed;
  std::uint64_t num_vertices;
  std::uint64_t col_index;
  std::uint64_t row_index;
  std::int32_t  val;
  std::uint64_t numtrials;
  std::uint64_t size_mat;
  bool          verbose;
};

template <typename RegType>
struct run_rademacher_test {
  std::string name() const { return "rademacher tests"; }

  void operator()(const Parameters& params) const {
    std::vector<int32_t> test_vec(params.numtrials);
    for (int i = 0; i < params.numtrials; i++) {
      std::uint64_t new_seed = params.seed + i;
      test_vec[i] = krowkee::transform::fwht::rademacher_flip<int32_t>(
          params.val, params.col_index, new_seed);
    }
    int    sum(std::accumulate(test_vec.begin(), test_vec.end(), 0));
    double mean((double)sum / (std::abs(params.val) * params.numtrials + 0.0));
    if (params.verbose) {
      std::cout << "empirical mean of " << params.numtrials
                << " variables : " << mean << std::endl;
    }
    CHECK_CONDITION(std::abs(mean) < 0.01, "unbiased mean");
  }
};

struct run_uniform_sample_test {
  std::string name() const { return "uniform sampling"; }

  void operator()(const Parameters& params) const {
    std::vector<uint64_t> test_vec_unif(params.sketch_size);
    std::int32_t          sum_indices    = 0;
    int                   input_size_int = (int)params.num_vertices;
    int                   histArray[params.num_vertices] = {0};
    for (int i = 0; i < params.numtrials; i++) {
      std::uint64_t new_seed = params.seed + i;

      test_vec_unif = krowkee::transform::fwht::uniform_sample_vec(
          params.num_vertices, params.sketch_size, params.row_index, new_seed);

      // std::cout << "\t" << test_vec_unif << std::endl;
      for (int j = 0; j < params.sketch_size; j++) {
        histArray[test_vec_unif[j]]++;
      }
    }
    int max_hist_entry =
        *std::max_element(histArray, histArray + input_size_int);
    int min_hist_entry =
        *std::min_element(histArray, histArray + input_size_int);
    int sum_elements(std::accumulate(histArray, histArray + input_size_int, 0));
    double mean_hist_entry = (double)sum_elements / input_size_int;
    int    denom           = params.numtrials;
    double rel_min(
        std::abs((double)(mean_hist_entry - min_hist_entry) / mean_hist_entry));
    double rel_max(
        std::abs((double)(mean_hist_entry - max_hist_entry) / mean_hist_entry));
    if (params.verbose) {
      std::cout << "uniform sample histograph has min and max entries ("
                << min_hist_entry << ", " << max_hist_entry << ") - relative ("
                << rel_min << ", " << rel_max << ")" << std::endl;
    }
    CHECK_CONDITION(rel_min < 0.02 && rel_max < 0.02, "uniform histogram");
  }
};

/**
 * @note[BWP] Only run if verbose
 */
struct count_set_bits_test {
  std::string name() const { return "bit set counting"; }

  void operator()(const Parameters& params) const {
    for (int i(0); i < 500; i++) {
      std::cout << "(" << i << ","
                << krowkee::transform::fwht::count_set_bits(i) << ") ";
    }
    std::cout << std::endl;
  }
};

struct get_parity_test {
  std::string name() const { return "parity"; }

  void operator()(const Parameters& params) const {
    bool parity_success(true);
    for (int i(0); i < params.numtrials; ++i) {
      if (krowkee::transform::fwht::get_parity(i) != i % 2) {
        parity_success = false;
      };
    }
    CHECK_CONDITION(parity_success == true, "parity");
  }
};

template <typename RegType>
struct get_hadamard_element_test {
  std::string name() const { return "Hadamard element lookup"; }

  void operator()(const Parameters& params) const {
    RegType hadamard[params.size_mat][params.size_mat];
    RegType hadamard_truth[params.size_mat][params.size_mat];
    hadamard[0][0]       = RegType(1);
    hadamard_truth[0][0] = RegType(1);
    for (uint64_t x = 1; x < params.size_mat; x += x) {
      for (uint64_t i = 0; i < x; i++) {
        for (uint64_t j = 0; j < x; j++) {
          hadamard_truth[i + x][j]     = hadamard_truth[i][j];
          hadamard_truth[i][j + x]     = hadamard_truth[i][j];
          hadamard_truth[i + x][j + x] = RegType(-1) * hadamard_truth[i][j];
          RegType elem_ij(
              krowkee::transform::fwht::get_hadamard_element<RegType>(i, j));
          hadamard[i + x][j]     = elem_ij;
          hadamard[i][j + x]     = elem_ij;
          hadamard[i + x][j + x] = RegType(-1) * elem_ij;
        }
      }
    }
    bool success(true);
    for (uint64_t i = 0; i < params.size_mat; i++) {
      for (uint64_t j = 0; j < params.size_mat; j++) {
        if (hadamard[i][j] != hadamard_truth[i][j]) {
          success = false;
        }
      }
    }
    CHECK_CONDITION(success, "Hadamard element generation");
  }
};

template <typename RegType>
struct get_sketch_vector_test {
  std::string name() const { return "update vector"; }

  void operator()(const Parameters& params) const {
    std::vector<int32_t> test_sketch_vec =
        krowkee::transform::fwht::get_sketch_vector<int32_t>(
            params.val, params.row_index, params.col_index, params.num_vertices,
            params.sketch_size, params.seed);
    {
      bool size_success = test_sketch_vec.size() == params.sketch_size;
      if (!size_success) {
        std::cout << "got " << test_sketch_vec.size() << " while expecting "
                  << params.sketch_size << std::endl;
      }
      CHECK_CONDITION(size_success, "size");
    }
    {
      int    sum(std::accumulate(std::begin(test_sketch_vec),
                                 std::end(test_sketch_vec), 0.0));
      double rel_mag((double)sum / (params.sketch_size * params.val));

      if (params.verbose) {
        std::cout << "Update sum, should be close to zero: "
                  << std::accumulate(std::begin(test_sketch_vec),
                                     std::end(test_sketch_vec), 0.0) /
                         (params.sketch_size * params.val)
                  << std::endl;
        std::cout << "The sketch: " << test_sketch_vec << std::endl;
      }
      CHECK_CONDITION(rel_mag < 0.1,
                      "register sum relative magnitude near zero");
    }
    {
      std::vector<int32_t> second_sketch_vec =
          krowkee::transform::fwht::get_sketch_vector<int32_t>(
              params.val, params.row_index, params.col_index,
              params.num_vertices, params.sketch_size, params.seed);
      bool agree_success = test_sketch_vec == second_sketch_vec;
      CHECK_CONDITION(agree_success, "agreement");
    }
  }
};

struct orthonormality_test {
  std::string name() const { return "orthonormality"; }

  void operator()(const Parameters& params) const {
    const std::uint64_t col_index(2);
    const std::uint64_t row_index(10);

    std::int32_t        sum_row     = 0;
    std::int32_t        sum_col     = 0;
    std::uint64_t       max_power   = 128;
    const std::uint64_t row_index_2 = 100;
    const std::uint64_t col_index_2 = row_index_2;
    for (int i = 0; i < max_power; i++) {
      sum_row += krowkee::transform::fwht::get_hadamard_element<std::int32_t>(
                     row_index, i) *
                 krowkee::transform::fwht::get_hadamard_element<std::int32_t>(
                     row_index_2, i);
      sum_col += krowkee::transform::fwht::get_hadamard_element<std::int32_t>(
                     i, col_index) *
                 krowkee::transform::fwht::get_hadamard_element<std::int32_t>(
                     i, col_index_2);
    }
    bool row_success = sum_row == 0;
    if (!row_success) {
      std::cout << "row sum: " << sum_row;
    }
    CHECK_CONDITION(row_success, "row orthonormality");
    bool col_success = sum_col == 0;
    if (!col_success) {
      std::cout << "col sum: " << sum_col;
    }
    CHECK_CONDITION(col_success, "column orthonormality");
  }
};

void print_help(char* exe_name) {
  std::cout << "\nusage:  " << exe_name << "\n"
            << "-\tv, --verbose  - print additional debug information.\n"
            << "-\th, --help  - print this line and exit\n"
            << std::endl;
}

// note{bwp} Should make this instrumentable at some point.
void parse_args(int argc, char** argv, Parameters& params) {
  int c;

  while (1) {
    int                  option_index(0);
    static struct option long_options[] = {{"verbose", no_argument, NULL, 'v'},
                                           {NULL, 0, NULL, 0}};

    int curind = optind;
    c          = getopt_long(argc, argv, "-:vh", long_options, &option_index);
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

int main(int argc, char** argv) {
  std::uint64_t count(10000);
  std::uint64_t sketch_size(10);
  std::uint64_t seed(3);
  std::uint64_t num_vertices(100);
  std::uint64_t col_index(2);
  std::uint64_t row_index(10);
  std::int32_t  val(15);
  // Run all of my test scripts
  std::uint64_t numtrials(1000000);
  std::uint64_t size_mat(512);
  bool          verbose(false);

  Parameters params{count,     sketch_size, seed,      num_vertices, col_index,
                    row_index, val,         numtrials, size_mat,     verbose};

  parse_args(argc, argv, params);

  if (params.verbose) {
    do_test<count_set_bits_test>(params);
  }
  do_test<get_parity_test>(params);
  do_test<run_rademacher_test<std::int32_t>>(params);
  do_test<run_uniform_sample_test>(params);
  do_test<get_hadamard_element_test<std::int32_t>>(params);
  do_test<get_sketch_vector_test<std::int32_t>>(params);
  do_test<orthonormality_test>(params);

  return (0);
}
