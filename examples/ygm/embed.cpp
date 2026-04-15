// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <krowkee/sketch.hpp>
#include <krowkee/util/runtime.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/map.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

ygm::container::bag<std::string> get_file_paths(
    ygm::comm &comm, const std::filesystem::path &input_path) {
  namespace fs = std::filesystem;

  if (!fs::exists(input_path)) {
    throw std::runtime_error("Path does not exist: " + input_path.string());
  }

  ygm::container::bag<std::string> result(comm);

  if (fs::is_regular_file(input_path)) {
    if (comm.rank0()) {
      result.async_insert(fs::absolute(input_path).string());
    }
  } else if (fs::is_directory(input_path)) {
    if (comm.rank0()) {
      for (const auto &entry : fs::directory_iterator(input_path)) {
        if (fs::is_regular_file(entry.path())) {
          result.async_insert(fs::absolute(entry.path()).string());
        }
      }
    }
  } else {
    throw std::runtime_error(
        "Path is neither a regular file nor a directory: " +
        input_path.string());
  }

  return result;
}

struct parameters {
  std::string   input_path;
  std::string   output_path;
  std::size_t   range_size;
  std::size_t   replication_count;
  std::uint32_t seed;

  parameters() {}
};

void usage(ygm::comm &comm) {
  comm.cerr0()
      << "embed usage:"
      << "\n\t-i <str>\t- Path to input (file or directory of files)"
      << "\n\t-o <str>\t- Path to output (will create directory if not exists)"
      << "\n\t-r <int>\t- Power-of-2 range size"
      << "\n\t-R <int>\t- Power-of-2 replication count"
      << "\n\t-s <int>\t- Random seed"
      << "\n\t-h\t\t- Print help" << std::endl;
}

parameters parse_args(int argc, char **argv, ygm::comm &comm) {
  parameters params{};
  int        c;
  bool       print_help = false;
  bool       satisfied  = false;

  // Suppress error messages from getopt
  extern int opterr;
  opterr = 0;

  while ((c = getopt(argc, argv, "i:o:r:R:s:h")) != -1) {
    switch (c) {
      case 'h':
        print_help = true;
        break;
      case 'i':
        params.input_path = optarg;
        break;
      case 'o':
        params.output_path = optarg;
        break;
      case 'r':
        params.range_size = atoll(optarg);
        break;
      case 'R':
        params.replication_count = atoll(optarg);
        break;
      case 's':
        params.seed = atoi(optarg);
        break;
      default:
        comm.cerr0() << "Unrecognized option: " << char(optopt) << std::endl;
        print_help = true;
        break;
    }
  }

  if (print_help || params.input_path == "" || params.output_path == "" ||
      params.range_size == 0 || params.replication_count == 0) {
    usage(comm);
    exit(-1);
  }

  return params;
}

template <std::size_t RangeSize, std::size_t ReplicationCount>
struct embed {
  constexpr static std::size_t range_size        = RangeSize;
  constexpr static std::size_t replication_count = ReplicationCount;
  using register_type                            = double;
  using sketch_type =
      krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                                 std::shared_ptr>;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  void operator()(ygm::comm &comm, const parameters params) const {
    ygm::container::bag<std::string> path_bag =
        get_file_paths(comm, params.input_path);

    ygm::container::map<std::size_t, std::vector<register_type>> embed_map(
        comm);

    transform_ptr_type transform_ptr(
        std::make_shared<transform_type>(params.seed));

    path_bag.for_all(
        [&comm, &transform_ptr, &embed_map](const std::string &path) {
          std::ifstream ifs(path);
          if (!ifs.good()) {
            std::cerr << "error opening file: " << path << std::endl;
          }

          std::string line;
          while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::size_t        index;
            register_type      feature;
            sketch_type        sketch(transform_ptr);
            int                counter(0);
            iss >> index;  // assume that the first word is the index
            while (iss >> feature) {
              // assume feature vectors are dense. skip zeros otherwise.
              sketch.insert(counter++, feature);
            }
            embed_map.async_insert(index, sketch.scaled_registers());
          }
        });
    comm.barrier();

    // Create the output path if it does not already exist.
    std::error_code ec;
    if (!std::filesystem::exists(params.output_path, ec)) {
      if (!std::filesystem::create_directories(params.output_path, ec) && ec) {
        throw std::runtime_error(
            "Failed to create directory: " + params.output_path +
            ", error: " + ec.message());
      }
    }

    // create the rank-wise output file
    std::ostringstream oss;
    oss << params.output_path << "/" << comm.rank() << ".txt";
    const std::filesystem::path out_path(oss.str());
    std::ofstream               ofs(out_path);
    if (!ofs.is_open()) {
      throw std::runtime_error("Failed to open file: " + out_path.string());
    }

    // dump sketches to file rankwise, line-by-line
    embed_map.for_all([&ofs](const std::size_t                 index,
                             const std::vector<register_type> &embedding) {
      ofs << index;
      for (const auto &feature : embedding) {
        ofs << " " << feature;
      }
      ofs << "\n";
    });
    comm.barrier();

    if (!ofs) {
      throw std::runtime_error("Failed while writing to file: " +
                               out_path.string());
    }
  }
};

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    parameters params = parse_args(argc, argv, world);
    krowkee::dispatch<embed, void>{params.range_size, params.replication_count}(
        world, params);
  }
}