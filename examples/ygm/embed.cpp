// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for detaisketch.
//
// SPDX-License-Identifier: MIT

#include <krowkee/sketch.hpp>
#include <krowkee/util/parameters.hpp>
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

template <std::size_t RangeSize, std::size_t ReplicationCount>
struct embed {
  constexpr static std::size_t range_size        = RangeSize;
  constexpr static std::size_t replication_count = ReplicationCount;
  using register_type                            = double;
  using sketch_type =
      krowkee::sketch::SparseJLT<register_type, range_size, replication_count,
                                 std::shared_ptr>;
  using registers_type     = sketch_type::registers_type;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = typename sketch_type::transform_ptr_type;

  void operator()(ygm::comm &comm, const auto params) const {
    ygm::container::bag<std::string> path_bag =
        get_file_paths(comm, params.input_path());

    ygm::container::map<std::size_t, registers_type> embed_map(comm);

    transform_ptr_type transform_ptr(
        std::make_shared<transform_type>(params.seed()));

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
    if (!std::filesystem::exists(params.output_path(), ec)) {
      if (!std::filesystem::create_directories(params.output_path(), ec) &&
          ec) {
        throw std::runtime_error(
            "Failed to create directory: " + params.output_path() +
            ", error: " + ec.message());
      }
    }

    // create the rank-wise output file
    std::ostringstream oss;
    oss << params.output_path() << "/" << comm.rank() << ".txt";
    const std::filesystem::path out_path(oss.str());
    std::ofstream               ofs(out_path);
    if (!ofs.is_open()) {
      throw std::runtime_error("Failed to open file: " + out_path.string());
    }

    // dump sketches to file rankwise, line-by-line
    embed_map.for_all(
        [&ofs](const std::size_t index, const registers_type &embedding) {
          ofs << index << embedding << std::endl;
        });
    comm.barrier();

    if (!ofs) {
      throw std::runtime_error("Failed while writing to file: " +
                               out_path.string());
    }
  }
};

// Here we create a simple struct to hold CLI parameters using
// krowkee::parameters.
struct parameters : public krowkee::parameters {
  using base_type = krowkee::parameters;

  krowkee::parameter<std::string>   input_path;
  krowkee::parameter<std::string>   output_path;
  krowkee::parameter<std::uint64_t> range_size;
  krowkee::parameter<std::uint64_t> replication_count;
  krowkee::parameter<int>           seed;

  parameters()
      : base_type(),
        input_path("input_path", "Path to input (file or directory of files)",
                   'i', true, ""),
        output_path("input_path", "Path to input (file or directory of files)",
                    'o', true, ""),
        range_size("range_size", "Range of hash functors", 'r', true, 32),
        replication_count("replication_count",
                          "Number of tiled sketch transforms", 'R', true, 4),
        seed("seed", "Random seed", 's', true, 4) {
    _params.push_back(&input_path);
    _params.push_back(&output_path);
    _params.push_back(&range_size);
    _params.push_back(&replication_count);
    _params.push_back(&seed);
  }

  bool _help_needed() const override {
    bool ret = base_type::_help_needed();
    if (input_path() == "") {
      std::cout << "Must indicate an input file or directory." << std::endl;
      return true;
    }
    if (output_path() == "") {
      std::cout << "Must indicate an output file or directory." << std::endl;
      return true;
    }
    return ret;
  }
};

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    parameters params = krowkee::parse_cmd_line<parameters>(argc, argv);
    if (world.rank0()) {
      std::cout << params << std::endl;
    }

    krowkee::dispatch<embed, void>{params.range_size(),
                                   params.replication_count()}(world, params);
  }
}