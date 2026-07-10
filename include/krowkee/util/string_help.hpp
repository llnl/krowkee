// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <krowkee/util/sketch_types.hpp>

#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace krowkee {

using krowkee::util::sketch_impl_type;

template <typename Out>
void split(const std::string &s, char delim, Out result) {
  std::istringstream iss(s);
  std::string        item;
  while (std::getline(iss, item, delim)) {
    *result++ = item;
  }
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

template <typename ParametersType>
std::string container_name(const ParametersType &params) {
  if (params.use_map() == true) {
    return "ygm::map";
  } else {
    return "ygm::array";
  }
}

void repeat_cmd_line(int argc, char **argv) {
  std::cout << "CMD line:";
  for (int i = 0; i < argc; ++i) {
    std::cout << " " << argv[i];
  }
  std::cout << std::endl;
}

template <typename... Args>
std::string concatenate_printable(Args &&...args) {
  std::stringstream ss;
  (ss << ... << args);
  return ss.str();
}

std::ostream &operator<<(std::ostream &os, const sketch_impl_type &st) {
  if (st == sketch_impl_type::cst) {
    os << "cst";
    // } else if (st == sketch_impl_type::sparse_cst) {
    //   os << "sparse_cst";
    // } else if (st == sketch_impl_type::promotable_cst) {
    //   os << "promotable_cst";
  } else if (st == sketch_impl_type::fwht) {
    os << "fwht";
  }
  return os;
}

}  // namespace krowkee
