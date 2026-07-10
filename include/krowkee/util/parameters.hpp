// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other
// powersqueeze Project Developers.See the top-level COPYRIGHT file for details.

#pragma once

#include <krowkee/util/string_help.hpp>

#include <unistd.h>
#include <any>

namespace krowkee {

template <typename T>
struct parameter {
  using value_type = T;

 protected:
  std::string _name;
  std::string _usage;
  const char  _flag;
  bool        _parameterized;
  T           _value;

 public:
  parameter(std::string name, std::string usage, char flag, bool parameterized,
            T value)
      : _name(name),
        _usage(usage),
        _flag(flag),
        _parameterized(parameterized),
        _value(value) {}

  const T &operator()() const { return _value; }
  void     set(T val) { _value = val; }

  char flag() const { return _flag; }

  void usage(std::stringstream &ss) const {
    ss << " -" << flag() << ((_parameterized) ? " <arg>" : "\t") << "\t"
       << _usage << std::endl;
  }

  void print(std::ostream &os) const {
    os << "\t" << _name << ":\t" << _value << std::endl;
  }

  void print_name(std::ostream &os) const { os << _name; }
  void print_value(std::ostream &os) const { os << _value; }

  template <typename C, typename OptType>
  void parse(C c, OptType optarg);

  std::string parse_str() const {
    std::stringstream ss;
    ss << flag();
    if (_parameterized) {
      ss << ":";
    }
    return ss.str();
  }
};

template <>
template <typename C, typename OptType>
void parameter<int>::parse(C c, OptType optarg) {
  if (c == _flag) {
    set(atoi(optarg));
  }
}
template <>
template <typename C, typename OptType>
void parameter<float>::parse(C c, OptType optarg) {
  if (c == _flag) {
    set(atof(optarg));
  }
}
template <>
template <typename C, typename OptType>
void parameter<double>::parse(C c, OptType optarg) {
  if (c == _flag) {
    set(atof(optarg));
  }
}
template <>
template <typename C, typename OptType>
void parameter<std::uint64_t>::parse(C c, OptType optarg) {
  if (c == _flag) {
    set(atoll(optarg));
  }
}
template <>
template <typename C, typename OptType>
void parameter<bool>::parse(C c, OptType optarg) {
  if (c == _flag) {
    set(true);
  }
}
template <>
template <typename C, typename OptType>
void parameter<std::string>::parse(C c, OptType optarg) {
  if (c == _flag) {
    set(optarg);
  }
}

struct parameters {
 protected:
  std::vector<std::any> _params;

 public:
  parameter<bool> print_help;

  parameters()
      : print_help("print_help?", "Indicates whether to print usage string",
                   'h', false, false) {
    _params.push_back(&print_help);
  }

  void usage() const {
    std::stringstream ss;
    ss << "Usage:\n";
    for (const auto &param : _params) {
      if (param.type() == typeid(parameter<std::uint64_t> *)) {
        std::any_cast<parameter<std::uint64_t> *>(param)->usage(ss);
      } else if (param.type() == typeid(parameter<int> *)) {
        std::any_cast<parameter<int> *>(param)->usage(ss);
      } else if (param.type() == typeid(parameter<float> *)) {
        std::any_cast<parameter<float> *>(param)->usage(ss);
      } else if (param.type() == typeid(parameter<double> *)) {
        std::any_cast<parameter<double> *>(param)->usage(ss);
      } else if (param.type() == typeid(parameter<std::string> *)) {
        std::any_cast<parameter<std::string> *>(param)->usage(ss);
      } else if (param.type() == typeid(parameter<bool> *)) {
        std::any_cast<parameter<bool> *>(param)->usage(ss);
      }
    }
    std::cerr << ss.str();
  }

  std::string csv_names() const {
    std::stringstream ss;
    const std::size_t elem_count{_params.size()};
    std::size_t       counter{0};
    for (const auto &param : _params) {
      if (param.type() == typeid(parameter<std::uint64_t> *)) {
        std::any_cast<parameter<std::uint64_t> *>(param)->print_name(ss);
      } else if (param.type() == typeid(parameter<int> *)) {
        std::any_cast<parameter<int> *>(param)->print_name(ss);
      } else if (param.type() == typeid(parameter<float> *)) {
        std::any_cast<parameter<float> *>(param)->print_name(ss);
      } else if (param.type() == typeid(parameter<double> *)) {
        std::any_cast<parameter<double> *>(param)->print_name(ss);
      } else if (param.type() == typeid(parameter<std::string> *)) {
        std::any_cast<parameter<std::string> *>(param)->print_name(ss);
      } else if (param.type() == typeid(parameter<bool> *)) {
        std::any_cast<parameter<bool> *>(param)->print_name(ss);
      }
      if (++counter < elem_count) {
        ss << ",";
      }
    }
    return ss.str();
  }

  std::string csv_values() const {
    std::stringstream ss;
    const std::size_t elem_count{_params.size()};
    std::size_t       counter{0};
    counter = 0;
    for (const auto &param : _params) {
      if (param.type() == typeid(parameter<std::uint64_t> *)) {
        std::any_cast<parameter<std::uint64_t> *>(param)->print_value(ss);
      } else if (param.type() == typeid(parameter<int> *)) {
        std::any_cast<parameter<int> *>(param)->print_value(ss);
      } else if (param.type() == typeid(parameter<float> *)) {
        std::any_cast<parameter<float> *>(param)->print_value(ss);
      } else if (param.type() == typeid(parameter<double> *)) {
        std::any_cast<parameter<double> *>(param)->print_value(ss);
      } else if (param.type() == typeid(parameter<std::string> *)) {
        std::any_cast<parameter<std::string> *>(param)->print_value(ss);
      } else if (param.type() == typeid(parameter<bool> *)) {
        std::any_cast<parameter<bool> *>(param)->print_value(ss);
      }
      if (++counter < elem_count) {
        ss << ",";
      }
    }
    return ss.str();
  }

  std::ostream &print(std::ostream &os) const {
    for (const auto &param : _params) {
      if (param.type() == typeid(parameter<std::uint64_t> *)) {
        std::any_cast<parameter<std::uint64_t> *>(param)->print(os);
      } else if (param.type() == typeid(parameter<int> *)) {
        std::any_cast<parameter<int> *>(param)->print(os);
      } else if (param.type() == typeid(parameter<float> *)) {
        std::any_cast<parameter<float> *>(param)->print(os);
      } else if (param.type() == typeid(parameter<double> *)) {
        std::any_cast<parameter<double> *>(param)->print(os);
      } else if (param.type() == typeid(parameter<std::string> *)) {
        std::any_cast<parameter<std::string> *>(param)->print(os);
      } else if (param.type() == typeid(parameter<bool> *)) {
        std::any_cast<parameter<bool> *>(param)->print(os);
      }
    }
    return os;
  }

  template <typename C, typename OptType>
  void parse(C c, OptType optarg) {
    for (const auto &param : _params) {
      if (param.type() == typeid(parameter<std::uint64_t> *)) {
        std::any_cast<parameter<std::uint64_t> *>(param)->parse(c, optarg);
      } else if (param.type() == typeid(parameter<int> *)) {
        std::any_cast<parameter<int> *>(param)->parse(c, optarg);
      } else if (param.type() == typeid(parameter<float> *)) {
        std::any_cast<parameter<float> *>(param)->parse(c, optarg);
      } else if (param.type() == typeid(parameter<double> *)) {
        std::any_cast<parameter<double> *>(param)->parse(c, optarg);
      } else if (param.type() == typeid(parameter<std::string> *)) {
        std::any_cast<parameter<std::string> *>(param)->parse(c, optarg);
      } else if (param.type() == typeid(parameter<bool> *)) {
        std::any_cast<parameter<bool> *>(param)->parse(c, optarg);
      }
    }
  }

  virtual bool _help_needed() const { return print_help(); }

  bool check_help_needed() const { return _help_needed(); }

  virtual void _apply_defaults() { return; }

  std::string parse_str() {
    std::stringstream ss;
    for (const auto &param : _params) {
      if (param.type() == typeid(parameter<std::uint64_t> *)) {
        ss << std::any_cast<parameter<std::uint64_t> *>(param)->parse_str();
      } else if (param.type() == typeid(parameter<int> *)) {
        ss << std::any_cast<parameter<int> *>(param)->parse_str();
      } else if (param.type() == typeid(parameter<float> *)) {
        ss << std::any_cast<parameter<float> *>(param)->parse_str();
      } else if (param.type() == typeid(parameter<double> *)) {
        ss << std::any_cast<parameter<double> *>(param)->parse_str();
      } else if (param.type() == typeid(parameter<std::string> *)) {
        ss << std::any_cast<parameter<std::string> *>(param)->parse_str();
      } else if (param.type() == typeid(parameter<bool> *)) {
        ss << std::any_cast<parameter<bool> *>(param)->parse_str();
      }
    }
    return ss.str();
  }
};

std::ostream &operator<<(std::ostream &os, const parameters &params) {
  return params.print(os);
}

template <typename ParametersType>
ParametersType parse_cmd_line(int argc, char **argv) {
  ParametersType params{};
  int            c;
  std::string    str = params.parse_str() + " ";
  while ((c = getopt(argc, argv, str.c_str())) != -1) {
    params.parse(c, optarg);
  }
  if (params.check_help_needed()) {
    params.usage();
    exit(-1);
  }
  params._apply_defaults();

  return params;
}

}  // namespace krowkee
