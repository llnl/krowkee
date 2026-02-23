// Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <krowkee/container/compacting_map.hpp>

#include <krowkee/hash/util.hpp>

#include <krowkee/util/cmap_types.hpp>
#include <krowkee/util/runtime.hpp>

#if __has_include(<boost/container/flat_map.hpp>)
#include <boost/container/flat_map.hpp>
#endif

#if __has_include(<cereal/cereal.hpp>)
#include <check_archive.hpp>
#endif

#include <getopt.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using krowkee::chirp;
using krowkee::do_test;
using krowkee::make_shared_functor;
using krowkee::print_line;

using cmap_type = krowkee::container::compacting_map<int, int>;

#if __has_include(<boost/container/flat_map.hpp>)
using cmap_boost_type =
    krowkee::container::compacting_map<int, int, boost::container::flat_map>;
#endif

using map_type  = std::map<int, int>;
using pair_type = std::pair<int, int>;

using Clock   = std::chrono::system_clock;
using ns_type = std::chrono::nanoseconds;

std::ostream &operator<<(std::ostream &os, const pair_type &pair) {
  os << "(" << pair.first << "," << pair.second << ")";
  return os;
}

std::vector<int> get_random_vector(const int count, const std::uint32_t seed) {
  std::vector<int> vec;
  vec.reserve(count);
  for (int i(1); i <= count; ++i) {
    vec.push_back(i);
  }
  std::random_device rd;
  std::mt19937       g(seed);
  std::shuffle(std::begin(vec), std::end(vec), g);
  return vec;
}

/**
 * Struct bundling the experiment parameters.
 */
struct Parameters {
  std::vector<int> to_insert;
  std::uint32_t    count;
  std::size_t      thresh;
  std::uint32_t    seed;
  bool             verbose;
};

struct clear_check {
  const char *name() { return "clear check"; }

  template <typename MapType>
  void operator()(MapType &cm, const Parameters params) const {
    bool init_empty = cm.empty();

    CHECK_CONDITION(init_empty == true, "initial empty");

    cm.insert({1, 1});
    bool not_empty = cm.empty();
    CHECK_CONDITION(not_empty == false, "post-insert not empty");

    cm.clear();
    bool clear_empty = cm.empty();
    CHECK_CONDITION(clear_empty == true, "post-clear empty");
  }
};

struct insert_check {
  const char *name() { return "insert check"; }

  template <typename MapType>
  void operator()(MapType &cm, const Parameters params) const {
    int  counter(0);
    bool all_success(true);

    auto cm_start(Clock::now());
    std::for_each(std::begin(params.to_insert), std::end(params.to_insert),
                  [&](const int i) {
                    bool success = cm.insert({i, i});
                    // auto [iter, success] = cm.insert({i, i});
                    counter++;
                    if (success == false) {
                      std::cout << "FAILED " << pair_type{i, i} << std::endl;
                      all_success == false;
                    }
                    if (params.verbose == true) {
                      std::cout << "on " << counter << "th insert (size "
                                << cm.size() << ") : " << pair_type{i, i} << ":"
                                << std::endl;
                      std::cout << cm.print_state() << std::endl << std::endl;
                    }
                  });
    CHECK_CONDITION(all_success, "iterative insert");

    map_type map;
    auto     map_start(Clock::now());
    std::for_each(std::begin(params.to_insert), std::end(params.to_insert),
                  [&](const int i) {
                    auto [iter, success] = map.insert({i, i});
                  });
    auto end(Clock::now());

    {  // Test whether attempting to insert a dynamic element correctly fails
      bool success = cm.insert({params.to_insert[params.count - 1], 1});
      // auto [iter, success] = cm.insert({to_insert[count - 1], 1});
      CHECK_CONDITION(success == false, "dynamic insert");
    }
    {  // Test whether attempting to insert an archive element correctly fails
      bool success = cm.insert({params.to_insert[0], 1});
      // auto [iter, success] = cm.insert({to_insert[0], 1});
      CHECK_CONDITION(success == false, "archive insert");
    }
    auto cm_ns(
        std::chrono::duration_cast<ns_type>(map_start - cm_start).count());
    auto map_ns(std::chrono::duration_cast<ns_type>(end - map_start).count());
    std::cout << "Inserted " << params.count
              << " elements into compact map with compaction threshold "
              << cm.compaction_threshold() << " in " << cm_ns
              << " ns versus std::map in " << map_ns << " ns ("
              << ((double)cm_ns / (double)map_ns) << " slowdown)" << std::endl;
  }
};

struct find_check {
  const char *name() { return "find check"; }

  template <typename MapType>
  void operator()(MapType &cm, const Parameters params) const {
    {  // Test whether * returns the correct reference from the find() iterator
      auto q            = cm.find(params.to_insert[params.count - 1]);
      bool find_success = (*q).second == params.to_insert[params.count - 1];
      CHECK_CONDITION(find_success == true, "positive find ((*).second)");
    }
    {  // Test whether -> returns the correct reference from the find() iterator
      auto q            = cm.find(params.to_insert[params.count - 1]);
      bool find_success = q->second == params.to_insert[params.count - 1];
      CHECK_CONDITION(find_success == true, "positive find (*->second)");
    }
    {  // Test whether find() returns end() iterator on unseen element.
      auto q         = cm.find(-10);
      bool finds_end = q == std::end(cm);
      CHECK_CONDITION(finds_end == true, "negative find");
    }
  }
};

template <typename MapType, typename T>
void check_throws_bad_key_at(const MapType &cm, T badkey) {
  auto val = cm.at(badkey);
}

template <typename MapType, typename T>
void check_throws_bad_key_find(const MapType &cm, T badkey) {
  auto val = cm.find(badkey);
}

struct accessor_check {
  const char *name() { return "accessor check"; }

  template <typename MapType>
  void operator()(MapType &cm, const Parameters params) const {
    auto key = params.to_insert[params.count - 1];
    {
      auto q          = cm.find(key);
      auto val1       = cm[key];
      auto val2       = cm.at(key);
      bool acc_agrees = q->second == val1;
      bool at_agrees  = q->second == val2;
      CHECK_CONDITION(acc_agrees == true, "key accessor (operator)");
      CHECK_CONDITION(at_agrees == true, "key accessor (at)");
    }
    {
      int bad_key = -1;
      CHECK_THROWS<std::out_of_range>(check_throws_bad_key_at<MapType, int>,
                                      "bad key accessor (at)", cm, bad_key);
    }
    {
      int newval = 1;
      cm[key]    = newval;
      auto q     = cm.find(key);
      auto val   = cm[key];
      bool reset = val == newval && val == q->second;
      CHECK_CONDITION(reset == true, "key reset operator");
    }
    {
      bool caught(false);
      int  newkey = -1;
      auto val    = cm[newkey];
      CHECK_THROWS<std::logic_error>(check_throws_bad_key_find<MapType, int>,
                                     "bad key find", cm, newkey);
      cm.compactify();
      auto q             = cm.find(newkey);
      bool unset_key_def = val == 0 && val == q->second;
      CHECK_CONDITION(unset_key_def == true, "new unset key");
    }
    {
      int newkey = -2;
      int newval = -2;
      cm[newkey] = newval;
      auto val   = cm[newkey];
      cm.compactify();
      auto q          = cm.find(newkey);
      bool newkey_set = val == newval && val == q->second;
      CHECK_CONDITION(newkey_set == true, "new set key");
    }
  }
};

struct erase_check {
  const char *name() { return "erase check"; }

  template <typename MapType>
  void operator()(MapType &cm, const Parameters params) const {
    int newkey1 = -1;
    int newkey2 = -2;
    {
      std::size_t success1 = cm.erase(newkey1);
      std::size_t success2 = cm.erase(newkey2);
      bool        erasure_success =
          success1 == 1 && success2 == 1 && cm.erased_count() == 2;
      CHECK_CONDITION(erasure_success == true, "archive erase");
    }
    {
      int bad_key = -1;
      CHECK_THROWS<std::out_of_range>(check_throws_bad_key_at<MapType, int>,
                                      "at access erased key", cm, bad_key);
    }
    {
      cm.insert({newkey1, newkey1});
      bool insert_erased_key = cm[newkey1] == newkey1 && cm.erased_count() == 1;
      CHECK_CONDITION(insert_erased_key == true, "insert set erased key");
    }
    {
      cm[newkey2] = newkey2;
      bool acc_assign_erased_key =
          cm[newkey2] == newkey2 && cm.erased_count() == 0;
      CHECK_CONDITION(acc_assign_erased_key == true,
                      "accessor (operator) assignment set erased key");
    }
    int newkey3 = -3;
    {
      cm.insert({newkey3, newkey3});
      int  success       = cm.erase(newkey3);
      bool dynamic_erase = success == 1 && cm.erased_count() == 0;
      CHECK_CONDITION(dynamic_erase == true, "dynamic erase");
    }
    {
      cm[newkey1]         = newkey1;
      auto q              = cm.find(newkey1);
      int  success        = cm.erase(q);
      bool iterator_erase = success == 1 && cm.erased_count() == 1;
      CHECK_CONDITION(iterator_erase == true, "iterator erase");
    }
  }
};

struct const_funcs_check {
  const char *name() { return "const funcs check"; }

  template <typename MapType>
  void operator()(const MapType &cm, const Parameters params) const {
    int good_key = params.to_insert[0];
    int bad_key  = -4;
    {
      auto q           = cm.at(good_key);
      bool at_good_key = q == good_key;
      CHECK_CONDITION(at_good_key == true, "const at (good key)");
    }
    CHECK_THROWS<std::out_of_range>(check_throws_bad_key_at<MapType, int>,
                                    "const at (bad key)", cm, bad_key);
    {
      auto q               = cm.at(good_key, bad_key);
      bool at_good_key_def = q == good_key;
      CHECK_CONDITION(at_good_key_def == true,
                      "const at w/ default (good key) test");
    }
    {
      auto q              = cm.at(bad_key, bad_key);
      bool at_bad_key_def = q == bad_key;
      CHECK_CONDITION(at_bad_key_def == true, "const at w/ default (bad key)");
    }
    {
      auto q             = cm.find(good_key);
      bool find_good_key = q->second == good_key;
      CHECK_CONDITION(find_good_key == true, "const find (good key)");
    }
    {
      auto q                = cm.find(bad_key);
      bool not_find_bad_key = q == std::cend(cm);
      CHECK_CONDITION(not_find_bad_key == true, "const find (bad key)");
    }
    {
      for (const auto &p : cm) {
        if (params.verbose == true) {
          std::cout << p << " ";
        }
      }
      if (params.verbose == true) {
        std::cout << std::endl;
      }
      CHECK_CONDITION(true, "const iteration");
    }
  }
};

struct copy_check {
  const char *name() { return "copy check"; }

  template <typename MapType>
  void operator()(const MapType &cm, const Parameters params) const {
    int new_key = -4;
    {
      MapType cm2(cm);
      bool    copy_success = cm2 == cm;
      CHECK_CONDITION(copy_success == true, "copy constructor");
      cm2[new_key]         = new_key;
      MapType cm3          = cm2;
      bool    swap_success = cm3 == cm2 && cm3 != cm;
      CHECK_CONDITION(swap_success == true, "swap constructor");
    }
  }
};

#if __has_include(<cereal/cereal.hpp>)
template <typename StreamType, typename ArchiveType, typename MapType>
void check_throws_uncompacted_archive(const MapType &cm) {
  StreamType  os;
  ArchiveType oarchive(os);
  oarchive(cm);
}

struct serialize_check {
  const char *name() { return "serialize check"; }

  template <typename MapType>
  void operator()(const MapType &cm, const Parameters params) const {
    CHECK_THROWS<std::logic_error>(
        check_throws_uncompacted_archive<std::stringstream,
                                         cereal::BinaryOutputArchive, MapType>,
        "uncompacted archive (binary)", cm);
    {
      MapType cm1(cm);
      cm1.compactify();
      CHECK_ALL_ARCHIVES(cm1, "compacting map");
    }
  }
};
#endif

template <typename MapType>
void do_experiment(const Parameters params) {
  MapType cm(params.thresh);

  print_line();
  print_line();
  std::cout << "Experiment inserting " << params.count << " elements to "
            << cm.full_name() << std::endl;
  print_line();
  print_line();
  std::cout << std::endl;
  do_test<clear_check>(cm, params);
  do_test<insert_check>(cm, params);
  do_test<find_check>(cm, params);
  do_test<accessor_check>(cm, params);
  do_test<erase_check>(cm, params);
  do_test<const_funcs_check>(cm, params);
  do_test<copy_check>(cm, params);
#if __has_include(<cereal/cereal.hpp>)
  do_test<serialize_check>(cm, params);
#endif
  std::cout << std::endl;
}

void print_help(char *exe_name) {
  std::cout << "\nusage:  " << exe_name << "\n"
            << "\t-c, --count <int>     - number of insertions\n"
            << "\t-m, --map-type <str>  - map type (std, boost)\n"
            << "\t-s, --seed <int>      - random seed\n"
            << "\t-t, --thresh <int>    - compaction threshold\n"
            << "\t-v, --verbose         - print additional debug information.\n"
            << "\t-h, --help            - print this line and exit\n"
            << std::endl;
}

void parse_args(int argc, char **argv, Parameters &params) {
  int c;

  while (1) {
    int                  option_index(0);
    static struct option long_options[] = {
        {"count", required_argument, NULL, 'c'},
        {"seed", required_argument, NULL, 's'},
        {"thresh", required_argument, NULL, 't'},
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}};

    int curind = optind;
    c = getopt_long(argc, argv, "-:c:s:t:vh", long_options, &option_index);
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
      case 's':
        params.seed = std::atol(optarg);
        break;
      case 't':
        params.thresh = std::atoll(optarg);
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

int main(int argc, char **argv) {
  std::uint32_t  count(1000);
  std::uint32_t  seed(0);
  std::size_t    thresh(5);
  bool           verbose(false);

  Parameters params{
      get_random_vector(count, seed), count, thresh, seed, verbose};

  parse_args(argc, argv, params);

  params.to_insert = get_random_vector(params.count, params.seed);

  do_experiment<cmap_type>(params);
#if __has_include(<boost/container/flat_map.hpp>)
  do_experiment<cmap_boost_type>(params);
#endif
  return 0;
}
