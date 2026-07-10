// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <parameters.hpp>

#include <chrono>
#include <memory>
#include <vector>

struct Stopwatch {
  Stopwatch() { reset(); }

  double elapsed() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(now - _start);
    return (double)elapsed.count() / 100000;
  }

  void reset() { _start = std::chrono::steady_clock::now(); }

 private:
  std::chrono::time_point<std::chrono::steady_clock> _start;
};

//// Init timing

template <typename ContainerType>
double time_container_init(const Parameters &params) {
  Stopwatch     stopwatch;
  ContainerType con(params);
  return stopwatch.elapsed();
}

template <typename SketchType>
double time_sketch_init(const Parameters &params) {
  using sketch_type        = SketchType;
  using transform_type     = typename sketch_type::transform_type;
  using transform_ptr_type = std::shared_ptr<transform_type>;

  transform_ptr_type transform_ptr =
      std::make_shared<transform_type>(params.seed);

  Stopwatch   stopwatch;
  sketch_type sketch(transform_ptr, params);
  return stopwatch.elapsed();
}

template <typename ContainerType, typename... ContainerTypes>
void profile_container_init(
    std::vector<std::pair<std::string, double>> &profiles,
    const Parameters                            &params) {
  double init_time = 0.0;
  for (std::size_t i(0); i < params.iterations; ++i) {
    init_time += time_container_init<ContainerType>(params);
  }
  init_time /= params.iterations;
  profiles.push_back({ContainerType::name(), init_time});
  if constexpr (sizeof...(ContainerTypes) > 0) {
    profile_container_init<ContainerTypes...>(profiles, params);
  }
  return;
}

template <typename... ContainerTypes>
std::vector<std::pair<std::string, double>> profile_container_init(
    const Parameters &params) {
  std::vector<std::pair<std::string, double>> profiles;
  if constexpr (sizeof...(ContainerTypes) > 0) {
    profile_container_init<ContainerTypes...>(profiles, params);
  }
  return profiles;
}

template <typename SketchType, typename... SketchTypes>
void profile_sketch_init(std::vector<std::pair<std::string, double>> &profiles,
                         const Parameters                            &params) {
  double init_time = 0.0;
  for (std::size_t i(0); i < params.iterations; ++i) {
    init_time += time_sketch_init<SketchType>(params);
  }
  init_time /= params.iterations;
  profiles.push_back({SketchType::full_name(), init_time});
  if constexpr (sizeof...(SketchTypes) > 0) {
    profile_sketch_init<SketchTypes...>(profiles, params);
  }
  return;
}

template <typename... SketchTypes>
std::vector<std::pair<std::string, double>> profile_sketch_init(
    const Parameters &params) {
  std::vector<std::pair<std::string, double>> profiles;
  if constexpr (sizeof...(SketchTypes) > 0) {
    profile_sketch_init<SketchTypes...>(profiles, params);
  }
  return profiles;
}

//// Histogram timing

template <typename ContainerType, typename SampleType>
double time_insert(const std::vector<SampleType> &samples, ContainerType &con) {
  Stopwatch stopwatch;
  for (const SampleType &sample : samples) {
    con.insert(sample);
  }
  return stopwatch.elapsed();
}

template <typename ContainerType, typename SampleType>
double time_insert(const std::vector<SampleType> &samples,
                   const Parameters              &params) {
  ContainerType con(params);
  return time_insert(samples, con);
}

template <typename SampleType, typename HistType, typename... HistTypes>
void profile_container_histogram(
    std::vector<std::pair<std::string, double>> &profiles,
    const std::vector<SampleType> &samples, const Parameters &params) {
  double hist_time = time_insert<HistType>(samples, params);
  profiles.push_back({HistType::name(), hist_time});
  if constexpr (sizeof...(HistTypes) > 0) {
    profile_container_histogram<SampleType, HistTypes...>(profiles, samples,
                                                          params);
  }
  return;
}

template <typename SampleType, typename... HistTypes>
std::vector<std::pair<std::string, double>> profile_container_histogram(
    const std::vector<SampleType> &samples, const Parameters &params) {
  std::vector<std::pair<std::string, double>> profiles;
  if constexpr (sizeof...(HistTypes) > 0) {
    profile_container_histogram<SampleType, HistTypes...>(profiles, samples,
                                                          params);
  }
  return profiles;
}

template <typename SampleType, typename SketchType, typename... SketchTypes>
void profile_sketch_histogram(
    std::vector<std::pair<std::string, double>> &profiles,
    const std::vector<SampleType> &samples, const Parameters &params) {
  using sketch_type           = SketchType;
  using transform_type        = typename sketch_type::transform_type;
  using transform_ptr_typeype = std::shared_ptr<transform_type>;

  transform_ptr_typeype transform_ptr(
      std::make_shared<transform_type>(params.seed));
  sketch_type sketch(transform_ptr, params);
  double      sketch_time = time_insert(samples, sketch);
  profiles.push_back({sketch_type::full_name(), sketch_time});
  if constexpr (sizeof...(SketchTypes) > 0) {
    profile_sketch_histogram<SampleType, SketchTypes...>(profiles, samples,
                                                         params);
  }
  return;
}

template <typename SampleType, typename... SketchTypes>
std::vector<std::pair<std::string, double>> profile_sketch_histogram(
    const std::vector<SampleType> &samples, const Parameters &params) {
  std::vector<std::pair<std::string, double>> profiles;
  if constexpr (sizeof...(SketchTypes) > 0) {
    profile_sketch_histogram<SampleType, SketchTypes...>(profiles, samples,
                                                         params);
  }
  return profiles;
}
