/* Copyright (c) 2023 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include <string>
#include <unordered_map>
#include <utility>

#include "xpuautoshard/common/hsp_inference/hsp_inference.h"
#include "xpuautoshard/common/op_desc.h"
#include "xpuautoshard/common/ref_base.h"

namespace as {

class HspInferenceFactory;
using HspInferenceFactoryRef = Ref<HspInferenceFactory>;

/**
 * @brief A factory for creating HspInference objects
 *
 */
class HspInferenceFactory {
 public:
  virtual ~HspInferenceFactory() = default;
  /**
   * @brief Get a factory for creating HspInference objects for given `op_name`
   *
   * @param op_name
   * @return HspInferenceFactory&
   */
  static HspInferenceFactory& get(const std::string& op_name);

  /**
   * @brief Register a factory on an op `op_name`.
   *
   * @param op_name
   * @param factory
   */
  static void registerFactory(const std::string& op_name,
                              HspInferenceFactoryRef factory);

  /**
   * @brief Create an HspInference object with the op metadata specified in
   * `op_desc`
   *
   * @param op_desc
   * @return HspInferenceRef
   */
  virtual HspInferenceRef create(OpDescRef op_desc) = 0;

 private:
  static std::unordered_map<std::string, HspInferenceFactoryRef> factory_map_;
};

/**
 * @brief Define the HSP inference factory class for op with `op_name` and with
 * `HSP_INF_CLS`
 *
 */
#define DEFINE_HSP_INFERENCE_FACTORY(HSP_INF_CLS)               \
  class HSP_INF_CLS##Factory : public as::HspInferenceFactory { \
   public:                                                      \
    HspInferenceRef create(OpDescRef op_desc) override {        \
      return makeRef<HSP_INF_CLS, HspInference>(op_desc);       \
    }                                                           \
  };

/**
 * @brief Define the HSP inference class
 *
 */
#define DEFINE_HSP_INFERENCE(HSP_INF_CLS)                                   \
  class HSP_INF_CLS : public HspInference {                                 \
   public:                                                                  \
    virtual ~HSP_INF_CLS() = default;                                       \
    explicit HSP_INF_CLS(OpDescRef op_desc) : as::HspInference(op_desc) {}  \
    bool doInfer(const as::ShardingPropertyRefVec& input_props,             \
                 const as::ShardingPropertyRefVec& output_props) override { \
      return HSP_INF_CLS##Infer(*op_desc_, input_props, output_props);      \
    }                                                                       \
  };

#define DEFINE_HSP_INFERENCE_AND_FACTORY(HSP_INF_CLS) \
  DEFINE_HSP_INFERENCE(HSP_INF_CLS)                   \
  DEFINE_HSP_INFERENCE_FACTORY(HSP_INF_CLS)

namespace detail {

/**
 * @brief A util class for invoking initialization functions at the global
 * scope.
 *
 */
template <typename Factory>
class Registrator {
 public:
  Registrator() {}
  Registrator& operator+(const std::string& op_name) & {
    HspInferenceFactory::registerFactory(
        op_name, makeRef<Factory, HspInferenceFactory>());
    return *this;
  }
  Registrator&& operator+(const std::string& op_name) && {
    HspInferenceFactory::registerFactory(
        op_name, makeRef<Factory, HspInferenceFactory>());
    return std::move(*this);
  }
};

}  // namespace detail

#define _GLOBAL_HSP_FACTORY_REGNAME_FOR(HSP_INF_CLS) HSP_INF_CLS##_factory_reg__

#define REGISTER_HSP_INFERENCE_FACTORY(HSP_INF_CLS)          \
  static auto _GLOBAL_HSP_FACTORY_REGNAME_FOR(HSP_INF_CLS) = \
      detail::Registrator<HSP_INF_CLS##Factory>()

#define REGISTER_HSP_INFERENCE_FACTORY_FOR(HSP_INF_CLS, op_name) \
  REGISTER_HSP_INFERENCE_FACTORY(HSP_INF_CLS) + (op_name)

#define DEFINE_AND_REGISTER_HSP_INFERENCE(HSP_INF_CLS, op_name) \
  DEFINE_HSP_INFERENCE_AND_FACTORY(HSP_INF_CLS)                 \
  REGISTER_HSP_INFERENCE_FACTORY_FOR(HSP_INF_CLS, op_name)

}  // namespace as
