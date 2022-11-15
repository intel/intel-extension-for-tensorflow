/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_COMMON_RELU_OP_H_
#define ITEX_CORE_KERNELS_COMMON_RELU_OP_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "itex/core/kernels/common/eltwise_base.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/numeric_op.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::algorithm;
using dnnl::eltwise_backward;
using dnnl::eltwise_forward;
using dnnl::engine;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;

namespace itex {
template <typename Device, typename T>
class ReluBaseOp : public EltwiseBaseOp<Device, T> {
 public:
  explicit ReluBaseOp(OpKernelConstruction* context, dnnl::algorithm algo,
                      float alpha, float beta)
      : EltwiseBaseOp<Device, T>(context, algo, alpha, beta) {}
};

template <typename Device, typename T>
class ReluOp : public ReluBaseOp<Device, T> {
 public:
  explicit ReluOp(OpKernelConstruction* context)
      : ReluBaseOp<Device, T>(context, dnnl::algorithm::eltwise_relu, 0.0f,
                              0.0f) {}
};

template <typename Device, typename T>
class EluOp : public ReluBaseOp<Device, T> {
 public:
  explicit EluOp(OpKernelConstruction* context)
      : ReluBaseOp<Device, T>(context, dnnl::algorithm::eltwise_elu, 1.0f,
                              0.0f) {}
};

template <typename Device, typename T>
class Relu6Op : public ReluBaseOp<Device, T> {
 public:
  explicit Relu6Op(OpKernelConstruction* context)
      : ReluBaseOp<Device, T>(context, dnnl::algorithm::eltwise_clip_v2, 0.0f,
                              6.0f) {}
};

template <typename Device, typename T>
class LeakyReluOp : public ReluBaseOp<Device, T> {
 public:
  explicit LeakyReluOp(OpKernelConstruction* context)
      : ReluBaseOp<Device, T>(context, dnnl::algorithm::eltwise_relu, 0.0f,
                              0.0f) {
    float alpha;
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES(
        context, alpha <= 1,
        errors::InvalidArgument("OneDNN LeakyRelu only supports alpha <= 1. "
                                "alpha is: ",
                                alpha));

    this->alpha_ = alpha;
  }
};

template <typename Device, typename T>
class GeluOp : public ReluBaseOp<Device, T> {
 public:
  explicit GeluOp(OpKernelConstruction* context)
      : ReluBaseOp<Device, T>(context, dnnl::algorithm::eltwise_gelu_erf, 0.0f,
                              0.0f) {
    if (context->HasAttr("approximate")) {
      OP_REQUIRES_OK(context, context->GetAttr("approximate", &approximate_));
      this->alg_kind_ = approximate_ ? algorithm::eltwise_gelu_tanh
                                     : algorithm::eltwise_gelu_erf;
    }
  }

 private:
  bool approximate_ = true;
};

template <typename Device, typename T>
class MishOp : public ReluBaseOp<Device, T> {
 public:
  explicit MishOp(OpKernelConstruction* context)
      : ReluBaseOp<Device, T>(context, dnnl::algorithm::eltwise_mish, 0.0f,
                              0.0f) {}
};

template <typename Device, typename T>
class SwishOp : public ReluBaseOp<Device, T> {
 public:
  explicit SwishOp(OpKernelConstruction* context)
      : ReluBaseOp<Device, T>(context, dnnl::algorithm::eltwise_swish, 1.0f,
                              0.0f) {
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &this->alpha_));
  }
};

template <typename Device, typename T>
class ReluGradBaseOp : public EltwiseGradBaseOp<Device, T> {
 public:
  explicit ReluGradBaseOp(OpKernelConstruction* context, dnnl::algorithm algo,
                          float alpha, float beta)
      : EltwiseGradBaseOp<Device, T>(context, algo, alpha, beta) {}
};

template <typename Device, typename T>
class ReluGradOp : public ReluGradBaseOp<Device, T> {
 public:
  explicit ReluGradOp(OpKernelConstruction* context)
      : ReluGradBaseOp<Device, T>(context, dnnl::algorithm::eltwise_relu, 0.0f,
                                  0.0f) {}

  int GetDiffDstIndex() const override { return 0; }
  int GetSrcIndex() const override { return 1; }
  int GetDiffSrcIndex() const override { return 0; }
  int GetTypeOfInputTensorFromFwdOp() const override { return DNNL_ARG_SRC; }
};

template <typename Device, typename T>
class EluGradOp : public ReluGradBaseOp<Device, T> {
 public:
  explicit EluGradOp(OpKernelConstruction* context)
      : ReluGradBaseOp<Device, T>(
            context, dnnl::algorithm::eltwise_elu_use_dst_for_bwd, 1.0f, 0.0f) {
  }

  int GetDiffDstIndex() const override { return 0; }
  int GetSrcIndex() const override { return 1; }
  int GetDiffSrcIndex() const override { return 0; }
  int GetTypeOfInputTensorFromFwdOp() const override { return DNNL_ARG_DST; }
};

template <typename Device, typename T>
class Relu6GradOp : public ReluGradBaseOp<Device, T> {
 public:
  explicit Relu6GradOp(OpKernelConstruction* context)
      : ReluGradBaseOp<Device, T>(
            context, dnnl::algorithm::eltwise_clip_v2_use_dst_for_bwd, 0.0f,
            6.0f) {}

  int GetDiffDstIndex() const override { return 0; }
  int GetSrcIndex() const override { return 1; }
  int GetDiffSrcIndex() const override { return 0; }
  int GetTypeOfInputTensorFromFwdOp() const override { return DNNL_ARG_DST; }
};

template <typename Device, typename T>
class LeakyReluGradOp : public ReluGradBaseOp<Device, T> {
 public:
  explicit LeakyReluGradOp(OpKernelConstruction* context)
      : ReluGradBaseOp<Device, T>(context, dnnl::algorithm::eltwise_relu, 0.0f,
                                  0.0f) {
    float alpha;
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES(
        context, alpha <= 1,
        errors::InvalidArgument("OneDNN LeakyRelu only supports alpha <= 1. "
                                "alpha is: ",
                                alpha));

    this->alpha_ = alpha;
  }

  int GetDiffDstIndex() const override { return 0; }
  int GetSrcIndex() const override { return 1; }
  int GetDiffSrcIndex() const override { return 0; }
  int GetTypeOfInputTensorFromFwdOp() const override { return DNNL_ARG_SRC; }
};

template <typename Device, typename T>
class GeluGradOp : public ReluGradBaseOp<Device, T> {
 public:
  explicit GeluGradOp(OpKernelConstruction* context)
      : ReluGradBaseOp<Device, T>(context, dnnl::algorithm::eltwise_relu, 0.0f,
                                  0.0f) {
    if (context->HasAttr("approximate")) {
      OP_REQUIRES_OK(context, context->GetAttr("approximate", &approximate_));
      this->alg_kind_ = approximate_ ? algorithm::eltwise_gelu_tanh
                                     : algorithm::eltwise_gelu_erf;
    }
  }

  int GetDiffDstIndex() const override { return 0; }
  int GetSrcIndex() const override { return 1; }
  int GetDiffSrcIndex() const override { return 0; }
  int GetTypeOfInputTensorFromFwdOp() const override { return DNNL_ARG_SRC; }

 private:
  bool approximate_ = true;
};

template <typename Device, typename T>
class SwishGradOp : public ReluGradBaseOp<Device, T> {
 public:
  explicit SwishGradOp(OpKernelConstruction* context)
      : ReluGradBaseOp<Device, T>(context, dnnl::algorithm::eltwise_swish, 1.0f,
                                  0.0f) {
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &this->alpha_));
  }

  int GetDiffDstIndex() const override { return 0; }
  int GetSrcIndex() const override { return 1; }
  int GetDiffSrcIndex() const override { return 0; }
  int GetTypeOfInputTensorFromFwdOp() const override { return DNNL_ARG_SRC; }
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_COMMON_RELU_OP_H_
