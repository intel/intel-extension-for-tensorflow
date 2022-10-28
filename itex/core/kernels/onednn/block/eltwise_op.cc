/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/kernels/onednn/block/eltwise_op.h"

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::algorithm;
using dnnl::eltwise_forward;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;

namespace itex {
template <typename Device, typename T>
class OneDnnReluOp : public OneDnnEltwiseBaseOp<Device, T> {
 public:
  explicit OneDnnReluOp(OpKernelConstruction* context)
      : OneDnnEltwiseBaseOp<Device, T>(context, dnnl::algorithm::eltwise_relu,
                                       0.0f, 0.0f) {}
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnRelu")                  \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<TYPE>("T")       \
                              .HostMemory("features_meta")     \
                              .HostMemory("activations_meta"), \
                          OneDnnReluOp<GPUDevice, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#else
#define REGISTER_KERNEL(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OneDnnRelu").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnReluOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

template <typename Device, typename T>
class OneDnnReluGradOp : public OneDnnEltwiseGradBaseOp<Device, T> {
 public:
  explicit OneDnnReluGradOp(OpKernelConstruction* context)
      : OneDnnEltwiseGradBaseOp<Device, T>(
            context, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f) {}

  int GetDiffDstIndex() const override { return 0; }
  int GetSrcIndex() const override { return 1; }
  int GetDiffSrcIndex() const override { return 0; }
  int GetTypeOfInputTensorFromFwdOp() const override { return DNNL_ARG_SRC; }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnReluGrad")            \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<TYPE>("T")     \
                              .HostMemory("gradients_meta")  \
                              .HostMemory("features_meta")   \
                              .HostMemory("backprops_meta"), \
                          OneDnnReluGradOp<GPUDevice, TYPE>)
TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#else
#define REGISTER_KERNEL(TYPE)                                               \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_OneDnnReluGrad").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnReluGradOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

template <typename Device, typename T>
class OneDnnLeakyReluOp : public OneDnnEltwiseBaseOp<Device, T> {
 public:
  explicit OneDnnLeakyReluOp(OpKernelConstruction* context)
      : OneDnnEltwiseBaseOp<Device, T>(context, dnnl::algorithm::eltwise_relu,
                                       0.0f, 0.0f) {
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

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnLeakyRelu")             \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<TYPE>("T")       \
                              .HostMemory("features_meta")     \
                              .HostMemory("activations_meta"), \
                          OneDnnLeakyReluOp<GPUDevice, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#else
#define REGISTER_KERNEL(TYPE)                                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_OneDnnLeakyRelu").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnLeakyReluOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

template <typename Device, typename T>
class OneDnnLeakyReluGradOp : public OneDnnEltwiseGradBaseOp<Device, T> {
 public:
  explicit OneDnnLeakyReluGradOp(OpKernelConstruction* context)
      : OneDnnEltwiseGradBaseOp<Device, T>(
            context, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f) {
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

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnLeakyReluGrad")       \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<TYPE>("T")     \
                              .HostMemory("gradients_meta")  \
                              .HostMemory("features_meta")   \
                              .HostMemory("backprops_meta"), \
                          OneDnnLeakyReluGradOp<GPUDevice, TYPE>)
TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#else
#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnLeakyReluGrad")    \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          OneDnnLeakyReluGradOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

template <typename Device, typename T>
class OneDnnGeluOp : public OneDnnEltwiseBaseOp<Device, T> {
 public:
  ~OneDnnGeluOp() {}

  explicit OneDnnGeluOp(OpKernelConstruction* context)
      : OneDnnEltwiseBaseOp<Device, T>(
            context, dnnl::algorithm::eltwise_gelu_erf, 0.0f, 0.0f) {
    bool approximate;
    OP_REQUIRES_OK(context, context->GetAttr("approximate", &approximate));
    this->approximate_ = approximate;
    if (approximate == true) {
      this->algo_ = dnnl::algorithm::eltwise_gelu_tanh;
    } else {
      this->algo_ = dnnl::algorithm::eltwise_gelu_erf;
    }
  }

 private:
  bool approximate_;
};

template <typename Device, typename T>
class OneDnnGeluGradOp : public OneDnnEltwiseGradBaseOp<Device, T> {
 public:
  explicit OneDnnGeluGradOp(OpKernelConstruction* context)
      : OneDnnEltwiseGradBaseOp<Device, T>(
            context, dnnl::algorithm::eltwise_gelu_erf, 0.0f, 0.0f) {
    bool approximate;
    OP_REQUIRES_OK(context, context->GetAttr("approximate", &approximate));
    this->approximate_ = approximate;
    if (approximate == true) {
      this->algo_ = dnnl::algorithm::eltwise_gelu_tanh;
    } else {
      this->algo_ = dnnl::algorithm::eltwise_gelu_erf;
    }
  }

  int GetDiffDstIndex() const override { return 0; }
  int GetSrcIndex() const override { return 1; }
  int GetDiffSrcIndex() const override { return 0; }
  int GetTypeOfInputTensorFromFwdOp() const override { return DNNL_ARG_SRC; }

 private:
  bool approximate_;
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnGelu")                  \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<TYPE>("T")       \
                              .HostMemory("features_meta")     \
                              .HostMemory("activations_meta"), \
                          OneDnnGeluOp<GPUDevice, TYPE>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(TYPE)                                \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnGeluGrad")            \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<TYPE>("T")     \
                              .HostMemory("gradients_meta")  \
                              .HostMemory("features_meta")   \
                              .HostMemory("backprops_meta"), \
                          OneDnnGeluGradOp<GPUDevice, TYPE>);
TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#else
#define REGISTER_KERNEL(TYPE)                                               \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_OneDnnGelu").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),     \
      OneDnnGeluOp<CPUDevice, TYPE>);                                       \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_OneDnnGeluGrad").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnGeluGradOp<CPUDevice, TYPE>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL)
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

template <typename Device, typename T>
class OneDnnSwishOp : public OneDnnEltwiseBaseOp<Device, T> {
 public:
  explicit OneDnnSwishOp(OpKernelConstruction* context)
      : OneDnnEltwiseBaseOp<Device, T>(context, dnnl::algorithm::eltwise_swish,
                                       1.0f, 0.0f) {
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &this->alpha_));
  }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnSwish")                 \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<TYPE>("T")       \
                              .HostMemory("features_meta")     \
                              .HostMemory("activations_meta"), \
                          OneDnnSwishOp<GPUDevice, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#else
#define REGISTER_KERNEL(TYPE)                                            \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_OneDnnSwish").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnSwishOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

template <typename Device, typename T>
class OneDnnSwishGradOp : public OneDnnEltwiseGradBaseOp<Device, T> {
 public:
  explicit OneDnnSwishGradOp(OpKernelConstruction* context)
      : OneDnnEltwiseGradBaseOp<Device, T>(
            context, dnnl::algorithm::eltwise_swish, 1.0f, 0.0f) {
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &this->alpha_));
  }

  int GetDiffDstIndex() const override { return 0; }
  int GetSrcIndex() const override { return 1; }
  int GetDiffSrcIndex() const override { return 0; }
  int GetTypeOfInputTensorFromFwdOp() const override { return DNNL_ARG_SRC; }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnSwishGrad")           \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<TYPE>("T")     \
                              .HostMemory("gradients_meta")  \
                              .HostMemory("features_meta")   \
                              .HostMemory("backprops_meta"), \
                          OneDnnSwishGradOp<GPUDevice, TYPE>)
TF_CALL_GPU_BACKWARD_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#else
#define REGISTER_KERNEL(TYPE)                                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_OneDnnSwishGrad").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnSwishGradOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

}  // namespace itex
