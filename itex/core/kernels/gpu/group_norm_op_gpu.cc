/* Copyright (c) 2021-2023 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/group_norm_op_gpu.h"

#include "itex/core/kernels/gpu/group_norm_op.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T, typename U>
void ComputeMeanAndVar(OpKernelContext* context, const T* input, U* temp_mean,
                       U* temp_var, const InputShape& shape) {
  const auto& d = context->eigen_gpu_device();
  int num_to_reduce = shape.num_hw * shape.chans_per_group;
  bool use_one_kernel = (num_to_reduce < 64 * 1024);
  if (use_one_kernel) {
    if (shape.chans_per_group < 32) {
      impl::LaunchMeanAndVarKernel<16>(d, input, temp_mean, temp_var, shape);
    } else {
      impl::LaunchMeanAndVarKernel<32>(d, input, temp_mean, temp_var, shape);
    }

  } else {
    int scaled_hw = std::min(512, shape.num_hw);

    // allocate temporary for sum and square sum
    auto scratch_shape =
        TensorShape({2 * shape.num_batches * shape.num_groups * scaled_hw});
    Tensor temp_t;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::v(),
                                                   scratch_shape, &temp_t));
    U* temp_sum = temp_t.flat<U>().data();
    U* temp_sqr = temp_sum + shape.num_batches * shape.num_groups * scaled_hw;

    if (shape.chans_per_group < 32) {
      impl::LaunchPartialSumKernel<16>(d, input, temp_sum, temp_sqr, shape,
                                       scaled_hw);
    } else {
      impl::LaunchPartialSumKernel<32>(d, input, temp_sum, temp_sqr, shape,
                                       scaled_hw);
    }
    if (shape.chans_per_group < 32) {
      impl::LaunchMeanFromPartialKernel<16>(d, temp_sum, temp_sqr, temp_mean,
                                            temp_var, shape, scaled_hw);
    } else {
      impl::LaunchMeanFromPartialKernel<32>(d, temp_sum, temp_sqr, temp_mean,
                                            temp_var, shape, scaled_hw);
    }
  }
}

template <typename T, typename U>
void DoNormalization(OpKernelContext* context, const T* input, const T* gamma,
                     const T* beta, const U* temp_mean, const U* temp_var,
                     float epsilon, T* output, bool use_scale, bool use_center,
                     const InputShape& shape) {
  const auto& d = context->eigen_gpu_device();
  if (use_scale) {
    if (use_center) {
      impl::LaunchNormalizationKernel<true, true>(
          d, input, gamma, beta, temp_mean, temp_var, epsilon, output, shape);
    } else {
      impl::LaunchNormalizationKernel<true, false>(
          d, input, gamma, beta, temp_mean, temp_var, epsilon, output, shape);
    }
  } else {
    if (use_center) {
      impl::LaunchNormalizationKernel<false, true>(
          d, input, gamma, beta, temp_mean, temp_var, epsilon, output, shape);
    } else {
      impl::LaunchNormalizationKernel<false, false>(
          d, input, gamma, beta, temp_mean, temp_var, epsilon, output, shape);
    }
  }
}

}  // end namespace

namespace functor {

template <typename T>
struct GroupNormFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::Flat output,
                  typename TTypes<T>::ConstVec gamma,
                  typename TTypes<T>::ConstVec beta, float epsilon,
                  bool use_scale, bool use_center, const InputShape& shape) {
    // allocate temporary for mean and variance
    auto scratch_shape =
        TensorShape({shape.num_batches * shape.num_groups * 2});

    // using U: float as computation type
    using U = float;
    Tensor temp_t;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::v(),
                                                   scratch_shape, &temp_t));
    U* temp_mean = temp_t.flat<U>().data();
    U* temp_var = temp_mean + shape.num_batches * shape.num_groups;

    ComputeMeanAndVar(context, input.data(), temp_mean, temp_var, shape);
    DoNormalization(context, input.data(), gamma.data(), beta.data(), temp_mean,
                    temp_var, epsilon, output.data(), use_scale, use_center,
                    shape);
  }
};

}  // end namespace functor

// Instantiate the GPU implementation
#define DEFINE_GPU_KERNELS(T) \
  template struct functor::GroupNormFunctor<GPUDevice, T>;

DEFINE_GPU_KERNELS(float);
DEFINE_GPU_KERNELS(Eigen::half);
DEFINE_GPU_KERNELS(Eigen::bfloat16);
#undef DEFINE_GPU_KERNELS

}  // end namespace itex
