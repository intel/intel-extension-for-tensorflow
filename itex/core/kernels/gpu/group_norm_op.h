/* Copyright (c) 2021-2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_GROUP_NORM_OP_H_
#define ITEX_CORE_KERNELS_GPU_GROUP_NORM_OP_H_

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

struct InputShape {
  int num_batches;
  int num_hw;
  int num_channels;
  int num_groups;
  int chans_per_group;
};

namespace functor {

template <typename Device, typename T>
struct GroupNormFunctor {
  void operator()(OpKernelContext* context, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::Flat output,
                  typename TTypes<float>::Flat reserve_space_1,
                  typename TTypes<float>::Flat reserve_space_2,
                  typename TTypes<T>::ConstVec gamma,
                  typename TTypes<T>::ConstVec beta, float epsilon,
                  bool use_scale, bool use_center, const InputShape& shape);
};

template <typename Device, typename T>
struct GroupNormGradFunctor {
  void operator()(OpKernelContext* context, typename TTypes<T>::ConstFlat x,
                  typename TTypes<float>::ConstFlat mean,
                  typename TTypes<float>::ConstFlat var,
                  typename TTypes<T>::ConstVec gamma,
                  typename TTypes<T>::ConstFlat grad_y, float epsilon,
                  int group, int num_batches, int num_HW, int channel_per_group,
                  int channel, typename TTypes<T>::Flat dx,
                  typename TTypes<T>::Flat dscale,
                  typename TTypes<T>::Flat doffset);
};

}  // end namespace functor
}  // end namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_GROUP_NORM_OP_H_
