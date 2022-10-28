/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_IMAGE_CROP_AND_RESIZE_OP_H_
#define ITEX_CORE_KERNELS_GPU_IMAGE_CROP_AND_RESIZE_OP_H_

#include <string>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

template <typename Device, typename T>
struct CropAndResize {
  // We assume that the tensor sizes are correct.
  bool operator()(const OpKernelContext* context,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  const string& method_name, float extrapolation_value,
                  typename TTypes<float, 4>::Tensor crops);
};

template <typename Device, typename T>
struct CropAndResizeBackpropImage {
  // We assume that the tensor sizes are correct.
  bool operator()(const OpKernelContext* context,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<T, 4>::Tensor grads_image,
                  const string& method_name);
};

template <typename Device, typename T>
struct CropAndResizeBackpropBoxes {
  // We assume that the tensor sizes are correct.
  bool operator()(const Device& d, typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<float, 2>::Tensor grads_boxes);
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_IMAGE_CROP_AND_RESIZE_OP_H_
