/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_SNAPSHOT_OP_H_
#define ITEX_CORE_KERNELS_GPU_SNAPSHOT_OP_H_

#include "itex/core/utils/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

// Functor used by SnapshotOp.
template <typename Device, typename Scalar>
struct Snapshot {
  void operator()(const Device& device,
                  typename TTypes<Scalar>::ConstTensor input,
                  typename TTypes<Scalar>::Tensor output) {
    device.memcpy(output.data(), input.data(), input.size() * sizeof(Scalar));
  }
};

template <typename Scalar>
struct Snapshot<Eigen::GpuDevice, Scalar> {
  void operator()(const Eigen::GpuDevice& device,
                  typename TTypes<Scalar>::ConstTensor input,
                  typename TTypes<Scalar>::Tensor output) {
    // will support memcpy in eigen
    dpcppMemcpyDtoDAsync(output.data(), input.data(),
                         input.size() * sizeof(Scalar), device.stream());
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_SNAPSHOT_OP_H_
