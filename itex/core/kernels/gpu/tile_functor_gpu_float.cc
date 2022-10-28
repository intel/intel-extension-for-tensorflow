/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/tile_functor.h"
#include "itex/core/kernels/gpu/tile_functor_gpu.h"

namespace itex {
namespace functor {
using Eigen::GpuDevice;

template struct Tile<GpuDevice, float, int32>;
template struct Tile<GpuDevice, float, int64>;
}  // namespace functor
}  // namespace itex

// Put bfloat16 here for we can't add tile_functor_gpu_bfloat16.cu.cc
// to core/kernel/BUILD file. Because if_dpcpp() is not iterable
// for gpu_src. And if we add it to gpu_src directly. It may affect
// other gpu compiles.
namespace itex {
namespace functor {
using Eigen::GpuDevice;

template struct Tile<GpuDevice, Eigen::bfloat16, int32>;
template struct Tile<GpuDevice, Eigen::bfloat16, int64>;
}  // namespace functor
}  // namespace itex
