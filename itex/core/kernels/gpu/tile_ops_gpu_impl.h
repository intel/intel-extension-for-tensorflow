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

#ifndef ITEX_CORE_KERNELS_GPU_TILE_OPS_GPU_IMPL_H_
#define ITEX_CORE_KERNELS_GPU_TILE_OPS_GPU_IMPL_H_

#include <cstdio>

#include "itex/core/kernels/gpu/tile_ops_impl.h"
#include "itex/core/utils/numeric_types.h"

#define DEFINE_DIM(T, NDIM)                                       \
  namespace itex {                                                \
  namespace functor {                                             \
  template struct TileGrad<Eigen::GpuDevice, T, NDIM>;            \
  template struct ReduceAndReshape<Eigen::GpuDevice, T, NDIM, 1>; \
  }                                                               \
  }

#ifdef ITEX_ENABLE_DOUBLE
#define DEFINE_TILE_OPS(NDIM)       \
  DEFINE_DIM(int16, NDIM)           \
  DEFINE_DIM(int32, NDIM)           \
  DEFINE_DIM(int64, NDIM)           \
  DEFINE_DIM(Eigen::half, NDIM)     \
  DEFINE_DIM(Eigen::bfloat16, NDIM) \
  DEFINE_DIM(float, NDIM)           \
  DEFINE_DIM(complex64, NDIM)       \
  DEFINE_DIM(double, NDIM)          \
  DEFINE_DIM(complex128, NDIM)
#else
#define DEFINE_TILE_OPS(NDIM)       \
  DEFINE_DIM(int16, NDIM)           \
  DEFINE_DIM(int32, NDIM)           \
  DEFINE_DIM(int64, NDIM)           \
  DEFINE_DIM(Eigen::half, NDIM)     \
  DEFINE_DIM(Eigen::bfloat16, NDIM) \
  DEFINE_DIM(float, NDIM)           \
  DEFINE_DIM(complex64, NDIM)
#endif  // ITEX_ENABLE_DOUBLE

#endif  // ITEX_CORE_KERNELS_GPU_TILE_OPS_GPU_IMPL_H_
