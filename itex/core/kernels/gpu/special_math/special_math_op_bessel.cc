/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/common/cwise_ops_common.h"
#include "itex/core/kernels/gpu/special_math/special_math_op_misc_impl.h"

namespace itex {

REGISTER3(UnaryOp, GPU, "BesselI0", functor::bessel_i0, Eigen::half, float,
          Eigen::bfloat16);
REGISTER3(UnaryOp, GPU, "BesselI1", functor::bessel_i1, Eigen::half, float,
          Eigen::bfloat16);
REGISTER3(UnaryOp, GPU, "BesselI0e", functor::bessel_i0e, Eigen::half, float,
          Eigen::bfloat16);
REGISTER3(UnaryOp, GPU, "BesselI1e", functor::bessel_i1e, Eigen::half, float,
          Eigen::bfloat16);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER(UnaryOp, GPU, "BesselI0", functor::bessel_i0, double);
REGISTER(UnaryOp, GPU, "BesselI1", functor::bessel_i1, double);
REGISTER(UnaryOp, GPU, "BesselI0e", functor::bessel_i0e, double);
REGISTER(UnaryOp, GPU, "BesselI1e", functor::bessel_i1e, double);
#endif

REGISTER3(UnaryOp, GPU, "BesselK0", functor::bessel_k0, Eigen::half, float,
          Eigen::bfloat16);
REGISTER3(UnaryOp, GPU, "BesselK1", functor::bessel_k1, Eigen::half, float,
          Eigen::bfloat16);
REGISTER3(UnaryOp, GPU, "BesselK0e", functor::bessel_k0e, Eigen::half, float,
          Eigen::bfloat16);
REGISTER3(UnaryOp, GPU, "BesselK1e", functor::bessel_k1e, Eigen::half, float,
          Eigen::bfloat16);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER(UnaryOp, GPU, "BesselK0", functor::bessel_k0, double);
REGISTER(UnaryOp, GPU, "BesselK1", functor::bessel_k1, double);
REGISTER(UnaryOp, GPU, "BesselK0e", functor::bessel_k0e, double);
REGISTER(UnaryOp, GPU, "BesselK1e", functor::bessel_k1e, double);
#endif

REGISTER3(UnaryOp, GPU, "BesselJ0", functor::bessel_j0, Eigen::half, float,
          Eigen::bfloat16);
REGISTER3(UnaryOp, GPU, "BesselJ1", functor::bessel_j1, Eigen::half, float,
          Eigen::bfloat16);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER(UnaryOp, GPU, "BesselJ0", functor::bessel_j0, double);
REGISTER(UnaryOp, GPU, "BesselJ1", functor::bessel_j1, double);
#endif

REGISTER3(UnaryOp, GPU, "BesselY0", functor::bessel_y0, Eigen::half, float,
          Eigen::bfloat16);
REGISTER3(UnaryOp, GPU, "BesselY1", functor::bessel_y1, Eigen::half, float,
          Eigen::bfloat16);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER(UnaryOp, GPU, "BesselY0", functor::bessel_y0, double);
REGISTER(UnaryOp, GPU, "BesselY1", functor::bessel_y1, double);
#endif

}  // namespace itex
