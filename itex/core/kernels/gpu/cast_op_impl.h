/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_GPU_CAST_OP_IMPL_H_
#define ITEX_CORE_KERNELS_GPU_CAST_OP_IMPL_H_

#include "itex/core/kernels/gpu/cast_op.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#ifdef ITEX_ENABLE_DOUBLE
#define CURRY_TYPES3_NO_HALF(FN, arg0, arg1) \
  FN(arg0, arg1, bool);                      \
  FN(arg0, arg1, uint8);                     \
  FN(arg0, arg1, uint16);                    \
  FN(arg0, arg1, uint32);                    \
  FN(arg0, arg1, uint64);                    \
  FN(arg0, arg1, int8);                      \
  FN(arg0, arg1, int16);                     \
  FN(arg0, arg1, int32);                     \
  FN(arg0, arg1, int64);                     \
  FN(arg0, arg1, float);                     \
  FN(arg0, arg1, double);                    \
  FN(arg0, arg1, std::complex<float>);       \
  FN(arg0, arg1, std::complex<double>);
#else
#define CURRY_TYPES3_NO_HALF(FN, arg0, arg1) \
  FN(arg0, arg1, bool);                      \
  FN(arg0, arg1, uint8);                     \
  FN(arg0, arg1, uint16);                    \
  FN(arg0, arg1, uint32);                    \
  FN(arg0, arg1, uint64);                    \
  FN(arg0, arg1, int8);                      \
  FN(arg0, arg1, int16);                     \
  FN(arg0, arg1, int32);                     \
  FN(arg0, arg1, int64);                     \
  FN(arg0, arg1, float);                     \
  FN(arg0, arg1, std::complex<float>);
#endif  // ITEX_ENABLE_DOUBLE

#define CURRY_TYPES3_NO_BF16(FN, arg0, arg1) \
  CURRY_TYPES3_NO_HALF(FN, arg0, arg1)       \
  FN(arg0, arg1, Eigen::half);

#define CURRY_TYPES3(FN, arg0, arg1)   \
  CURRY_TYPES3_NO_BF16(FN, arg0, arg1) \
  FN(arg0, arg1, Eigen::bfloat16);

#define CAST_CASE(DEVICE, IN, OUT)                                       \
  if (DataTypeToEnum<OUT>::value == dst_dtype) {                         \
    return [](OpKernelContext& context, const Tensor& inp, Tensor* out,  \
              bool truncate) {                                           \
      itex::functor::CastFunctor<DEVICE, OUT, IN> func;                  \
      func(context.eigen_gpu_device(), out->flat<OUT>(), inp.flat<IN>(), \
           truncate);                                                    \
    };                                                                   \
  }

namespace itex {

CastFunctorType GetGpuCastFromBool(DataType dst_dtype);

CastFunctorType GetGpuCastFromUint8(DataType dst_dtype);

CastFunctorType GetGpuCastFromUint16(DataType dst_dtype);

CastFunctorType GetGpuCastFromInt8(DataType dst_dtype);

CastFunctorType GetGpuCastFromUint32(DataType dst_dtype);

CastFunctorType GetGpuCastFromUint64(DataType dst_dtype);

CastFunctorType GetGpuCastFromInt16(DataType dst_dtype);

CastFunctorType GetGpuCastFromInt32(DataType dst_dtype);

CastFunctorType GetGpuCastFromInt64(DataType dst_dtype);

CastFunctorType GetGpuCastFromHalf(DataType dst_dtype);

CastFunctorType GetGpuCastFromFloat(DataType dst_dtype);

CastFunctorType GetGpuCastFromBfloat(DataType dst_dtype);

CastFunctorType GetGpuCastFromComplex64(DataType dst_dtype);

#ifdef ITEX_ENABLE_DOUBLE
CastFunctorType GetGpuCastFromDouble(DataType dst_dtype);

CastFunctorType GetGpuCastFromComplex128(DataType dst_dtype);
#endif  // ITEX_ENABLE_DOUBLE
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_CAST_OP_IMPL_H_
