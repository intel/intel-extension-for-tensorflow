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

#include "itex/core/kernels/gpu/cast_op_impl.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

CastFunctorType GetGpuCastFromBool(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, bool);
  return nullptr;
}

CastFunctorType GetGpuCastFromHalf(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, Eigen::half);
  return nullptr;
}

CastFunctorType GetGpuCastFromInt8(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, int8);
  return nullptr;
}

CastFunctorType GetGpuCastFromInt16(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, int16);
  return nullptr;
}

CastFunctorType GetGpuCastFromInt32(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, int32);
  return nullptr;
}

CastFunctorType GetGpuCastFromInt64(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, int64);
  return nullptr;
}

CastFunctorType GetGpuCastFromUint8(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, uint8);
  return nullptr;
}

CastFunctorType GetGpuCastFromUint16(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, uint16);
  return nullptr;
}

CastFunctorType GetGpuCastFromUint32(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, uint32);
  return nullptr;
}

CastFunctorType GetGpuCastFromUint64(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, uint64);
  return nullptr;
}

CastFunctorType GetGpuCastFromFloat(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, float);
  return nullptr;
}

CastFunctorType GetGpuCastFromBfloat(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, Eigen::bfloat16);
  return nullptr;
}

CastFunctorType GetGpuCastFromComplex64(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, std::complex<float>);
  return nullptr;
}

#ifdef ITEX_ENABLE_DOUBLE
CastFunctorType GetGpuCastFromDouble(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, double);
  return nullptr;
}

CastFunctorType GetGpuCastFromComplex128(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, std::complex<double>);
  return nullptr;
}
#endif  // ITEX_ENABLE_DOUBLE
}  // namespace itex
