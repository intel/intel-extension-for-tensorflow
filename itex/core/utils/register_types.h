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

#ifndef ITEX_CORE_UTILS_REGISTER_TYPES_H_
#define ITEX_CORE_UTILS_REGISTER_TYPES_H_

#include "itex/core/utils/numeric_types.h"
#include "itex/core/utils/types.h"

// When registering a CPU kernel, it can be used to set the priority attribute
// of the kernel. Priority allows to register duplicated kernel in same device,
// and the high priority kernel will always be called.
const int CPU_PRIORITY = 1;

// Two sets of macros:
// - TF_CALL_float, TF_CALL_half, etc. which call the given macro with
//   the type name as the only parameter - except on platforms for which
//   the type should not be included.
// - Macros to apply another macro to lists of supported types. These also call
//   into TF_CALL_float, TF_CALL_double, etc. so they filter by target platform
//   as well.
// If you change the lists of types, please also update the list in types.cc.
//
// See example uses of these macros in core/ops.
//
//
// Each of these TF_CALL_XXX_TYPES(m) macros invokes the macro "m" multiple
// times by passing each invocation a data type supported by TensorFlow.
//
// The different variations pass different subsets of the types.
// TF_CALL_ALL_TYPES(m) applied "m" to all types supported by TensorFlow.
// The set of types depends on the compilation platform.
//.
// This can be used to register a different template instantiation of
// an OpKernel for different signatures, e.g.:
/*
   #define REGISTER_PARTITION(type)                                      \
     REGISTER_KERNEL_BUILDER(                                            \
         Name("Partition").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
         PartitionOp<type>);
   TF_CALL_ALL_TYPES(REGISTER_PARTITION)
   #undef REGISTER_PARTITION
*/

// All types are supported, so all macros are invoked.
//
// Note: macros are defined in same order as types in types.proto, for
// readability.
#define TF_CALL_double(m) m(double)  // NOLINT(readability/casting)
#define TF_CALL_float(m) m(float)    // NOLINT(readability/casting)
#define TF_CALL_int32(m) m(::itex::int32)
#define TF_CALL_uint32(m) m(::itex::uint32)
#define TF_CALL_uint8(m) m(::itex::uint8)
#define TF_CALL_int16(m) m(::itex::int16)

#define TF_CALL_int8(m) m(::itex::int8)
#define TF_CALL_int64(m) m(::itex::int64)
#define TF_CALL_uint64(m) m(::itex::uint64)
#define TF_CALL_bool(m) m(bool)  // NOLINT(readability/casting)

#define TF_CALL_bfloat16(m) m(Eigen::bfloat16)

#define TF_CALL_uint16(m) m(::itex::uint16)
#define TF_CALL_half(m) m(Eigen::half)

#define TF_CALL_complex64(m) m(std::complex<float>)
#define TF_CALL_complex128(m) m(std::complex<double>)

#define TF_CALL_qint8(m) m(::itex::qint8)
#define TF_CALL_quint8(m) m(::itex::quint8)
#define TF_CALL_qint32(m) m(::itex::qint32)
#define TF_CALL_qint16(m) m(::itex::qint16)
#define TF_CALL_quint16(m) m(::itex::quint16)

// Defines for sets of types.
#define TF_CALL_INTEGRAL_TYPES(m)                                       \
  TF_CALL_uint64(m) TF_CALL_int64(m) TF_CALL_uint32(m) TF_CALL_int32(m) \
      TF_CALL_uint16(m) TF_CALL_int16(m) TF_CALL_uint8(m) TF_CALL_int8(m)

#define TF_CALL_FLOAT_TYPES(m) \
  TF_CALL_half(m) TF_CALL_bfloat16(m) TF_CALL_float(m) TF_CALL_double(m)

#define TF_CALL_REAL_NUMBER_TYPES(m) \
  TF_CALL_INTEGRAL_TYPES(m)          \
  TF_CALL_FLOAT_TYPES(m)

#define TF_CALL_COMPLEX_TYPES(m) TF_CALL_complex64(m) TF_CALL_complex128(m)

// Call "m" for all number types, including complex types
#define TF_CALL_NUMBER_TYPES(m) \
  TF_CALL_REAL_NUMBER_TYPES(m) TF_CALL_COMPLEX_TYPES(m)

#define TF_CALL_POD_TYPES(m) TF_CALL_NUMBER_TYPES(m) TF_CALL_bool(m)

// Call "m" on all number types supported on CPU.
#define TF_CALL_CPU_NUMBER_TYPES(m) \
  TF_CALL_float(m) TF_CALL_bfloat16(m) TF_CALL_half(m)

// Call "m" on all number types supported on CPU without half.
#define TF_CALL_CPU_NUMBER_TYPES_WITHOUT_HALF(m) \
  TF_CALL_float(m) TF_CALL_bfloat16(m)

// Call "m" on all number types supported on GPU.
#define TF_CALL_GPU_NUMBER_TYPES(m) \
  TF_CALL_half(m) TF_CALL_float(m) TF_CALL_bfloat16(m)

#define TF_CALL_GPU_BACKWARD_NUMBER_TYPES(m) \
  TF_CALL_bfloat16(m) TF_CALL_float(m)

// Call "m" on all types supported on GPU.
#define TF_CALL_GPU_ALL_TYPES(m) TF_CALL_GPU_NUMBER_TYPES(m) TF_CALL_bool(m)

#define TF_CALL_QUANTIZED_TYPES(m) \
  TF_CALL_qint8(m) TF_CALL_quint8(m) TF_CALL_qint32(m)

#endif  // ITEX_CORE_UTILS_REGISTER_TYPES_H_
