/* Copyright (c) 2023 Intel Corporation

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

// Exposes the family of BLAS routines as pre-canned high performance calls for
// use in conjunction with the StreamExecutor abstraction.
//
// Note that this interface is optionally supported by platforms; see
// StreamExecutor::SupportsBlas() for details.
//
// This abstraction makes it simple to entrain BLAS operations on GPU data into
// a Stream -- users typically will not use this API directly, but will use the
// Stream builder methods to entrain these operations "under the hood". For
// example:
//
//  DeviceMemory<float> x = stream_exec->AllocateArray<float>(1024);
//  DeviceMemory<float> y = stream_exec->AllocateArray<float>(1024);
//  // ... populate x and y ...
//  Stream stream{stream_exec};
//  stream
//    .Init()
//    .ThenBlasAxpy(1024, 5.5, x, 1, &y, 1);
//  SE_CHECK_OK(stream.BlockHostUntilDone());
//
// By using stream operations in this manner the user can easily intermix custom
// kernel launches (via StreamExecutor::ThenLaunch()) with these pre-canned BLAS
// routines.

#ifndef ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_BLAS_H_
#define ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_BLAS_H_

#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "itex/core/compiler/xla/stream_executor/data_type.h"
#include "itex/core/compiler/xla/stream_executor/device_memory.h"
#include "itex/core/compiler/xla/stream_executor/lib/array_slice.h"
#include "itex/core/compiler/xla/stream_executor/lib/statusor.h"
#include "itex/core/compiler/xla/stream_executor/platform/port.h"

namespace Eigen {
struct half;
}  // namespace Eigen

namespace stream_executor {

class Stream;
class ScratchAllocator;

template <typename ElemT>
class DeviceMemory;

template <typename ElemT>
class HostOrDeviceScalar;

template <typename T>
using DeviceMemorySlice = port::ArraySlice<DeviceMemory<T>*>;  // non-absl ok

namespace blas {

// Specifies whether the input matrix will be transposed or
// transposed+conjugated before any BLAS operations.
enum class Transpose { kNoTranspose, kTranspose, kConjugateTranspose };

// Returns a name for t.
std::string TransposeString(Transpose t);

// Specifies whether the upper or lower triangular part of a
// symmetric/Hermitian matrix is used.
enum class UpperLower { kUpper, kLower };

// Returns a name for ul.
std::string UpperLowerString(UpperLower ul);

// Specifies whether a matrix is unit triangular.
enum class Diagonal { kUnit, kNonUnit };

// Returns a name for d.
std::string DiagonalString(Diagonal d);

// Specifies whether a Hermitian matrix appears on the left or right in
// operation.
enum class Side { kLeft, kRight };

// Returns a name for s.
std::string SideString(Side s);

// Type with which intermediate computations of a blas routine are performed.
//
// Some blas calls can perform computations with a type that's different than
// the type of their inputs/outputs.  This lets you e.g. multiply two matrices
// of int8s using float32s to store the matmul's intermediate values.
enum class ComputationType {
  kF16,  // 16-bit floating-point
  kF32,  // 32-bit floating-point
  kF64,  // 64-bit floating-point
  kI32,  // 32-bit integer
  // The below values use float32 for accumulation, but allow the inputs and
  // outputs to be downcast to a lower precision:
  kF16AsF32,   // Allow downcast to F16 precision.
  kBF16AsF32,  // Allow downcast to BF16 precision.
  kTF32AsF32,  // Allow downcast to TF32 precision.
};

// Converts a ComputationType to a string.
std::string ComputationTypeString(ComputationType ty);

std::ostream& operator<<(std::ostream& os, ComputationType ty);

// Opaque identifier for an "algorithm" used by a blas routine.  This functions
// as a hint to the blas library.
typedef int64_t AlgorithmType;
constexpr AlgorithmType kDefaultAlgorithm = -1;
constexpr AlgorithmType kDefaultBlasGemm = -2;
constexpr AlgorithmType kDefaultBlasGemv = -3;
constexpr AlgorithmType kNoAlgorithm = -4;

// blas uses -1 to represent the default algorithm. This happens to match up
// with the CUBLAS_GEMM_DFALT constant, so cuda_blas.cc is using static_cast
// to convert from AlgorithmType to cublasGemmAlgo_t, and uses a static_assert
// to ensure that this assumption does not break.
// If another blas implementation uses a different value for the default
// algorithm, then it needs to convert kDefaultGemmAlgo to that value
// (e.g. via a function called ToWhateverGemmAlgo).
constexpr AlgorithmType kDefaultGemmAlgo = -1;

// Describes the result of a performance experiment, usually timing the speed of
// a particular AlgorithmType.
//
// If the call we were benchmarking failed (a common occurrence; not all
// algorithms are valid for all calls), is_valid() will be false.
class ProfileResult {
 public:
  bool is_valid() const { return is_valid_; }
  void set_is_valid(bool val) { is_valid_ = val; }
  AlgorithmType algorithm() const { return algorithm_; }
  void set_algorithm(AlgorithmType val) { algorithm_ = val; }
  float elapsed_time_in_ms() const { return elapsed_time_in_ms_; }
  void set_elapsed_time_in_ms(float val) { elapsed_time_in_ms_ = val; }

 private:
  bool is_valid_ = false;
  AlgorithmType algorithm_ = kDefaultAlgorithm;
  float elapsed_time_in_ms_ = std::numeric_limits<float>::max();
};

class AlgorithmConfig {
 public:
  AlgorithmConfig() : algorithm_(kDefaultAlgorithm) {}
  explicit AlgorithmConfig(AlgorithmType algorithm) : algorithm_(algorithm) {}
  AlgorithmType algorithm() const { return algorithm_; }
  void set_algorithm(AlgorithmType val) { algorithm_ = val; }
  bool operator==(const AlgorithmConfig& other) const {
    return this->algorithm_ == other.algorithm_;
  }
  bool operator!=(const AlgorithmConfig& other) const {
    return !(*this == other);
  }
  std::string ToString() const;

 private:
  AlgorithmType algorithm_;
};

// Opaque identifier specifying the precision to use in gemm calls.
typedef int64_t ComputePrecision;
constexpr ComputePrecision kDefaultComputePrecision = 0;

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  DeviceMemoryBase data;
  Transpose transpose;
  int64_t num_rows;
  int64_t num_cols;
  int64_t stride;

  int64_t reduced_dim() const {
    return transpose == Transpose::kTranspose ? num_rows : num_cols;
  }

  template <typename T>
  DeviceMemory<T> cast() const {
    return DeviceMemory<T>(data);
  }
};

}  // namespace blas
}  // namespace stream_executor

#endif  // ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_BLAS_H_
