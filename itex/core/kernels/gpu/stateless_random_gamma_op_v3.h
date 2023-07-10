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

#ifndef ITEX_CORE_KERNELS_GPU_STATELESS_RANDOM_GAMMA_OP_V3_H_
#define ITEX_CORE_KERNELS_GPU_STATELESS_RANDOM_GAMMA_OP_V3_H_

#include "itex/core/kernels/common/random_ops_util.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/lib/random/philox_random.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {

namespace functor {

template <typename Device, typename T>
struct StatelessRandomGammaFunctor {
  static Status Fill(OpKernelContext* ctx, const T* alpha_flat,
                     int64 num_samples, int64 num_alphas,
                     int64 samples_per_alpha,
                     const random::PhiloxRandom& random, T* samples_flat);
};

}  // namespace functor

// Buffer that holds multiple samples. Operator()(random::PhiloxRandom*) returns
// a single sample from this buffer. If the buffer is empty, it first generates
// new samples using the provided distribution.
//
// If the call to Distribution::operator() returns samples[0...N-1], then this
// class returns samples in the following order:
//
//   samples[N-1], samples[N-2],..., samples[1], samples[0]
//
// For comparison, random::SingleSampleAdapter returns samples in
// the following order:
//
//   samples[0], samples[1],...,samples[N-2], samples[N-1].
//
template <class Distribution>
class RandomSampleBuffer {
 public:
  typedef typename Distribution::ResultElementType ResultElementType;

  PHILOX_DEVICE_INLINE
  explicit RandomSampleBuffer(Distribution* distribution)
      : distribution_(distribution), remaining_numbers_(0) {}

  PHILOX_DEVICE_INLINE
  ResultElementType operator()(random::PhiloxRandom* random) {
    if (remaining_numbers_ == 0) {
      results_ = (*distribution_)(random);
      remaining_numbers_ = Distribution::kResultElementCount;
    }

    remaining_numbers_--;
    return results_[remaining_numbers_];
  }

  // Mark this buffer as empty. The next call to operator() will fill it
  // with new random numbers.
  PHILOX_DEVICE_INLINE
  void Clear() { remaining_numbers_ = 0; }

 private:
  typedef typename Distribution::ResultType ResultType;

  Distribution* distribution_;
  ResultType results_;
  int remaining_numbers_;
};

template <typename T>
Status GetScalar(const Tensor& tensor, int input_idx, T* result) {
  auto dtype = DataTypeToEnum<T>::v();
  if (tensor.dims() != 0) {
    return errors::InvalidArgument("input ", std::to_string(input_idx),
                                   " (0-based) must have shape [], not ",
                                   tensor.shape().DebugString());
  }
  if (tensor.dtype() != dtype) {
    return errors::InvalidArgument("dtype of input ", std::to_string(input_idx),
                                   " (0-based) must be ", DataTypeString(dtype),
                                   ", not ", DataTypeString(tensor.dtype()));
  }
  *result = tensor.flat<T>()(0);
  return Status::OK();
}

inline Status CheckKeyCounterShape(Algorithm const& alg,
                                   TensorShape const& key_shape,
                                   TensorShape const& counter_shape) {
  if (!(key_shape.dims() == 1 && key_shape.dim_size(0) == RNG_KEY_SIZE)) {
    return errors::InvalidArgument(
        "key must have shape [", RNG_KEY_SIZE, "], not ",
        key_shape.DebugString(),
        ". (Note that batched keys are not supported yet.)");
  }
  auto counter_size = GetCounterSize(alg);
  if (!(counter_shape.dims() == 1 &&
        counter_shape.dim_size(0) >= counter_size)) {
    return errors::InvalidArgument(
        "counter must be a vector with length at least ", counter_size,
        "; got shape: ", counter_shape.DebugString(),
        ". (Note that batched counters are not supported yet.)");
  }
  return Status::OK();
}

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_STATELESS_RANDOM_GAMMA_OP_V3_H_
