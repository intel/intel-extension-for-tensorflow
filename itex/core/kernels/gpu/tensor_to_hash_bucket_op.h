/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_TENSOR_TO_HASH_BUCKET_OP_H_
#define ITEX_CORE_KERNELS_GPU_TENSOR_TO_HASH_BUCKET_OP_H_

#include <string>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/fingerprint.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/stringprintf.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace functor {

template <typename Device, typename T>
struct LaunchTensorToHashBucket {
  void operator()(OpKernelContext* c, const int64_t num_buckets, const T* input,
                  const int num_elems, int64_t* output) {
    string format = "%";
    switch (DataTypeToEnum<T>::value) {
      case DT_INT8:
      case DT_INT16:
      case DT_INT32:
        strings::Appendf(&format, "d");
        break;
      case DT_INT64:
        strings::Appendf(&format, "lld");
        break;
      default:
        bool type_not_supported = true;
        OP_REQUIRES(
            c, !type_not_supported,
            errors::InvalidArgument("Type not supported: ",
                                    DataTypeString(DataTypeToEnum<T>::value)));
    }

    for (int i = 0; i < num_elems; ++i) {
      string input_str = strings::Printf(format.c_str(), input[i]);
      const uint64 input_hash = Fingerprint64(input_str);
      const uint64 bucket_id = input_hash % num_buckets;
      // The number of buckets is always in the positive range of int64 so is
      // the resulting bucket_id. Casting the bucket_id from uint64 to int64 is
      // safe.
      output[i] = static_cast<int64_t>(bucket_id);
    }
  }
};

template <typename T>
struct LaunchTensorToHashBucket<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* c, const int64_t num_buckets, const T* input,
                  const int num_elems, int64_t* output);
};
}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_TENSOR_TO_HASH_BUCKET_OP_H_
