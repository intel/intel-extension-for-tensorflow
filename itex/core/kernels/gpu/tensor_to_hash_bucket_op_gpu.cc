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

#include <farmhash.h>

#include "itex/core/kernels/gpu/tensor_to_hash_bucket_op.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/farmhash_gpu/src/farmhash_gpu.h"

namespace itex {

namespace {

// We set the buffer size to 20 as it is sufficient to cover the number of
// digits in any integer type.
constexpr int kSharedMemBufferSizePerThread = 20;

template <typename T>
inline void FillDigits(T val, int num_digits, int* i, char* buf) {
  // eigen_assert(num_digits <= kSharedMemBufferSizePerThread - (*i));

  int factor = (val < 0 ? -1 : 1);

  int num_digits_a = num_digits;
  do {
    int digit = static_cast<int>((val % 10) * factor);
    buf[(*i) + num_digits - 1] = digit + '0';
    val /= 10;
    num_digits--;
  } while (val != 0);

  (*i) += num_digits_a;
}

template <typename T>
inline int IntegerToString(T val, char* buf) {
  int num_digits = 0;
  T val_a = val;
  do {
    val_a = val_a / 10;
    num_digits++;
  } while (val_a != 0);

  int i = 0;
  if (val < 0) {
    buf[i++] = '-';
  }

  FillDigits(val, num_digits, &i, buf);

  return i;
}

template <typename T>
struct ComputeHashes {
  ComputeHashes(const T* vals, int vals_size, int64 num_buckets, int64* hashes,
                sycl::local_accessor<char, 1> local_acc)
      : vals(vals),
        vals_size(vals_size),
        num_buckets(num_buckets),
        hashes(hashes),
        local_acc(local_acc) {}
  void operator()(sycl::nd_item<1> item) const {
    auto gid = item.get_global_linear_id();
    if (gid >= vals_size) return;

    auto lid = item.get_local_linear_id();
    char* s = local_acc.get_pointer().get();
    int size =
        IntegerToString(vals[gid], s + lid * kSharedMemBufferSizePerThread);
    uint64_t a_hash = ::util_gpu::Fingerprint64(
        s + lid * kSharedMemBufferSizePerThread, size);
    int64 a_bucket = static_cast<int64_t>(a_hash % num_buckets);
    hashes[gid] = a_bucket;
  }
  const T* vals;
  int vals_size;
  int64 num_buckets;
  int64* hashes;
  sycl::local_accessor<char, 1> local_acc;
};

}  // end namespace

namespace functor {

template <typename T>
void LaunchTensorToHashBucket<Eigen::GpuDevice, T>::operator()(
    OpKernelContext* c, const int64 num_buckets, const T* input,
    const int num_elems, int64* output) {
  const Eigen::GpuDevice& d = c->eigen_gpu_device();
  auto* stream = d.stream();
  if (num_elems > 0) {
    constexpr size_t kThreadsLimitInBlock = 1024;

    const int smem_bytes_allowed =
        stream->get_device()
            .template get_info<sycl::info::device::local_mem_size>();

    auto smem_bytes_per_thread = kSharedMemBufferSizePerThread * sizeof(char);
    size_t thread_per_block = std::min(
        kThreadsLimitInBlock, smem_bytes_allowed / smem_bytes_per_thread);

    auto smem_bytes_per_block = thread_per_block * smem_bytes_per_thread;

    int num_wg = (num_elems + thread_per_block - 1) / thread_per_block;
    sycl::nd_range<1> range(num_wg * thread_per_block, thread_per_block);

    stream->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<char, 1> local_acc(smem_bytes_per_block, cgh);
      ComputeHashes<T> task(input, num_elems, num_buckets, output, local_acc);
      cgh.parallel_for<ComputeHashes<T>>(range, task);
    });
  }
}

}  // namespace functor

#define REGISTER_FUNCTORS(type) \
  template struct functor::LaunchTensorToHashBucket<Eigen::GpuDevice, type>;

TF_CALL_INTEGRAL_TYPES(REGISTER_FUNCTORS);

#undef REGISTER_FUNCTORS

}  // namespace itex
