/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_RANDOM_OP_GPU_H_
#define ITEX_CORE_KERNELS_GPU_RANDOM_OP_GPU_H_

#include "itex/core/kernels/common/random_ops_util.h"
#include "itex/core/utils/lib/random/philox_random.h"
#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

template <typename Device, class Distribution>
struct FillPhiloxRandom;

typedef Eigen::GpuDevice GPUDevice;

template <class Distribution>
struct FillPhiloxRandom<GPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  random::PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64 size,
                  Distribution dist, const uint64* key = nullptr,
                  const uint64* counter = nullptr);
};

template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomKernel;

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, false> {
  typedef typename Distribution::ResultElementType T;

  FillPhiloxRandomKernel(T* data, int64_t size, const random::PhiloxRandom& gen,
                         const Distribution& dist, const uint64* key,
                         const uint64* counter)
      : data_(data),
        size_(size),
        gen_(gen),
        dist_(dist),
        key_(key),
        counter_(counter) {}
  void operator()(sycl::nd_item<1> item) const {
    const int kGroupSize = Distribution::kResultElementCount;

    const size_t item_id = item.get_global_id(0);
    int32_t offset = item_id * kGroupSize;

    if (key_ != nullptr && counter_ != nullptr) {
      random::PhiloxRandom* tmp_gen_ptr =
          const_cast<random::PhiloxRandom*>(&gen_);
      *tmp_gen_ptr = GetPhiloxRandomFromCounterKeyMem(counter_, key_);
    }
    gen_.Skip(item_id);

    const typename Distribution::ResultType samples = dist_(&gen_);
    if (offset + kGroupSize <= size_) {
      for (int i = 0; i < kGroupSize; ++i) {
        data_[offset + i] = samples[i];
      }
    } else {
      for (int i = 0; i < kGroupSize; ++i) {
        if (offset + i >= size_) {
          return;
        }
        data_[offset + i] = samples[i];
      }
    }
  }

 private:
  T* data_;
  int64_t size_;
  random::PhiloxRandom gen_;
  Distribution dist_;
  const uint64* key_;
  const uint64* counter_;
};

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  FillPhiloxRandomKernel(T* data, int64_t size, const random::PhiloxRandom& gen,
                         const Distribution& dist, const uint64* key,
                         const uint64* counter)
      : data_(data),
        size_(size),
        gen_(gen),
        dist_(dist),
        key_(key),
        counter_(counter) {}

  void operator()(sycl::nd_item<1> item) const {
    using random::PhiloxRandom;
    using random::SingleSampleAdapter;

    const int kReservedSamplesPerOutput = 256;
    const int kGroupSize = Distribution::kResultElementCount;
    const int kGeneratorSkipPerOutputGroup = kGroupSize *
                                             kReservedSamplesPerOutput /
                                             PhiloxRandom::kResultElementCount;
    const size_t item_id = item.get_global_id(0);
    const int32_t total_item_count = item.get_global_range()[0];
    int64_t group_index = item_id;
    int64_t offset = group_index * kGroupSize;
    const int64_t size = size_;
    if (key_ != nullptr && counter_ != nullptr) {
      random::PhiloxRandom* tmp_gen_ptr =
          const_cast<random::PhiloxRandom*>(&gen_);
      *tmp_gen_ptr = GetPhiloxRandomFromCounterKeyMem(counter_, key_);
    }

    while (offset < size) {
      // Since each output takes a variable number of samples, we need to
      // realign the generator to the beginning for the current output group
      PhiloxRandom gen = gen_;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      const typename Distribution::ResultType samples = dist_(&single_samples);

      for (int i = 0; i < kGroupSize; ++i) {
        if (offset >= size) {
          return;
        }
        data_[offset] = samples[i];
        ++offset;
      }
      offset += (total_item_count - 1) * kGroupSize;
      group_index += total_item_count;
    }
  }

 private:
  T* data_;
  int64_t size_;
  random::PhiloxRandom gen_;
  Distribution dist_;
  const uint64* key_;
  const uint64* counter_;
};

template <typename T>
class FillRandomKernel;

template <class Distribution>
void FillPhiloxRandomKernelLaunch(
    const int32 workgroup_size, const int32 num_workgroups, gpuStream_t stream,
    random::PhiloxRandom base_gen,
    typename Distribution::ResultElementType* data, int64 size,
    Distribution dist, const uint64* key, const uint64* counter) {
  stream->submit([&](sycl::handler& cgh) {
    FillPhiloxRandomKernel<Distribution,
                           Distribution::kVariableSamplesPerOutput>
        task(data, size, base_gen, dist, key, counter);
    cgh.parallel_for<FillRandomKernel<Distribution> >(
        sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                          sycl::range<1>(workgroup_size)),
        task);
  });
}

// Partial specialization for GPU
template <class Distribution>
void FillPhiloxRandom<GPUDevice, Distribution>::operator()(
    OpKernelContext* context, const GPUDevice& d, random::PhiloxRandom gen,
    typename Distribution::ResultElementType* data, int64 size,
    Distribution dist, const uint64* key, const uint64* counter) {
  if (size == 0) return;
  auto* stream = context->GetDeviceStream();
  const int32 workgroup_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const int dist_group_size =
      Distribution::kResultElementCount * workgroup_size;
  const int32 num_workgroups = (size + dist_group_size - 1) / dist_group_size;
  FillPhiloxRandomKernelLaunch<Distribution>(workgroup_size, num_workgroups,
                                             d.stream(), gen, data, size, dist,
                                             key, counter);
}

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_RANDOM_OP_GPU_H_
