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

#ifndef ITEX_CORE_KERNELS_GPU_FUSED_RANDOM_OP_GPU_H_
#define ITEX_CORE_KERNELS_GPU_FUSED_RANDOM_OP_GPU_H_

#include "itex/core/kernels/common/random_ops_util.h"
#include "itex/core/utils/lib/random/philox_random.h"
#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {

template <typename T>
struct Identity {
  inline T operator()(T x, T i) const { return x; }
};

#define MAX_DIMS 12

template <typename Device, class Distribution,
          typename Op = Identity<typename Distribution::ResultElementType>>
struct FillPhiloxRandomFused;

typedef Eigen::GpuDevice GPUDevice;

template <class Distribution, typename Op>
struct FillPhiloxRandomFused<GPUDevice, Distribution, Op> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  random::PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64 size,
                  Distribution dist,
                  const typename Distribution::ResultElementType* compare_data,
                  const int* random_shape, const int* compare_shape, int dims,
                  const uint64* key = nullptr, const uint64* counter = nullptr,
                  Op op = Identity<typename Distribution::ResultElementType>());
};

template <class Distribution, bool VariableSamplesPerOutput,
          typename Op = Identity<typename Distribution::ResultElementType>>
struct FillPhiloxRandomFusedKernel;

template <class Distribution, typename Op>
struct FillPhiloxRandomFusedKernel<Distribution, false, Op> {
  typedef typename Distribution::ResultElementType T;

  FillPhiloxRandomFusedKernel(
      T* data, int64_t size, const random::PhiloxRandom& gen,
      const Distribution& dist, const T* compare_data, const int* random_shape,
      const int* compare_shape, int dims, const uint64* key,
      const uint64* counter,
      Op op = Identity<typename Distribution::ResultElementType>())
      : data_(data),
        size_(size),
        gen_(gen),
        dist_(dist),
        compare_data_(compare_data),
        random_shape_(random_shape),
        compare_shape_(compare_shape),
        dims_(dims),
        key_(key),
        counter_(counter),
        op_(op) {}
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
        data_[offset + i] = op_(samples[i], GetCompareData(offset + i));
      }
    } else {
      for (int i = 0; i < kGroupSize; ++i) {
        if (offset + i >= size_) {
          return;
        }
        data_[offset + i] = op_(samples[i], GetCompareData(offset + i));
      }
    }
  }

 private:
  inline T GetCompareData(int64_t offset) const {
    if (dims_ == 0) {
      return compare_data_[0];
    }
    return T(0);
  }

  T* data_;
  int64_t size_;
  random::PhiloxRandom gen_;
  Distribution dist_;
  const T* compare_data_;
  const int* random_shape_;
  const int* compare_shape_;
  int dims_;
  const uint64* key_;
  const uint64* counter_;
  Op op_;
};

template <class Distribution, typename Op>
struct FillPhiloxRandomFusedKernel<Distribution, true, Op> {
  typedef typename Distribution::ResultElementType T;
  FillPhiloxRandomFusedKernel(
      T* data, int64_t size, const random::PhiloxRandom& gen,
      const Distribution& dist, const T* compare_data, const int* random_shape,
      const int* compare_shape, int dims, const uint64* key,
      const uint64* counter,
      Op op = Identity<typename Distribution::ResultElementType>())
      : data_(data),
        size_(size),
        gen_(gen),
        dist_(dist),
        compare_data_(compare_data),
        random_shape_(random_shape),
        compare_shape_(compare_shape),
        dims_(dims),
        key_(key),
        counter_(counter),
        op_(op) {}

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
        data_[offset] = op_(samples[i], GetCompareData(offset));
        ++offset;
      }
      offset += (total_item_count - 1) * kGroupSize;
      group_index += total_item_count;
    }
  }

 private:
  inline T GetCompareData(int64_t offset) const {
    if (dims_ == 0) {
      return compare_data_[0];
    }
  }

  T* data_;
  int64_t size_;
  random::PhiloxRandom gen_;
  Distribution dist_;
  const T* compare_data_;
  const int* random_shape_;
  const int* compare_shape_;
  int dims_;
  const uint64* key_;
  const uint64* counter_;
  Op op_;
};

template <typename T, typename Op>
class FillRandomFusedKernel;

template <class Distribution, typename Op>
void FillPhiloxRandomFusedKernelLaunch(
    const int32 workgroup_size, const int32 num_workgroups, gpuStream_t stream,
    random::PhiloxRandom base_gen,
    typename Distribution::ResultElementType* data, int64 size,
    Distribution dist,
    const typename Distribution::ResultElementType* compare_data,
    const int* random_shape, const int* compare_shape, int dims,
    const uint64* key, const uint64* counter, Op op) {
  stream->submit([&](sycl::handler& cgh) {
    FillPhiloxRandomFusedKernel<Distribution,
                                Distribution::kVariableSamplesPerOutput, Op>
        task(data, size, base_gen, dist, compare_data, random_shape,
             compare_shape, dims, key, counter, op);
    cgh.parallel_for<FillRandomFusedKernel<Distribution, Op>>(
        sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                          sycl::range<1>(workgroup_size)),
        task);
  });
}

// Partial specialization for GPU
template <class Distribution, typename Op>
void FillPhiloxRandomFused<GPUDevice, Distribution, Op>::operator()(
    OpKernelContext* context, const GPUDevice& d, random::PhiloxRandom gen,
    typename Distribution::ResultElementType* data, int64 size,
    Distribution dist,
    const typename Distribution::ResultElementType* compare_data,
    const int* random_shape, const int* compare_shape, int dims,
    const uint64* key, const uint64* counter, Op op) {
  if (size == 0) return;
  auto* stream = context->GetDeviceStream();
  const int32 workgroup_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const int dist_group_size =
      Distribution::kResultElementCount * workgroup_size;
  const int32 num_workgroups = (size + dist_group_size - 1) / dist_group_size;
  FillPhiloxRandomFusedKernelLaunch<Distribution, Op>(
      workgroup_size, num_workgroups, d.stream(), gen, data, size, dist,
      compare_data, random_shape, compare_shape, dims, key, counter, op);
}

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FUSED_RANDOM_OP_GPU_H_
