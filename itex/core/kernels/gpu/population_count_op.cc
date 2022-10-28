/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/population_count_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct ComputePopulationCountKernel {
  ComputePopulationCountKernel(const T* input, uint8* output, int total_size)
      : input_(input), output_(output), total_size_(total_size) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_id()[0];
    if (id >= total_size_) return;
    output_[id] = sycl::popcount(*(input_ + id));
  }

 private:
  const T* input_;
  uint8* output_;
  const int total_size_;
};

template <typename T>
class PopulationCountKernel;

template <typename T>
struct PopulationCountDPCPP {
  void operator()(const GPUDevice& d, const T* input, uint8* output,
                  OpKernelContext* ctx, int elements) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (elements + total_threads - 1) / total_threads;
    stream->submit([&](sycl::handler& cgh) {
      ComputePopulationCountKernel<T> task(input, output, elements);
      cgh.parallel_for<PopulationCountKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};

}  // namespace functor

template <typename Device, typename T>
class PopulationCountOp : public OpKernel {
 public:
  explicit PopulationCountOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    const Device& device = ctx->template eigen_device<Device>();

    functor::PopulationCountDPCPP<T> popcnt;
    popcnt(device, input.flat<T>().data(), output->flat<uint8>().data(), ctx,
           input.NumElements());
  }
};

#define REGISTER_POPULATION_COUNT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("PopulationCount").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PopulationCountOp<GPUDevice, type>)

TF_CALL_uint8(REGISTER_POPULATION_COUNT);
TF_CALL_int8(REGISTER_POPULATION_COUNT);
TF_CALL_uint16(REGISTER_POPULATION_COUNT);
TF_CALL_int16(REGISTER_POPULATION_COUNT);
TF_CALL_int32(REGISTER_POPULATION_COUNT);
TF_CALL_int64(REGISTER_POPULATION_COUNT);

#undef REGISTER_POPULATION_COUNT

}  // namespace itex
