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

#include "itex/core/kernels/gpu/bucketize_op.h"

#include <algorithm>

#include "itex/core/kernels/gpu/gpu_device_array.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
struct BucketizeKernel {
  BucketizeKernel(GpuDeviceArrayStruct<float> boundaries_data, size_t num_elems,
                  size_t boundaries_size, const T* in, int32_t* out)
      : boundaries_data(boundaries_data),
        num_elems(num_elems),
        boundaries_size(boundaries_size),
        in(in),
        out(out) {}
  void operator()(sycl::nd_item<1> item) const {
    const float* boundaries = GetGpuDeviceArrayOnDevice(&boundaries_data);
    for (size_t id = item.get_global_linear_id(); id < num_elems;
         id += item.get_global_range(0)) {
      T value = in[id];
      int32 bucket = 0;
      int32 count = boundaries_size;
      while (count > 0) {
        int32 l = bucket;
        int32 step = count / 2;
        l += step;
        if (!(value < static_cast<T>(boundaries[l]))) {
          bucket = ++l;
          count -= step + 1;
        } else {
          count = step;
        }
      }
      out[id] = bucket;
    }
  }

 private:
  GpuDeviceArrayStruct<float> boundaries_data;
  size_t num_elems;
  size_t boundaries_size;
  const T* in;
  int32_t* out;
};

template <typename T>
struct BucketizeFunctor<GPUDevice, T> {
  // PRECONDITION: boundaries_vector must be sorted.
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& input,
                        const std::vector<float>& boundaries_vector,
                        const typename TTypes<int32, 1>::Tensor& output) {
    auto boundaries_size = boundaries_vector.size();
    GpuDeviceArrayOnHost<float> boundaries_array(context, boundaries_size);
    TF_RETURN_IF_ERROR(boundaries_array.Init());
    for (int i = 0; i < boundaries_vector.size(); ++i) {
      boundaries_array.Set(i, boundaries_vector[i]);
    }
    TF_RETURN_IF_ERROR(boundaries_array.Finalize());

    auto& stream = context->eigen_gpu_device().stream();

    const size_t num_elems = input.size();
    const size_t work_group_size =
        (stream->get_device())
            .template get_info<sycl::info::device::max_work_group_size>();
    const int hw_eu_count =
        (stream->get_device())
            .template get_info<sycl::ext::intel::info::device::gpu_eu_count>();
    const int hw_threads_per_eu =
        (stream->get_device())
            .template get_info<
                sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
    const int max_sub_group_size =
        (stream->get_device())
            .template get_info<sycl::info::device::sub_group_sizes>()
            .back();
    const size_t max_hw_workitem_count =
        hw_eu_count * hw_threads_per_eu * max_sub_group_size;
    const size_t workitem_count = std::min(max_hw_workitem_count, num_elems);
    size_t num_wg = (workitem_count + work_group_size - 1) / work_group_size;

    stream->submit([&](sycl::handler& cgh) {
      auto in = input.data();
      auto out = output.data();
      GpuDeviceArrayStruct<float> boundaries_data = boundaries_array.data();
      BucketizeKernel<T> kernel_functor(boundaries_data, num_elems,
                                        boundaries_size, in, out);
      cgh.parallel_for<BucketizeKernel<T> >(
          sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                            sycl::range<1>(work_group_size)),
          kernel_functor);
    });

    return Status::OK();
  }
};

}  // namespace functor

template <typename Device, typename T>
class BucketizeOp : public OpKernel {
 public:
  explicit BucketizeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("boundaries", &boundaries_));
    OP_REQUIRES(context, std::is_sorted(boundaries_.begin(), boundaries_.end()),
                errors::InvalidArgument("Expected sorted boundaries"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const auto input = input_tensor.flat<T>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();
    if (input.size() > 0) {
      OP_REQUIRES_OK(context, functor::BucketizeFunctor<Device, T>::Compute(
                                  context, input, boundaries_, output));
    }
  }

 private:
  std::vector<float> boundaries_;
};

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Bucketize").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BucketizeOp<GPUDevice, T>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(float);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL(double);
#endif
#undef REGISTER_KERNEL

}  // namespace itex
