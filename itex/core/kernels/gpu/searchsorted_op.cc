/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/searchsorted_op.h"

#include <limits>

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T, typename OutType>
struct UpperBoundKernel {
  UpperBoundKernel(size_t num_work_items, int num_inputs, int num_values,
                   const T* values_ptr, const T* inputs_ptr,
                   OutType* output_ptr)
      : num_work_items(num_work_items),
        num_inputs(num_inputs),
        num_values(num_values),
        values_ptr(values_ptr),
        inputs_ptr(inputs_ptr),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }
    auto batch_id = id / num_values;
    T val = values_ptr[id];
    auto start = inputs_ptr + batch_id * num_inputs;
    // TODO(itex): oneDPL is directly implementing
    // oneapi::dpl::upper_bound with std::upper_bound. Update it when
    // they have optimized implementation.
    output_ptr[id] = std::upper_bound(start, start + num_inputs, val) - start;
  }

 private:
  size_t num_work_items;
  int num_inputs;
  int num_values;
  const T* values_ptr;
  const T* inputs_ptr;
  OutType* output_ptr;
};

template <typename T, typename OutType>
struct UpperBoundFunctor<GPUDevice, T, OutType> {
  static Status Compute(OpKernelContext* ctx,
                        const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        int batch_size, int num_inputs, int num_values,
                        typename TTypes<OutType, 1>::Tensor* output) {
    if (values.size() == 0) return Status::OK();
    auto* stream = ctx->GetDeviceStream();
    auto work_group_size =
        stream->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = batch_size * num_values;
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
    if (sorted_inputs.size() == 0) {
      output->device(ctx->eigen_gpu_device()) = output->constant(OutType(0));
    } else {
      stream->submit([&](sycl::handler& cgh) {
        auto inputs_ptr = sorted_inputs.data();
        auto values_ptr = values.data();
        auto output_ptr = (*output).data();
        UpperBoundKernel<T, OutType> task(num_work_items, num_inputs,
                                          num_values, values_ptr, inputs_ptr,
                                          output_ptr);
        cgh.parallel_for<UpperBoundKernel<T, OutType>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                              sycl::range<1>(work_group_size)),
            task);
      });
    }

    return Status::OK();
  }
};

template <typename T, typename OutType>
struct LowerBoundKernel {
  LowerBoundKernel(size_t num_work_items, int num_inputs, int num_values,
                   const T* values_ptr, const T* inputs_ptr,
                   OutType* output_ptr)
      : num_work_items(num_work_items),
        num_inputs(num_inputs),
        num_values(num_values),
        values_ptr(values_ptr),
        inputs_ptr(inputs_ptr),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }
    auto batch_id = id / num_values;
    T val = values_ptr[id];
    auto start = inputs_ptr + batch_id * num_inputs;
    // TODO(itex): oneDPL is directly implementing
    // oneapi::dpl::lower_bound with std::lower_bound. Update it when
    // they have optimized implementation.
    output_ptr[id] = std::lower_bound(start, start + num_inputs, val) - start;
  }

 private:
  size_t num_work_items;
  int num_inputs;
  int num_values;
  const T* values_ptr;
  const T* inputs_ptr;
  OutType* output_ptr;
};

template <typename T, typename OutType>
struct LowerBoundFunctor<GPUDevice, T, OutType> {
  static Status Compute(OpKernelContext* ctx,
                        const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        int batch_size, int num_inputs, int num_values,
                        typename TTypes<OutType, 1>::Tensor* output) {
    if (values.size() == 0) return Status::OK();
    auto* stream = ctx->GetDeviceStream();
    auto work_group_size =
        stream->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = batch_size * num_values;
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
    if (sorted_inputs.size() == 0) {
      output->device(ctx->eigen_gpu_device()) = output->constant(OutType(0));
    } else {
      stream->submit([&](sycl::handler& cgh) {
        auto inputs_ptr = sorted_inputs.data();
        auto values_ptr = values.data();
        auto output_ptr = (*output).data();
        LowerBoundKernel<T, OutType> task(num_work_items, num_inputs,
                                          num_values, values_ptr, inputs_ptr,
                                          output_ptr);

        cgh.parallel_for<LowerBoundKernel<T, OutType>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                              sycl::range<1>(work_group_size)),
            task);
      });
    }

    return Status::OK();
  }
};

}  // namespace functor

template <typename Device, typename T, typename OutType>
class UpperBoundOp : public OpKernel {
 public:
  explicit UpperBoundOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& sorted_inputs_t = ctx->input(0);
    const Tensor& values_t = ctx->input(1);

    // must have same batch dim_size for both
    OP_REQUIRES(ctx, sorted_inputs_t.dim_size(0) == values_t.dim_size(0),
                Status(TF_Code::TF_INVALID_ARGUMENT,
                       "Leading dim_size of both tensors must match."));

    // this is required because we do indexing in int32 on the GPU
    OP_REQUIRES(ctx, values_t.NumElements() < std::numeric_limits<int>::max(),
                Status(TF_Code::TF_INVALID_ARGUMENT,
                       "values tensor size must less than INT_MAX"));

    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, values_t.shape(), &output_t));

    if (output_t->dtype() == DT_INT32) {
      OP_REQUIRES(ctx,
                  FastBoundsCheck(sorted_inputs_t.dim_size(1),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("trailing dim_size must less than "
                                          "INT_MAX for int32 output type, was ",
                                          sorted_inputs_t.dim_size(1)));
    }

    auto output = output_t->template flat<OutType>();
    const auto sorted_inputs = sorted_inputs_t.template flat<T>();
    const auto values = values_t.template flat<T>();
    OP_REQUIRES_OK(
        ctx, functor::UpperBoundFunctor<Device, T, OutType>::Compute(
                 ctx, sorted_inputs, values, sorted_inputs_t.dim_size(0),
                 sorted_inputs_t.dim_size(1), values_t.dim_size(1), &output));
  }
};

template <typename Device, typename T, typename OutType>
class LowerBoundOp : public OpKernel {
 public:
  explicit LowerBoundOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& sorted_inputs_t = ctx->input(0);
    const Tensor& values_t = ctx->input(1);

    // must have same batch dim_size for both
    OP_REQUIRES(ctx, sorted_inputs_t.dim_size(0) == values_t.dim_size(0),
                Status(TF_Code::TF_INVALID_ARGUMENT,
                       "Leading dim_size of both tensors must match."));

    // this is required because we do indexing in int32 on the GPU
    OP_REQUIRES(ctx, values_t.NumElements() < std::numeric_limits<int>::max(),
                Status(TF_Code::TF_INVALID_ARGUMENT,
                       "values tensor size must less than INT_MAX"));

    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, values_t.shape(), &output_t));

    if (output_t->dtype() == DT_INT32) {
      OP_REQUIRES(ctx,
                  FastBoundsCheck(sorted_inputs_t.dim_size(1),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("trailing dim_size must less than "
                                          "INT_MAX for int32 output type, was ",
                                          sorted_inputs_t.dim_size(1)));
    }

    auto output = output_t->template flat<OutType>();
    const auto sorted_inputs = sorted_inputs_t.template flat<T>();
    const auto values = values_t.template flat<T>();
    OP_REQUIRES_OK(
        ctx, functor::LowerBoundFunctor<Device, T, OutType>::Compute(
                 ctx, sorted_inputs, values, sorted_inputs_t.dim_size(0),
                 sorted_inputs_t.dim_size(1), values_t.dim_size(1), &output));
  }
};

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          UpperBoundOp<GPUDevice, type, int32>);  \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int64>("out_type"), \
                          UpperBoundOp<GPUDevice, type, int64>);  \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          LowerBoundOp<GPUDevice, type, int32>);  \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int64>("out_type"), \
                          LowerBoundOp<GPUDevice, type, int64>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNELS

}  // namespace itex
