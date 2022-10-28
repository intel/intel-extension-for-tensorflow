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

#include "itex/core/kernels/gpu/roll_op.h"

#include <algorithm>

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/common_shape_fns.h"
#include "itex/core/utils/gtl/array_slice.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/register_types_traits.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tshift, typename Taxis>
class RollOp : public OpKernel {
 public:
  explicit RollOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input = context->input(0);
    const Tensor& shift = context->input(1);
    const Tensor& axis = context->input(2);

    auto shift_flat = shift.flat<Tshift>();
    auto axis_flat = axis.flat<Taxis>();

    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(input.shape()),
                errors::InvalidArgument("input must be 1-D or higher"));
    OP_REQUIRES(context, shift.shape().dims() <= 1,
                errors::InvalidArgument(
                    "shift must be a scalar or a 1-D vector. Found: ",
                    shift.shape().DebugString()));
    OP_REQUIRES(context, axis.shape().dims() <= 1,
                errors::InvalidArgument(
                    "axis must be a scalar or a 1-D vector. Found: ",
                    axis.shape().DebugString()));
    OP_REQUIRES(
        context, shift.shape() == axis.shape(),
        errors::InvalidArgument("shift and axis must have the same size"));
    const int64 num_elements = input.NumElements();
    const int num_shifts = static_cast<int>(shift_flat.size());
    const int num_dims = input.dims();

    // if there are any duplicate axes, shift_mod_sum will have the
    // total modulo sum of shifts for each dimension
    gtl::InlinedVector<int32, 4> shift_mod_sum(num_dims, 0);
    for (int i = 0; i < num_shifts; i++) {
      int axis = axis_flat(i);
      if (axis < 0) {
        axis += num_dims;
      }
      OP_REQUIRES(context, FastBoundsCheck(axis, num_dims),
                  errors::InvalidArgument("axis ", axis, " is out of range"));
      const int ds = std::max<int>(static_cast<int>(input.dim_size(axis)), 1);
      const int sum = shift_mod_sum[axis] + static_cast<int>(shift_flat(i));
      // modulo that works with negatives: ((x % y) + y) % y
      shift_mod_sum[axis] = (sum % ds + ds) % ds;
    }
    // the size of each dimension
    gtl::InlinedVector<int32, 4> dim_size(num_dims);
    // threshold[i] is the index that the roll starts to wrap back to the front
    gtl::InlinedVector<int32, 4> threshold(num_dims);
    // dim_range is the number of indices over in the flattened tensor
    // you need to skip in order to make it over from one side of a dimension
    // to the other. Used to make the shifts wrap around after a threshold.
    gtl::InlinedVector<int64, 4> dim_range(num_dims);
    int64 dim_size_prod = 1;  // dimension size product
    // inner shift dimension (inner most shifted dimension)
    int64 isd = 0;
    for (int i = num_dims - 1; i >= 0; i--) {
      if (isd == 0 && shift_mod_sum[i] != 0) isd = i;
      const int ds = std::max<int>(static_cast<int>(input.dim_size(i)), 1);
      dim_size[i] = ds;
      threshold[i] = (ds - shift_mod_sum[i]) % ds;
      dim_size_prod *= static_cast<int64>(input.dim_size(i));
      dim_range[i] = dim_size_prod;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    auto input_flat = input.flat<T>().data();
    auto output_flat = output->flat<T>().data();

    functor::Roll<Device, T>()(context, num_elements, num_dims, dim_size,
                               input_flat, output_flat, threshold, dim_range,
                               isd);
  }
};

namespace functor {

template <typename T>
struct RollKernel {
  RollKernel(size_t num_work_items, int num_dims, const int64* dim_range_ptr,
             const int32* dim_size_ptr, const int32* threshold_ptr,
             const T* input, T* output)
      : num_work_items(num_work_items),
        num_dims(num_dims),
        dim_range_ptr(dim_range_ptr),
        dim_size_ptr(dim_size_ptr),
        threshold_ptr(threshold_ptr),
        input(input),
        output(output) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) {
      return;
    }

    int64 offset = 0;
    for (int i = 0; i < num_dims; i++) {
      const int64 stride = dim_range_ptr[i] / dim_size_ptr[i];
      const int shift = dim_size_ptr[i] - threshold_ptr[i];
      const int indx = (id / stride) % dim_size_ptr[i];
      const int shifted_indx = (indx + shift) % dim_size_ptr[i];
      offset += (shifted_indx - indx) * stride;
    }
    output[id + offset] = input[id];
  }

 private:
  size_t num_work_items;
  int num_dims;
  const int64* dim_range_ptr;
  const int32* dim_size_ptr;
  const int32* threshold_ptr;
  const T* input;
  T* output;
};

template <typename T>
struct Roll<GPUDevice, T> {
  void operator()(const OpKernelContext* context, const int64 num_elements,
                  const int num_dims, const gtl::ArraySlice<int32> dim_size,
                  const T* input, T* output,
                  const gtl::ArraySlice<int32> threshold,
                  const gtl::ArraySlice<int64> dim_range, const int64 isd) {
    if (!num_elements) return;
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    auto dim_bytes = sizeof(int32) * dim_size.size();
    auto dim_buf = d.allocate(dim_bytes);

    auto thres_bytes = sizeof(int32) * threshold.size();
    auto thres_buf = d.allocate(thres_bytes);

    auto range_bytes = sizeof(int64) * dim_range.size();
    auto range_buf = d.allocate(range_bytes);

    d.memcpyHostToDevice(dim_buf, dim_size.data(), dim_bytes);
    d.memcpyHostToDevice(thres_buf, threshold.data(), thres_bytes);
    d.memcpyHostToDevice(range_buf, dim_range.data(), range_bytes);

    auto stream = context->eigen_gpu_device().stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_work_items = num_elements;
    auto num_wg = (num_work_items + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      auto dim_size_ptr = reinterpret_cast<const int32*>(dim_buf);
      auto threshold_ptr = reinterpret_cast<const int32*>(thres_buf);
      auto dim_range_ptr = reinterpret_cast<const int64*>(range_buf);
      RollKernel<T> task(num_work_items, num_dims, dim_range_ptr, dim_size_ptr,
                         threshold_ptr, input, output);

      cgh.parallel_for<RollKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * work_group_size),
                            sycl::range<1>(work_group_size)),
          task);
    });

    d.deallocate(dim_buf);
    d.deallocate(thres_buf);
    d.deallocate(range_buf);
  }
};

}  // namespace functor

#define REGISTER_KERNEL(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int32>("Taxis")    \
                              .HostMemory("shift")               \
                              .HostMemory("axis"),               \
                          RollOp<GPUDevice, type, int32, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int32>("Taxis")    \
                              .HostMemory("shift")               \
                              .HostMemory("axis"),               \
                          RollOp<GPUDevice, type, int64, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int64>("Taxis")    \
                              .HostMemory("shift")               \
                              .HostMemory("axis"),               \
                          RollOp<GPUDevice, type, int32, int64>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int64>("Taxis")    \
                              .HostMemory("shift")               \
                              .HostMemory("axis"),               \
                          RollOp<GPUDevice, type, int64, int64>)

TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
TF_CALL_uint32(REGISTER_KERNEL);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
TF_CALL_complex64(REGISTER_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNEL);
TF_CALL_complex128(REGISTER_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNEL
}  // namespace itex
