/* Copyright (c) 2023 Intel Corporation

Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/sparse_reorder_op.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "itex/core/kernels/gpu/unique_op.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

using GPUDevice = Eigen::GpuDevice;

namespace {

struct IndicesFlattenKernel {
  IndicesFlattenKernel(const int64* indices, const int64 nnz, const int64* dims,
                       const int64 ndims, int64* flat_indices)
      : indices_(indices),
        nnz_(nnz),
        dims_(dims),
        ndims_(ndims),
        flat_indices_(flat_indices) {}

  void operator()(sycl::nd_item<1> item) const {
    eigen_assert(ndims_ >= 1);

    auto id = item.get_global_linear_id();
    if (id >= nnz_) return;

    int64 output_idx = indices_[id * ndims_ + ndims_ - 1];
    int64 strides = 1;
    for (int i = ndims_ - 2; i >= 0; i--) {
      strides *= dims_[i + 1];
      output_idx += indices_[id * ndims_ + i] * strides;
    }
    flat_indices_[id] = output_idx;
  }

 private:
  const int64* indices_;
  const int64 nnz_;
  const int64* dims_;
  const int64 ndims_;
  int64* flat_indices_;
};

template <typename T>
struct PermuteIndicesAndValuesKernel {
  PermuteIndicesAndValuesKernel(const int64* indices, const T* values,
                                const int64 nnz, const int64 ndims,
                                const int64* permutation,
                                int64* reordered_indices, T* reordered_values)
      : indices_(indices),
        values_(values),
        nnz_(nnz),
        ndims_(ndims),
        permutation_(permutation),
        reordered_indices_(reordered_indices),
        reordered_values_(reordered_values) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= nnz_) return;

    for (int i = 0; i < ndims_; i++) {
      reordered_indices_[id * ndims_ + i] =
          indices_[permutation_[id] * ndims_ + i];
    }
    reordered_values_[id] = values_[permutation_[id]];
  }

 private:
  const int64* indices_;
  const T* values_;
  const int64 nnz_;
  const int64 ndims_;
  const int64* permutation_;
  int64* reordered_indices_;
  T* reordered_values_;
};

}  // namespace

namespace functor {

template <typename T>
struct SparseReorderFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* c, const Tensor& input_ind,
                  const Tensor& input_val, const Tensor& input_shape_in) {
    const Eigen::GpuDevice& d = c->eigen_gpu_device();

    const int64 num_elems = input_ind.dims() > 0 ? input_ind.dim_size(0) : 1;
    const int64 num_dims = input_ind.dims() > 1 ? input_ind.dim_size(1) : 1;

    auto indices = input_ind.template flat<int64_t>().data();
    auto values = input_val.template flat<T>().data();
    auto dims = input_shape_in.template flat<int64_t>().data();

    if (num_elems == 0) {
      c->set_output(0, input_ind);
      c->set_output(1, input_val);
      return;
    }

    Tensor flat_indices_tensor;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT64, TensorShape({num_elems}),
                                       &flat_indices_tensor));
    auto flat_indices = flat_indices_tensor.template flat<int64_t>().data();

    auto stream = d.stream();
    auto wg_size =
        stream->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (num_elems + wg_size - 1) / wg_size;

    sycl::nd_range<1> kernel_range(num_wg * wg_size, wg_size);
    IndicesFlattenKernel flatten_kernel(indices, num_elems, dims, num_dims,
                                        flat_indices);
    stream->parallel_for<IndicesFlattenKernel>(kernel_range, flatten_kernel);

    Tensor permutation_tensor;
    OP_REQUIRES_OK(
        c, c->allocate_temp(DT_INT64, {num_elems}, &permutation_tensor));
    auto permutation_data = permutation_tensor.template flat<int64_t>().data();

    if (num_elems > std::numeric_limits<int32_t>::max()) {
      OP_REQUIRES(
          c, false,
          errors::InvalidArgument("Number of inputs exceeds max int32 limits, "
                                  "which is not supported on GPU currently."));
    }
    OP_REQUIRES_OK(
        c,
        ::itex::impl::DispatchRadixSort<int64_t, int64_t, /*KEYS_PER_ITEM=*/8,
                                        /*GROUP_SIZE=*/256,
                                        /*SUBGROUP_SIZE*/ 16>(
            c, static_cast<int32_t>(num_elems),
            /*keys_in = */ flat_indices,
            /*indices_in = */ static_cast<int64*>(nullptr),
            /*keys_out = */ static_cast<int64*>(nullptr),
            /*indices_out = */ permutation_data));

    // Free temporary tensor that is no longer needed.
    flat_indices_tensor = Tensor();
    flat_indices = nullptr;

    Tensor* reordered_ind_tensor = nullptr;
    Tensor* reordered_val_tensor = nullptr;
    OP_REQUIRES_OK(
        c, c->allocate_output(0, input_ind.shape(), &reordered_ind_tensor));
    OP_REQUIRES_OK(
        c, c->allocate_output(1, input_val.shape(), &reordered_val_tensor));
    auto reordered_ind_data =
        reordered_ind_tensor->template flat<int64_t>().data();
    auto reordered_val_data = reordered_val_tensor->template flat<T>().data();

    PermuteIndicesAndValuesKernel<T> permute_kernel(
        indices, values, num_elems, num_dims, permutation_data,
        reordered_ind_data, reordered_val_data);
    stream->parallel_for<PermuteIndicesAndValuesKernel<T>>(kernel_range,
                                                           permute_kernel);
  }
};

}  // namespace functor

template <typename Device, typename T>
class SparseReorderOp : public OpKernel {
 public:
  explicit SparseReorderOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_ind = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_ind.shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_ind.shape().DebugString()));

    const Tensor& input_val = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_val.shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_val.shape().DebugString()));

    const Tensor& input_shape_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape_in.shape().DebugString()));

    functor::SparseReorderFunctor<Device, T>()(context, input_ind, input_val,
                                               input_shape_in);
  }
};

#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseReorder").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SparseReorderOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS);
REGISTER_GPU_KERNELS(bool);
#undef REGISTER_GPU_KERNELS

}  // namespace itex
