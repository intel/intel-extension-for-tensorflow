/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/kernels/gpu/sparse_concat_op.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "itex/core/kernels/gpu/gpu_device_array.h"
#include "itex/core/kernels/gpu/unique_op.h"
#include "itex/core/utils/bits.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/overflow.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {

template <typename T>
struct SparseConcatKernel {
  SparseConcatKernel(int64 output_nnz, int rank, int concat_dim,
                     bool need_to_sort,
                     GpuDeviceArrayStruct<const int64*> ind_ptrs_data,
                     GpuDeviceArrayStruct<const T*> val_ptrs_data,
                     GpuDeviceArrayStruct<int64_t> nnz_scan_data,
                     GpuDeviceArrayStruct<int64_t> concat_size_scan_data,
                     GpuDeviceArrayStruct<int64_t> output_shape_data,
                     int64* output_inds, T* output_vals,
                     int64* output_flat_inds)
      : output_nnz_(output_nnz),
        rank_(rank),
        concat_dim_(concat_dim),
        need_to_sort_(need_to_sort),
        ind_ptrs_data_(ind_ptrs_data),
        val_ptrs_data_(val_ptrs_data),
        nnz_scan_data_(nnz_scan_data),
        concat_size_scan_data_(concat_size_scan_data),
        output_shape_data_(output_shape_data),
        output_inds_(output_inds),
        output_vals_(output_vals),
        output_flat_inds_(output_flat_inds) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= output_nnz_) return;

    const auto ind_ptrs = GetGpuDeviceArrayOnDevice(&ind_ptrs_data_);
    const auto val_ptrs = GetGpuDeviceArrayOnDevice(&val_ptrs_data_);
    const int64* nnz_scan = GetGpuDeviceArrayOnDevice(&nnz_scan_data_);
    const int64* concat_size_scan =
        GetGpuDeviceArrayOnDevice(&concat_size_scan_data_);
    const int64* output_shape = GetGpuDeviceArrayOnDevice(&output_shape_data_);
    const int64 num_inputs = ind_ptrs_data_.size;
    const int64 input_num =
        std::upper_bound(nnz_scan, nnz_scan + num_inputs, id) - nnz_scan - 1;
    const int64 input_nz = id - nnz_scan[input_num];
    const int64 ind_offset = concat_size_scan[input_num];
    if (!need_to_sort_) {
      output_vals_[id] = val_ptrs[input_num][input_nz];
    }
    int64 flat_ind = 0;
    for (int j = 0; j < rank_; ++j) {
      const int64 output_ind = ind_ptrs[input_num][input_nz * rank_ + j] +
                               (j == concat_dim_ ? ind_offset : 0);
      if (!need_to_sort_) {
        output_inds_[id * rank_ + j] = output_ind;
      } else {
        flat_ind = flat_ind * output_shape[j] + output_ind;
        output_flat_inds_[id] = flat_ind;
      }
    }
  }

 private:
  int64 output_nnz_;
  int rank_;
  int concat_dim_;
  bool need_to_sort_;
  GpuDeviceArrayStruct<const int64*> ind_ptrs_data_;
  GpuDeviceArrayStruct<const T*> val_ptrs_data_;
  GpuDeviceArrayStruct<int64_t> nnz_scan_data_;
  GpuDeviceArrayStruct<int64_t> concat_size_scan_data_;
  GpuDeviceArrayStruct<int64_t> output_shape_data_;
  int64* output_inds_;
  T* output_vals_;
  int64* output_flat_inds_;
};

template <typename T>
struct SparseConcatPermuteKernel {
  SparseConcatPermuteKernel(int64 output_nnz, int rank,
                            GpuDeviceArrayStruct<const T*> val_ptrs_data,
                            GpuDeviceArrayStruct<int64_t> nnz_scan_data,
                            GpuDeviceArrayStruct<int64_t> output_shape_data,
                            const int64* output_flat_inds,
                            const int64* permutation, int64* output_inds,
                            T* output_vals)
      : output_nnz_(output_nnz),
        rank_(rank),
        val_ptrs_data_(val_ptrs_data),
        nnz_scan_data_(nnz_scan_data),
        output_shape_data_(output_shape_data),
        output_flat_inds_(output_flat_inds),
        permutation_(permutation),
        output_inds_(output_inds),
        output_vals_(output_vals) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= output_nnz_) return;

    const auto val_ptrs = GetGpuDeviceArrayOnDevice(&val_ptrs_data_);
    const int64* nnz_scan = GetGpuDeviceArrayOnDevice(&nnz_scan_data_);
    const int64* output_shape = GetGpuDeviceArrayOnDevice(&output_shape_data_);
    const int64 num_inputs = val_ptrs_data_.size;

    const int64 permuted_nz = permutation_[id];
    const int64 input_num =
        std::upper_bound(nnz_scan, nnz_scan + num_inputs, permuted_nz) -
        nnz_scan - 1;
    const int64 input_nz = permuted_nz - nnz_scan[input_num];
    output_vals_[id] = val_ptrs[input_num][input_nz];
    int64 output_flat_ind = output_flat_inds_[permuted_nz];
    for (int j = rank_ - 1; j >= 0; --j) {
      const int64 output_dim_size = output_shape[j];
      output_inds_[id * rank_ + j] = output_flat_ind % output_dim_size;
      output_flat_ind /= output_dim_size;
    }
  }

 private:
  int64 output_nnz_;
  int rank_;
  GpuDeviceArrayStruct<const T*> val_ptrs_data_;
  GpuDeviceArrayStruct<int64_t> nnz_scan_data_;
  GpuDeviceArrayStruct<int64_t> output_shape_data_;
  const int64* output_flat_inds_;
  const int64* permutation_;
  int64* output_inds_;
  T* output_vals_;
};

}  // namespace

template <typename T>
struct SparseConcatFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const OpInputList& inds,
                  const OpInputList& vals, const OpInputList& shapes,
                  int concat_dim) {
    const int N = inds.size();
    const TensorShape input_shape0(shapes[0].vec<int64_t>());
    const int rank = input_shape0.dims();

    // The input non-zeros are assumed to be sorted by increasing dimension
    // number (i.e., row-major order), so if the concatenation is along the
    // first dimension then they remain in order and we can directly compute the
    // output indices and values. To concatenate along other dimensions, we
    // first compute the flattened (1D) row-major output indices, then sort
    // these to obtain the required permutation, and finally gather the permuted
    // input values.

    GpuDeviceArrayOnHost<const int64*> ind_ptrs(context, N);
    GpuDeviceArrayOnHost<const T*> val_ptrs(context, N);
    GpuDeviceArrayOnHost<int64_t> nnz_scan(context, N + 1);
    GpuDeviceArrayOnHost<int64_t> concat_size_scan(context, N + 1);
    OP_REQUIRES_OK(context, ind_ptrs.Init());
    OP_REQUIRES_OK(context, val_ptrs.Init());
    OP_REQUIRES_OK(context, nnz_scan.Init());
    OP_REQUIRES_OK(context, concat_size_scan.Init());
    int64 nnz_sum = 0;
    int64 concat_size_sum = 0;
    nnz_scan.Set(0, nnz_sum);
    concat_size_scan.Set(0, concat_size_sum);
    for (int i = 0; i < N; ++i) {
      ind_ptrs.Set(i, inds[i].matrix<int64_t>().data());
      val_ptrs.Set(i, vals[i].vec<T>().data());
      nnz_sum += inds[i].dim_size(0);
      nnz_scan.Set(i + 1, nnz_sum);
      const TensorShape current_shape(shapes[i].vec<int64_t>());
      concat_size_sum += current_shape.dim_size(concat_dim);
      concat_size_scan.Set(i + 1, concat_size_sum);
    }
    OP_REQUIRES_OK(context, ind_ptrs.Finalize());
    OP_REQUIRES_OK(context, val_ptrs.Finalize());
    OP_REQUIRES_OK(context, nnz_scan.Finalize());
    OP_REQUIRES_OK(context, concat_size_scan.Finalize());
    const int64 output_nnz = nnz_sum;
    const int64 output_concat_size = concat_size_sum;

    const bool need_to_sort = concat_dim != 0;

    GpuDeviceArrayOnHost<int64_t> output_shape(context, rank);
    int64 output_dense_elements;
    if (need_to_sort) {
      OP_REQUIRES_OK(context, output_shape.Init());
      output_dense_elements = 1;
      for (int j = 0; j < rank; ++j) {
        int64 output_dim_size =
            j == concat_dim ? output_concat_size : input_shape0.dim_size(j);
        output_shape.Set(j, output_dim_size);
        output_dense_elements *= output_dim_size;
      }
      OP_REQUIRES_OK(context, output_shape.Finalize());
    }

    int64* output_inds_ptr = nullptr;
    T* output_vals_ptr = nullptr;
    int64* output_flat_inds_ptr = nullptr;
    Tensor output_flat_inds;
    if (need_to_sort) {
      // SparseConcatKernel will (only) produce output_flat_inds.
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT64, TensorShape({output_nnz}),
                                            &output_flat_inds));
      output_flat_inds_ptr = output_flat_inds.vec<int64_t>().data();
    } else {
      OP_REQUIRES_OK(
          context, allocate_outputs(context, rank, output_nnz, &output_inds_ptr,
                                    &output_vals_ptr));
    }

    const GPUDevice& device = context->eigen_gpu_device();
    auto stream = device.stream();
    auto wg_size =
        stream->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_wg = (output_nnz + wg_size - 1) / wg_size;
    sycl::nd_range<1> kernel_range(num_wg * wg_size, wg_size);
    SparseConcatKernel<T> concat_kernel(
        output_nnz, rank, concat_dim, need_to_sort, ind_ptrs.data(),
        val_ptrs.data(), nnz_scan.data(), concat_size_scan.data(),
        (need_to_sort ? output_shape.data() : GpuDeviceArrayStruct<int64_t>()),
        output_inds_ptr, output_vals_ptr, output_flat_inds_ptr);
    stream->parallel_for<SparseConcatKernel<T>>(kernel_range, concat_kernel);

    if (!need_to_sort) return;

    OP_REQUIRES_OK(context,
                   allocate_outputs(context, rank, output_nnz, &output_inds_ptr,
                                    &output_vals_ptr));

    Tensor permutation;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({output_nnz}),
                                          &permutation));
    int64* permutation_ptr = permutation.vec<int64_t>().data();

    if (output_nnz > std::numeric_limits<int32_t>::max()) {
      OP_REQUIRES(
          context, false,
          errors::InvalidArgument("Number of inputs exceeds max int32 limits, "
                                  "which is not supported on GPU currently."));
    }

    OP_REQUIRES_OK(
        context,
        ::itex::impl::DispatchRadixSort<int64_t, int64_t, /*KEYS_PER_ITEM=*/8,
                                        /*GROUP_SIZE=*/256,
                                        /*SUBGROUP_SIZE*/ 16>(
            context, /*size=*/static_cast<int32_t>(output_nnz),
            /*keys_in=*/output_flat_inds_ptr,
            /*indices_in=*/static_cast<int64*>(nullptr),
            /*keys_out=*/static_cast<int64*>(nullptr),
            /*indices_out=*/permutation_ptr,
            /*num_bits=*/Log2Ceiling64(output_dense_elements)));

    SparseConcatPermuteKernel<T> permute_kernel(
        output_nnz, rank, val_ptrs.data(), nnz_scan.data(), output_shape.data(),
        output_flat_inds_ptr, permutation_ptr, output_inds_ptr,
        output_vals_ptr);
    stream->parallel_for<SparseConcatPermuteKernel<T>>(kernel_range,
                                                       permute_kernel);
  }

 private:
  Status allocate_outputs(OpKernelContext* context, int rank, int64 output_nnz,
                          int64** output_inds_ptr, T** output_vals_ptr) const {
    Tensor* output_inds = nullptr;
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({output_nnz, rank}), &output_inds));
    *output_inds_ptr = output_inds->matrix<int64_t>().data();
    Tensor* output_vals = nullptr;
    TF_RETURN_IF_ERROR(
        context->allocate_output(1, TensorShape({output_nnz}), &output_vals));
    *output_vals_ptr = output_vals->vec<T>().data();
    return OkStatus();
  }
};

#define DEFINE_SPARSE_CONCAT_FUNCTOR(T) \
  template struct SparseConcatFunctor<GPUDevice, T>;
TF_CALL_POD_TYPES(DEFINE_SPARSE_CONCAT_FUNCTOR);

#undef DEFINE_SPARSE_CONCAT_FUNCTOR

}  // namespace functor

template <typename Device, typename T>
class SparseConcatOp : public OpKernel {
 public:
  explicit SparseConcatOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("concat_dim", &concat_dim_attr_));
  }

  void Compute(OpKernelContext* context) override {
    // `indices`, `values`, and `shapes` all have N inputs(N >= 2)
    auto num_inputs = context->num_inputs();
    auto num_inputs_each = num_inputs / 3;

    OpInputList inds(context, 0, num_inputs_each);
    // OP_REQUIRES_OK(context, context->input_list("indices", &inds));
    const int N = inds.size();
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(inds[i].shape()),
                  errors::InvalidArgument(
                      "Input indices should be a matrix but received shape ",
                      inds[i].shape().DebugString(), " at position ", i));
    }

    OpInputList vals(context, num_inputs_each, num_inputs_each * 2);
    // OP_REQUIRES_OK(context, context->input_list("values", &vals));
    OP_REQUIRES(context, vals.size() == N,
                errors::InvalidArgument("Expected ", N, " input values, got ",
                                        vals.size()));
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(vals[i].shape()),
                  errors::InvalidArgument(
                      "Input values should be a vector but received shape ",
                      vals[i].shape().DebugString(), " at position ", i));
    }

    OpInputList shapes(context, num_inputs_each * 2, num_inputs);
    // OP_REQUIRES_OK(context, context->input_list("shapes", &shapes));
    OP_REQUIRES(context, shapes.size() == N,
                errors::InvalidArgument("Expected ", N, " input shapes, got ",
                                        shapes.size()));
    bool overflow_ocurred = false;
    for (int i = 0; i < N; i++) {
      int64_t new_num_elements = 1;
      OP_REQUIRES(context, TensorShapeUtils::IsVector(shapes[i].shape()),
                  errors::InvalidArgument(
                      "Input shapes should be a vector but received shape ",
                      shapes[i].shape().DebugString(), " at position ", i));
      auto input_shape_vector = shapes[i].vec<int64_t>();
      for (int j = 0; j < input_shape_vector.size(); j++) {
        new_num_elements =
            MultiplyWithoutOverflow(new_num_elements, input_shape_vector(j));
        if (new_num_elements < 0) {
          overflow_ocurred = true;
          break;
        }
      }

      if (overflow_ocurred) {
        break;
      }
    }

    OP_REQUIRES(
        context, !overflow_ocurred,
        errors::Internal("Encountered overflow from large input shape."));

    const TensorShape input_shape(shapes[0].vec<int64_t>());
    const int input_rank = input_shape.dims();
    const int concat_dim = (concat_dim_attr_ < 0)
                               ? input_rank + concat_dim_attr_
                               : concat_dim_attr_;
    OP_REQUIRES(context, concat_dim >= 0 && concat_dim < input_rank,
                errors::InvalidArgument("Concat dimension must be in range [",
                                        -input_rank, ", ", input_rank,
                                        "), got ", concat_dim_attr_));
    TensorShape output_shape = input_shape;
    for (int i = 1; i < N; ++i) {
      const TensorShape current_shape(shapes[i].vec<int64_t>());
      OP_REQUIRES(
          context, current_shape.dims() == input_rank,
          errors::InvalidArgument(
              "Ranks of all input tensors must match: expected ", input_rank,
              " but got ", current_shape.dims(), " at position ", i));
      for (int j = 0; j < input_rank; ++j) {
        if (j != concat_dim) {
          OP_REQUIRES(
              context, input_shape.dim_size(j) == current_shape.dim_size(j),
              errors::InvalidArgument(
                  "Input shapes must match: expected ", input_shape.dim_size(j),
                  " for dimension ", j, " but got ", current_shape.dim_size(j),
                  " at position ", i));
        } else {
          output_shape.set_dim(
              j, output_shape.dim_size(j) + current_shape.dim_size(j));
        }
      }
    }

    Tensor* output_shape_out = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(2, TensorShape({output_shape.dims()}),
                                          &output_shape_out));
    auto output_shape_t = output_shape_out->vec<int64_t>();
    for (int j = 0; j < output_shape.dims(); ++j) {
      output_shape_t(j) = output_shape.dim_size(j);
    }

    int64_t output_nnz = 0;
    for (int i = 0; i < N; ++i) {
      output_nnz += inds[i].dim_size(0);
    }
    if (output_nnz == 0) {
      Tensor* output_inds = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({0, input_rank}),
                                              &output_inds));
      Tensor* output_vals = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(1, TensorShape({0}), &output_vals));
      return;  // No work to do
    }

    functor::SparseConcatFunctor<Device, T>()(context, inds, vals, shapes,
                                              concat_dim);
  }

 private:
  int concat_dim_attr_;
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseConcat")            \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("shapes")       \
                              .HostMemory("output_shape") \
                              .TypeConstraint<type>("T"), \
                          SparseConcatOp<GPUDevice, type>)
TF_CALL_POD_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace itex
