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

#include "itex/core/kernels/gpu/scatter_nd_op.h"

#include <algorithm>
#include <limits>

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/kernels/gpu/dense_update_functor.h"
#include "itex/core/kernels/gpu/inplace_ops_functor.h"
#include "itex/core/kernels/gpu/training_op_helpers.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/refcount.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "itex/core/utils/util.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T, scatter_nd_op::UpdateOp Op>
struct LeftUpdate {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(T* out, const T& val);
};

template <typename T>
struct LeftUpdate<T, scatter_nd_op::UpdateOp::ASSIGN> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(T* out, const T& val) {
    *out = val;
  }
};

// Maozhou: atomic_ref does NOT support half/bf16 type
template <typename T>
struct LeftUpdate<T, scatter_nd_op::UpdateOp::ADD> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(T* out, const T& val) {
    DpcppAtomicAdd(out, val);
  }
};

template <typename T>
struct LeftUpdate<T, scatter_nd_op::UpdateOp::SUB> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(T* out, const T& val) {
    DpcppAtomicSub(out, val);
  }
};

template <typename T>
struct LeftUpdate<T, scatter_nd_op::UpdateOp::MAX> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(T* out, const T& val) {
    DpcppAtomicMax(out, val);
  }
};

template <typename T>
struct LeftUpdate<T, scatter_nd_op::UpdateOp::MIN> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(T* out, const T& val) {
    DpcppAtomicMin(out, val);
  }
};

// Specializations for std::complex, updating real and imaginary part
// individually. Even though this is not an atomic op anymore, it is safe
// because there is only one type of op per kernel.
template <typename T>
struct LeftUpdate<std::complex<T>, scatter_nd_op::UpdateOp::ADD> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(
      std::complex<T>* out, const std::complex<T>& val) {
    T* ptr = reinterpret_cast<T*>(out);
    DpcppAtomicAdd(ptr, val.real());
    DpcppAtomicAdd(ptr + 1, val.imag());
  }
};

template <typename T>
struct LeftUpdate<std::complex<T>, scatter_nd_op::UpdateOp::SUB> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(
      std::complex<T>* out, const std::complex<T>& val) {
    T* ptr = reinterpret_cast<T*>(out);
    DpcppAtomicSub(ptr, val.real());
    DpcppAtomicSub(ptr + 1, val.imag());
  }
};
}  // namespace

namespace functor {

template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM,
          typename = void>
struct ScatterNdOpKernel {
  ScatterNdOpKernel(
      size_t batch_size, const Index* indices,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      Eigen::array<int64_t, IXDIM> batch_strides, Index slice_size,
      const T* updates, T* out)
      : batch_size(batch_size),
        indices(indices),
        output_shape_prefix(output_shape_prefix),
        batch_strides(batch_strides),
        slice_size(slice_size),
        updates(updates),
        out(out) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= batch_size) {
      return;
    }
    // TODO(itex): move out of kernel
    auto update = LeftUpdate<T, op>();
    Index i = 0;
    bool out_of_bounds = false;
#pragma unroll
    for (int dim = 0; dim < IXDIM; ++dim) {
      int offset = (IXDIM * index + dim);
      const Index ix_d = internal::SubtleMustCopy(indices[offset]);
      out_of_bounds |= !FastBoundsCheck(ix_d, output_shape_prefix[dim]);
      i += ix_d * batch_strides[dim] * slice_size;
    }
    if (!out_of_bounds) {
      for (int si = 0; si < slice_size; ++si) {
        update(out + i + si, updates[index * slice_size + si]);
      }
    }
  }

 private:
  size_t batch_size;
  const Index* indices;
  const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix;
  Eigen::array<int64_t, IXDIM> batch_strides;
  Index slice_size;
  const T* updates;
  T* out;
};

template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
struct ScatterNdOpKernel<
    T, Index, op, IXDIM,
    typename std::enable_if_t<(std::is_same_v<T, Eigen::half> ||
                               std::is_same_v<T, Eigen::bfloat16>),
                              void>> {
  ScatterNdOpKernel(
      size_t batch_size, const Index* indices,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      Eigen::array<int64_t, IXDIM> batch_strides, Index slice_size,
      const T* updates, float* out_fp32)
      : batch_size(batch_size),
        indices(indices),
        output_shape_prefix(output_shape_prefix),
        batch_strides(batch_strides),
        slice_size(slice_size),
        updates(updates),
        out_fp32(out_fp32) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= batch_size) {
      return;
    }
    // TODO(itex): move out of kernel
    auto update = LeftUpdate<float, op>();
    Index i = 0;
    bool out_of_bounds = false;
#pragma unroll
    for (int dim = 0; dim < IXDIM; ++dim) {
      int offset = (IXDIM * index + dim);
      const Index ix_d = internal::SubtleMustCopy(indices[offset]);
      out_of_bounds |= !FastBoundsCheck(ix_d, output_shape_prefix[dim]);
      i += ix_d * batch_strides[dim] * slice_size;
    }
    if (!out_of_bounds) {
      for (int si = 0; si < slice_size; ++si) {
        update(out_fp32 + i + si,
               static_cast<float>(updates[index * slice_size + si]));
      }
    }
  }

 private:
  size_t batch_size;
  const Index* indices;
  const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix;
  Eigen::array<int64_t, IXDIM> batch_strides;
  Index slice_size;
  const T* updates;
  float* out_fp32;
};

// Functor used by ScatterOp to do the computations.
template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
struct ScatterNdFunctor<
    GPUDevice, T, Index, op, IXDIM,
    typename std::enable_if<!std::is_same<T, Eigen::bfloat16>::value &&
                            !std::is_same<T, Eigen::half>::value>::type> {
  Index operator()(
      const GPUDevice& d, const Index slice_size,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      typename TTypes<T, 2>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, 2>::Tensor Toutput,
      typename TTypes<float, 2>::Tensor Toutput_fp32) {
    // TODO(itex): The performance of this for small indices (large slices)
    // is poor. Write a kernel whose splitting is independent of the slice size.
    // See the gather_nd kernel for an example.
    const Eigen::DenseIndex batch_size = Tindices.dimension(0);
    // Maozhou: pybind11 crash if submit kernel when batch_size == 0
    if (batch_size == 0) {
      return -1;
    }

    // Index batch_strides[IXDIM];
    Eigen::array<int64, IXDIM> batch_strides;
    for (int dim = IXDIM - 1; dim >= 0; --dim) {
      if (dim == IXDIM - 1) {
        batch_strides[dim] = 1;
      } else {
        batch_strides[dim] =
            batch_strides[dim + 1] * output_shape_prefix[dim + 1];
      }
    }

    auto stream = d.stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .get_info<sycl::info::device::max_work_group_size>();
    auto num_work_groups = (batch_size + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      auto indices = Tindices.data();
      auto updates = Tupdates.data();
      auto out = Toutput.data();

      sycl::nd_range<1> kernel_range(
          sycl::range<1>(work_group_size * num_work_groups),
          sycl::range<1>(work_group_size));
      ScatterNdOpKernel<T, Index, op, IXDIM> task(
          batch_size, indices, output_shape_prefix, batch_strides, slice_size,
          updates, out);
      cgh.parallel_for<ScatterNdOpKernel<T, Index, op, IXDIM>>(kernel_range,
                                                               task);
    });

    return -1;
  }
};

template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
struct ScatterNdFunctor<
    GPUDevice, T, Index, op, IXDIM,
    typename std::enable_if<std::is_same<T, Eigen::bfloat16>::value ||
                            std::is_same<T, Eigen::half>::value>::type> {
  Index operator()(
      const GPUDevice& d, const Index slice_size,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      typename TTypes<T, 2>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, 2>::Tensor Toutput,
      typename TTypes<float, 2>::Tensor Toutput_fp32) {
    // TODO(itex): The performance of this for small indices (large slices)
    // is poor. Write a kernel whose splitting is independent of the slice size.
    // See the gather_nd kernel for an example.
    const Eigen::DenseIndex batch_size = Tindices.dimension(0);
    // Maozhou: pybind11 crash if submit kernel when batch_size == 0
    if (batch_size == 0) {
      return -1;
    }

    // Index batch_strides[IXDIM];
    Eigen::array<int64, IXDIM> batch_strides;
    for (int dim = IXDIM - 1; dim >= 0; --dim) {
      if (dim == IXDIM - 1) {
        batch_strides[dim] = 1;
      } else {
        batch_strides[dim] =
            batch_strides[dim + 1] * output_shape_prefix[dim + 1];
      }
    }

    auto stream = d.stream();
    auto work_group_size =
        (*stream)
            .get_device()
            .get_info<sycl::info::device::max_work_group_size>();
    auto num_work_groups = (batch_size + work_group_size - 1) / work_group_size;
    stream->submit([&](sycl::handler& cgh) {
      auto indices = Tindices.data();
      auto updates = Tupdates.data();
      auto out_fp32 = Toutput_fp32.data();

      sycl::nd_range<1> kernel_range(
          sycl::range<1>(work_group_size * num_work_groups),
          sycl::range<1>(work_group_size));
      ScatterNdOpKernel<T, Index, op, IXDIM> task(
          batch_size, indices, output_shape_prefix, batch_strides, slice_size,
          updates, out_fp32);
      cgh.parallel_for<ScatterNdOpKernel<T, Index, op, IXDIM>>(kernel_range,
                                                               task);
    });
    const Eigen::DenseIndex sizes = Toutput.size();
    ConvertFromFp32<GPUDevice, T>(d, sizes, Toutput_fp32.data(),
                                  Toutput.data());
    return -1;
  }
};
}  // namespace functor

#define DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, IXDIM) \
  template struct functor::ScatterNdFunctor<GPUDevice, T, Index, op, IXDIM>;

#define DECLARE_GPU_SPECS_INDEX_OP(T, Index, op)     \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 1); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 2); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 3); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 4); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 5); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 6); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 7);

#define DECLARE_GPU_SPECS_INDEX(T, Index)                                \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ASSIGN); \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ADD);    \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::SUB);

#define DECLARE_GPU_SPECS_INDEX_MINMAX(T, Index)                     \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::MAX) \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::MIN);

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

#define DECLARE_GPU_SPECS_MINMAX(T)         \
  DECLARE_GPU_SPECS_INDEX_MINMAX(T, int32); \
  DECLARE_GPU_SPECS_INDEX_MINMAX(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS_MINMAX);
TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int32(DECLARE_GPU_SPECS_MINMAX);
TF_CALL_int64(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS_MINMAX);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_MINMAX
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_INDEX_MINMAX
#undef DECLARE_GPU_SPECS_INDEX_OP

namespace functor {

// Returns true if the three tensors have valid number of elements
// If shape_input has 0 elements, then we need to have indices and updates with
// exactly 0 elements too, otherwise we should error. If indices has 0 elements
// then updates should also have 0 elements, otherwise we should error.
bool ValidEmptyOutputShape(int64 num_inputs, int64 num_indices,
                           int64 num_updates) {
  if (num_indices == 0 && num_updates == 0) {
    return true;  // regardless of num_inputs ?= 0, covers both cases
  }
  // now we want all 3 tensors to have values
  return (num_inputs != 0 && num_indices != 0 && num_updates != 0);
}

// Check whether updates.shape = indices.shape[:batch_dim] +
// params_shape[slice_dim:]
Status ValidateUpdateShape(const TensorShape& params_shape,
                           const Tensor& indices, const Tensor& updates) {
  const int64 slice_dim =
      (indices.dims() > 1) ? indices.dim_size(indices.dims() - 1) : 1;
  const int64 batch_dim = (indices.dims() > 1) ? indices.dims() - 1 : 1;

  auto shape_err_prefix = [&]() {
    return errors::InvalidArgument(
        "Dimensions [0,", batch_dim,
        ") of indices[shape=", indices.shape().DebugString(),
        "] must match dimensions [0,", batch_dim,
        ") of updates[shape=", updates.shape().DebugString(), "]");
  };
  auto shape_err_suffix = [&]() {
    return errors::InvalidArgument(
        "Dimensions [", slice_dim, ",", params_shape.dims(),
        ") of input[shape=", params_shape.DebugString(),
        "] must match dimensions [", slice_dim, ",", updates.dims(),
        ") of updates[shape=", updates.shape().DebugString(), "]");
  };

  if (updates.dims() < batch_dim) return shape_err_prefix();
  if (params_shape.dims() < slice_dim + (updates.dims() - batch_dim)) {
    return shape_err_suffix();
  }
  if (updates.dims() != batch_dim + params_shape.dims() - slice_dim) {
    return shape_err_suffix();
  }
  for (int d = 0; d < batch_dim; ++d) {
    if (updates.dim_size(d) != indices.dim_size(d)) return shape_err_prefix();
  }
  for (int d = 0; d < updates.dims() - batch_dim; ++d) {
    if (updates.dim_size(d + batch_dim) !=
        params_shape.dim_size(d + slice_dim)) {
      return shape_err_suffix();
    }
  }
  return Status::OK();
}

template <typename Index>
Status PrepareAndValidateInputs(const TensorShape& params_shape,
                                const Tensor& indices, const Tensor& updates,
                                int64* slice_dim, Index* num_updates,
                                Index* slice_size) {
  const TensorShape& indices_shape(indices.shape());
  const TensorShape& updates_shape(updates.shape());

  if (!TensorShapeUtils::IsVectorOrHigher(params_shape)) {
    return errors::InvalidArgument("Output must be at least 1-D, ",
                                   "got shape: ", params_shape.DebugString());
  }

  if (!ValidEmptyOutputShape(params_shape.num_elements(),
                             indices_shape.num_elements(),
                             updates_shape.num_elements())) {
    return errors::InvalidArgument(
        "Indices and updates specified for empty output.  indices shape: ",
        indices.shape().DebugString());
  }

  if (updates.dim_size(0) != indices.dim_size(0)) {
    return errors::InvalidArgument(
        "Dimensions [0,1) of indices[shape=", indices_shape.DebugString(),
        "] = ", indices.dim_size(0), " must match dimensions [0,1) of updates[",
        "shape=", updates_shape.DebugString(), "] = ", updates.dim_size(0));
  }
  TF_RETURN_IF_ERROR(ValidateUpdateShape(params_shape, indices, updates));

  // Check that we have enough index space
  const int64 N_big = indices.NumElements();
  if (N_big > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("indices has too many elements for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", N_big, " > ",
                                   std::numeric_limits<Index>::max());
  }
  if (params_shape.dim_size(0) > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("params_shape[0] too large for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", params_shape.dim_size(0),
                                   " > ", std::numeric_limits<Index>::max());
  }

  // Calculate the number of dimensions in indices
  *slice_dim = (indices_shape.dims() > 1)
                   ? indices_shape.dim_size(indices_shape.dims() - 1)
                   : 1;

  // Calculate the number of elements that make up each slice of our updated
  // tensor. This allows us to work with flattened tensors and copy over whole
  // slices at a time.
  Index total_nd = params_shape.dims();

  int64 slice_size_big = 1;
  for (int64 i = *slice_dim; i < total_nd; ++i) {
    slice_size_big *= params_shape.dim_size(i);
  }

  if (slice_size_big > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument(
        "slice size is too large for indexing: ", slice_size_big, " > ",
        std::numeric_limits<Index>::max());
  }

  *slice_size = static_cast<Index>(slice_size_big);

  const int64 safe_slice_dim = (*slice_dim < 1) ? 1 : *slice_dim;
  *num_updates = indices_shape.num_elements() / safe_slice_dim;

  return Status::OK();
}

template <typename Device, typename Index>
class IndexFlattener {
 public:
  inline typename TTypes<Index, 2>::ConstTensor operator()(
      OpKernelContext*, const Tensor& indices) {
    return indices.flat_inner_dims<Index>();
  }
};

template <typename T>
class ScatterNdFill;

template <typename Device, typename T,
          bool IsBf16Half = std::is_same<T, Eigen::bfloat16>::value ||
                            std::is_same<T, Eigen::half>::value>
struct InitOutputTensor;

template <typename Device, typename T>
struct InitOutputTensor<Device, T, false> {
  void operator()(OpKernelContext* c, Tensor* out, Tensor* out_fp32,
                  bool allocate) {
    if (allocate) {
      functor::SetZeroFunctor<Device, T> fill;
      fill(c->eigen_device<Device>(), out->flat<T>());
    }
  }
};

template <typename Device, typename T>
struct InitOutputTensor<Device, T, true> {
  void operator()(OpKernelContext* c, Tensor* out, Tensor* out_fp32,
                  bool allocate) {
    if (allocate) {
      functor::SetZeroFunctor<Device, float> fill;
      fill(c->eigen_device<Device>(), out_fp32->flat<float>());
    } else {
      auto total_size = out->NumElements();
      ConvertToFp32<Device, T>(c->eigen_device<Device>(), total_size,
                               out->flat<T>().data(),
                               out_fp32->flat<float>().data());
    }
  }
};

template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp Op>
Status DoScatterNd(OpKernelContext* c, const Tensor& indices,
                   const Tensor& updates, const TensorShape& shape, Tensor* out,
                   bool allocate) {
  int64 slice_dim;
  Index num_updates;
  Index slice_size;
  TF_RETURN_IF_ERROR(PrepareAndValidateInputs<Index>(
      shape, indices, updates, &slice_dim, &num_updates, &slice_size));

  IndexFlattener<Device, Index> index_flattener;
  auto indices_flat = index_flattener(c, indices);
  auto updates_flat = updates.shaped<T, 2>({num_updates, slice_size});

  Tensor out_fp32;
  TF_RETURN_IF_ERROR(c->allocate_temp(
      DataTypeToEnum<float>::value,
      TensorShape({shape.num_elements() / slice_size, slice_size}), &out_fp32));
  auto output_fp32_matrix = out_fp32.shaped<float, 2>(
      {shape.num_elements() / slice_size, slice_size});

  if (allocate) {
    AllocatorAttributes alloc_attr;
    TF_RETURN_IF_ERROR(
        c->allocate_temp(DataTypeToEnum<T>::value, shape, out, alloc_attr));
  } else {
    ITEX_CHECK_NOTNULL(out);
  }

  if (shape.num_elements() == 0) {
    return Status::OK();
  }

  InitOutputTensor<Device, T> init_output;
  init_output(c, out, &out_fp32, allocate);

  auto output_matrix =
      out->shaped<T, 2>({shape.num_elements() / slice_size, slice_size});

  Index bad_i = -1;
  if (shape.num_elements() > 0) {
    switch (slice_dim) {
#define PARAMS_CASE(IXDIM)                                               \
  case IXDIM: {                                                          \
    typename Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix; \
    for (int i = 0; i < IXDIM; ++i) {                                    \
      output_shape_prefix[i] = shape.dim_size(i);                        \
    }                                                                    \
    functor::ScatterNdFunctor<Device, T, Index, Op, IXDIM> functor;      \
    bad_i = functor(c->eigen_device<Device>(), slice_size,               \
                    output_shape_prefix, output_matrix, indices_flat,    \
                    updates_flat, output_matrix, output_fp32_matrix);    \
  } break
      PARAMS_CASE(1);
      PARAMS_CASE(2);
      PARAMS_CASE(3);
      PARAMS_CASE(4);
      PARAMS_CASE(5);
      PARAMS_CASE(6);
      PARAMS_CASE(7);
#undef PARAMS_CASE
      default:
        return errors::InvalidArgument(
            "Only indices.shape[-1] values between 1 and 5 "
            "are currently supported.  Requested rank: ",
            slice_dim);
    }
  }
  if (bad_i >= 0) {
    auto slice_shape = indices.shape();
    slice_shape.RemoveLastDims(1);
    return errors::InvalidArgument(
        "indices", SliceDebugString(slice_shape, bad_i), " = [",
        absl::StrJoin(
            gtl::ArraySlice<Index>(&indices_flat(bad_i, 0), slice_dim), ", "),
        "] does not index into shape ", shape.DebugString());
  }
  return Status::OK();
}
}  // namespace functor

// Returns true if the three tensors have valid number of elements
// If shape_input has 0 elements, then we need to have indices and updates with
// exactly 0 elements too, otherwise we should error. If indices has 0 elements
// then updates should also have 0 elements, otherwise we should error.
bool ValidEmptyOutputShape(int64 num_inputs, int64 num_indices,
                           int64 num_updates) {
  if (num_indices == 0 && num_updates == 0) {
    return true;  // regardless of num_inputs ?= 0, covers both cases
  }
  // now we want all 3 tensors to have values
  return (num_inputs != 0 && num_indices != 0 && num_updates != 0);
}

template <typename Device, typename T, typename Index>
class ScatterNdOp : public OpKernel {
 public:
  explicit ScatterNdOp(OpKernelConstruction* c) : OpKernel(c) {
    // TODO(itex):
    //    const DataType dt = DataTypeToEnum<T>::v();
    //    const DataType index_t = DataTypeToEnum<Index>::v();
    //    OP_REQUIRES_OK(c, c->MatchSignature({index_t, dt, index_t}, {dt}));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& indices = c->input(0);
    const Tensor& updates = c->input(1);
    const Tensor& shape_input = c->input(2);

    OP_REQUIRES(c, indices.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Indices shape must have rank at least one. Found:",
                    indices.shape().DebugString()));
    OP_REQUIRES(c, updates.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Updates shape must have rank at least one. Found:",
                    updates.shape().DebugString()));

    auto vec = shape_input.flat<Index>();
    TensorShape shape;
    OP_REQUIRES_OK(c,
                   TensorShapeUtils::MakeShape(vec.data(), vec.size(), &shape));

    OP_REQUIRES(c,
                ValidEmptyOutputShape(shape_input.NumElements(),
                                      indices.shape().num_elements(),
                                      updates.shape().num_elements()),
                errors::InvalidArgument(
                    "Indices and updates specified for empty output shape"));

    const int64 outer_dims = indices.shape().dims() - 1;

    for (int i = 0; i < outer_dims; ++i) {
      OP_REQUIRES(
          c, indices.shape().dim_size(i) == updates.shape().dim_size(i),
          errors::InvalidArgument(
              "Dimensions [0,", outer_dims,
              ") of indices[shape=", indices.shape().DebugString(),
              "] must match dimensions [0,", outer_dims,
              ") of updates[shape=", updates.shape().DebugString(), "]"));
    }

    const int64 ix = indices.shape().dim_size(outer_dims);
    OP_REQUIRES(c, updates.shape().dims() - outer_dims == shape.dims() - ix,
                errors::InvalidArgument(
                    "Dimensions [", ix, ",", shape.dims(), ") of input[shape=",
                    shape.DebugString(), "] must match dimensions [",
                    outer_dims, ",", updates.shape().dims(),
                    ") of updates[shape=", updates.shape().DebugString(), "]"));

    for (int i = 0; i + outer_dims < updates.shape().dims(); ++i) {
      OP_REQUIRES(
          c, updates.shape().dim_size(i + outer_dims) == shape.dim_size(ix + i),
          errors::InvalidArgument("Dimensions [", ix, ",", shape.dims(),
                                  ") of input[shape=", shape.DebugString(),
                                  "] must match dimensions [", outer_dims, ",",
                                  updates.shape().dims(), ") of updates[shape=",
                                  updates.shape().DebugString(), "]"));
    }
    OP_REQUIRES(c, shape_input.dims() == 1,
                errors::InvalidArgument("Shape must be a vector"));

    Tensor out;
    OP_REQUIRES_OK(
        c, functor::DoScatterNd<Device, T, Index, scatter_nd_op::UpdateOp::ADD>(
               c, indices, updates, shape, &out, true /*allocate*/));
    c->set_output(0, out);
  }
};

#define REGISTER_SCATTER_ND_KERNEL_INDEX(type, index_type, dev, name) \
  REGISTER_KERNEL_BUILDER(Name(name)                                  \
                              .Device(DEVICE_##dev)                   \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("shape"),                   \
                          ScatterNdOp<dev##Device, type, index_type>)

#define REGISTER_SCATTER_ND_KERNEL(type, dev, name)         \
  REGISTER_SCATTER_ND_KERNEL_INDEX(type, int32, dev, name); \
  REGISTER_SCATTER_ND_KERNEL_INDEX(type, int64, dev, name)

#define REGISTER_SCATTER_ND(type, dev) \
  REGISTER_SCATTER_ND_KERNEL(type, dev, "ScatterNd");

#define REGISTER_SCATTER_ND_GPU(type) REGISTER_SCATTER_ND(type, GPU);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_GPU);
TF_CALL_int32(REGISTER_SCATTER_ND_GPU);
TF_CALL_int64(REGISTER_SCATTER_ND_GPU);
TF_CALL_complex64(REGISTER_SCATTER_ND_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_SCATTER_ND_GPU);
TF_CALL_complex128(REGISTER_SCATTER_ND_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_SCATTER_ND_GPU
#undef REGISTER_SCATTER_ND
#undef REGISTER_SCATTER_ND_KERNEL
#undef REGISTER_SCATTER_ND_KERNEL_INDEX

template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp op>
class TensorScatterOp : public OpKernel {
 public:
  explicit TensorScatterOp(OpKernelConstruction* c) : OpKernel(c) {
    // TODO(itex): data type check
    // const DataType dt = DataTypeToEnum<T>::v();
    // const DataType index_t = DataTypeToEnum<Index>::v();
    // OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t, dt}, {dt}));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& input = c->input(0);
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    OP_REQUIRES(c, indices.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Indices shape must have rank at least one. Found:",
                    indices.shape().DebugString()));
    OP_REQUIRES(c, updates.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Updates shape must have rank at least one. Found:",
                    updates.shape().DebugString()));

    TensorShape shape = input.shape();

    OP_REQUIRES(c,
                ValidEmptyOutputShape(shape.num_elements(),
                                      indices.shape().num_elements(),
                                      updates.shape().num_elements()),
                errors::InvalidArgument(
                    "Indices and updates specified for empty output shape"));

    const int64 outer_dims = indices.shape().dims() - 1;

    for (int i = 0; i < outer_dims; ++i) {
      OP_REQUIRES(c, indices.shape().dim_size(i) == updates.shape().dim_size(i),
                  errors::InvalidArgument(
                      "Outer dimensions of indices and update must match. "
                      "Indices shape: ",
                      indices.shape().DebugString(),
                      ", updates shape:", updates.shape().DebugString()));
    }

    const int64 ix = indices.shape().dim_size(outer_dims);
    OP_REQUIRES(
        c, updates.shape().dims() - outer_dims == shape.dims() - ix,
        errors::InvalidArgument("Inner dimensions of output shape must match "
                                "inner dimensions of updates shape. Output: ",
                                shape.DebugString(),
                                " updates: ", updates.shape().DebugString()));
    for (int i = 0; i + outer_dims < updates.shape().dims(); ++i) {
      OP_REQUIRES(
          c, updates.shape().dim_size(i + outer_dims) == shape.dim_size(ix + i),
          errors::InvalidArgument(
              "The inner ", shape.dims() - ix,
              " dimensions of output.shape=", shape.DebugString(),
              " must match the inner ", updates.shape().dims() - outer_dims,
              " dimensions of updates.shape=", updates.shape().DebugString()));
    }

    Tensor* out;
    // TODO(itex): Check whether at graph construction time this output was
    // marked either for no forwarding or with a reservation for this input?
    int forwarded_input = 0;
    OP_REQUIRES_OK(c, c->forward_input_or_allocate_output(
                          {0}, 0, input.shape(), &out, &forwarded_input));
    // does not forward successfully and allocates new output
    if (forwarded_input < 0) {
      OP_REQUIRES_OK(
          c, itex::functor::DoCopy(c->eigen_device<Device>(), input, out));
    }
    OP_REQUIRES_OK(c, functor::DoScatterNd<Device, T, Index, op>(
                          c, indices, updates, shape, out, false /*allocate*/));
  }
};

#define REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, index_type, \
                                                          dev)              \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterUpdate")                       \
                              .Device(DEVICE_##dev)                         \
                              .TypeConstraint<type>("T")                    \
                              .TypeConstraint<index_type>("Tindices"),      \
                          TensorScatterOp<dev##Device, type, index_type,    \
                                          scatter_nd_op::UpdateOp::ASSIGN>)

#define REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterAdd")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::ADD>)

#define REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterSub")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::SUB>)

#define REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterMin")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::MIN>)

#define REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE(type, index_type, dev) \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterMax")                            \
                              .Device(DEVICE_##dev)                           \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          TensorScatterOp<dev##Device, type, index_type,      \
                                          scatter_nd_op::UpdateOp::MAX>)

#define REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE(type, int64, GPU);

#define REGISTER_SCATTER_ND_TENSOR_ADD_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE(type, int64, GPU);

#define REGISTER_SCATTER_ND_TENSOR_SUB_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE(type, int64, GPU);

#define REGISTER_SCATTER_ND_TENSOR_MIN_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE(type, int64, GPU);

#define REGISTER_SCATTER_ND_TENSOR_MAX_GPU(type)                    \
  REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE(type, int32, GPU); \
  REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE(type, int64, GPU);

#define REGISTER_SCATTER_ND_TENSOR_ADD_SUB_GPU(type) \
  REGISTER_SCATTER_ND_TENSOR_ADD_GPU(type);          \
  REGISTER_SCATTER_ND_TENSOR_SUB_GPU(type);

#define REGISTER_SCATTER_ND_TENSOR_MIN_MAX_GPU(type) \
  REGISTER_SCATTER_ND_TENSOR_MIN_GPU(type);          \
  REGISTER_SCATTER_ND_TENSOR_MAX_GPU(type);

TF_CALL_int32(REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU);
TF_CALL_int32(REGISTER_SCATTER_ND_TENSOR_ADD_SUB_GPU);
TF_CALL_int32(REGISTER_SCATTER_ND_TENSOR_MIN_MAX_GPU);
TF_CALL_int64(REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU);
TF_CALL_int64(REGISTER_SCATTER_ND_TENSOR_ADD_SUB_GPU);
TF_CALL_int64(REGISTER_SCATTER_ND_TENSOR_MIN_MAX_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_TENSOR_ADD_SUB_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_TENSOR_MIN_MAX_GPU);
TF_CALL_complex64(REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU);
TF_CALL_complex64(REGISTER_SCATTER_ND_TENSOR_ADD_SUB_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU);
TF_CALL_double(REGISTER_SCATTER_ND_TENSOR_ADD_SUB_GPU);
TF_CALL_double(REGISTER_SCATTER_ND_TENSOR_MIN_MAX_GPU);
TF_CALL_complex128(REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU);
TF_CALL_complex128(REGISTER_SCATTER_ND_TENSOR_ADD_SUB_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_SCATTER_ND_TENSOR_UPDATE_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_ADD_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_SUB_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_MIN_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_MAX_TYPE_INDEX_TYPE
#undef REGISTER_SCATTER_ND_TENSOR_UPDATE_GPU
#undef REGISTER_SCATTER_ND_TENSOR_ADD_GPU
#undef REGISTER_SCATTER_ND_TENSOR_SUB_GPU
#undef REGISTER_SCATTER_ND_TENSOR_MIN_GPU
#undef REGISTER_SCATTER_ND_TENSOR_MAX_GPU
#undef REGISTER_SCATTER_ND_TENSOR_ADD_SUB_GPU
#undef REGISTER_SCATTER_ND_TENSOR_MIN_MAX_GPU

// TODO(itex): remove InputTensorType template argument when we have
// TF_InputIsRef C-API. Currently, we cannot distinguish normal tensor or ref
// tensor, during kernel execution. So we have to use InputTensorType to
// distinguish them during compiling.

enum class InputTensorType { ResourceTensor, RefTensor, NormalTensor };

template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp Op, InputTensorType TensorType>
class ScatterNdUpdateOp : public OpKernel {
 public:
  explicit ScatterNdUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
    // TODO(itex): data type check
    if (c->HasAttr("use_locking")) {
      OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
    }
  }

  void Compute(OpKernelContext* c) override {
    if (TensorType == InputTensorType::ResourceTensor || use_exclusive_lock_) {
      auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
          c, /* do_lock */ use_exclusive_lock_, /* sparse */ true, {0});
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  DataType dtype_;
  bool use_exclusive_lock_ = false;

  void DoCompute(OpKernelContext* c) {
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    Tensor params;
    TensorShape params_shape;

    if (TensorType == InputTensorType::ResourceTensor) {
      OP_REQUIRES_OK(c, GetInputTensorFromVariable<Device, T>(
                            c, 0, /* lock_held */ use_exclusive_lock_,
                            /* sparse */ true, &params));
      params_shape = params.shape();
    } else if (TensorType == InputTensorType::RefTensor) {
      params = c->mutable_input(0, use_exclusive_lock_);
      params_shape = params.shape();
      c->forward_ref_input_to_ref_output(0, 0);
      OP_REQUIRES(c, params.IsInitialized(),
                  errors::FailedPrecondition("Null ref for params"));
    } else {
      Tensor* params_ptr;
      params_shape = c->input(0).shape();
      if (!c->forward_input_or_allocate_output({0}, 0, params_shape,
                                               &params_ptr)
               .ok()) {
        // We weren't able to forward the input to output, so just
        // allocate a new output tensor and copy the values over.
        OP_REQUIRES_OK(c, c->allocate_output(0, params_shape, &params_ptr));
        params = *params_ptr;
        functor::DenseUpdate<Device, T, ASSIGN> copy;
        const Tensor& input_copy = c->input(0);
        copy(c->eigen_device<Device>(), params.flat<T>(), input_copy.flat<T>());
      } else {
        params = *params_ptr;
      }
    }

    OP_REQUIRES_OK(
        c, functor::DoScatterNd<Device, T, Index, Op>(
               c, indices, updates, params_shape, &params, false /*allocate*/));
  }
};

#define REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, index_type, dev, name,   \
                                                op)                            \
  REGISTER_KERNEL_BUILDER(Name(name)                                           \
                              .Device(DEVICE_##dev)                            \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ScatterNdUpdateOp<dev##Device, type, index_type, op, \
                                            InputTensorType::RefTensor>)

#define REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX(type, index_type,     \
                                                         dev, name, op)        \
  REGISTER_KERNEL_BUILDER(Name(name)                                           \
                              .Device(DEVICE_##dev)                            \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices")          \
                              .HostMemory("ref"),                              \
                          ScatterNdUpdateOp<dev##Device, type, index_type, op, \
                                            InputTensorType::ResourceTensor>)

#define REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int64, dev, name, op)

#define REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(type, dev, name, op)    \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int32, dev, name, \
                                                   op);                    \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL_INDEX(type, int64, dev, name, op)

#define REGISTER_SCATTER_ND_ADD_SUB(type, dev)                          \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdAdd",          \
                                    scatter_nd_op::UpdateOp::ADD);      \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdSub",          \
                                    scatter_nd_op::UpdateOp::SUB);      \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                           \
      type, dev, "ResourceScatterNdAdd", scatter_nd_op::UpdateOp::ADD); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                           \
      type, dev, "ResourceScatterNdSub", scatter_nd_op::UpdateOp::SUB);

#define REGISTER_SCATTER_ND_UPDATE(type, dev)                         \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdUpdate",     \
                                    scatter_nd_op::UpdateOp::ASSIGN); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                         \
      type, dev, "ResourceScatterNdUpdate", scatter_nd_op::UpdateOp::ASSIGN);

#define REGISTER_SCATTER_ND_MIN_MAX(type, dev)                          \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdMax",          \
                                    scatter_nd_op::UpdateOp::MAX);      \
  REGISTER_SCATTER_ND_UPDATE_KERNEL(type, dev, "ScatterNdMin",          \
                                    scatter_nd_op::UpdateOp::MIN);      \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                           \
      type, dev, "ResourceScatterNdMin", scatter_nd_op::UpdateOp::MIN); \
  REGISTER_RESOURCE_SCATTER_ND_UPDATE_KERNEL(                           \
      type, dev, "ResourceScatterNdMax", scatter_nd_op::UpdateOp::MAX);

#define REGISTER_SCATTER_ND_ADD_SUB_GPU(type) \
  REGISTER_SCATTER_ND_ADD_SUB(type, GPU);

#define REGISTER_SCATTER_ND_UPDATE_GPU(type) \
  REGISTER_SCATTER_ND_UPDATE(type, GPU);

#define REGISTER_SCATTER_ND_MIN_MAX_GPU(type) \
  REGISTER_SCATTER_ND_MIN_MAX(type, GPU);

TF_CALL_int32(REGISTER_SCATTER_ND_UPDATE_GPU);
TF_CALL_int32(REGISTER_SCATTER_ND_ADD_SUB_GPU);
TF_CALL_int32(REGISTER_SCATTER_ND_MIN_MAX_GPU);
TF_CALL_int64(REGISTER_SCATTER_ND_UPDATE_GPU);
TF_CALL_int64(REGISTER_SCATTER_ND_ADD_SUB_GPU);
TF_CALL_int64(REGISTER_SCATTER_ND_MIN_MAX_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_UPDATE_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_ADD_SUB_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_MIN_MAX_GPU);
TF_CALL_complex64(REGISTER_SCATTER_ND_UPDATE_GPU);
TF_CALL_complex64(REGISTER_SCATTER_ND_ADD_SUB_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_SCATTER_ND_UPDATE_GPU);
TF_CALL_double(REGISTER_SCATTER_ND_ADD_SUB_GPU);
TF_CALL_double(REGISTER_SCATTER_ND_MIN_MAX_GPU);
TF_CALL_complex128(REGISTER_SCATTER_ND_UPDATE_GPU);
TF_CALL_complex128(REGISTER_SCATTER_ND_ADD_SUB_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#define REGISTER_SCATTER_ND_NONALIASING_ADD_INDEX(type, index_type, dev, name, \
                                                  op)                          \
  REGISTER_KERNEL_BUILDER(Name(name)                                           \
                              .Device(DEVICE_##dev)                            \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ScatterNdUpdateOp<dev##Device, type, index_type, op, \
                                            InputTensorType::NormalTensor>)

#define REGISTER_SCATTER_ND_NONALIASING_ADD(type)                          \
  REGISTER_SCATTER_ND_NONALIASING_ADD_INDEX(type, int32, GPU,              \
                                            "ScatterNdNonAliasingAdd",     \
                                            scatter_nd_op::UpdateOp::ADD); \
  REGISTER_SCATTER_ND_NONALIASING_ADD_INDEX(type, int64, GPU,              \
                                            "ScatterNdNonAliasingAdd",     \
                                            scatter_nd_op::UpdateOp::ADD)

TF_CALL_int32(REGISTER_SCATTER_ND_NONALIASING_ADD);
TF_CALL_int64(REGISTER_SCATTER_ND_NONALIASING_ADD);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ND_NONALIASING_ADD);
TF_CALL_complex64(REGISTER_SCATTER_ND_NONALIASING_ADD);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_SCATTER_ND_NONALIASING_ADD);
TF_CALL_complex128(REGISTER_SCATTER_ND_NONALIASING_ADD);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_SCATTER_ND_NONALIASING_ADD
#undef REGISTER_SCATTER_ND_NONALIASING_ADD_INDEX

#undef REGISTER_SCATTER_ND_ADD_SUB
#undef REGISTER_SCATTER_ND_ADD_SUB_GPU
#undef REGISTER_SCATTER_ND_MIN_MAX
#undef REGISTER_SCATTER_ND_MIN_MAX_GPU
#undef REGISTER_SCATTER_ND_UPDATE
#undef REGISTER_SCATTER_ND_UPDATE_GPU
}  // namespace itex
