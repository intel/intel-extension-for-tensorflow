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

#include "itex/core/kernels/gpu/inplace_ops_functor.h"

#include "itex/core/utils/status.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
namespace functor {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T, InplaceOpType op>
void DoInPlaceOpImpl(const T* src, const int32* rowids, T* dst, int work_items,
                     const int64 rows, const int64 cols,
                     sycl::nd_item<1> item) {
  auto idx = item.get_global_linear_id();
  if (idx >= work_items) return;

  int64 r = idx / cols;
  int64 c = idx % cols;
  r = (rowids[r] % rows + rows) % rows;  // Guard index range.
  switch (op) {
    case I_UPDATE:
      dst[r * cols + c] = src[idx];
      break;
    case I_ADD:
      dst[r * cols + c] += src[idx];
      break;
    case I_SUB:
      dst[r * cols + c] -= src[idx];
      break;
  }
}

template <typename T, InplaceOpType op>
struct DoInPlaceOpKernel {
  DoInPlaceOpKernel(const T* src, const int32* rowids, T* dst, int work_items,
                    const int64 rows, const int64 cols)
      : src(src),
        rowids(rowids),
        dst(dst),
        work_items(work_items),
        rows(rows),
        cols(cols) {}
  void operator()(sycl::nd_item<1> item) const {
    DoInPlaceOpImpl<T, op>(src, rowids, dst, work_items, rows, cols, item);
  }

 private:
  const T* src;
  const int32* rowids;
  T* dst;
  int work_items;
  const int64 rows;
  const int64 cols;
};

template <typename T>
void DoInplaceOp(const GPUDevice& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
  const int64 nelem = v.NumElements();

  auto Ty = y->flat_outer_dims<T>();
  const int64 nrows = Ty.dimension(0);
  const int64 ncols = Ty.dimension(1);

  auto& stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_wg = (nelem + group_size - 1) / group_size;

  const T* src = v.flat<T>().data();
  const int32* rowids = i.flat<int32>().data();
  T* dst = y->flat<T>().data();

  switch (op) {
    case I_UPDATE:
      stream->submit([&](sycl::handler& cgh) {
        DoInPlaceOpKernel<T, I_UPDATE> task(src, rowids, dst, nelem, nrows,
                                            ncols);
        cgh.parallel_for<DoInPlaceOpKernel<T, I_UPDATE>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                              sycl::range<1>(group_size)),
            task);
      });
      break;
    case I_ADD:
      stream->submit([&](sycl::handler& cgh) {
        DoInPlaceOpKernel<T, I_ADD> task(src, rowids, dst, nelem, nrows, ncols);

        cgh.parallel_for<DoInPlaceOpKernel<T, I_ADD>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                              sycl::range<1>(group_size)),
            task);
      });
      break;
    case I_SUB:
      stream->submit([&](sycl::handler& cgh) {
        DoInPlaceOpKernel<T, I_SUB> task(src, rowids, dst, nelem, nrows, ncols);

        cgh.parallel_for<DoInPlaceOpKernel<T, I_SUB>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                              sycl::range<1>(group_size)),
            task);
      });
      break;
  }
}

template <bool>
void DoInplaceOp(const GPUDevice& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
  const int64 nelem = v.NumElements();

  auto Ty = y->flat_outer_dims<bool>();
  const int64 nrows = Ty.dimension(0);
  const int64 ncols = Ty.dimension(1);
  auto& stream = d.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_wg = (nelem + group_size - 1) / group_size;
  if (op == I_UPDATE) {
    stream->submit([&](sycl::handler& cgh) {
      DoInPlaceOpKernel<bool, I_UPDATE> task(
          v.flat<bool>().data(), i.flat<int32>().data(), y->flat<bool>().data(),
          nelem, nrows, ncols);
      cgh.parallel_for<DoInPlaceOpKernel<bool, I_UPDATE>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });
  }
}

template <>
Status DoInplace(const GPUDevice& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
  ITEX_CHECK_EQ(v.dtype(), y->dtype());
  switch (v.dtype()) {
#define CASE(type)                     \
  case DataTypeToEnum<type>::value:    \
    DoInplaceOp<type>(d, op, i, v, y); \
    break;

    CASE(bool)
    CASE(float)
#ifdef ITEX_ENABLE_DOUBLE
    CASE(double)
#endif  // ITEX_ENABLE_DOUBLE
    CASE(Eigen::half)
    CASE(Eigen::bfloat16)
    CASE(itex::int32)
    CASE(itex::int64)
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(v.dtype()));
  }
  return Status::OK();
}

// Copies x into y.
template <>
Status DoCopy(const GPUDevice& d, const Tensor& x, Tensor* y) {
  ITEX_CHECK_EQ(x.dtype(), y->dtype());
  switch (x.dtype()) {
#define CASE(type)                              \
  case DataTypeToEnum<type>::value:             \
    y->flat<type>().device(d) = x.flat<type>(); \
    break;

    CASE(float)
#ifdef ITEX_ENABLE_DOUBLE
    CASE(double)
#endif  // ITEX_ENABLE_DOUBLE
    CASE(Eigen::half)
    CASE(Eigen::bfloat16)
    CASE(complex64)
    CASE(complex128)
    CASE(int32)
    CASE(int64)
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported dtype: ",
                                     DataTypeString(x.dtype()));
  }
  return Status::OK();
}

// Copies x into y.
template <>
Status DoCopy(const CPUDevice& d, const Tensor& x, Tensor* y) {
  ITEX_CHECK_EQ(x.dtype(), y->dtype());
  switch (x.dtype()) {
#define CASE(type)                              \
  case DataTypeToEnum<type>::value:             \
    y->flat<type>().device(d) = x.flat<type>(); \
    break;

    CASE(float)
    CASE(double)
    CASE(Eigen::half)
    CASE(Eigen::bfloat16)
    CASE(complex64)
    CASE(complex128)
    CASE(int32)
    CASE(int64)
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported dtype: ",
                                     DataTypeString(x.dtype()));
  }
  return Status::OK();
}
}  // namespace functor
}  // namespace itex
