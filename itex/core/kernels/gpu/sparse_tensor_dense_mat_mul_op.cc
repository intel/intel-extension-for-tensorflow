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

#include "itex/core/kernels/gpu/sparse_tensor_dense_mat_mul_op.h"

#include <limits>
#include <utility>

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/types.h"

namespace itex {

template <typename Device, typename T, typename Tindices>
class SparseTensorDenseMatMulOp : public OpKernel {
 public:
  explicit SparseTensorDenseMatMulOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_a", &adjoint_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_b", &adjoint_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a_indices = ctx->input(0);
    const Tensor& a_values = ctx->input(1);
    const Tensor& a_shape = ctx->input(2);
    const Tensor& b = ctx->input(3);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("Tensor 'b' is not a matrix"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a_shape.shape()),
                errors::InvalidArgument("Tensor 'a_shape' is not a vector"));

    OP_REQUIRES(
        ctx, a_shape.NumElements() == 2,
        errors::InvalidArgument("Tensor 'a_shape' must have 2 elements"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a_values.shape()),
                errors::InvalidArgument("Tensor 'a_values' is not a vector"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a_indices.shape()),
                errors::InvalidArgument("Tensor 'a_indices' is not a matrix"));

    const int64 nnz = a_indices.shape().dim_size(0);
    OP_REQUIRES(ctx, nnz == a_values.NumElements(),
                errors::InvalidArgument("Number of rows of a_indices does not "
                                        "match number of entries in a_values"));

    OP_REQUIRES(
        ctx, a_indices.shape().dim_size(1) == a_shape.NumElements(),
        errors::InvalidArgument("Number of columns of a_indices does not match "
                                "number of entries in a_shape"));

    auto a_shape_t = a_shape.vec<int64>();
    const int64 outer_left = (adjoint_a_) ? a_shape_t(1) : a_shape_t(0);
    const int64 outer_right =
        (adjoint_b_) ? b.shape().dim_size(0) : b.shape().dim_size(1);
    const int64 inner_left = (adjoint_a_) ? a_shape_t(0) : a_shape_t(1);
    const int64 inner_right =
        (adjoint_b_) ? b.shape().dim_size(1) : b.shape().dim_size(0);

    OP_REQUIRES(
        ctx, inner_right == inner_left,
        errors::InvalidArgument(
            "Cannot multiply A and B because inner dimension does not match: ",
            inner_left, " vs. ", inner_right,
            ".  Did you forget a transpose?  "
            "Dimensions of A: [",
            a_shape_t(0), ", ", a_shape_t(1),
            ").  Dimensions of B: ", b.shape().DebugString()));

    TensorShape out_shape({outer_left, outer_right});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a_values.NumElements() == 0 || b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    CallSparseTensorDenseMatMulFunctor(ctx, a_values, a_indices, b, out);
  }

 private:
  bool adjoint_a_ = false;
  bool adjoint_b_ = false;

  void CallSparseTensorDenseMatMulFunctor(OpKernelContext* ctx,
                                          const Tensor& a_values,
                                          const Tensor& a_indices,
                                          const Tensor& b, Tensor* out) {
#define MAYBE_ADJOINT(ADJ_A, ADJ_B)                                            \
  if (adjoint_a_ == ADJ_A && adjoint_b_ == ADJ_B) {                            \
    Status functor_status = functor::SparseTensorDenseMatMulFunctor<           \
        T, Tindices, ADJ_A, ADJ_B>::Compute(ctx, out->matrix<T>(),             \
                                            a_indices.matrix<Tindices>(),      \
                                            a_values.vec<T>(), b.matrix<T>()); \
    OP_REQUIRES_OK(ctx, functor_status);                                       \
  }

    MAYBE_ADJOINT(false, false)
    MAYBE_ADJOINT(false, true)
    MAYBE_ADJOINT(true, false)
    MAYBE_ADJOINT(true, true)

#undef MAYBE_ADJOINT
  }
};

namespace functor {

template <typename Tidx>
inline Tidx DivUp(Tidx a, Tidx b) {
  return (a + b - 1) / b;
}

constexpr auto GLOBAL_SPACE = sycl::access::address_space::global_space;
// TODO(itex): ElemSize is sensitive to inputs, how to make performance more
// stable?
constexpr int ElemSize = 4;

template <typename T, typename Tsum, typename Tindices, bool ADJ_A, bool ADJ_B>
struct SparseTensorDenseMatmulKernel {
  SparseTensorDenseMatmulKernel(int num_work_items, int out_cols, int out_rows,
                                int b_cols, int n, int total_size,
                                Tindices* a_idx_ptr, const T* b_ptr,
                                T* a_val_ptr, Tsum* out_ptr)
      : num_work_items(num_work_items),
        out_cols(out_cols),
        out_rows(out_rows),
        b_cols(b_cols),
        n(n),
        total_size(total_size),
        a_idx_ptr(a_idx_ptr),
        b_ptr(b_ptr),
        a_val_ptr(a_val_ptr),
        out_ptr(out_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items) return;
    id = ElemSize * id;

    // out_{i,j} = \sum{a_{i,k} * b_{k,j}}
    int i, k;
    bool load_a = true;
    int a_ix = id / out_cols;
    int j = id % out_cols;
#pragma unroll
    for (int elem = 0; elem < ElemSize; ++elem) {
      if (load_a) {
        sycl::vec<Tindices, 2> tmp;
        tmp.load(a_ix, sycl::multi_ptr<Tindices, GLOBAL_SPACE>(a_idx_ptr));
        i = tmp.x();
        k = tmp.y();
        if (ADJ_A) std::swap(i, k);
        load_a = false;
      }
      // if a_row is out of range, skip
      if (FastBoundsCheck(i, out_rows)) {
        int out_idx = i * out_cols + j;
        auto atm = sycl::atomic_ref<Tsum, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device, GLOBAL_SPACE>(
            out_ptr[out_idx]);
        // if a_row is in range, but a_col is out of range, set output
        // as NaN
        if (!FastBoundsCheck(k, n)) {
          atm.fetch_add(std::numeric_limits<Tsum>::quiet_NaN());
        } else {
          const T a_val = a_val_ptr[a_ix];
          const int b_idx = ADJ_B ? (j * b_cols + k) : (k * b_cols + j);
          const T b_val = b_ptr[b_idx];
          atm.fetch_add(static_cast<Tsum>(a_val * b_val));
        }
      }
      if ((++id) >= total_size) {
        return;
      }
      if ((++j) >= out_cols) {
        j = 0;
        load_a = true;
        a_ix = a_ix + 1;
      }
    }
  }

 private:
  int num_work_items;
  int out_cols;
  int out_rows;
  int b_cols;
  int n;
  int total_size;
  Tindices* a_idx_ptr;
  const T* b_ptr;
  T* a_val_ptr;
  Tsum* out_ptr;
};

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B>
Status SparseTensorDenseMatMulFunctor<T, Tindices, ADJ_A, ADJ_B>::Compute(
    OpKernelContext* ctx, typename TTypes<T>::Matrix out,
    typename TTypes<Tindices>::ConstMatrix a_indices,
    typename TTypes<T>::ConstVec a_values, typename TTypes<T>::ConstMatrix b) {
  const int nnz = a_values.size();
  const int b_rows = b.dimension(0);
  const int b_cols = b.dimension(1);
  const int out_rows = out.dimension(0);
  const int out_cols = out.dimension(1);
  const int n = (ADJ_B) ? b_cols : b_rows;

  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  using Tsum = typename SumType<T>::type;
  Tsum* maybe_temp_out_data = nullptr;
  Tensor temp_out_t;
  bool sum_type_is_different = !std::is_same<T, Tsum>::value;
  if (sum_type_is_different) {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DataTypeToEnum<Tsum>::value,
        TensorShape({out.dimension(0), out.dimension(1)}), &temp_out_t));
    auto temp_out = temp_out_t.matrix<Tsum>();
    maybe_temp_out_data = temp_out.data();
    temp_out.device(d) = temp_out.constant(Tsum(0));
  } else {
    // Note: The reinterpret cast is only required to avoid a compilation
    // error; it is only used if Tsum == T.
    maybe_temp_out_data = reinterpret_cast<Tsum*>(out.data());
    out.device(d) = out.constant(T(0));
  }

  auto* stream = d.stream();
  // TODO(itex): why small work-group size is better?
  const int wg_size = 256;
  const int total_size = out_cols * nnz;
  const int num_work_items = DivUp(total_size, ElemSize);
  stream->submit([&](sycl::handler& cgh) {
    auto a_idx_ptr = const_cast<Tindices*>(a_indices.data());
    auto a_val_ptr = const_cast<T*>(a_values.data());
    const T* b_ptr = b.data();
    Tsum* maybe_temp_out_ptr = maybe_temp_out_data;
    SparseTensorDenseMatmulKernel<T, Tsum, Tindices, ADJ_A, ADJ_B> task(
        num_work_items, out_cols, out_rows, b_cols, n, total_size, a_idx_ptr,
        b_ptr, a_val_ptr, maybe_temp_out_ptr);
    cgh.parallel_for<
        SparseTensorDenseMatmulKernel<T, Tsum, Tindices, ADJ_A, ADJ_B>>(
        sycl::nd_range<1>(
            sycl::range<1>(DivUp(num_work_items, wg_size) * wg_size),
            sycl::range<1>(wg_size)),
        task);
  });

  if (sum_type_is_different) {
    out.device(d) = temp_out_t.matrix<Tsum>().template cast<T>();
  }

  return Status::OK();
}
}  // namespace functor

#define DEFINE(T, Tindices)                                                   \
  template struct functor::SparseTensorDenseMatMulFunctor<T, Tindices, false, \
                                                          false>;             \
  template struct functor::SparseTensorDenseMatMulFunctor<T, Tindices, false, \
                                                          true>;              \
  template struct functor::SparseTensorDenseMatMulFunctor<T, Tindices, true,  \
                                                          false>;             \
  template struct functor::SparseTensorDenseMatMulFunctor<T, Tindices, true,  \
                                                          true>;

DEFINE(float, int32);
DEFINE(float, int64);
#undef DEFINE

#define REGISTER_GPU(T, Tindices)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorDenseMatMul")           \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<Tindices>("Tindices") \
                              .HostMemory("a_shape"),               \
                          SparseTensorDenseMatMulOp<GPUDevice, T, Tindices>);

#define REGISTER_KERNELS_GPU(T) \
  REGISTER_GPU(T, int64);       \
  REGISTER_GPU(T, int32)

REGISTER_KERNELS_GPU(float);
REGISTER_KERNELS_GPU(Eigen::bfloat16);
REGISTER_KERNELS_GPU(Eigen::half);
// TODO(itex): compiler does not support the complex64/complex128 registration
// for SparseTensorDenseMatMul complex64/complex128 registration should be
// opened when compiler supports registration REGISTER_KERNELS_GPU(complex64);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNELS_GPU(double);
// REGISTER_KERNELS_GPU(complex128);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_GPU
#undef REGISTER_KERNELS_GPU

}  // namespace itex
