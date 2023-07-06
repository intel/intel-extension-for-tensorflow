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

#ifndef ITEX_CORE_KERNELS_GPU_LINALG_MATRIX_TRIANGULAR_SOLVE_OP_IMPL_H_
#define ITEX_CORE_KERNELS_GPU_LINALG_MATRIX_TRIANGULAR_SOLVE_OP_IMPL_H_
#if ITEX_USE_MKL
#include <string>
#include <utility>
#include <vector>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/matmul_op.h"
#include "itex/core/kernels/common/transpose_functor.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "mkl.h"  // NOLINT(build/include_subdir)
#include "oneapi/mkl/lapack.hpp"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Scalar>
struct LaunchBatchMatrixTriangularSolve;

template <typename Device, typename Scalar>
class BaseMatrixTriangularSolveOp : public OpKernel {
 public:
  explicit BaseMatrixTriangularSolveOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("lower", &lower_));
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  ~BaseMatrixTriangularSolveOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    ValidateInputTensors(ctx, in0, in1);
    if (!ctx->status().ok()) {
      return;
    }

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            in0.shape().DebugString(), " vs. ", in1.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();
    auto batch_size = bcast.output_batch_size();
    auto d0 = in0.dim_size(in0.dims() - 2);
    auto d1 = in0.dim_size(in0.dims() - 1);
    Tensor in0_reshaped;
    OP_REQUIRES(
        ctx,
        in0_reshaped.CopyFrom(in0, TensorShape({bcast.x_batch_size(), d0, d1})),
        errors::Internal("Failed to reshape In[0] from ",
                         in0.shape().DebugString()));
    auto d2 = in1.dim_size(in1.dims() - 2);
    auto d3 = in1.dim_size(in1.dims() - 1);
    Tensor in1_reshaped;
    OP_REQUIRES(
        ctx,
        in1_reshaped.CopyFrom(in1, TensorShape({bcast.y_batch_size(), d2, d3})),
        errors::Internal("Failed to reshape In[1] from ",
                         in1.shape().DebugString()));
    if (adjoint_) std::swap(d0, d1);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().DebugString(), " ", in1.shape().DebugString(),
                    " ", lower_, " ", adjoint_));
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    Tensor out_reshaped;
    OP_REQUIRES(ctx,
                out_reshaped.CopyFrom(*out, TensorShape({batch_size, d0, d3})),
                errors::Internal("Failed to reshape output from ",
                                 out->shape().DebugString()));
    LaunchBatchMatrixTriangularSolve<Device, Scalar>::Launch(
        ctx, in0_reshaped, in1_reshaped, adjoint_, lower_, bcast,
        &out_reshaped);
  }

 private:
  virtual void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                                    const Tensor& in1) = 0;
  bool lower_;
  bool adjoint_;
};

template <class Device, class Scalar>
class MatrixTriangularSolveOp
    : public BaseMatrixTriangularSolveOp<Device, Scalar> {
 public:
  explicit MatrixTriangularSolveOp(OpKernelConstruction* context)
      : BaseMatrixTriangularSolveOp<Device, Scalar>(context) {}

  ~MatrixTriangularSolveOp() override {}

 private:
  void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                            const Tensor& in1) override {
    const auto in0_num_dims = in0.dims();
    OP_REQUIRES(
        ctx, in0_num_dims >= 2,
        errors::InvalidArgument("In[0] ndims must be >= 2: ", in0_num_dims));

    const auto in1_num_dims = in1.dims();
    OP_REQUIRES(
        ctx, in1_num_dims >= 2,
        errors::InvalidArgument("In[1] ndims must be >= 2: ", in1_num_dims));

    const auto in0_last_dim = in0.dim_size(in0_num_dims - 1);
    const auto in0_prev_dim = in0.dim_size(in0_num_dims - 2);
    OP_REQUIRES(ctx, in0_last_dim == in0_prev_dim,
                errors::InvalidArgument(
                    "In[0] matrices in the last dimensions must be square (",
                    in0_last_dim, " =/= ", in0_prev_dim, ")"));
  }
};

template <typename Scalar>
struct LaunchBatchMatrixTriangularSolve<GPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adjoint, bool lower,
                     const MatMulBCast& bcast, Tensor* out) {
    auto* stream = context->GetDeviceStream();
    const uint64 m = in_x.dim_size(1);
    const uint64 n = out->dim_size(2);

    //  Do a memcpy when we don't need to broadcast.
    if (!bcast.IsBroadcastingRequired() || out->shape() == in_y.shape()) {
      auto* src_device_mem = const_cast<Scalar*>(in_y.flat<Scalar>().data());
      auto* dst_device_mem = const_cast<Scalar*>(out->flat<Scalar>().data());
      DeviceMemcpy<GPUDevice>(dst_device_mem, src_device_mem,
                              bcast.y_batch_size() * m * n * sizeof(Scalar),
                              stream);
    } else {
      auto device = stream->get_device();
      std::vector<Scalar*> out_ptrs;
      std::vector<const Scalar*> b_tmp_ptrs;
      auto* b_base_ptr = const_cast<Scalar*>(in_y.flat<Scalar>().data());
      const std::vector<int64>& b_batch_indices = bcast.y_batch_indices();
      for (int64 i = 0; i < bcast.y_batch_size(); ++i) {
        b_tmp_ptrs.push_back(b_base_ptr + i * m * n);
      }
      for (int64 i = 0; i < bcast.output_batch_size(); ++i) {
        const Scalar* src_device_mem = b_tmp_ptrs[b_batch_indices[i]];
        Scalar* dst_device_mem =
            const_cast<Scalar*>(out->flat<Scalar>().data()) + i * m * n;
        DeviceMemcpy<GPUDevice>(dst_device_mem, src_device_mem,
                                m * n * sizeof(Scalar), stream);
      }
    }
    if (out->NumElements() == 0) {
      return;
    }

    // oneMKL assumes column-major, while TF uses row-major, and oneMKL does not
    // support so called `CUBLAS_RIGHT_SIDE` mode as CUDA does in TensorFlow:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsm
    Tensor out_trans;
    TensorShape out_trans_shape(out->shape());
    out_trans_shape.RemoveLastDims(2);
    out_trans_shape.AddDim(n);
    out_trans_shape.AddDim(m);
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Scalar>::v(),
                                          out_trans_shape, &out_trans));

    const auto& device = context->eigen_device<GPUDevice>();
    // for n == 1, we cannot use D2D memcpy since it do transpose only,
    // instead of conjugate transpose for complext data types
    OP_REQUIRES_OK(context,
                   DoConjugateMatrixTranspose(device, *out, &out_trans));

    const int64 batch_size = bcast.output_batch_size();
    std::vector<const Scalar*> a_ptrs;
    std::vector<Scalar*> out_ptrs;
    std::vector<const Scalar*> a_tmp_ptrs;
    a_ptrs.reserve(batch_size);
    out_ptrs.reserve(batch_size);
    a_tmp_ptrs.reserve(bcast.x_batch_size());
    auto* a_base_ptr = const_cast<Scalar*>(in_x.flat<Scalar>().data());
    auto* out_base_ptr = const_cast<Scalar*>(out_trans.flat<Scalar>().data());

    if (!bcast.IsBroadcastingRequired()) {
      for (int64 i = 0; i < batch_size; ++i) {
        a_ptrs.push_back(a_base_ptr + i * m * m);
        out_ptrs.push_back(out_base_ptr + i * m * n);
      }
    } else {
      const std::vector<int64>& a_batch_indices = bcast.x_batch_indices();
      for (int64 i = 0; i < bcast.x_batch_size(); ++i) {
        a_tmp_ptrs.push_back(a_base_ptr + i * m * m);
      }
      for (int64 i = 0; i < batch_size; ++i) {
        a_ptrs.push_back(a_tmp_ptrs[a_batch_indices[i]]);
        out_ptrs.push_back(out_base_ptr + i * m * n);
      }
    }

    // set `mkl::uplo` and `mkl::transpose` to use a as columon-major directly
    auto uplo = lower ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
    auto trans = adjoint ? oneapi::mkl::transpose::nontrans
                         : oneapi::mkl::transpose::conjtrans;
    auto diag = oneapi::mkl::diag::nonunit;

    int64_t nrhs = n;
    int64_t lda = m;
    int64_t ldb = m;

    int64_t info = 0;
    std::string err_msg;
    try {
      std::int64_t scratchpad_size =
          oneapi::mkl::lapack::trtrs_scratchpad_size<Scalar>(
              *stream, uplo, trans, diag, m, nrhs, lda, ldb);
      Tensor scratchpad;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Scalar>::v(),
                                            {scratchpad_size}, &scratchpad));
      for (int batch = 0; batch < batch_size; ++batch) {
        oneapi::mkl::lapack::trtrs(*stream, uplo, trans, diag, m, nrhs,
                                   const_cast<Scalar*>(a_ptrs[batch]), lda,
                                   const_cast<Scalar*>(out_ptrs[batch]), ldb,
                                   static_cast<Scalar*>(scratchpad.data()),
                                   scratchpad_size);
      }
    } catch (oneapi::mkl::lapack::exception const& e) {
      info = e.info();
      err_msg = std::string(e.what());
    }
    OP_REQUIRES(context, info == 0,
                errors::Internal("Unexpected exception caught during call to "
                                 "LAPACK API, error message: ",
                                 err_msg));

    // copy output back
    OP_REQUIRES_OK(context, DoConjugateMatrixTranspose(device, out_trans, out));
  }
};
}  // namespace itex
#endif  // ITEX_USE_MKL
#endif  // ITEX_CORE_KERNELS_GPU_LINALG_MATRIX_TRIANGULAR_SOLVE_OP_IMPL_H_
