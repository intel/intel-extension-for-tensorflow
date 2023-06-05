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

#include "itex/core/kernels/common/matmul_op.h"

#include "itex/core/kernels/common/batch_matmul_op.h"
#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/onednn/onednn_post_op_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
using LocalAcc = sycl::accessor<T, 2, sycl::access::mode::read_write,
                                sycl::access::target::local>;

using dnnl::memory;

// This template is used when input dimension is equal to three, or larger than
// three while no broadcasting is required.
template <typename T, int c_M, int c_P, int BS_X, int BS_Y, int TILE_K,
          int TILE_AB>
struct BatchMatMulCustomKernel {
  BatchMatMulCustomKernel(const T* A, const T* B, T* C, const int bs,
                          const int M, const int N, const int P,
                          LocalAcc<T> Asub, LocalAcc<T> Bsub, bool adj_A,
                          bool adj_B, bool bcast_A, bool bcast_B)
      : A_(A),
        B_(B),
        C_(C),
        bs_(bs),
        M_(M),
        N_(N),
        P_(P),
        Asub_(Asub),
        Bsub_(Bsub),
        adj_A_(adj_A),
        adj_B_(adj_B),
        bcast_A_(bcast_A),
        bcast_B_(bcast_B) {}
  inline void operator()(sycl::nd_item<3> item) const {
    // Use register tile for better performance
    T a_reg[TILE_AB][c_M];
    T b_reg[TILE_AB][c_P];
    T sum[c_M][c_P] = {{T(0)}};

    int block_i = item.get_group(1);    // block col
    int block_j = item.get_group(2);    // block row
    int batch_idx = item.get_group(0);  // batch
    int item_i = item.get_local_id(1);  // item col in block
    int item_j = item.get_local_id(2);  // item row in block

    const T* A_per_batch = A_ + !bcast_A_ * batch_idx * M_ * N_;
    const T* B_per_batch = B_ + !bcast_B_ * batch_idx * N_ * P_;
    T* C_per_batch = C_ + batch_idx * M_ * P_;

    // reduction in N direction
    for (int n = 0; n < N_; n += TILE_K) {
// Load matrix A into local memory
#pragma unroll
      for (int k = 0; k < DivUp(TILE_K, BS_Y); ++k) {
#pragma unroll
        for (int m = 0; m < c_M; m++) {
          int tile_k_x = k * BS_Y + item_j;
          int tile_k_y = m * BS_X + item_i;
          int global_y = block_i * c_M * BS_X + tile_k_y;
          int global_x = n + tile_k_x;
          if (global_y < M_ && global_x < N_) {
            if (!adj_A_)
              Asub_[tile_k_y][tile_k_x] = A_per_batch[global_y * N_ + global_x];
            else
              Asub_[tile_k_y][tile_k_x] = A_per_batch[global_x * M_ + global_y];
          } else {
            Asub_[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

// Load matrix B into local memory
#pragma unroll
      for (int k = 0; k < DivUp(TILE_K, BS_X); ++k) {
#pragma unroll
        for (int p = 0; p < c_P; p++) {
          int tile_k_x = p * BS_Y + item_j;
          int tile_k_y = k * BS_X + item_i;
          int global_y = n + tile_k_y;
          int global_x = block_j * c_P * BS_Y + tile_k_x;
          if (global_y < N_ && global_x < P_) {
            if (!adj_B_)
              Bsub_[tile_k_y][tile_k_x] = B_per_batch[global_y * P_ + global_x];
            else
              Bsub_[tile_k_y][tile_k_x] = B_per_batch[global_x * N_ + global_y];
          } else {
            Bsub_[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

      item.barrier(sycl::access::fence_space::local_space);

      for (int k = 0; k < TILE_K; k += TILE_AB) {
// load SLM to registers
#pragma unroll
        for (int t_ab = 0; t_ab < TILE_AB; ++t_ab) {
#pragma unroll
          for (int m = 0; m < c_M; ++m) {
            a_reg[t_ab][m] = Asub_[item_i * c_M + m][k + t_ab];
          }
        }
#pragma unroll
        for (int t_ab = 0; t_ab < TILE_AB; ++t_ab) {
#pragma unroll
          for (int p = 0; p < c_P; ++p) {
            b_reg[t_ab][p] = Bsub_[k + t_ab][item_j * c_P + p];
          }
        }

        for (int t_ab = 0; t_ab < TILE_AB; t_ab++) {
#pragma unroll
          for (int m = 0; m < c_M; m++) {
#pragma unroll
            for (int p = 0; p < c_P; p++) {
              sum[m][p] += a_reg[t_ab][m] * b_reg[t_ab][p];
            }
          }
        }  // end of t_ab
      }

      // wait all memory has been consumed
      item.barrier(sycl::access::fence_space::local_space);
    }

    // write results back
    int i = c_M * item.get_global_id(1);  // col
    int j = c_P * item.get_global_id(2);  // row
#pragma unroll
    for (int m = 0; m < c_M; ++m) {
#pragma unroll
      for (int p = 0; p < c_P; ++p) {
        int global_i = i + m;
        int global_j = j + p;
        if (global_i < M_ && global_j < P_) {
          C_per_batch[global_i * P_ + global_j] = sum[m][p];
        }
      }
    }
  }

  const T* A_;
  const T* B_;
  T* C_;
  const int bs_;
  const int M_;
  const int N_;
  const int P_;
  LocalAcc<T> Asub_;
  LocalAcc<T> Bsub_;
  const bool adj_A_, adj_B_;
  const bool bcast_A_, bcast_B_;
};

// This template is used when input dimension is larger than three and
// broadcasting is required.
template <typename T, int c_M, int c_P, int BS_X, int BS_Y, int TILE_K,
          int TILE_AB>
struct BatchMatMulWithBcastKernel {
  BatchMatMulWithBcastKernel(const T* A, const T* B, T* C, const int bs,
                             const int M, const int N, const int P,
                             LocalAcc<T> Asub, LocalAcc<T> Bsub, bool adj_A,
                             bool adj_B, const int64_t* A_offset,
                             const int64_t* B_offset)
      : A_(A),
        B_(B),
        C_(C),
        bs_(bs),
        M_(M),
        N_(N),
        P_(P),
        Asub_(Asub),
        Bsub_(Bsub),
        adj_A_(adj_A),
        adj_B_(adj_B),
        A_offset_(A_offset),
        B_offset_(B_offset) {}
  inline void operator()(sycl::nd_item<3> item) const {
    // Use register tile for better performance
    T a_reg[TILE_AB][c_M];
    T b_reg[TILE_AB][c_P];
    T sum[c_M][c_P] = {{T(0)}};

    int block_i = item.get_group(1);    // block col
    int block_j = item.get_group(2);    // block row
    int batch_idx = item.get_group(0);  // batch
    int item_i = item.get_local_id(1);  // item col in block
    int item_j = item.get_local_id(2);  // item row in block

    const T* A_per_batch = A_ + A_offset_[batch_idx] * M_ * N_;
    const T* B_per_batch = B_ + B_offset_[batch_idx] * N_ * P_;
    T* C_per_batch = C_ + batch_idx * M_ * P_;

    // reduction in N direction
    for (int n = 0; n < N_; n += TILE_K) {
// Load matrix A into local memory
#pragma unroll
      for (int k = 0; k < DivUp(TILE_K, BS_Y); ++k) {
#pragma unroll
        for (int m = 0; m < c_M; m++) {
          int tile_k_x = k * BS_Y + item_j;
          int tile_k_y = m * BS_X + item_i;
          int global_y = block_i * c_M * BS_X + tile_k_y;
          int global_x = n + tile_k_x;
          if (global_y < M_ && global_x < N_) {
            if (!adj_A_)
              Asub_[tile_k_y][tile_k_x] = A_per_batch[global_y * N_ + global_x];
            else
              Asub_[tile_k_y][tile_k_x] = A_per_batch[global_x * M_ + global_y];
          } else {
            Asub_[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

// Load matrix B into local memory
#pragma unroll
      for (int k = 0; k < DivUp(TILE_K, BS_X); ++k) {
#pragma unroll
        for (int p = 0; p < c_P; p++) {
          int tile_k_x = p * BS_Y + item_j;
          int tile_k_y = k * BS_X + item_i;
          int global_y = n + tile_k_y;
          int global_x = block_j * c_P * BS_Y + tile_k_x;
          if (global_y < N_ && global_x < P_) {
            if (!adj_B_)
              Bsub_[tile_k_y][tile_k_x] = B_per_batch[global_y * P_ + global_x];
            else
              Bsub_[tile_k_y][tile_k_x] = B_per_batch[global_x * N_ + global_y];
          } else {
            Bsub_[tile_k_y][tile_k_x] = T(0);
          }
        }
      }

      item.barrier(sycl::access::fence_space::local_space);

      for (int k = 0; k < TILE_K; k += TILE_AB) {
// load SLM to registers
#pragma unroll
        for (int t_ab = 0; t_ab < TILE_AB; ++t_ab) {
#pragma unroll
          for (int m = 0; m < c_M; ++m) {
            a_reg[t_ab][m] = Asub_[item_i * c_M + m][k + t_ab];
          }
        }
#pragma unroll
        for (int t_ab = 0; t_ab < TILE_AB; ++t_ab) {
#pragma unroll
          for (int p = 0; p < c_P; ++p) {
            b_reg[t_ab][p] = Bsub_[k + t_ab][item_j * c_P + p];
          }
        }

        for (int t_ab = 0; t_ab < TILE_AB; t_ab++) {
#pragma unroll
          for (int m = 0; m < c_M; m++) {
#pragma unroll
            for (int p = 0; p < c_P; p++) {
              sum[m][p] += a_reg[t_ab][m] * b_reg[t_ab][p];
            }
          }
        }  // end of t_ab
      }

      // wait all memory has been consumed
      item.barrier(sycl::access::fence_space::local_space);
    }

    // write results back
    int i = c_M * item.get_global_id(1);  // col
    int j = c_P * item.get_global_id(2);  // row
#pragma unroll
    for (int m = 0; m < c_M; ++m) {
#pragma unroll
      for (int p = 0; p < c_P; ++p) {
        int global_i = i + m;
        int global_j = j + p;
        if (global_i < M_ && global_j < P_) {
          C_per_batch[global_i * P_ + global_j] = sum[m][p];
        }
      }
    }
  }

  const T* A_;
  const T* B_;
  T* C_;
  const int bs_;
  const int M_;
  const int N_;
  const int P_;
  LocalAcc<T> Asub_;
  LocalAcc<T> Bsub_;
  const bool adj_A_, adj_B_;
  const int64_t* A_offset_;
  const int64_t* B_offset_;
};

template <typename T, int c_M, int c_P, int BS_X, int BS_Y, int TILE_K,
          int TILE_AB>
void LaunchBmmCustomKernel(OpKernelContext* ctx, const T* A, const T* B, T* C,
                           const int M, const int N, const int P,
                           const bool adj_A, const bool adj_B,
                           const int src_dims, const MatMulBCast& bcast) {
  auto stream = ctx->eigen_gpu_device().stream();

  const int MN = DivUp(M, c_M);
  const int PN = DivUp(P, c_P);

  const bool bcast_A = bcast.is_x_bcast();
  const bool bcast_B = bcast.is_y_bcast();
  const bool is_bcast_required = bcast.IsBroadcastingRequired();
  auto bs = bcast.output_batch_size();

  sycl::range<3> global{static_cast<size_t>(bs),
                        static_cast<size_t>(RoundUp(MN, BS_X)),
                        static_cast<size_t>(RoundUp(PN, BS_Y))};
  sycl::range<3> local{1, BS_X, BS_Y};
  Tensor A_offset_tensor, B_offset_tensor;

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc<T> Asub(sycl::range<2>{c_M * BS_X, TILE_K}, cgh);
    LocalAcc<T> Bsub(sycl::range<2>{TILE_K, c_P * BS_Y}, cgh);
    if (src_dims > 3 && is_bcast_required) {
      const std::vector<int64_t>& x_batch_indices = bcast.x_batch_indices();
      const std::vector<int64_t>& y_batch_indices = bcast.y_batch_indices();
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<int64_t>::value,
                                        TensorShape({bs}), &A_offset_tensor));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<int64_t>::value,
                                        TensorShape({bs}), &B_offset_tensor));
      stream
          ->memcpy(GetTensorBuffer<int64_t>(&A_offset_tensor),
                   x_batch_indices.data(), bs * sizeof(int64_t))
          .wait();
      stream
          ->memcpy(GetTensorBuffer<int64_t>(&B_offset_tensor),
                   y_batch_indices.data(), bs * sizeof(int64_t))
          .wait();
      BatchMatMulWithBcastKernel<T, c_M, c_P, BS_X, BS_Y, TILE_K, TILE_AB> task(
          A, B, C, bs, M, N, P, Asub, Bsub, adj_A, adj_B,
          static_cast<int64_t*>(GetTensorBuffer<int64_t>(&A_offset_tensor)),
          static_cast<int64_t*>(GetTensorBuffer<int64_t>(&B_offset_tensor)));
      cgh.parallel_for<
          BatchMatMulWithBcastKernel<T, c_M, c_P, BS_X, BS_Y, TILE_K, TILE_AB>>(
          sycl::nd_range<3>(global, local), task);
    } else {
      BatchMatMulCustomKernel<T, c_M, c_P, BS_X, BS_Y, TILE_K, TILE_AB> task(
          A, B, C, bs, M, N, P, Asub, Bsub, adj_A, adj_B, bcast_A, bcast_B);
      cgh.parallel_for<
          BatchMatMulCustomKernel<T, c_M, c_P, BS_X, BS_Y, TILE_K, TILE_AB>>(
          sycl::nd_range<3>(global, local), task);
    }
  });
}

template <typename Device, typename Tin, typename Tout,
          bool is_legacy_matmul = false>
class BatchMatMulCustomOp : public OpKernel {
 public:
  explicit BatchMatMulCustomOp(OpKernelConstruction* context)
      : OpKernel(context) {
    if (is_legacy_matmul) {
      // The old MatMul kernel has "transpose_a/transpose_b" attributes.
      OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &adj_x_));
      OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &adj_y_));
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
      OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
    }
  }

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(&mu_compute_);

    dst_tensor_ = nullptr;
    const Tensor& src_tensor = context->input(0);
    const Tensor& weights_tensor = context->input(1);
    auto src_tensor_shape = src_tensor.shape();
    auto weights_tensor_shape = weights_tensor.shape();

    OP_REQUIRES_OK(context,
                   ValidateInputTensors(context, src_tensor, weights_tensor));

    MatMulBCast bcast(src_tensor.shape().dim_sizes(),
                      weights_tensor.shape().dim_sizes());
    OP_REQUIRES(context, bcast.IsValid(),
                errors::InvalidArgument(
                    "In[0] and In[1] must have compatible batch dimensions: ",
                    src_tensor.shape().DebugString(), " vs. ",
                    weights_tensor.shape().DebugString()));

    // dst(bs, m,n) = \sigma{src(bs, m,k) * weights(bs, k, n)} + bias(bs, m,n)
    // Get the actual m & n to set dst_shape, and MatMulBCast will calculate the
    // shape of batches for us
    const int kSrcDims = src_tensor.dims();
    const auto m = adj_x_ ? src_tensor.dim_size(kSrcDims - 1)
                          : src_tensor.dim_size(kSrcDims - 2);
    const auto k = adj_x_ ? src_tensor.dim_size(kSrcDims - 2)
                          : src_tensor.dim_size(kSrcDims - 1);
    const int kWeightDims = weights_tensor.dims();
    const auto k_weights = adj_y_ ? weights_tensor.dim_size(kWeightDims - 1)
                                  : weights_tensor.dim_size(kWeightDims - 2);
    const auto n = adj_y_ ? weights_tensor.dim_size(kWeightDims - 2)
                          : weights_tensor.dim_size(kWeightDims - 1);
    OP_REQUIRES(context, k == k_weights,
                errors::InvalidArgument(
                    "Matrix size-incompatible: In[0]: ",
                    src_tensor.shape().DebugString(),
                    ", In[1]: ", weights_tensor.shape().DebugString()));
    dst_shape_ = bcast.output_batch_shape();
    dst_shape_.AddDim(m);
    dst_shape_.AddDim(n);
    OP_REQUIRES(
        context, dst_shape_.dims() <= 6,
        errors::InvalidArgument(
            "Rank of output tensor must be <= 6, but is ", dst_shape_.dims(),
            ". Current implementation supports up to rank 6 tensors."));

    OP_REQUIRES_OK(context, context->allocate_output(kDstIndex_, dst_shape_,
                                                     &dst_tensor_));
    if (dst_shape_.num_elements() == 0) {
      return;
    }
    if (src_tensor.NumElements() == 0 || weights_tensor.NumElements() == 0) {
      functor::SetZeroFunctor<Device, double> f;
      f(context->eigen_device<Device>(), dst_tensor_->flat<double>());
      return;
    }

    enum { c_M = 1, c_P = 1, BS_X = 16, BS_Y = 16, TILE_K = 16, TILE_AB = 2 };
    LaunchBmmCustomKernel<double, c_M, c_P, BS_X, BS_Y, TILE_K, TILE_AB>(
        context, reinterpret_cast<double*>(src_tensor.data()),
        reinterpret_cast<double*>(weights_tensor.data()),
        reinterpret_cast<double*>(dst_tensor_->data()), m, k, n, adj_x_, adj_y_,
        kSrcDims, bcast);
  }

 protected:
  bool adj_x_ = false;
  bool adj_y_ = false;
  bool trans_x_ = false;
  bool trans_y_ = false;
  const int kSrcIndex_ = 0, kDstIndex_ = 0, kWeightIndex_ = 1;

 private:
  Status ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                              const Tensor& in1) {
    const int ndims = in0.dims();
    if (is_legacy_matmul) {
      if (ndims != 2) {
        return errors::InvalidArgument("In[0] and In[1] ndims must be == 2: ",
                                       ndims);
      }
    } else {
      if (ndims < 2) {
        return errors::InvalidArgument("In[0] and In[1] ndims must be >= 2: ",
                                       ndims);
      }
    }
    return Status::OK();
  }

  mutex mu_compute_;
  Tensor* dst_tensor_;
  std::vector<int64> input_dims_, weights_dims_;
  TensorShape dst_shape_;
};

#define REGISTER_MATMUL_GPU(TYPE)                                            \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("BatchMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),      \
      BatchMatMulOp<GPUDevice, TYPE, TYPE, TYPE>);                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("BatchMatMulV2").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),    \
      BatchMatMulOp<GPUDevice, TYPE, TYPE, TYPE>);                           \
  REGISTER_KERNEL_BUILDER(Name("BatchMatMulV3")                              \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<TYPE>("Ta")                    \
                              .TypeConstraint<TYPE>("Tb")                    \
                              .TypeConstraint<TYPE>("Tout"),                 \
                          BatchMatMulOp<GPUDevice, TYPE, TYPE, TYPE>);       \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedBatchMatMulV2")                    \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<TYPE>("T"),                    \
                          BatchMatMulOp<GPUDevice, TYPE, TYPE, TYPE>);       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),           \
      MatMulOp<GPUDevice, TYPE, TYPE, TYPE>)                                 \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ITEXFusedMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      MatMulOp<GPUDevice, TYPE, TYPE, TYPE>)                                 \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedMatMulWithSum")                    \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<TYPE>("T"),                    \
                          MatMulOp<GPUDevice, TYPE, TYPE, TYPE>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_MATMUL_GPU);
#undef REGISTER_MATMUL_GPU

#ifdef ITEX_ENABLE_DOUBLE
REGISTER_KERNEL_BUILDER(
    Name("MatMul").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    BatchMatMulCustomOp<GPUDevice, double, double, true>)
REGISTER_KERNEL_BUILDER(
    Name("BatchMatMul").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    BatchMatMulCustomOp<GPUDevice, double, double>);
REGISTER_KERNEL_BUILDER(
    Name("BatchMatMulV2").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    BatchMatMulCustomOp<GPUDevice, double, double>);
REGISTER_KERNEL_BUILDER(Name("BatchMatMulV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("Ta")
                            .TypeConstraint<double>("Tb")
                            .TypeConstraint<double>("Tout"),
                        BatchMatMulCustomOp<GPUDevice, double, double>);
#endif

#define REGISTER_MATMUL_GRAD_GPU(TYPE)                    \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedMatMulGrad")    \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          FusedMatMulGradOp<GPUDevice, TYPE, TYPE>)
TF_CALL_float(REGISTER_MATMUL_GRAD_GPU);
TF_CALL_bfloat16(REGISTER_MATMUL_GRAD_GPU);
#undef REGISTER_MATMUL_GRAD_GPU

#define REGISTER_BF32MATMUL_GPU(TYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("_ITEXAccMatMul")                    \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .TypeConstraint<float>("Tout")        \
                              .TypeConstraint<float>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, float>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMul")               \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .TypeConstraint<float>("Tout")        \
                              .TypeConstraint<float>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, float>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMulWithSum")        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .TypeConstraint<float>("Tout")        \
                              .TypeConstraint<float>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, float>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_BF32MATMUL_GPU);
#undef REGISTER_BF32MATMUL_GPU

#define REGISTER_BF32MATMUL_GPU(TYPE)                                         \
  REGISTER_KERNEL_BUILDER(Name("_ITEXAccMatMul")                              \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tout")                  \
                              .TypeConstraint<Eigen::bfloat16>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, Eigen::bfloat16>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMul")                         \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tout")                  \
                              .TypeConstraint<Eigen::bfloat16>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, Eigen::bfloat16>); \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMulGrad")                     \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tgrad"),                \
                          FusedMatMulGradOp<GPUDevice, TYPE, float>);         \
  REGISTER_KERNEL_BUILDER(Name("_ITEXFusedAccMatMulWithSum")                  \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<float>("Tout")                  \
                              .TypeConstraint<Eigen::bfloat16>("Tpost"),      \
                          MatMulOp<GPUDevice, TYPE, float, Eigen::bfloat16>);

TF_CALL_bfloat16(REGISTER_BF32MATMUL_GPU);
#undef REGISTER_BF32MATMUL_GPU

// Concrete Native BatchMatMul INT8 V1 API (deprecated) kernel implementation
#define REGISTER_NATIVE_KERNEL(op, kernel, lhs_type, rhs_type, output_type,   \
                               is_v2, output_type_name)                       \
  REGISTER_KERNEL_BUILDER(Name(op)                                            \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<lhs_type>("T1")                 \
                              .TypeConstraint<rhs_type>("T2")                 \
                              .TypeConstraint<output_type>(output_type_name)  \
                                  HOSTMEMORYLIST,                             \
                          kernel TEMPLATE_ARGS(GPUDevice, lhs_type, rhs_type, \
                                               output_type, is_v2));

#define REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, output_type, \
                                                 is_v2, output_type_name) \
  REGISTER_NATIVE_KERNEL(op, kernel, qint8, qint8, output_type, is_v2,    \
                         output_type_name);

#define REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES(op, kernel, is_v2,             \
                                                output_type_name)              \
  REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, float, is_v2,           \
                                           output_type_name);                  \
  REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, Eigen::bfloat16, is_v2, \
                                           output_type_name);                  \
  REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES(op, kernel, Eigen::half, is_v2,     \
                                           output_type_name);

// Concrete Native BatchMatMul INT8 V2 API (latest) kernel implementation
#define TEMPLATE_ARGS(Device, lhs_type, rhs_type, output_type, is_v2) \
<Device, lhs_type, rhs_type, output_type, is_v2>
#define HOSTMEMORYLIST .HostMemoryList4("min_x", "max_x", "min_y", "max_y")
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES("_QuantizedBatchMatMulV2AndDequantize",
                                        QuantizedBatchMatMulV2Op, false,
                                        "Toutput");
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES(
    "_QuantizedFusedBatchMatMulV2AndDequantize", QuantizedBatchMatMulV2Op,
    false, "Toutput");
#undef HOSTMEMORYLIST

#define HOSTMEMORYLIST .HostMemoryList2("host_inputs", "host_outputs")
REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES("_QuantizedBatchMatMul",
                                        QuantizedBatchMatMulV2Op, true, "Tout");
#undef HOSTMEMORYLIST

#undef TEMPLATE_ARGS

#undef REGISTER_NATIVE_KERNEL_ALL_OUTPUT_TYPES
#undef REGISTER_NATIVE_KERNEL_ALL_LHS_RHS_TYPES
#undef REGISTER_NATIVE_KERNEL

}  // namespace itex
