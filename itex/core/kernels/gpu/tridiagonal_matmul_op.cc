/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Scalar>
class TridiagonalMatMulKernelTask;

template <typename Scalar>
void TridiagonalMatMulKernel(const Eigen::GpuDevice& device, int batch_size,
                             int m, int n, const Scalar* superdiag,
                             const Scalar* maindiag, const Scalar* subdiag,
                             const Scalar* rhs, Scalar* product) {
  const int total_count = batch_size * m * n;
  auto stream = device.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (total_count + group_size - 1) / group_size;

  stream->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<TridiagonalMatMulKernelTask<Scalar>>(
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                          sycl::range<1>(group_size)),
        [=](sycl::nd_item<1> item) {
          auto out_idx = item.get_global_linear_id();
          if (out_idx >= total_count) return;
          int row_id = out_idx / n;
          Scalar result = maindiag[row_id] * rhs[out_idx];
          if (row_id % m != 0) {
            result = result + subdiag[row_id] * rhs[out_idx - n];
          }
          if ((row_id + 1) % m != 0) {
            result = result + superdiag[row_id] * rhs[out_idx + n];
          }
          product[out_idx] = result;
        });
  });
}

template <typename Scalar>
class TridiagonalMatMulOpGpu : public OpKernel {
 public:
  explicit TridiagonalMatMulOpGpu(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) final {
    const Tensor& superdiag = context->input(0);
    const Tensor& maindiag = context->input(1);
    const Tensor& subdiag = context->input(2);
    const Tensor& rhs = context->input(3);

    const int ndims = rhs.dims();
    int64 batch_size = 1;
    for (int i = 0; i < ndims - 2; i++) {
      batch_size *= rhs.dim_size(i);
    }
    const int m = rhs.dim_size(ndims - 2);
    const int n = rhs.dim_size(ndims - 1);

    // Allocate output.
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs.shape(), &output));

    const GPUDevice& device = context->eigen_device<GPUDevice>();
    TridiagonalMatMulKernel<Scalar>(
        device, batch_size, m, n, superdiag.flat<Scalar>().data(),
        maindiag.flat<Scalar>().data(), subdiag.flat<Scalar>().data(),
        rhs.flat<Scalar>().data(), output->flat<Scalar>().data());
  }
};

#define REGISTER_LINALG_OP_GPU(OpName, OpClass, Scalar) \
  REGISTER_KERNEL_BUILDER(                              \
      Name(OpName).Device(DEVICE_GPU).TypeConstraint<Scalar>("T"), OpClass)

REGISTER_LINALG_OP_GPU("TridiagonalMatMul", TridiagonalMatMulOpGpu<float>,
                       float);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER_LINALG_OP_GPU("TridiagonalMatMul", TridiagonalMatMulOpGpu<double>,
                       double);
#endif
#undef REGISTER_LINALG_OP_GPU

}  // namespace itex
