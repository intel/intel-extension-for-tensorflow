/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/kernels/gpu/rms_norm_op.h"

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename U>
class RMSNormOp : public OpKernel {
 public:
  explicit RMSNormOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(context, context->GetAttr("use_scale", &use_scale_));
    OP_REQUIRES_OK(context, context->GetAttr("use_center", &use_center_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& gamma = context->input(1);
    const Tensor& beta = context->input(2);

    OP_REQUIRES(context, !use_scale_ || gamma.dims() == 1,
                errors::InvalidArgument("gamma must be 1-dimensional",
                                        gamma.shape().DebugString()));
    OP_REQUIRES(context, !use_center_ || beta.dims() == 1,
                errors::InvalidArgument("beta must be 1-dimensional",
                                        beta.shape().DebugString()));
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be at least 1-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, gamma.dim_size(0) == input.dim_size(input.dims() - 1),
                errors::InvalidArgument(
                    "gamma's size", gamma.shape().DebugString(),
                    " must be equal to input's last-dimensional size, but got",
                    input.shape().DebugString()));
    OP_REQUIRES(context, beta.dim_size(0) == input.dim_size(input.dims() - 1),
                errors::InvalidArgument(
                    "beta's size", beta.shape().DebugString(),
                    " must be equal to input's last-dimensional size, but got",
                    input.shape().DebugString()));

    int cols = input.dim_size(input.dims() - 1);
    int rows = input.NumElements() / cols;

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    functor::RMSNormFunctor<Device, T, U>()(
        context, input.flat<T>(), output->template flat<T>(), gamma.vec<U>(),
        beta.vec<U>(), epsilon_, use_scale_, use_center_, rows, cols);
  }

 private:
  bool use_scale_;
  bool use_center_;
  float epsilon_;
};

namespace functor {

template <typename T, typename U>
struct RMSNormFunctor<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::Flat output,
                  typename TTypes<U>::ConstVec gamma,
                  typename TTypes<U>::ConstVec beta, float epsilon,
                  bool use_scale, bool use_center, int rows, int cols) {
    auto launcher = &launch_rms_norm<T, U, 1, 4, 1024, 16>;

    if (cols <= 128) {
      launcher = &launch_rms_norm<T, U, 4, 1, 128, 8>;
    } else if (cols <= 512) {
      launcher = &launch_rms_norm<T, U, 4, 1, 512, 16>;
    } else if (cols <= 1024) {
      launcher = &launch_rms_norm<T, U, 1, 4, 1024, 16>;
    } else if (cols <= 2048) {
      launcher = &launch_rms_norm<T, U, 1, 8, 2048, 16>;
    } else if (cols <= 8192) {
      launcher = &launch_rms_norm<T, U, 1, 16, 8192, 16>;
    } else {
      /* TODO(itex): support welford updating for large cols. */
      context->SetStatus(errors::InvalidArgument("Unsupported shape"));
      return;
    }

    Params params;
    params.rows = rows;
    params.cols = cols;
    params.input = const_cast<T*>(input.data());
    params.output = output.data();
    params.gamma = const_cast<U*>(gamma.data());
    params.beta = const_cast<U*>(beta.data());
    params.epsilon = epsilon;

    launcher(context, params, use_scale, use_center);
  }
};

}  // namespace functor

#define REGISTER_GPU_KERNEL(T, U)                      \
  REGISTER_KERNEL_BUILDER(Name("ItexRmsNorm")          \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .TypeConstraint<U>("U"), \
                          RMSNormOp<GPUDevice, T, U>);

REGISTER_GPU_KERNEL(float, float);
REGISTER_GPU_KERNEL(Eigen::half, float);
REGISTER_GPU_KERNEL(Eigen::bfloat16, float);
#undef REGISTER_GPU_KERNEL

}  // end namespace itex
