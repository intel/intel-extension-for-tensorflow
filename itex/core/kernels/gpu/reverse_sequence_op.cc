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

#include "itex/core/kernels/gpu/reverse_sequence_op.h"

#include <memory>
#include <vector>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Tlen>
void CheckErrors(OpKernelContext* context, int batch_dim, int seq_dim) {
  const Tensor& input = context->input(0);
  const Tensor& seq_lens = context->input(1);

  auto seq_lens_t = seq_lens.vec<Tlen>();

  std::vector<Tlen> seq_lens_vec(seq_lens_t.size());

  // Copy seq_len info down for validity checks
  context->eigen_device<Device>().memcpyDeviceToHost(
      seq_lens_vec.data(), seq_lens_t.data(), sizeof(Tlen) * seq_lens_t.size());

  OP_REQUIRES(context, batch_dim != seq_dim,
              errors::InvalidArgument("batch_dim == seq_dim == ", seq_dim));
  OP_REQUIRES(context, seq_dim < input.dims(),
              errors::InvalidArgument("seq_dim must be < input.dims()", "( ",
                                      seq_dim, " vs. ", input.dims(), ")"));
  OP_REQUIRES(context, batch_dim < input.dims(),
              errors::InvalidArgument("batch_dim must be < input.dims()", "( ",
                                      batch_dim, " vs. ", input.dims(), ")"));
  OP_REQUIRES(context, seq_lens.NumElements() == input.dim_size(batch_dim),
              errors::InvalidArgument("len(seq_lens) != input.dims(", batch_dim,
                                      "), ", "(", seq_lens.NumElements(),
                                      " vs. ", input.dim_size(batch_dim), ")"));

  for (size_t d = 0; d < seq_lens_vec.size(); ++d) {
    OP_REQUIRES(context, seq_lens_vec[d] >= 0,
                errors::InvalidArgument("seq_lens(", d, ") < 0"));
    OP_REQUIRES(context, seq_lens_vec[d] <= input.dim_size(seq_dim),
                errors::InvalidArgument("seq_lens(", d, ") > input.dims(",
                                        seq_dim, ")"));
  }
}

void CheckErrorsGPU(OpKernelContext* context, int batch_dim, int seq_dim) {
  const Tensor& input = context->input(0);
  const Tensor& seq_lens = context->input(1);

  OP_REQUIRES(context, batch_dim != seq_dim,
              errors::InvalidArgument("batch_dim == seq_dim == ", seq_dim));
  OP_REQUIRES(context, seq_dim < input.dims(),
              errors::InvalidArgument("seq_dim must be < input.dims()", "( ",
                                      seq_dim, " vs. ", input.dims(), ")"));
  OP_REQUIRES(context, batch_dim < input.dims(),
              errors::InvalidArgument("batch_dim must be < input.dims()", "( ",
                                      batch_dim, " vs. ", input.dims(), ")"));

  OP_REQUIRES(context, seq_lens.NumElements() == input.dim_size(batch_dim),
              errors::InvalidArgument("len(seq_lens) != input.dims(", batch_dim,
                                      "), ", "(", seq_lens.NumElements(),
                                      " vs. ", input.dim_size(batch_dim), ")"));
}

template <>
void CheckErrors<GPUDevice, int32>(OpKernelContext* context, int batch_dim,
                                   int seq_dim) {
  CheckErrorsGPU(context, batch_dim, seq_dim);
}

template <>
void CheckErrors<GPUDevice, int64>(OpKernelContext* context, int batch_dim,
                                   int seq_dim) {
  CheckErrorsGPU(context, batch_dim, seq_dim);
}

template <typename Device, typename T, typename Tlen>
class ReverseSequenceOp : public OpKernel {
 public:
  explicit ReverseSequenceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("batch_dim", &batch_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("seq_dim", &seq_dim_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& seq_lens = context->input(1);

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsVector(seq_lens.shape()),
                errors::InvalidArgument("seq_lens input must be 1-dim, not ",
                                        seq_lens.dims()));

    auto seq_lens_t = seq_lens.vec<Tlen>();

    CheckErrors<Device, Tlen>(context, batch_dim_, seq_dim_);
    if (!context->status().ok()) return;

    const int input_dims = input.dims();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

#define HANDLE_DIM(NDIM)                                                      \
  case NDIM:                                                                  \
    functor::ReverseSequence<Device, T, Tlen, NDIM>::Compute(                 \
        context->eigen_device<Device>(), input.tensor<T, NDIM>(), batch_dim_, \
        seq_dim_, seq_lens_t, output->tensor<T, NDIM>());                     \
    break;

    switch (input_dims) {
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "ReverseSequenceOp : Unhandled input dimensions: ",
                        input_dims));
    }
  }

 private:
  int32 batch_dim_;
  int32 seq_dim_;

  TF_DISALLOW_COPY_AND_ASSIGN(ReverseSequenceOp);
};

#define DEFINE_GPU_SPEC(T, Tlen, dims)                       \
  template class generator::ReverseGenerator<T, Tlen, dims>; \
  template struct functor::ReverseSequence<GPUDevice, T, Tlen, dims>;

#define DEFINE_GPU_SPEC_LEN(T, dims) \
  DEFINE_GPU_SPEC(T, int32, dims);   \
  DEFINE_GPU_SPEC(T, int64, dims);

#define DEFINE_GPU_SPECS(T)  \
  DEFINE_GPU_SPEC_LEN(T, 2); \
  DEFINE_GPU_SPEC_LEN(T, 3); \
  DEFINE_GPU_SPEC_LEN(T, 4); \
  DEFINE_GPU_SPEC_LEN(T, 5);

TF_CALL_float(DEFINE_GPU_SPECS);
TF_CALL_half(DEFINE_GPU_SPECS);
TF_CALL_bfloat16(DEFINE_GPU_SPECS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_GPU_SPECS);
#endif

#define REGISTER_REVERSE_SEQUENCE_DPCPP(type, len_type)          \
  REGISTER_KERNEL_BUILDER(Name("ReverseSequence")                \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<len_type>("Tlen"), \
                          ReverseSequenceOp<GPUDevice, type, len_type>);

#define REGISTER_REVERSE_SEQUENCE_DPCPP_LEN(type) \
  REGISTER_REVERSE_SEQUENCE_DPCPP(type, int32);   \
  REGISTER_REVERSE_SEQUENCE_DPCPP(type, int64);

TF_CALL_float(REGISTER_REVERSE_SEQUENCE_DPCPP_LEN);
TF_CALL_half(REGISTER_REVERSE_SEQUENCE_DPCPP_LEN);
TF_CALL_bfloat16(REGISTER_REVERSE_SEQUENCE_DPCPP_LEN);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_REVERSE_SEQUENCE_DPCPP_LEN);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_REVERSE_SEQUENCE_DPCPP
#undef REGISTER_REVERSE_SEQUENCE_DPCPP_LEN

}  // namespace itex
