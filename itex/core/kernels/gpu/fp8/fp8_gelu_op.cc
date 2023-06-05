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

#include "itex/core/kernels/gpu/fp8/fp8_gelu_gpu.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
class Fp8GeluOp : public OpKernel {
 public:
  explicit Fp8GeluOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype", &fp8_dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index", &fp8_meta_index_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& inp = context->input(0);
    const Tensor& amax = context->input(1);
    const Tensor& scale = context->input(2);

    auto amax_ptr =
        const_cast<float*>(amax.flat<float>().data()) + fp8_meta_index_;
    auto scale_ptr = scale.flat<float>().data() + fp8_meta_index_;

    auto inp_shape = inp.shape();

    Tensor* gelu_out;
    OP_REQUIRES_OK(context, context->allocate_output(0, inp_shape, &gelu_out));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_, output_t, int8,
        functor::Fp8Gelu<T, output_t>(context, inp.flat<T>().data(),
                                      gelu_out->flat<int8>().data(), amax_ptr,
                                      scale_ptr, inp.NumElements()););
  }

 private:
  std::string fp8_dtype_;
  int fp8_meta_index_;
};

#define REGISTER_FP8_GELU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Fp8Gelu").Device(DEVICE_GPU).TypeConstraint<T>("in_dtype"), \
      Fp8GeluOp<T>);

REGISTER_FP8_GELU(float);
REGISTER_FP8_GELU(Eigen::bfloat16);
#undef REGISTER_FP8_QUANTIZE_FUSION

}  // namespace itex
