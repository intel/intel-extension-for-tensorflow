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

#include "itex/core/kernels/gpu/fp8/fp8_quantize_gpu.h"
#include "itex/core/utils/plugin_tensor.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename SrcT>
class Fp8QuantizeOp : public OpKernel {
 public:
  explicit Fp8QuantizeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype", &fp8_dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index", &fp8_meta_index_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& inp = context->input(0);
    Tensor& amax = const_cast<Tensor&>(context->input(1));
    const Tensor& scale = context->input(2);

    auto amax_ptr = amax.flat<float>().data() + fp8_meta_index_;
    auto scale_ptr = scale.flat<float>().data() + fp8_meta_index_;

    Tensor* out;
    OP_REQUIRES_OK(context, context->allocate_output(0, inp.shape(), &out));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_, output_t, int8,
        functor::Fp8Quantize<SrcT, output_t>(context, inp.flat<SrcT>().data(),
                                             out->flat<int8>().data(), amax_ptr,
                                             scale_ptr, inp.NumElements()););
  }

 private:
  std::string fp8_dtype_;
  int fp8_meta_index_;
};

template <typename Device, typename DstT>
class Fp8DequantizeOp : public OpKernel {
 public:
  explicit Fp8DequantizeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fp8_dtype", &fp8_dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fp8_meta_index", &fp8_meta_index_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& inp = context->input(0);
    const Tensor& scale_inv = context->input(1);

    auto scale_inv_ptr = scale_inv.flat<float>().data() + fp8_meta_index_;

    Tensor* out;
    OP_REQUIRES_OK(context, context->allocate_output(0, inp.shape(), &out));

    FP8_TYPE_SWITCH(
        context, fp8_dtype_, input_t, int8,
        functor::Fp8Dequantize<input_t, DstT>(
            context, inp.flat<int8>().data(), out->flat<DstT>().data(),
            scale_inv_ptr, inp.NumElements()););
  }

 private:
  std::string fp8_dtype_;
  int fp8_meta_index_;
};

#define REGISTER_QUANTIZATION(T)                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Fp8Quantize").Device(DEVICE_GPU).TypeConstraint<T>("in_dtype"),    \
      Fp8QuantizeOp<GPUDevice, T>);                                            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Fp8Dequantize").Device(DEVICE_GPU).TypeConstraint<T>("out_dtype"), \
      Fp8DequantizeOp<GPUDevice, T>);

REGISTER_QUANTIZATION(Eigen::bfloat16);
REGISTER_QUANTIZATION(Eigen::half);
REGISTER_QUANTIZATION(float);

#undef REGISTER_QUANTIZATION

};  // namespace itex
