/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/kernels/gpu/aggregate_ops.h"

#include "itex/core/kernels/gpu/reduction_itex_gpu_kernels.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AddNOp : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (!context->ValidateInputsAreSameShape()) return;
    const Tensor& input0 = context->input(0);
    const int num = context->num_inputs();

    if (num == 1) {
      context->set_output(0, input0);
      return;
    }

    // Try to forward and accumulate the result in one of the input buffers.
    gtl::InlinedVector<int, 8> input_indices(num);
    std::iota(input_indices.begin(), input_indices.end(), 0);
    Tensor* output = nullptr;
    /* TODO(itex): This way is blocked by RefCountIsOne().
    int candidate_input_index = 0;
    int forwarded_input = -1;
    for (int input_idx = 0; input_idx < num; ++input_idx) {
      candidate_input_index = input_idx;
      gtl::ArraySlice<int> candidate_input_indices(&candidate_input_index, 1);
      OP_REQUIRES_OK(&context, context->forward_input_or_allocate_output(
                                    candidate_input_indices, 0, input0.shape(),
                                    &output, &forwarded_input));
      if (forwarded_input != -1) break;
    }
    if (forwarded_input > 0) {
      // Move the forwarded buffer to the front so we don't double count
      // anything if there are more than 8 inputs.
      input_indices[0] = forwarded_input;
      input_indices[forwarded_input] = 0;
    }
    */
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input0.shape(), &output));
    auto To = output->flat<T>();

#define I(IDX) context->input(input_indices[IDX]).template flat<T>()

    static const int kWidth = 8;
    int r = num % kWidth;

    switch (r) {
      case 2: {
        functor::Add2Functor<Device, T> functor2;
        functor2(context->eigen_gpu_device(), To, I(0), I(1));
        break;
      }
      case 3: {
        functor::Add3Functor<Device, T> functor3;
        functor3(context->eigen_gpu_device(), To, I(0), I(1), I(2));
        break;
      }
      case 4: {
        functor::Add4Functor<Device, T> functor4;
        functor4(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3));
        break;
      }
      case 5: {
        functor::Add5Functor<Device, T> functor5;
        functor5(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4));
        break;
      }
      case 6: {
        functor::Add6Functor<Device, T> functor6;
        functor6(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4),
                 I(5));
        break;
      }
      case 7: {
        functor::Add7Functor<Device, T> functor7;
        functor7(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4),
                 I(5), I(6));
        break;
      }
      case 0: {
        functor::Add8Functor<Device, T> functor8;
        functor8(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4),
                 I(5), I(6), I(7));
        r = 8;
        break;
      }
      case 1: {
        functor::Add9Functor<Device, T> functor9;
        functor9(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4),
                 I(5), I(6), I(7), I(8));
        r = 9;
        break;
      }
    }

    for (; r < num; r += kWidth) {
      functor::Add8pFunctor<Device, T> functor8p;
      functor8p(context->eigen_gpu_device(), To, I(r), I(r + 1), I(r + 2),
                I(r + 3), I(r + 4), I(r + 5), I(r + 6), I(r + 7));
    }
#undef I
  }
};

template <typename Device>
class AddNOp<Device, Variant> : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    auto binary_add_func = [](TF_OpKernelContext* tf_ctx, TF_Tensor* tf_a,
                              TF_Tensor* tf_b, TF_Tensor* tf_out) {
      OpKernelContext ctx(tf_ctx);
      Tensor a(tf_a);
      Tensor b(tf_b);
      Tensor out(tf_out);
      switch (out.dtype()) {
#define DTYPE_CASE(dtype)                                  \
  case DataTypeToEnum<dtype>::value:                       \
    out.flat<dtype>().device(ctx.eigen_device<Device>()) = \
        a.flat<dtype>() + b.flat<dtype>();                 \
    break;
        TF_CALL_NUMBER_TYPES(DTYPE_CASE)
        default:
          break;
#undef DTYPE_CASE
      }
    };

    TF_OpKernelContext* tf_ctx = ctx->Get();
    TF_Status* tf_status = TF_NewStatus();
    TF_AddNVariant(tf_ctx, binary_add_func, tf_status);
    Status status = StatusFromTF_Status(tf_status);
    ITEX_CHECK_OK(status);
    TF_DeleteStatus(tf_status);
  }
};

#define REGISTER_KERNEL(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("AddN").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      AddNOp<GPUDevice, TYPE>)
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_bfloat16(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
TF_CALL_complex64(REGISTER_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNEL);
TF_CALL_complex128(REGISTER_KERNEL);
#endif
REGISTER_KERNEL(Variant)
#undef REGISTER_KERNEL

template <typename Device, typename T>
class FusedAddNOp;

template <typename T>
struct squareHalf {
  inline T operator()(const T& x) const { return static_cast<T>(0.5) * x * x; }
};

template <typename T>
void FusedAddNKernel(OpKernelContext* context, const int num, T* output) {
  Tensor scratch;
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                 TensorShape({num}), &scratch));

  for (int i = 0; i < num; ++i) {
    const Tensor& input = context->input(i);
    const int nelems = input.NumElements();

    LaunchFullReduction<const T, T, T, sycl::plus<T>, squareHalf<T>>(
        context, input.flat<T>().data(), scratch.flat<T>().data() + i, T(0),
        nelems, sycl::plus<T>(), squareHalf<T>());
  }
  LaunchFullReduction<const T, T, T, sycl::plus<T>>(
      context, scratch.flat<T>().data(), output, T(0), num, sycl::plus<T>());
}

template <typename T>
class FusedAddNOp<GPUDevice, T> : public OpKernel {
 public:
  explicit FusedAddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const int num = context->num_inputs();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    FusedAddNKernel<T>(context, num, output->flat<T>().data());
  }
};

#define REGISTER_FUSEDADDN(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("_FusedAddN").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      FusedAddNOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER_FUSEDADDN);
#undef REGISTER_FUSEDADDN
}  // namespace itex
