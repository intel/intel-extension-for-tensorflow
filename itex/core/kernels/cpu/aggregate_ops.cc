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

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

template <typename Device, typename T>
class AddNOp : public OpKernel {
 public:
  ~AddNOp() {}
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  TensorShape GetTensorShape(OpKernelContext* context, size_t src_index) {
    const Tensor& src_tensor = context->input(src_index);
    return src_tensor.shape();
  }

  void Compute(OpKernelContext* context) override {
    if (!context->ValidateInputsAreSameShape()) return;
    const int num_inputs = context->num_inputs();
    const size_t kSrc0Idx = 0;
    const size_t kOutputIdx = 0;
    const TensorShape src0_shape = GetTensorShape(context, kSrc0Idx);

    // Try to forward input to output if only got 1 input.
    if (num_inputs == 1) {
      context->set_output(0, context->input(0));
      return;
    }

    // Check if the input shape is same
    if (!context->ValidateInputsAreSameShape()) return;

    try {
      Tensor* dst_tensor = nullptr;
      TensorShape output_tf_shape = src0_shape;

      // Nothing to compute, return.
      if (src0_shape.num_elements() == 0) {
        OP_REQUIRES_OK(context, context->allocate_output(
                                    kOutputIdx, output_tf_shape, &dst_tensor));
        return;
      }

      // Fallback to eigen if input is scalar.
      if (src0_shape.dims() == 0) {
        // Try to forward and accumulate the result in one of the input buffers.
        gtl::InlinedVector<int, 8> input_indices(num_inputs);
        std::iota(input_indices.begin(), input_indices.end(), 0);
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, src0_shape, &output));
        auto To = output->flat<T>();

#define I(IDX) context->input(input_indices[IDX]).template flat<T>()

        static const int kWidth = 8;
        int r = num_inputs % kWidth;

        switch (r) {
          case 2: {
            functor::Add2Functor<Device, T> functor2;
            functor2(context->eigen_device<Device>(), To, I(0), I(1));
            break;
          }
          case 3: {
            functor::Add3Functor<Device, T> functor3;
            functor3(context->eigen_device<Device>(), To, I(0), I(1), I(2));
            break;
          }
          case 4: {
            functor::Add4Functor<Device, T> functor4;
            functor4(context->eigen_device<Device>(), To, I(0), I(1), I(2),
                     I(3));
            break;
          }
          case 5: {
            functor::Add5Functor<Device, T> functor5;
            functor5(context->eigen_device<Device>(), To, I(0), I(1), I(2),
                     I(3), I(4));
            break;
          }
          case 6: {
            functor::Add6Functor<Device, T> functor6;
            functor6(context->eigen_device<Device>(), To, I(0), I(1), I(2),
                     I(3), I(4), I(5));
            break;
          }
          case 7: {
            functor::Add7Functor<Device, T> functor7;
            functor7(context->eigen_device<Device>(), To, I(0), I(1), I(2),
                     I(3), I(4), I(5), I(6));
            break;
          }
          case 0: {
            functor::Add8Functor<Device, T> functor8;
            functor8(context->eigen_device<Device>(), To, I(0), I(1), I(2),
                     I(3), I(4), I(5), I(6), I(7));
            r = 8;
            break;
          }
          case 1: {
            functor::Add9Functor<Device, T> functor9;
            functor9(context->eigen_device<Device>(), To, I(0), I(1), I(2),
                     I(3), I(4), I(5), I(6), I(7), I(8));
            r = 9;
            break;
          }
        }

        for (; r < num_inputs; r += kWidth) {
          functor::Add8pFunctor<Device, T> functor8p;
          functor8p(context->eigen_device<Device>(), To, I(r), I(r + 1),
                    I(r + 2), I(r + 3), I(r + 4), I(r + 5), I(r + 6), I(r + 7));
        }
#undef I
        return;
      }

      // Create memory descriptor for OneDnn.
      std::vector<dnnl::memory::desc> srcs_pd;
      for (int src_idx = 0; src_idx < num_inputs; ++src_idx) {
        dnnl::memory::desc src_md;
        const Tensor& src_tensor = context->input(src_idx);
        auto dims = TFShapeToOneDnnDims(src_tensor.shape());
        src_md = CreatePlainMemDescWithFormatTag<T>(dims);

        srcs_pd.push_back(src_md);
      }

      // Allocate output
      std::vector<float> coeff(num_inputs, 1.0);
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      dnnl::primitive_attr attr;
#ifdef ITEX_ONEDNN_3_0
      auto sum_pd =
          dnnl::sum::primitive_desc(onednn_engine, coeff, srcs_pd, attr);
#else
      auto sum_pd =
          dnnl::sum::primitive_desc(coeff, srcs_pd, onednn_engine, attr);
#endif

      Tensor scratchpad_tensor;
      int64 scratchpad_size = sum_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(sum_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      OP_REQUIRES_OK(context, context->allocate_output(
                                  kOutputIdx, output_tf_shape, &dst_tensor));

      // Create Sum op, and submit for execution.
      dnnl::sum sum_op(sum_pd);
      dnnl::memory dst_mem = CreateDnnlMemory(sum_pd.dst_desc(), onednn_engine,
                                              GetTensorBuffer<T>(dst_tensor));
      std::unordered_map<int, dnnl::memory> net_args = {
          {DNNL_ARG_DST, dst_mem}};
      for (int src_idx = 0; src_idx < num_inputs; ++src_idx) {
        dnnl::memory src_mem =
            CreateDnnlMemory(srcs_pd[src_idx], onednn_engine,
                             GetTensorBuffer<T>(&context->input(src_idx)));
        net_args.insert({DNNL_ARG_MULTIPLE_SRC + src_idx, src_mem});
      }

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      sum_op.execute(onednn_stream, net_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }
};

#define REGISTER_ADDN(T)                                           \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("_ITEXAddN").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      AddNOp<CPUDevice, T>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_ADDN);
#undef REGISTER_ADDN
}  // namespace itex
