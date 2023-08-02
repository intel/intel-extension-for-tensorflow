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
class OneDnnAddNOp : public OpKernel {
 public:
  ~OneDnnAddNOp() {}
  explicit OneDnnAddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  TensorShape GetTensorShape(OpKernelContext* context, size_t src_index) {
    const Tensor& src_tensor = context->input(src_index);
    OneDnnShape src_onednn_shape;
    GetOneDnnShape(context, src_index, &src_onednn_shape);
    return src_onednn_shape.IsOneDnnTensor() ? src_onednn_shape.GetTfShape()
                                             : src_tensor.shape();
  }

  // Return first tensor index which is in OneDnn layout, or -1 with no OneDnn
  // input.
  int FindOneDnnInputIndex(OpKernelContext* context) {
    int onednn_index = -1;
    const int num_inputs = context->num_inputs() / 2;

    OneDnnShape src_onednn_shape;
    for (size_t i = 0; i < num_inputs; ++i) {
      GetOneDnnShape(context, i, &src_onednn_shape);
      if (src_onednn_shape.IsOneDnnTensor()) {
        onednn_index = i;
        break;
      }
    }

    return onednn_index;
  }

  void Compute(OpKernelContext* context) override {
    const int num_inputs = context->num_inputs() / 2;
    const size_t kSrc0Idx = 0;
    const size_t kOutputIdx = 0;
    const TensorShape src0_shape = GetTensorShape(context, kSrc0Idx);

    // Try to forward input to output if only got 1 input.
    // Back to normal process if forward failed.
    if (num_inputs == 1) {
      const int kUnsuccess = -1;
      int is_forward_success = kUnsuccess;
      OneDnnShape src_onednn_shape;
      GetOneDnnShape(context, kSrc0Idx, &src_onednn_shape);
      Tensor* dst_tensor;

      ForwardOrAllocateOutputSetOneDnnShape(
          context, kSrc0Idx, kOutputIdx, &dst_tensor, src0_shape,
          src_onednn_shape, &is_forward_success);
      if (is_forward_success != kUnsuccess) return;
    }

    // Check if the input shape is same
    for (size_t i = 1; i < num_inputs; ++i) {
      if (!src0_shape.IsSameSize(GetTensorShape(context, i))) {
        context->SetStatus(
            errors::InvalidArgument("Inputs to operation _OneDnnAddN must have "
                                    "the same size and shape.  Input 0: ",
                                    src0_shape.DebugString(), " != input : ", i,
                                    GetTensorShape(context, i).DebugString()));
        return;
      }
    }

    try {
      Tensor* dst_tensor = nullptr;
      TensorShape output_tf_shape = src0_shape;
      OneDnnShape output_onednn_shape;

      // Nothing to compute, return.
      if (src0_shape.num_elements() == 0) {
        output_onednn_shape.SetOneDnnTensor(false);
        AllocateOutputSetOneDnnShape(context, kOutputIdx, &dst_tensor,
                                     output_tf_shape, output_onednn_shape);
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

        OneDnnShape dst_onednn_shape;
        dst_onednn_shape.SetOneDnnTensor(false);
        AllocateMetaData(context, 0, dst_onednn_shape);
        return;
      }

      bool has_onednn_input = false;
      int onednn_input_index = FindOneDnnInputIndex(context);
      OneDnnTensorFormat onednn_data_format =
          OneDnnTensorFormat::FORMAT_INVALID;
      TensorFormat tf_data_format;
      dnnl::memory::format_tag dnn_fmt = dnnl::memory::format_tag::any;
      std::vector<dnnl::memory::desc> srcs_pd_blocked;
      if (onednn_input_index >= 0) {
        has_onednn_input = true;
        OneDnnShape src_onednn_shape;
        GetOneDnnShape(context, onednn_input_index, &src_onednn_shape);
        dnnl::memory::desc src_md = src_onednn_shape.GetOneDnnLayout();
        srcs_pd_blocked = std::vector<dnnl::memory::desc>(num_inputs, src_md);
        // OneDnn input has the data format information.
        onednn_data_format = src_onednn_shape.GetTfDataFormat();
        TensorShape src_tf_shape = src_onednn_shape.GetTfShape();
        if (src_tf_shape.dims() == 4 || src_tf_shape.dims() == 5) {
          tf_data_format = OneDnnDataFormatToTFDataFormat(onednn_data_format);
          dnn_fmt = OneDnnTensorFormatToTag(onednn_data_format);
        }
      }

      // Create memory descriptor for OneDnn.
      // If all input in Tensorflow format, create block memory descriptor,
      // else convert TF format to OneDnn memory descriptor
      std::vector<dnnl::memory::desc> srcs_pd;
      for (int src_idx = 0; src_idx < num_inputs; ++src_idx) {
        OneDnnShape src_onednn_shape;
        GetOneDnnShape(context, src_idx, &src_onednn_shape);
        dnnl::memory::desc src_md;
        const Tensor& src_tensor = context->input(src_idx);

        if (src_onednn_shape.IsOneDnnTensor()) {
          src_md = src_onednn_shape.GetOneDnnLayout();
        } else {
          if (has_onednn_input) {
            dnnl::memory::dims src_dims;
            if (src_tensor.dims() == 4 || src_tensor.dims() == 5) {
              src_dims = TFShapeToOneDnnDimsInNC(
                  src_tensor.shape(), tf_data_format, src_tensor.dims() == 4);
              src_md = dnnl::memory::desc(src_dims, OneDnnType<T>(), dnn_fmt);
            } else {
              src_dims = TFShapeToOneDnnDims(src_tensor.shape());
              src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
            }
          } else {
            auto dims = TFShapeToOneDnnDims(src_tensor.shape());
            src_md = CreatePlainMemDescWithFormatTag<T>(dims);
          }
        }
        srcs_pd.push_back(dnnl::memory::desc(src_md));
      }

      // Allocate output
      std::vector<float> coeff(num_inputs, 1.0);
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      auto sum_pd =
          has_onednn_input
              ? dnnl::sum::primitive_desc(onednn_engine, coeff, srcs_pd_blocked,
                                          attr)
              : dnnl::sum::primitive_desc(onednn_engine, coeff, srcs_pd, attr);
      Tensor scratchpad_tensor;
      int64 scratchpad_size = sum_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(sum_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      SetOutputTensorShape(sum_pd.dst_desc(), onednn_data_format,
                           &output_tf_shape, &output_onednn_shape,
                           has_onednn_input);
      AllocateOutputSetOneDnnShape(context, kOutputIdx, &dst_tensor,
                                   output_tf_shape, output_onednn_shape);

      // Create Sum op, and submit for execution.
      sum_op_ = dnnl::sum(sum_pd);
      dnnl::memory dst_mem = CreateDnnlMemory(sum_pd.dst_desc(), onednn_engine,
                                              GetTensorBuffer<T>(dst_tensor));
      std::unordered_map<int, dnnl::memory> net_args = {
          {DNNL_ARG_DST, dst_mem}};
      std::vector<Tensor> src_reorder_tensor_vec(num_inputs);
      for (int src_idx = 0; src_idx < num_inputs; ++src_idx) {
        dnnl::memory src_mem =
            CreateDnnlMemory(srcs_pd[src_idx], onednn_engine,
                             GetTensorBuffer<T>(&context->input(src_idx)));
        dnnl::memory reorder_mem;
        bool is_src_reordered = (srcs_pd[src_idx] != sum_pd.src_desc(src_idx));
        if (is_src_reordered) {
          int64 src_reorder_size =
              sum_pd.src_desc(src_idx).get_size() / sizeof(T);
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DataTypeToEnum<T>::v(),
                                      TensorShape({src_reorder_size}),
                                      &src_reorder_tensor_vec[src_idx]));

          reorder_mem = CreateDnnlMemory(
              sum_pd.src_desc(src_idx), onednn_engine,
              GetTensorBuffer<T>(&src_reorder_tensor_vec[src_idx]));
          ReorderMemory(*context, &src_mem, &reorder_mem, onednn_engine);
        }

        net_args.insert({DNNL_ARG_MULTIPLE_SRC + src_idx,
                         is_src_reordered ? reorder_mem : src_mem});
      }
      net_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      sum_op_.execute(onednn_stream, net_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  // OneDNN sum primitive creates scale buffer and it crashes in some cases.
  // This is a workaround to make sum primitive as a class member.
  // https://github.com/oneapi-src/oneDNN/blob/rls-v3.1/src/gpu/ocl/many_inputs_sum.hpp#L93
  dnnl::sum sum_op_;
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_ADDN(T)                                 \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnAddN")            \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("inputs_meta") \
                              .HostMemory("sum_meta")    \
                              .TypeConstraint<T>("T"),   \
                          OneDnnAddNOp<GPUDevice, T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ADDN);
#undef REGISTER_ADDN
#else
#define REGISTER_ADDN(T)                                             \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("_OneDnnAddN").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OneDnnAddNOp<CPUDevice, T>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_ADDN);
#undef REGISTER_ADDN
#endif
}  // namespace itex
