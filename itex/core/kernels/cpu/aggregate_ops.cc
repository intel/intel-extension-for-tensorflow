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
#include "itex/core/utils/mutex.h"
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
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {
    ITEX_CHECK_OK(
        ReadBoolFromEnvVar("ITEX_CACHE_ONEDNN_OBJECT", false, &enable_cache_));
    OP_REQUIRES_OK(context, context->GetAttr("N", &num_inputs_));
  }

  TensorShape GetTensorShape(OpKernelContext* context, size_t src_index) {
    const Tensor& src_tensor = context->input(src_index);
    return src_tensor.shape();
  }

  void RunObjectCache(OpKernelContext* context, const dnnl::stream& stream,
                      Tensor* dst_tensor) {
    for (int src_idx = 0; src_idx < num_inputs_; ++src_idx) {
      src_mem_list_[src_idx].set_data_handle(context->tensor_data(src_idx));
    }
    dst_mem_.set_data_handle(
        reinterpret_cast<T*>(GetTensorBuffer<T>(dst_tensor)));
    sum_op_.execute(stream, net_args_);
  }

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(&mu_compute_);

    if (!context->ValidateInputsAreSameShape()) return;
    const size_t kSrc0Idx = 0;
    TensorShape src0_shape = GetTensorShape(context, kSrc0Idx);

    // Try to forward input to output if only got 1 input.
    if (num_inputs_ == 1) {
      context->set_output(0, context->input(0));
      return;
    }

    Tensor* dst_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(kOutputIdx_, src0_shape,
                                                     &dst_tensor));

    // Nothing to compute, return.
    if (src0_shape.num_elements() == 0) {
      return;
    }

    // Fallback to eigen if input is scalar.
    if (src0_shape.dims() == 0) {
      // Try to forward and accumulate the result in one of the input buffers.
      gtl::InlinedVector<int, 8> input_indices(num_inputs_);
      std::iota(input_indices.begin(), input_indices.end(), 0);
      auto To = dst_tensor->flat<T>();

#define I(IDX) context->input(input_indices[IDX]).template flat<T>()

      static const int kWidth = 8;
      int r = num_inputs_ % kWidth;

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
          functor4(context->eigen_device<Device>(), To, I(0), I(1), I(2), I(3));
          break;
        }
        case 5: {
          functor::Add5Functor<Device, T> functor5;
          functor5(context->eigen_device<Device>(), To, I(0), I(1), I(2), I(3),
                   I(4));
          break;
        }
        case 6: {
          functor::Add6Functor<Device, T> functor6;
          functor6(context->eigen_device<Device>(), To, I(0), I(1), I(2), I(3),
                   I(4), I(5));
          break;
        }
        case 7: {
          functor::Add7Functor<Device, T> functor7;
          functor7(context->eigen_device<Device>(), To, I(0), I(1), I(2), I(3),
                   I(4), I(5), I(6));
          break;
        }
        case 0: {
          functor::Add8Functor<Device, T> functor8;
          functor8(context->eigen_device<Device>(), To, I(0), I(1), I(2), I(3),
                   I(4), I(5), I(6), I(7));
          r = 8;
          break;
        }
        case 1: {
          functor::Add9Functor<Device, T> functor9;
          functor9(context->eigen_device<Device>(), To, I(0), I(1), I(2), I(3),
                   I(4), I(5), I(6), I(7), I(8));
          r = 9;
          break;
        }
      }

      for (; r < num_inputs_; r += kWidth) {
        functor::Add8pFunctor<Device, T> functor8p;
        functor8p(context->eigen_device<Device>(), To, I(r), I(r + 1), I(r + 2),
                  I(r + 3), I(r + 4), I(r + 5), I(r + 6), I(r + 7));
      }
#undef I
      return;
    }

    dnnl::engine onednn_engine = CreateDnnlEngine<Device>(*context);
    dnnl::stream onednn_stream = CreateDnnlStream(*context, onednn_engine);

    if (enable_cache_ && is_init_ && context->is_input_same(0, input_dims_)) {
      RunObjectCache(context, onednn_stream, dst_tensor);
      return;
    }

    input_dims_.clear();
    net_args_.clear();
    src_mem_list_.clear();
    for (int i = 0; i < src0_shape.dims(); ++i) {
      input_dims_.push_back(src0_shape.dim_size(i));
    }

    try {
      // Create memory descriptor for OneDnn.
      std::vector<dnnl::memory::desc> srcs_pd;
      for (int src_idx = 0; src_idx < num_inputs_; ++src_idx) {
        dnnl::memory::desc src_md;
        const Tensor& src_tensor = context->input(src_idx);
        auto dims = TFShapeToOneDnnDims(src_tensor.shape());
        src_md = CreatePlainMemDescWithFormatTag<T>(dims);

        srcs_pd.push_back(src_md);
      }

      // Allocate output
      std::vector<float> coeff(num_inputs_, 1.0);
      dnnl::primitive_attr attr;
#ifdef ITEX_ONEDNN_3_0
      auto sum_pd =
          dnnl::sum::primitive_desc(onednn_engine, coeff, srcs_pd, attr);
#else
      auto sum_pd =
          dnnl::sum::primitive_desc(coeff, srcs_pd, onednn_engine, attr);
#endif

      // Create Sum op, and submit for execution.
      sum_op_ = dnnl::sum(sum_pd);
      dst_mem_ = CreateDnnlMemory(sum_pd.dst_desc(), onednn_engine,
                                  GetTensorBuffer<T>(dst_tensor));
      net_args_.insert({DNNL_ARG_DST, dst_mem_});
      for (int src_idx = 0; src_idx < num_inputs_; ++src_idx) {
        auto src_mem =
            CreateDnnlMemory(srcs_pd[src_idx], onednn_engine,
                             GetTensorBuffer<T>(&context->input(src_idx)));
        src_mem_list_.push_back(src_mem);
        net_args_.insert({DNNL_ARG_MULTIPLE_SRC + src_idx, src_mem});
      }
      sum_op_.execute(onednn_stream, net_args_);
      is_init_ = true;
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
  mutex mu_compute_;

  std::vector<int64> input_dims_;
  std::vector<dnnl::memory> src_mem_list_;
  std::unordered_map<int, dnnl::memory> net_args_;

  dnnl::memory dst_mem_;
  dnnl::sum sum_op_;

  bool is_init_ = false;
  bool enable_cache_ = false;

  const size_t kOutputIdx_ = 0;
  int num_inputs_ = 0;
};

#define REGISTER_ADDN(T)                                           \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("_ITEXAddN").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      AddNOp<CPUDevice, T>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_ADDN);
#undef REGISTER_ADDN
}  // namespace itex
