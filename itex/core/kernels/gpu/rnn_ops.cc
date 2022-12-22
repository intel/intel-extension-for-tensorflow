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

#include "itex/core/kernels/gpu/rnn_ops.h"

#include <string>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {

using GPUDevice = Eigen::GpuDevice;

namespace {

Status ParseRNNMode(const string& str, RnnMode* rnn_mode) {
  if (str == "rnn_relu") {
    *rnn_mode = RnnMode::kRnnRelu;
  } else if (str == "rnn_tanh") {
    *rnn_mode = RnnMode::kRnnTanh;
  } else if (str == "lstm") {
    *rnn_mode = RnnMode::kRnnLstm;
  } else if (str == "gru") {
    *rnn_mode = RnnMode::kRnnGru;
  } else {
    return errors::InvalidArgument("Invalid RNN mode: ", str);
  }
  return Status::OK();
}

}  // namespace

// ------------------------------------------------------------------
// A common base class for RNN kernels. It extracts common attributes
class RnnCommonKernel : public OpKernel {
 protected:
  RnnModelConfig rmc_;

  explicit RnnCommonKernel(OpKernelConstruction* context) : OpKernel(context) {
    std::string str;
    OP_REQUIRES_OK(context, context->GetAttr("rnn_mode", &str));
    OP_REQUIRES_OK(context, ParseRNNMode(str, &rmc_.rnn_mode));
    OP_REQUIRES_OK(context, context->GetAttr("dropout", &rmc_.dropout));
    OP_REQUIRES_OK(context, context->GetAttr("recurrent_dropout",
                                             &rmc_.recurrent_dropout));
    OP_REQUIRES_OK(context, context->GetAttr("num_proj", &rmc_.num_proj));
    OP_REQUIRES_OK(context,
                   context->GetAttr("var_seq_length", &rmc_.var_seq_length));
  }

  Status ExtractInput(OpKernelContext* context, const Tensor** input,
                      const Tensor** input_h, const Tensor** input_c,
                      const Tensor** params, const Tensor** seq_lengths,
                      const Tensor** dp_mask, const Tensor** rec_dp_mask) {
    TF_RETURN_IF_ERROR(context->input("input", input));
    if ((*input)->dims() != 3) {
      return errors::InvalidArgument("input must be 3-D, got ",
                                     (*input)->shape().DebugString());
    }

    TF_RETURN_IF_ERROR(context->input("input_h", input_h));
    if ((*input_h)->dims() != 2) {
      return errors::InvalidArgument("input_h must be 2-D, got ",
                                     (*input_h)->shape().DebugString());
    }

    if (rmc_.HasInputC()) {
      TF_RETURN_IF_ERROR(context->input("input_c", input_c));
      if ((*input_c)->dims() != 2) {
        return errors::InvalidArgument("input_c must be 2-D, got ",
                                       (*input_c)->shape().DebugString());
      }
    }

    TF_RETURN_IF_ERROR(context->input("params", params));
    if ((*params)->dims() != 1) {
      return errors::InvalidArgument("params must be 1-D, got ",
                                     (*params)->shape().DebugString());
    }

    if (rmc_.var_seq_length) {
      TF_RETURN_IF_ERROR(context->input("sequence_lengths", seq_lengths));
      if ((*seq_lengths)->dims() != 1) {
        return errors::InvalidArgument("sequence_lengths must be 1-D, got ",
                                       (*seq_lengths)->shape().DebugString());
      }
    }

    if (rmc_.HasDpMask()) {
      TF_RETURN_IF_ERROR(context->input("dropout_mask", dp_mask));
      if ((*dp_mask)->dims() != 2) {
        return errors::InvalidArgument("dropout_mask must be 2-D, got ",
                                       (*dp_mask)->shape().DebugString());
      }
    }
    if (rmc_.HasRecDpMask()) {
      TF_RETURN_IF_ERROR(context->input("recurrent_dropout_mask", rec_dp_mask));
      if ((*rec_dp_mask)->dims() != 2) {
        return errors::InvalidArgument(
            "recurrent_dropout_mask must be 2-D, got ",
            (*rec_dp_mask)->shape().DebugString());
      }
    }

    // assign model shapes
    rmc_.max_seq_length = (*input)->dim_size(0);
    rmc_.batch_size = (*input)->dim_size(1);
    rmc_.input_size = (*input)->dim_size(2);
    rmc_.output_size = (*input_h)->dim_size(1);

    rmc_.input_shape = (*input)->shape();
    rmc_.output_shape =
        TensorShape({rmc_.max_seq_length, rmc_.batch_size, rmc_.output_size});

    rmc_.hidden_state_shape = TensorShape({rmc_.batch_size, rmc_.output_size});
    if ((*input_h)->shape() != rmc_.hidden_state_shape) {
      return errors::InvalidArgument(
          "invalid input_h shape: ", (*input_h)->shape().DebugString(),
          "expected: ", rmc_.hidden_state_shape.DebugString());
    }

    if (rmc_.var_seq_length) {
      if ((*seq_lengths)->dim_size(0) != rmc_.batch_size) {
        return errors::InvalidArgument("invalid sequence_lengths size: ",
                                       (*seq_lengths)->shape().DebugString());
      }
    }

    if (rmc_.rnn_mode == RnnMode::kRnnLstm) {
      rmc_.num_gates = 4;
    } else if (rmc_.rnn_mode == RnnMode::kRnnGru) {
      rmc_.num_gates = 3;
    } else {
      rmc_.num_gates = 1;
    }

    rmc_.params_shape = (*params)->shape();
    int params_size = rmc_.num_gates * rmc_.output_size *
                      (rmc_.input_size + rmc_.output_size + 1);
    if ((*params)->NumElements() != params_size) {
      return errors::InvalidArgument(
          "invalid params shape size: ", (*params)->shape().DebugString(),
          "expected: ", params_size);
    }

    if (rmc_.HasInputC()) {
      rmc_.cell_size = (*input_c)->dim_size(1);
      rmc_.cell_state_shape = (*input_c)->shape();
      if (rmc_.num_proj == 0) {
        if ((*input_h)->shape() != (*input_c)->shape()) {
          return errors::InvalidArgument(
              "input_h and input_c must have the same shape ",
              (*input_h)->shape().DebugString(), " ",
              (*input_c)->shape().DebugString());
        }
      } else {
        if ((*input_h)->dim_size(0) != (*input_c)->dim_size(0) ||
            (*input_h)->dim_size(1) > (*input_c)->dim_size(1) ||
            rmc_.num_proj != (*input_h)->dim_size(1)) {
          return errors::InvalidArgument(
              "invalid input_h and input_c w/ projection size: ", rmc_.num_proj,
              " ", (*input_h)->shape().DebugString(), " ",
              (*input_c)->shape().DebugString());
        }
      }
    } else {
      // dummy cell_state_shape
      rmc_.cell_size = 0;
      rmc_.cell_state_shape = TensorShape({});
    }

    if (rmc_.is_training) {
      // workspace structure (training):
      // 1. gates: (max_seq_length, num_gates, batch_size, ouput_size)
      // 2. masked_input: (max_seq_length, num_gates, batch_size, input_size)
      // 3. masked_h_prev: (max_seq_length, num_gates, batch_size, output_size)
      // 4. c_states: (max_seq_length, batch_size, cell_size)
      const int ss = rmc_.max_seq_length * rmc_.num_gates * rmc_.batch_size;
      int size = ss * rmc_.output_size;
      if (rmc_.HasDpMask()) {
        size += ss * rmc_.input_size;
      }
      if (rmc_.HasRecDpMask()) {
        size += ss * rmc_.output_size;
      }
      if (rmc_.HasInputC()) {
        size += rmc_.max_seq_length * rmc_.batch_size * rmc_.cell_size;
      }
      rmc_.workspace_shape = TensorShape({size});
    } else {
      rmc_.workspace_shape = TensorShape({});
    }

    return Status::OK();
  }
};

// ------------------------------------------------------------------
// RNN OP
// ------------------------------------------------------------------
template <typename Device, typename T>
class RnnOp : public RnnCommonKernel {
 public:
  explicit RnnOp(OpKernelConstruction* context) : RnnCommonKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &rmc_.is_training));
  }

  void Compute(OpKernelContext* context) override {
    // Extract inputs
    const Tensor* input = nullptr;
    const Tensor* input_h = nullptr;
    const Tensor* input_c = nullptr;
    const Tensor* params = nullptr;
    const Tensor* seq_lengths = nullptr;
    const Tensor* dp_mask = nullptr;
    const Tensor* rec_dp_mask = nullptr;
    OP_REQUIRES_OK(context,
                   ExtractInput(context, &input, &input_h, &input_c, &params,
                                &seq_lengths, &dp_mask, &rec_dp_mask));
    // printf("debug: %s", rmc_.DebugString().c_str());

    // Allocate outputs
    Tensor* output = nullptr;
    Tensor* output_h = nullptr;
    Tensor* output_c = nullptr;
    Tensor* workspace = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, rmc_.output_shape, &output));
    OP_REQUIRES_OK(context, context->allocate_output(1, rmc_.hidden_state_shape,
                                                     &output_h));
    OP_REQUIRES_OK(
        context, context->allocate_output(2, rmc_.cell_state_shape, &output_c));
    OP_REQUIRES_OK(
        context, context->allocate_output(3, rmc_.workspace_shape, &workspace));

    // Call RNN functor
    input_gemm_.SetContext(context);
    h_gemm_.SetContext(context);
    functor::RnnFunctor<Device, T> func;
    func(context, rmc_, input, input_h, input_c, params, seq_lengths, dp_mask,
         rec_dp_mask, output, output_h, output_c, workspace, &input_gemm_,
         &h_gemm_);
  }
  MatMulFunctor<Device, T, T, T, true> input_gemm_;
  MatMulFunctor<Device, T, T, T, true> h_gemm_;
};

// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void RnnFunctor<GPUDevice, T>::operator()(                                  \
      OpKernelContext* context, const RnnModelConfig& model_config,           \
      const Tensor* input, const Tensor* input_h, const Tensor* input_c,      \
      const Tensor* params, const Tensor* seq_lengths, const Tensor* dp_mask, \
      const Tensor* rec_dp_mask, Tensor* output, Tensor* output_h,            \
      Tensor* output_c, Tensor* workspace,                                    \
      MatMulFunctor<GPUDevice, T, T, T, true>* input_gemm,                    \
      MatMulFunctor<GPUDevice, T, T, T, true>* h_gemm);                       \
  extern template struct RnnFunctor<GPUDevice, T>;

TF_CALL_half(DECLARE_GPU_SPEC);
TF_CALL_float(DECLARE_GPU_SPEC);
TF_CALL_bfloat16(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC

}  // namespace functor

#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ItexRnn").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RnnOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_bfloat16(REGISTER_GPU);
#undef REGISTER_GPU

// ------------------------------------------------------------------
// RNN GRADIENT OP
// ------------------------------------------------------------------
template <typename Device, typename T>
class RnnGradOp : public RnnCommonKernel {
 public:
  explicit RnnGradOp(OpKernelConstruction* context) : RnnCommonKernel(context) {
    rmc_.is_training = true;
  }

  void Compute(OpKernelContext* context) override {
    // Extract inputs
    const Tensor* input = nullptr;
    const Tensor* input_h = nullptr;
    const Tensor* input_c = nullptr;
    const Tensor* params = nullptr;
    const Tensor* seq_lengths = nullptr;
    const Tensor* dp_mask = nullptr;
    const Tensor* rec_dp_mask = nullptr;
    OP_REQUIRES_OK(context,
                   ExtractInput(context, &input, &input_h, &input_c, &params,
                                &seq_lengths, &dp_mask, &rec_dp_mask));
    // printf("%s", rmc_.DebugString().c_str());

    // Extract gradient inputs
    const Tensor* output = nullptr;
    const Tensor* output_h = nullptr;
    const Tensor* output_c = nullptr;
    const Tensor* workspace = nullptr;
    const Tensor* output_backprop = nullptr;
    const Tensor* output_h_backprop = nullptr;
    const Tensor* output_c_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   ExtractGradInputs(context, &output, &output_h, &output_c,
                                     &workspace, &output_backprop,
                                     &output_h_backprop, &output_c_backprop));

    // Allocate outputs
    Tensor* input_backprop = nullptr;
    Tensor* input_h_backprop = nullptr;
    Tensor* input_c_backprop = nullptr;
    Tensor* params_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, rmc_.input_shape,
                                                     &input_backprop));
    OP_REQUIRES_OK(context, context->allocate_output(1, rmc_.hidden_state_shape,
                                                     &input_h_backprop));
    OP_REQUIRES_OK(context, context->allocate_output(2, rmc_.cell_state_shape,
                                                     &input_c_backprop));
    OP_REQUIRES_OK(context, context->allocate_output(3, rmc_.params_shape,
                                                     &params_backprop));
    // Call RNN gradient functor
    functor::RnnGradFunctor<Device, T> func;
    hidden_gemm_.SetContext(context);
    input_gemm_.SetContext(context);
    params_wei_ih_gemm_.SetContext(context);
    params_wei_hh_gemm_.SetContext(context);
    func(context, rmc_, input, input_h, input_c, params, seq_lengths, dp_mask,
         rec_dp_mask, output, output_h, output_c, workspace, output_backprop,
         output_h_backprop, output_c_backprop, input_backprop, input_h_backprop,
         input_c_backprop, params_backprop, &hidden_gemm_, &input_gemm_,
         &params_wei_ih_gemm_, &params_wei_hh_gemm_);
  }

 private:
  Status ExtractGradInputs(OpKernelContext* context, const Tensor** output,
                           const Tensor** output_h, const Tensor** output_c,
                           const Tensor** workspace,
                           const Tensor** output_backprop,
                           const Tensor** output_h_backprop,
                           const Tensor** output_c_backprop) {
    TF_RETURN_IF_ERROR(context->input("output", output));
    TF_RETURN_IF_ERROR(context->input("output_backprop", output_backprop));
    TF_RETURN_IF_ERROR(context->input("output_h", output_h));
    TF_RETURN_IF_ERROR(context->input("output_h_backprop", output_h_backprop));
    if (rmc_.HasInputC()) {
      TF_RETURN_IF_ERROR(context->input("output_c", output_c));
      TF_RETURN_IF_ERROR(
          context->input("output_c_backprop", output_c_backprop));
    }
    TF_RETURN_IF_ERROR(context->input("workspace", workspace));

    if ((*output)->shape() != rmc_.output_shape) {
      return errors::InvalidArgument("Invalid output shape, got ",
                                     (*output)->shape().DebugString());
    }

    if ((*output_backprop)->shape() != rmc_.output_shape) {
      return errors::InvalidArgument("Invalid output_backprop shape, got ",
                                     (*output_backprop)->shape().DebugString());
    }

    if ((*output_h)->shape() != rmc_.hidden_state_shape) {
      return errors::InvalidArgument("Invalid output_h shape, got ",
                                     (*output_h)->shape().DebugString());
    }

    if ((*output_h_backprop)->shape() != rmc_.hidden_state_shape) {
      return errors::InvalidArgument(
          "Invalid output_h_backprop shape, got ",
          (*output_h_backprop)->shape().DebugString());
    }

    if (rmc_.HasInputC()) {
      if ((*output_c)->shape() != rmc_.cell_state_shape) {
        return errors::InvalidArgument("Invalid output_c shape, got ",
                                       (*output_c)->shape().DebugString());
      }
      if ((*output_c_backprop)->shape() != rmc_.cell_state_shape) {
        return errors::InvalidArgument(
            "Invalid output_c_backprop shape, got ",
            (*output_c_backprop)->shape().DebugString());
      }
    }

    if ((*workspace)->shape() != rmc_.workspace_shape) {
      return errors::InvalidArgument(
          "Invalid workspace shape, got ", (*workspace)->shape().DebugString(),
          " expected: ", rmc_.workspace_shape.DebugString());
    }

    return Status::OK();
  }

  MatMulFunctor<Device, T, T, T, false> hidden_gemm_;
  MatMulFunctor<Device, T, T, T, true> input_gemm_;
  MatMulFunctor<Device, T, T, T, true> params_wei_ih_gemm_;
  MatMulFunctor<Device, T, T, T, true> params_wei_hh_gemm_;
};

// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void RnnGradFunctor<GPUDevice, T>::operator()(                               \
      OpKernelContext* context, const RnnModelConfig& model_config,            \
      const Tensor* input, const Tensor* input_h, const Tensor* input_c,       \
      const Tensor* params, const Tensor* seq_lengths, const Tensor* dp_mask,  \
      const Tensor* rec_dp_mask, const Tensor* output, const Tensor* output_h, \
      const Tensor* output_c, const Tensor* workspace,                         \
      const Tensor* output_backprop, const Tensor* output_h_backprop,          \
      const Tensor* output_c_backprop, Tensor* input_backprop,                 \
      Tensor* input_h_backprop, Tensor* input_c_backprop,                      \
      Tensor* params_backprop, MatMulFunctor<GPUDevice, T, T, T, false>*,      \
      MatMulFunctor<GPUDevice, T, T, T, true>*,                                \
      MatMulFunctor<GPUDevice, T, T, T, true>*,                                \
      MatMulFunctor<GPUDevice, T, T, T, true>*);                               \
  extern template struct RnnGradFunctor<GPUDevice, T>;

TF_CALL_half(DECLARE_GPU_SPEC);
TF_CALL_float(DECLARE_GPU_SPEC);
TF_CALL_bfloat16(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC

}  // namespace functor

#define REGISTER_GPU(T)                                              \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ItexRnnGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RnnGradOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_bfloat16(REGISTER_GPU);
#undef REGISTER_GPU

}  // namespace itex
