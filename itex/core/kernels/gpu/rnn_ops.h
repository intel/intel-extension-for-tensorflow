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

#ifndef ITEX_CORE_KERNELS_GPU_RNN_OPS_H_
#define ITEX_CORE_KERNELS_GPU_RNN_OPS_H_

#include <string>

#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/stringprintf.h"
#include "itex/core/utils/tensor_types.h"

namespace itex {

enum class RnnMode {
  kRnnRelu = 0,
  kRnnTanh = 1,
  kRnnLstm = 2,
  kRnnGru = 3,
};

struct RnnModelConfig {
  // input attribute
  RnnMode rnn_mode;
  float dropout;
  float recurrent_dropout;
  int num_proj;
  bool var_seq_length;
  bool is_training;

  // model shapes
  int max_seq_length;
  int batch_size;
  int input_size;
  int output_size;
  int cell_size;
  int num_gates;
  TensorShape input_shape;
  TensorShape output_shape;
  TensorShape hidden_state_shape;
  TensorShape cell_state_shape;
  TensorShape params_shape;
  TensorShape workspace_shape;

  bool HasInputC() const { return rnn_mode == RnnMode::kRnnLstm; }
  bool HasDpMask() const { return dropout > 0 && dropout < 1; }
  bool HasRecDpMask() const {
    return recurrent_dropout > 0 && recurrent_dropout < 1;
  }

  string DebugString() const {
    return strings::Printf(
        "rnn_mode: %d, dropout: %f, recurrent_dropout: %f, num_proj: %d, "
        "var_seq_length: %d, is_training: %d\n"
        "seq_length: %d, batch_size: %d, input_size: %d, output size: %d, "
        "cell_size: %d, num_gates: %d\n",
        rnn_mode, dropout, recurrent_dropout, num_proj, var_seq_length,
        is_training, max_seq_length, batch_size, input_size, output_size,
        cell_size, num_gates);
  }
};

namespace functor {

template <typename Device, typename T>
struct RnnFunctor {
  void operator()(OpKernelContext* context, const RnnModelConfig& model_config,
                  const Tensor* input, const Tensor* input_h,
                  const Tensor* input_c, const Tensor* params,
                  const Tensor* seq_lengths, const Tensor* dp_mask,
                  const Tensor* rec_dp_mask, Tensor* output, Tensor* output_h,
                  Tensor* output_c, Tensor* workspace);
};

template <typename Device, typename T>
struct RnnGradFunctor {
  void operator()(OpKernelContext* context, const RnnModelConfig& model_config,
                  const Tensor* input, const Tensor* input_h,
                  const Tensor* input_c, const Tensor* params,
                  const Tensor* seq_lengths, const Tensor* dp_mask,
                  const Tensor* rec_dp_mask, const Tensor* output,
                  const Tensor* output_h, const Tensor* output_c,
                  const Tensor* workspace, const Tensor* output_backprop,
                  const Tensor* output_h_backprop,
                  const Tensor* output_c_backprop, Tensor* input_backprop,
                  Tensor* input_h_backprop, Tensor* input_c_backprop,
                  Tensor* params_backprop);
};

}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_RNN_OPS_H_
