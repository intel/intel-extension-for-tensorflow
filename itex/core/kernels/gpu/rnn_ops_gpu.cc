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

#include "itex/core/kernels/gpu/rnn_ops_gpu.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "itex/core/kernels/gpu/reduction_itex_gpu_kernels.h"
#include "itex/core/kernels/gpu/rnn_ops.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

namespace internal {

// ------------------------------------------------------------------
// UTILS

template <typename T>
void AssignStorage(const RnnModelConfig& rmc, T* workspace, T* scratch,
                   T** gates, T** masked_input, T** masked_h_prev, T** c_states,
                   T** igates, T** hgates_t) {
  // workspace: gates, masked_input, masked_h_prev, c_states (training)
  // scratch: igates, hgates_t (required)
  //          masked_input, masked_h_prev, c_states (inference)
  const int in_size = rmc.num_gates * rmc.batch_size * rmc.input_size;
  const int total_in_size = in_size * rmc.max_seq_length;
  const int out_size = rmc.num_gates * rmc.batch_size * rmc.output_size;
  const int total_out_size = out_size * rmc.max_seq_length;

  // igates: (max_seq_length, num_gates, batch_size, output_size)
  *igates = scratch;
  int scratch_offset = total_out_size;

  // hgates_t: (num_gates, batch_size, output_size)
  *hgates_t = scratch + scratch_offset;
  scratch_offset += out_size;

  if (rmc.is_training) {
    // gates: (max_seq_length, num_gates, batch_size, ouput_size)
    *gates = workspace;
    int workspace_offset = total_out_size;

    if (rmc.HasDpMask()) {
      // masked_input: (max_seq_length, num_gates, batch_size, input_size)
      *masked_input = workspace + workspace_offset;
      workspace_offset += total_in_size;
    }
    if (rmc.HasRecDpMask()) {
      // masked_h_prev: (max_seq_length, num_gates, batch_size, output_size)
      *masked_h_prev = workspace + workspace_offset;
      workspace_offset += total_out_size;
    }
    if (rmc.HasInputC()) {
      // c_states: (max_seq_length, batch_size, cell_size)
      *c_states = workspace + workspace_offset;
    }
  } else {
    if (rmc.HasDpMask()) {
      // masked_input: (max_seq_length, num_gates, batch_size, input_size)
      *masked_input = scratch + scratch_offset;
      scratch_offset += total_in_size;
    }
    if (rmc.HasRecDpMask()) {
      // masked_h_prev: (num_gates, batch_size, output_size)
      *masked_h_prev = scratch + scratch_offset;
      scratch_offset += out_size;
    }
    if (rmc.HasInputC()) {
      // c_states: (2, batch_size, cell_size)
      *c_states = scratch + scratch_offset;
    }
  }
}

// ------------------------------------------------------------------
// LSTM CELL IMPLEMENTATIONS

template <typename T>
struct LSTMCell {
  LSTMCell(const RnnModelConfig& rmc,
           MatMulFunctor<GPUDevice, T, T, T, true>* h_gemm, const T* params,
           T* hgates)
      : num_gates_(rmc.num_gates),
        batch_size_(rmc.batch_size),
        output_size_(rmc.output_size),
        has_rec_dropout_(rmc.HasRecDpMask()),
        is_training_(rmc.is_training),
        w_hh_(params + rmc.input_size * rmc.output_size * rmc.num_gates),
        bias_(w_hh_ + rmc.output_size * rmc.output_size * rmc.num_gates),
        hgates_(hgates),
        h_gemm_(h_gemm) {}

  void operator()(const GPUDevice& d, const T* input, const T* h_prev,
                  const T* c_prev, const T* igates, T* h_next, T* c_next,
                  T* gates) const {
    // compute hgates
    if (has_rec_dropout_) {
      h_gemm_->Compute(const_cast<T*>(h_prev),
                       {num_gates_, batch_size_, output_size_}, false,
                       const_cast<T*>(w_hh_),
                       {num_gates_, output_size_, output_size_}, true,
                       !is_training_, hgates_);
    } else {
      h_gemm_->Compute(const_cast<T*>(h_prev), {1, batch_size_, output_size_},
                       false, const_cast<T*>(w_hh_),
                       {num_gates_, output_size_, output_size_}, true,
                       !is_training_, hgates_);
    }

    // compute h_next and c_next
    if (is_training_) {
      LstmEltwise<T, true>(d, igates, hgates_, bias_, c_prev, h_next, c_next,
                           gates, batch_size_, output_size_);
    } else {
      LstmEltwise<T, false>(d, igates, hgates_, bias_, c_prev, h_next, c_next,
                            gates, batch_size_, output_size_);
    }
  }

 private:
  const int num_gates_;
  const int batch_size_;
  const int output_size_;
  const bool has_rec_dropout_;
  const bool is_training_;
  const T* w_hh_;
  const T* bias_;
  T* hgates_;
  MatMulFunctor<GPUDevice, T, T, T, true>* h_gemm_;
};

// ------------------------------------------------------------------
// LSTM IMPLEMENTATIONS

template <typename T>
void LstmImpl(const GPUDevice& d, const RnnModelConfig& rmc, const T* input,
              const T* hx, const T* cx, const T* params, const T* dp_mask,
              const T* rec_dp_mask, T* output, T* hy, T* cy, T* workspace,
              T* scratch, MatMulFunctor<GPUDevice, T, T, T, true>* input_gemm,
              MatMulFunctor<GPUDevice, T, T, T, true>* h_gemm) {
  // Assign workspace and scratch
  T* gates = nullptr;          // in workspace or null
  T* masked_input = nullptr;   // in workspace (training) or scratch
  T* masked_h_prev = nullptr;  // in workspace (training) or scratch
  T* c_states = nullptr;       // in workspace (training) or scratch
  T* igates = nullptr;         // in scratch
  T* hgates_t = nullptr;       // in scratch
  AssignStorage(rmc, workspace, scratch, &gates, &masked_input, &masked_h_prev,
                &c_states, &igates, &hgates_t);

  const int hidden_stride = rmc.batch_size * rmc.output_size;
  const int cell_stride = rmc.batch_size * rmc.cell_size;
  const int gate_stride = rmc.num_gates * hidden_stride;
  int input_stride = rmc.batch_size * rmc.input_size;

  // Initialize LSTM cell and arguments of its functor
  LSTMCell<T> lstm_cell{rmc, h_gemm, params, hgates_t};

  const T* input_t = input;
  const T* h_prev = hx;
  const T* c_prev = cx;
  const T* igates_t = igates;

  T* h_next = output;
  T* c_next = c_states;
  T* gates_t = gates;

  // Apply the input dropout mask if needed
  if (rmc.HasDpMask()) {
    // Apply the input dropout mask
    ApplyMask(d, input, dp_mask, masked_input, rmc.num_gates,
              rmc.max_seq_length, rmc.batch_size, rmc.input_size);

    // Compute the igates of input at all timesteps
    input_gemm->Compute(
        const_cast<T*>(masked_input),
        {rmc.max_seq_length, rmc.num_gates, rmc.batch_size, rmc.input_size},
        false, const_cast<T*>(params),
        {1, rmc.num_gates, rmc.output_size, rmc.input_size}, true,
        !rmc.is_training, igates);
    input_t = masked_input;
    input_stride *= rmc.num_gates;  // update input stride
  } else {
    // There is no dropout for input
    // Compute the igates of input at all timesteps
    input_gemm->Compute(const_cast<T*>(input),
                        {rmc.max_seq_length, 1, rmc.batch_size, rmc.input_size},
                        false, const_cast<T*>(params),
                        {1, rmc.num_gates, rmc.output_size, rmc.input_size},
                        true, !rmc.is_training, igates);
  }

  // Iterate all the time steps
  for (int i = 0, flag = 1; i < rmc.max_seq_length; ++i) {
    // apply the recurrent dropout mask if needed
    if (rmc.HasRecDpMask()) {
      ApplyMask(d, h_prev, rec_dp_mask, masked_h_prev, rmc.num_gates, 1,
                rmc.batch_size, rmc.output_size);
      h_prev = masked_h_prev;
      // move to next step position
      if (rmc.is_training) {
        masked_h_prev += rmc.num_gates * hidden_stride;
      }
    }

    if (!rmc.is_training && i == rmc.max_seq_length - 1) {
      c_next = cy;
    }

    // Call the cell functor
    lstm_cell(d, input_t, h_prev, c_prev, igates_t, h_next, c_next, gates_t);

    // prepare for the next iteration
    input_t += input_stride;
    h_prev = h_next;
    c_prev = c_next;
    igates_t += gate_stride;
    h_next += hidden_stride;

    if (rmc.is_training) {
      c_next += cell_stride;
      gates_t += gate_stride;
    } else {
      c_next += flag * cell_stride;
      flag *= -1;
    }
  }

  // Copy to hy and cy
  auto stream = d.stream();
  stream->memcpy(hy, output + hidden_stride * (rmc.max_seq_length - 1),
                 hidden_stride * sizeof(T));
  if (rmc.is_training) {
    stream->memcpy(cy, c_states + cell_stride * (rmc.max_seq_length - 1),
                   cell_stride * sizeof(T));
  }
}

}  // namespace internal

namespace functor {

template <typename T>
struct RnnFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const RnnModelConfig& rmc,
                  const Tensor* input, const Tensor* input_h,
                  const Tensor* input_c, const Tensor* params,
                  const Tensor* seq_lengths, const Tensor* dp_mask,
                  const Tensor* rec_dp_mask, Tensor* output, Tensor* output_h,
                  Tensor* output_c, Tensor* workspace,
                  MatMulFunctor<GPUDevice, T, T, T, true>* input_gemm,
                  MatMulFunctor<GPUDevice, T, T, T, true>* h_gemm) {
    // Inputs data
    auto input_data = input->template flat<T>().data();
    auto input_h_data = input_h->template flat<T>().data();
    auto params_data = params->template flat<T>().data();

    const T* input_c_data = nullptr;
    if (rmc.HasInputC()) {
      input_c_data = input_c->template flat<T>().data();
    }

    const T* dp_mask_data = nullptr;
    if (rmc.HasDpMask()) {
      dp_mask_data = dp_mask->template flat<T>().data();
    }

    const T* rec_dp_mask_data = nullptr;
    if (rmc.HasRecDpMask()) {
      rec_dp_mask_data = rec_dp_mask->template flat<T>().data();
    }

    // Allocate temporary storage (scratch):
    // 1. igates: (max_seq_length, num_gates, batch_size, output_size)
    // 2. hgates_t: (num_gates, batch_size, output_size)
    // 3. masked_input (optional):
    //     (max_seq_length, num_gates, batch_size, input_size)
    // 4. masked_h_prev (optional): (num_gates, batch_size, output_size)
    // 5. c_states (optional): (2 * batch_size * cell_size)
    Tensor scratch_t;

    const int in_size = rmc.num_gates * rmc.batch_size * rmc.input_size;
    const int total_in_size = in_size * rmc.max_seq_length;
    const int out_size = rmc.num_gates * rmc.batch_size * rmc.output_size;
    const int total_out_size = out_size * rmc.max_seq_length;

    int scratch_size = total_out_size + out_size;  // igates + hgates_t
    if (!rmc.is_training) {
      if (rmc.HasDpMask()) {
        scratch_size += total_in_size;  // masked_input
      }
      if (rmc.HasRecDpMask()) {
        scratch_size += out_size;  // masked_h_prev
      }
      if (rmc.HasInputC()) {
        scratch_size += 2 * rmc.batch_size * rmc.cell_size;  // c_states
      }
    }
    auto scratch_shape = TensorShape({scratch_size});

    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                   scratch_shape, &scratch_t));
    auto scratch_data = scratch_t.template flat<T>().data();

    // Outputs data
    auto output_data = output->template flat<T>().data();
    auto output_h_data = output_h->template flat<T>().data();
    auto output_c_data = output_c->template flat<T>().data();
    auto workspace_data = workspace->template flat<T>().data();

    if (rmc.rnn_mode == RnnMode::kRnnLstm) {
      if (rmc.var_seq_length) {
        ITEX_LOG(ERROR) << "not implemented";
      } else {
        internal::LstmImpl(context->eigen_device<GPUDevice>(), rmc, input_data,
                           input_h_data, input_c_data, params_data,
                           dp_mask_data, rec_dp_mask_data, output_data,
                           output_h_data, output_c_data, workspace_data,
                           scratch_data, input_gemm, h_gemm);
      }
    }
  }
};

}  // namespace functor

// Instantiate the GPU implementation
#define REGISTER_FUNCTOR(T) template struct functor::RnnFunctor<GPUDevice, T>;

TF_CALL_half(REGISTER_FUNCTOR);
TF_CALL_float(REGISTER_FUNCTOR);
TF_CALL_bfloat16(REGISTER_FUNCTOR);
#undef REGISTER_FUNCTOR

// ------------------------------------------------------------------
// RNN GRADIENT IMPLEMENTATIONS
// ------------------------------------------------------------------

namespace internal {

// ------------------------------------------------------------------
// UTILS

template <typename T>
void AssignWorkspace(const RnnModelConfig& rmc, const T* workspace,
                     const T** gates, const T** masked_input,
                     const T** masked_h_prev, const T** c_state) {
  const int size = rmc.max_seq_length * rmc.num_gates * rmc.batch_size;

  // gates: (max_seq_length, num_gates, batch_size, ouput_size)
  *gates = workspace;
  int offset = size * rmc.output_size;

  if (rmc.HasDpMask()) {
    // masked_input: (max_seq_length, num_gates, batch_size, input_size)
    *masked_input = workspace + offset;
    offset += size * rmc.input_size;
  }
  if (rmc.HasRecDpMask()) {
    // masked_h_prev: (max_seq_length, num_gates, batch_size, output_size)
    *masked_h_prev = workspace + offset;
    offset += size * rmc.output_size;
  }
  if (rmc.HasInputC()) {
    // c_state: (max_seq_length * batch_size * cell_size)
    *c_state = workspace + offset;
  }
}

template <typename T>
void AssignGradStorage1(const RnnModelConfig& rmc, T* scratch, T** gates_grad,
                        T** h_prev_grad, T** dh_prev_grad, T** c_next_grad,
                        T** c_prev_grad) {
  const int out_size = rmc.batch_size * rmc.output_size;
  const int cell_size = rmc.batch_size * rmc.cell_size;

  // gates_grad: (max_seq_length, num_gates, batch_size, output_size)
  *gates_grad = scratch;
  int offset = rmc.max_seq_length * rmc.num_gates * out_size;

  // h_prev_grad: (batch_size, output_size)
  *h_prev_grad = scratch + offset;
  offset += out_size;

  // dh_prev_grad: (num_gates, batch_size, output_size)
  *dh_prev_grad = scratch + offset;
  offset += rmc.num_gates * out_size;

  if (rmc.HasInputC()) {
    // c_next_grad: (batch_size, cell_size)
    *c_next_grad = scratch + offset;
    offset += cell_size;

    // c_prev_grad: (batch_size, cell_size)
    *c_prev_grad = scratch + offset;
  }
}

template <typename T>
void AssignGradStorage2(const RnnModelConfig& rmc, T* scratch, T** h_states,
                        T** dx_grad, T** dw_ih_grad, T** dw_hh_grad,
                        T** dbias_grad) {
  int offset = 0;

  if (!rmc.HasRecDpMask()) {
    // h_states: (max_seq_length, batch_size, output_size)
    *h_states = scratch;
    offset = rmc.max_seq_length * rmc.batch_size * rmc.output_size;
  }

  // Others share the remaining temporary storage
  // dx_grad: (max_seq_length, num_gates, batch_size, input_size)
  *dx_grad = scratch + offset;

  // dw_ih_grad: (max_seq_length, num_gates, input_size, output_size)
  *dw_ih_grad = scratch + offset;

  // dw_hh_grad: (max_seq_length, num_gates, output_size, output_size)
  *dw_hh_grad = scratch + offset;

  // dbias_grad: (max_seq_length, num_gates, output_size)
  *dbias_grad = scratch + offset;
}

// ------------------------------------------------------------------
// LSTM CELL GRAD IMPLEMENTATIONS

template <typename T>
struct LSTMCellGrad {
  LSTMCellGrad(const RnnModelConfig& rmc, const T* params, const T* rec_dp_mask,
               T* h_prev_grad, T* dh_prev_grad)
      : num_gates_(rmc.num_gates),
        batch_size_(rmc.batch_size),
        output_size_(rmc.output_size),
        has_rec_dropout_(rmc.HasRecDpMask()),
        w_hh_(params + rmc.num_gates * rmc.input_size * rmc.output_size),
        rec_dp_mask_(rec_dp_mask),
        h_prev_grad_(h_prev_grad),
        dh_prev_grad_(dh_prev_grad) {}

  void operator()(const GPUDevice& d, const T* c_next, const T* c_prev,
                  const T* gates, const T* output_grad, const T* c_next_grad,
                  T* c_prev_grad, T* gates_grad,
                  MatMulFunctor<GPUDevice, T, T, T, false>* hidden_gemm) const {
    // compute LSTM gates gradients
    LstmGradEltwise(d, c_prev, c_next, output_grad, h_prev_grad_, c_next_grad,
                    gates, c_prev_grad, gates_grad, batch_size_, output_size_);

    // compute dh_prev_grad
    hidden_gemm->Compute(gates_grad, {num_gates_, batch_size_, output_size_},
                         false, const_cast<T*>(w_hh_),
                         {num_gates_, output_size_, output_size_}, false, false,
                         dh_prev_grad_);
    // compute h_prev_grad
    if (has_rec_dropout_) {
      ApplyMaskThenReduce<T, true>(d, dh_prev_grad_, rec_dp_mask_, h_prev_grad_,
                                   1, num_gates_, batch_size_ * output_size_);
    } else {
      ApplyMaskThenReduce<T, false>(d, dh_prev_grad_, rec_dp_mask_,
                                    h_prev_grad_, 1, num_gates_,
                                    batch_size_ * output_size_);
    }
  }

 private:
  const int num_gates_;
  const int batch_size_;
  const int output_size_;
  const bool has_rec_dropout_;
  const T* w_hh_;
  const T* rec_dp_mask_;
  T* h_prev_grad_;
  T* dh_prev_grad_;
};

// ------------------------------------------------------------------
// LSTM GRAD IMPLEMENTATIONS

template <typename T>
void LstmGradImpl(OpKernelContext* context, const RnnModelConfig& rmc,
                  const T* input, const T* hx, const T* cx, const T* params,
                  const T* dp_mask, const T* rec_dp_mask, const T* output,
                  const T* workspace, const T* output_grad, T* input_grad,
                  T* hx_grad, T* cx_grad, T* params_grad, T* scratch1,
                  T* scratch2,
                  MatMulFunctor<GPUDevice, T, T, T, false>* hidden_gemm,
                  MatMulFunctor<GPUDevice, T, T, T, true>* input_gemm,
                  MatMulFunctor<GPUDevice, T, T, T, true>* params_wei_ih_gemm,
                  MatMulFunctor<GPUDevice, T, T, T, true>* params_wei_hh_gemm) {
  auto d = context->eigen_device<GPUDevice>();
  auto stream = d.stream();

  // Assign workspace
  const T* gates_ws = nullptr;
  const T* masked_input_ws = nullptr;
  const T* masked_h_prev_ws = nullptr;
  const T* c_states_ws = nullptr;
  AssignWorkspace(rmc, workspace, &gates_ws, &masked_input_ws,
                  &masked_h_prev_ws, &c_states_ws);

  // Assign scratch1
  T* gates_grad = nullptr;
  T* h_prev_grad = nullptr;
  T* dh_prev_grad = nullptr;
  T* c_next_grad = nullptr;
  T* c_prev_grad = nullptr;
  AssignGradStorage1(rmc, scratch1, &gates_grad, &h_prev_grad, &dh_prev_grad,
                     &c_next_grad, &c_prev_grad);

  const int hidden_stride = rmc.batch_size * rmc.output_size;
  const int cell_stride = rmc.batch_size * rmc.cell_size;
  const int gate_stride = rmc.num_gates * hidden_stride;

  stream->memset(h_prev_grad, 0, hidden_stride * sizeof(T));
  stream->memset(c_next_grad, 0, cell_stride * sizeof(T));

  // Initialize LSTM Gradient Cell and arguments of its functor
  LSTMCellGrad<T> lstm_cell_grad{rmc, params, rec_dp_mask, h_prev_grad,
                                 dh_prev_grad};

  // Iterate all the time steps, compute gates gradients
  for (int i = rmc.max_seq_length - 1; i >= 0; --i) {
    const T* c_next = c_states_ws + i * cell_stride;
    const T* c_prev = (i == 0) ? cx : c_next - cell_stride;
    const T* gates_t = gates_ws + i * gate_stride;
    const T* output_grad_t = output_grad + i * hidden_stride;
    T* gates_grad_t = gates_grad + i * gate_stride;

    lstm_cell_grad(d, c_next, c_prev, gates_t, output_grad_t, c_next_grad,
                   c_prev_grad, gates_grad_t, hidden_gemm);

    std::swap(c_prev_grad, c_next_grad);
  }

  // TODO(itex): maybe remove below copy in next optimization
  //  Copy hx_grad and cx_grad
  stream->memcpy(hx_grad, h_prev_grad, hidden_stride * sizeof(T));
  stream->memcpy(cx_grad, c_next_grad, cell_stride * sizeof(T));

  // ----------------------------------------------------------------
  // Assign scratch2 for the next computations
  T* h_states = nullptr;
  T* dx_grad = nullptr;
  T* dw_ih_grad = nullptr;
  T* dw_hh_grad = nullptr;
  T* dbias_grad = nullptr;
  AssignGradStorage2(rmc, scratch2, &h_states, &dx_grad, &dw_ih_grad,
                     &dw_hh_grad, &dbias_grad);

  // ----------------------------------------------------------------
  // Compute input gradient

  // Compute input gradient deltas (dx_grad)
  const T* w_ih = params;
  input_gemm->Compute(
      gates_grad,
      {rmc.max_seq_length, rmc.num_gates, rmc.batch_size, rmc.output_size},
      false, const_cast<T*>(w_ih),
      {1, rmc.num_gates, rmc.output_size, rmc.input_size}, false, false,
      dx_grad);
  if (rmc.HasDpMask()) {
    ApplyMaskThenReduce<T, true>(d, dx_grad, dp_mask, input_grad,
                                 rmc.max_seq_length, rmc.num_gates,
                                 rmc.batch_size * rmc.input_size);
  } else {
    ApplyMaskThenReduce<T, false>(d, dx_grad, dp_mask, input_grad,
                                  rmc.max_seq_length, rmc.num_gates,
                                  rmc.batch_size * rmc.input_size);
  }

  // ----------------------------------------------------------------
  // Compute w_ih gradient
  T* w_ih_grad = params_grad;

  // Compute w_ih gradient deltas (dw_ih_grad)
  if (rmc.HasDpMask()) {
    params_wei_ih_gemm->Compute(
        gates_grad,
        {rmc.max_seq_length, rmc.num_gates, rmc.batch_size, rmc.output_size},
        true, const_cast<T*>(masked_input_ws),
        {rmc.max_seq_length, rmc.num_gates, rmc.batch_size, rmc.input_size},
        false, false, dw_ih_grad);
  } else {
    params_wei_ih_gemm->Compute(
        gates_grad,
        {rmc.max_seq_length, rmc.num_gates, rmc.batch_size, rmc.output_size},
        true, const_cast<T*>(input),
        {rmc.max_seq_length, 1, rmc.batch_size, rmc.input_size}, false, false,
        dw_ih_grad);
  }
  // always using float as an internal computation type
  LaunchColReduction<T, T, float, sycl::plus<float>>(
      context, dw_ih_grad, w_ih_grad, 0.0f, 1, rmc.max_seq_length,
      rmc.num_gates * rmc.output_size * rmc.input_size, sycl::plus<float>());

  // ----------------------------------------------------------------
  // Compute w_hh gradient
  T* w_hh_grad = params_grad + rmc.num_gates * rmc.input_size * rmc.output_size;

  // Compute w_hh gradient deltas (dw_hh_grad)
  if (rmc.HasRecDpMask()) {
    params_wei_hh_gemm->Compute(
        gates_grad,
        {rmc.max_seq_length, rmc.num_gates, rmc.batch_size, rmc.output_size},
        true, const_cast<T*>(masked_h_prev_ws),
        {rmc.max_seq_length, rmc.num_gates, rmc.batch_size, rmc.output_size},
        false, false, dw_hh_grad);
  } else {
    stream->memcpy(h_states, hx, hidden_stride * sizeof(T));
    stream->memcpy(h_states + hidden_stride, output,
                   (rmc.max_seq_length - 1) * hidden_stride * sizeof(T));
    params_wei_hh_gemm->Compute(
        gates_grad,
        {rmc.max_seq_length, rmc.num_gates, rmc.batch_size, rmc.output_size},
        true, h_states,
        {rmc.max_seq_length, 1, rmc.batch_size, rmc.output_size}, false, false,
        dw_hh_grad);
  }
  // always using float as an internal computation type
  LaunchColReduction<T, T, float, sycl::plus<float>>(
      context, dw_hh_grad, w_hh_grad, 0.0f, 1, rmc.max_seq_length,
      rmc.num_gates * rmc.output_size * rmc.output_size, sycl::plus<float>());

  // ----------------------------------------------------------------
  // TODO(itex): change below two reduction to one kernel in next optimization
  // Compute bias gradient
  T* bias_grad = w_hh_grad + rmc.num_gates * rmc.output_size * rmc.output_size;

  // Compute bias gradient deltas (dbias_grad)
  // always using float as an internal computation type
  LaunchColReduction<T, T, float, sycl::plus<float>>(
      context, gates_grad, dbias_grad, 0.0f, rmc.num_gates * rmc.max_seq_length,
      rmc.batch_size, rmc.output_size, sycl::plus<float>());
  LaunchColReduction<T, T, float, sycl::plus<float>>(
      context, dbias_grad, bias_grad, 0.0f, 1, rmc.max_seq_length,
      rmc.num_gates * rmc.output_size, sycl::plus<float>());
}

}  // namespace internal

namespace functor {

template <typename T>
struct RnnGradFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const RnnModelConfig& rmc,
                  const Tensor* input, const Tensor* input_h,
                  const Tensor* input_c, const Tensor* params,
                  const Tensor* seq_lengths, const Tensor* dp_mask,
                  const Tensor* rec_dp_mask, const Tensor* output,
                  const Tensor* output_h, const Tensor* output_c,
                  const Tensor* workspace, const Tensor* output_backprop,
                  const Tensor* output_h_backprop,
                  const Tensor* output_c_backprop, Tensor* input_backprop,
                  Tensor* input_h_backprop, Tensor* input_c_backprop,
                  Tensor* params_backprop,
                  MatMulFunctor<GPUDevice, T, T, T, false>* hidden_gemm,
                  MatMulFunctor<GPUDevice, T, T, T, true>* input_gemm,
                  MatMulFunctor<GPUDevice, T, T, T, true>* params_wei_ih_gemm,
                  MatMulFunctor<GPUDevice, T, T, T, true>* params_wei_hh_gemm) {
    // Inputs data
    auto input_data = input->template flat<T>().data();
    auto input_h_data = input_h->template flat<T>().data();
    auto params_data = params->template flat<T>().data();
    auto output_data = output->template flat<T>().data();
    auto workspace_data = workspace->template flat<T>().data();
    auto output_backprop_data = output_backprop->template flat<T>().data();

    const T* input_c_data = nullptr;
    const T* output_c_data = nullptr;
    if (rmc.HasInputC()) {
      input_c_data = input_c->template flat<T>().data();
      output_c_data = output_c->template flat<T>().data();
    }

    const T* dp_mask_data = nullptr;
    if (rmc.HasDpMask()) {
      dp_mask_data = dp_mask->template flat<T>().data();
    }

    const T* rec_dp_mask_data = nullptr;
    if (rmc.HasRecDpMask()) {
      rec_dp_mask_data = rec_dp_mask->template flat<T>().data();
    }

    // Allocate temporary storage: scratch1 and scratch2
    // scratch1:
    //  1. gates_grad: (max_seq_length, num_gates, batch_size, output_size)
    //  2. h_prev_grad: (batch_size, output_size)
    //  3. dh_prev_grad: (num_gates, batch_size, output_size)
    //  4. c_next_grad: (batch_size, cell_size)
    //  5. c_prev_grad: (batch_size, cell_size)
    Tensor scratch1_t;

    const int sg_size = rmc.max_seq_length * rmc.num_gates;
    const int out_size = rmc.batch_size * rmc.output_size;
    const int cell_size = rmc.batch_size * rmc.cell_size;

    int scratch1_size = (sg_size + 1 + rmc.num_gates) * out_size;
    if (rmc.HasInputC()) {
      scratch1_size += 2 * cell_size;
    }
    auto scratch1_shape = TensorShape({scratch1_size});
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::v(), scratch1_shape,
                                        &scratch1_t));

    // scratch2: (2-5 can share temporary storage)
    //  1. h_states: (max_seq_length, batch_size, output_size)
    //  2. dx_grad: (max_seq_length, num_gates, batch_size, input_size)
    //  3. dw_ih_grad: (max_seq_length, num_gates, output_size, input_size)
    //  4. dw_hh_grad: (max_seq_length, num_gates, output_size, output_size)
    //  5. dbias_grad: (max_seq_length, num_gates, output_size)
    Tensor scratch2_t;

    const int size2 = rmc.batch_size * rmc.input_size;
    const int size3 = rmc.input_size * rmc.input_size;
    const int size4 = rmc.output_size * rmc.output_size;
    const int max_size = std::max(size2, std::max(size3, size4));

    int scratch2_size = max_size * rmc.max_seq_length * rmc.num_gates;
    if (!rmc.HasRecDpMask()) {
      scratch2_size += rmc.max_seq_length * rmc.batch_size * rmc.output_size;
    }

    auto scratch2_shape = TensorShape({scratch2_size});
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::v(), scratch2_shape,
                                        &scratch2_t));

    auto scratch1_data = scratch1_t.template flat<T>().data();
    auto scratch2_data = scratch2_t.template flat<T>().data();

    // Outputs data
    auto input_backprop_data = input_backprop->template flat<T>().data();
    auto input_h_backprop_data = input_h_backprop->template flat<T>().data();
    auto input_c_backprop_data = input_c_backprop->template flat<T>().data();
    auto params_backprop_data = params_backprop->template flat<T>().data();

    if (rmc.rnn_mode == RnnMode::kRnnLstm) {
      if (rmc.var_seq_length) {
        ITEX_LOG(ERROR) << "not implemented";
      } else {
        internal::LstmGradImpl(
            context, rmc, input_data, input_h_data, input_c_data, params_data,
            dp_mask_data, rec_dp_mask_data, output_data, workspace_data,
            output_backprop_data, input_backprop_data, input_h_backprop_data,
            input_c_backprop_data, params_backprop_data, scratch1_data,
            scratch2_data, hidden_gemm, input_gemm, params_wei_ih_gemm,
            params_wei_hh_gemm);
      }
    }
  }
};

}  // namespace functor

// Instantiate the GPU implementation
#define REGISTER_FUNCTOR(T) \
  template struct functor::RnnGradFunctor<GPUDevice, T>;

TF_CALL_half(REGISTER_FUNCTOR);
TF_CALL_float(REGISTER_FUNCTOR);
TF_CALL_bfloat16(REGISTER_FUNCTOR);
#undef REGISTER_FUNCTOR

}  // namespace itex
