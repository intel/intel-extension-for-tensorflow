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

#include "itex/core/ops/shape_inference_fns.h"

#include "itex/core/ops/utils/logging.h"
#include "itex/core/ops/utils/status.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

void empty_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {}

void unchanged_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
  TF_DeleteShapeHandle(handle);
}

void unknown_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeInferenceContextSetUnknownShape(ctx, status);
}

// Below function only work when tensorflow version >= 2.10.0
// as there is a bug in TF_ShapeInferenceContextGetInput and
// TF_ShapeInferenceContextSetOutput when index > 0 in tensorflow
void rnn_forward_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* input_handle = TF_NewShapeHandle();
  TF_ShapeHandle* input_h_handle = TF_NewShapeHandle();
  TF_ShapeHandle* input_c_handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, input_handle, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status));

  TF_ShapeInferenceContextGetInput(ctx, 1, input_h_handle, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status));

  TF_ShapeInferenceContextGetInput(ctx, 2, input_c_handle, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status));

  // input shape: (t, bs, ic)
  TF_ShapeHandle* i_handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextSubshape(ctx, input_handle, 0, 1, i_handle, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status));

  TF_ShapeHandle* output_handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextConcatenateShapes(ctx, i_handle, input_h_handle,
                                            output_handle, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status));
  TF_ShapeInferenceContextSetOutput(ctx, 0, output_handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 1, input_h_handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 2, input_c_handle, status);
  TF_ShapeHandle* unknown_handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextSetOutput(ctx, 3, unknown_handle, status);

  TF_DeleteShapeHandle(input_handle);
  TF_DeleteShapeHandle(input_h_handle);
  TF_DeleteShapeHandle(input_c_handle);
  TF_DeleteShapeHandle(i_handle);
  TF_DeleteShapeHandle(output_handle);
  TF_DeleteShapeHandle(unknown_handle);
}

void layer_norm_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
  TF_DeleteShapeHandle(handle);
  handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 1, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 1, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 2, handle, status);
  TF_DeleteShapeHandle(handle);
}

void layer_norm_grad_shape_fn(TF_ShapeInferenceContext* ctx,
                              TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
  TF_DeleteShapeHandle(handle);
  handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 2, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 1, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 2, handle, status);
  TF_DeleteShapeHandle(handle);
}

void itex_layer_norm_grad_shape_fn(TF_ShapeInferenceContext* ctx,
                                   TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
  TF_DeleteShapeHandle(handle);
  handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 2, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 1, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 2, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 3, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 4, handle, status);
  TF_DeleteShapeHandle(handle);
}

void group_norm_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
  TF_DeleteShapeHandle(handle);
  TF_ShapeHandle* unknown_handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextSetOutput(ctx, 1, unknown_handle,
                                    status);  // (batch_size, group)
  TF_ShapeInferenceContextSetOutput(ctx, 2, unknown_handle,
                                    status);  // (batch_size, group)
  TF_DeleteShapeHandle(unknown_handle);
}

void itex_group_norm_grad_shape_fn(TF_ShapeInferenceContext* ctx,
                                   TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
  TF_DeleteShapeHandle(handle);
  handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 2, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 1, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 2, handle, status);
  TF_DeleteShapeHandle(handle);
}

void apply_adam_with_weight_decay_shape_fn(TF_ShapeInferenceContext* ctx,
                                           TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 10, handle, status);  // grad
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
}

void rotary_embedding_shape_fn(TF_ShapeInferenceContext* ctx,
                               TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* q_handle = TF_NewShapeHandle();
  // [beam*batch, sequence length, embed_dim]
  TF_ShapeInferenceContextGetInput(ctx, 0, q_handle, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status));

  TF_ShapeInferenceContextSetOutput(ctx, 0, q_handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 1, q_handle, status);
  TF_DeleteShapeHandle(q_handle);
}

void scaled_dot_product_attention_inf_shape_fn(TF_ShapeInferenceContext* ctx,
                                               TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* q_handle = TF_NewShapeHandle();
  TF_ShapeHandle* res_handle = TF_NewShapeHandle();
  TF_ShapeHandle* n_handle = TF_NewShapeHandle();
  TF_ShapeHandle* f_handle = TF_NewShapeHandle();
  TF_ShapeHandle* h_handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, q_handle, status);
  TF_ShapeInferenceContextSubshape(ctx, q_handle, 0, 1, res_handle, status);
  TF_ShapeInferenceContextSubshape(ctx, q_handle, 1, 2, n_handle, status);
  TF_ShapeInferenceContextSubshape(ctx, q_handle, 2, 3, f_handle, status);
  TF_ShapeInferenceContextSubshape(ctx, q_handle, 3, 4, h_handle, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status));

  TF_ShapeInferenceContextConcatenateShapes(ctx, res_handle, f_handle,
                                            res_handle, status);
  TF_ShapeInferenceContextConcatenateShapes(ctx, res_handle, n_handle,
                                            res_handle, status);
  TF_ShapeInferenceContextConcatenateShapes(ctx, res_handle, h_handle,
                                            res_handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 0, res_handle, status);
  TF_DeleteShapeHandle(q_handle);
  TF_DeleteShapeHandle(res_handle);
  TF_DeleteShapeHandle(n_handle);
  TF_DeleteShapeHandle(f_handle);
  TF_DeleteShapeHandle(h_handle);
}
