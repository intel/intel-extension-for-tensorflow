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

#ifndef ITEX_CORE_OPS_SHAPE_INFERENCE_FNS_H_
#define ITEX_CORE_OPS_SHAPE_INFERENCE_FNS_H_
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

#ifdef __cplusplus
extern "C" {
#endif
void empty_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status);
void unchanged_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status);
void unknown_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status);
void rnn_forward_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status);

void layer_norm_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status);
void layer_norm_grad_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status);
void itex_layer_norm_grad_shape_fn(TF_ShapeInferenceContext* ctx,
                                   TF_Status* status);

void apply_adam_with_weight_decay_shape_fn(TF_ShapeInferenceContext* ctx,
                                           TF_Status* status);
void rotary_embedding_shape_fn(TF_ShapeInferenceContext* ctx,
                               TF_Status* status);
#ifdef __cplusplus
}
#endif

#endif  // ITEX_CORE_OPS_SHAPE_INFERENCE_FNS_H_
