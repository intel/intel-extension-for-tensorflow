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

#include "itex/core/kernels/cpu/mha_op.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/utils/env_var.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename T>
class MHAOp : public OpKernel {
 public:
  explicit MHAOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("is_inference", &is_inference));
    if (!is_inference) {
      OP_REQUIRES_OK(context, context->GetAttr("use_dropout", &use_dropout));
      OP_REQUIRES_OK(context, context->GetAttr("dropout_prob", &dropout_prob));
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("use_causal", &use_causal));
    }
    OP_REQUIRES_OK(context, context->GetAttr("use_mask", &use_mask));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& query = context->input(0);
    const Tensor& key = context->input(1);
    const Tensor& value = context->input(2);
    Tensor atten_mask;
    if (use_mask) atten_mask = context->input(3);
    Tensor dropout_mask;
    if (!is_inference) dropout_mask = context->input(4);

    int64_t batch_size = query.dim_size(0);
    int64_t num_heads = query.dim_size(1);
    int64_t q_seq_len = query.dim_size(2);
    int64_t head_size = query.dim_size(3);
    int64_t k_seq_len = key.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, {batch_size, q_seq_len, num_heads, head_size}, &output));

#define CALL_FMHA_FUNC(T, qSplitSize, kvSplitSize)                            \
  FmhaFunctor<T, qSplitSize, kvSplitSize>()(                                  \
      query, key, value, batch_size, q_seq_len, num_heads, head_size,         \
      k_seq_len, use_mask, use_causal, use_dropout, atten_mask, dropout_mask, \
      dropout_prob, output)

    if (q_seq_len >= 768) {
      CALL_FMHA_FUNC(T, 256, 512);
    } else if (q_seq_len >= 192) {
      CALL_FMHA_FUNC(T, 64, 512);
    } else {
      CALL_FMHA_FUNC(T, 32, 512);
    }
  }

 private:
  float dropout_prob = 0;
  bool use_mask = false;
  bool use_causal = false;
  bool use_dropout = false;
  bool is_inference = false;
};

#define REGISTER_MHA_INF_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("ScaledDotProductAttentionInference") \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<type>("T"),            \
                          MHAOp<type>);

REGISTER_MHA_INF_CPU(Eigen::bfloat16);
REGISTER_MHA_INF_CPU(float);
#undef REGISTER_MHA_INF_GPU

}  // namespace itex
