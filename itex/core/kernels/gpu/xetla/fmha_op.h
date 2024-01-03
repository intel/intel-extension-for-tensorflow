#ifndef ITEX_CORE_KERNELS_GPU_XETLA_FMHA_OP_H_
#define ITEX_CORE_KERNELS_GPU_XETLA_FMHA_OP_H_

#include <dlfcn.h>
#include <stdlib.h>

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/kernels/gpu/xetla/fmha_backward.h"
#include "itex/core/kernels/gpu/xetla/fmha_forward.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/common_shape_fns.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/padding.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

/// @brief Main execution function for flash mha forward.
template <typename T, bool kUseBias = false, bool kIsCausal = false,
          bool kIsDropout = false, bool kIsTraining = false>
void fmha_forward(OpKernelContext* context, const Tensor& query,
                  const Tensor& key, const Tensor& value, const Tensor& bias,
                  const Tensor& dropout, float dropout_prob, Tensor* output,
                  Tensor* l, uint32_t num_batches, uint32_t num_heads,
                  uint32_t head_size, uint32_t num_queries, uint32_t num_keys,
                  float head_scale) {
  ITEX_GPUStream* sycl_queue = context->GetDeviceStream();
  const T* query_ptr = query.template flat<T>().data();
  const T* key_ptr = key.template flat<T>().data();
  const T* value_ptr = value.template flat<T>().data();

  const T* bias_ptr = nullptr;
  if constexpr (kUseBias) {
    bias_ptr = bias.template flat<T>().data();
  }
  const bool* dropout_mask_ptr = nullptr;
  if constexpr (kIsDropout) {
    dropout_mask_ptr = dropout.template flat<bool>().data();
  }

  T* output_ptr = output->template flat<T>().data();
  float* l_ptr = nullptr;
  if (l != nullptr) {
    l_ptr = l->template flat<float>().data();
  }
  using InT = typename std::conditional<std::is_same<T, Eigen::bfloat16>::value,
                                        gpu::xetla::bf16, sycl::half>::type;

#define CALL_IMPL_FUNC(P)                                                      \
  gpu::xetla::fmha::fmha_forward_impl<P, InT, kUseBias, kIsCausal, kIsDropout, \
                                      kIsTraining>(                            \
      sycl_queue, reinterpret_cast<InT*>(const_cast<T*>(query_ptr)),           \
      reinterpret_cast<InT*>(const_cast<T*>(key_ptr)),                         \
      reinterpret_cast<InT*>(const_cast<T*>(value_ptr)),                       \
      reinterpret_cast<InT*>(const_cast<T*>(bias_ptr)),                        \
      reinterpret_cast<uint8_t*>(const_cast<bool*>(dropout_mask_ptr)),         \
      dropout_prob, reinterpret_cast<InT*>(output_ptr), l_ptr, num_batches,    \
      num_heads, head_size, num_queries, num_keys, head_scale)

  if (head_size <= 64) {
    CALL_IMPL_FUNC(gpu::xetla::fmha_policy_64x128x64);
  } else if (head_size <= 128) {
    CALL_IMPL_FUNC(gpu::xetla::fmha_policy_64x128x128);
  } else if (head_size <= 256) {
    if (num_keys <= 256) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_32x256x256);
    } else {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_64x512x256);
    }
  } else {
    ITEX_DCHECK(false) << "No policy available for current head_size "
                       << head_size << "\n";
  }
#undef CALL_IMPL_FUNC
}

/// @brief Main execution function for fmha backward.
template <typename T, bool kUseMask = false, bool kUseDropout = true>
void fmha_backward(OpKernelContext* context, const Tensor& query,
                   const Tensor& key, const Tensor& value, const Tensor& out,
                   const Tensor& bias, const Tensor& grad_out,
                   const Tensor& dropout_mask, float dropout_prob,
                   const Tensor& dp_sum, const Tensor& l, Tensor* grad_query,
                   Tensor* grad_query_accum, Tensor* grad_key,
                   Tensor* grad_value, uint32_t num_batches, uint32_t num_heads,
                   uint32_t head_size, uint32_t num_queries,
                   uint32_t num_keys) {
  ITEX_GPUStream* sycl_queue = context->GetDeviceStream();
  const T* query_ptr = query.template flat<T>().data();
  const T* key_ptr = key.template flat<T>().data();
  const T* value_ptr = value.template flat<T>().data();
  const T* out_ptr = out.template flat<T>().data();
  const T* grad_ptr = grad_out.template flat<T>().data();
  float* dp_sum_ptr = const_cast<float*>(dp_sum.template flat<float>().data());
  const float* L_ptr = l.template flat<float>().data();

  T* dq_ptr = grad_query->template flat<T>().data();
  T* dk_ptr = grad_key->template flat<T>().data();
  T* dv_ptr = grad_value->template flat<T>().data();
  float* dq_accum_ptr = grad_query_accum->template flat<float>().data();

  const bool* dropout_mask_ptr = nullptr;
  if constexpr (kUseDropout) {
    dropout_mask_ptr = dropout_mask.template flat<bool>().data();
  }
  const T* bias_ptr = nullptr;
  if constexpr (kUseMask) {
    bias_ptr = bias.template flat<T>().data();
  }

  using InT = typename std::conditional<std::is_same<T, Eigen::bfloat16>::value,
                                        gpu::xetla::bf16, sycl::half>::type;

#define CAST(ptr, src_t, dst_t) \
  reinterpret_cast<dst_t*>(const_cast<src_t*>(ptr))

  if (num_keys <= 512 && num_queries <= 512 && head_size <= 64) {
    gpu::xetla::fmha::fmha_backward_impl<gpu::xetla::fmha_bwd_policy_512x512x64,
                                         InT, kUseMask, kUseDropout>(
        sycl_queue, CAST(query_ptr, T, InT), CAST(key_ptr, T, InT),
        CAST(value_ptr, T, InT), CAST(out_ptr, T, InT), CAST(bias_ptr, T, InT),
        CAST(grad_ptr, T, InT), CAST(dropout_mask_ptr, bool, uint8_t),
        dropout_prob, dp_sum_ptr, const_cast<float*>(L_ptr),
        CAST(dq_ptr, T, InT), dq_accum_ptr, CAST(dk_ptr, T, InT),
        CAST(dv_ptr, T, InT), num_batches, num_heads, head_size, num_queries,
        num_keys);
  } else {
    ITEX_DCHECK(false)
        << "No policy available for current shape: (B, N, H, F, T): "
        << num_batches << " " << num_heads << " " << head_size << " "
        << num_queries << " " << num_keys << std::endl;
  }
#undef CAST
}

template <typename Device, typename T>
class FlashScaledDotProductAttentionOp : public OpKernel {
 public:
  explicit FlashScaledDotProductAttentionOp(OpKernelConstruction* context)
      : OpKernel(context) {
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
    if (use_dropout) dropout_mask = context->input(4);

    const TensorShape& query_shape = query.shape();
    const TensorShape& key_shape = key.shape();

    int b = query_shape.dim_size(0);
    int n = query_shape.dim_size(1);
    int f = query_shape.dim_size(2);
    int h = query_shape.dim_size(3);
    int t = key_shape.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({b, f, n, h}), &output));
    Tensor* l = nullptr;
    if (!is_inference) {
      OP_REQUIRES_OK(context,
                     context->allocate_output(1, TensorShape({b, n, f}), &l));
    }
    BOOL_SWITCH(use_causal, kIsCausal, [&] {
      BOOL_SWITCH(use_mask, kIsMask, [&] {
        BOOL_SWITCH(use_dropout, kIsDropout, [&] {
          BOOL_SWITCH(!is_inference, kIsTraining, [&] {
            fmha_forward<T, kIsMask, kIsCausal, kIsDropout, kIsTraining>(
                context, query, key, value, atten_mask, dropout_mask,
                dropout_prob, output, l, b, n, h, f, t, 1. / sqrt(h));
          });
        });
      });
    });
  };

 private:
  float dropout_prob = 0;
  bool use_dropout = false;
  bool use_mask = false;
  bool use_causal = false;
  bool is_inference = false;
};

template <typename Device, typename T>
class FlashScaledDotProductAttentionGradOp : public OpKernel {
 public:
  explicit FlashScaledDotProductAttentionGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dropout_prob", &dropout_prob));
    OP_REQUIRES_OK(context, context->GetAttr("use_mask", &use_mask));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& query = context->input(0);
    const Tensor& key = context->input(1);
    const Tensor& value = context->input(2);
    const Tensor& out = context->input(3);
    const Tensor& bias = context->input(4);
    const Tensor& grad_out = context->input(5);
    const Tensor& l = context->input(6);
    const Tensor& dropout_mask = context->input(7);

    const TensorShape& query_shape = query.shape();
    const TensorShape& key_shape = key.shape();

    int b = query_shape.dim_size(0);
    int n = query_shape.dim_size(1);
    int f = query_shape.dim_size(2);
    int h = query_shape.dim_size(3);
    int t = key_shape.dim_size(2);

    Tensor* q_grad = nullptr;
    Tensor* k_grad = nullptr;
    Tensor* v_grad = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, query_shape, &q_grad));
    OP_REQUIRES_OK(context, context->allocate_output(1, key_shape, &k_grad));
    OP_REQUIRES_OK(context, context->allocate_output(2, key_shape, &v_grad));

    Tensor grad_query_accum, dp_sum;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {b, n, f, h},
                                                   &grad_query_accum));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_FLOAT, {b, n, f}, &dp_sum));

    BOOL_SWITCH(use_mask, kUseMask, [&] {
      BOOL_SWITCH(dropout_prob > 0.f, kUseDropout, [&] {
        fmha_backward<T, kUseMask, kUseDropout>(
            context, query, key, value, out, bias, grad_out, dropout_mask,
            dropout_prob, dp_sum, l, q_grad, &grad_query_accum, k_grad, v_grad,
            b, n, h, f, t);
      });
    });
  }

 private:
  float dropout_prob = 0;
  bool use_mask = false;
};

#undef BOOL_SWITCH
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_XETLA_FMHA_OP_H_
