#ifndef ITEX_CORE_KERNELS_GPU_XETLA_MHA_OP_H_
#define ITEX_CORE_KERNELS_GPU_XETLA_MHA_OP_H_

#define UseFlashAttention 0

#if UseFlashAttention
#include "itex/core/kernels/gpu/xetla/fmha_forward.h"
#else
#include "itex/core/kernels/gpu/xetla/non_flash_sdp/mha_backward.h"
#include "itex/core/kernels/gpu/xetla/non_flash_sdp/mha_forward.h"
#endif

#include <dlfcn.h>
#include <stdlib.h>

#include "itex/core/kernels/common/fill_functor.h"
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

#if UseFlashAttention
#define CALL_IMPL_FUNC(P)                                                    \
  gpu::xetla::fmha_forward_impl<P, InT, kUseMask, kUseDropout>(              \
      sycl_queue, reinterpret_cast<InT*>(query_ptr),                         \
      reinterpret_cast<InT*>(key_ptr), reinterpret_cast<InT*>(value_ptr),    \
      reinterpret_cast<InT*>(bias_ptr),                                      \
      reinterpret_cast<uint8_t*>(dropout_mask_ptr), dropout_prob,            \
      reinterpret_cast<InT*>(output_ptr), num_batches, num_heads, head_size, \
      num_queries, num_keys)

/// @brief Main execution function for flash mha forward.
template <typename T, bool kUseMask = false, bool kUseDropout = false>
void fmha_forward(OpKernelContext* context, const Tensor& query,
                  const Tensor& key, const Tensor& value,
                  const Tensor& atten_mask, const Tensor& dropout_mask,
                  float dropout_prob, Tensor* output, uint32_t num_batches,
                  uint32_t num_heads, uint32_t head_size, uint32_t num_queries,
                  uint32_t num_keys) {
  ITEX_GPUStream* sycl_queue = context->GetDeviceStream();
  const T* query_ptr = query.template flat<T>().data();
  const T* key_ptr = key.template flat<T>().data();
  const T* value_ptr = value.template flat<T>().data();

  const T* bias_ptr = nullptr;
  const bool* dropout_mask_ptr = nullptr;
  if constexpr (kUseMask) {
    bias_ptr = atten_mask.template flat<T>().data();
  }
  if constexpr (kUseDropout) {
    dropout_mask_ptr = dropout_mask.template flat<bool>().data();
  }

  T* output_ptr = output->template flat<T>().data();

  // TODO(zw) support float ?
  using InT = typename std::conditional<std::is_same<T, Eigen::bfloat16>::value,
                                        gpu::xetla::bf16, sycl::half>::type;

  if (head_size <= 64) {
    if (num_queries == 512 && num_keys == 512) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f512_t512_h64);
    } else if (num_queries == 384 && num_keys == 384) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f384_t384_h64);
    } else if (num_queries == 4096 && num_keys == 4096) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f4096_t4096_h64);
    } else if (num_queries == 4096 && num_keys == 77) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f4096_t77_h64);
    }
  } else if (head_size <= 96) {
    if (num_queries == 1024 && num_keys == 1024) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f1024_t1024_h96);
    } else if (num_queries == 1024 && num_keys == 77) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f1024_t77_h96);
    }
  } else if (head_size <= 128) {
    CALL_IMPL_FUNC(gpu::xetla::fmha_policy_h128);
  } else if (head_size <= 160) {
    if (num_queries == 256 && num_keys == 256) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f256_t256_h160);
    } else if (num_queries == 256 && num_keys == 77) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f256_t77_h160);
    }
    if (num_queries == 64 && num_keys == 64) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f64_t64_h160);
    } else if (num_queries == 64 && num_keys == 77) {
      CALL_IMPL_FUNC(gpu::xetla::fmha_policy_f64_t77_h160);
    }
  } else {
    std::cout << "No policy available for current shape: (b, n, h, f, t): "
              << num_batches << " " << num_heads << " " << head_size << " "
              << num_queries << " " << num_keys << "\n";
    return;
  }
}
#undef CALL_IMPL_FUNC
#else
template <typename T, bool kUseMask = false, bool kUseDropout = false>
void mha_forward(OpKernelContext* context, const Tensor& query,
                 const Tensor& key, const Tensor& value,
                 const Tensor& atten_mask, const Tensor& dropout_mask,
                 float dropout_prob, Tensor* output, Tensor* attn,
                 Tensor* attn_dp, uint32_t num_batches, uint32_t num_heads,
                 uint32_t head_size, uint32_t num_queries, uint32_t num_keys) {
  ITEX_GPUStream* sycl_queue = context->GetDeviceStream();
  const T* query_ptr = query.template flat<T>().data();
  const T* key_ptr = key.template flat<T>().data();
  const T* value_ptr = value.template flat<T>().data();

  const T* atten_mask_ptr = nullptr;
  if constexpr (kUseMask) {
    atten_mask_ptr = atten_mask.template flat<T>().data();
  }

  const bool* dropout_mask_ptr = nullptr;
  const T* attn_ptr = nullptr;
  const T* attn_dp_ptr = nullptr;
  if constexpr (kUseDropout) {
    dropout_mask_ptr = dropout_mask.template flat<bool>().data();
    attn_ptr = attn->template flat<T>().data();
    attn_dp_ptr = attn_dp->template flat<T>().data();
  }

  T* output_ptr = output->template flat<T>().data();

  using InT = typename std::conditional<std::is_same<T, Eigen::bfloat16>::value,
                                        gpu::xetla::bf16, sycl::half>::type;
#define CAST(ptr, src_t, dst_t) \
  reinterpret_cast<dst_t*>(const_cast<src_t*>(ptr))

  if (num_queries <= 512 && head_size <= 128) {
    gpu::xetla::mha::mha_forward_impl<gpu::xetla::mha_policy_32x512x128, InT,
                                      kUseMask, kUseDropout>(
        sycl_queue, CAST(query_ptr, T, InT), CAST(key_ptr, T, InT),
        CAST(value_ptr, T, InT), CAST(atten_mask_ptr, T, InT),
        CAST(dropout_mask_ptr, bool, uint8_t), dropout_prob,
        CAST(output_ptr, T, InT), CAST(attn_ptr, T, InT),
        CAST(attn_dp_ptr, T, InT), num_batches, num_heads, head_size,
        num_queries, num_keys);
  } else {
    std::cout << "No policy available for current shape: (B, N, H, F, T): "
              << num_batches << " " << num_heads << " " << head_size << " "
              << num_queries << " " << num_keys << std::endl;
  }
#undef CAST
}

/// @brief Main execution function for mha backward.
/// Some fuzzy params are listed below.
/// @param attention_probs_dp Is the attention probs after dropout from forward
/// @param grad_score Is used to temporarily store softmax results
template <typename T, bool UseDropout = true>
void mha_backward(OpKernelContext* context, const Tensor& query,
                  const Tensor& key, const Tensor& value,
                  const Tensor& attention_probs,
                  const Tensor& attention_probs_dp, const Tensor& grad_out,
                  const Tensor& dropout_mask, float dropout_prob,
                  Tensor* grad_score, Tensor* grad_query, Tensor* grad_key,
                  Tensor* grad_value, uint32_t num_batches, uint32_t num_heads,
                  uint32_t head_size, uint32_t num_queries, uint32_t num_keys) {
  ITEX_GPUStream* sycl_queue = context->GetDeviceStream();
  const T* query_ptr = query.template flat<T>().data();
  const T* key_ptr = key.template flat<T>().data();
  const T* value_ptr = value.template flat<T>().data();
  const T* attn_ptr = attention_probs.template flat<T>().data();
  const T* attn_dp_ptr = attention_probs_dp.template flat<T>().data();
  const T* grad_ptr = grad_out.template flat<T>().data();

  T* dq_ptr = grad_query->template flat<T>().data();
  T* dk_ptr = grad_key->template flat<T>().data();
  T* dv_ptr = grad_value->template flat<T>().data();

  T* grad_score_ptr = grad_score->template flat<T>().data();

  const bool* dropout_mask_ptr = nullptr;
  if constexpr (UseDropout) {
    dropout_mask_ptr = dropout_mask.template flat<bool>().data();
  }

  using InT = typename std::conditional<std::is_same<T, Eigen::bfloat16>::value,
                                        gpu::xetla::bf16, sycl::half>::type;

#define CAST(ptr, src_t, dst_t) \
  reinterpret_cast<dst_t*>(const_cast<src_t*>(ptr))

  if (num_keys <= 512 && num_queries <= 512 && head_size <= 64) {
    gpu::xetla::mha::mha_backward_impl<gpu::xetla::mha_policy_32x512x128, InT>(
        sycl_queue, CAST(query_ptr, T, InT), CAST(key_ptr, T, InT),
        CAST(value_ptr, T, InT), CAST(attn_ptr, T, InT),
        CAST(attn_dp_ptr, T, InT), CAST(grad_ptr, T, InT),
        CAST(dropout_mask_ptr, bool, uint8_t), dropout_prob,
        CAST(grad_score_ptr, T, InT), CAST(dq_ptr, T, InT),
        CAST(dk_ptr, T, InT), CAST(dv_ptr, T, InT), num_batches, num_heads,
        head_size, num_queries, num_keys);
  } else {
    std::cout << "No policy available for current shape: (B, N, H, F, T): "
              << num_batches << " " << num_heads << " " << head_size << " "
              << num_queries << " " << num_keys << std::endl;
  }
#undef CAST
}
#endif

template <typename Device, typename T>
class ScaledDotProductAttentionOp : public OpKernel {
 public:
  explicit ScaledDotProductAttentionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_mask", &use_mask));
    OP_REQUIRES_OK(context, context->GetAttr("use_dropout", &use_dropout));
    OP_REQUIRES_OK(context, context->GetAttr("dropout_prob", &dropout_prob));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& query = context->input(0);
    const Tensor& key = context->input(1);
    const Tensor& value = context->input(2);
    const Tensor& atten_mask = context->input(3);
    const Tensor& dropout_mask = context->input(4);

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

    Tensor* attn = nullptr;
    Tensor* attn_dp = nullptr;
    if (use_dropout) {
      OP_REQUIRES_OK(context, context->allocate_output(
                                  1, TensorShape({b, n, f, t}), &attn));
      OP_REQUIRES_OK(context, context->allocate_output(
                                  2, TensorShape({b, n, f, t}), &attn_dp));
    }

#if UseFlashAttention
    if (use_dropout) {
      if (use_mask) {
        fmha_forward<T, true, true>(context, query, key, value, atten_mask,
                                    dropout_mask, dropout_prob, output, b, n, h,
                                    f, t);
      } else {
        fmha_forward<T, false, true>(context, query, key, value, atten_mask,
                                     dropout_mask, dropout_prob, output, b, n,
                                     h, f, t);
      }
    } else {
      if (use_mask) {
        fmha_forward<T, true, false>(context, query, key, value, atten_mask,
                                     dropout_mask, dropout_prob, output, b, n,
                                     h, f, t);
      } else {
        fmha_forward<T, false, false>(context, query, key, value, atten_mask,
                                      dropout_mask, dropout_prob, output, b, n,
                                      h, f, t);
      }
    }
#else
    if (use_dropout) {
      if (use_mask) {
        mha_forward<T, true, true>(context, query, key, value, atten_mask,
                                   dropout_mask, dropout_prob, output, attn,
                                   attn_dp, b, n, h, f, t);
      } else {
        mha_forward<T, false, true>(context, query, key, value, atten_mask,
                                    dropout_mask, dropout_prob, output, attn,
                                    attn_dp, b, n, h, f, t);
      }
    } else {
      if (use_mask) {
        mha_forward<T, true, false>(context, query, key, value, atten_mask,
                                    dropout_mask, dropout_prob, output, attn,
                                    attn_dp, b, n, h, f, t);
      } else {
        mha_forward<T, false, false>(context, query, key, value, atten_mask,
                                     dropout_mask, dropout_prob, output, attn,
                                     attn_dp, b, n, h, f, t);
      }
    }
#endif
  };

 private:
  float dropout_prob;
  bool use_mask;
  bool use_dropout;
};

template <typename Device, typename T>
class ScaledDotProductAttentionGradOp : public OpKernel {
 public:
  explicit ScaledDotProductAttentionGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dropout_prob", &dropout_prob));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& query = context->input(0);
    const Tensor& key = context->input(1);
    const Tensor& value = context->input(2);
    const Tensor& dropout_mask = context->input(3);
    const Tensor& atten = context->input(4);
    const Tensor& atten_dp = context->input(5);
    const Tensor& grads = context->input(6);

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

    Tensor grad_score;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                   {b, n, f, t}, &grad_score));

    mha_backward<T>(context, query, key, value, atten, atten_dp, grads,
                    dropout_mask, dropout_prob, &grad_score, q_grad, k_grad,
                    v_grad, b, n, h, f, t);
  }

 private:
  float dropout_prob;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_XETLA_MHA_OP_H_
