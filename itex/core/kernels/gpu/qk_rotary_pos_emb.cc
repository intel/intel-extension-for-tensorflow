#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct ComputeRotaryPositionalEmbeddingKernel {
  ComputeRotaryPositionalEmbeddingKernel(const T* q, const T* k, const T* sin,
                                         const T* cos, T* output_q, T* output_k,
                                         int Length, int Num_heads,
                                         int head_dim, int rotary_dim,
                                         int total_size)
      : q_(q),
        k_(k),
        sin_(sin),
        cos_(cos),
        output_q_(output_q),
        output_k_(output_k),
        Length_(Length),
        Num_heads_(Num_heads),
        head_dim_(head_dim),
        rotary_dim_(rotary_dim),
        total_size_(total_size) {}
  // total size = q_rotary_size
  // [Batch*beam,length,num_attention_heads(16), head_dim(256)]
  // sin, cos: [Batch*beam,length,1, 64]
  // rotate on first rotary_dim(32 * 2) elements on head_dim
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= total_size_) return;
    // b,l
    int x = id / (Num_heads_ * rotary_dim_);
    int n = (id - x * Num_heads_ * rotary_dim_) / rotary_dim_;
    int idx = id % rotary_dim_;
    // out_q[b,l,n,2*idx] = q[b,l,n,2*idx] * cos[b,l,0,2*idx] - q[b,l,n,2*idx+1]
    // * sin[b,l,0,2*idx] out_q[b,l,n,2*idx+1] = q[b,l,n,2*idx+1] *
    // cos[b,l,0,2*idx+1] + q[b,l,n,2*idx] * sin[b,l,0,2*idx+1] k is the same

    // [b,l,n,2*idx], [b,l,n,2*idx+1]
    int x1 = 2 * idx + n * head_dim_ + x * head_dim_ * Num_heads_;
    int x2 = 1 + 2 * idx + n * head_dim_ + x * head_dim_ * Num_heads_;
    int s_x1 = 2 * idx + x * rotary_dim_ * 2;
    int s_x2 = 1 + 2 * idx + x * rotary_dim_ * 2;
    float tmp_q_x1 = static_cast<float>(q_[x1]);
    float tmp_q_x2 = static_cast<float>(q_[x2]);
    float tmp_k_x1 = static_cast<float>(k_[x1]);
    float tmp_k_x2 = static_cast<float>(k_[x2]);
    float tmp_sin_x1 = static_cast<float>(sin_[s_x1]);
    float tmp_sin_x2 = static_cast<float>(sin_[s_x2]);
    float tmp_cos_x1 = static_cast<float>(cos_[s_x1]);
    float tmp_cos_x2 = static_cast<float>(cos_[s_x2]);
    output_q_[x1] =
        static_cast<T>(tmp_q_x1 * tmp_cos_x1 - tmp_q_x2 * tmp_sin_x1);
    output_q_[x2] =
        static_cast<T>(tmp_q_x2 * tmp_cos_x2 + tmp_q_x1 * tmp_sin_x2);
    output_k_[x1] =
        static_cast<T>(tmp_k_x1 * tmp_cos_x1 - tmp_k_x2 * tmp_sin_x1);
    output_k_[x2] =
        static_cast<T>(tmp_k_x2 * tmp_cos_x2 + tmp_k_x1 * tmp_sin_x2);
  }

 private:
  const T* q_;
  const T* k_;
  const T* sin_;
  const T* cos_;
  T* output_q_;
  T* output_k_;
  const int Length_;
  const int Num_heads_;
  const int head_dim_;
  const int rotary_dim_;
  const int total_size_;
};

template <typename T>
struct ComputeRotaryPositionalEmbeddingKernelBcast {
  ComputeRotaryPositionalEmbeddingKernelBcast(const T* q, const T* k,
                                              const T* sin, const T* cos,
                                              T* output_q, T* output_k, int B,
                                              int Length, int Num_heads,
                                              int head_dim, int rotary_dim,
                                              int total_size)
      : q_(q),
        k_(k),
        sin_(sin),
        cos_(cos),
        output_q_(output_q),
        output_k_(output_k),
        B_(B),
        Length_(Length),
        Num_heads_(Num_heads),
        head_dim_(head_dim),
        rotary_dim_(rotary_dim),
        total_size_(total_size) {}
  // total size = q_rotary_size
  // [Batch*beam,length,num_attention_heads(16), head_dim(256)]
  // sin, cos: [1,length,1, 64]
  // rotate on first rotary_dim(32 * 2) elements on head_dim
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= total_size_) return;
    int x = id / (Num_heads_ * rotary_dim_);
    int n = (id - x * Num_heads_ * rotary_dim_) / rotary_dim_;
    int idx = id % rotary_dim_;
    int l_sin = x / B_;
    // out_q[b,l,n,2*idx] = q[b,l,n,2*idx] * cos[b,l,0,2*idx] - q[b,l,n,2*idx+1]
    // * sin[b,l,0,2*idx] out_q[b,l,n,2*idx+1] = q[b,l,n,2*idx+1] *
    // cos[b,l,0,2*idx+1] + q[b,l,n,2*idx] * sin[b,l,0,2*idx+1] k is the same

    // [b,l,n,2*idx], [b,l,n,2*idx+1]
    int x1 = 2 * idx + n * head_dim_ + x * head_dim_ * Num_heads_;
    int x2 = 1 + 2 * idx + n * head_dim_ + x * head_dim_ * Num_heads_;
    int s_x1 = 2 * idx + l_sin * rotary_dim_ * 2;
    int s_x2 = 1 + 2 * idx + l_sin * rotary_dim_ * 2;
    float tmp_q_x1 = static_cast<float>(q_[x1]);
    float tmp_q_x2 = static_cast<float>(q_[x2]);
    float tmp_k_x1 = static_cast<float>(k_[x1]);
    float tmp_k_x2 = static_cast<float>(k_[x2]);
    float tmp_sin_x1 = static_cast<float>(sin_[s_x1]);
    float tmp_sin_x2 = static_cast<float>(sin_[s_x2]);
    float tmp_cos_x1 = static_cast<float>(cos_[s_x1]);
    float tmp_cos_x2 = static_cast<float>(cos_[s_x2]);
    output_q_[x1] =
        static_cast<T>(tmp_q_x1 * tmp_cos_x1 - tmp_q_x2 * tmp_sin_x1);
    output_q_[x2] =
        static_cast<T>(tmp_q_x2 * tmp_cos_x2 + tmp_q_x1 * tmp_sin_x2);
    output_k_[x1] =
        static_cast<T>(tmp_k_x1 * tmp_cos_x1 - tmp_k_x2 * tmp_sin_x1);
    output_k_[x2] =
        static_cast<T>(tmp_k_x2 * tmp_cos_x2 + tmp_k_x1 * tmp_sin_x2);
  }

 private:
  const T* q_;
  const T* k_;
  const T* sin_;
  const T* cos_;
  T* output_q_;
  T* output_k_;
  const int B_;
  const int Length_;
  const int Num_heads_;
  const int head_dim_;
  const int rotary_dim_;
  const int total_size_;
};

template <typename T>
class RotaryPositionalEmbeddingKernel;

template <typename T>
class RotaryPositionalEmbeddingKernelBcast;

template <typename T>
struct RotaryPositionalEmbedding {
  void operator()(const GPUDevice& d, const T* q, const T* k, const T* sin,
                  const T* cos, T* output_q, T* output_k, OpKernelContext* ctx,
                  bool bcast, int B, int Length, int Num_heads, int head_dim,
                  int rotary_dim, int total_size) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (total_size + total_threads - 1) / total_threads;
    if (bcast) {
      stream->submit([&](sycl::handler& cgh) {
        ComputeRotaryPositionalEmbeddingKernelBcast<T> task(
            q, k, sin, cos, output_q, output_k, B, Length, Num_heads, head_dim,
            rotary_dim, total_size);
        cgh.parallel_for<RotaryPositionalEmbeddingKernelBcast<T>>(
            sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                              sycl::range<1>(total_threads)),
            task);
      });
    } else {
      stream->submit([&](sycl::handler& cgh) {
        ComputeRotaryPositionalEmbeddingKernel<T> task(
            q, k, sin, cos, output_q, output_k, Length, Num_heads, head_dim,
            rotary_dim, total_size);
        cgh.parallel_for<RotaryPositionalEmbeddingKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                              sycl::range<1>(total_threads)),
            task);
      });
    }
  }
};
}  // namespace functor

template <typename Device, typename T>
class QKRotaryPositionalEmbeddingOp : public OpKernel {
 public:
  explicit QKRotaryPositionalEmbeddingOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rotary_dim", &this->rotary_dim_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("num_attention_heads", &this->num_attention_heads_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("head_dim", &this->head_dim_));
  }

  void Compute(OpKernelContext* ctx) override {
    // input q,k: [beam*batch, sequence length, embed_dim(4096)]
    // q,k: reshape[...,num_attention_heads(16), head_dim(256)],
    // apply_rotary_pos_emb

    const Tensor& q = ctx->input(0);
    const Tensor& k = ctx->input(1);
    const Tensor& sin = ctx->input(2);
    const Tensor& cos = ctx->input(3);
    Tensor* output_q = nullptr;
    Tensor* output_k = nullptr;
    ctx->set_output(0, q);
    ctx->set_output(1, k);
    output_q = ctx->mutable_output(0);
    output_k = ctx->mutable_output(1);

    TensorShape q_shape = q.shape();
    int B = q_shape.dim_size(0);  // Batch * Beam
    int Length = q_shape.dim_size(1);
    int rotary_dim_half = rotary_dim_ / 2;

    const Device& device = ctx->template eigen_device<Device>();
    bool bcast = sin.shape().dim_size(0) == 1 && sin.shape().dim_size(0) != B;
    functor::RotaryPositionalEmbedding<T> embedding;
    embedding(device, q.flat<T>().data(), k.flat<T>().data(),
              sin.flat<T>().data(), cos.flat<T>().data(),
              output_q->flat<T>().data(), output_k->flat<T>().data(), ctx,
              bcast, B, Length, num_attention_heads_, head_dim_,
              rotary_dim_half,
              B * Length * num_attention_heads_ * rotary_dim_half);

    output_q->set_shape({B, Length, num_attention_heads_, head_dim_});
    output_k->set_shape({B, Length, num_attention_heads_, head_dim_});
  }

 private:
  int rotary_dim_;
  int num_attention_heads_;
  int head_dim_;
};

#define REGISTER_QK_ROTARY_EMBEDDING(type)                    \
  REGISTER_KERNEL_BUILDER(Name("QKRotaryPositionalEmbedding") \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<type>("T"),     \
                          QKRotaryPositionalEmbeddingOp<GPUDevice, type>)

TF_CALL_float(REGISTER_QK_ROTARY_EMBEDDING);
TF_CALL_half(REGISTER_QK_ROTARY_EMBEDDING);
TF_CALL_bfloat16(REGISTER_QK_ROTARY_EMBEDDING);

#undef REGISTER_QK_ROTARY_EMBEDDING

}  // namespace itex
