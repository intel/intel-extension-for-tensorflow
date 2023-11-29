#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define BEAM_MAX 16

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T, typename Index>
struct ComputeBeamSelectKernel {
  ComputeBeamSelectKernel(const T* cache, const Index* indices, T* output_cache,
                          int batch, int beam, int num_heads, int length,
                          int head_dim, int input_length, int total_size)
      : cache_(cache),
        indices_(indices),
        output_cache_(output_cache),
        batch_(batch),
        beam_(beam),
        num_heads_(num_heads),
        length_(length),
        head_dim_(head_dim),
        input_length_(input_length),
        total_size_(total_size) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= total_size_) return;
    int head_dim_id = id % head_dim_;
    int length_id = id / head_dim_ % (length_ - input_length_);
    int num_heads_id =
        id / (head_dim_ * (length_ - input_length_)) % num_heads_;
    int batch_id = id / (head_dim_ * (length_ - input_length_) * num_heads_);
    T tmp[BEAM_MAX];
#pragma unroll
    for (int beam_id = 0; beam_id < beam_; beam_id++) {
      tmp[beam_id] =
          cache_[head_dim_id + (length_id + input_length_) * head_dim_ +
                 num_heads_id * length_ * head_dim_ +
                 beam_id * num_heads_ * length_ * head_dim_ +
                 batch_id * beam_ * num_heads_ * length_ * head_dim_];
    }
#pragma unroll
    for (int beam_id = 0; beam_id < beam_; beam_id++) {
      output_cache_[head_dim_id + (length_id + input_length_) * head_dim_ +
                    num_heads_id * length_ * head_dim_ +
                    beam_id * num_heads_ * length_ * head_dim_ +
                    batch_id * beam_ * num_heads_ * length_ * head_dim_] =
          tmp[indices_[beam_id + beam_ * batch_id]];
    }
  }

 private:
  const T* cache_;
  const Index* indices_;
  T* output_cache_;
  const int batch_;
  const int beam_;
  const int num_heads_;
  const int length_;
  const int head_dim_;
  const int input_length_;
  const int total_size_;
};

template <typename T, typename Index>
class BeamSelectKernel;

template <typename T, typename Index>
struct BeamSelect {
  void operator()(const GPUDevice& d, const T* cache, const Index* indices,
                  T* output_cache, OpKernelContext* ctx, int batch, int beam,
                  int num_heads, int length, int head_dim, int input_length,
                  int total_size) {
    auto* stream = ctx->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups = (total_size + total_threads - 1) / total_threads;
    stream->submit([&](sycl::handler& cgh) {
      ComputeBeamSelectKernel<T, Index> task(
          cache, indices, output_cache, batch, beam, num_heads, length,
          head_dim, input_length, total_size);
      cgh.parallel_for<BeamSelectKernel<T, Index>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });
  }
};
}  // namespace functor

template <typename Device, typename T, typename Index>
class BeamSelectOp : public OpKernel {
 public:
  explicit BeamSelectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_length", &this->input_length_));
  }

  void Compute(OpKernelContext* ctx) override {
    // input k cache or v cache: [batch, beam, num_heads, sequence length,
    // hidden_size] input beam indices: [batch, beam] always inplace, permute kv
    // cache [:,:,:,input_length,:] according to beam indices.

    const Tensor& cache = ctx->input(0);
    const Tensor& indices = ctx->input(1);

    OP_REQUIRES(ctx, cache.dims() == 5,
                errors::InvalidArgument("cache must be 5-dimensional",
                                        cache.shape().DebugString()));

    OP_REQUIRES(ctx, indices.dims() == 2,
                errors::InvalidArgument("beam indices must be 2-dimensional",
                                        indices.shape().DebugString()));

    Tensor* output_cache = nullptr;
    ctx->set_output(0, cache);
    output_cache = ctx->mutable_output(0);

    TensorShape cache_shape = cache.shape();
    int batch = cache_shape.dim_size(0);
    int beam = cache_shape.dim_size(1);
    int num_heads = cache_shape.dim_size(2);
    int length = cache_shape.dim_size(3);
    int head_dim = cache_shape.dim_size(4);

    OP_REQUIRES(
        ctx, batch == indices.shape().dim_size(0),
        errors::InvalidArgument(
            "First dim of cache and indies must equal to number of batches",
            batch));

    OP_REQUIRES(
        ctx, beam == indices.shape().dim_size(1),
        errors::InvalidArgument(
            "Second dim of cache and indies must equal to number of beams",
            beam));

    OP_REQUIRES(ctx, beam <= BEAM_MAX,
                errors::InvalidArgument("beam number should less or equal to",
                                        BEAM_MAX));

    const Device& device = ctx->template eigen_device<Device>();

    functor::BeamSelect<T, Index> select;
    select(device, cache.flat<T>().data(), indices.flat<Index>().data(),
           output_cache->flat<T>().data(), ctx, batch, beam, num_heads, length,
           head_dim, input_length_,
           batch * (length - input_length_) * num_heads * head_dim);

    output_cache->set_shape({batch, beam, num_heads, length, head_dim});
  }

 private:
  int input_length_;
};

#define REGISTER_BEAM_SELECT(type)                               \
  REGISTER_KERNEL_BUILDER(Name("BeamSelectKVCache")              \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Index"),   \
                          BeamSelectOp<GPUDevice, type, int32>); \
  REGISTER_KERNEL_BUILDER(Name("BeamSelectKVCache")              \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64_t>("Index"), \
                          BeamSelectOp<GPUDevice, type, int64_t>);

TF_CALL_float(REGISTER_BEAM_SELECT);
TF_CALL_half(REGISTER_BEAM_SELECT);
TF_CALL_bfloat16(REGISTER_BEAM_SELECT);

#undef REGISTER_BEAM_SELECT
#undef BEAM_MAX

}  // namespace itex
