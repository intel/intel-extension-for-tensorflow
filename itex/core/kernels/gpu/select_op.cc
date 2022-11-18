/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/common/cwise_ops_common.h"
#include "itex/core/kernels/gpu/cwise_op.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

// if cond is scalar, don't need to load then_flat and else_flat at the same
// time.
template <typename T, int vec_size>
struct SelectScalarGpuKernel {
  using Tvec = typename BaseTypeVectorize<T, vec_size>::type;
  SelectScalarGpuKernel(const bool* cond, const T* then_flat,
                        const T* else_flat, T* out, int num_work_items,
                        int vectorized_items, int vectorized_range)
      : cond_(cond),
        then_flat_(then_flat),
        else_flat_(else_flat),
        out_(out),
        num_work_items_(num_work_items),
        vectorized_items_(vectorized_items),
        vectorized_range_(vectorized_range) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= num_work_items_) {
      return;
    }

    bool selector = cond_[0];
    if (id < vectorized_items_) {
      auto out_id = id * vec_size;
      *(reinterpret_cast<Tvec*>(out_ + out_id)) =
          selector ? *(reinterpret_cast<const Tvec*>(then_flat_ + out_id))
                   : *(reinterpret_cast<const Tvec*>(else_flat_ + out_id));
    } else {
      auto out_id = vectorized_range_ + (id - vectorized_items_);
      out_[out_id] = selector ? then_flat_[out_id] : else_flat_[out_id];
    }
  }

 private:
  const bool* cond_;
  const T* then_flat_;
  const T* else_flat_;
  T* out_;
  int num_work_items_;
  int vectorized_items_;
  int vectorized_range_;
};

namespace functor {

template <typename T, int NDIMS>
struct BCastSelectFunctor<GPUDevice, T, NDIMS> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T, NDIMS>::Tensor output_tensor,
                  typename TTypes<bool, NDIMS>::ConstTensor cond_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor then_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor else_tensor,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> cond_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> then_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> else_bcast) {
    const bool then_bcast_all_one = AllOne<NDIMS>(then_bcast);
    const bool else_bcast_all_one = AllOne<NDIMS>(else_bcast);

#define KERNEL_INT_TYPE(IntTypePattern)                                        \
  if (then_bcast_all_one && else_bcast_all_one) {                              \
    IntTypePattern(output_tensor).device(d) =                                  \
        IntTypePattern(cond_tensor)                                            \
            .broadcast(cond_bcast)                                             \
            .select(IntTypePattern(then_tensor), IntTypePattern(else_tensor)); \
  } else if (then_bcast_all_one) {                                             \
    IntTypePattern(output_tensor).device(d) =                                  \
        IntTypePattern(cond_tensor)                                            \
            .broadcast(cond_bcast)                                             \
            .select(IntTypePattern(then_tensor),                               \
                    IntTypePattern(else_tensor).broadcast(else_bcast));        \
  } else if (else_bcast_all_one) {                                             \
    IntTypePattern(output_tensor).device(d) =                                  \
        IntTypePattern(cond_tensor)                                            \
            .broadcast(cond_bcast)                                             \
            .select(IntTypePattern(then_tensor).broadcast(then_bcast),         \
                    IntTypePattern(else_tensor));                              \
  } else {                                                                     \
    IntTypePattern(output_tensor).device(d) =                                  \
        IntTypePattern(cond_tensor)                                            \
            .broadcast(cond_bcast)                                             \
            .select(IntTypePattern(then_tensor).broadcast(then_bcast),         \
                    IntTypePattern(else_tensor).broadcast(else_bcast));        \
  }

    KERNEL_INT_TYPE(To32Bit);

#undef KERNEL_INT_TYPE
  }
};

template <typename T>
struct SelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    To32Bit(out).device(d) =
        To32Bit(cond_flat).select(To32Bit(then_flat), To32Bit(else_flat));
  }
};

template <typename T>
struct SelectScalarFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstScalar cond,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    constexpr int bytes_num = 16;
    constexpr int vec_size = bytes_num / sizeof(T);
    int out_elements = out.size();
    int vectorized_items = out_elements / vec_size;
    int vectorized_range = vectorized_items * vec_size;
    int num_work_items = vectorized_items + (out_elements - vectorized_range);

    auto& stream = d.stream();
    auto workgroup_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups =
        (num_work_items + workgroup_size - 1) / workgroup_size;

    stream->submit([&](sycl::handler& cgh) {
      SelectScalarGpuKernel<T, vec_size> task(
          cond.data(), then_flat.data(), else_flat.data(), out.data(),
          num_work_items, vectorized_items, vectorized_range);
      cgh.parallel_for<SelectScalarGpuKernel<T, vec_size>>(
          sycl::nd_range<1>(sycl::range<1>(num_workgroups * workgroup_size),
                            sycl::range<1>(workgroup_size)),
          task);
    });
  }
};

template <typename T>
struct BatchSelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims) {
    const int batch = cond_vec.size();
    const int all_but_batch = then_flat_outer_dims.dimension(1);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 2> broadcast_dims{{ 1, all_but_batch }};
    Eigen::Tensor<int, 2>::Dimensions reshape_dims{{ batch, 1 }};
#else
    Eigen::IndexList<Eigen::type2index<1>, int> broadcast_dims;
    broadcast_dims.set(1, all_but_batch);
    Eigen::IndexList<int, Eigen::type2index<1>> reshape_dims;
    reshape_dims.set(0, batch);
#endif

    output_flat_outer_dims.device(d) =
        cond_vec.reshape(reshape_dims)
            .broadcast(broadcast_dims)
            .select(then_flat_outer_dims, else_flat_outer_dims);
  }
};

#define SELECT_FUNCTOR(T)                              \
  template struct SelectFunctor<GPUDevice, T>;         \
  template struct SelectScalarFunctor<GPUDevice, T>;   \
  template struct BatchSelectFunctor<GPUDevice, T>;    \
  template struct BCastSelectFunctor<GPUDevice, T, 1>; \
  template struct BCastSelectFunctor<GPUDevice, T, 2>; \
  template struct BCastSelectFunctor<GPUDevice, T, 3>; \
  template struct BCastSelectFunctor<GPUDevice, T, 4>; \
  template struct BCastSelectFunctor<GPUDevice, T, 5>; \
  template struct BCastSelectFunctor<GPUDevice, T, 6>; \
  template struct BCastSelectFunctor<GPUDevice, T, 7>; \
  template struct BCastSelectFunctor<GPUDevice, T, 8>;

SELECT_FUNCTOR(bool);
SELECT_FUNCTOR(int32);
SELECT_FUNCTOR(int64);
TF_CALL_GPU_NUMBER_TYPES(SELECT_FUNCTOR);
#undef SELECT_FUNCTOR

template <typename T>
struct SelectScalarHandler {
  void operator()(OpKernelContext* ctx, const Tensor* cond, const Tensor* then,
                  const Tensor* else_) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {1, 2}, 0, then->shape(), &output));

    if (output->NumElements() > 0) {
      functor::SelectScalarFunctor<GPUDevice, T> func;
      TTypes<bool>::ConstScalar cond_scalar = cond->scalar<bool>();
      func(ctx->eigen_gpu_device(), output->flat<T>(), cond_scalar,
           then->flat<T>(), else_->flat<T>());
    }
  }
};

}  // namespace functor

template <typename T>
void ComputeBroadcasting(OpKernelContext* ctx, const Tensor* cond,
                         const Tensor* then, const Tensor* else_) {
  // Preliminary validation of sizes.
  OP_REQUIRES(
      ctx, TensorShapeUtils::IsVector(cond->shape()),
      errors::InvalidArgument("'cond' must be a vector, but saw shape: ",
                              cond->shape().DebugString()));
  OP_REQUIRES(
      ctx,
      FastBoundsCheck(cond->NumElements(),
                      std::numeric_limits<Eigen::DenseIndex>::max()),
      errors::InvalidArgument("cond vector larger than ",
                              std::numeric_limits<Eigen::DenseIndex>::max()));
  OP_REQUIRES(
      ctx,
      FastBoundsCheck(then->flat_outer_dims<T>().dimension(1),
                      std::numeric_limits<Eigen::DenseIndex>::max()),
      errors::InvalidArgument("flat outer dims dim 1 size >= ",
                              std::numeric_limits<Eigen::DenseIndex>::max()));

  OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(then->shape()),
              errors::InvalidArgument(
                  "'then' must be at least a vector, but saw shape: ",
                  then->shape().DebugString()));
  OP_REQUIRES(
      ctx, then->shape().dim_size(0) == cond->NumElements(),
      errors::InvalidArgument(
          "Number of batches of 'then' must match size of 'cond', but saw: ",
          then->shape().dim_size(0), " vs. ", cond->NumElements()));
  OP_REQUIRES(
      ctx, then->shape().IsSameSize(else_->shape()),
      errors::InvalidArgument(
          "'then' and 'else' must have the same size.  but received: ",
          then->shape().DebugString(), " vs. ", else_->shape().DebugString()));

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {1, 2}, 0, then->shape(), &output));
  if (output->NumElements() > 0) {
    functor::BatchSelectFunctor<GPUDevice, T> func;
    func(ctx->eigen_gpu_device(), output->flat_outer_dims<T>(),
         cond->vec<bool>(), then->flat_outer_dims<T>(),
         else_->flat_outer_dims<T>());
  }
}

template <typename T>
void ComputeElementwise(OpKernelContext* ctx, const Tensor* cond,
                        const Tensor* then, const Tensor* else_) {
  if (!ctx->ValidateInputsAreSameShape()) return;
  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {1, 2}, 0, then->shape(), &output));
  if (output->NumElements() > 0) {
    functor::SelectFunctor<GPUDevice, T> func;
    func(ctx->eigen_gpu_device(), output->flat<T>(), cond->flat<bool>(),
         then->flat<T>(), else_->flat<T>());
  }
}

template <typename T>
void ComputeScalar(OpKernelContext* ctx, const Tensor* cond, const Tensor* then,
                   const Tensor* else_) {
  OP_REQUIRES(
      ctx, then->shape().IsSameSize(else_->shape()),
      errors::InvalidArgument(
          "'then' and 'else' must have the same size.  but received: ",
          then->shape().DebugString(), " vs. ", else_->shape().DebugString()));

  functor::SelectScalarHandler<T> handler;
  handler(ctx, cond, then, else_);
}

template <typename Device, typename T>
class SelectOp : public OpKernel {
 public:
  explicit SelectOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& cond = ctx->input(0);
    const Tensor& then = ctx->input(1);
    const Tensor& else_ = ctx->input(2);

    if (TensorShapeUtils::IsScalar(cond.shape())) {
      ComputeScalar<T>(ctx, &cond, &then, &else_);
      return;
    }

    bool broadcasting = (TensorShapeUtils::IsVector(cond.shape()) &&
                         !TensorShapeUtils::IsVector(then.shape()));

    if (broadcasting) {
      ComputeBroadcasting<T>(ctx, &cond, &then, &else_);
    } else {
      ComputeElementwise<T>(ctx, &cond, &then, &else_);
    }
  }
};

template <typename Device, typename T>
class SelectV2Op : public OpKernel {
 public:
  explicit SelectV2Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* cond = &ctx->input(0);
    const Tensor* then = &ctx->input(1);
    const Tensor* else_ = &ctx->input(2);

    // TODO(itex): support more select pattern in itex
    if (cond->NumElements() == 1 && then->shape().IsSameSize(else_->shape())) {
      functor::SelectScalarHandler<T> handler;
      handler(ctx, cond, then, else_);
      return;
    }

    // The `cond`, `then`, and `else` are broadcastable (bcast.IsValid()),
    // This matches the behavior of numpy.
    BCastList<3> bcast({cond->shape().dim_sizes(), then->shape().dim_sizes(),
                        else_->shape().dim_sizes()},
                       false);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "condition ", cond->shape().DebugString(), ", then ",
                    then->shape().DebugString(), ", and else ",
                    else_->shape().DebugString(), " must be broadcastable"));

    // Broadcast `cond`, `then` and `else` to combined shape,
    // in order to obtain the reshape.
    BCast cond_bcast(bcast.output_shape(), cond->shape().dim_sizes(), false);
    BCast then_bcast(bcast.output_shape(), then->shape().dim_sizes(), false);
    BCast else_bcast(bcast.output_shape(), else_->shape().dim_sizes(), false);
    OP_REQUIRES(
        ctx,
        cond_bcast.IsValid() && then_bcast.IsValid() && else_bcast.IsValid(),
        errors::InvalidArgument("condition ", cond->shape().DebugString(),
                                ", then ", then->shape().DebugString(),
                                ", and else ", else_->shape().DebugString(),
                                " must be broadcastable"));

    // Combined shape should be the final shape.
    OP_REQUIRES(
        ctx,
        cond_bcast.output_shape() == bcast.output_shape() &&
            then_bcast.output_shape() == bcast.output_shape() &&
            else_bcast.output_shape() == bcast.output_shape(),
        errors::InvalidArgument("condition ", cond->shape().DebugString(),
                                ", then ", then->shape().DebugString(),
                                ", and else ", else_->shape().DebugString(),
                                " must be broadcastable to the same shape"));

    Tensor* output = nullptr;
    const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {1, 2}, 0, output_shape, &output));

    if (output->NumElements() == 0) {
      return;
    }

#define HANDLE_DIM(NDIMS)                                            \
  {                                                                  \
    functor::BCastSelectFunctor<Device, T, NDIMS> func;              \
    func(ctx->eigen_device<Device>(),                                \
         output->shaped<T, NDIMS>(bcast.result_shape()),             \
         cond->template shaped<bool, NDIMS>(cond_bcast.y_reshape()), \
         then->template shaped<T, NDIMS>(then_bcast.y_reshape()),    \
         else_->template shaped<T, NDIMS>(else_bcast.y_reshape()),   \
         BCast::ToIndexArray<NDIMS>(cond_bcast.y_bcast()),           \
         BCast::ToIndexArray<NDIMS>(then_bcast.y_bcast()),           \
         BCast::ToIndexArray<NDIMS>(else_bcast.y_bcast()));          \
  }

    const int ndims = static_cast<int>(bcast.result_shape().size());
    switch (ndims) {
      case 1:
        HANDLE_DIM(1);
        break;
      case 2:
        HANDLE_DIM(2);
        break;
      case 3:
        HANDLE_DIM(3);
        break;
      case 4:
        HANDLE_DIM(4);
        break;
      case 5:
        HANDLE_DIM(5);
        break;
      case 6:
        HANDLE_DIM(6);
        break;
      case 7:
        HANDLE_DIM(7);
        break;
      case 8:
        HANDLE_DIM(8);
        break;
      default:
        ctx->SetStatus(errors::Unimplemented(
            "Broadcast between ", ctx->input(0).shape().DebugString(), " and ",
            ctx->input(1).shape().DebugString(), " is not supported yet."));
        break;
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SelectV2Op);
};

#define REGISTER_KERNEL(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Select").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),   \
      SelectOp<GPUDevice, TYPE>);                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SelectV2").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      SelectV2Op<GPUDevice, TYPE>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
TF_CALL_bool(REGISTER_KERNEL);
TF_CALL_complex64(REGISTER_KERNEL);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNEL);
TF_CALL_complex128(REGISTER_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNEL

}  // namespace itex
