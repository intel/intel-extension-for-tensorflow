/* Copyright (c) 2021-2023 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/group_norm_op_gpu.h"

#include "itex/core/kernels/gpu/group_norm_op.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T, typename U>
void ComputeMeanAndVar(OpKernelContext* context, const T* input, U* temp_mean,
                       U* temp_var, const InputShape& shape) {
  const auto& d = context->eigen_gpu_device();
  int num_to_reduce = shape.num_hw * shape.chans_per_group;
  bool use_one_kernel = (num_to_reduce < 64 * 1024);
  if (use_one_kernel) {
    if (shape.chans_per_group < 32) {
      impl::LaunchMeanAndVarKernel<16>(d, input, temp_mean, temp_var, shape);
    } else {
      impl::LaunchMeanAndVarKernel<32>(d, input, temp_mean, temp_var, shape);
    }

  } else {
    int scaled_hw = std::min(512, shape.num_hw);

    // allocate temporary for sum and square sum
    auto scratch_shape =
        TensorShape({2 * shape.num_batches * shape.num_groups * scaled_hw});
    Tensor temp_t;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::v(),
                                                   scratch_shape, &temp_t));
    U* temp_sum = temp_t.flat<U>().data();
    U* temp_sqr = temp_sum + shape.num_batches * shape.num_groups * scaled_hw;

    if (shape.chans_per_group < 32) {
      impl::LaunchPartialSumKernel<16>(d, input, temp_sum, temp_sqr, shape,
                                       scaled_hw);
    } else {
      impl::LaunchPartialSumKernel<32>(d, input, temp_sum, temp_sqr, shape,
                                       scaled_hw);
    }
    if (shape.chans_per_group < 32) {
      impl::LaunchMeanFromPartialKernel<16>(d, temp_sum, temp_sqr, temp_mean,
                                            temp_var, shape, scaled_hw);
    } else {
      impl::LaunchMeanFromPartialKernel<32>(d, temp_sum, temp_sqr, temp_mean,
                                            temp_var, shape, scaled_hw);
    }
  }
}

template <typename T, typename U>
void DoNormalization(OpKernelContext* context, const T* input, const T* gamma,
                     const T* beta, const U* temp_mean, const U* temp_var,
                     float epsilon, T* output, bool use_scale, bool use_center,
                     const InputShape& shape) {
  const auto& d = context->eigen_gpu_device();
  if (use_scale) {
    if (use_center) {
      impl::LaunchNormalizationKernel<true, true>(
          d, input, gamma, beta, temp_mean, temp_var, epsilon, output, shape);
    } else {
      impl::LaunchNormalizationKernel<true, false>(
          d, input, gamma, beta, temp_mean, temp_var, epsilon, output, shape);
    }
  } else {
    if (use_center) {
      impl::LaunchNormalizationKernel<false, true>(
          d, input, gamma, beta, temp_mean, temp_var, epsilon, output, shape);
    } else {
      impl::LaunchNormalizationKernel<false, false>(
          d, input, gamma, beta, temp_mean, temp_var, epsilon, output, shape);
    }
  }
}

template <typename T, bool is_scaleoffset>
void TwoColReduction(OpKernelContext* ctx, const T* input, const T* grad_y,
                     const T* gamma, const float* mean, const float* var,
                     float epsilon, int group, int num_batches, int num_HW,
                     int channel_per_group, int channel, T* result,
                     T* result2) {
  int extend_x, extend_y, extend_z;
  if constexpr (is_scaleoffset) {
    extend_x = 1;
    extend_y = num_batches * num_HW;
    extend_z = channel;
  } else {
    extend_x = 1;
    extend_y = num_HW * channel_per_group;
    extend_z = group * num_batches;
  }
  static constexpr int SubGroupSize = ColReductionPolicy::SUB_GROUP_SIZE;
  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();
  int max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  int tile_z = extend_z > 64 ? 16 : extend_z;
  int num_segments_z = DivUp(extend_z, tile_z);

  int num_sg = 1;
  int tile_y = 8;
  // tile_y * tile_z should be approxmately equal to SubGroupSize * num_sg
  compute_tile(&num_sg, &tile_y, extend_y, tile_z,
               max_group_size / SubGroupSize);
  int steps = ceil_log2(tile_y);

  // 64 Xe cores on PVC, extend_x * tile_y * tile_z * num_segments_z > 64 *
  // max_group_size
  bool is_full_occu = (extend_x * num_segments_z) >
                      (64 * (max_group_size / (num_sg * SubGroupSize)));

  static constexpr int MAX_ELEMENTS_PER_THREAD_FOR_SMALL_REDUCE = 32;
  int elems_per_thread = DivUp(extend_y, tile_y);

  if (!is_full_occu &&
      elems_per_thread > MAX_ELEMENTS_PER_THREAD_FOR_SMALL_REDUCE) {
    // At least 4 round of launch
    int max_elems_per_thread =
        DivUp(num_segments_z * extend_x * (extend_y / tile_y),
              (4 * 64 * (max_group_size / (num_sg * SubGroupSize))));
    int min_elems_per_thread =
        DivUp(DivUp(extend_y, tile_y) * tile_z, max_group_size * 16);
    elems_per_thread = std::max(max_elems_per_thread, min_elems_per_thread);

    int num_segments_y = DivUp(extend_y, tile_y * elems_per_thread);
    Tensor scratch_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::value,
                            TensorShape({extend_x * num_segments_y * extend_z}),
                            &scratch_tensor));
    T* inter_out = scratch_tensor.flat<T>().data();

    Tensor scratch_tensor2;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::value,
                            TensorShape({extend_x * num_segments_y * extend_z}),
                            &scratch_tensor2));
    T* inter_out2 = scratch_tensor2.flat<T>().data();

    sycl::range<3> local(1, 1, num_sg * SubGroupSize);
    sycl::range<3> global(extend_x, num_segments_y, num_segments_z * local[2]);
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<T> scratch(tile_y * tile_z, cgh);
      reduciton_helper::LocalAcc<T> scratch2(tile_y * tile_z, cgh);
      impl::TwoColReduceKernel<T, reduciton_helper::LocalAcc<T>, true,
                               is_scaleoffset>
          task(grad_y, input, gamma, mean, var, inter_out, inter_out2, extend_y,
               extend_z, tile_y, num_segments_y, elems_per_thread, tile_z,
               steps, num_batches, num_HW, group, channel_per_group, epsilon,
               scratch, scratch2);
      cgh.parallel_for<impl::TwoColReduceKernel<
          T, reduciton_helper::LocalAcc<T>, true, is_scaleoffset>>(
          sycl::nd_range<3>{global, local}, task);
    });
    compute_tile(max_group_size, &tile_y, num_segments_y, tile_z);

    steps = ceil_log2(tile_y);
    elems_per_thread = DivUp(num_segments_y, tile_y);
    local = sycl::range<3>{1, 1, static_cast<size_t>(max_group_size)};
    global = sycl::range<3>{static_cast<size_t>(extend_x), 1,
                            num_segments_z * local[2]};
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<T> scratch(tile_y * tile_z, cgh);
      reduciton_helper::LocalAcc<T> scratch2(tile_y * tile_z, cgh);
      impl::TwoColReduceKernel<T, reduciton_helper::LocalAcc<T>, false,
                               is_scaleoffset>
          task(inter_out, inter_out2, gamma, mean, var, result, result2,
               num_segments_y, extend_z, tile_y, 1, elems_per_thread, tile_z,
               steps, num_batches, num_HW, group, channel_per_group, epsilon,
               scratch, scratch2);
      cgh.parallel_for<impl::TwoColReduceKernel<
          T, reduciton_helper::LocalAcc<T>, false, is_scaleoffset>>(
          sycl::nd_range<3>{global, local}, task);
    });

    return;
  } else {
    sycl::range<3> local(1, 1, num_sg * SubGroupSize);
    sycl::range<3> global(extend_x, 1, num_segments_z * local[2]);

    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<T> scratch(tile_y * tile_z, cgh);
      reduciton_helper::LocalAcc<T> scratch2(tile_y * tile_z, cgh);
      impl::TwoColReduceKernel<T, reduciton_helper::LocalAcc<T>, true,
                               is_scaleoffset>
          task(grad_y, input, gamma, mean, var, result, result2, extend_y,
               extend_z, tile_y, 1, elems_per_thread, tile_z, steps,
               num_batches, num_HW, group, channel_per_group, epsilon, scratch,
               scratch2);
      cgh.parallel_for<impl::TwoColReduceKernel<
          T, reduciton_helper::LocalAcc<T>, true, is_scaleoffset>>(
          sycl::nd_range<3>{global, local}, task);
    });
    return;
  }
}
}  // end namespace

namespace functor {

template <typename T>
struct GroupNormFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::Flat output,
                  typename TTypes<float>::Flat reserve_space_1,
                  typename TTypes<float>::Flat reserve_space_2,
                  typename TTypes<T>::ConstVec gamma,
                  typename TTypes<T>::ConstVec beta, float epsilon,
                  bool use_scale, bool use_center, const InputShape& shape) {
    // allocate temporary for mean and variance
    auto scratch_shape =
        TensorShape({shape.num_batches * shape.num_groups * 2});

    // using U: float as computation type
    using U = float;
    U* temp_mean = reserve_space_1.data();
    U* temp_var = reserve_space_2.data();

    ComputeMeanAndVar(context, input.data(), temp_mean, temp_var, shape);
    DoNormalization(context, input.data(), gamma.data(), beta.data(), temp_mean,
                    temp_var, epsilon, output.data(), use_scale, use_center,
                    shape);
  }
};

template <typename T>
struct GroupNormGradFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, typename TTypes<T>::ConstFlat x,
                  typename TTypes<float>::ConstFlat mean,
                  typename TTypes<float>::ConstFlat var,
                  typename TTypes<T>::ConstVec gamma,
                  typename TTypes<T>::ConstFlat grad_y, float epsilon,
                  int group, int num_batches, int num_HW, int channel_per_group,
                  int channel, typename TTypes<T>::Flat dx,
                  typename TTypes<T>::Flat dscale,
                  typename TTypes<T>::Flat doffset) {
    TwoColReduction<T, true>(context, x.data(), grad_y.data(), gamma.data(),
                             mean.data(), var.data(), epsilon, group,
                             num_batches, num_HW, channel_per_group, channel,
                             doffset.data(), dscale.data());
    Tensor reduce_result;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::v(),
                                          TensorShape({num_batches, group}),
                                          &reduce_result));
    Tensor reduce_result2;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::v(),
                                          TensorShape({num_batches, group}),
                                          &reduce_result2));
    TwoColReduction<T, false>(
        context, x.data(), grad_y.data(), gamma.data(), mean.data(), var.data(),
        epsilon, group, num_batches, num_HW, channel_per_group, channel,
        reduce_result.flat<T>().data(), reduce_result2.flat<T>().data());
    auto* stream = context->eigen_gpu_device().stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups =
        (num_batches * num_HW * channel + total_threads - 1) / total_threads;
    stream->submit([&](sycl::handler& cgh) {
      impl::ComputeDxKernel<T> task(
          x.data(), grad_y.data(), gamma.data(), mean.data(), var.data(),
          reduce_result.flat<T>().data(), reduce_result2.flat<T>().data(),
          epsilon, group, num_batches, num_HW, channel_per_group, dx.data());
      cgh.parallel_for<impl::ComputeDxKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(total_threads * num_workgroups),
                            sycl::range<1>(total_threads)),
          task);
    });

    // ComputeDx(context, x.data(), grad_y.data(), gamma.data(), mean.data(),
    // var.data(), epsilon, group, num_batches, num_HW, channel_per_group,
    // channel, dx.data());
    // doffset.device(d) = grad_y.reshape(normed_shape).sum(Eigen::DSizes<Index,
    // 2>(0,1)); x_hat.reshape(reshape_dims).device(d) =
    // (x.reshape(reshape_dims).template cast<float>() -
    // mean.reshape(mean_shape).broadcast(reshape_dims)) / (var +
    // var.constant(epsilon)).sqrt().reshape(mean_shape).broadcast(reshape_dims);
    // dscale.device(d) = (x_hat * grad_y.template
    // cast<float>()).reshape(normed_shape).sum(Eigen::DSizes<Index,
    // 2>(0,1)).template cast<T>(); dx_hat.reshape(normed_shape).device(d) =
    // (grad_y.reshape(normed_shape) *
    // gamma.reshape(gamma_shape).broadcast(normed_shape)).template
    // cast<float>(); int N = channel_per_group * num_HW;
    // x_hat.reshape(reshape_dims).device(d)=(dx_hat.constant(N)*dx_hat).reshape(reshape_dims)
    // - (dx_hat.reshape(reshape_dims)).sum(Eigen::DSizes<Index,
    // 2>(1,3)).reshape(mean_shape).broadcast(reshape_dims) -
    // x_hat.reshape(reshape_dims)*(((dx_hat*x_hat).reshape(reshape_dims)).sum(Eigen::DSizes<Index,
    // 2>(1,3)).reshape(mean_shape).broadcast(reshape_dims));
    // dx.reshape(reshape_dims).device(d) =
    // (x_hat.reshape(reshape_dims)/((var.constant(N) * (var +
    // var.constant(epsilon)).sqrt()).reshape(mean_shape).broadcast(reshape_dims))).template
    // cast<T>();

    // dx.reshape(reshape_dims).device(d) = ((N * dx_hat.reshape(reshape_dims) -
    // (dx_hat.reshape(reshape_dims)).sum(Eigen::DSizes<Index,
    // 2>(1,3)).reshape(mean_shape).broadcast(reshape_dims) -
    // x_hat.reshape(reshape_dims)*((dx_hat*x_hat).reshape(reshape_dims)).sum(Eigen::DSizes<Index,
    // 2>(1,3)).reshape(mean_shape).broadcast(reshape_dims))/(N * (var +
    // epsilon).sqrt().reshape(mean_shape).broadcast(reshape_dims))).template
    // cast<T>();
  }
};

}  // end namespace functor

// Instantiate the GPU implementation
#define DEFINE_GPU_KERNELS(T)                              \
  template struct functor::GroupNormFunctor<GPUDevice, T>; \
  template struct functor::GroupNormGradFunctor<GPUDevice, T>;

DEFINE_GPU_KERNELS(float);
DEFINE_GPU_KERNELS(Eigen::half);
DEFINE_GPU_KERNELS(Eigen::bfloat16);
#undef DEFINE_GPU_KERNELS

}  // end namespace itex
