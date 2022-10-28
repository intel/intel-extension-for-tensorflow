/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/in_topk_op.h"
#include "itex/core/kernels/gpu/reduction_ops.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"

namespace Eigen {
namespace internal {

template <typename T>
struct MaskReducer {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    internal::scalar_sum_op<T> sum_op;
    T tmp = t < 0 ? -1 : t;
    *accum = sum_op(*accum, tmp);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p,
                                                          Packet* accum) const {
    Packet tmp = p < 0 ? -1 : p;
    *accum = padd<Packet>(*accum, p);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    internal::scalar_cast_op<int, T> conv;
    return conv(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(initialize());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet
  finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T
  finalizeBoth(const T saccum, const Packet& vaccum) const {
    internal::scalar_sum_op<T> sum_op;
    return sum_op(saccum, predux(vaccum));
  }
};
template <typename T, typename Device>
struct reducer_traits<MaskReducer<T>, Device> {
  enum {
    Cost = NumTraits<T>::AddCost,
    PacketAccess = PacketType<T, Device>::HasAdd,
    IsStateful = false,
    IsExactlyAssociative = NumTraits<T>::IsInteger
  };
};

}  // namespace internal
}  // namespace Eigen

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T, typename TargetT>
struct ComputePredictionMaskKernel {
  ComputePredictionMaskKernel(T* predictions, TargetT* targets, int64* mask,
                              int num_targets, int num_classes)
      : predictions_(predictions),
        targets_(targets),
        mask_(mask),
        num_targets_(num_targets),
        num_classes_(num_classes) {}
  void operator()(sycl::nd_item<1> item) const {
    auto i = item.get_global_id()[0];
    if (i >= num_targets_ * num_classes_) return;
    const int batch_index = i / num_classes_;
    TargetT target_idx = targets_[batch_index];

    if (!FastBoundsCheck(target_idx, num_classes_)) {
      mask_[i] = -1;
      return;
    }

    T prediction = predictions_[i];
    T target_prediction = predictions_[batch_index * num_classes_ + target_idx];

    if (!Eigen::numext::isfinite(prediction) ||
        !Eigen::numext::isfinite(target_prediction)) {
      mask_[i] = -1;
    } else {
      mask_[i] = prediction > target_prediction ? 1 : 0;
    }
  }

 private:
  T* predictions_;
  TargetT* targets_;
  int64* mask_;
  int num_targets_;
  int num_classes_;
};

template <typename T, typename TargetT>
void LaunchMaskKernel(const gpuStream_t& stream, const int32 num_workgroup,
                      const int32 workgroup_size, const T* predict,
                      const TargetT* target, const int64* mask, int num_targets,
                      int num_classes) {
  stream->submit([&](sycl::handler& cgh) {
    ComputePredictionMaskKernel<T, TargetT> task(
        const_cast<T*>(predict), const_cast<TargetT*>(target),
        const_cast<int64*>(mask), num_targets, num_classes);
    cgh.parallel_for<ComputePredictionMaskKernel<T, TargetT>>(
        sycl::nd_range<1>(sycl::range<1>(num_workgroup * workgroup_size),
                          sycl::range<1>(workgroup_size)),
        task);
  });
}

template <typename T, typename TargetT>
struct InTopKFunctor<GPUDevice, T, TargetT> {
  template <int ndims>
  using Dims = Eigen::DSizes<Eigen::Index, ndims>;

  void operator()(OpKernelContext* context,
                  typename TTypes<T, 2>::ConstTensor predictions,
                  typename TTypes<TargetT>::ConstVec targets, const TopKArg k,
                  typename TTypes<bool>::Vec output) {
    const Eigen::Index num_targets = predictions.dimension(0);
    const Eigen::Index num_classes = predictions.dimension(1);

    OP_REQUIRES(
        context, num_targets * num_classes < std::numeric_limits<int>::max(),
        errors::InvalidArgument(
            "Number of targets * number of classes must be less than INT_MAX"));

    // Temporary storage for a mask computed by  `ComputePredictionMaskKernel`.
    Tensor predictions_mask;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_INT64,
                                        TensorShape({num_targets, num_classes}),
                                        &predictions_mask));

    // Number of predictions for each target that are larger than the target
    // prediction (or -1 if we can't compute this number, because not all
    // predictions are finite or target class is out of range).
    Tensor num_larger_prediction;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({num_targets}),
                                          &num_larger_prediction));

    const auto& d = context->eigen_gpu_device();
    auto& stream = d.stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroups =
        (num_targets * num_classes + total_threads - 1) / total_threads;

    // Compute a mask for all predictions.
    LaunchMaskKernel<T, TargetT>(stream, num_workgroups, total_threads,
                                 predictions.data(), targets.data(),
                                 predictions_mask.flat<int64>().data(),
                                 num_targets, num_classes);

    // Reduce prediction masks to number of predictions larger than the target
    // prediction, or to the negative value if we can't compute an answer.
    {
      auto in = predictions_mask.matrix<int64>();
      auto out = num_larger_prediction.flat<int64>();

      // reduce the 1 dim(along class of each target)
      Eigen::array<int, 1> reduce_axis({1});
      ReduceEigenImpl<decltype(out), decltype(in), decltype(reduce_axis),
                      Eigen::internal::MaskReducer<int64>>
          impl;
      impl(d, out, in, reduce_axis, Eigen::internal::MaskReducer<int64>());
    }

    // Compute if target prediction is in top K predictions.
    auto cnt = num_larger_prediction.flat<int64>();

    if (k.k_tensor != nullptr) {
      if (k.k_tensor->dtype() == DT_INT32) {
        output.device(d) =
            (cnt >= cnt.constant(0)) &&
            (cnt < k.k_tensor->flat<int32>().template cast<int64>().broadcast(
                       Dims<1>(num_targets)));
      } else {
        output.device(d) =
            (cnt >= cnt.constant(0)) &&
            (cnt < k.k_tensor->flat<int64>().broadcast(Dims<1>(num_targets)));
      }
    } else {
      output.device(d) =
          (cnt >= cnt.constant(0)) && (cnt < targets.constant(k.k_value));
    }
  }
};

}  // namespace functor

// Definition of the GPU implementations declared in in_topk_op.cc.
#define DEFINE_GPU_KERNELS(T, TARGET_T) \
  template struct functor::InTopKFunctor<GPUDevice, T, TARGET_T>;

DEFINE_GPU_KERNELS(float, int32);
DEFINE_GPU_KERNELS(float, int64);

#undef DEFINE_GPU_KERNELS

}  // namespace itex
