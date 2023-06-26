/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/kernels/gpu/fill_empty_rows_functor.h"

#include "itex/core/kernels/gpu/full_reduction_kernels.h"
#include "itex/core/utils/register_types.h"

namespace itex {

using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename T, typename Tindex>
struct GatherOriginalGradValuesKernel {
  GatherOriginalGradValuesKernel(const Tindex n,
                                 const Tindex* reverse_index_map,
                                 const T* grad_values, T* d_values,
                                 bool* visited)
      : n_(n),
        reverse_index_map_(reverse_index_map),
        grad_values_(grad_values),
        d_values_(d_values),
        visited_(visited) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= n_) return;

    Tindex output_i = reverse_index_map_[id];
    d_values_[id] = grad_values_[output_i];
    visited_[output_i] = true;
  }

 private:
  const Tindex n_;
  const Tindex* reverse_index_map_;
  const T* grad_values_;
  T* d_values_;
  bool* visited_;
};

template <typename T, typename Tindex>
struct ZeroMaskedValues {
  ZeroMaskedValues(const bool* mask, const T* values, const Tindex n_full,
                   T* out)
      : mask_(mask), values_(values), n_full_(n_full), out_(out) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= n_full_) return;

    // true means return zero instead of value
    out_[id] = mask_[id] ? T(0) : values_[id];
  }

 private:
  const bool* mask_;
  const T* values_;
  const Tindex n_full_;
  T* out_;
};

}  // namespace

namespace functor {

template <typename T, typename Tindex>
struct FillEmptyRowsGrad<GPUDevice, T, Tindex> {
  Status operator()(OpKernelContext* context,
                    typename TTypes<Tindex>::ConstVec reverse_index_map,
                    typename TTypes<T>::ConstVec grad_values,
                    typename TTypes<T>::Vec d_values,
                    typename TTypes<T>::Scalar d_default_value) {
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    const Tindex N = reverse_index_map.dimension(0);
    const Tindex N_full = grad_values.dimension(0);

    Tensor visited_t;
    TF_RETURN_IF_ERROR(
        context->allocate_temp(DT_BOOL, TensorShape({N_full}), &visited_t));
    auto visited = visited_t.vec<bool>();
    visited.device(device) = visited.constant(false);

    auto stream = device.stream();
    auto wg_size =
        stream->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    if (N > 0) {
      auto num_wg = (N + wg_size - 1) / wg_size;
      sycl::nd_range<1> kernel_range(num_wg * wg_size, wg_size);
      GatherOriginalGradValuesKernel<T, Tindex> gather_kernel(
          N, reverse_index_map.data(), grad_values.data(), d_values.data(),
          visited.data());
      stream->parallel_for<GatherOriginalGradValuesKernel<T, Tindex>>(
          kernel_range, gather_kernel);
    }

    // Now we mask out the visited values and sum the remaining ones (which
    // correspond to the empty rows in the forward input) to compute
    // d_default_value.
    if (N_full > 0) {
      Tensor masked_values;
      TF_RETURN_IF_ERROR(context->allocate_temp(
          DataTypeToEnum<T>::v(), TensorShape({N_full}), &masked_values));
      auto masked_values_ptr = masked_values.flat<T>().data();
      auto num_wg = (N_full - wg_size + 1) / wg_size;
      sycl::nd_range<1> kernel_range(num_wg * wg_size, wg_size);
      ZeroMaskedValues<T, Tindex> zero_mask_kernel(
          visited.data(), grad_values.data(), N_full, masked_values_ptr);
      stream->parallel_for<ZeroMaskedValues<T, Tindex>>(kernel_range,
                                                        zero_mask_kernel);

      itex::LaunchFullReduction<T, T, T, sycl::plus<T>>(
          context, masked_values_ptr, d_default_value.data(), T(0), N_full,
          sycl::plus<T>());
    }

    return OkStatus();
  }
};

}  // namespace functor

#define DEFINE_INT64(T) \
  template struct functor::FillEmptyRowsGrad<GPUDevice, T, int64>;
// TODO(itex): add bf16/fp16 back when eigen::vec is ready
TF_CALL_INTEGRAL_TYPES(DEFINE_INT64);
TF_CALL_float(DEFINE_INT64);
TF_CALL_double(DEFINE_INT64);
#undef DEFINE_INT64

}  // namespace itex
