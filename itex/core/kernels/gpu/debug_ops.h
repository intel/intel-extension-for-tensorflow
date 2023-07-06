/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_DEBUG_OPS_H_
#define ITEX_CORE_KERNELS_GPU_DEBUG_OPS_H_

#include <algorithm>
#include <limits>

#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/devices/gpu/gpu_device_plugin.h"
#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Tin, typename Tout>
class DebugNumericSummaryV2Op;

template <typename Tin, typename Tout>
struct CurtHealthKernel {
  CurtHealthKernel(size_t input_size, size_t total_items, const Tin* input_data,
                   Tout* output)
      : input_size(input_size),
        total_items(total_items),
        input_data(input_data),
        output(output) {}
  void operator()(sycl::item<1> item) const {
    auto id = item.get_id(0);
    while (id < input_size) {
      if (Eigen::numext::isinf(input_data[id]) ||
          Eigen::numext::isnan(input_data[id])) {
        output[0] = 1.0;
      }
      id += total_items;
    }
  }

 private:
  size_t input_size;
  size_t total_items;
  const Tin* input_data;
  Tout* output;
};

template <typename Tin, typename Tout>
struct ConciseHealthKernel {
  ConciseHealthKernel(size_t input_size, size_t total_items,
                      const Tin* input_data, Tout* output)
      : input_size(input_size),
        total_items(total_items),
        input_data(input_data),
        output(output) {}
  void operator()(sycl::item<1> item) const {
    auto id = item.get_id(0);
    Tout accum[3] = {0.0, 0.0, 0.0};
    while (id < input_size) {
      if (Eigen::numext::isinf(input_data[id])) {
        if (input_data[id] < static_cast<Tin>(0.f)) {
          ++accum[0];
        } else {
          ++accum[1];
        }
      }
      if (Eigen::numext::isnan(input_data[id])) {
        ++accum[2];
      }
      id += total_items;
    }
    ItexAtomicAdd(output, accum[0]);
    ItexAtomicAdd(output + 1, accum[1]);
    ItexAtomicAdd(output + 2, accum[2]);
  }

 private:
  size_t input_size;
  size_t total_items;
  const Tin* input_data;
  Tout* output;
};

template <typename Tin, typename Tout>
struct FullHealthKernel {
  FullHealthKernel(size_t input_size, size_t total_items, const Tin* input_data,
                   Tout* output)
      : input_size(input_size),
        total_items(total_items),
        input_data(input_data),
        output(output) {}
  void operator()(sycl::item<1> item) const {
    auto id = item.get_id(0);
    Tout accum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    while (id < input_size) {
      if (Eigen::numext::isinf(input_data[id])) {
        if (input_data[id] < static_cast<Tin>(0.f)) {
          ++accum[0];
        } else {
          ++accum[1];
        }
      } else if (Eigen::numext::isnan(input_data[id])) {
        ++accum[2];
      } else {
        if (input_data[id] < static_cast<Tin>(0.f)) {
          ++accum[3];
        } else if (input_data[id] == static_cast<Tin>(0.f)) {
          ++accum[4];
        } else {
          ++accum[5];
        }
      }
      id += total_items;
    }

    ItexAtomicAdd(output, accum[0]);
    ItexAtomicAdd(output + 1, accum[1]);
    ItexAtomicAdd(output + 2, accum[2]);
    ItexAtomicAdd(output + 3, accum[3]);
    ItexAtomicAdd(output + 4, accum[4]);
    ItexAtomicAdd(output + 5, accum[5]);
  }

 private:
  size_t input_size;
  size_t total_items;
  const Tin* input_data;
  Tout* output;
};

template <typename Tin, typename Tout>
struct ReduceInfNanThreeSlotsKernel {
  ReduceInfNanThreeSlotsKernel(size_t input_size, size_t total_items,
                               const Tin* input_data, Tout* output)
      : input_size(input_size),
        total_items(total_items),
        input_data(input_data),
        output(output) {}
  void operator()(sycl::item<1> item) const {
    auto id = item.get_id(0);
    while (id < input_size) {
      if (Eigen::numext::isinf(input_data[id])) {
        if (input_data[id] < static_cast<Tin>(0.f)) {
          output[0] = -std::numeric_limits<Tout>::infinity();
        } else {
          output[1] = std::numeric_limits<Tout>::infinity();
        }
      }
      if (Eigen::numext::isnan(input_data[id])) {
        output[2] = std::numeric_limits<Tout>::quiet_NaN();
      }
      id += total_items;
    }
  }

 private:
  size_t input_size;
  size_t total_items;
  const Tin* input_data;
  Tout* output;
};

template <typename Tin, typename Tout>
class DebugNumericSummaryV2Op<GPUDevice, Tin, Tout> : public OpKernel {
 public:
  explicit DebugNumericSummaryV2Op(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("tensor_debug_mode", &tensor_debug_mode_));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_id", &tensor_id_));
  }

  void Compute(OpKernelContext* context) override {
    Tensor* output_tensor;
    Tout tensor_id = static_cast<Tout>(tensor_id_);
    const Tensor& tensor = context->input(0);
    const Tout num_elem = static_cast<Tout>(tensor.NumElements());

    auto input = tensor.flat<Tin>();
    auto* ITEX_GPU_stream = context->GetDeviceStream();
    OP_REQUIRES(context, ITEX_GPU_stream != nullptr,
                errors::Internal("No GPU stream available."));

    // Disregard lossy cast if mode is REDUCE_INF_NAN_THREE_SLOTS because
    // that mode does not make use of tensor_id.
    if (tensor_debug_mode_ != 8) {
      OP_REQUIRES(
          context, tensor_id_ <= kMaxTensorId,
          errors::InvalidArgument("DebugNumericSummaryV2Op requires "
                                  "tensor_id to be less than or equal to "
                                  "(2^",
                                  std::numeric_limits<Tout>::digits,
                                  "). Given tensor_id:", tensor_id_));
    }

    if (tensor_debug_mode_ == 2) {  // CURT_HEALTH.
      TensorShape shape({2});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      ITEX_GPU_stream->fill<Tout>(output_tensor->flat<Tout>().data(), Tout(0),
                                  2);
      ITEX_GPU_stream->memcpy(output_tensor->data(), &tensor_id, sizeof(Tout))
          .wait();

      if (num_elem == 0) return;

      auto total_items =
          ITEX_GPU_stream->get_device()
              .template get_info<sycl::info::device::max_work_group_size>();

      ITEX_GPU_stream->submit([&](sycl::handler& cgh) {
        auto input_data = input.data();
        auto input_size = input.size();
        auto output = output_tensor->flat<Tout>().data() + 1;
        CurtHealthKernel<Tin, Tout> task(input_size, total_items, input_data,
                                         output);
        cgh.parallel_for<CurtHealthKernel<Tin, Tout>>(
            sycl::range<1>(total_items), task);
      });
    } else if (tensor_debug_mode_ == 3) {  // CONCISE_HEALTH.
      TensorShape shape({5});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      ITEX_GPU_stream->fill<Tout>(output_tensor->flat<Tout>().data(), Tout(0),
                                  5);
      const Tout static_output[] = {tensor_id, num_elem};
      ITEX_GPU_stream
          ->memcpy(output_tensor->data(), &static_output, 2 * sizeof(Tout))
          .wait();

      if (num_elem == 0) return;

      auto total_items =
          ITEX_GPU_stream->get_device()
              .template get_info<sycl::info::device::max_work_group_size>();

      ITEX_GPU_stream->submit([&](sycl::handler& cgh) {
        auto input_data = input.data();
        auto input_size = input.size();
        auto output = output_tensor->flat<Tout>().data() + 2;
        ConciseHealthKernel<Tin, Tout> task(input_size, total_items, input_data,
                                            output);
        cgh.parallel_for<ConciseHealthKernel<Tin, Tout>>(
            sycl::range<1>(total_items), task);
      });
    } else if (tensor_debug_mode_ == 4) {  // FULL HEALTH
      TensorShape shape({11});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      ITEX_GPU_stream->fill<Tout>(output_tensor->flat<Tout>().data(), Tout(0),
                                  11);

      int num_dims = tensor.dims();
      const Tout static_output[] = {tensor_id, -1.0,
                                    static_cast<Tout>(tensor.dtype()),
                                    static_cast<Tout>(num_dims), num_elem};
      ITEX_GPU_stream
          ->memcpy(output_tensor->data(),
                   static_cast<const void*>(static_output), 5 * sizeof(Tout))
          .wait();

      if (num_elem == 0) return;

      auto total_items =
          ITEX_GPU_stream->get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      ITEX_GPU_stream->submit([&](sycl::handler& cgh) {
        auto input_data = input.data();
        auto input_size = input.size();
        auto output = output_tensor->flat<Tout>().data() + 5;
        FullHealthKernel<Tin, Tout> task(input_size, total_items, input_data,
                                         output);
        cgh.parallel_for<FullHealthKernel<Tin, Tout>>(
            sycl::range<1>(total_items), task);
      });
    } else if (tensor_debug_mode_ == 5) {  // SHAPE
      TensorShape shape({10});

      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      ITEX_GPU_stream->fill<Tout>(output_tensor->flat<Tout>().data(), Tout(0),
                                  11);

      int num_dims = tensor.dims();
      Tout static_output[10] = {tensor_id,
                                static_cast<Tout>(tensor.dtype()),
                                static_cast<Tout>(num_dims),
                                num_elem,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0};
      // Tensor shape: right pad zeros, truncate head
      int dim_idx = 4;
      for (int i = std::max(0, num_dims - 6); i < num_dims; ++i) {
        static_output[dim_idx++] = static_cast<Tout>(tensor.dim_size(i));
      }

      ITEX_GPU_stream
          ->memcpy(output_tensor->data(),
                   static_cast<const void*>(static_output), 10 * sizeof(Tout))
          .wait();
    } else if (tensor_debug_mode_ == 8) {  // REDUCE_INF_NAN_THREE_SLOTS.
      TensorShape shape({3});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      ITEX_GPU_stream->fill<Tout>(output_tensor->flat<Tout>().data(), Tout(0),
                                  output_tensor->flat<Tout>().size());

      if (num_elem == 0) return;

      auto total_items =
          ITEX_GPU_stream->get_device()
              .template get_info<sycl::info::device::max_work_group_size>();

      ITEX_GPU_stream->submit([&](sycl::handler& cgh) {
        auto input_data = input.data();
        auto input_size = input.size();
        auto output = output_tensor->flat<Tout>().data();
        ReduceInfNanThreeSlotsKernel<Tin, Tout> task(input_size, total_items,
                                                     input_data, output);
        cgh.parallel_for<ReduceInfNanThreeSlotsKernel<Tin, Tout>>(
            sycl::range<1>(total_items), task);
      });
    } else {
      context->SetStatus(errors::Unimplemented(
          "Unimplemented tensor debug mode: ", tensor_debug_mode_));
    }
  }

 private:
  int tensor_debug_mode_;
  int64 tensor_id_;
  static constexpr int64 kMaxTensorId = 1L << std::numeric_limits<Tout>::digits;
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_DEBUG_OPS_H_
