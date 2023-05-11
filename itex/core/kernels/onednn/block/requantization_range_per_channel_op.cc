/* Copyright (c) 2021-2022 Intel Corporation

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

#include <cmath>
#include <limits>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename Device, typename T>
class OneDnnRequantizationRangePerChannelOp : public OpKernel {
 public:
  explicit OneDnnRequantizationRangePerChannelOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("clip_value_max", &clip_value_max_));
  }

  void Compute(OpKernelContext* context) override {
    // TODO(itex): This kernel implementation now only supports plain NHWC
    // format. Deal with the block layout for GPU
    const Tensor& input = context->input(kInputTensorIndex);
    const Tensor& input_min = context->input(kInputMinIndex);
    const Tensor& input_max = context->input(kInputMaxIndex);

    const size_t depth = input_max.NumElements();
    OP_REQUIRES(
        context, input_min.dim_size(0) == depth,
        errors::InvalidArgument("input_min has incorrect size, expected ",
                                depth, " was ", input_min.dim_size(0)));
    OP_REQUIRES(
        context, input_max.dim_size(0) == depth,
        errors::InvalidArgument("input_max has incorrect size, expected ",
                                depth, " was ", input_max.dim_size(0)));

    const float* input_min_data = input_min.flat<float>().data();
    const float* input_max_data = input_max.flat<float>().data();
    std::vector<float> ranges(depth);
    bool is_non_negative = true;
    Eigen::array<int, 2> shuffling({1, 0});
    auto input_matrix = input.flat_inner_dims<qint32>();

    // TODO(itex): verify performance of not transposing and finding the min
    // max directly from input_matrix vs the one presented below of transposing
    // and using the transposed matrix as the transposing operation in itself
    // might be more costly. Note that this operation is a calibration step for
    // quantization and will cease to exist in the final inference graph(will
    // exist as a const node).

    // TODO(itex): make clear why we shuffle index 0 and 1 here? The kernel
    // implemenation works well now. But for NHWC int8 Convolution, the channel
    // dim seems to be 3?
    auto transposed_input = input_matrix.shuffle(shuffling);

    // Find the ranges of each channel in parallel.
    float out_min_max = std::numeric_limits<float>::min();

    // Note: this kernel is only used for quantization calibration, and is not
    // included in final generated pb. So the kernel performance is not very
    // important and additional memcpy is ok here.

    Device d = context->eigen_device<Device>();

#ifndef INTEL_CPU_ONLY
    // Minimum and maxinum for each channel, tensor on device
    Tensor min_device, max_device;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<qint32>::v(),
                                          TensorShape({}), &min_device));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<qint32>::v(),
                                          TensorShape({}), &max_device));
#endif  // INTEL_CPU_ONLY

    // Minimum and maxinum for each channel, tensor on host.
    // The scale passed to OneDnn need to be float on host.
    Tensor min_host, max_host;
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<qint32>::v(),
                                                   TensorShape({}), &min_host,
                                                   alloc_attr));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<qint32>::v(),
                                                   TensorShape({}), &max_host,
                                                   alloc_attr));

    // TODO(itex): Add parallel_for for both CPU and GPU
    for (int64_t i = 0; i < depth; ++i) {
#ifndef INTEL_CPU_ONLY
      auto* ITEX_GPU_stream = context->GetDeviceStream();
      min_device.flat<qint32>().device(d) =
          transposed_input.chip<0>(i).minimum();
      max_device.flat<qint32>().device(d) =
          transposed_input.chip<0>(i).maximum();

      ITEX_GPU_stream->memcpy(min_host.flat<qint32>().data(),
                              min_device.flat<qint32>().data(), sizeof(qint32));
      ITEX_GPU_stream
          ->memcpy(max_host.flat<qint32>().data(),
                   max_device.flat<qint32>().data(), sizeof(qint32))
          .wait();
#else
      min_host.flat<qint32>().device(d) = transposed_input.chip<0>(i).minimum();
      max_host.flat<qint32>().device(d) = transposed_input.chip<0>(i).maximum();
#endif  // INTEL_CPU_ONLY

      const int32_t min_per_channel = min_host.flat<qint32>()(0);
      const int32_t max_per_channel = max_host.flat<qint32>()(0);

      const int32_t abs_max =
          std::max(std::abs(min_per_channel), std::abs(max_per_channel));
      float scale =
          std::max(std::abs(input_min_data[i]), std::abs(input_max_data[i]));
      ranges[i] =
          scale * static_cast<float>(abs_max) / static_cast<float>(1L << 31);
      if (min_per_channel < 0) is_non_negative = false;

      // Thread-local out_min_max.
      out_min_max = std::max(out_min_max, ranges[i]);
    }

    // All local out_min_max gets max-reduced into one global out_min_max at
    // the end of the loop by specifying reduction(max:out_min_max) along with
    // omp parallel for.

    // Fixing max to clip_value_max_ (example 6.0 to support relu6)
    if (out_min_max > clip_value_max_) out_min_max = clip_value_max_;

    Tensor* output_min = nullptr;
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(kOutputMinIndex, {}, &output_min));
    OP_REQUIRES_OK(context,
                   context->allocate_output(kOutputMaxIndex, {}, &output_max));
    output_min->flat<float>()(0) = is_non_negative ? 0.0f : -out_min_max;
    output_max->flat<float>()(0) = out_min_max;
  }

 private:
  float clip_value_max_ = std::numeric_limits<float>::infinity();
  const int kInputTensorIndex = 0;
  const int kInputMinIndex = 1;
  const int kInputMaxIndex = 2;
  const int kOutputMinIndex = 0;
  const int kOutputMaxIndex = 1;
};

// TODO(itex): May rename the op to _OneDnnRequantizationRangePerChannel,
// when we take intel-tensorflow as tf-proper backend
#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)               \
  REGISTER_KERNEL_BUILDER(                  \
      Name("RequantizationRangePerChannel") \
          .Device(DEVICE_GPU)               \
          .TypeConstraint<TYPE>("T")        \
          .HostMemory("input_min")          \
          .HostMemory("input_max")          \
          .HostMemory("output_min")         \
          .HostMemory("output_max"),        \
      OneDnnRequantizationRangePerChannelOp<GPUDevice, TYPE>)
TF_CALL_qint32(REGISTER_KERNEL);
#endif  // INTEL_CPU_ONLY

}  // namespace itex
