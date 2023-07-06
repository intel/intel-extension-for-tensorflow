/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/compiler/xla/service/gpu/gpu_conv_runner.h"

#include <string>

#include "itex/core/compiler/xla/service/gpu/stream_executor_util.h"
namespace itex_xla {
namespace gpu {

Status RunGpuConv(const OneDnnConvPrimitive& onednn_primitive,
                  const GpuConvDescriptor& conv_descriptor,
                  absl::Span<const se::DeviceMemoryBase> operand_buffers,
                  se::DeviceMemoryBase result_buffer,
                  const Thunk::ExecuteParams& params) {
  void* input_data;
  void* filter_data;
  void* output_data;
  void* bias_data = nullptr;
  void* side_input_data = nullptr;

  switch (conv_descriptor.kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
      input_data = const_cast<void*>(operand_buffers[0].opaque());
      filter_data = const_cast<void*>(operand_buffers[1].opaque());
      output_data = const_cast<void*>(result_buffer.opaque());
      break;
    case CudnnConvKind::kBackwardInput:
      input_data = const_cast<void*>(result_buffer.opaque());
      filter_data = const_cast<void*>(operand_buffers[1].opaque());
      output_data = const_cast<void*>(operand_buffers[0].opaque());

      break;
    case CudnnConvKind::kBackwardFilter:
      input_data = const_cast<void*>(operand_buffers[0].opaque());
      filter_data = const_cast<void*>(result_buffer.opaque());
      output_data = const_cast<void*>(operand_buffers[1].opaque());
      break;
    default:
      return InternalError("Unkown convolution kind");
  }

  if (conv_descriptor.kind == CudnnConvKind::kForwardActivation) {
    bias_data = const_cast<void*>(operand_buffers[2].opaque());
    if (operand_buffers.size() >= 4) {
      side_input_data = const_cast<void*>(operand_buffers[3].opaque());
    }
  }
  onednn_primitive.src_memory.set_data_handle(input_data);
  onednn_primitive.filter_memory.set_data_handle(filter_data);
  onednn_primitive.dst_memory.set_data_handle(output_data);
  if (bias_data != nullptr) {
    onednn_primitive.bias_memory.set_data_handle(bias_data);
  }
  try {
    if (conv_descriptor.kind == CudnnConvKind::kForward ||
        conv_descriptor.kind == CudnnConvKind::kForwardActivation) {
      if (onednn_primitive.has_reorder) {
        onednn_primitive.filter_reorder_primitive.execute(
            onednn_primitive.stream, onednn_primitive.reorder_args);
      }
      onednn_primitive.fwd_primitive.execute(
          onednn_primitive.stream, onednn_primitive.fwd_primitives_args);
    } else if (conv_descriptor.kind == CudnnConvKind::kBackwardInput) {
      if (onednn_primitive.has_reorder) {
        onednn_primitive.filter_reorder_primitive.execute(
            onednn_primitive.stream, onednn_primitive.reorder_args);
      }
      onednn_primitive.bwd_input_primitive.execute(
          onednn_primitive.stream, onednn_primitive.bwd_input_primitive_args);
    } else if (conv_descriptor.kind == CudnnConvKind::kBackwardFilter) {
      onednn_primitive.bwd_filter_primitive.execute(
          onednn_primitive.stream, onednn_primitive.bwd_filter_primitive_args);
      if (onednn_primitive.has_reorder) {
        onednn_primitive.filter_reorder_primitive.execute(
            onednn_primitive.stream, onednn_primitive.reorder_args);
      }
    } else {
      return InternalError("Unkown convolutuion kind");
    }
  } catch (dnnl::error& e) {
    std::string error_msg = "Status: " + std::to_string(e.status) +
                            ", message: " + std::string(e.message) +
                            ", in file " + std::string(__FILE__) + ":" +
                            std::to_string(__LINE__);
    std::cout << error_msg << std::endl;
  }
}

}  // namespace gpu
}  // namespace itex_xla
