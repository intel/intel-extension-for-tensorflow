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

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::algorithm;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::reorder;

namespace itex {
template <typename Device, typename T>
class OneDnnToTfOp : public OpKernel {
 public:
  explicit OneDnnToTfOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const size_t kSrcIndex = 0;  // index of src input tensor
    const size_t kDstIndex = 0;  // index of dst output tensor

    const Tensor& src_tensor = context->input(kSrcIndex);
    OneDnnShape src_shape;
    GetOneDnnShape(context, kSrcIndex, &src_shape);

    // If input is already in Tf format, then set input tensor to output.
    if (!src_shape.IsOneDnnTensor()) {
      context->set_output(kDstIndex, src_tensor);
      ITEX_VLOG(3) << "OneDnnToTfOp: No conversion needed, "
                   << "setting input to output";
      return;
    }

    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      // Get OneDnn memory desc of input tensor.
      auto src_onednn_md = src_shape.GetOneDnnLayout();
      // Get Tf memory desc of input tensor, in other words, the expected output
      // layout after onednntotf
      auto src_tf_md = src_shape.GetTfLayout();

      // Allocate output tensor.
      TensorShape dst_shape = src_shape.GetTfShape();

      if (src_onednn_md == src_tf_md) {
        Tensor dst_tensor;
        // if onednn layout and tf layout are the same, just forward tensor
        ITEX_VLOG(3) << "OneDnnToTfOp: Input tensor is plain layout, but "
                        "IsOneDnnTensor() = True. The implementation of the op "
                        "before _OneDnnTotf may be improved";
        ITEX_CHECK(dst_tensor.CopyFrom(src_tensor, dst_shape));
        context->set_output(kDstIndex, dst_tensor);
        return;
      }

      Tensor* dst_tensor;
      // Allocate new buffer for output tensor
      OP_REQUIRES_OK(
          context, context->allocate_output(kDstIndex, dst_shape, &dst_tensor));

      // No matter src_onednn_md is equal to src_tf_md or not, use `onednn
      // reorder` to reorder or copy input tensor to output tensor
      const T* src_data = src_tensor.flat<T>().data();
      T* dst_data = dst_tensor->flat<T>().data();
      auto src_mem =
          CreateDnnlMemory(src_onednn_md, onednn_engine,
                           static_cast<void*>(const_cast<T*>(src_data)));
      auto reorder_mem = CreateDnnlMemory(src_tf_md, onednn_engine,
                                          static_cast<void*>(dst_data));
      ReorderMemory(*context, &src_mem, &reorder_mem, onednn_engine);
    } catch (dnnl::error& e) {
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception: Status: ", e.status,
                          ", message: ", StringPiece(e.message), ", in file ",
                          __FILE__, ":", __LINE__));
    }
  }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                            \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnToTf")            \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("input_meta"), \
                          OneDnnToTfOp<GPUDevice, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
#else
#define REGISTER_KERNEL(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OneDnnToTf").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnToTfOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
#endif  // INTEL_CPU_ONLY

}  // namespace itex
