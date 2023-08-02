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
using dnnl::softmax_forward;

namespace itex {
template <typename Device, typename T>
class OneDnnSoftmaxOp : public OpKernel {
 public:
  explicit OneDnnSoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = context->input(src_index);
    OneDnnShape src_onednn_shape;
    GetOneDnnShape(context, src_index, &src_onednn_shape);
    TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                   ? src_onednn_shape.GetTfShape()
                                   : src_tensor.shape();
    const int input_dims = src_tf_shape.dims();

    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      int axis;
      memory::dims src_dims;
      memory::desc src_md;

      // Create src md
      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
        // axis in TF order
        axis = input_dims - 1;
        // axis in OneDnn order
        axis = src_onednn_shape.TfDimIdx(axis);
        src_md = src_onednn_shape.GetOneDnnLayout();
      } else {
        src_dims = TFShapeToOneDnnDims(src_tf_shape);
        axis = input_dims - 1;
        // Create `plain` onednn memory descriptor
        src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
      }

      // Create softmax primitive
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      auto fwd_pd = softmax_forward::primitive_desc(
          onednn_engine, prop_kind::forward_training,
          dnnl::algorithm::softmax_accurate, src_md, src_md, axis, attr);
      auto fwd_primitive = softmax_forward(fwd_pd);

      // Create src memory
      T* src_data =
          static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
      auto src_mem = CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                                      static_cast<void*>(src_data));

      // Set output layout
      OneDnnShape dst_onednn_shape;
      TensorShape dst_tf_shape;

      dst_tf_shape = src_tf_shape;
      SetOutputTensorShape(
          fwd_pd.dst_desc(), src_onednn_shape.GetTfDataFormat(), &dst_tf_shape,
          &dst_onednn_shape, src_onednn_shape.IsOneDnnTensor());

      // Allocate output data tensor and meta tensor
      Tensor* dst_tensor = nullptr;

      // If the input and output tensor shape is always the same, try to use
      // `ForwardOrAllocateOutputSetOneDnnShape(`. Otherwise, please use
      // `AllocateOutputSetOneDnnShape`
      ForwardOrAllocateOutputSetOneDnnShape(context, src_index, dst_index,
                                            &dst_tensor, dst_tf_shape,
                                            dst_onednn_shape);

      // Create dst memory
      T* dst_data = dst_tensor->flat<T>().data();
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      static_cast<void*>(dst_data));

      Tensor scratchpad_tensor;
      int64 scratchpad_size = fwd_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(fwd_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      // Execute softmax primitive
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> fwd_primitive_args = {
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_DST, dst_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
      fwd_primitive.execute(onednn_stream, fwd_primitive_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                              \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnSoftmax")           \
                              .Device(DEVICE_GPU)          \
                              .TypeConstraint<TYPE>("T")   \
                              .HostMemory("logits_meta")   \
                              .HostMemory("softmax_meta"), \
                          OneDnnSoftmaxOp<GPUDevice, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#else
#define REGISTER_KERNEL(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_OneDnnSoftmax").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnSoftmaxOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);

#endif  // INTEL_CPU_ONLY
}  // namespace itex
