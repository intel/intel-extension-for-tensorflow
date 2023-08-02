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

#include "itex/core/kernels/common/host_data_cache.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::memory;
using dnnl::primitive;

namespace itex {
template <typename Device, typename Toutput>
class OneDnnRequantizePerChannelOp : public OpKernel {
 public:
  explicit OneDnnRequantizePerChannelOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));
    OP_REQUIRES(context, out_type_ == DT_QINT8 || out_type_ == DT_QUINT8,
                errors::InvalidArgument(
                    "out_type must be qint8 or quint8, but got: ", out_type_));
  }
  void Compute(OpKernelContext* context) override {
    try {
      // TODO(itex): This kernel implementation now only supports plain NHWC
      // format. Deal with the block layout for GPU
      const Tensor& input = context->input(kInputTensorIndex);
      const Tensor& input_min_vec = context->input(kInputMinVecIndex);
      float* input_min_vec_data =
          const_cast<float*>(input_min_vec.flat<float>().data());
      const Tensor& input_max_vec = context->input(kInputMaxVecIndex);
      float* input_max_vec_data =
          const_cast<float*>(input_max_vec.flat<float>().data());

      const Tensor& input_requested_min =
          context->input(this->kRequestMinIndex);
      const float input_requested_min_float =
          input_requested_min.flat<float>()(0);
      const Tensor& input_requested_max =
          context->input(this->kRequestMaxIndex);
      const float input_requested_max_float =
          input_requested_max.flat<float>()(0);

      size_t depth = input_min_vec.NumElements();
      OP_REQUIRES(
          context, input.dims() == 4,
          errors::InvalidArgument("Current RequantizePerChannel operator"
                                  "supports 4D tensors only."));
      OP_REQUIRES(
          context, input_min_vec.dim_size(0) == depth,
          errors::InvalidArgument("input_min has incorrect size, expected ",
                                  depth, " was ", input_min_vec.dim_size(0)));
      OP_REQUIRES(
          context, input_max_vec.dim_size(0) == depth,
          errors::InvalidArgument("input_max has incorrect size, expected ",
                                  depth, " was ", input_max_vec.dim_size(0)));

      if (out_type_ == DT_QINT8) ITEX_DCHECK(input_requested_min_float < 0.0f);

      const float factor = (out_type_ == DT_QINT8) ? 127.0f : 255.0f;
      const float requested_min_max =
          std::max(std::abs(input_requested_min_float),
                   std::abs(input_requested_max_float));
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(kOutputTensorIndex,
                                                       input.shape(), &output));

      std::vector<float> scales(depth);
      for (int i = 0; i < depth; ++i) {
        float min_max_from_vec = std::max(std::abs(input_min_vec_data[i]),
                                          std::abs(input_max_vec_data[i]));
        scales[i] = factor * (min_max_from_vec / requested_min_max /
                              static_cast<float>(1L << 31));
      }

      dnnl::primitive_attr reorder_attr;
      reorder_attr.set_scales_mask(DNNL_ARG_SRC, 2);

      memory::dims dims_onednn_order =
          TFShapeToOneDnnDimsInNC(input.shape(), FORMAT_NHWC);
      memory::desc input_md = memory::desc(
          dims_onednn_order, OneDnnType<qint32>(), memory::format_tag::nhwc);
      memory::desc output_md =
          (out_type_ == DT_QINT8)
              ? memory::desc(dims_onednn_order, OneDnnType<qint8>(),
                             memory::format_tag::nhwc)
              : memory::desc(dims_onednn_order, OneDnnType<quint8>(),
                             memory::format_tag::nhwc);

      void* input_buf =
          static_cast<void*>(const_cast<qint32*>(input.flat<qint32>().data()));
      void* output_buf;
      if (out_type_ == DT_QINT8) {
        output_buf = static_cast<void*>(
            const_cast<qint8*>(output->flat<qint8>().data()));
      } else {
        output_buf = static_cast<void*>(
            const_cast<quint8*>(output->flat<quint8>().data()));
      }
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      float* output_scale_ptr =
          output_scale_cache_.GetCachedPtr(context, scales.data(), depth);
      dnnl::memory output_scales_mem({{static_cast<dnnl_dim_t>(depth)},
                                      dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::x},
                                     onednn_engine,
                                     reinterpret_cast<void*>(output_scale_ptr));
      auto src_mem = CreateDnnlMemory(input_md, onednn_engine, input_buf);
      auto dst_mem = CreateDnnlMemory(output_md, onednn_engine, output_buf);

      dnnl::reorder reorder_prim =
          dnnl::reorder(src_mem, dst_mem, reorder_attr);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> reorder_args = {
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_DST, dst_mem},
          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, output_scales_mem},
      };
      reorder_prim.execute(onednn_stream, reorder_args);

      Tensor* output_min = nullptr;
      Tensor* output_max = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(kOutputMinIndex, {}, &output_min));
      OP_REQUIRES_OK(
          context, context->allocate_output(kOutputMaxIndex, {}, &output_max));

      output_min->flat<float>()(0) = input_requested_min_float;
      output_max->flat<float>()(0) = input_requested_max_float;
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + std::string(e.message) + ", in file " +
                         std::string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  const int kInputTensorIndex = 0;
  const int kInputMinVecIndex = 1;
  const int kInputMaxVecIndex = 2;
  const int kRequestMinIndex = 3;
  const int kRequestMaxIndex = 4;
  const int kOutputTensorIndex = 0;
  const int kOutputMinIndex = 1;
  const int kOutputMaxIndex = 2;
  // TODO(itex): use template para T instead of out_type_
  DataType out_type_;
  HostDataCache<Device, float> output_scale_cache_;
};

// TODO(itex): Enable OneDnnRequantizePerChannel for CPUDevice
#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(Name("RequantizePerChannel")            \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<qint32>("T")        \
                              .TypeConstraint<TYPE>("out_type")   \
                              .HostMemory("input_min")            \
                              .HostMemory("input_max")            \
                              .HostMemory("requested_output_min") \
                              .HostMemory("requested_output_max") \
                              .HostMemory("output_min")           \
                              .HostMemory("output_max"),          \
                          OneDnnRequantizePerChannelOp<GPUDevice, TYPE>)
TF_CALL_qint8(REGISTER_KERNEL);
TF_CALL_quint8(REGISTER_KERNEL);
#endif

}  // namespace itex
