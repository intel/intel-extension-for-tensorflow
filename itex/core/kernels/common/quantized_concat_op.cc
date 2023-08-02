/* Copyright (c) 2022 Intel Corporation

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

#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {

template <typename Device, typename T>
class QuantizedConcatOp : public OpKernel {
 public:
  ~QuantizedConcatOp() {}
  explicit QuantizedConcatOp(OpKernelConstruction* context)
      : OpKernel(context), axis_attribute_name_("axis") {}

  void Compute(OpKernelContext* context) override {
    // TODO(itex): It seems "Concat" is a dead core op, only "ConcatV2" is
    // used. We may not need to deal with values_input_start_index_ at all, just
    // consider the starting index is always 0;

    // num_inputs include input/axis tensor and input/axis meta tensors
    const int num_inputs = context->num_inputs();

    OP_REQUIRES(
        context, num_inputs > 3,
        errors::InvalidArgument("Number of values must larger than 1, but got ",
                                num_inputs - 1));
    // "Concat" and "ConcatV2" have different input order
    values_input_start_index_ = 0;
    values_input_end_index_ = (num_inputs - 1) / 3 - 1;
    // Values input is from 0 to N-1.
    // In quantized case, N = (num_inputs - 1) / 3.
    // See the OP definition in onednn_nn_ops.cc.
    axis_input_index_ = values_input_end_index_ + 1;

    const Tensor& concat_dim_tensor = context->input(axis_input_index_);

    OP_REQUIRES(context,
                (TensorShapeUtils::IsScalar(concat_dim_tensor.shape()) ||
                 (TensorShapeUtils::IsVector(concat_dim_tensor.shape()) &&
                  concat_dim_tensor.shape().dim_size(0) == 1)),
                errors::InvalidArgument(
                    axis_attribute_name_,
                    " tensor should be a scalar integer, but got shape ",
                    concat_dim_tensor.shape().DebugString()));
    int64 concat_dim;
    // In case of ConcatV2, "axis" could be int32 or int64
    OP_REQUIRES(
        context,
        (concat_dim_tensor.dtype() == DT_INT32 ||
         concat_dim_tensor.dtype() == DT_INT64),
        errors::InvalidArgument(axis_attribute_name_,
                                " tensor should be int32 or int64, but got ",
                                DataTypeString(concat_dim_tensor.dtype())));

    if (concat_dim_tensor.dtype() == DT_INT32) {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int32>()());
    } else {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int64>()());
    }

    // num of actual input tensors
    const int N = values_input_end_index_ - values_input_start_index_ + 1;

    const Tensor& src0_tensor = context->input(values_input_start_index_);
    const TensorShape src0_shape = src0_tensor.shape();
    const int input_dims = src0_shape.dims();
    // Here axis_in_eigen is in Eigen plain layout sequence
    int32 axis_in_eigen = concat_dim < 0 ? concat_dim + input_dims : concat_dim;

    // concat_dim==0 allows concatenating a list of scalars into a vector.
    OP_REQUIRES(
        context,
        (0 <= axis_in_eigen && axis_in_eigen < input_dims) || concat_dim == 0,
        errors::InvalidArgument(
            "ConcatOp : Expected concatenating dimensions in the range "
            "[",
            -input_dims, ", ", input_dims, "), but got ", concat_dim));

    int64 output_concat_dim = 0;

    bool all_empty_inputs = true;

    // Sanity check for input tensors shape and calculate output concat dim
    for (int i = 0; i < N; ++i) {
      const Tensor& src_tensor = context->input(values_input_start_index_ + i);
      const TensorShape src_tf_shape = src_tensor.shape();
      all_empty_inputs = all_empty_inputs && (src_tf_shape.num_elements() == 0);
      OP_REQUIRES(
          context, src_tf_shape.dims() == input_dims,
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              src_tf_shape.DebugString(), " vs. shape[", i,
              "] = ", src_tf_shape.DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == axis_in_eigen) {
          continue;
        }
        OP_REQUIRES(
            context, src_tf_shape.dim_size(j) == src_tf_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                src_tf_shape.DebugString(), " vs. shape[", i,
                "] = ", src_tf_shape.DebugString()));
      }
      output_concat_dim +=
          src_tf_shape.dims() > 0 ? src_tf_shape.dim_size(axis_in_eigen) : 1;
    }

    float input_min = 0, input_max = 0;
    input_min = context->input(N + 1).flat<float>()(0);
    input_max = context->input(2 * N + 1).flat<float>()(0);
    const float eps = 1.0e-6;
    float min, max;
    for (int i = 1; i < N; ++i) {
      min = context->input(N + 1 + i).flat<float>()(0);
      max = context->input(2 * N + 1 + i).flat<float>()(0);
      if (fabs(input_min - min) > eps || fabs(input_max - max) > eps) {
        OP_REQUIRES_OK(context,
                       errors::Aborted("TODO: Not implemented the case "
                                       "unequal input_mins / input_maxes."));
        // TODO(itex): support such condition in the future.
        break;
      }
    }

    try {
      Tensor* dst_tensor = nullptr;
      TensorShape output_tf_shape;

      const int kOutputIdx = 0;
      // Nothing to compute, return.
      if (all_empty_inputs) {
        // Although elements are 0 anyway, the output shape is changed.
        output_tf_shape = src0_shape;
        output_tf_shape.set_dim(axis_in_eigen, output_concat_dim);
        OP_REQUIRES_OK(context, context->allocate_output(
                                    kOutputIdx, output_tf_shape, &dst_tensor));
        return;
      }

      // Create memory descriptor for OneDnn.
      // If all input in Tensorflow format, create block memory descriptor,
      // else convert TF format to OneDnn memory descriptor
      std::vector<dnnl::memory::desc> srcs_pd;
      for (int src_idx = 0; src_idx < N; ++src_idx) {
        dnnl::memory::desc src_md;
        const Tensor& src_tensor =
            context->input(src_idx + values_input_start_index_);
        auto dims = TFShapeToOneDnnDims(src_tensor.shape());
        src_md = CreatePlainMemDescWithFormatTag<T>(dims);
        srcs_pd.push_back(src_md);
      }
      dnnl::memory::dims dst_dims = srcs_pd[0].get_dims();
      // Only difference between output dims and each input dims is the concat
      // dim
      dst_dims[axis_in_eigen] = output_concat_dim;

      // Allocate output
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      dnnl::concat::primitive_desc concat_pd;

      // For all plain layout input, we need to explicitly choose the output
      // memory desc, otherwise OneDnn may automatically choose format we
      // don't want. E.g. the inputs are all default format "abcd", the output
      // format chosen by OneDnn may be "acdb"
      dnnl::memory::desc dst_md = CreatePlainMemDescWithFormatTag<T>(dst_dims);
      concat_pd = dnnl::concat::primitive_desc(onednn_engine, dst_md,
                                               axis_in_eigen, srcs_pd, attr);
      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          concat_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(concat_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));

      output_tf_shape = OneDnnDimsToTFShape(dst_dims);

      OP_REQUIRES_OK(context, context->allocate_output(
                                  kOutputIdx, output_tf_shape, &dst_tensor));

      // Create Concat op, and submit for execution.
      dnnl::concat concat_prim = dnnl::concat(concat_pd);
      dnnl::memory dst_mem = CreateDnnlMemory(
          concat_pd.dst_desc(), onednn_engine, GetTensorBuffer<T>(dst_tensor));
      std::unordered_map<int, dnnl::memory> net_args = {
          {DNNL_ARG_DST, dst_mem}, {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
      for (int src_idx = 0; src_idx < N; ++src_idx) {
        dnnl::memory src_mem = CreateDnnlMemory(
            concat_pd.src_desc(src_idx), onednn_engine,
            GetTensorBuffer<T>(
                &context->input(src_idx + values_input_start_index_)));
        net_args.insert({DNNL_ARG_MULTIPLE_SRC + src_idx, src_mem});
      }

      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      concat_prim.execute(onednn_stream, net_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
    context->set_output(1, context->input(N + 1));
    context->set_output(2, context->input(N * 2 + 1));
  }

 private:
  const char* const axis_attribute_name_;
  int values_input_start_index_;
  int values_input_end_index_;
  int axis_input_index_;
};

// TODO(itex): Add completed implementation for GPU in the future
#ifndef INTEL_CPU_ONLY
REGISTER_KERNEL_BUILDER(Name("QuantizedConcatV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("axis")
                            .HostMemory("input_mins")
                            .HostMemory("input_maxes")
                            .HostMemory("output_min")
                            .HostMemory("output_max")
                            .TypeConstraint<quint8>("T"),
                        QuantizedConcatOp<GPUDevice, quint8>);

REGISTER_KERNEL_BUILDER(Name("QuantizedConcatV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("axis")
                            .HostMemory("input_mins")
                            .HostMemory("input_maxes")
                            .HostMemory("output_min")
                            .HostMemory("output_max")
                            .TypeConstraint<qint8>("T"),
                        QuantizedConcatOp<GPUDevice, qint8>);
#else
REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizedConcatV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T"),
                        QuantizedConcatOp<CPUDevice, quint8>);

REGISTER_KERNEL_BUILDER(Name("_ITEXQuantizedConcatV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T"),
                        QuantizedConcatOp<CPUDevice, qint8>);
#endif
}  // namespace itex
