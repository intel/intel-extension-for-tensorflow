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

#include "itex/core/kernels/common/host_data_cache.h"
#include "itex/core/kernels/common/no_ops.h"
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

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

template <typename Device, typename T, AxisArgumentName AxisArgName>
class OneDnnConcatOp : public OpKernel {
 public:
  ~OneDnnConcatOp() {}
  explicit OneDnnConcatOp(OpKernelConstruction* context)
      : OpKernel(context),
        is_v2_(AxisArgName == NAME_IS_AXIS),
        axis_attribute_name_(is_v2_ ? "axis" : "concat_dim"),
        quantized_input(std::is_same<T, qint8>::value ||
                        std::is_same<T, quint8>::value) {}

  // Return first tensor index which is in OneDnn layout, or -1 with no OneDnn
  // input.
  int FindOneDnnInputIndex(OpKernelContext* context) {
    int onednn_index = -1;
    const int N = quantized_input ? (context->num_inputs() / 2 - 1) / 3
                                  : context->num_inputs() / 2 - 1;

    OneDnnShape src_onednn_shape;
    for (size_t i = 0; i < N; ++i) {
      GetOneDnnShape(context, values_input_start_index_ + i, &src_onednn_shape);
      if (src_onednn_shape.IsOneDnnTensor()) {
        onednn_index = i;
        break;
      }
    }

    return onednn_index;
  }

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
    if (is_v2_) {
      values_input_start_index_ = 0;
      values_input_end_index_ =
          quantized_input ? (num_inputs - 2) / 6 - 1 : num_inputs / 2 - 2;
      // Values input is from 0 to N-1.
      // In quantized case, N = (num_inputs - 2) / 6.
      // See the OP definition in onednn_nn_ops.cc.
      axis_input_index_ = values_input_end_index_ + 1;
    } else {
      axis_input_index_ = 0;
      values_input_start_index_ = 1;
      values_input_end_index_ = num_inputs / 2 - 1;
      if (quantized_input) {
        OP_REQUIRES_OK(context,
                       errors::Aborted("Quantized Concat V1 not supported "));
      }
    }

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
    if (is_v2_) {
      OP_REQUIRES(
          context,
          (concat_dim_tensor.dtype() == DT_INT32 ||
           concat_dim_tensor.dtype() == DT_INT64),
          errors::InvalidArgument(axis_attribute_name_,
                                  " tensor should be int32 or int64, but got ",
                                  DataTypeString(concat_dim_tensor.dtype())));
    } else {
      OP_REQUIRES(context, (concat_dim_tensor.dtype() == DT_INT32),
                  errors::InvalidArgument(
                      axis_attribute_name_, " tensor should be int32, but got ",
                      DataTypeString(concat_dim_tensor.dtype())));
    }

    if (concat_dim_tensor.dtype() == DT_INT32) {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int32>()());
    } else {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int64>()());
    }

    // num of actual input tensors
    const int N = values_input_end_index_ - values_input_start_index_ + 1;

    // TODO(itex): implement fall back to Eigen, whether memory desc are
    // different

    const Tensor& src0_tensor = context->input(values_input_start_index_);
    OneDnnShape src0_onednn_shape;
    GetOneDnnShape(context, values_input_start_index_, &src0_onednn_shape);
    const TensorShape src0_shape = src0_onednn_shape.IsOneDnnTensor()
                                       ? src0_onednn_shape.GetTfShape()
                                       : src0_tensor.shape();
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
      OneDnnShape src_onednn_shape;
      GetOneDnnShape(context, values_input_start_index_ + i, &src_onednn_shape);
      const Tensor& src_tensor = context->input(values_input_start_index_ + i);
      const TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                           ? src_onednn_shape.GetTfShape()
                                           : src_tensor.shape();
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
    std::vector<float> requant_scales(N);
    if (quantized_input) {
      input_min = context->input(N + 1).flat<float>()(0);
      input_max = context->input(2 * N + 1).flat<float>()(0);
      for (int i = 1; i < N; ++i) {
        float cur_min = context->input(N + 1 + i).flat<float>()(0);
        float cur_max = context->input(2 * N + 1 + i).flat<float>()(0);
        input_min = std::min(input_min, cur_min);
        input_max = std::max(input_max, cur_max);
      }
      const float input_min_max =
          std::max(std::abs(input_min), std::abs(input_max));

      const float factor = (src0_tensor.dtype() == DT_QINT8) ? 127.0f : 255.0f;
      for (int i = 0; i < N; ++i) {
        float cur_min = context->input(N + 1 + i).flat<float>()(0);
        float cur_max = context->input(2 * N + 1 + i).flat<float>()(0);
        float cur_min_max = std::max(std::abs(cur_min), std::abs(cur_max));
        requant_scales[i] = factor * (cur_min_max / input_min_max /
                                      static_cast<float>(1L << 31));
      }
    }

    try {
      Tensor* dst_tensor = nullptr;
      TensorShape output_tf_shape;
      OneDnnShape output_onednn_shape;

      const int kOutputIdx = 0;
      // Nothing to compute, return.
      if (all_empty_inputs) {
        // Although elements are 0 anyway, the output shape is changed.
        output_onednn_shape.SetOneDnnTensor(false);
        output_tf_shape = src0_shape;
        output_tf_shape.set_dim(axis_in_eigen, output_concat_dim);
        AllocateOutputSetOneDnnShape(context, kOutputIdx, &dst_tensor,
                                     output_tf_shape, output_onednn_shape);
        return;
      }

      bool has_onednn_input = false;
      int onednn_input_index = FindOneDnnInputIndex(context);
      OneDnnTensorFormat onednn_data_format =
          OneDnnTensorFormat::FORMAT_INVALID;
      TensorFormat tf_data_format;
      OneDnnShape src_onednn_shape;
      dnnl::memory::format_tag dnn_fmt = dnnl::memory::format_tag::undef;
      // Here axis_in_eigen is in OneDnn memory desc sequence
      int axis_in_onednn;

      if (onednn_input_index >= 0) {
        has_onednn_input = true;
        GetOneDnnShape(context, onednn_input_index + values_input_start_index_,
                       &src_onednn_shape);
        // OneDnn input has the data format information.
        onednn_data_format = src_onednn_shape.GetTfDataFormat();
        if (src_onednn_shape.GetTfShape().dims() == 4 ||
            src_onednn_shape.GetTfShape().dims() == 5) {
          tf_data_format = OneDnnDataFormatToTFDataFormat(onednn_data_format);
          dnn_fmt = OneDnnTensorFormatToTag(onednn_data_format);
        }
        // Convert concat dims from Eigen axis to block OneDnn axis
        axis_in_onednn = src_onednn_shape.TfDimIdx(axis_in_eigen);
      } else {
        axis_in_onednn = axis_in_eigen;
      }

      // Create memory descriptor for OneDnn.
      // If all input in Tensorflow format, create block memory descriptor,
      // else convert TF format to OneDnn memory descriptor
      std::vector<dnnl::memory::desc> srcs_pd;
      for (int src_idx = 0; src_idx < N; ++src_idx) {
        GetOneDnnShape(context, src_idx + values_input_start_index_,
                       &src_onednn_shape);
        dnnl::memory::desc src_md;
        const Tensor& src_tensor =
            context->input(src_idx + values_input_start_index_);

        if (src_onednn_shape.IsOneDnnTensor()) {
          src_md = src_onednn_shape.GetOneDnnLayout();
        } else {
          if (has_onednn_input) {
            dnnl::memory::dims src_dims;
            if (src_tensor.dims() == 4 || src_tensor.dims() == 5) {
              src_dims = TFShapeToOneDnnDimsInNC(
                  src_tensor.shape(), tf_data_format, src_tensor.dims() == 4);
              src_md = dnnl::memory::desc(src_dims, OneDnnType<T>(), dnn_fmt);
            } else {
              src_dims = TFShapeToOneDnnDims(src_tensor.shape());
              src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
            }
          } else {
            auto dims = TFShapeToOneDnnDims(src_tensor.shape());
            src_md = CreatePlainMemDescWithFormatTag<T>(dims);
          }
        }
        srcs_pd.push_back(src_md);
      }
#ifdef ITEX_ONEDNN_3_0
      dnnl::memory::dims dst_dims = srcs_pd[0].get_dims();
#else
      dnnl::memory::dims dst_dims = srcs_pd[0].dims();
#endif
      // Only difference between output dims and each input dims is the concat
      // dim
      dst_dims[axis_in_onednn] = output_concat_dim;

      // Allocate output
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      // Set scales for each input
      if (quantized_input) {
        for (int src_idx = 0; src_idx < N; ++src_idx) {
#ifdef ITEX_ONEDNN_3_0
          attr.set_scales_mask(DNNL_ARG_MULTIPLE_SRC + src_idx, 0);
#else
          attr.set_scales(DNNL_ARG_MULTIPLE_SRC + src_idx, 0,
                          {requant_scales[src_idx]});
#endif
        }
      }

      dnnl::concat::primitive_desc concat_pd;
      if (has_onednn_input) {
#ifdef ITEX_ONEDNN_3_0
        concat_pd = dnnl::concat::primitive_desc(onednn_engine, axis_in_onednn,
                                                 srcs_pd, attr);
#else
        concat_pd = dnnl::concat::primitive_desc(axis_in_onednn, srcs_pd,
                                                 onednn_engine, attr);
#endif
      } else {
        // For all plain layout input, we need to explicitly choose the output
        // memory desc, otherwise OneDnn may automatically choose format we
        // don't want. E.g. the inputs are all default format "abcd", the output
        // format chosen by OneDnn may be "acdb"
        dnnl::memory::desc dst_md =
            CreatePlainMemDescWithFormatTag<T>(dst_dims);
#ifdef ITEX_ONEDNN_3_0
        concat_pd = dnnl::concat::primitive_desc(onednn_engine, dst_md,
                                                 axis_in_onednn, srcs_pd, attr);
#else
        concat_pd = dnnl::concat::primitive_desc(dst_md, axis_in_onednn,
                                                 srcs_pd, onednn_engine, attr);
#endif
      }

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

      SetOutputTensorShape(concat_pd.dst_desc(), onednn_data_format,
                           &output_tf_shape, &output_onednn_shape,
                           has_onednn_input);
      AllocateOutputSetOneDnnShape(context, kOutputIdx, &dst_tensor,
                                   output_tf_shape, output_onednn_shape);

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
#ifdef ITEX_ONEDNN_3_0
      if (quantized_input) {
        float* requant_scale_ptr = requant_scale_cache_.GetCachedPtr(
            context, requant_scales.data(), requant_scales.size());
        for (int src_idx = 0; src_idx < N; ++src_idx) {
          dnnl::memory cur_scales_mem(
              {{static_cast<dnnl_dim_t>(1)},
               dnnl::memory::data_type::f32,
               dnnl::memory::format_tag::x},
              onednn_engine,
              reinterpret_cast<void*>(requant_scale_ptr + src_idx));
          net_args.insert({DNNL_ARG_MULTIPLE_SRC + src_idx, cur_scales_mem});
        }
      }
#endif
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
    if (quantized_input) {
      Tensor* output_min = nullptr;
      Tensor* output_max = nullptr;
      OneDnnShape output_min_onednn_shape, output_max_onednn_shape;
      output_min_onednn_shape.SetOneDnnTensor(false);
      output_max_onednn_shape.SetOneDnnTensor(false);
      AllocateOutputSetOneDnnShape(context, 1, &output_min, {},
                                   output_min_onednn_shape);
      AllocateOutputSetOneDnnShape(context, 2, &output_max, {},
                                   output_max_onednn_shape);
      output_min->flat<float>()(0) = input_min;
      output_max->flat<float>()(0) = input_max;
    }
  }

 private:
  bool is_v2_;
  const char* const axis_attribute_name_;
  bool quantized_input;
  int values_input_start_index_;
  int values_input_end_index_;
  int axis_input_index_;
#ifdef ITEX_ONEDNN_3_0
  HostDataCache<Device, float> requant_scale_cache_;
#endif
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_CONCAT(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConcat")              \
                              .Device(DEVICE_GPU)            \
                              .HostMemory("concat_dim")      \
                              .HostMemory("concat_dim_meta") \
                              .HostMemory("values_meta")     \
                              .HostMemory("output_meta")     \
                              .TypeConstraint<T>("T"),       \
                          OneDnnConcatOp<GPUDevice, T, NAME_IS_CONCAT_DIM>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_CONCAT);
#undef REGISTER_CONCAT

#define REGISTER_CONCATV2(T)                             \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnConcatV2")        \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("axis")        \
                              .HostMemory("values_meta") \
                              .HostMemory("axis_meta")   \
                              .HostMemory("output_meta") \
                              .TypeConstraint<T>("T"),   \
                          OneDnnConcatOp<GPUDevice, T, NAME_IS_AXIS>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_CONCATV2);
#undef REGISTER_CONCATV2

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConcatV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("axis")
                            .HostMemory("input_mins")
                            .HostMemory("input_maxes")
                            .HostMemory("values_meta")
                            .HostMemory("axis_meta")
                            .HostMemory("input_mins_meta")
                            .HostMemory("input_maxes_meta")
                            .HostMemory("output_min")
                            .HostMemory("output_max")
                            .HostMemory("output_meta")
                            .HostMemory("output_min_meta")
                            .HostMemory("output_max_meta")
                            .TypeConstraint<quint8>("T"),
                        OneDnnConcatOp<GPUDevice, quint8, NAME_IS_AXIS>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConcatV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("axis")
                            .HostMemory("input_mins")
                            .HostMemory("input_maxes")
                            .HostMemory("values_meta")
                            .HostMemory("axis_meta")
                            .HostMemory("input_mins_meta")
                            .HostMemory("input_maxes_meta")
                            .HostMemory("output_min")
                            .HostMemory("output_max")
                            .HostMemory("output_meta")
                            .HostMemory("output_min_meta")
                            .HostMemory("output_max_meta")
                            .TypeConstraint<qint8>("T"),
                        OneDnnConcatOp<GPUDevice, qint8, NAME_IS_AXIS>);

#else
#define REGISTER_CONCAT(T)                                             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("_OneDnnConcat").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OneDnnConcatOp<CPUDevice, T, NAME_IS_CONCAT_DIM>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_CONCAT);
#undef REGISTER_CONCAT

#define REGISTER_CONCATV2(T)                                             \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_OneDnnConcatV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OneDnnConcatOp<CPUDevice, T, NAME_IS_AXIS>);
TF_CALL_CPU_NUMBER_TYPES(REGISTER_CONCATV2);
#undef REGISTER_CONCATV2

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConcatV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T"),
                        OneDnnConcatOp<CPUDevice, quint8, NAME_IS_AXIS>);

REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedConcatV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T"),
                        OneDnnConcatOp<CPUDevice, qint8, NAME_IS_AXIS>);
#endif
}  // namespace itex
