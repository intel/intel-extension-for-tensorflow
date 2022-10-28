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
#include "itex/core/utils/quantization_util.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// TODO(itex): For intel-tf proper, the Quantize and Dequantize op only have
// narrow_range implementation, regardless the `narrow_range` flag is True or
// False. Currently, ITEX also follow such logic.

using dnnl::memory;
using dnnl::primitive;
using dnnl::primitive_attr;
using dnnl::reorder;
namespace itex {
template <typename Device, typename T>
class OneDnnDequantizeReshapeOp : public OpKernel {
 public:
  explicit OneDnnDequantizeReshapeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_string));
    OP_REQUIRES(context,
                (mode_string == "MIN_COMBINED" || mode_string == "MIN_FIRST" ||
                 mode_string == "SCALED"),
                errors::InvalidArgument("Mode string must be 'MIN_COMBINED',"
                                        " 'MIN_FIRST', or 'SCALED', is '" +
                                        mode_string + "'"));
    if (mode_string == "MIN_COMBINED") {
      mode_ = QuantizeMode::MIN_COMBINED;
    } else if (mode_string == "MIN_FIRST") {
      mode_ = QuantizeMode::MIN_FIRST;
    } else if (mode_string == "SCALED") {
      mode_ = QuantizeMode::SCALED;
    }

    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* context) override {
    const int kSrcDataIndex = 0;
    const int kSrcMinRangeIndex = 1;
    const int kSrcMaxRangeIndex = 2;
    const int kShapeIndex = 3;
    const int kDstDataIndex = 0;

    const Tensor& src_tensor = context->input(kSrcDataIndex);
    const Tensor& input_min_range = context->input(kSrcMinRangeIndex);
    const Tensor& input_max_range = context->input(kSrcMaxRangeIndex);
    const Tensor& sizes = context->input(kShapeIndex);

    int num_slices = 1;
    if (axis_ > -1) {
      num_slices = input_min_range.NumElements();
    }

    std::vector<float> min_range(num_slices);
    std::vector<float> max_range(num_slices);

    if (num_slices == 1) {
      min_range[0] = input_min_range.template flat<float>()(0);
      max_range[0] = input_max_range.template flat<float>()(0);
    } else {
      auto min_ranges = input_min_range.template vec<float>();
      auto max_ranges = input_max_range.template vec<float>();
      for (int i = 0; i < num_slices; ++i) {
        min_range[i] = min_ranges(i);
        max_range[i] = max_ranges(i);
      }
    }

    // Calculating scales and zeropoints for quantization.
    std::vector<float> scale_factor(num_slices, 0);
    std::vector<int32> zero_points(num_slices, 0);

    GetScaleAndZeropointAndAlignMinMax<T>(
        min_range.data(), max_range.data(), mode_, QuantDequantFlag::Dequantize,
        num_slices, scale_factor.data(), zero_points.data());

    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      // Get src_md
      OneDnnShape src_onednn_shape;
      GetOneDnnShape(context, kSrcDataIndex, &src_onednn_shape);
      TensorShape input_shape = src_onednn_shape.IsOneDnnTensor()
                                    ? src_onednn_shape.GetTfShape()
                                    : src_tensor.shape();
      memory::dims src_dims = src_onednn_shape.IsOneDnnTensor()
                                  ? src_onednn_shape.GetSizesAsOneDnnDims()
                                  : TFShapeToOneDnnDims(src_tensor.shape());

      const int64 nelems = src_onednn_shape.IsOneDnnTensor()
                               ? input_shape.num_elements()
                               : src_tensor.NumElements();
      memory::desc src_md;
      // Get dst_md.
      memory::dims dst_dims = src_dims;
      memory::desc dst_md;

      if (src_onednn_shape.IsOneDnnTensor()) {
        src_md = src_onednn_shape.GetOneDnnLayout();
        dst_md = src_onednn_shape.GetTfLayout();
      } else {
        src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
        dst_md = src_md;
      }
      dst_md.data.data_type = memory::convert_to_c(OneDnnType<float>());
      // Set the scale factor for quantize
      primitive_attr post_ops_attr;
      if (num_slices == 1) {
        post_ops_attr.set_output_scales(0, scale_factor);
        if (mode_ == QuantizeMode::MIN_FIRST) {
          post_ops_attr.set_zero_points(DNNL_ARG_SRC, 0, zero_points);
        }
      } else {
        int mask = static_cast<int>(std::pow(2, axis_));
        post_ops_attr.set_output_scales(mask, scale_factor);
        if (mode_ == QuantizeMode::MIN_FIRST) {
          post_ops_attr.set_zero_points(DNNL_ARG_SRC, mask, zero_points);
        }
      }

      // Create Reorder primitive
      auto fwd_pd = reorder::primitive_desc(
          onednn_engine, src_md, onednn_engine, dst_md, post_ops_attr);
      auto fwd_primitive = reorder(fwd_pd);

      // Compute the output shape.  Determine product of specified
      // dimensions, and find the index of the unspecified one.
      TensorShape shape;
      int64 product = 1;
      int unknown_index = -1;
      bool sizes_has_zero_dim = false;
      switch (sizes.dtype()) {
        case DT_INT32:
          OP_REQUIRES_OK(context,
                         ValidateSizes<int32>(sizes, &product, &unknown_index,
                                              &shape, &sizes_has_zero_dim));
          break;
        case DT_INT64:
          OP_REQUIRES_OK(context,
                         ValidateSizes<int64>(sizes, &product, &unknown_index,
                                              &shape, &sizes_has_zero_dim));
          break;
        default:
          context->CtxFailure(errors::InvalidArgument(
              "desired shape must be a DT_INT32 or DT_INT64 vector, not a ",
              DataTypeString(sizes.dtype())));
          return;
      }
      if (unknown_index != -1) {
        int64 input_num_elements = 1;
        bool input_has_zero_dim = false;
        for (int dim = 0; dim < input_shape.dims(); ++dim) {
          // For zero dimension, we don't count it into `input_num_elements`
          // unless `sizes` has no zero dimension, so we are still able to
          // infer shapes for other dimensions.
          if (input_shape.dim_size(dim) > 0 || !sizes_has_zero_dim) {
            input_num_elements *= input_shape.dim_size(dim);
          } else {
            input_has_zero_dim = true;
          }
        }

        const int64 missing = input_num_elements / product;
        if (!input_has_zero_dim) {
          OP_REQUIRES(
              context, product * missing == input_num_elements,
              errors::InvalidArgument(
                  "Input to reshape is a tensor with ", input_num_elements,
                  " values, but the requested shape requires a multiple of ",
                  product));
        }
        shape.set_dim(unknown_index, missing);
      }
      OP_REQUIRES(
          context, shape.num_elements() == nelems,
          errors::InvalidArgument("Input to reshape is a tensor with ", nelems,
                                  " values, but the requested shape has ",
                                  shape.num_elements()));
      // Reorder is needed when OneDnn layout and TF layout are different
      Tensor* dst_tensor;
      // Allocate new buffer for output tensor
      OP_REQUIRES_OK(
          context, context->allocate_output(kDstDataIndex, shape, &dst_tensor));

      // Create src and dst memory
      auto src_mem = CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                                      GetTensorBuffer<T>(&src_tensor));
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      GetTensorBuffer<float>(dst_tensor));

      // Execute Reorder primitive
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> fwd_primitive_args = {
          {DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}};
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

 private:
  template <typename Tshape>
  Status ValidateSizes(const Tensor& sizes, int64* product, int* unknown_index,
                       TensorShape* shape, bool* has_zero_dim) {
    *product = 1;
    *unknown_index = -1;
    *has_zero_dim = false;
    const int64 num_dims = sizes.NumElements();
    auto Svec = sizes.flat<Tshape>();
    for (int d = 0; d < num_dims; ++d) {
      const Tshape size = Svec(d);
      if (size == -1) {
        if (*unknown_index != -1) {
          return errors::InvalidArgument(
              "Only one input size may be -1, not both ", *unknown_index,
              " and ", d);
        }
        *unknown_index = d;
        shape->AddDim(1);
      } else if (size < 0) {
        return errors::InvalidArgument("Size ", d,
                                       " must be non-negative, not ", size);
      } else if (size == 0) {
        // We don't include zero-sized dimension in product, so that we can
        // still calculate number of elements for non-zero-sized dimensions and
        // therefore infer their shapes.
        shape->AddDim(size);
        *has_zero_dim = true;
      } else {
        shape->AddDim(size);
        (*product) *= size;
      }
    }
    return Status::OK();
  }

 private:
  QuantizeMode mode_;
  int axis_;
  bool narrow_range_;
};
#ifndef INTEL_CPU_ONLY

#define REGISTER_KERNEL(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnFusedDequantizeWithReshape") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("T")            \
                              .HostMemory("min_range")              \
                              .HostMemory("max_range")              \
                              .HostMemory("shape")                  \
                              .HostMemory("input_meta")             \
                              .HostMemory("min_range_meta")         \
                              .HostMemory("max_range_meta")         \
                              .HostMemory("shape_meta")             \
                              .HostMemory("output_meta"),           \
                          OneDnnDequantizeReshapeOp<GPUDevice, TYPE>);

TF_CALL_qint8(REGISTER_KERNEL);
TF_CALL_quint8(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#endif  // INTEL_CPU_ONLY
}  // namespace itex
