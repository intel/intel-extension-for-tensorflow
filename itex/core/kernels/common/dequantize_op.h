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

#ifndef ITEX_CORE_KERNELS_COMMON_DEQUANTIZE_OP_H_
#define ITEX_CORE_KERNELS_COMMON_DEQUANTIZE_OP_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/kernels/common/host_data_cache.h"
#include "itex/core/utils/errors.h"
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
template <typename Device, typename T, typename dtype>
class DequantizeOp : public OpKernel {
 public:
  explicit DequantizeOp(OpKernelConstruction* context) : OpKernel(context) {
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
    const int kDstDataIndex = 0;

    const Tensor& src_tensor = context->input(kSrcDataIndex);
    const Tensor& input_min_range = context->input(kSrcMinRangeIndex);
    const Tensor& input_max_range = context->input(kSrcMaxRangeIndex);

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
      TensorShape src_tf_shape = src_tensor.shape();
      memory::dims src_dims = TFShapeToOneDnnDims(src_tensor.shape());
      memory::desc src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);

      // Get dst_md.
      memory::dims dst_dims = src_dims;
      memory::desc dst_md = CreatePlainMemDescWithFormatTag<dtype>(dst_dims);

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

      TensorShape dst_tf_shape;
      dst_tf_shape = OneDnnDimsToTFShape(dst_dims);

      // Allocate output's data tensor and meta tensor
      Tensor* dst_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  kDstDataIndex, dst_tf_shape, &dst_tensor));

      // Create src and dst memory
      auto src_mem = CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                                      GetTensorBuffer<T>(&src_tensor));
      auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                      GetTensorBuffer<dtype>(dst_tensor));

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
  QuantizeMode mode_;
  int axis_;
  bool narrow_range_;
#ifdef ITEX_ONEDNN_3_0
  HostDataCache<Device, float> output_scale_cache_;
  HostDataCache<Device, int32> zero_point_cache_;
#endif
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_DEQUANTIZE_OP_H_
