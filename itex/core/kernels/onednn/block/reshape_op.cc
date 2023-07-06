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

using dnnl::memory;

namespace itex {
template <typename Device, typename T>
class OneDnnReshapeOp : public OpKernel {
 public:
  explicit OneDnnReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& src_tensor = context->input(kSrcIndex);
    const Tensor& sizes = context->input(kShapeIndex);

    OneDnnShape src_onednn_shape;
    GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
    TensorShape input_shape = src_onednn_shape.IsOneDnnTensor()
                                  ? src_onednn_shape.GetTfShape()
                                  : src_tensor.shape();
    const int64 nelems = src_onednn_shape.IsOneDnnTensor()
                             ? input_shape.num_elements()
                             : src_tensor.NumElements();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes.shape().DebugString()));

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

    if (src_onednn_shape.IsOneDnnTensor()) {
      try {
        // Reshape is just a logical view change operation for a tensor.
        // It does not change underlying layout.
        // 1. For plain layout (src_onednn_md == dst_tf_md), we don't need to
        // actually change the Tensor physical layout. Just copy the input
        // tensor with new shape.
        // 2. For block layout (src_onednn_md != dst_tf_md), which is different
        // layout compared to normal Tensorflow tensor. We need to reorder
        // tensor and then put it in the shape expected by Tensorflow.

        // Get OneDnn layout of input tensor.
        auto src_onednn_md = src_onednn_shape.GetOneDnnLayout();
        // Get expected Tensorflow layout of input tensor.
        auto dst_tf_md = src_onednn_shape.GetTfLayout();

        if (src_onednn_md == dst_tf_md) {
          // if onednn layout and tf layout are the same, just forward tensor
          // with new shape
          ITEX_VLOG(3)
              << "Reshape: Input tensor is plain layout, but "
                 "IsOneDnnTensor() = True. The implementation of the op "
                 "before _OneDnnTotf may be improved";
          Tensor dst_tensor;
          ITEX_CHECK(dst_tensor.CopyFrom(src_tensor, shape));
          context->set_output(kDstIndex, dst_tensor);
          return;
        }

        // Reorder is needed when OneDnn layout and TF layout are different
        Tensor* dst_tensor;
        // Allocate new buffer for output tensor
        OP_REQUIRES_OK(context,
                       context->allocate_output(kDstIndex, shape, &dst_tensor));

        // If dst tensor is empty, we do not need to execute reorder primitive.
        if (shape.num_elements() == 0) return;

        auto onednn_engine = CreateDnnlEngine<Device>(*context);
        auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

        auto src_mem = CreateDnnlMemory(src_onednn_md, onednn_engine,
                                        GetTensorBuffer<T>(&src_tensor));
        auto dst_mem = CreateDnnlMemory(dst_tf_md, onednn_engine,
                                        GetTensorBuffer<T>(dst_tensor));
        ReorderMemory(*context, &src_mem, &dst_mem, onednn_engine);
      } catch (dnnl::error& e) {
        string error_msg = "Status: " + std::to_string(e.status) +
                           ", message: " + string(e.message) + ", in file " +
                           string(__FILE__) + ":" + std::to_string(__LINE__);
        OP_REQUIRES_OK(
            context,
            errors::Aborted("Operation received an exception:", error_msg));
      }
    } else {
      // If input tensor is not in OneDnn layout, then just copy input tensor
      // to output with specified shape.
      Tensor dst_tensor;
      ITEX_CHECK(dst_tensor.CopyFrom(src_tensor, shape));
      context->set_output(kDstIndex, dst_tensor);
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

  const int kSrcIndex = 0;
  const int kShapeIndex = 1;
  const int kDstIndex = 0;
};

// INT8 Reshape = FP32 Reshape + Pass min/max tensor
template <typename Device, typename T>
class OneDnnQuantizedReshapeOp : public OneDnnReshapeOp<Device, T> {
 public:
  explicit OneDnnQuantizedReshapeOp(OpKernelConstruction* context)
      : OneDnnReshapeOp<Device, T>(context) {}

  void Compute(OpKernelContext* context) override {
    // This call processes inputs 1 and 2 to write output 0.
    OneDnnReshapeOp<Device, T>::Compute(context);
    if (!context->status().ok()) {
      return;
    }

    const int kSrcMinRangeIndex = 2;
    const int kSrcMaxRangeIndex = 3;
    const int kDstMinRangeIndex = 1;
    const int kDstMaxRangeIndex = 2;

    const auto& input_min_float_tensor = context->input(kSrcMinRangeIndex);
    const auto& input_min_float_shape = input_min_float_tensor.shape();
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(input_min_float_shape) ||
                    (TensorShapeUtils::IsVector(input_min_float_shape) &&
                     (input_min_float_shape.dim_size(0) == 1)),
                errors::InvalidArgument(
                    "input_min must be a scalar or a vector of 1 element"));

    const auto& input_max_float_tensor = context->input(kSrcMaxRangeIndex);
    const auto& input_max_float_shape = input_max_float_tensor.shape();
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(input_max_float_shape) ||
                    (TensorShapeUtils::IsVector(input_max_float_shape) &&
                     (input_max_float_shape.dim_size(0) == 1)),
                errors::InvalidArgument(
                    "input_max must be a scalar or a vector of 1 element"));

    context->set_output(kDstMinRangeIndex, input_min_float_tensor);
    context->set_output(kDstMaxRangeIndex, input_max_float_tensor);
  }
};

// FP32 kernel registration
#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                            \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnReshape")         \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("shape")       \
                              .HostMemory("tensor_meta") \
                              .HostMemory("shape_meta"), \
                          OneDnnReshapeOp<GPUDevice, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#else
#define REGISTER_KERNEL(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_OneDnnReshape").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnReshapeOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

// INT8 kernel registration
#ifndef INTEL_CPU_ONLY
// OneDnn gpu kernel registration
#define REGISTER_KERNEL(TYPE)                                             \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_OneDnnQuantizedReshape")                                     \
          .Device(DEVICE_GPU)                                             \
          .HostMemoryList3("shape", "input_min", "input_max")             \
          .HostMemoryList4("tensor_meta", "shape_meta", "input_min_meta", \
                           "input_max_meta")                              \
          .HostMemoryList2("output_min", "output_max")                    \
          .TypeConstraint<TYPE>("T"),                                     \
      OneDnnQuantizedReshapeOp<GPUDevice, TYPE>);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#else
// OneDnn cpu kernel registration
#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedReshape") \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          OneDnnQuantizedReshapeOp<CPUDevice, TYPE>);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY
}  // namespace itex
