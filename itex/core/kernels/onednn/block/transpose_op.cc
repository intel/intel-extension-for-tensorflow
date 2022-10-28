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

#include <memory>
#include <vector>

#include "itex/core/kernels/common/no_ops.h"
#include "itex/core/kernels/common/transpose_functor.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"

namespace itex {

using dnnl::memory;

memory::dims OneDnnReorderStrides(const memory::dims& strides,
                                  const OneDnnShape& src_onednn_shape) {
  memory::dims reordered_strides(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    reordered_strides[src_onednn_shape.TfDimIdx(i)] = strides[i];
  }
  return reordered_strides;
}

template <typename Device, typename T, bool is_conjugate = false>
Status DoTranspose(OpKernelContext* context, const Tensor& src_tensor,
                   gtl::ArraySlice<int32> perm,
                   const OneDnnShape& src_onednn_shape, Tensor* dst_tensor) {
  auto tensor_dims = dst_tensor->dims();
  if (tensor_dims < 2) return Status::OK();

  if (src_onednn_shape.IsOneDnnTensor() || tensor_dims <= MAX_NDIMS) {
    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      memory::desc src_md;
      memory::dims src_dims;
      if (src_onednn_shape.IsOneDnnTensor()) {
        src_dims = src_onednn_shape.GetSizesAsOneDnnDims();
        src_md = src_onednn_shape.GetOneDnnLayout();
      } else {
        src_dims = TFShapeToOneDnnDims(src_tensor.shape());
        memory::dims src_strides = CalculateTFStrides(src_dims);
        src_md = memory::desc(src_dims, OneDnnType<T>(), src_strides);
      }
      auto src_mem = CreateDnnlMemory(src_md, onednn_engine,
                                      GetTensorBuffer<T>(&src_tensor));

      // Sequential strides for dst tensor
      memory::dims dst_dims = TFShapeToOneDnnDims(dst_tensor->shape());
      memory::dims dst_strides = CalculateTFStrides(dst_dims);

      // Strides change due to permutation semantic for tranpose op
      dst_strides = internal::ReorderStrides(dst_strides, perm);

      // Strides change due to NHWC/NCHW semantic for OneDnn layout
      if (src_onednn_shape.IsOneDnnTensor()) {
        dst_strides = OneDnnReorderStrides(dst_strides, src_onednn_shape);
      }

      memory::desc dst_md =
          memory::desc(src_dims, OneDnnType<T>(), dst_strides);
      auto dst_mem = CreateDnnlMemory(dst_md, onednn_engine,
                                      GetTensorBuffer<T>(dst_tensor));

      ReorderMemory(*context, &src_mem, &dst_mem, onednn_engine);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + std::string(e.message) + ", in file " +
                         std::string(__FILE__) + ":" + std::to_string(__LINE__);
      return errors::Aborted("Operation received an exception:", error_msg);
    }
  } else {
    // Plain layout input with dim > MAX_NDIMS will fall back to eigen
    // implementation
    const Device& d = context->eigen_device<Device>();

    switch (tensor_dims) {
      case 2:
        internal::TransposeUsingEigen<Device, T, 2>(d, src_tensor, perm,
                                                    is_conjugate, dst_tensor);
        break;
      case 3:
        internal::TransposeUsingEigen<Device, T, 3>(d, src_tensor, perm,
                                                    is_conjugate, dst_tensor);
        break;
      case 4:
        internal::TransposeUsingEigen<Device, T, 4>(d, src_tensor, perm,
                                                    is_conjugate, dst_tensor);
        break;
      case 5:
        internal::TransposeUsingEigen<Device, T, 5>(d, src_tensor, perm,
                                                    is_conjugate, dst_tensor);
        break;
      case 6:
        internal::TransposeUsingEigen<Device, T, 6>(d, src_tensor, perm,
                                                    is_conjugate, dst_tensor);
        break;
      case 7:
        internal::TransposeUsingEigen<Device, T, 7>(d, src_tensor, perm,
                                                    is_conjugate, dst_tensor);
        break;
      case 8:
        internal::TransposeUsingEigen<Device, T, 8>(d, src_tensor, perm,
                                                    is_conjugate, dst_tensor);
        break;
      default:
        ITEX_CHECK(false) << "Max supported dim number is 8, got "
                          << tensor_dims;
        break;
    }
  }

  return Status::OK();
}

template <typename Device, typename T, bool is_conjugate = false>
class OneDnnTransposeOp : public OpKernel {
 public:
  explicit OneDnnTransposeOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const int kSrcIndex = 0;      // index of src input tensor
    const int kPermuteIndex = 1;  // index of permute input tensor
    const int kDstIndex = 0;      // index of dst tensor
    const Tensor& src_tensor = context->input(kSrcIndex);
    const Tensor& perm = context->input(kPermuteIndex);

    OneDnnShape src_onednn_shape;
    GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
    TensorShape src_tf_shape = src_onednn_shape.IsOneDnnTensor()
                                   ? src_onednn_shape.GetTfShape()
                                   : src_tensor.shape();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsVector(perm.shape()),
                errors::InvalidArgument("perm must be a vector, not ",
                                        perm.shape().DebugString()));

    // Although Tperm may be an int64 type, an int32 is sufficient to hold
    // dimension range values, so the narrowing here should be safe.
    std::vector<int32> permutation;
    const int dims = src_tf_shape.dims();
    if (perm.dtype() == DT_INT32) {
      OP_REQUIRES_OK(context, internal::PermutationHelper<int32>(perm, dims,
                                                                 &permutation));
    } else {
      OP_REQUIRES_OK(context, internal::PermutationHelper<int64>(perm, dims,
                                                                 &permutation));
    }

    TensorShape dst_shape;
    // Check whether permutation is a permutation of integers of [0 .. dims).
    gtl::InlinedVector<bool, 8> bits(dims);
    bool is_identity = true;
    for (int i = 0; i < dims; ++i) {
      const int32 d = permutation[i];
      OP_REQUIRES(
          context, 0 <= d && d < dims,
          errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
      bits[d] = true;
      const auto dim_size = src_tf_shape.dim_size(d);
      dst_shape.AddDim(dim_size);
      if (d != i) {
        is_identity = false;
      }
    }
    for (int i = 0; i < dims; ++i) {
      OP_REQUIRES(
          context, bits[i],
          errors::InvalidArgument(i, " is missing from {",
                                  str_util::Join(permutation, ","), "}."));
    }

    // 0-D, 1-D, and identity transposes do nothing.
    if (!is_conjugate && !src_onednn_shape.IsOneDnnTensor() &&
        (dims <= 1 || is_identity)) {
      context->set_output(kDstIndex, src_tensor);
      return;
    } else if (!is_conjugate && !src_onednn_shape.IsOneDnnTensor() &&
               internal::NonSingletonDimensionsAlign(src_tensor.shape(),
                                                     permutation)) {
      // TODO(itex): Here is a quick path for plain layout tensor.
      // To avoid transpose, and just do the reshape, when permutation and dims
      // meet certain conditions. Investigate could block layout also benefits
      // from the quick path.
      Tensor output;
      ITEX_CHECK(output.CopyFrom(src_tensor, dst_shape));
      context->set_output(kDstIndex, output);
      return;
    }

    // Allocate output, since _OneDnnTranspose's output is always plain format.
    // We use normal allocate_output function
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(kDstIndex, dst_shape, &output));

    // Do the transpose computation
    if (dst_shape.num_elements() > 0) {
      OP_REQUIRES_OK(context, DoTranspose<Device, T, is_conjugate>(
                                  context, src_tensor, permutation,
                                  src_onednn_shape, output));
    }
  }
};

// INT8 Transpose = FP32 Transpose + Pass min/max tensor
template <typename Device, typename T>
class OneDnnQuantizedTransposeOp : public OneDnnTransposeOp<Device, T> {
 public:
  explicit OneDnnQuantizedTransposeOp(OpKernelConstruction* context)
      : OneDnnTransposeOp<Device, T>(context) {}

  void Compute(OpKernelContext* context) override {
    // This call processes inputs 1 and 2 to write output 0.
    OneDnnTransposeOp<Device, T>::Compute(context);
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
                    "min_x must be a scalar or a vector of 1 element"));

    const auto& input_max_float_tensor = context->input(kSrcMaxRangeIndex);
    const auto& input_max_float_shape = input_max_float_tensor.shape();
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(input_max_float_shape) ||
                    (TensorShapeUtils::IsVector(input_max_float_shape) &&
                     (input_max_float_shape.dim_size(0) == 1)),
                errors::InvalidArgument(
                    "max_x must be a scalar or a vector of 1 element"));

    context->set_output(kDstMinRangeIndex, input_min_float_tensor);
    context->set_output(kDstMaxRangeIndex, input_max_float_tensor);
  }
};

// FP32 kernel registration
#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                            \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnTranspose")       \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("perm")        \
                              .HostMemory("x_meta")      \
                              .HostMemory("perm_meta"),  \
                          OneDnnTransposeOp<GPUDevice, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#else
#define REGISTER_KERNEL(TYPE)                                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_OneDnnTranspose").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnTransposeOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

// INT8 kernel registration
#ifndef INTEL_CPU_ONLY
// OneDnn gpu kernel registration
#define REGISTER_KERNEL(TYPE)                                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_OneDnnQuantizedTranspose")                                       \
          .Device(DEVICE_GPU)                                                 \
          .HostMemoryList3("perm", "min_x", "max_x")                          \
          .HostMemoryList4("x_meta", "perm_meta", "min_x_meta", "max_x_meta") \
          .HostMemoryList2("min_y", "max_y")                                  \
          .TypeConstraint<TYPE>("T"),                                         \
      OneDnnQuantizedTransposeOp<GPUDevice, TYPE>);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#else
// Dummy cpu kernel registration
#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("_QuantizedTranspose")     \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          NoImplementOp);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
// OneDnn cpu kernel registration
#define REGISTER_KERNEL(TYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnQuantizedTranspose") \
                              .Device(DEVICE_CPU)           \
                              .TypeConstraint<TYPE>("T"),   \
                          OneDnnQuantizedTransposeOp<CPUDevice, TYPE>);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // INTEL_CPU_ONLY

}  // namespace itex
