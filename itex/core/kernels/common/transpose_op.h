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

#ifndef ITEX_CORE_KERNELS_COMMON_TRANSPOSE_OP_H_
#define ITEX_CORE_KERNELS_COMMON_TRANSPOSE_OP_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/kernels/common/transpose_functor.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {

// Transpose of N-dimensional tensor using oneDNN
template <typename Device, typename T>
Status TransposeND(OpKernelContext* context, const Tensor& in, Tensor* out,
                   const gtl::ArraySlice<int32>& perm) {
  try {
    auto onednn_engine = CreateDnnlEngine<Device>(*context);
    auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

    dnnl::memory::dims in_dims = TFShapeToOneDnnDims(in.shape());
    dnnl::memory::dims out_dims = TFShapeToOneDnnDims(out->shape());
    dnnl::memory::dims in_strides = CalculateTFStrides(in_dims);
    // Reorder output strides based on permutation requested.
    dnnl::memory::dims out_strides =
        internal::ReorderStrides(CalculateTFStrides(out_dims), perm);

    dnnl::memory::desc in_md =
        dnnl::memory::desc(in_dims, OneDnnType<T>(), in_strides);
    auto in_mem = CreateDnnlMemory(
        in_md, onednn_engine,
        const_cast<void*>(static_cast<const void*>(in.flat<T>().data())));

    // Output dimensions are same as input dimensions. We adjust the layout
    // using strides.
    dnnl::memory::desc out_md =
        dnnl::memory::desc(in_dims, OneDnnType<T>(), out_strides);
    auto out_mem = CreateDnnlMemory(
        out_md, onednn_engine,
        const_cast<void*>(static_cast<const void*>(out->flat<T>().data())));

    auto transpose_reorder_primitive = dnnl::reorder(in_mem, out_mem);
    std::unordered_map<int, dnnl::memory> transpose_reorder_args = {
        {DNNL_ARG_SRC, in_mem}, {DNNL_ARG_DST, out_mem}};
    transpose_reorder_primitive.execute(onednn_stream, transpose_reorder_args);
    return Status::OK();
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + std::string(e.message) + ", in file " +
                       std::string(__FILE__) + ":" + std::to_string(__LINE__);
    return errors::Aborted("Operation received an exception:", error_msg);
  }
}

// output = TransposeOp(T<any> input, T<int32> perm) takes a tensor
// of type T and rank N, and a permutation of 0, 1, ..., N-1. It
// shuffles the dimensions of the input tensor according to permutation.
//
// Specifically, the returned tensor output meets the following condition:
// 1) output.dims() == input.dims();
// 2) output.dim_size(i) == input.dim_size(perm[i]);
// 3) output.tensor<T, N>(i_0, i_1, ..., i_N-1) ==
//      input.tensor<T, N>(j_0, j_1, ..., j_N-1),
//    where i_s == j_{perm[s]}
//
// REQUIRES: perm is a vector of int32.
// REQUIRES: input.dims() == perm.size().
// REQUIRES: perm is a permutation.
template <typename Device, typename T, bool is_conjugate = false>
class TransposeOp : public OpKernel {
 public:
  explicit TransposeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& perm = ctx->input(1);
    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm.shape()),
                errors::InvalidArgument("perm must be a vector, not ",
                                        perm.shape().DebugString()));

    // Although Tperm may be an int64 type, an int32 is sufficient to hold
    // dimension range values, so the narrowing here should be safe.
    std::vector<int32> permutation;
    const int dims = input.dims();
    if (perm.dtype() == DT_INT32) {
      OP_REQUIRES_OK(
          ctx, internal::PermutationHelper<int32>(perm, dims, &permutation));
    } else {
      OP_REQUIRES_OK(
          ctx, internal::PermutationHelper<int64>(perm, dims, &permutation));
    }
    TensorShape shape;

    // Check whether permutation is a permutation of integers of [0 .. dims).
    gtl::InlinedVector<bool, 8> bits(dims);
    bool is_identity = true;
    for (int i = 0; i < dims; ++i) {
      const int32 d = permutation[i];
      OP_REQUIRES(
          ctx, 0 <= d && d < dims,
          errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
      bits[d] = true;
      const auto dim_size = input.dim_size(d);
      shape.AddDim(dim_size);
      if (d != i) {
        is_identity = false;
      }
    }
    for (int i = 0; i < dims; ++i) {
      OP_REQUIRES(
          ctx, bits[i],
          errors::InvalidArgument(i, " is missing from {",
                                  str_util::Join(permutation, ","), "}."));
    }

    // 0-D, 1-D, and identity transposes do nothing.
    if (!is_conjugate && (dims <= 1 || is_identity)) {
      ctx->set_output(0, input);
      return;
    } else if (!is_conjugate && internal::NonSingletonDimensionsAlign(
                                    input.shape(), permutation)) {
      Tensor output;
      OP_REQUIRES(ctx, output.CopyFrom(input, shape),
                  errors::Unknown("Error reshaping Tensor."));
      ctx->set_output(0, output);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    if (shape.num_elements() > 0) {
      OP_REQUIRES_OK(ctx, DoTranspose(ctx, input, permutation, output));
    }
  }

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) {
    // oneDNN has different define for MAX_NDIMS and MKLDNN_MAX_NDIMS(12)
    // all gpu primitive is using MAX_NDIMS, align with it first
    // Need check with oneDNN team
    if (!is_conjugate) {
      if (in.dims() <= MAX_NDIMS) {
        switch (in.dtype()) {
          case DT_FLOAT:
            return TransposeND<Device, float>(ctx, in, out, perm);
            break;
          case DT_HALF:
            return TransposeND<Device, Eigen::half>(ctx, in, out, perm);
            break;
          case DT_BFLOAT16:
            return TransposeND<Device, Eigen::bfloat16>(ctx, in, out, perm);
            break;
          case DT_QUINT8:
            return TransposeND<Device, quint8>(ctx, in, out, perm);
            break;
          case DT_QINT8:
            return TransposeND<Device, qint8>(ctx, in, out, perm);
            break;
          case DT_INT8:
            return TransposeND<Device, int8>(ctx, in, out, perm);
            break;
          default:
            break;
        }
      }
      return ::itex::DoTranspose(ctx->eigen_device<Device>(), in, perm, out);
    } else {
      return ::itex::DoConjugateTranspose(ctx->eigen_device<Device>(), in, perm,
                                          out);
    }
  }
};

// INT8 Transpose = FP32 Transpose + Pass min/max tensor
template <typename Device, typename T>
class QuantizedTransposeOp : public TransposeOp<Device, T> {
 public:
  explicit QuantizedTransposeOp(OpKernelConstruction* context)
      : TransposeOp<Device, T>(context) {}

  void Compute(OpKernelContext* context) override {
    TransposeOp<Device, T>::Compute(context);
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

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_COMMON_TRANSPOSE_OP_H_
