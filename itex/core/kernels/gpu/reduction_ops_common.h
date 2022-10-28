/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_REDUCTION_OPS_COMMON_H_
#define ITEX_CORE_KERNELS_GPU_REDUCTION_OPS_COMMON_H_

#include "itex/core/kernels/common/transpose_functor.h"
#include "itex/core/kernels/gpu/reduction_ops.h"
#include "itex/core/kernels/gpu/reduction_utils.h"
#include "itex/core/utils/allocator.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
struct Constants {
  // Derive Index type. int (32-bit) or long (64-bit) depending on the
  // compile-time configuration. "float" here is not relevant.
  typedef TTypes<float>::Tensor::Index Index;
  Eigen::array<Index, 1> kZero;
  Eigen::array<Index, 1> kOne;
  Eigen::array<Index, 2> kZeroTwo;

  Constants() {
    kZero[0] = 0;
    kOne[0] = 1;
    kZeroTwo[0] = 0;
    kZeroTwo[1] = 2;
  }
};

#if defined(EIGEN_HAS_INDEX_LIST)
struct ConstantsBase {
  const Eigen::IndexList<Eigen::type2index<0>> kZero;
  const Eigen::IndexList<Eigen::type2index<1>> kOne;
  const Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<2>> kZeroTwo;
};
template <>
struct Constants<GPUDevice> : ConstantsBase {};
#endif  // EIGEN_HAS_INDEX_LIST

template <int NDIMS, bool ReduceFirstAxis>
struct ReduceAxies {
  typedef TTypes<float>::Tensor::Index Index;
  Eigen::array<Index, NDIMS> value;
  ReduceAxies() {
#pragma unroll
    for (int index = 0, i = !ReduceFirstAxis; index < NDIMS; ++index, i += 2)
      value[index] = i;
  }
};

template <typename Device, typename T, typename Tidx, typename Reducer>
class ReductionOp : public OpKernel {
 public:
  explicit ReductionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // TODO(itex): same as UnaryOp_Create()
    //  const DataType dt = DataTypeToEnum<T>::v();
    //  const DataType pt = DataTypeToEnum<Tidx>::v();
    //  OP_REQUIRES_OK(ctx, ctx->MatchSignature({dt, pt}, {dt}));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    const Tensor& axes = ctx->input(1);
    ITEX_VLOG(3) << "data shape: " << data.shape().DebugString();

    ReductionHelper helper;
    OP_REQUIRES_OK(ctx, helper.Simplify(data, axes, keep_dims_));
    ITEX_CHECK_GE(helper.ndims(), 0);

    bool is_scalar_identity = functor::ReducerTraits<Reducer>::IsScalarIdentity;
    bool is_trivial = helper.ndims() == 0 ||
                      (helper.ndims() == 1 && !helper.reduce_first_axis());
    if (is_scalar_identity && is_trivial) {
      // Special case. Reduces nothing.  It is unclear why this is
      // necessary, but tests fail without it.  Look into why this
      // case occurs.
      Tensor out;
      if (!out.CopyFrom(data, helper.out_shape())) {
        ctx->SetStatus(errors::Internal("Error during reduction copy."));
      }
      ctx->set_output(0, out);
      return;
    }

    // A temporary tensor whose size matches the size of the reduced
    // output.
    Tensor tmp_out;
    typedef functor::ReduceFunctor<Reducer> Functor;
    Constants<GPUDevice> constants;
    Reducer reducer;
    if (data.NumElements() > 0 && is_trivial && !is_scalar_identity) {
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
                                  TensorShape({data.NumElements()}), &tmp_out));
      Functor::Reduce(ctx, tmp_out.flat<T>(),
                      data.shaped<T, 2>({1, data.NumElements()}),
                      constants.kZero, reducer);
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
                                             helper.out_reshape(), &tmp_out));

      if (tmp_out.NumElements() == 0) {
        // Nothing to do, fall through to final reshaping.
      } else if (data.NumElements() == 0) {
        // Degenerate reduction where the input is empty but the output is
        // nonempty (thus tmp_out.NumElements() > 0), and we must fill the
        // output with identity elements.  Example: tf.reduce_sum(tf.zeros((0,
        // 3)), [0]). Eigen sometimes crashes in this case, so we do it
        // manually.
        Functor::FillIdentity(ctx->eigen_gpu_device(), tmp_out.flat<T>(),
                              reducer);
      } else if ((helper.ndims() == 1) && helper.reduce_first_axis()) {
        // Reduce to a scalar.
        Functor::Reduce(ctx, helper.out<T, 0>(&tmp_out), helper.in<T, 1>(data),
                        ReduceAxies<1, true>().value, reducer);
      } else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
        // Can be viewed as a reduction of a matrix along 1st dimension.
        Functor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                        ReduceAxies<1, true>().value, reducer);
      } else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
        // Can be viewed as a reduction of a matrix along 2nd dimension.
        Functor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                        ReduceAxies<1, false>().value, reducer);

      } else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
        // Can be viewed as a reduction of a 3D tensor along 2nd dimension.
        Functor::Reduce(ctx, helper.out<T, 2>(&tmp_out), helper.in<T, 3>(data),
                        constants.kOne, reducer);
      } else {
        // If we don't hit one of the cases above, transpose the data so that
        // all reduced dimensions are last and reuse the 2-D -> 1-D case.
        Tensor data_reshaped;
        OP_REQUIRES(ctx, data_reshaped.CopyFrom(data, helper.data_reshape()),
                    errors::Internal("Error during reduction copy."));
        Tensor shuffled;
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_temp(DataTypeToEnum<T>::value,
                                          helper.shuffled_shape(), &shuffled));
        OP_REQUIRES_OK(ctx, DoTranspose(ctx->eigen_gpu_device(), data_reshaped,
                                        helper.permutation(), &shuffled));
        const int64_t unreduced = tmp_out.NumElements();
        const int64_t reduced = shuffled.NumElements() / unreduced;
        const Tensor& const_shuffled = shuffled;
        Functor::Reduce(ctx, tmp_out.flat<T>(),
                        const_shuffled.shaped<T, 2>({unreduced, reduced}),
                        constants.kOne, reducer);
      }
    }

    // Set the real output using the contents of the reduction but the
    // real expected output shape.  The number of elements should
    // match between the two shapes.
    Tensor out;
    if (!out.CopyFrom(tmp_out, helper.out_shape())) {
      ctx->SetStatus(errors::Internal("Error during reduction copy."));
    }
    ctx->set_output(0, out);
  }

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_REDUCTION_OPS_COMMON_H_
