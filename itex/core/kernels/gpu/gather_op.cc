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

#include "itex/core/kernels/gpu/gather_functor.h"
#include "itex/core/kernels/gpu/gather_functor_batched.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class GatherOp : public OpKernel {
 public:
  explicit GatherOp(OpKernelConstruction* context) : OpKernel(context) {
    if (context->HasAttr("batch_dims")) {
      ITEX_CHECK_EQ(Status::OK(), context->GetAttr("batch_dims", &batch_dims_));
    } else {
      batch_dims_ = 0;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& params = context->input(0);
    const Tensor& indices = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // GatherV2 added an axis argument. For backwards compatibility with Gather,
    // fall back to axis 0 if the op does not have an axis input.
    int64_t axis = 0;
    bool axis_is_set = false;  // Indicates whether the axis argument was set.
    if (context->num_inputs() == 3) {
      axis_is_set = true;
      const Tensor& axis_tensor = context->input(2);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                  errors::InvalidArgument("axis must be scalar"));

      if (axis_tensor.dtype() == DT_INT32) {
        axis = axis_tensor.scalar<int32>()();
      } else if (axis_tensor.dtype() == DT_INT64) {
        axis = axis_tensor.scalar<int64_t>()();
      } else {
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("axis must be int32 or int64."));
      }
    }

    int64_t min_params_dim = axis < 0 ? -axis : axis + 1;
    OP_REQUIRES(
        context, params.dims() >= min_params_dim,
        errors::InvalidArgument("Shape must be at least rank ", min_params_dim,
                                " but is rank ", params.dims()));

    if (axis < 0) {
      axis = params.dims() + axis;
    }

    // Modify only a local copy of batch_dims_.
    int32_t batch_dims = batch_dims_;
    if (batch_dims != 0) {
      OP_REQUIRES(context,
                  batch_dims >= -indices.dims() && batch_dims <= indices.dims(),
                  errors::InvalidArgument("Expected batch_dims in the range [",
                                          -indices.dims(), ", ", indices.dims(),
                                          "], but got ", batch_dims));

      if (batch_dims < 0) {
        batch_dims = indices.dims() + batch_dims;
      }

      if (!axis_is_set) axis = batch_dims;

      OP_REQUIRES(context, batch_dims < params.dims(),
                  errors::InvalidArgument("batch_dims (", batch_dims,
                                          ") must be less than rank(params) (",
                                          params.dims(), ")."));

      OP_REQUIRES(context, axis >= batch_dims,
                  errors::InvalidArgument("batch_dims (", batch_dims,
                                          ") must be less than or equal to ",
                                          "axis (", axis, ")."));
      for (int i = 0; i < batch_dims; ++i) {
        OP_REQUIRES(context, params.dim_size(i) == indices.dim_size(i),
                    errors::InvalidArgument(
                        "params.shape[", i, "]: ", params.dim_size(i),
                        " should be equal to indices.shape[", i,
                        "]: ", indices.dim_size(i)));
      }
    }

    // Check that we have enough index space
    int64_t gather_dim_size = params.dim_size(axis);
    const int64_t N = indices.NumElements();
    OP_REQUIRES(
        context, gather_dim_size <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[", axis, "] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", gather_dim_size, " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is params.shape[:axis] + indices.shape[batch_dims:] +
    // params.shape[axis + 1:].
    TensorShape result_shape;
    int64_t batch_size = 1;
    int64_t outer_size = 1;
    int64_t inner_size = 1;

    for (int i = 0; i < batch_dims; ++i) {
      result_shape.AddDim(params.dim_size(i));
      batch_size *= params.dim_size(i);
    }
    for (int i = batch_dims; i < axis; ++i) {
      result_shape.AddDim(params.dim_size(i));
      outer_size *= params.dim_size(i);
    }
    for (int i = batch_dims; i < indices.dims(); ++i) {
      result_shape.AddDim(indices.dim_size(i));
    }
    for (int i = axis + 1; i < params.dims(); ++i) {
      result_shape.AddDim(params.dim_size(i));
      inner_size *= params.dim_size(i);
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, result_shape, &out));
    if (N == 0) return;
    if (inner_size == 0) return;

    const int64_t params_size = params.NumElements();
    if (params_size == 0) {
      // When params_size is 0, the data pointer of params tensor maybe a host
      // pointer. If we use a host pointer in sycl kernel even if the code is
      // in impossible condition branch, we will get an error -50
      // (CL_INVALID_ARG_VALUE). Here we workaround this case. All indices will
      // be out of range in this condition, so the output value will be zero
      // according to definition of GatherV2.

      context->eigen_device<Device>().stream()->memset(
          out->flat<T>().data(), 0, out->NumElements() * sizeof(T));
      return;
    }

    int64_t bad_i = -1;
    auto indices_flat = indices.flat<Index>();
    if (batch_dims > 0) {
      auto params_flat = params.shaped<T, 4>(
          {batch_size, outer_size, gather_dim_size, inner_size});
      auto out_flat = out->shaped<T, 4>(
          {batch_size, outer_size, N / batch_size, inner_size});

      functor::GatherFunctorBatched<Device, T, Index> functor;
      bad_i = functor(context, params_flat, indices_flat, out_flat);
    } else {
      auto params_flat =
          params.shaped<T, 3>({outer_size, gather_dim_size, inner_size});
      auto out_flat = out->shaped<T, 3>({outer_size, N, inner_size});

      functor::GatherFunctor<Device, T, Index> functor;
      bad_i = functor(context, params_flat, indices_flat, out_flat);
    }
    OP_REQUIRES(
        context, bad_i < 0,
        errors::InvalidArgument(
            "indices", SliceDebugString(indices.shape(), bad_i), " = ",
            indices_flat(bad_i), " is not in [0, ", gather_dim_size, ")"));
  }

 private:
  int32 batch_dims_;
};

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("Gather")                               \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherOp<dev##Device, type, index_type>);    \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                             \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices")  \
                              .HostMemory("axis"),                     \
                          GatherOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64_t)

#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_int32(REGISTER_GATHER_GPU);
TF_CALL_int64(REGISTER_GATHER_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_GATHER_GPU);
TF_CALL_complex64(REGISTER_GATHER_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GATHER_GPU);
TF_CALL_complex128(REGISTER_GATHER_GPU)
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_GATHER_GPU

#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

};  // namespace itex
