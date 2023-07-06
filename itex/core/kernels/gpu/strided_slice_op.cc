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

#include "itex/core/kernels/gpu/strided_slice_op.h"

#include "itex/core/kernels/gpu/inplace_ops_functor.h"
#include "itex/core/kernels/gpu/strided_slice_op_impl.h"
#include "itex/core/kernels/gpu/strided_slice_op_util.h"
#include "itex/core/kernels/gpu/training_op_helpers.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/types.h"

// TODO(itex): remove InputTensorType template argument when we have
// TF_InputIsRef C-API. Currently, we cannot distinguish normal tensor or ref
// tensor, during kernel execution. So we have to use InputTensorType to
// distinguish them during compiling.

enum class InputTensorType { ResourceTensor, RefTensor, NormalTensor };

namespace itex {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class StridedSliceOp : public OpKernel {
 public:
  explicit StridedSliceOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask_));
  }

  void Compute(OpKernelContext* context) override {
    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    OP_REQUIRES_OK(
        context, ValidateStridedSliceOp(
                     &context->input(1), &context->input(2), context->input(3),
                     context->input(0).shape(), begin_mask_, end_mask_,
                     ellipsis_mask_, new_axis_mask_, shrink_axis_mask_,
                     &processing_shape, &final_shape, &is_identity,
                     &is_simple_slice, &slice_dim0, &begin, &end, &strides));
    const Tensor& input = context->input(0);

    // Optimization #1, slice is a no-op plus reshape
    if (is_identity) {
      Tensor tmp;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                     final_shape, &tmp));
      OP_REQUIRES(context, tmp.CopyFrom(input, final_shape),
                  errors::Internal("Copy failed"));
      context->set_output(0, tmp);
      return;
    }

    // TODO(itex): Enable it with Tensor::Slice().
    // Optimization #2, slice is memory contiguous (only occurs in dim 0)

    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, final_shape, &result));
    const int input_dims = input.dims();
    const int processing_dims = processing_shape.dims();

    if (processing_shape.num_elements() > 0) {
#define HANDLE_DIM(NDIM)                                                       \
  if (processing_dims == NDIM) {                                               \
    HandleStridedSliceCase<Device, T, NDIM>(context, begin, end, strides,      \
                                            processing_shape, is_simple_slice, \
                                            result);                           \
    return;                                                                    \
  }

      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);
      HANDLE_DIM(8);

#undef HANDLE_DIM

      OP_REQUIRES(context, false,
                  errors::Unimplemented("Unhandled input dimensions ",
                                        input_dims, "  ", processing_dims));
    }
  }

 private:
  int32 begin_mask_;
  int32 end_mask_;
  int32 ellipsis_mask_;
  int32 new_axis_mask_;
  int32 shrink_axis_mask_;
};

template <typename Device, typename T>
class StridedSliceGradOp : public OpKernel {
 public:
  explicit StridedSliceGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask_));
  }

  void Compute(OpKernelContext* context) override {
    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    TensorShape input_shape;
    const Tensor& input_shape_tensor = context->input(0);
    OP_REQUIRES(
        context, input_shape_tensor.dims() == 1,
        errors::InvalidArgument("shape must be 1-D, got shape.shape = ",
                                input_shape_tensor.shape().DebugString()));
    if (input_shape_tensor.dtype() == DT_INT32) {
      OP_REQUIRES_OK(
          context, TensorShapeUtils::MakeShape(input_shape_tensor.vec<int32>(),
                                               &input_shape));
    } else if (input_shape_tensor.dtype() == DT_INT64) {
      OP_REQUIRES_OK(
          context, TensorShapeUtils::MakeShape(input_shape_tensor.vec<int64>(),
                                               &input_shape));
    } else {
      ITEX_LOG(FATAL) << "shape must have type int32 or int64.";
    }

    OP_REQUIRES_OK(
        context,
        ValidateStridedSliceOp(
            &context->input(1), &context->input(2), context->input(3),
            input_shape, begin_mask_, end_mask_, ellipsis_mask_, new_axis_mask_,
            shrink_axis_mask_, &processing_shape, &final_shape, &is_identity,
            &is_simple_slice, &slice_dim0, &begin, &end, &strides));

    // Check to make sure dy is consistent with the original slice
    TensorShape dy_shape = context->input(4).shape();
    OP_REQUIRES(
        context, final_shape == dy_shape,
        errors::InvalidArgument("shape of dy was ", dy_shape.DebugString(),
                                " instead of ", final_shape.DebugString()));

    if (!context->status().ok()) return;

    // const int input_dims = input.dims();
    const int processing_dims = processing_shape.dims();
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &result));

    if (processing_shape.dims() == 0) {
      const Tensor& in = context->input(4);
      OP_REQUIRES(context, result->CopyFrom(in, processing_shape),
                  errors::Internal("Copy failed"));
      return;
    }

#define HANDLE_DIM(NDIM)                                                      \
  if (processing_dims == NDIM) {                                              \
    HandleStridedSliceGradCase<Device, T, NDIM>(context, begin, end, strides, \
                                                processing_shape,             \
                                                is_simple_slice, result);     \
    return;                                                                   \
  }

    HANDLE_DIM(1);
    HANDLE_DIM(2);
    HANDLE_DIM(3);
    HANDLE_DIM(4);
    HANDLE_DIM(5);
    HANDLE_DIM(6);
    HANDLE_DIM(7);

#undef HANDLE_DIM
  }

 private:
  int32 begin_mask_;
  int32 end_mask_;
  int32 ellipsis_mask_;
  int32 new_axis_mask_;
  int32 shrink_axis_mask_;
};

template <typename Device, typename T, InputTensorType TensorType>
class StridedSliceAssignOp : public OpKernel {
 public:
  explicit StridedSliceAssignOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
  }

  void Compute(OpKernelContext* context) override {
    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    Tensor* old_lhs = nullptr;
    Tensor tmp;
    if (TensorType == InputTensorType::NormalTensor) {
      const Tensor& input = context->input(0);

      int forwarded_input;
      OP_REQUIRES_OK(context,
                     context->forward_input_or_allocate_output(
                         {0}, 0, input.shape(), &old_lhs, &forwarded_input));
      if (forwarded_input < 0) {
        OP_REQUIRES_OK(context,
                       itex::functor::DoCopy(context->eigen_device<Device>(),
                                             input, old_lhs));
      }
    } else {
      if (TensorType == InputTensorType::RefTensor) {
        // Condition for input is ref tensor
        context->forward_ref_input_to_ref_output(0, 0);
        tmp = context->mutable_input(0, true);
        old_lhs = &tmp;
      } else {
        // Condition for input is DT_RESOURCE
        auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
            context, /* do_lock */ true, /* sparse */ true, {0});
        OP_REQUIRES_OK(context, GetInputTensorFromVariable<Device, T>(
                                    context, 0, /* lock_held unused */ true,
                                    /* sparse */ true, &tmp));
        OP_REQUIRES(context, tmp.dtype() == DataTypeToEnum<T>::value,
                    errors::InvalidArgument(
                        "l-value dtype ", DataTypeString(tmp.dtype()),
                        " does not match r-value dtype ",
                        DataTypeString(DataTypeToEnum<T>::value)));

        old_lhs = &tmp;
      }
    }

    OP_REQUIRES_OK(
        context, ValidateStridedSliceOp(
                     &context->input(1), &context->input(2), context->input(3),
                     old_lhs->shape(), begin_mask, end_mask, ellipsis_mask,
                     new_axis_mask, shrink_axis_mask, &processing_shape,
                     &final_shape, &is_identity, &is_simple_slice, &slice_dim0,
                     &begin, &end, &strides));

    if (processing_shape.num_elements()) {
      const Tensor& input = context->input(4);
      TensorShape input_shape = input.shape();
      TensorShape original_shape = old_lhs->shape();
      // TODO(itex): We only should need input_shape to be broadcastable to
      // final_shape
      OP_REQUIRES(
          context, final_shape == input_shape,
          errors::Unimplemented(
              "sliced l-value shape ", final_shape.DebugString(),
              " does not match r-value shape ", input_shape.DebugString(),
              ". Automatic broadcasting not ", "yet implemented."));
      const int processing_dims = processing_shape.dims();

// 0-dimensional case implies the left and right are exactly the same
// scalar shape

// Handle general dimensions
#define HANDLE_DIM(NDIM)                                                       \
  if (processing_dims == NDIM) {                                               \
    HandleStridedSliceAssignCase<Device, T, NDIM>()(context, begin, end,       \
                                                    strides, processing_shape, \
                                                    is_simple_slice, old_lhs); \
    return;                                                                    \
  }
      HANDLE_DIM(0);
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);
      HANDLE_DIM(8);
#undef HANDLE_DIM

      OP_REQUIRES(context, false,
                  errors::Unimplemented("Unhandled input dimensions ",
                                        processing_dims));
    }
  }

 private:
  int32 begin_mask, end_mask;
  int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

#define REGISTER_GPU(type)                                                  \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")                              \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<type>("T")                    \
                              .HostMemory("begin")                          \
                              .HostMemory("end")                            \
                              .HostMemory("strides"),                       \
                          StridedSliceOp<GPUDevice, type>)                  \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")                          \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<type>("T")                    \
                              .HostMemory("shape")                          \
                              .HostMemory("begin")                          \
                              .HostMemory("end")                            \
                              .HostMemory("strides"),                       \
                          StridedSliceGradOp<GPUDevice, type>)              \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StridedSliceAssign")                                            \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<type>("T")                                        \
          .HostMemory("begin")                                              \
          .HostMemory("end")                                                \
          .HostMemory("strides"),                                           \
      StridedSliceAssignOp<GPUDevice, type, InputTensorType::RefTensor>)    \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorStridedSliceUpdate")                                      \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<type>("T")                                        \
          .HostMemory("begin")                                              \
          .HostMemory("end")                                                \
          .HostMemory("strides"),                                           \
      StridedSliceAssignOp<GPUDevice, type, InputTensorType::NormalTensor>) \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ResourceStridedSliceAssign")                                    \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<type>("T")                                        \
          .HostMemory("ref")                                                \
          .HostMemory("begin")                                              \
          .HostMemory("end")                                                \
          .HostMemory("strides"),                                           \
      StridedSliceAssignOp<GPUDevice, type, InputTensorType::ResourceTensor>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_bool(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);

TF_CALL_uint8(REGISTER_GPU);
TF_CALL_int8(REGISTER_GPU);
TF_CALL_int16(REGISTER_GPU);
TF_CALL_uint32(REGISTER_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_int64(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
#endif  // ITEX_ENABLE_DOUBLE

// A special GPU kernel for int32.
REGISTER_KERNEL_BUILDER(Name("StridedSlice")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("output"),
                        StridedSliceOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("shape")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("dy")
                            .HostMemory("output"),
                        StridedSliceGradOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(
    Name("StridedSliceAssign")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T")
        .HostMemory("ref")
        .HostMemory("begin")
        .HostMemory("end")
        .HostMemory("strides"),
    StridedSliceAssignOp<CPUDevice, int32, InputTensorType::RefTensor>);
REGISTER_KERNEL_BUILDER(
    Name("ResourceStridedSliceAssign")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T")
        .HostMemory("ref")
        .HostMemory("begin")
        .HostMemory("end")
        .HostMemory("strides"),
    StridedSliceAssignOp<CPUDevice, int32, InputTensorType::ResourceTensor>);
REGISTER_KERNEL_BUILDER(
    Name("TensorStridedSliceUpdate")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T")
        .HostMemory("input")
        .HostMemory("begin")
        .HostMemory("end")
        .HostMemory("strides"),
    StridedSliceAssignOp<CPUDevice, int32, InputTensorType::NormalTensor>);
#undef REGISTER_GPU

}  // namespace itex
