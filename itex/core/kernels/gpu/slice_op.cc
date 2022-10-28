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

#include "itex/core/kernels/gpu/slice_op.h"

#include "itex/core/kernels/gpu/ops_util.h"
#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace {

gtl::InlinedVector<int64, 4> IntTensorToInt64Vec(const Tensor& tensor) {
  gtl::InlinedVector<int64, 4> out;
  if (tensor.dtype() == DT_INT32) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int32>()(i));
    }
  } else if (tensor.dtype() == DT_INT64) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int64>()(i));
    }
  } else {
    ITEX_LOG(FATAL) << "begin must be either int32 or int64";
  }
  return out;
}

}  // namespace

// Shared code that is not dependent on the type of T.  We do this to reduce
// code size by not duplicating all this for all T (float, double, int32, etc.)
static void SharedValidation(OpKernelContext* context,
                             TensorShape* output_shape, bool* is_identity,
                             bool* slice_dim0,
                             gtl::InlinedVector<int64, 4>* begin,
                             gtl::InlinedVector<int64, 4>* size) {
  const Tensor& input = context->input(0);
  const Tensor& begin_tensor = context->input(1);
  const Tensor& size_tensor = context->input(2);

  OP_REQUIRES(
      context,
      begin_tensor.shape().dims() == 1 && size_tensor.shape().dims() == 1 &&
          begin_tensor.NumElements() == input.dims() &&
          size_tensor.NumElements() == input.dims(),
      errors::InvalidArgument(
          "Expected begin and size arguments to be 1-D tensors of size ",
          input.dims(), ", but got shapes ", begin_tensor.shape().DebugString(),
          " and ", size_tensor.shape().DebugString(), " instead."));

  const int input_dims = input.dims();
  *begin = IntTensorToInt64Vec(begin_tensor);
  *size = IntTensorToInt64Vec(size_tensor);
  for (int i = 0; i < input_dims; ++i) {
    if ((*size)[i] == -1) {
      // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
      (*size)[i] = input.dim_size(i) - (*begin)[i];
    }
  }

  *is_identity = true;
  *slice_dim0 = true;
  for (int i = 0; i < input_dims; ++i) {
    int64 b = (*begin)[i];
    int64 s = (*size)[i];
    if (input.dim_size(i) == 0) {
      OP_REQUIRES(
          context, b == 0 && s == 0,
          errors::InvalidArgument("Expected begin[", i, "] == 0 (got ", b,
                                  ") and size[", i, "] == 0 ", "(got ", s,
                                  ") when ", "input.dim_size(", i, ") == 0"));
    } else {
      OP_REQUIRES(context, 0 <= b && b <= input.dim_size(i),
                  errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                          input.dim_size(i), "], but got ", b));
      OP_REQUIRES(
          context, 0 <= s && b + s <= input.dim_size(i),
          errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                  input.dim_size(i) - b, "], but ", "got ", s));
    }
    output_shape->AddDim(s);
    const bool take_all = (b == 0) && (s == input.dim_size(i));
    (*is_identity) &= take_all;
    (*slice_dim0) &= (i == 0) || take_all;
  }
}

// Extracted out code in SliceOp::Compute so that OneDnnSliceOp can reuse this
// generic code
template <typename T>
static void SharedSliceCommonCases(OpKernelContext* context,
                                   TensorShape* output_shape,
                                   gtl::InlinedVector<int64, 4>* begin,
                                   gtl::InlinedVector<int64, 4>* size,
                                   Tensor** result, bool* done,
                                   bool* sub_slice) {
  bool is_identity = true;
  bool slice_dim0 = true;
  *done = false;

  SharedValidation(context, output_shape, &is_identity, &slice_dim0, begin,
                   size);
  if (!context->status().ok()) return;
  const Tensor& input = context->input(0);
  if (is_identity) {
    ITEX_VLOG(3) << "Slice identity";
    context->set_output(0, input);
    *done = true;
    return;
  }

  // TODO(itex): should use input.Slice after Tensor.Slice is
  // implemented.
  if (slice_dim0 &&
      IsDim0SliceAligned<T>(input.shape(), (*begin)[0], (*size)[0])) {
    ITEX_VLOG(3) << "Slice dim 0: " << input.shape().DebugString();
    ITEX_CHECK_GE(input.dims(), 1);  // Otherwise, is_identity should be true.
    // context->set_output(0, input.Slice((*begin)[0], (*begin)[0] +
    // (*size)[0]));
    *sub_slice = true;
  }

  OP_REQUIRES_OK(context, context->allocate_output(0, *output_shape, result));
}

template <typename Device, typename T, int NDIM>
void HandleCase(OpKernelContext* context, const gtl::ArraySlice<int64>& begin,
                const gtl::ArraySlice<int64>& size, Tensor* result) {
  Eigen::DSizes<Eigen::DenseIndex, NDIM> indices;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes;
  for (int i = 0; i < NDIM; ++i) {
    indices[i] = begin[i];
    sizes[i] = size[i];
  }

  functor::Slice<Device, T, NDIM>()(
      context->eigen_gpu_device(), result->tensor<T, NDIM>(),
      context->input(0).tensor<T, NDIM>(), indices, sizes);
}

template <typename Device, typename T>
class SliceOp : public OpKernel {
 public:
  explicit SliceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TensorShape output_shape;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> size;
    Tensor* result = nullptr;
    bool done = false, sub_slice = false;
    SharedSliceCommonCases<T>(context, &output_shape, &begin, &size, &result,
                              &done, &sub_slice);
    if (!context->status().ok() || done == true) return;

    const Tensor& input = context->input(0);
    const int input_dims = input.dims();

    if (output_shape.num_elements() > 0 && sub_slice) {
      int64_t slice_size = input.NumElements() / input.dim_size(0);
      if (input.NumElements() <= kint32max) {
        functor::SubSliceFunctor<T, int>()(
            context->eigen_gpu_device(), input.flat<T>().data(),
            result->flat<T>().data(), begin[0], begin[0] + size[0], slice_size);
      } else {
        functor::SubSliceFunctor<T, int64_t>()(
            context->eigen_gpu_device(), input.flat<T>().data(),
            result->flat<T>().data(), begin[0], begin[0] + size[0], slice_size);
      }
      return;
    }

    if (output_shape.num_elements() > 0) {
#define HANDLE_DIM(NDIM)                                       \
  if (input_dims == NDIM) {                                    \
    HandleCase<Device, T, NDIM>(context, begin, size, result); \
    return;                                                    \
  }

      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);

#undef HANDLE_DIM

      OP_REQUIRES(
          context, false,
          errors::Unimplemented("SliceOp : Unhandled input dimensions"));
    }
  }
};

#define DEFINE_GPU_KERNELS(T)                       \
  template struct functor::Slice<GPUDevice, T, 1>;  \
  template struct functor::Slice<GPUDevice, T, 2>;  \
  template struct functor::Slice<GPUDevice, T, 3>;  \
  template struct functor::Slice<GPUDevice, T, 4>;  \
  template struct functor::Slice<GPUDevice, T, 5>;  \
  template struct functor::Slice<GPUDevice, T, 6>;  \
  template struct functor::Slice<GPUDevice, T, 7>;  \
  template struct functor::SubSliceFunctor<T, int>; \
  template struct functor::SubSliceFunctor<T, int64_t>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_bool(DEFINE_GPU_KERNELS);
TF_CALL_int8(DEFINE_GPU_KERNELS);
TF_CALL_int32(DEFINE_GPU_KERNELS);
TF_CALL_int64(DEFINE_GPU_KERNELS);
TF_CALL_complex64(DEFINE_GPU_KERNELS);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_GPU_KERNELS);
TF_CALL_complex128(DEFINE_GPU_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE

#undef DEFINE_GPU_KERNELS

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Slice")                  \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("begin")       \
                              .HostMemory("size"),       \
                          SliceOp<GPUDevice, type>)

TF_CALL_int64(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_GPU
}  // namespace itex
