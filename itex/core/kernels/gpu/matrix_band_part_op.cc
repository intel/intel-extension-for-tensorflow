/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/matrix_band_part_op.h"

#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Scalar>
struct MatrixBandParttKernel {
  MatrixBandParttKernel(int m, int n, size_t size, int num_lower_diags,
                        int num_upper_diags, const Scalar* in_ptr,
                        Scalar* out_ptr)
      : m(m),
        n(n),
        size(size),
        num_lower_diags(num_lower_diags),
        num_upper_diags(num_upper_diags),
        in_ptr(in_ptr),
        out_ptr(out_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index < size) {
      const int col = index % n;
      const int row = (index / n) % m;
      const int band_start = (num_lower_diags < 0 ? 0 : row - num_lower_diags);
      const int band_end =
          (num_upper_diags < 0 ? n : row + num_upper_diags + 1);
      if (col < band_start || col >= band_end) {
        out_ptr[index] = Scalar();
      } else {
        out_ptr[index] = in_ptr[index];
      }
    }
  }

 private:
  int m;
  int n;
  size_t size;
  int num_lower_diags;
  int num_upper_diags;
  const Scalar* in_ptr;
  Scalar* out_ptr;
};

template <typename Scalar>
struct MatrixBandPartFunctor<GPUDevice, Scalar> {
  void operator()(OpKernelContext* context, const GPUDevice& device,
                  int num_lower_diags, int num_upper_diags,
                  typename TTypes<Scalar, 3>::ConstTensor input,
                  typename TTypes<Scalar, 3>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int m = input.dimension(1);
    const int n = input.dimension(2);
    auto stream = context->GetDeviceStream();
    auto group_size = (*stream)
                          .get_device()
                          .get_info<sycl::info::device::max_work_group_size>();
    auto size = batch_size * m * n;
    auto num_wg = (size + group_size - 1) / group_size;
    stream->submit([&](sycl::handler& cgh) {
      auto in_ptr = input.data();
      auto out_ptr = output.data();
      MatrixBandParttKernel<Scalar> task(m, n, size, num_lower_diags,
                                         num_upper_diags, in_ptr, out_ptr);
      cgh.parallel_for<MatrixBandParttKernel<Scalar>>(
          sycl::nd_range<1>(sycl::range<1>(num_wg * group_size),
                            sycl::range<1>(group_size)),
          task);
    });
  }
};

#define DEFINE_GPU_SPEC(T) template struct MatrixBandPartFunctor<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);
TF_CALL_bool(DEFINE_GPU_SPEC);
TF_CALL_int32(DEFINE_GPU_SPEC);
TF_CALL_int64(DEFINE_GPU_SPEC);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(DEFINE_GPU_SPEC);
TF_CALL_complex128(DEFINE_GPU_SPEC);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_complex64(DEFINE_GPU_SPEC);
#undef DEFINE_GPU_SPEC
}  // namespace functor

template <typename Device, typename T>
class MatrixBandPartOp : public OpKernel {
 public:
  explicit MatrixBandPartOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));
    auto input_reshaped = input.flat_inner_dims<T, 3>();

    const Tensor& num_lower_in = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_lower_in.shape()),
                errors::InvalidArgument("num_lower must be scalar, got shape ",
                                        num_lower_in.shape().DebugString()));

    auto as_int64_scalar = [](const Tensor& tensor) -> int64 {
      if (tensor.dtype() == DT_INT32) {
        return tensor.scalar<int32>()();
      } else {
        return tensor.scalar<int64>()();
      }
    };
    const int64 num_lower = as_int64_scalar(num_lower_in);
    OP_REQUIRES(
        context, num_lower <= input_reshaped.dimension(1),
        errors::InvalidArgument(
            "num_lower must be negative or less or equal to number of rows (",
            input_reshaped.dimension(1), ") got: ", num_lower));

    const Tensor& num_upper_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_upper_in.shape()),
                errors::InvalidArgument("num_upper must be scalar, got shape ",
                                        num_upper_in.shape().DebugString()));
    const int64 num_upper = as_int64_scalar(num_upper_in);
    OP_REQUIRES(context, num_upper <= input_reshaped.dimension(2),
                errors::InvalidArgument("num_upper must be negative or less or "
                                        "equal to number of columns (",
                                        input_reshaped.dimension(2),
                                        ") got: ", num_upper));

    if (input.NumElements() == 0 ||
        ((num_lower < 0 || num_lower == input_reshaped.dimension(1)) &&
         (num_upper < 0 || num_upper == input_reshaped.dimension(2)))) {
      // This is a no-op.
      context->set_output(0, input);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input_shape, &output));
    auto output_reshaped = output->flat_inner_dims<T, 3>();
    functor::MatrixBandPartFunctor<Device, T> fn;
    fn(context, context->eigen_device<Device>(), num_lower, num_upper,
       input_reshaped, output_reshaped);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixBandPartOp);
};

#define REGISTER_MATRIX_BAND_PART_GPU(type)              \
  REGISTER_KERNEL_BUILDER(Name("MatrixBandPart")         \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("num_lower")   \
                              .HostMemory("num_upper"),  \
                          MatrixBandPartOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_MATRIX_BAND_PART_GPU);
TF_CALL_bool(REGISTER_MATRIX_BAND_PART_GPU);
TF_CALL_int32(REGISTER_MATRIX_BAND_PART_GPU);
TF_CALL_int64(REGISTER_MATRIX_BAND_PART_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_MATRIX_BAND_PART_GPU);
TF_CALL_complex128(REGISTER_MATRIX_BAND_PART_GPU);
#endif  // ITEX_ENABLE_DOUBLE
TF_CALL_complex64(REGISTER_MATRIX_BAND_PART_GPU);
#undef REGISTER_MATRIX_BAND_PART_GPU
}  // namespace itex
