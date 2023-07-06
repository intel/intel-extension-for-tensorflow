/* Copyright (c) 2021-2023 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/sparse_slice_grad_op.h"

#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;
namespace {

// Helper that wraps a multi-dimensional index and provides a comparison
// operator that shifts and then compares against another index. This is
// used for searching for an input index within the output indices.
struct MultiIndexComparator {
  EIGEN_DEVICE_FUNC MultiIndexComparator(int rank, const int64_t* indices,
                                         const int64_t* input_start)
      : rank_(rank), indices_(indices), input_start_(input_start) {}

  EIGEN_DEVICE_FUNC bool operator<(const MultiIndexComparator& input) const {
    for (int d = 0; d < rank_; ++d) {
      // Shift output index by the slice start.
      int64_t output_index_i = indices_[d] + input_start_[d];
      int64_t input_index_i = input.indices_[d];
      // Lexicographically compare the indexes.
      if (output_index_i < input_index_i) return true;
      if (output_index_i > input_index_i) return false;
    }
    return false;
  }

  EIGEN_DEVICE_FUNC bool operator==(const MultiIndexComparator& input) const {
    for (int d = 0; d < rank_; ++d) {
      // Shift output index by the slice start.
      int64_t output_index_i = indices_[d] + input_start_[d];
      int64_t input_index_i = input.indices_[d];
      if (output_index_i != input_index_i) return false;
    }
    return true;
  }

 private:
  int rank_;
  const int64_t* indices_;
  const int64_t* input_start_;
};

// Generate multi index through indices and other element
struct MultiIndexGenerator {
  MultiIndexGenerator(int rank, const int64_t* indices,
                      const int64_t* input_start)
      : rank_(rank), indices_(indices), input_start_(input_start) {}

  // It is unable to override operator '[]' because reference returned value is
  // mandatory for that operator. Use accessor function instead.
  EIGEN_DEVICE_FUNC MultiIndexComparator at(int64_t i) const {
    return {rank_, indices_ + i * rank_, input_start_};
  }

 private:
  int rank_;
  const int64_t* indices_;
  const int64_t* input_start_;
};

// Similar to std::lower_bound, this returns the index of the first comparator
// in MultiIndexGenerator[0] to MultiIndexGenerator[count] that is not less
// than `target_comparator`, or `count` if no such comparator is found.
int64_t lower_bound(const MultiIndexGenerator& generator, int64_t count,
                    const MultiIndexComparator& target_comparator) {
  int64_t step = 0;
  int64_t start_index = 0;
  while (count > 0) {
    step = count / 2;
    auto cur_comparator = generator.at(step + start_index);
    if (cur_comparator < target_comparator) {
      start_index += step + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }

  return start_index;
}

template <typename T>
struct SparseSliceGradKernel {
  SparseSliceGradKernel(const int64_t input_nnz, const int64_t output_nnz,
                        const MultiIndexGenerator input_indices,
                        const MultiIndexGenerator output_indices,
                        const T* backprop_val_grad, T* val_grad)
      : input_nnz_(input_nnz),
        output_nnz_(output_nnz),
        input_indices_(input_indices),
        output_indices_(output_indices),
        backprop_val_grad_(backprop_val_grad),
        val_grad_(val_grad) {}

  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id >= input_nnz_) return;

    const int64_t input_nz = id;
    const int64_t output_nz =
        lower_bound(output_indices_, output_nnz_, input_indices_.at(input_nz));

    if (output_nz < output_nnz_ &&
        output_indices_.at(output_nz) == input_indices_.at(input_nz)) {
      // Found the input index in the output, so copy its gradient value.
      val_grad_[input_nz] = backprop_val_grad_[output_nz];
    } else {
      // Not found, meaning it was not within the slice volume.
      val_grad_[input_nz] = T(0);
    }
  }

 private:
  const int64_t input_nnz_;
  const int64_t output_nnz_;
  const MultiIndexGenerator input_indices_;
  const MultiIndexGenerator output_indices_;
  const T* backprop_val_grad_;
  T* val_grad_;
};

template <typename T>
Status LaunchSparseSliceGrad(const GPUDevice& d, const int64_t input_nnz,
                             const int64_t output_nnz,
                             const MultiIndexGenerator input_indices,
                             const MultiIndexGenerator output_indices,
                             const T* backprop_val_grad, T* val_grad) {
  d.stream()->submit([&](sycl::handler& cgh) {
    auto max_group_size =
        d.stream()
            ->get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    auto num_work_group = (input_nnz + max_group_size - 1) / max_group_size;
    sycl::range<1> local_range(max_group_size);
    sycl::range<1> global_range(max_group_size * num_work_group);

    SparseSliceGradKernel<T> task(input_nnz, output_nnz, input_indices,
                                  output_indices, backprop_val_grad, val_grad);
    cgh.parallel_for<SparseSliceGradKernel<T>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

}  // namespace

namespace functor {

template <typename T>
struct SparseSliceGradFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* ctx,
                  typename TTypes<T>::ConstFlat backprop_val_grad,
                  typename TTypes<int64_t>::ConstMatrix input_indices_mat,
                  typename TTypes<int64_t>::ConstFlat input_start_flat,
                  typename TTypes<int64_t>::ConstMatrix output_indices_mat,
                  typename TTypes<T>::Flat val_grad) const {
    const int rank = input_indices_mat.dimension(1);

    MultiIndexGenerator input_indices(rank, input_indices_mat.data(),
                                      /*input_start=*/nullptr);
    MultiIndexGenerator output_indices(rank, output_indices_mat.data(),
                                       input_start_flat.data());

    const int64_t input_nnz = input_indices_mat.dimension(0);
    const int64_t output_nnz = output_indices_mat.dimension(0);

    const GPUDevice& device = ctx->eigen_gpu_device();

    auto status = LaunchSparseSliceGrad<T>(
        device, input_nnz, output_nnz, input_indices, output_indices,
        backprop_val_grad.data(), val_grad.data());
    OP_REQUIRES_OK(ctx, status);
  }
};

}  // namespace functor

template <typename T>
class SparseSliceGradOp : public OpKernel {
 public:
  explicit SparseSliceGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor *backprop_val_grad, *input_indices, *output_indices,
        *input_start;
    OP_REQUIRES_OK(ctx, ctx->input("backprop_val_grad", &backprop_val_grad));
    OP_REQUIRES_OK(ctx, ctx->input("input_indices", &input_indices));
    OP_REQUIRES_OK(ctx, ctx->input("input_start", &input_start));
    OP_REQUIRES_OK(ctx, ctx->input("output_indices", &output_indices));

    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsMatrix(input_indices->shape()) &&
            TensorShapeUtils::IsMatrix(output_indices->shape()),
        errors::InvalidArgument("Input and output indices should be matrices "
                                "but received shapes: ",
                                input_indices->shape().DebugString(), " and ",
                                output_indices->shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(backprop_val_grad->shape()),
        errors::InvalidArgument(
            "Input backprop_val_grad should be a vector but received shape: ",
            backprop_val_grad->shape().DebugString()));
    OP_REQUIRES(
        ctx, input_indices->dim_size(1) == output_indices->dim_size(1),
        errors::InvalidArgument("The input and output should have the same "
                                "ndims: got: ",
                                input_indices->dim_size(1), " and ",
                                output_indices->dim_size(1)));
    OP_REQUIRES(
        ctx, output_indices->dim_size(0) <= input_indices->dim_size(0),
        errors::InvalidArgument(
            "# rows of output_indices should be not greater "
            "than of input_indices, got ",
            output_indices->dim_size(0), " and ", input_indices->dim_size(0)));
    OP_REQUIRES(ctx,
                backprop_val_grad->NumElements() == output_indices->dim_size(0),
                errors::InvalidArgument(
                    "# elements of backprop_val_grad and # rows of "
                    "output_indices should match (#nnz of sum): got ",
                    backprop_val_grad->NumElements(), " and ",
                    output_indices->dim_size(0)));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_start->shape()),
                errors::InvalidArgument(
                    "The input_start should be a vector but received shape ",
                    input_start->shape().DebugString()));

    const int num_dims = input_indices->dim_size(1);
    OP_REQUIRES(ctx, num_dims == input_start->NumElements(),
                errors::InvalidArgument(
                    "Expected input_start to be a vector of length ", num_dims,
                    " but got length ", input_start->NumElements()));

    const int64_t input_nnz = input_indices->dim_size(0);

    Tensor* val_grad;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({input_nnz}), &val_grad));

    if (input_nnz == 0) return;

    functor::SparseSliceGradFunctor<GPUDevice, T>()(
        ctx, backprop_val_grad->flat<T>(), input_indices->matrix<int64_t>(),
        input_start->flat<int64_t>(), output_indices->matrix<int64_t>(),
        val_grad->flat<T>());
  }
};

#define REGISTER_KERNELS(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseSliceGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SparseSliceGradOp<type>)
TF_CALL_INTEGRAL_TYPES(REGISTER_KERNELS);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS);
TF_CALL_complex64(REGISTER_KERNELS);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_KERNELS);
TF_CALL_complex128(REGISTER_KERNELS);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_KERNELS

}  // namespace itex
