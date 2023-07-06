/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/kernels/gpu/split_lib.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tlen>
class SplitVOpBase : public OpKernel {
 public:
  explicit SplitVOpBase(OpKernelConstruction* c) : OpKernel(c) {}

  void ComputeEasyCases(OpKernelContext* context, bool* done,
                        std::vector<Tlen>* split_sizes_vec) {
    const int32 num_split = context->num_outputs();
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const Tensor& split_tensor = context->input(1);
    const Tensor& split_dim_tensor = context->input(2);

    OP_REQUIRES(context, split_dim_tensor.NumElements() == 1,
                errors::InvalidArgument("split_dim_tensor must have "
                                        "exactly one element."));

    const int32 split_dim_orig = split_dim_tensor.flat<int32>()(0);
    const int32 split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;

    OP_REQUIRES(
        context,
        split_tensor.dims() == 1 && split_tensor.NumElements() == num_split,
        errors::InvalidArgument("size of the split_tensor must be 1-D and have "
                                "the same elements as outputs got ",
                                split_tensor.dims(), " -D and ",
                                split_tensor.NumElements(), " elements"));

    auto split_sizes_d = split_tensor.vec<Tlen>();

    split_sizes_vec->resize(split_sizes_d.size());

    std::copy(split_sizes_d.data(), split_sizes_d.data() + split_sizes_d.size(),
              split_sizes_vec->begin());

    OP_REQUIRES(
        context, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    OP_REQUIRES(
        context, 0 <= split_dim && split_dim < input.dims(),
        errors::InvalidArgument("-input rank(-", input.dims(),
                                ") <= split_dim < input rank (", input.dims(),
                                "), but got ", split_dim_orig));

    Tlen input_size_split_dim = input_shape.dim_size(split_dim);

    // Special case 1: num_split == 1. Nothing to do.
    if (num_split == 1) {
      context->set_output(0, context->input(0));
      OP_REQUIRES(
          context, (*split_sizes_vec)[0] == input_size_split_dim,
          errors::InvalidArgument("If there is only one output, it must have "
                                  "the same size as the input. Input size: ",
                                  input_size_split_dim,
                                  " output size: ", (*split_sizes_vec)[0]));
      *done = true;
      return;
    }

    // Determine sizes of output, in case of a -1 input value
    int neg_one_dim = -1;
    Tlen determined_size = 0;
    for (int d = 0; d < split_sizes_vec->size(); ++d) {
      Tlen size = (*split_sizes_vec)[d];

      if (size == -1) {
        OP_REQUIRES(context, neg_one_dim == -1,
                    errors::InvalidArgument("There can only be one -1 in the "
                                            "input."));
        neg_one_dim = d;
      } else {
        determined_size += size;
      }
    }

    OP_REQUIRES(
        context,
        (neg_one_dim == -1 && determined_size == input_size_split_dim) ||
            (neg_one_dim >= 0 && determined_size <= input_size_split_dim),
        errors::InvalidArgument("Determined shape must either match "
                                "input shape along split_dim exactly if "
                                "fully specified, or be less than the size of "
                                "the input along split_dim if not fully "
                                "specified.  Got: ",
                                determined_size));

    if (neg_one_dim >= 0) {
      (*split_sizes_vec)[neg_one_dim] = input_size_split_dim - determined_size;
    }

    // TODO(itex): Special case 2: split along the 1st dimension. We can share
    // the underlying buffer.
  }

  template <typename IndexType>
  std::tuple<IndexType, IndexType, IndexType> SetDims(
      const TensorShape& input_shape, const int32 split_dim) const {
    static_assert(std::is_integral<IndexType>::value,
                  "IndexType must be an integer type");
    int32 prefix_dim_size = 1;
    for (int i = 0; i < split_dim; ++i) {
      prefix_dim_size *= input_shape.dim_size(i);
    }

    // Caller must ensure that dim_size and suffix_dim_size are <
    // std::numeric_limits<IndexType>::max()
    IndexType split_dim_size =
        static_cast<IndexType>(input_shape.dim_size(split_dim));

    IndexType suffix_dim_size = 1;
    for (int i = split_dim + 1; i < input_shape.dims(); ++i) {
      suffix_dim_size *= static_cast<IndexType>(input_shape.dim_size(i));
    }
    return std::make_tuple(prefix_dim_size, split_dim_size, suffix_dim_size);
  }
};

// Partial specialization for GPU
template <typename T, typename Tlen>
class SplitVOpGPU : public SplitVOpBase<GPUDevice, T, Tlen> {
 public:
  typedef SplitVOpBase<GPUDevice, T, Tlen> Base;
  explicit SplitVOpGPU(OpKernelConstruction* c) : Base(c) {}

  void Compute(OpKernelContext* context) override {
    bool done = false;
    std::vector<Tlen> split_sizes_vec;
    Base::ComputeEasyCases(context, &done, &split_sizes_vec);
    if (!context->status().ok() || done) {
      return;
    }

    const int32 num_split = context->num_outputs();
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const int32 split_dim_orig = context->input(2).flat<int32>()(0);
    const int32 split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;
    OP_REQUIRES(
        context,
        FastBoundsCheck(input.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Split on GPU requires input size "
                                "< max int32"));

    Eigen::DenseIndex prefix_dim_size;
    Eigen::DenseIndex split_dim_size;
    Eigen::DenseIndex suffix_dim_size;

    std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
        Base::template SetDims<Eigen::DenseIndex>(input_shape, split_dim);

    auto input_reshaped =
        input.shaped<T, 2>({prefix_dim_size, split_dim_size * suffix_dim_size});

    Eigen::DSizes<Eigen::DenseIndex, 2> indices{0, 0};

    for (int i = 0; i < num_split; ++i) {
      TensorShape output_shape(input_shape);
      output_shape.set_dim(split_dim, split_sizes_vec[i]);
      Tensor* result = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, output_shape, &result));

      Eigen::DSizes<Eigen::DenseIndex, 2> sizes{
          prefix_dim_size, split_sizes_vec[i] * suffix_dim_size};

      if (sizes.TotalSize() > 0) {
        auto result_shaped = result->shaped<T, 2>(
            {prefix_dim_size, split_sizes_vec[i] * suffix_dim_size});

        functor::Split<T, 2>()(context->eigen_device<GPUDevice>(),
                               result_shaped, input_reshaped, indices, sizes);
      }
      indices[1] += split_sizes_vec[i] * suffix_dim_size;
    }
  }
};

#define REGISTER_GPU(type, len_type)                            \
  REGISTER_KERNEL_BUILDER(Name("SplitV")                        \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<len_type>("Tlen") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("size_splits")        \
                              .HostMemory("split_dim"),         \
                          SplitVOpGPU<type, len_type>);

#define REGISTER_GPU_LEN(type) \
  REGISTER_GPU(type, int32);   \
  REGISTER_GPU(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_LEN);
// TF_CALL_int32(REGISTER_GPU_LEN);
TF_CALL_complex64(REGISTER_GPU_LEN);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_LEN);
TF_CALL_complex128(REGISTER_GPU_LEN);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_GPU_LEN
#undef REGISTER_GPU

}  // namespace itex
