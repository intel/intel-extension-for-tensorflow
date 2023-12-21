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

#include "itex/core/kernels/gpu/where_op.h"

#include <limits>
#include <utility>

#include "itex/core/kernels/gpu/scan_ops_gpu.h"

namespace itex {
namespace functor {

template <typename TIndex, typename T, int NDIM>
Eigen::array<TIndex, NDIM> CalculateStrides(
    typename TTypes<T, NDIM>::ConstTensor input) {
  const Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
  Eigen::array<TIndex, NDIM> strides;
  EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                       static_cast<int>(Eigen::RowMajor)),
                      INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);
  strides[NDIM - 1] = 1;
  for (int i = NDIM - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

template <typename InputT, typename OutputT>
struct NonZero {
  inline OutputT operator()(const InputT& x) const {
    return OutputT(x != InputT(0));
  }
};

template <typename T, typename TIndex>
Status InputCumSum<T, TIndex>::Compute(
    OpKernelContext* context, typename TTypes<T>::ConstFlat input,
    typename TTypes<TIndex>::Vec input_cumsum, TIndex num_elems) {
  TF_RETURN_IF_ERROR(launchFullScan<const T, TIndex, TIndex, sycl::plus<TIndex>,
                                    NonZero<T, TIndex>>(
      context, input.data(), input_cumsum.data(), TIndex(0),
      sycl::plus<TIndex>(), false, false, num_elems, NonZero<T, TIndex>()));
  return Status::OK();
}

// Explicit instantiation for SparseSliceOp.
template struct InputCumSum<int, int64_t>;

template <int NDIM, typename T, typename TIndex>
struct WhereKernel {
  WhereKernel(const T* input_ptr, TIndex* input_cumsum_ptr, int64_t* output_ptr,
              Eigen::array<TIndex, NDIM> strides, size_t input_size)
      : input_ptr(input_ptr),
        input_cumsum_ptr(input_cumsum_ptr),
        output_ptr(output_ptr),
        strides(strides),
        input_size(input_size) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id < input_size && (input_ptr[id] != T(0))) {
      auto row_lin = (input_cumsum_ptr[id] - 1) * NDIM;
#pragma unroll
      for (int c = 0; c < NDIM; ++c) {
        auto value = id / strides[c];
        output_ptr[row_lin + c] = value;
        id -= value * strides[c];
      }
    }
  }

 private:
  const T* input_ptr;
  TIndex* input_cumsum_ptr;
  int64_t* output_ptr;
  Eigen::array<TIndex, NDIM> strides;
  size_t input_size;
};

template <int NDIM, typename T, typename TIndex>
struct Where<GPUDevice, NDIM, T, TIndex> {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext*, const GPUDevice& d,
      typename TTypes<T, NDIM>::ConstTensor input,
      typename TTypes<TIndex>::Vec input_cumsum,
      typename TTypes<int64_t>::Matrix output) {
    if (output.dimension(0) == 0) {
      // Nothing to do.
      return Status::OK();
    }

    auto compute = [&d, input, input_cumsum, output](sycl::handler& cgh) {
      auto max_wg_size =
          d.stream()
              ->get_device()
              .template get_info<sycl::info::device::max_work_group_size>();
      const Eigen::array<TIndex, NDIM> strides =
          CalculateStrides<TIndex, T, NDIM>(input);
      auto input_ptr = input.data();
      auto input_cumsum_ptr = input_cumsum.data();
      auto output_ptr = output.data();
      auto input_size = input.size();
      sycl::nd_range<1> nd_rng(sycl::range<1>((input_size + max_wg_size - 1) /
                                              max_wg_size * max_wg_size),
                               sycl::range<1>(max_wg_size));
      WhereKernel<NDIM, T, TIndex> task(input_ptr, input_cumsum_ptr, output_ptr,
                                        strides, input_size);
      cgh.parallel_for<Where<GPUDevice, NDIM, T, TIndex>>(nd_rng, task);
    };
    d.stream()->submit(std::move(compute));

    return Status::OK();
  }
};

}  // namespace functor

template <typename T>
class WhereOp : public OpKernel {
 public:
  explicit WhereOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int input_dims = input.dims();

    // Although it's ok for USM, but it will
    // cause divided-by-0 error in Eigen.
    if (input.NumElements() == 0) {
      Tensor* output;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  0, TensorShape({0, input_dims}), &output));
      return;
    }

    if (input.NumElements() < std::numeric_limits<int32>::max()) {
      ComputeType<int32>(input, input_dims, context);
    } else {
      ComputeType<int64>(input, input_dims, context);
    }
  }

  template <typename Tindex>
  void ComputeType(const Tensor& input, const int input_dims,
                   OpKernelContext* context) {
    // Instead of doing a sum to compute num_true, we compute a cumsum to then
    // get the counter of true elements seen so far.
    Tindex input_size = input.NumElements();

    Tensor input_cumsum;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Tindex>::v(),
                                                   TensorShape({input_size}),
                                                   &input_cumsum));
    auto input_cumsum_t = input_cumsum.vec<Tindex>();

    // Push kernel to stream to get number of true elements.
    Status s = functor::InputCumSum<T, Tindex>::Compute(
        context, input.flat<T>(), input_cumsum_t, input_size);
    OP_REQUIRES_OK(context, s);

    // Copy num_true to host;
    Tindex num_true_host;
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    d.stream()
        ->memcpy(&num_true_host, input_cumsum_t.data() + input_size - 1,
                 sizeof(Tindex))
        .wait();

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({num_true_host, input_dims}), &output));

#define HANDLE_DIM(NDIM)                                            \
  case NDIM: {                                                      \
    Status s = functor::Where<GPUDevice, NDIM, T, Tindex>::Compute( \
        context, d, input.tensor<T, NDIM>(), input_cumsum_t,        \
        output->matrix<int64>());                                   \
    OP_REQUIRES_OK(context, s);                                     \
  } break;

    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "WhereOp: Unhandled input dimensions: ", input_dims));
    }
#undef HANDLE_DIM
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WhereOp);
};

#define REGISTER_WHERE_OP(T) \
  REGISTER_KERNEL_BUILDER(   \
      Name("Where").Device(DEVICE_GPU).TypeConstraint<T>("T"), WhereOp<T>);
TF_CALL_bool(REGISTER_WHERE_OP);
TF_CALL_int8(REGISTER_WHERE_OP);
TF_CALL_uint8(REGISTER_WHERE_OP);
TF_CALL_int32(REGISTER_WHERE_OP);
TF_CALL_int64(REGISTER_WHERE_OP);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_WHERE_OP);
TF_CALL_complex64(REGISTER_WHERE_OP);
TF_CALL_complex128(REGISTER_WHERE_OP);
#endif
TF_CALL_GPU_NUMBER_TYPES(REGISTER_WHERE_OP);
#undef REGISTER_WHERE_OP

}  // namespace itex
