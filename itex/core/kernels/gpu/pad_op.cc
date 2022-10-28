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

#include "itex/core/kernels/gpu/pad_op.h"

#include <utility>
#include <vector>

#include "itex/core/utils/allocator.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Tpadding, int Dims>
struct PadKernel {
  PadKernel(const T* in_data_, T* out_data_,
            const std::array<Tpadding, Dims * 2> paddings_array_,
            const std::array<int, Dims> out_dims_,
            const std::array<int, Dims> in_strides_,
            const std::array<int, Dims> out_strides_, const T pad_value_,
            size_t nelems_)
      : in_data(in_data_),
        out_data(out_data_),
        paddings_array(paddings_array_),
        out_dims(out_dims_),
        in_strides(in_strides_),
        out_strides(out_strides_),
        pad_value(pad_value_),
        nelems(nelems_) {}
  void operator()(sycl::nd_item<1> item) const {
    const int out_idx = item.get_global_linear_id();
    if (out_idx >= nelems) {
      return;
    }
    int in_idx = 0;
    int out_idx_tmp = out_idx;

    for (int i = 0; i < Dims - 1; ++i) {
      const int idx = out_idx_tmp / out_strides[i];
      if ((idx < paddings_array[2 * i]) ||
          (idx >= out_dims[i] - paddings_array[2 * i + 1])) {
        out_data[out_idx] = pad_value;
        return;
      }
      in_idx += (idx - paddings_array[2 * i]) * in_strides[i];
      out_idx_tmp -= idx * out_strides[i];
    }
    if ((out_idx_tmp < paddings_array[2 * Dims - 2]) ||
        (out_idx_tmp >= out_dims[Dims - 1] - paddings_array[2 * Dims - 1])) {
      out_data[out_idx] = pad_value;
      return;
    }
    in_idx += (out_idx_tmp - paddings_array[2 * Dims - 2]);
    out_data[out_idx] = in_data[in_idx];
    return;
  }

 private:
  const T* in_data;
  T* out_data;
  const std::array<Tpadding, Dims * 2> paddings_array;
  const std::array<int, Dims> out_dims;
  const std::array<int, Dims> in_strides;
  const std::array<int, Dims> out_strides;
  const T pad_value;
  size_t nelems;
};

template <typename T, typename Tpadding, int Dims, int VecSize>
struct ContinuousPadKernel {
  ContinuousPadKernel(const T* in_data_, T* out_data_,
                      const std::array<Tpadding, Dims * 2> paddings_array_,
                      const std::array<int, Dims> out_dims_,
                      const std::array<int, Dims> in_strides_,
                      const std::array<int, Dims> out_strides_,
                      const T pad_value_, size_t nelems_)
      : in_data(in_data_),
        out_data(out_data_),
        paddings_array(paddings_array_),
        out_dims(out_dims_),
        in_strides(in_strides_),
        out_strides(out_strides_),
        pad_value(pad_value_),
        nelems(nelems_) {}
  void operator()(sycl::nd_item<1> item) const {
    const int out_idx_start = item.get_global_linear_id() * VecSize;

    int in_idx = 0;
    int out_idx = out_idx_start;
    if (out_idx >= nelems) return;

    int out_idx_tmp = out_idx;
    bool is_pad = false;
#pragma unroll
    for (int i = 0; i < Dims - 1; ++i) {
      const int idx = out_idx_tmp / out_strides[i];
      if ((idx < paddings_array[2 * i]) ||
          (idx >= out_dims[i] - paddings_array[2 * i + 1])) {
        is_pad = true;
      }
      in_idx += (idx - paddings_array[2 * i]) * in_strides[i];
      out_idx_tmp -= idx * out_strides[i];
    }
    if ((out_idx_tmp < paddings_array[2 * Dims - 2]) ||
        (out_idx_tmp >= out_dims[Dims - 1] - paddings_array[2 * Dims - 1])) {
      is_pad = true;
    }
    in_idx += (out_idx_tmp - paddings_array[2 * Dims - 2]);

    if (is_pad) {
      sycl::vec<T, VecSize> data(pad_value);
      *(reinterpret_cast<sycl::vec<T, VecSize>*>(out_data + out_idx_start)) =
          data;
    } else {
      sycl::vec<T, VecSize> data =
          *(reinterpret_cast<const sycl::vec<T, VecSize>*>(in_data + in_idx));
      *(reinterpret_cast<sycl::vec<T, VecSize>*>(out_data + out_idx_start)) =
          data;
    }
  }

 private:
  const T* in_data;
  T* out_data;
  const std::array<Tpadding, Dims * 2> paddings_array;
  const std::array<int, Dims> out_dims;
  const std::array<int, Dims> in_strides;
  const std::array<int, Dims> out_strides;
  const T pad_value;
  size_t nelems;
};

template <typename Tpadding, int Dims, int VecSize>
struct ContinuousPadKernel<Eigen::half, Tpadding, Dims, VecSize> {
  ContinuousPadKernel(const Eigen::half* in_data_, Eigen::half* out_data_,
                      const std::array<Tpadding, Dims * 2> paddings_array_,
                      const std::array<int, Dims> out_dims_,
                      const std::array<int, Dims> in_strides_,
                      const std::array<int, Dims> out_strides_,
                      const Eigen::half pad_value_, size_t nelems_)
      : in_data(in_data_),
        out_data(out_data_),
        paddings_array(paddings_array_),
        out_dims(out_dims_),
        in_strides(in_strides_),
        out_strides(out_strides_),
        pad_value(pad_value_),
        nelems(nelems_) {}
  void operator()(sycl::nd_item<1> item) const {
    const int out_idx_start = item.get_global_linear_id() * VecSize;

    int in_idx = 0;
    int out_idx = out_idx_start;
    if (out_idx >= nelems) return;

    int out_idx_tmp = out_idx;
    bool is_pad = false;
#pragma unroll
    for (int i = 0; i < Dims - 1; ++i) {
      const int idx = out_idx_tmp / out_strides[i];
      if ((idx < paddings_array[2 * i]) ||
          (idx >= out_dims[i] - paddings_array[2 * i + 1])) {
        is_pad = true;
      }
      in_idx += (idx - paddings_array[2 * i]) * in_strides[i];
      out_idx_tmp -= idx * out_strides[i];
    }
    if ((out_idx_tmp < paddings_array[2 * Dims - 2]) ||
        (out_idx_tmp >= out_dims[Dims - 1] - paddings_array[2 * Dims - 1])) {
      is_pad = true;
    }
    in_idx += (out_idx_tmp - paddings_array[2 * Dims - 2]);

    if (is_pad) {
      sycl::vec<sycl::half, VecSize> data(pad_value.x);
      sycl::half* out_data_ptr =
          reinterpret_cast<sycl::half*>(out_data + out_idx_start);
      *(reinterpret_cast<sycl::vec<sycl::half, VecSize>*>(out_data_ptr)) = data;
    } else {
      sycl::half* out_data_ptr =
          reinterpret_cast<sycl::half*>(out_data + out_idx_start);
      const sycl::half* in_data_ptr =
          reinterpret_cast<const sycl::half*>(in_data + in_idx);
      sycl::vec<sycl::half, VecSize> data = *(
          reinterpret_cast<const sycl::vec<sycl::half, VecSize>*>(in_data_ptr));
      *(reinterpret_cast<sycl::vec<sycl::half, VecSize>*>(
          reinterpret_cast<sycl::half*>(out_data_ptr))) = data;
    }
  }

 private:
  const Eigen::half* in_data;
  Eigen::half* out_data;
  const std::array<Tpadding, Dims * 2> paddings_array;
  const std::array<int, Dims> out_dims;
  const std::array<int, Dims> in_strides;
  const std::array<int, Dims> out_strides;
  const Eigen::half pad_value;
  size_t nelems;
};

template <typename Tpadding, int Dims, int VecSize>
struct ContinuousPadKernel<Eigen::bfloat16, Tpadding, Dims, VecSize> {
  ContinuousPadKernel(const Eigen::bfloat16* in_data_,
                      Eigen::bfloat16* out_data_,
                      const std::array<Tpadding, Dims * 2> paddings_array_,
                      const std::array<int, Dims> out_dims_,
                      const std::array<int, Dims> in_strides_,
                      const std::array<int, Dims> out_strides_,
                      const Eigen::bfloat16 pad_value_, size_t nelems_)
      : in_data(in_data_),
        out_data(out_data_),
        paddings_array(paddings_array_),
        out_dims(out_dims_),
        in_strides(in_strides_),
        out_strides(out_strides_),
        pad_value(pad_value_),
        nelems(nelems_) {}
  void operator()(sycl::nd_item<1> item) const {
    const int out_idx_start = item.get_global_linear_id() * VecSize;

    int in_idx = 0;
    int out_idx = out_idx_start;
    if (out_idx >= nelems) return;

    int out_idx_tmp = out_idx;
    bool is_pad = false;
#pragma unroll
    for (int i = 0; i < Dims - 1; ++i) {
      const int idx = out_idx_tmp / out_strides[i];
      if ((idx < paddings_array[2 * i]) ||
          (idx >= out_dims[i] - paddings_array[2 * i + 1])) {
        is_pad = true;
      }
      in_idx += (idx - paddings_array[2 * i]) * in_strides[i];
      out_idx_tmp -= idx * out_strides[i];
    }
    if ((out_idx_tmp < paddings_array[2 * Dims - 2]) ||
        (out_idx_tmp >= out_dims[Dims - 1] - paddings_array[2 * Dims - 1])) {
      is_pad = true;
    }
    in_idx += (out_idx_tmp - paddings_array[2 * Dims - 2]);

    if (is_pad) {
      sycl::vec<uint16_t, VecSize> data(pad_value.value);
      uint16_t* out_data_ptr =
          reinterpret_cast<uint16_t*>(out_data + out_idx_start);
      *(reinterpret_cast<sycl::vec<uint16_t, VecSize>*>(out_data_ptr)) = data;
    } else {
      uint16_t* out_data_ptr =
          reinterpret_cast<uint16_t*>(out_data + out_idx_start);
      const uint16_t* in_data_ptr =
          reinterpret_cast<const uint16_t*>(in_data + in_idx);
      sycl::vec<uint16_t, VecSize> data =
          *(reinterpret_cast<const sycl::vec<uint16_t, VecSize>*>(in_data_ptr));
      *(reinterpret_cast<sycl::vec<uint16_t, VecSize>*>(
          reinterpret_cast<uint16_t*>(out_data_ptr))) = data;
    }
  }

 private:
  const Eigen::bfloat16* in_data;
  Eigen::bfloat16* out_data;
  const std::array<Tpadding, Dims * 2> paddings_array;
  const std::array<int, Dims> out_dims;
  const std::array<int, Dims> in_strides;
  const std::array<int, Dims> out_strides;
  const Eigen::bfloat16 pad_value;
  size_t nelems;
};

template <typename Device, typename T, typename Tpadding>
class PadOp : public OpKernel {
 public:
  explicit PadOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);

    static const int kMinDims = 0;
    static const int kMaxDims = 6;

    OP_REQUIRES(context, kMinDims <= in0.dims() && in0.dims() <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", in0.dims()));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                in1.shape().DebugString()));

    OP_REQUIRES(
        context, in0.dims() == in1.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            in1.shape().DebugString(), " ", in0.shape().DebugString()));

    T pad_value = T();
    if (context->num_inputs() == 3) {
      const Tensor& constant_values = context->input(2);
      OP_REQUIRES(
          context, TensorShapeUtils::IsScalar(constant_values.shape()),
          errors::InvalidArgument("constant_values must be a scalar. Found: ",
                                  constant_values.shape().DebugString()));
      pad_value = context->input(2).scalar<T>()();
    }

    // Compute the shape of the output tensor, and allocate it.
    TensorShape output_shape;
    typename TTypes<Tpadding>::ConstMatrix paddings = in1.matrix<Tpadding>();
    for (int d = 0; d < in0.dims(); ++d) {
      const Tpadding before_d =
          paddings(d, 0);                       // Pad before existing elements.
      const Tpadding after_d = paddings(d, 1);  // Pad after existing elements.
      OP_REQUIRES(context, before_d >= 0 && after_d >= 0,
                  errors::InvalidArgument("Paddings must be non-negative: ",
                                          before_d, " ", after_d));
      const int64 size_d = in0.dim_size(d);
      output_shape.AddDim(before_d + size_d + after_d);
    }

    auto copy_tensor_and_set_to_output = [context,
                                          &output_shape](const Tensor& from) {
      Tensor output;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(context->expected_output_dtype(0),
                                            output_shape, &output));
      ITEX_CHECK(output.CopyFrom(from, output_shape));
      context->set_output(0, output);
    };

    // If there is no padding to be done, forward the input to output.
    if (output_shape.num_elements() == in0.NumElements()) {
      // When num_elements == 0, shape may have changed.
      copy_tensor_and_set_to_output(in0);
      return;
    }

    // input is 0, just set output to pad value.
    if (in0.NumElements() == 0) {
      Tensor output;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(context->expected_output_dtype(0),
                                            output_shape, &output));
      auto out = output.flat<T>();
      out.device(context->eigen_device<Device>()) = out.constant(T(pad_value));
      context->set_output(0, output);
      return;
    }

    TensorShape collapsed_input_shape;
    TensorShape collapsed_output_shape;
    std::vector<std::pair<int, int>> collapsed_paddings_pair;
    Tensor collapsed_paddings;
    if (in0.dims() > 1 &&
        CollapseAdjacentNonPaddedDimensions(
            in0.shape(), in1, output_shape, &collapsed_input_shape,
            &collapsed_paddings_pair, &collapsed_output_shape)) {
      // Copy collapsed_paddings to collapsed_paddings_as_tensor.
      AllocatorAttributes alloc_attrs;
      alloc_attrs.set_on_host(true);
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(
              in1.dtype(),
              TensorShape(
                  {static_cast<int64>(collapsed_paddings_pair.size()), 2}),
              &collapsed_paddings, alloc_attrs));
      auto collapsed_paddings_as_matrix = collapsed_paddings.matrix<Tpadding>();
      for (size_t i = 0; i < collapsed_paddings_pair.size(); ++i) {
        collapsed_paddings_as_matrix(i, 0) = collapsed_paddings_pair[i].first;
        collapsed_paddings_as_matrix(i, 1) = collapsed_paddings_pair[i].second;
      }
      Tensor collapsed_input;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(in0.dtype(), collapsed_input_shape,
                                            &collapsed_input));
      ITEX_CHECK(collapsed_input.CopyFrom(in0, collapsed_input_shape));
      Tensor collapsed_output;
      OP_REQUIRES_OK(context, context->allocate_temp(collapsed_input.dtype(),
                                                     collapsed_output_shape,
                                                     &collapsed_output));
      const Tensor& collapsed_paddings_ref = collapsed_paddings;
      typename TTypes<Tpadding>::ConstMatrix collapsed_paddings_matrix =
          collapsed_paddings_ref.matrix<Tpadding>();

      OperateWithVariableRank(context, collapsed_input_shape.dims(),
                              collapsed_input, collapsed_paddings_matrix,
                              pad_value, collapsed_input_shape,
                              collapsed_output_shape, &collapsed_output);

      copy_tensor_and_set_to_output(collapsed_output);
    } else {
      Tensor output;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(context->expected_output_dtype(0),
                                            output_shape, &output));
      OperateWithVariableRank(context, in0.dims(), in0, paddings, pad_value,
                              in0.shape(), output_shape, &output);
      context->set_output(0, output);
    }
  }

 private:
  // Collapses adjacent dimensions that are not padded to one dimension for
  // speed. Returns true if any two dimensions are collapsed. For example,
  //
  //   Pad(input_shape=[8, 28, 28, 3],
  //       paddings=[[0, 0], [0, 0], [0, 0], [0, 1]]
  // is equivalent to
  //   Pad(input_shape=[6272, 3],
  //       paddings=[[0, 0], [0, 1]])
  //
  // input_shape: the original input shape.
  // paddings_as_tensor: the original paddings.
  // output_shape: the original output shape.
  // collapsed_input_shape: the input shape after collapsing.
  // collapsed_paddings_as_tensor: the paddings after collapsing.
  // collapsed_output_shape: the output shape after collapsing.
  bool CollapseAdjacentNonPaddedDimensions(
      const TensorShape& input_shape, const Tensor& paddings_as_tensor,
      const TensorShape& output_shape, TensorShape* collapsed_input_shape,
      std::vector<std::pair<int, int>>* collapsed_paddings,
      TensorShape* collapsed_output_shape) {
    bool collapsed = false;
    typename TTypes<Tpadding>::ConstMatrix paddings =
        paddings_as_tensor.matrix<Tpadding>();
    int i = 0;
    while (i < paddings.dimension(0)) {
      if (paddings(i, 0) != 0 || paddings(i, 1) != 0) {
        // If padded, copy the original dimension over.
        collapsed_input_shape->InsertDim(collapsed_input_shape->dims(),
                                         input_shape.dim_size(i));
        collapsed_output_shape->InsertDim(collapsed_output_shape->dims(),
                                          output_shape.dim_size(i));
        collapsed_paddings->push_back({paddings(i, 0), paddings(i, 1)});
        ++i;
      } else {
        // If not padded, find the next dimension that is padded and collapse
        // all dimensions in between to one dimension.
        int64 collapsed_input_dim_size = input_shape.dim_size(i);
        int64 collapsed_output_dim_size = output_shape.dim_size(i);
        ++i;
        while (i < paddings.dimension(0) && paddings(i, 0) == 0 &&
               paddings(i, 1) == 0) {
          collapsed = true;
          collapsed_input_dim_size *= input_shape.dim_size(i);
          collapsed_output_dim_size *= output_shape.dim_size(i);
          ++i;
        }
        collapsed_input_shape->InsertDim(collapsed_input_shape->dims(),
                                         collapsed_input_dim_size);
        collapsed_output_shape->InsertDim(collapsed_output_shape->dims(),
                                          collapsed_output_dim_size);
        collapsed_paddings->push_back({0, 0});
      }
    }

    return collapsed;
  }

  template <int Dims>
  void Operate(OpKernelContext* context,
               typename TTypes<T, Dims>::ConstTensor input,
               typename TTypes<Tpadding>::ConstMatrix paddings, T pad_value,
               const TensorShape& in_shape, const TensorShape& out_shape,
               Tensor* output) {
    ITEX_CHECK_EQ(Dims, paddings.dimension(0));
    ITEX_CHECK_EQ(2, paddings.dimension(1));
    typename TTypes<T, Dims>::Tensor output_t = output->tensor<T, Dims>();

    std::array<Tpadding, Dims * 2> paddings_array;
    for (int i = 0; i < Dims; ++i) {
      paddings_array[2 * i] = paddings(i, 0);
      paddings_array[2 * i + 1] = paddings(i, 1);
    }

    std::array<int, Dims> out_dims, in_strides, out_strides;
    for (int i = 0; i < Dims; i++) {
      out_dims[i] = out_shape.dim_size(i);
    }
    in_strides[Dims - 1] = 1;
    out_strides[Dims - 1] = 1;
    for (int i = Dims - 2; i >= 0; --i) {
      in_strides[i] = in_strides[i + 1] * in_shape.dim_size(i + 1);
      out_strides[i] = out_strides[i + 1] * out_shape.dim_size(i + 1);
    }

    int nelems = out_shape.num_elements();
    auto* stream = context->GetDeviceStream();
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    bool pad_on_last_dim =
        paddings_array[2 * Dims - 2] || paddings_array[2 * Dims - 1];
    // TODO(itex): maybe use a global variable to represent VecSize, not this
    // magic number 4
    constexpr int VecSize = 4 * sizeof(float) / sizeof(T);

    if (pad_on_last_dim || out_dims[Dims - 1] % VecSize) {
      auto num_wg = (nelems + group_size - 1) / group_size;
      sycl::nd_range<1> thread_range(num_wg * group_size, group_size);
      stream->submit([&](sycl::handler& cgh) {
        PadKernel<T, Tpadding, Dims> task(input.data(), output_t.data(),
                                          paddings_array, out_dims, in_strides,
                                          out_strides, pad_value, nelems);
        cgh.parallel_for<PadKernel<T, Tpadding, Dims>>(thread_range, task);
      });
    } else {
      auto group_elems = VecSize * group_size;
      auto num_wg = (nelems + group_elems - 1) / group_elems;
      sycl::nd_range<1> thread_range(num_wg * group_size, group_size);
      stream->submit([&](sycl::handler& cgh) {
        ContinuousPadKernel<T, Tpadding, Dims, VecSize> task(
            input.data(), output_t.data(), paddings_array, out_dims, in_strides,
            out_strides, pad_value, nelems);
        cgh.parallel_for<ContinuousPadKernel<T, Tpadding, Dims, VecSize>>(
            thread_range, task);
      });
    }
  }

  void OperateWithVariableRank(OpKernelContext* context, int fixed_dims,
                               const Tensor& input,
                               typename TTypes<Tpadding>::ConstMatrix paddings,
                               T pad_value, const TensorShape& in_shape,
                               const TensorShape& out_shape, Tensor* output) {
    // Invoke the dims-specific implementation.
    switch (fixed_dims) {
      case 1:
        // TODO(irving): Once Pad doesn't need a scalar special case,
        // change flat to tensor.  That is, once !allow_legacy_scalars().
        Operate<1>(context, input.flat<T>(), paddings, pad_value, in_shape,
                   out_shape, output);
        break;
      case 2:
        Operate<2>(context, input.tensor<T, 2>(), paddings, pad_value, in_shape,
                   out_shape, output);
        break;
      case 3:
        Operate<3>(context, input.tensor<T, 3>(), paddings, pad_value, in_shape,
                   out_shape, output);
        break;
      case 4:
        Operate<4>(context, input.tensor<T, 4>(), paddings, pad_value, in_shape,
                   out_shape, output);
        break;
      case 5:
        Operate<5>(context, input.tensor<T, 5>(), paddings, pad_value, in_shape,
                   out_shape, output);
        break;
      case 6:
        Operate<6>(context, input.tensor<T, 6>(), paddings, pad_value, in_shape,
                   out_shape, output);
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Only ranks up to 6 supported: ",
                                            input.shape().DebugString()));
    }
  }
};

#define REGISTER_GPU_KERNEL(T)                                      \
  REGISTER_KERNEL_BUILDER(Name("Pad")                               \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int32>("Tpaddings")   \
                              .HostMemory("paddings"),              \
                          PadOp<GPUDevice, T, int32>);              \
  REGISTER_KERNEL_BUILDER(Name("Pad")                               \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int64_t>("Tpaddings") \
                              .HostMemory("paddings"),              \
                          PadOp<GPUDevice, T, int64>);              \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                             \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int32>("Tpaddings")   \
                              .HostMemory("paddings")               \
                              .HostMemory("constant_values"),       \
                          PadOp<GPUDevice, T, int32>)               \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                             \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int64_t>("Tpaddings") \
                              .HostMemory("paddings")               \
                              .HostMemory("constant_values"),       \
                          PadOp<GPUDevice, T, int64>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);

#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_GPU_KERNEL);
#endif  // ITEX_ENABLE_DOUBLE

#undef REGISTER_GPU_KERNEL
}  // namespace itex
