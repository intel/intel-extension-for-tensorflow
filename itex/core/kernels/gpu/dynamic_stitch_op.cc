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

#include "itex/core/kernels/gpu/gpu_device_array.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

template <class T>
class DynamicStitchOpImplBase : public OpKernel {
 public:
  explicit DynamicStitchOpImplBase(OpKernelConstruction* c,
                                   const string& op_name)
      : OpKernel(c) {
    // TODO(itex): cannot get input and output types now
    // Compute expected input signature
    //    const DataType dt = DataTypeToEnum<T>::v();
    //    const int n = c->num_inputs() / 2;
    //    DataTypeVector expected;
    //    for (int i = 0; i < n; i++) {
    //      expected.push_back(DT_INT32);
    //    }
    //    for (int i = 0; i < n; i++) {
    //      expected.push_back(dt);
    //    }
    //    OP_REQUIRES_OK(c, c->MatchSignature(expected, {dt}));
  }

 protected:
  // Check if data0.shape[indices0.dims():] == data1.shape[indices1.dims():]
  static bool SameExtraShape(const Tensor& data0, const Tensor& indices0,
                             const Tensor& data1, const Tensor& indices1) {
    const int extra0 = data0.dims() - indices0.dims();
    const int extra1 = data1.dims() - indices1.dims();
    if (extra0 != extra1) return false;
    for (int i = 0; i < extra0; i++) {
      if (data0.dim_size(indices0.dims() + i) !=
          data1.dim_size(indices1.dims() + i)) {
        return false;
      }
    }
    return true;
  }

  void CheckArgsAndAllocateResult(OpKernelContext* c,
                                  OpInputList* indices_inputs,
                                  OpInputList* data_inputs, int* first_dim_size,
                                  int* data_elements_size,
                                  Tensor** result_ptr) {
    int32 max_index = -1;
    if (data_elements_size) {
      *data_elements_size = 0;
    }
    for (const Tensor& indices : *indices_inputs) {
      if (indices.NumElements() > 0) {
        Eigen::Tensor<int32, 0, Eigen::RowMajor> m =
            indices.flat<int32>().maximum();
        max_index = std::max(m(), max_index);
      }
      if (data_elements_size) {
        *data_elements_size += indices.NumElements();
      }
    }

    *first_dim_size = max_index + 1;

    // Validate that data[i].shape = indices[i].shape + constant
    //    OP_REQUIRES_OK(c, c->input_list("data", data_inputs));
    const Tensor& data0 = (*data_inputs)[0];
    const Tensor& indices0 = (*indices_inputs)[0];
    for (int input_num = 0; input_num < indices_inputs->size(); input_num++) {
      const Tensor& indices = (*indices_inputs)[input_num];
      const Tensor& data = (*data_inputs)[input_num];
      OP_REQUIRES(
          c, TensorShapeUtils::StartsWith(data.shape(), indices.shape()),
          errors::InvalidArgument("data[", input_num,
                                  "].shape = ", data.shape().DebugString(),
                                  " does not start with indices[", input_num,
                                  "].shape = ", indices.shape().DebugString()));
      OP_REQUIRES(
          c, input_num == 0 || SameExtraShape(data0, indices0, data, indices),
          errors::InvalidArgument(
              "Need data[0].shape[", indices0.dims(), ":] = data[", input_num,
              "].shape[", indices.dims(),
              ":], got data[0].shape = ", data0.shape().DebugString(),
              ", data[", input_num, "].shape = ", data.shape().DebugString(),
              ", indices[0].shape = ", indices0.shape().DebugString(),
              ", indices[", input_num,
              "].shape = ", indices.shape().DebugString()));
    }

    // Allocate result tensor of shape
    //   [*first_dim_size] + data.shape[indices.dims:]
    TensorShape result_shape;
    result_shape.AddDim(*first_dim_size);
    for (int d = indices0.dims(); d < data0.dims(); d++) {
      result_shape.AddDim(data0.dim_size(d));
    }
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, result_ptr));
  }
};

namespace {

template <typename T, int Mask>
struct DynamicStitchKernel {
  DynamicStitchKernel(const int32 slice_size, const int32 output_size,
                      GpuDeviceArrayStruct<int32> input_indices,
                      GpuDeviceArrayStruct<const T*> input_ptrs, T* output)
      : slice_size_(slice_size),
        output_size_(output_size),
        input_indices_(input_indices),
        input_ptrs_(input_ptrs),
        output_(output) {}
  void operator()(sycl::nd_item<1> item) const {
    auto idx = item.get_global_linear_id();
    int32* data_indices;
    T* const* data_ptrs;
    GetGpuDeviceArrayOnDeviceWithMask<Mask>(&input_indices_, &data_indices,
                                            &input_ptrs_, &data_ptrs);
    for (; idx < output_size_; idx += item.get_global_range(0)) {
      const int32 slice_id = idx / slice_size_;
      const int32 slice_offset = idx % slice_size_;
      const int32 input_index = data_indices[slice_id];
      if (input_index != -1) {
        output_[idx] = *(data_ptrs[input_index] + slice_offset);
      }
    }
  }

 private:
  const int32 slice_size_;
  const int32 output_size_;
  GpuDeviceArrayStruct<int32> input_indices_;
  GpuDeviceArrayStruct<const T*> input_ptrs_;
  T* output_;
};
}  // namespace

template <int Mask>
struct DispatchDynamicStitchKernel {
  template <typename T>
  void operator()(const Eigen::GpuDevice& d, const int32 slice_size,
                  const int32 first_dim_size, T* output,
                  const GpuDeviceArrayStruct<int>& input_indices,
                  const GpuDeviceArrayStruct<const T*>& input_ptrs) {
    const int32 output_size = first_dim_size * slice_size;
    auto stream = d.stream();
    auto max_wg_size = stream->get_device()
                           .get_info<sycl::info::device::max_work_group_size>();
    auto wg_count = (output_size + max_wg_size - 1) / max_wg_size;
    stream->submit([&](sycl::handler& cgh) {
      sycl::nd_range<1> kernel_range(sycl::range<1>(wg_count * max_wg_size),
                                     sycl::range<1>(max_wg_size));
      DynamicStitchKernel<T, Mask> task(slice_size, output_size, input_indices,
                                        input_ptrs, output);
      cgh.parallel_for<DynamicStitchKernel<T, Mask> >(kernel_range, task);
    });
  }
};

template <typename T>
void DynamicStitchGPUImpl(const Eigen::GpuDevice& d, const int32 slice_size,
                          const int32 first_dim_size,
                          const GpuDeviceArrayStruct<int>& input_indices,
                          const GpuDeviceArrayStruct<const T*>& input_ptrs,
                          T* output) {
  DispatchToGpuDeviceArrayInlined<2, 0, DispatchDynamicStitchKernel>::run(
      input_indices, input_ptrs, d, slice_size, first_dim_size, output);
}

#define REGISTER_GPU(T)                                  \
  template void DynamicStitchGPUImpl(                    \
      const Eigen::GpuDevice& d, const int32 slice_size, \
      const int32 first_dim_size,                        \
      const GpuDeviceArrayStruct<int>& input_indices,    \
      const GpuDeviceArrayStruct<const T*>& input_ptrs, T* output);

TF_CALL_int32(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

template <class T>
class DynamicStitchOpGPU : public DynamicStitchOpImplBase<T> {
 public:
  explicit DynamicStitchOpGPU(OpKernelConstruction* c)
      : DynamicStitchOpImplBase<T>(c, "DynamicStitchOp") {}

  void Compute(OpKernelContext* c) override {
    auto num_inputs = c->num_inputs();
    OP_REQUIRES(
        c, num_inputs > 0,
        errors::InvalidArgument("DynamicStitchOp: Must have some inputs"));
    OP_REQUIRES(c, num_inputs % 2 == 0,
                errors::InvalidArgument(
                    "DynamicStitchOp: Must have even number of arguments"));

    OpInputList indices_inputs(c, 0, num_inputs / 2);
    OpInputList data_inputs(c, num_inputs / 2, c->num_inputs());
    int first_dim_size;
    int data_elements_size;
    Tensor* merged = nullptr;
    this->CheckArgsAndAllocateResult(c, &indices_inputs, &data_inputs,
                                     &first_dim_size, &data_elements_size,
                                     &merged);
    if (!c->status().ok()) {
      // Avoid segmentation faults if merged cannot be allocated and an error is
      // passed back in the context.
      return;
    }

    // Currently we leave uninitialized any portions of
    // merged that aren't covered by an index in indices.
    if (first_dim_size > 0) {
      // because the collision requirements, we have to deal with
      // collision first before send data to gpu kernel.
      const int slice_size = merged->flat_outer_dims<T>().dimension(1);
      GpuDeviceArrayOnHost<int32> indices_flat(c, first_dim_size);
      GpuDeviceArrayOnHost<const T*> data_flat(c, data_elements_size);
      OP_REQUIRES_OK(c, indices_flat.Init());
      OP_REQUIRES_OK(c, data_flat.Init());
      // initialize the indices_flat (-1 represents missing indices)
      for (int i = 0; i < first_dim_size; ++i) {
        indices_flat.Set(i, -1);
      }

      // data_flat index
      int32 idx = 0;
      // sum of indices_inputs[i].NumElements() for compute indices_flat value.
      int32 base_size = 0;
      for (int i = 0; i < indices_inputs.size(); ++i) {
        auto indices_vec = indices_inputs[i].flat<int32>();
        auto data_ptr_base = data_inputs[i].template flat<T>().data();
        for (int j = 0; j < indices_vec.size(); ++j) {
          // indices_flat's indices represent the indices of output.
          // indices_flat's values represent the indices of input_data where the
          // data located.
          indices_flat.Set(indices_vec(j), base_size + j);
          data_flat.Set(
              idx, const_cast<T*>(reinterpret_cast<const T*>(data_ptr_base) +
                                  j * slice_size));
          ++idx;
        }
        base_size += indices_vec.size();
      }
      OP_REQUIRES_OK(c, indices_flat.Finalize());
      OP_REQUIRES_OK(c, data_flat.Finalize());

      auto out_ptr = merged->template flat<T>().data();
      DynamicStitchGPUImpl<T>(c->eigen_gpu_device(), slice_size, first_dim_size,
                              indices_flat.data(), data_flat.data(), out_ptr);
    }
  }
};

#define REGISTER_DYNAMIC_STITCH_GPU(type)                \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOpGPU<type>)

TF_CALL_int32(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_int64(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_complex64(REGISTER_DYNAMIC_STITCH_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_complex128(REGISTER_DYNAMIC_STITCH_GPU);
#endif  // ITEX_ENABLE_DOUBLE
#undef REGISTER_DYNAMIC_STITCH_GPU
}  // namespace itex
