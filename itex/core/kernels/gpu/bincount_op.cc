/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/bincount_op.h"

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"

namespace itex {
constexpr static auto read_write = sycl::access::mode::read_write;
constexpr static auto local_target = sycl::access::target::local;

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Tidx, typename T, bool has_weight, bool binary_count>
struct BincountKernel;

template <typename Tidx, typename T>
struct BincountKernel<Tidx, T, true, false> {
  BincountKernel(int64_t arr_size, const Tidx* arr_ptr, const Tidx num_bins,
                 T* output_ptr, const T* weights_ptr)
      : arr_size(arr_size),
        arr_ptr(arr_ptr),
        num_bins(num_bins),
        output_ptr(output_ptr),
        weights_ptr(weights_ptr) {}
  void operator()() const {
    for (int32 i = 0; i < arr_size; ++i) {
      Tidx value = arr_ptr[i];
      if (value < num_bins) {
        output_ptr[value] += weights_ptr[i];
      }
    }
  }

 private:
  int64_t arr_size;
  const Tidx* arr_ptr;
  const Tidx num_bins;
  T* output_ptr;
  const T* weights_ptr;
};

template <typename Tidx, typename T>
struct BincountKernel<Tidx, T, false, false> {
  BincountKernel(int64_t arr_size, const Tidx* arr_ptr, const Tidx num_bins,
                 T* output_ptr)
      : arr_size(arr_size),
        arr_ptr(arr_ptr),
        num_bins(num_bins),
        output_ptr(output_ptr) {}
  void operator()() const {
    for (int32 i = 0; i < arr_size; ++i) {
      Tidx value = arr_ptr[i];
      if (value < num_bins) {
        output_ptr[value] += T(1);
      }
    }
  }

 private:
  int64_t arr_size;
  const Tidx* arr_ptr;
  const Tidx num_bins;
  T* output_ptr;
};

template <typename Tidx, typename T>
struct BincountKernel<Tidx, T, false, true> {
  BincountKernel(int64_t arr_size, const Tidx* arr_ptr, const Tidx num_bins,
                 T* output_ptr)
      : arr_size(arr_size),
        arr_ptr(arr_ptr),
        num_bins(num_bins),
        output_ptr(output_ptr) {}
  void operator()() const {
    for (int32 i = 0; i < arr_size; ++i) {
      Tidx value = arr_ptr[i];
      if (value < num_bins) {
        output_ptr[value] = true;
      }
    }
  }

 private:
  int64_t arr_size;
  const Tidx* arr_ptr;
  const Tidx num_bins;
  T* output_ptr;
};

template <typename Tidx, typename T>
struct BincountFunctor<GPUDevice, Tidx, T, false> {
  static Status Compute(
      OpKernelContext* context,
      const typename TTypes<Tidx, 1>::ConstTensor& arr,
      const typename TTypes<T, 1>::ConstTensor& weights,
      typename TTypes<T, 1>::Tensor& output,  // NOLINT(runtime/references)
      const Tidx num_bins) {
    if (output.size() == 0) {
      return Status::OK();
    }

    auto* stream = context->GetDeviceStream();
    output.device(context->eigen_gpu_device()) = output.constant(T(0));

    if (arr.size() == 0) return Status::OK();

    if (weights.size()) {
      stream->submit([&](sycl::handler& cgh) {
        auto arr_ptr = arr.data();
        auto weights_ptr = weights.data();
        auto output_ptr = output.data();
        cgh.single_task<BincountKernel<Tidx, T, true, false> >([=]() {
          for (int32 i = 0; i < arr.size(); ++i) {
            Tidx value = arr_ptr[i];
            if (value < num_bins) {
              output_ptr[value] += weights_ptr[i];
            }
          }
        });
      });
    } else {
      stream->submit([&](sycl::handler& cgh) {
        auto arr_ptr = arr.data();
        auto output_ptr = output.data();
        cgh.single_task<BincountKernel<Tidx, T, false, false> >([=]() {
          for (int32 i = 0; i < arr.size(); ++i) {
            Tidx value = arr_ptr[i];
            if (value < num_bins) {
              output_ptr[value] += T(1);
            }
          }
        });
      });
    }
    return Status::OK();
  }
};

template <typename Tidx, typename T>
struct BincountFunctor<GPUDevice, Tidx, T, true> {
  static Status Compute(
      OpKernelContext* context,
      const typename TTypes<Tidx, 1>::ConstTensor& arr,
      const typename TTypes<T, 1>::ConstTensor& weights,
      typename TTypes<T, 1>::Tensor& output,  // NOLINT(runtime/references)
      const Tidx num_bins) {
    if (arr.size() == 0 || output.size() == 0) {
      return Status::OK();
    }

    auto* stream = context->GetDeviceStream();
    output.device(context->eigen_gpu_device()) = output.constant(T(0));
    stream->submit([&](sycl::handler& cgh) {
      auto arr_ptr = arr.data();
      auto output_ptr = output.data();
      cgh.single_task<BincountKernel<Tidx, T, false, true> >([=]() {
        for (int32 i = 0; i < arr.size(); ++i) {
          Tidx value = arr_ptr[i];
          if (value < num_bins) {
            output_ptr[value] = true;
          }
        }
      });
    });

    return Status::OK();
  }
};

}  // namespace functor

template <typename Device, typename T>
class BincountOp : public OpKernel {
 public:
  explicit BincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& arr_t = ctx->input(0);
    const Tensor& size_tensor = ctx->input(1);
    OP_REQUIRES(ctx, size_tensor.dims() == 0,
                errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                        size_tensor.dims()));
    int32 size = size_tensor.scalar<int32>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    const Tensor& weights_t = ctx->input(2);
    const auto arr = arr_t.flat<int32>();
    const auto weights = weights_t.flat<T>();
    Tensor* output_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({size}), &output_t));
    auto output = output_t->flat<T>();
    OP_REQUIRES_OK(ctx,
                   functor::BincountFunctor<Device, int32, T, false>::Compute(
                       ctx, arr, weights, output, size));
  }
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("Bincount")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("size")         \
                              .TypeConstraint<type>("T"), \
                          BincountOp<GPUDevice, type>)

TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_float(REGISTER_KERNELS);
TF_CALL_bfloat16(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <typename Tidx, typename T, bool binary_count, bool has_weight>
struct BincountColReduceKernel;

template <typename Tidx, typename T, bool binary_count>
struct BincountColReduceKernel<Tidx, T, binary_count, false> {
  BincountColReduceKernel(int num_rows, int num_cols, const Tidx* in_ptr,
                          Tidx num_bins, T* out_ptr)
      : num_rows(num_rows),
        num_cols(num_cols),
        in_ptr(in_ptr),
        num_bins(num_bins),
        out_ptr(out_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    const int nthreads = num_rows * num_cols;
    const int index = item.get_global_linear_id();
    if (index >= nthreads) return;

    Tidx bin = in_ptr[index];
    if (bin < num_bins) {
      int row = index / num_cols;
      int offset = row * num_bins + bin;
      if (binary_count) {
        out_ptr[offset] = T(1);
      } else {
        ItexAtomicAdd(out_ptr + offset, T(1));
      }
    }
  }

 private:
  int num_rows;
  int num_cols;
  const Tidx* in_ptr;
  Tidx num_bins;
  T* out_ptr;
};

template <typename Tidx, typename T, bool binary_count>
struct BincountColReduceKernel<Tidx, T, binary_count, true> {
  BincountColReduceKernel(int num_rows, int num_cols, const Tidx* in_ptr,
                          Tidx num_bins, T* out_ptr, const T* weights_ptr)
      : num_rows(num_rows),
        num_cols(num_cols),
        in_ptr(in_ptr),
        num_bins(num_bins),
        out_ptr(out_ptr),
        weights_ptr(weights_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    const int nthreads = num_rows * num_cols;
    const int index = item.get_global_linear_id();
    if (index >= nthreads) return;

    Tidx bin = in_ptr[index];
    if (bin < num_bins) {
      int row = index / num_cols;
      int offset = row * num_bins + bin;
      if (binary_count) {
        out_ptr[offset] = T(1);
      } else {
        T value = weights_ptr[index];
        ItexAtomicAdd(out_ptr + offset, value);
      }
    }
  }

 private:
  int num_rows;
  int num_cols;
  const Tidx* in_ptr;
  Tidx num_bins;
  T* out_ptr;
  const T* weights_ptr;
};

template <typename Tidx, typename T, bool binary_count, bool has_weight>
struct BincountColReduceSharedKernel;

template <typename Tidx, typename T, bool binary_count>
struct BincountColReduceSharedKernel<Tidx, T, binary_count, true> {
  BincountColReduceSharedKernel(
      int num_rows, int num_cols, const Tidx* in_ptr, Tidx num_bins, T* out_ptr,
      const T* weights_ptr,
      sycl::accessor<T, 1, read_write, local_target> local_acc)
      : num_rows(num_rows),
        num_cols(num_cols),
        in_ptr(in_ptr),
        num_bins(num_bins),
        out_ptr(out_ptr),
        weights_ptr(weights_ptr),
        local_acc(local_acc) {}
  void operator()(sycl::nd_item<1> item) const {
    const int out_size = num_rows * num_bins;
    T* shared_col_bins = local_acc.get_pointer();
    for (unsigned int binIdx = item.get_local_id(0); binIdx < out_size;
         binIdx += item.get_local_range(0)) {
      shared_col_bins[binIdx] = T(0);
    }
    sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);

    const int nthreads = num_rows * num_cols;
    const int index = item.get_global_linear_id();
    if (index < nthreads) {
      Tidx bin = in_ptr[index];
      if (bin < num_bins) {
        int row = index / num_cols;
        int offset = row * num_bins + bin;
        if (binary_count) {
          shared_col_bins[offset] = T(1);
        } else {
          T value = weights_ptr[index];
          ItexAtomicAdd<T, T, sycl::memory_order::relaxed,
                        sycl::memory_scope::work_group,
                        sycl::access::address_space::local_space>(
              shared_col_bins + offset, value);
        }
      }
    }
    sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);

    for (unsigned int binIdx = item.get_local_id(0); binIdx < out_size;
         binIdx += item.get_local_range(0)) {
      if (binary_count) {
        if (shared_col_bins[binIdx]) {
          out_ptr[binIdx] = shared_col_bins[binIdx];
        }
      } else {
        ItexAtomicAdd(out_ptr + binIdx, shared_col_bins[binIdx]);
      }
    }
  }

 private:
  int num_rows;
  int num_cols;
  const Tidx* in_ptr;
  Tidx num_bins;
  T* out_ptr;
  const T* weights_ptr;
  sycl::accessor<T, 1, read_write, local_target> local_acc;
};

template <typename Tidx, typename T, bool binary_count>
struct BincountColReduceSharedKernel<Tidx, T, binary_count, false> {
  BincountColReduceSharedKernel(
      int num_rows, int num_cols, const Tidx* in_ptr, Tidx num_bins, T* out_ptr,
      sycl::accessor<T, 1, read_write, local_target> local_acc)
      : num_rows(num_rows),
        num_cols(num_cols),
        in_ptr(in_ptr),
        num_bins(num_bins),
        out_ptr(out_ptr),
        local_acc(local_acc) {}
  void operator()(sycl::nd_item<1> item) const {
    const int out_size = num_rows * num_bins;
    T* shared_col_bins = local_acc.get_pointer();
    for (unsigned int binIdx = item.get_local_id(0); binIdx < out_size;
         binIdx += item.get_local_range(0)) {
      shared_col_bins[binIdx] = T(0);
    }
    sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);

    const int nthreads = num_rows * num_cols;
    const int index = item.get_global_linear_id();
    if (index < nthreads) {
      Tidx bin = in_ptr[index];
      if (bin < num_bins) {
        int row = index / num_cols;
        int offset = row * num_bins + bin;
        if (binary_count) {
          shared_col_bins[offset] = T(1);
        } else {
          ItexAtomicAdd<T, T, sycl::memory_order::relaxed,
                        sycl::memory_scope::work_group,
                        sycl::access::address_space::local_space>(
              shared_col_bins + offset, T(1));
        }
      }
    }
    sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);

    for (unsigned int binIdx = item.get_local_id(0); binIdx < out_size;
         binIdx += item.get_local_range(0)) {
      if (binary_count) {
        if (shared_col_bins[binIdx]) {
          out_ptr[binIdx] = shared_col_bins[binIdx];
        }
      } else {
        ItexAtomicAdd(out_ptr + binIdx, shared_col_bins[binIdx]);
      }
    }
  }

 private:
  int num_rows;
  int num_cols;
  const Tidx* in_ptr;
  Tidx num_bins;
  T* out_ptr;
  sycl::accessor<T, 1, read_write, local_target> local_acc;
};

template <typename Tidx, typename T, bool binary_count>
struct functor::BincountReduceFunctor<GPUDevice, Tidx, T, binary_count> {
  static Status Compute(
      OpKernelContext* context, const typename TTypes<Tidx, 2>::ConstTensor& in,
      const typename TTypes<T, 2>::ConstTensor& weights,
      typename TTypes<T, 2>::Tensor& out,  // NOLINT(runtime/references)
      const Tidx num_bins) {
    const int num_rows = in.dimension(0);
    const int num_cols = in.dimension(1);

    auto* stream = context->GetDeviceStream();

    // Use half of maximum shared memory, about 32KB.
    int smem_max =
        stream->get_device()
            .template get_info<sycl::info::device::local_mem_size>() /
        2;
    int smem_usage = out.size() * sizeof(T);
    auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    auto num_workgroup = (num_rows * num_cols + group_size - 1) / group_size;

    if (smem_usage < smem_max) {
      if (weights.size()) {
        stream->submit([&](sycl::handler& cgh) {
          auto in_ptr = in.data();
          auto weights_ptr = weights.data();
          auto out_ptr = out.data();

          sycl::accessor<T, 1, read_write, local_target> local_acc(
              sycl::range<1>(smem_usage), cgh);
          BincountColReduceSharedKernel<Tidx, T, binary_count, true>
              kernel_functor(num_rows, num_cols, in_ptr, num_bins, out_ptr,
                             weights_ptr, local_acc);
          cgh.parallel_for<
              BincountColReduceSharedKernel<Tidx, T, binary_count, true> >(
              sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                                sycl::range<1>(group_size)),
              kernel_functor);
        });
      } else {
        stream->submit([&](sycl::handler& cgh) {
          auto in_ptr = in.data();
          auto out_ptr = out.data();

          sycl::accessor<T, 1, read_write, local_target> local_acc(
              sycl::range<1>(smem_usage), cgh);
          BincountColReduceSharedKernel<Tidx, T, binary_count, false>
              kernel_functor(num_rows, num_cols, in_ptr, num_bins, out_ptr,
                             local_acc);
          cgh.parallel_for<
              BincountColReduceSharedKernel<Tidx, T, binary_count, false> >(
              sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                                sycl::range<1>(group_size)),
              kernel_functor);
        });
      }
      return Status::OK();
    }

    if (weights.size()) {
      stream->submit([&](sycl::handler& cgh) {
        auto in_ptr = in.data();
        auto weights_ptr = weights.data();
        auto out_ptr = out.data();
        BincountColReduceKernel<Tidx, T, binary_count, true> kernel_functor(
            num_rows, num_cols, in_ptr, num_bins, out_ptr, weights_ptr);
        cgh.parallel_for<BincountColReduceKernel<Tidx, T, binary_count, true> >(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            kernel_functor);
      });
    } else {
      stream->submit([&](sycl::handler& cgh) {
        auto in_ptr = in.data();
        auto out_ptr = out.data();
        BincountColReduceKernel<Tidx, T, binary_count, false> kernel_functor(
            num_rows, num_cols, in_ptr, num_bins, out_ptr);
        cgh.parallel_for<
            BincountColReduceKernel<Tidx, T, binary_count, false> >(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            kernel_functor);
      });
    }
    return Status::OK();
  }
};
template <typename Device, typename Tidx, typename T>
class DenseBincountOp : public OpKernel {
 public:
  explicit DenseBincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("binary_output", &binary_output_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    OP_REQUIRES(ctx, data.dims() <= 2,
                errors::InvalidArgument(
                    "Shape must be at most rank 2 but is rank ", data.dims()));

    const Tensor& size_t = ctx->input(1);
    const Tensor& weights = ctx->input(2);

    Tidx size = size_t.scalar<Tidx>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    Tensor* out_t;
    functor::SetZeroFunctor<Device, T> fill;
    if (data.dims() == 1) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({size}), &out_t));
      auto out = out_t->flat<T>();
      fill(ctx->eigen_device<Device>(), out);
      if (binary_output_) {
        OP_REQUIRES_OK(
            ctx, functor::BincountFunctor<Device, Tidx, T, true>::Compute(
                     ctx, data.flat<Tidx>(), weights.flat<T>(), out, size));
      } else {
        OP_REQUIRES_OK(
            ctx, functor::BincountFunctor<Device, Tidx, T, false>::Compute(
                     ctx, data.flat<Tidx>(), weights.flat<T>(), out, size));
      }
    } else if (data.dims() == 2) {
      const int64 num_rows = data.dim_size(0);
      auto weight_matrix =
          (weights.NumElements() == 0)
              ? weights.shaped<T, 2>(gtl::InlinedVector<int64, 2>(2, 0))
              : weights.matrix<T>();
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, TensorShape({num_rows, size}), &out_t));
      auto out = out_t->matrix<T>();
      fill(ctx->eigen_device<Device>(), out_t->flat<T>());
      if (binary_output_) {
        OP_REQUIRES_OK(
            ctx, functor::BincountReduceFunctor<Device, Tidx, T, true>::Compute(
                     ctx, data.matrix<Tidx>(), weight_matrix, out, size));
      } else {
        OP_REQUIRES_OK(
            ctx,
            functor::BincountReduceFunctor<Device, Tidx, T, false>::Compute(
                ctx, data.matrix<Tidx>(), weight_matrix, out, size));
      }
    }
  }

 private:
  bool binary_output_;
};

#define REGISTER_KERNELS(Tidx, T)                            \
  REGISTER_KERNEL_BUILDER(Name("DenseBincount")              \
                              .Device(DEVICE_GPU)            \
                              .HostMemory("size")            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<Tidx>("Tidx"), \
                          DenseBincountOp<GPUDevice, Tidx, T>);
#define REGISTER_GPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64, T);

TF_CALL_int32(REGISTER_GPU_KERNELS);
TF_CALL_float(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

}  // end namespace itex
