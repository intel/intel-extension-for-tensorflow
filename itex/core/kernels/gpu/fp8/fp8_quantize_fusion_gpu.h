/* Copyright (c) 2023 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_GPU_FP8_FP8_QUANTIZE_FUSION_GPU_H_
#define ITEX_CORE_KERNELS_GPU_FP8_FP8_QUANTIZE_FUSION_GPU_H_

#include "itex/core/kernels/gpu/fp8/utils.h"
#include "itex/core/utils/gpu_device_functions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {

constexpr int load_bytes = 8;
constexpr int store_bytes = 8;
constexpr int subgroup_size = 32;
constexpr int sg_per_workgroup = 32;
constexpr int partial_reduce_rows = 4;
constexpr int workgroup_size = 1024;

template <typename Tcomp, typename Tin>
Tcomp dgelu(const Tin val) {
  Tcomp cval = Tcomp(val);
  const Tcomp tanh_out =
      tanhf(0.79788456f * cval * (1.f + 0.044715f * cval * cval));
  return 0.5f * cval *
             ((1.f - tanh_out * tanh_out) *
              (0.79788456f + 0.1070322243f * cval * cval)) +
         0.5f * (1.f + tanh_out);
}

template <int nvec, int partial_reduce_rows, typename input_t,
          typename quantize_t, typename compute_t>
struct Fp8QuantizeDbiasFusedKernel {
  using Ivec = Vec<input_t, nvec>;
  using Cvec = Vec<compute_t, nvec>;
  using Qvec = Vec<quantize_t, nvec>;
  using BiasLocalMem = sycl::local_accessor<Cvec, 1>;
  Fp8QuantizeDbiasFusedKernel(const void* grad, void* quantize_grad,
                              void* workspace, float* amax_ptr,
                              const float* scale_ptr,
                              BiasLocalMem dbias_scratch, int row, int col)
      : grad_(grad),
        quantize_grad_(quantize_grad),
        workspace_(workspace),
        amax_(amax_ptr),
        scale_(scale_ptr),
        dbias_scratch_(dbias_scratch),
        row_(row),
        col_(col) {}

  [[intel::reqd_sub_group_size(subgroup_size)]] void operator()(
      sycl::nd_item<2> item) const {
    auto g = item.get_group();
    int gid_y = item.get_group(0);
    int gid_x = item.get_group(1);
    int local_id = item.get_local_linear_id();
    sycl::sub_group sg = item.get_sub_group();
    int sg_id = local_id / sg_per_workgroup;
    int id_in_sg = sg.get_local_id();

    int group_row = gid_y * partial_reduce_rows * sg_per_workgroup;
    int group_col = gid_x * nvec * subgroup_size;
    int sg_row = group_row + sg_id * partial_reduce_rows;
    int thread_col = group_col + id_in_sg * nvec;

    Ivec in[partial_reduce_rows];  // NOLINT
#pragma unroll
    for (int i = 0; i < partial_reduce_rows; ++i) {
      if (sg_row + i < row_) {
        in[i].load_from_elts(grad_, (sg_row + i) * col_ + thread_col,
                             col_ - thread_col);
      }
    }
    Cvec partial_dbias;
    partial_dbias.clear();
    float scale = *scale_, max = 0;
#pragma unroll
    for (int i = 0; i < partial_reduce_rows; ++i) {
      Qvec quantize_grad;
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        float tmp = static_cast<float>(in[i].data[j]);
        quantize_t quantize_tmp = quantize_t(scale * tmp);
        quantize_grad.data[j] = quantize_tmp;
        partial_dbias.data[j] += tmp;
        max = sycl::fmax(sycl::fabs(tmp), max);
      }
      // Store quantize result.
      if (sg_row + i < row_) {
        quantize_grad.store_to_elts(quantize_grad_,
                                    (sg_row + i) * col_ + thread_col,
                                    col_ - thread_col);
      }
    }

    // partial dbais
    dbias_scratch_[local_id] = partial_dbias;
    sycl::group_barrier(g);
    if (sg_id == 0) {
#pragma unroll
      for (int i = 1; i < sg_per_workgroup; ++i) {
        Cvec tmp = dbias_scratch_[local_id + i * subgroup_size];
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          partial_dbias.data[j] += tmp.data[j];
        }
      }
      partial_dbias.store_to_elts(workspace_, gid_y * col_ + thread_col,
                                  col_ - thread_col);
    }
    // update meta data
    auto group_max =
        sycl::reduce_over_group(item.get_group(), max, sycl::maximum<>());
    if (local_id == 0) {
      ItexAtomicMax(amax_, group_max);
    }
  }

 private:
  const void* grad_;
  void* quantize_grad_;
  void* workspace_;
  float* amax_;
  const float* scale_;
  BiasLocalMem dbias_scratch_;
  int row_;
  int col_;
};

template <int nvec, int partial_reduce_rows, typename input_t,
          typename quantize_t, typename compute_t>
struct Fp8QuantizeDbiasDgeluFusedKernel {
  using Ivec = Vec<input_t, nvec>;
  using Cvec = Vec<compute_t, nvec>;
  using Qvec = Vec<quantize_t, nvec>;
  using BiasLocalMem = sycl::local_accessor<Cvec, 1>;
  Fp8QuantizeDbiasDgeluFusedKernel(const void* grad, const void* gelu_inp,
                                   void* dgelu, void* workspace,
                                   float* amax_ptr, const float* scale_ptr,
                                   BiasLocalMem dbias_scratch, int row, int col)
      : grad_(grad),
        gelu_inp_(gelu_inp),
        dgelu_(dgelu),
        workspace_(workspace),
        amax_(amax_ptr),
        scale_(scale_ptr),
        dbias_scratch_(dbias_scratch),
        row_(row),
        col_(col) {}

  [[intel::reqd_sub_group_size(subgroup_size)]] void operator()(
      sycl::nd_item<2> item) const {
    auto g = item.get_group();
    int gid_y = item.get_group(0);
    int gid_x = item.get_group(1);
    int local_id = item.get_local_linear_id();
    sycl::sub_group sg = item.get_sub_group();
    int sg_id = local_id / sg_per_workgroup;
    int id_in_sg = sg.get_local_id();

    int group_row = gid_y * partial_reduce_rows * sg_per_workgroup;
    int group_col = gid_x * nvec * subgroup_size;
    int sg_row = group_row + sg_id * partial_reduce_rows;
    int thread_col = group_col + id_in_sg * nvec;

    Ivec grad_vec[partial_reduce_rows];      // NOLINT
    Ivec gelu_inp_vec[partial_reduce_rows];  // NOLINT
#pragma unroll
    for (int i = 0; i < partial_reduce_rows; ++i) {
      if (sg_row + i < row_) {
        grad_vec[i].load_from_elts(grad_, (sg_row + i) * col_ + thread_col,
                                   col_ - thread_col);
        gelu_inp_vec[i].load_from_elts(
            gelu_inp_, (sg_row + i) * col_ + thread_col, col_ - thread_col);
      }
    }
    Cvec partial_dbias;
    partial_dbias.clear();
    Cvec after_dgelu[partial_reduce_rows];  // NOLINT
#pragma unroll
    for (int i = 0; i < partial_reduce_rows; ++i) {
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        after_dgelu[i].data[j] =
            dgelu<compute_t>(compute_t(gelu_inp_vec[i].data[j])) *
            compute_t(grad_vec[i].data[j]);
      }
    }
    float scale = *scale_, max = 0;
#pragma unroll
    for (int i = 0; i < partial_reduce_rows; ++i) {
      Qvec dgelu;
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        float tmp = static_cast<float>(after_dgelu[i].data[j]);
        quantize_t dgelu_tmp = quantize_t(scale * tmp);
        dgelu.data[j] = dgelu_tmp;
        partial_dbias.data[j] += tmp;
        max = sycl::fmax(sycl::fabs(tmp), max);
      }
      // Store dgelu.
      if (sg_row + i < row_) {
        dgelu.store_to_elts(dgelu_, (sg_row + i) * col_ + thread_col,
                            col_ - thread_col);
      }
    }
    // partial dbais
    dbias_scratch_[local_id] = partial_dbias;
    sycl::group_barrier(g);
    if (sg_id == 0) {
#pragma unroll
      for (int i = 1; i < sg_per_workgroup; ++i) {
        Cvec tmp = dbias_scratch_[local_id + i * subgroup_size];
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          partial_dbias.data[j] += tmp.data[j];
        }
      }
      partial_dbias.store_to_elts(workspace_, gid_y * col_ + thread_col,
                                  col_ - thread_col);
    }
    // update meta data
    auto group_max =
        sycl::reduce_over_group(item.get_group(), max, sycl::maximum<>());
    if (local_id == 0) {
      ItexAtomicMax(amax_, group_max);
    }
  }

 private:
  const void* grad_;
  const void* gelu_inp_;
  void* dgelu_;
  void* workspace_;
  float* amax_;
  const float* scale_;
  BiasLocalMem dbias_scratch_;
  int row_;
  int col_;
};

template <int nvec, typename compute_t, typename output_t>
struct ReduceDbiasKernel {
  using Cvec = Vec<compute_t, nvec>;
  using Ovec = Vec<output_t, nvec>;
  using Oscalar = typename Ovec::Scalar_type;
  ReduceDbiasKernel(void* workspace, void* dbias_out, int row, int col)
      : workspace_(workspace), dbias_out_(dbias_out), row_(row), col_(col) {}
  void operator()(sycl::nd_item<1> item) const {
    int id = item.get_global_linear_id();
    if (id * nvec >= col_) {
      return;
    }

    Cvec acc_vec;
    acc_vec.clear();
    Cvec ldg_vec;
    for (int i = 0; i < row_; ++i) {
      ldg_vec.load_from_elts(workspace_, i * col_ + id * nvec,
                             col_ - id * nvec);
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        acc_vec.data[j] += ldg_vec.data[j];
      }
    }

    Ovec out_vec;
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      out_vec.data[i] = Oscalar(acc_vec.data[i]);
    }
    out_vec.store_to_elts(dbias_out_, id * nvec, col_ - id * nvec);
  }

 private:
  void* workspace_;
  void* dbias_out_;
  int row_;
  int col_;
};

namespace functor {

template <typename compute_t, typename output_t>
void DbiasReduction(OpKernelContext* context, void* workspace, void* dbias_out,
                    int row, int col) {
  constexpr int reduce_dbias_store_bytes = 8;
  constexpr int reduce_dbias_nvec = reduce_dbias_store_bytes / sizeof(output_t);
  int num_workgroup = DivUp(col, reduce_dbias_nvec * workgroup_size);
  auto* stream = context->GetDeviceStream();
  stream->submit([&](sycl::handler& cgh) {
    ReduceDbiasKernel<reduce_dbias_nvec, compute_t, output_t> task(
        workspace, dbias_out, row, col);
    cgh.parallel_for<ReduceDbiasKernel<reduce_dbias_nvec, compute_t, output_t>>(
        sycl::nd_range<1>(num_workgroup * workgroup_size, workgroup_size),
        task);
  });
}

template <typename input_t, typename quantize_t>
void Fp8QuantizeDbiasFused(OpKernelContext* context, const void* grad,
                           void* quantize_grad, void* dbias, float* amax,
                           const float* scale, int row, int col) {
  constexpr int nvec = load_bytes / sizeof(input_t);
  int sg_rows = partial_reduce_rows * sg_per_workgroup;
  int sg_cols = nvec * subgroup_size;
  int sg_m = DivUp(row, sg_rows);
  int sg_n = DivUp(col, sg_cols);
  Tensor dbias_workspace;
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::v(),
                                                 TensorShape({sg_m, col}),
                                                 &dbias_workspace));
  using kernel_type = Fp8QuantizeDbiasFusedKernel<nvec, partial_reduce_rows,
                                                  input_t, quantize_t, float>;
  auto* stream = context->GetDeviceStream();
  stream->submit([&](sycl::handler& cgh) {
    using Cvec = typename kernel_type::Cvec;
    sycl::local_accessor<Cvec, 1> dbias_scratch(
        sg_per_workgroup * subgroup_size, cgh);
    kernel_type task(grad, quantize_grad, dbias_workspace.flat<float>().data(),
                     amax, scale, dbias_scratch, row, col);
    cgh.parallel_for<kernel_type>(
        sycl::nd_range<2>(
            sycl::range<2>(sg_m * sg_per_workgroup, sg_n * subgroup_size),
            sycl::range<2>(sg_per_workgroup, subgroup_size)),
        task);
  });

  DbiasReduction<float, input_t>(context, dbias_workspace.flat<float>().data(),
                                 dbias, sg_m, col);
}

template <typename input_t, typename quantize_t>
void Fp8QuantizeDbiasDgeluFused(OpKernelContext* context, const void* grad,
                                const void* gelu_inp, void* dbias, void* dgelu,
                                float* amax, const float* scale, int row,
                                int col) {
  constexpr int nvec = load_bytes / sizeof(input_t);
  int sg_rows = partial_reduce_rows * sg_per_workgroup;
  int sg_cols = nvec * subgroup_size;
  int sg_m = DivUp(row, sg_rows);
  int sg_n = DivUp(col, sg_cols);

  Tensor dbias_workspace;
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::v(),
                                                 TensorShape({sg_m, col}),
                                                 &dbias_workspace));
  using kernel_type =
      Fp8QuantizeDbiasDgeluFusedKernel<nvec, partial_reduce_rows, input_t,
                                       quantize_t, float>;
  auto* stream = context->GetDeviceStream();
  stream->submit([&](sycl::handler& cgh) {
    using Cvec = typename kernel_type::Cvec;
    sycl::local_accessor<Cvec, 1> dbias_scratch(
        sg_per_workgroup * subgroup_size, cgh);
    kernel_type task(grad, gelu_inp, dgelu,
                     dbias_workspace.flat<float>().data(), amax, scale,
                     dbias_scratch, row, col);
    cgh.parallel_for<kernel_type>(
        sycl::nd_range<2>(
            sycl::range<2>(sg_m * sg_per_workgroup, sg_n * subgroup_size),
            sycl::range<2>(sg_per_workgroup, subgroup_size)),
        task);
  });

  DbiasReduction<float, input_t>(context, dbias_workspace.flat<float>().data(),
                                 dbias, sg_m, col);
}

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_FP8_FP8_QUANTIZE_FUSION_GPU_H_
