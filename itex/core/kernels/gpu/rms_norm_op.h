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

#ifndef ITEX_CORE_KERNELS_GPU_RMS_NORM_OP_H_
#define ITEX_CORE_KERNELS_GPU_RMS_NORM_OP_H_

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

struct Params {
  int rows;
  int cols;
  void* input;
  void* output;
  void* gamma;
  void* beta;
  float epsilon;
};

template <typename Elt_type, int NUM_ELTS>
struct Vec {
  using Vec_type = typename BaseTypeVectorize<Elt_type, NUM_ELTS>::type;
  using Scalar_type = typename BaseTypeVectorize<Elt_type, NUM_ELTS>::scalar;

  Vec_type data;

  template <typename S>
  inline void to(Vec<S, NUM_ELTS>& other) {  // NOLINT(*)
#pragma unroll
    for (int it = 0; it < NUM_ELTS; it++) {
      other.data[it] = typename Vec<S, NUM_ELTS>::Scalar_type(this->data[it]);
    }
  }

  inline void load_from(const void* base_ptr, int idx = 0) {
    this->data = reinterpret_cast<const Vec_type*>(base_ptr)[idx];
  }

  // Pointer is cast to vector type
  inline void store_to(void* base_ptr, int idx = 0) const {
    reinterpret_cast<Vec_type*>(base_ptr)[idx] = this->data;
  }

  // Pointer is cast to element type. Loads min(count, NUM_ELT)
  // elements and any remaining elements are set to zero.
  inline void load_from_elts(const void* base_ptr, int idx = 0,
                             int count = NUM_ELTS) {
    const Scalar_type* elt_ptr =
        reinterpret_cast<const Scalar_type*>(base_ptr) + idx;
    if (count < NUM_ELTS || idx % NUM_ELTS != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELTS; it++) {
        this->data[it] = (it < count ? elt_ptr[it] : Scalar_type(0.f));
      }
    } else {
      this->load_from(elt_ptr);
    }
  }

  // Pointer is cast to element type. Stores min(count, NUM_ELT)
  // elements.
  inline void store_to_elts(void* base_ptr, int idx = 0,
                            int count = NUM_ELTS) const {
    Scalar_type* elt_ptr = static_cast<Scalar_type*>(base_ptr) + idx;
    if (count < NUM_ELTS || idx % NUM_ELTS != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELTS; it++) {
        if (it < count) {
          elt_ptr[it] = this->data[it];
        }
      }
    } else {
      this->store_to(elt_ptr);
    }
  }
};

template <typename T, typename Smem, int SUBGROUPS_M, int SUBGROUPS_N,
          int SUBGROUP_SIZE>
struct Reducer {
  enum { SMEM_SIZE = SUBGROUPS_M * SUBGROUPS_N };

  inline Reducer(sycl::group<1>& g, sycl::sub_group& sg,  // NOLINT
                 int sg_m, int sg_n, int lane, const Smem& scratch)
      : g_(g),
        sg_(sg),
        sg_m_(sg_m),
        sg_n_(sg_n),
        lane_(lane),
        scratch_(scratch) {}

  template <typename Op>
  inline T allreduce(T data, const Op& op) {
    data = reduce(data, op);
    if (this->lane_ == 0) {
      scratch_[this->sg_m_ * SUBGROUPS_N + this->sg_n_] = data;
    }
    sycl::group_barrier(this->g_);
    T out = T(0);
#pragma unroll
    for (int it = 0; it < SUBGROUPS_N; it++) {
      out = op(out, scratch_[this->sg_m_ * SUBGROUPS_N + it]);
    }
    return out;
  }

  template <typename Op>
  inline T reduce(T data, const Op& op) {
// only lane 0 holds the result!
#pragma unroll
    for (int it = SUBGROUP_SIZE / 2; it > 0; it /= 2) {
      data = op(data, sycl::shift_group_left(sg_, data, it));
    }
    return data;
  }

  sycl::group<1>& g_;
  sycl::sub_group& sg_;
  int sg_m_;
  int sg_n_;
  int lane_;
  const Smem& scratch_;
};

// For a input tensor with shape [rows, cols] and a weight tensor with shape
// [cols], RMSNormKernel use SUBGROUPS_N subgroups to norm one row, and each
// group will norm SUBGROUPS_M rows.
template <typename T, typename U, int SUBGROUPS_M, int SUBGROUPS_N,
          int BYTES_PER_LOAD, int HIDDEN_SIZE, bool use_scale, bool use_center>
struct RMSNormKernel {
  static constexpr int SUBGROUP_SIZE = 32;
  static constexpr int WORKITEMS_PER_ROW = SUBGROUPS_N * SUBGROUP_SIZE;
  static constexpr int NUM_ELTS = BYTES_PER_LOAD / sizeof(U);
  static constexpr int LOADS = HIDDEN_SIZE / (WORKITEMS_PER_ROW * NUM_ELTS);
  static constexpr int WORKITEMS_PER_GROUP = SUBGROUPS_M * WORKITEMS_PER_ROW;
  static constexpr int COLS_PER_GROUP = NUM_ELTS * WORKITEMS_PER_ROW;
  static constexpr int SMEM_SIZE = SUBGROUPS_M * SUBGROUPS_N;

  using index_t = int;
  using Tvec = Vec<T, NUM_ELTS>;
  using Uvec = Vec<U, NUM_ELTS>;
  using Smem = sycl::local_accessor<U, 1>;
  using Reducer = Reducer<U, Smem, SUBGROUPS_M, SUBGROUPS_N, SUBGROUP_SIZE>;

  RMSNormKernel(Params params, Smem scratch)
      : params_(params), scratch_(scratch) {}

  [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    auto g = item.get_group();
    auto sg = item.get_sub_group();
    const index_t wkitem_id = item.get_local_linear_id();
    const index_t group_id = item.get_group(0);
    const index_t lane = sg.get_local_id();
    const index_t sg_id = wkitem_id / SUBGROUP_SIZE;
    const index_t sg_m = sg_id / SUBGROUPS_N;
    const index_t sg_n = sg_id % SUBGROUPS_N;

    const index_t group_row = group_id * SUBGROUPS_M;
    const index_t vec_col = sg_n * SUBGROUP_SIZE + lane;

    Reducer reducer(g, sg, sg_m, sg_n, lane, scratch_);

    // Load weights
    Uvec gamma[LOADS];
    Uvec beta[LOADS];
    if constexpr (use_scale || use_center) {
#pragma unroll
      for (int it = 0, col = vec_col * NUM_ELTS;
           it < LOADS && col < params_.cols; ++it, col += COLS_PER_GROUP) {
        if constexpr (use_scale)
          gamma[it].load_from_elts(params_.gamma, col, params_.cols - col);
        if constexpr (use_center)
          beta[it].load_from_elts(params_.beta, col, params_.cols - col);
      }
    }

    // Load input
    const int row = group_row + sg_m;
    Uvec Uinput[LOADS];
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && row < params_.rows && col < params_.cols;
         it++, col += COLS_PER_GROUP) {
      Tvec Tinput;
      Tinput.load_from_elts(params_.input, row * params_.cols + col,
                            params_.cols - col);
      Tinput.to(Uinput[it]);
    }

    // Compute Root Mean Square
    U square_sum = 0.f;
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && row < params_.rows && col < params_.cols;
         it++, col += COLS_PER_GROUP) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        U temp = Uinput[it].data[jt];
        square_sum += temp * temp;
      }
    }
    square_sum = reducer.allreduce(square_sum, sycl::plus<U>());
    U rms = sycl::rsqrt(square_sum / params_.cols + U(params_.epsilon));

    // Compute output
#pragma unroll
    for (int it = 0, col = vec_col * NUM_ELTS;
         it < LOADS && row < params_.rows && col < params_.cols;
         it++, col += COLS_PER_GROUP) {
      // Compute output values
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        Uinput[it].data[jt] *= rms;
        if constexpr (use_scale) Uinput[it].data[jt] *= gamma[it].data[jt];
        if constexpr (use_center) Uinput[it].data[jt] += beta[it].data[jt];
      }

      // Store output
      Tvec Toutput;
      Uinput[it].to(Toutput);
      Toutput.store_to_elts(params_.output, row * params_.cols + col,
                            params_.cols - col);
    }
  }

 private:
  Params params_;
  Smem scratch_;
};

template <typename T, typename U, int SUBGROUPS_M, int SUBGROUPS_N,
          int HIDDEN_SIZE, int BYTES_PER_LOAD>
void launch_rms_norm(OpKernelContext* context, const Params& params,
                     bool use_scale, bool use_center) {
  using kernel_type = RMSNormKernel<T, U, SUBGROUPS_M, SUBGROUPS_N,
                                    BYTES_PER_LOAD, HIDDEN_SIZE, false, false>;

  // Set kernel launch parameter.
  constexpr int workitems_per_group = kernel_type::WORKITEMS_PER_GROUP;
  auto num_wg = DivUp(params.rows, SUBGROUPS_M);
  auto stream = context->GetDeviceStream();
  if (use_scale && use_center) {
    stream->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<U, 1> scratch(kernel_type::SMEM_SIZE, cgh);
      RMSNormKernel<T, U, SUBGROUPS_M, SUBGROUPS_N, BYTES_PER_LOAD, HIDDEN_SIZE,
                    true, true>
          task(params, scratch);
      cgh.parallel_for<RMSNormKernel<T, U, SUBGROUPS_M, SUBGROUPS_N,
                                     BYTES_PER_LOAD, HIDDEN_SIZE, true, true>>(
          sycl::nd_range<1>(num_wg * workitems_per_group, workitems_per_group),
          task);
    });
  } else if (use_scale && !use_center) {
    stream->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<U, 1> scratch(kernel_type::SMEM_SIZE, cgh);
      RMSNormKernel<T, U, SUBGROUPS_M, SUBGROUPS_N, BYTES_PER_LOAD, HIDDEN_SIZE,
                    true, false>
          task(params, scratch);
      cgh.parallel_for<RMSNormKernel<T, U, SUBGROUPS_M, SUBGROUPS_N,
                                     BYTES_PER_LOAD, HIDDEN_SIZE, true, false>>(
          sycl::nd_range<1>(num_wg * workitems_per_group, workitems_per_group),
          task);
    });
  } else if (!use_scale && use_center) {
    stream->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<U, 1> scratch(kernel_type::SMEM_SIZE, cgh);
      RMSNormKernel<T, U, SUBGROUPS_M, SUBGROUPS_N, BYTES_PER_LOAD, HIDDEN_SIZE,
                    false, true>
          task(params, scratch);
      cgh.parallel_for<RMSNormKernel<T, U, SUBGROUPS_M, SUBGROUPS_N,
                                     BYTES_PER_LOAD, HIDDEN_SIZE, false, true>>(
          sycl::nd_range<1>(num_wg * workitems_per_group, workitems_per_group),
          task);
    });
  } else {
    stream->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<U, 1> scratch(kernel_type::SMEM_SIZE, cgh);
      RMSNormKernel<T, U, SUBGROUPS_M, SUBGROUPS_N, BYTES_PER_LOAD, HIDDEN_SIZE,
                    false, false>
          task(params, scratch);
      cgh.parallel_for<
          RMSNormKernel<T, U, SUBGROUPS_M, SUBGROUPS_N, BYTES_PER_LOAD,
                        HIDDEN_SIZE, false, false>>(
          sycl::nd_range<1>(num_wg * workitems_per_group, workitems_per_group),
          task);
    });
  }
}

namespace functor {

template <typename Device, typename T, typename U>
struct RMSNormFunctor {
  void operator()(OpKernelContext* context, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::Flat output,
                  typename TTypes<U>::ConstVec gamma,
                  typename TTypes<U>::ConstVec beta, float epsilon,
                  bool use_scale, bool use_center, int rows, int cols);
};

}  // end namespace functor
}  // end namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_RMS_NORM_OP_H_
