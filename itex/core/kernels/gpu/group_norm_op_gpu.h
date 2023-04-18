/* Copyright (c) 2021-2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_KERNELS_GPU_GROUP_NORM_OP_GPU_H_
#define ITEX_CORE_KERNELS_GPU_GROUP_NORM_OP_GPU_H_
#include <algorithm>

#include "itex/core/kernels/gpu/col_reduction_kernels.h"
#include "itex/core/kernels/gpu/group_norm_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace impl {

template <typename T>
using LocalAcc = sycl::local_accessor<T, 1>;

// --------------------// GroupMeanVar //-------------------- //

template <int SUB_GROUP_SIZE, typename T>
void GroupMeanVar(const sycl::nd_item<2>& item, T* par_sum, T* par_sqr,
                  int total, T* lmem) {
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_id();
  int sg_local_id = sg.get_local_id();
  int num_sg = sg.get_group_linear_range();

  // compute sum of each sub group
  T sum = *par_sum;
  T sqr = *par_sqr;
#pragma unroll
  for (int s = SUB_GROUP_SIZE >> 1; s > 0; s >>= 1) {
    sum += sycl::shift_group_left(sg, sum, s);
    sqr += sycl::shift_group_left(sg, sqr, s);
  }

  if (sg_local_id == 0) {
    lmem[sg_id] = sum;
    lmem[sg_id + num_sg] = sqr;
  }
  item.barrier(sycl::access::fence_space::local_space);

  // compute total sum and mean by one sub group
  if (sg_id == 0) {
    sum = 0;
    sqr = 0;
    for (int i = sg_local_id; i < num_sg; i += SUB_GROUP_SIZE) {
      sum += lmem[i];
      sqr += lmem[i + num_sg];
    }

#pragma unroll
    for (int s = SUB_GROUP_SIZE >> 1; s > 0; s >>= 1) {
      if (s < num_sg) {
        sum += sycl::shift_group_left(sg, sum, s);
        sqr += sycl::shift_group_left(sg, sqr, s);
      }
    }

    sum = sum / total;
    sqr = sqr / total - sum * sum;
    (*par_sum) = sum;
    (*par_sqr) = sqr;
  }
}

// --------------------// MeanAndVarKernel //-------------------- //

template <int SUB_GROUP_SIZE, typename T, typename U>
struct MeanAndVarKernel {
  MeanAndVarKernel(const T* input, LocalAcc<U> scratch, U* temp_mean,
                   U* temp_var, const InputShape& shape, int sx, int sy)
      : input_(input),
        scratch_(scratch),
        temp_mean_(temp_mean),
        temp_var_(temp_var),
        num_hw_(shape.num_hw),
        num_channels_(shape.num_channels),
        chans_per_group_(shape.chans_per_group),
        sx_(sx),
        sy_(sy) {}

  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(
      sycl::nd_item<2> item) const {
    int batch_id = item.get_group(0);
    int group_id = item.get_group(1);
    int id = item.get_local_id(1);

    const T* p_input = input_ + batch_id * num_hw_ * num_channels_ +
                       group_id * chans_per_group_;

    int iy = id / sx_;
    int ix = id - iy * sx_;

    U sum = U(0);
    U sqr = U(0);
    for (int jy = iy; jy < num_hw_; jy += sy_) {
      const T* pval = p_input + jy * num_channels_;

      for (int jx = ix; jx < chans_per_group_; jx += sx_) {
        U value = static_cast<U>(pval[jx]);
        sum += value;
        sqr += value * value;
      }
    }

    U* lmem = scratch_.get_pointer().get();
    int total = num_hw_ * chans_per_group_;
    GroupMeanVar<SUB_GROUP_SIZE>(item, &sum, &sqr, total, lmem);

    if (id == 0) {
      int offset = batch_id * (num_channels_ / chans_per_group_) + group_id;
      temp_mean_[offset] = sum;
      temp_var_[offset] = sqr;
    }
  }

 private:
  const T* input_;
  LocalAcc<U> scratch_;
  U* temp_mean_;
  U* temp_var_;
  int num_hw_;
  int num_channels_;
  int chans_per_group_;
  int sx_;
  int sy_;
};

// Compute mean and variance in one kernel
template <int SUB_GROUP_SIZE, typename T, typename U>
void LaunchMeanAndVarKernel(const GPUDevice& d, const T* input, U* temp_mean,
                            U* temp_var, const InputShape& shape) {
  auto stream = d.stream();
  int group_size = (*stream)
                       .get_device()
                       .get_info<sycl::info::device::max_work_group_size>();

  int sx = SUB_GROUP_SIZE;
  while (sx << 1 <= shape.chans_per_group) sx <<= 1;
  sx = std::min(sx, group_size);
  int sy = group_size / sx;

  // shared local memory size
  size_t lmem_size = group_size / SUB_GROUP_SIZE * 2;

  // Create the range object
  sycl::range<2> global(shape.num_batches, shape.num_groups * group_size);
  sycl::range<2> local(1, group_size);
  sycl::nd_range<2> range(global, local);

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc<U> scratch(sycl::range<1>{lmem_size}, cgh);
    MeanAndVarKernel<SUB_GROUP_SIZE, T, U> task(input, scratch, temp_mean,
                                                temp_var, shape, sx, sy);
    cgh.parallel_for<MeanAndVarKernel<SUB_GROUP_SIZE, T, U>>(range, task);
  });
}

// --------------------// PartialSumKernel //-------------------- //

template <int SUB_GROUP_SIZE, typename T, typename U, int VECSize>
struct PartialSumKernel {
  PartialSumKernel(const T* input, LocalAcc<U> scratch, U* temp_sum,
                   U* temp_sqr, const InputShape& shape, int scaled_hw)
      : input_(input),
        scratch_(scratch),
        temp_sum_(temp_sum),
        temp_sqr_(temp_sqr),
        num_hw_(shape.num_hw),
        num_channels_(shape.num_channels),
        num_groups_(shape.num_groups),
        chans_per_group_(shape.chans_per_group),
        scaled_hw_(scaled_hw) {}

  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(
      sycl::nd_item<2> item) const {
    int batch_id = item.get_group(0);
    int scaled_id = item.get_group(1);
    int group_size = item.get_local_range(1);
    int id = item.get_local_id(1);

    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();
    int num_sg = sg.get_group_linear_range();

    const T* p_input = input_ + batch_id * num_hw_ * num_channels_;
    U* lmem = scratch_.get_pointer().get();

    for (int idx = id * VECSize; idx < num_channels_;
         idx += group_size * VECSize) {
      sycl::vec<U, VECSize> sum{0};
      sycl::vec<U, VECSize> sqr{0};

      for (int ihw = scaled_id; ihw < num_hw_; ihw += scaled_hw_) {
        sycl::vec<U, VECSize> value;
        PacketLoad(p_input, ihw * num_channels_ + idx, &value);

        for (int j = 0; j < VECSize; ++j) {
          U acc = value[j];
          sum[j] += acc;
          sqr[j] += acc * acc;
        }
      }

      *(reinterpret_cast<sycl::vec<U, VECSize>*>(lmem + idx)) = sum;
      *(reinterpret_cast<sycl::vec<U, VECSize>*>(lmem + idx + num_channels_)) =
          sqr;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int group_id = sg_id; group_id < num_groups_; group_id += num_sg) {
      U* data = lmem + group_id * chans_per_group_;

      U sum = U(0);
      U sqr = U(0);
      for (int i = sg_local_id; i < chans_per_group_; i += SUB_GROUP_SIZE) {
        sum += data[i];
        sqr += data[i + num_channels_];
      }

#pragma unroll
      for (int s = SUB_GROUP_SIZE >> 1; s > 0; s >>= 1) {
        sum += sycl::shift_group_left(sg, sum, s);
        sqr += sycl::shift_group_left(sg, sqr, s);
      }

      if (sg_local_id == 0) {
        int offset = batch_id * scaled_hw_ * num_groups_ +
                     group_id * scaled_hw_ + scaled_id;
        temp_sum_[offset] = sum;
        temp_sqr_[offset] = sqr;
      }
    }
  }

 private:
  const T* input_;
  LocalAcc<U> scratch_;
  U* temp_sum_;
  U* temp_sqr_;
  int num_hw_;
  int num_channels_;
  int num_groups_;
  int chans_per_group_;
  int scaled_hw_;
};

// Compute sum and square sum of data in each group
template <int SUB_GROUP_SIZE, typename T, typename U>
void LaunchPartialSumKernel(const GPUDevice& d, const T* input, U* temp_sum,
                            U* temp_sqr, const InputShape& shape,
                            int scaled_hw) {
  auto stream = d.stream();
  size_t max_group_size =
      (*stream)
          .get_device()
          .get_info<sycl::info::device::max_work_group_size>();

  int VECSize = 4;
  while (shape.chans_per_group % VECSize != 0 && VECSize > 1) {
    VECSize >>= 1;
  }

  size_t group_size = SUB_GROUP_SIZE;
  while (group_size << 1 <= (shape.num_channels / VECSize)) group_size <<= 1;
  group_size = std::min(group_size, max_group_size);

  // shared local memory size
  size_t lmem_size = shape.num_channels << 1;

  // Create the range object
  sycl::range<2> global(shape.num_batches, scaled_hw * group_size);
  sycl::range<2> local(1, group_size);
  sycl::nd_range<2> range(global, local);

  if (VECSize == 4) {
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<U> scratch(sycl::range<1>{lmem_size}, cgh);
      PartialSumKernel<SUB_GROUP_SIZE, T, U, 4> task(
          input, scratch, temp_sum, temp_sqr, shape, scaled_hw);
      cgh.parallel_for<PartialSumKernel<SUB_GROUP_SIZE, T, U, 4>>(range, task);
    });
  } else if (VECSize == 2) {
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<U> scratch(sycl::range<1>{lmem_size}, cgh);
      PartialSumKernel<SUB_GROUP_SIZE, T, U, 2> task(
          input, scratch, temp_sum, temp_sqr, shape, scaled_hw);
      cgh.parallel_for<PartialSumKernel<SUB_GROUP_SIZE, T, U, 2>>(range, task);
    });
  } else {
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<U> scratch(sycl::range<1>{lmem_size}, cgh);
      PartialSumKernel<SUB_GROUP_SIZE, T, U, 1> task(
          input, scratch, temp_sum, temp_sqr, shape, scaled_hw);
      cgh.parallel_for<PartialSumKernel<SUB_GROUP_SIZE, T, U, 1>>(range, task);
    });
  }
}

// --------------------// MeanFromPartialKernel //-------------------- //

template <int SUB_GROUP_SIZE, typename T>
struct MeanFromPartialKernel {
  MeanFromPartialKernel(const T* temp_sum, const T* temp_sqr,
                        LocalAcc<T> scratch, T* temp_mean, T* temp_var,
                        const InputShape& shape, int scaled_hw)
      : temp_sum_(temp_sum),
        temp_sqr_(temp_sqr),
        scratch_(scratch),
        temp_mean_(temp_mean),
        temp_var_(temp_var),
        num_hw_(shape.num_hw),
        num_channels_(shape.num_channels),
        num_groups_(shape.num_groups),
        chans_per_group_(shape.chans_per_group),
        scaled_hw_(scaled_hw) {}

  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(
      sycl::nd_item<2> item) const {
    int batch_id = item.get_group(0);
    int group_id = item.get_group(1);
    int group_size = item.get_local_range(1);
    int id = item.get_local_id(1);

    int offset = batch_id * scaled_hw_ * num_groups_ + group_id * scaled_hw_;
    const T* p_sum = temp_sum_ + offset;
    const T* p_sqr = temp_sqr_ + offset;

    T sum = 0;
    T sqr = 0;
    for (int i = id; i < scaled_hw_; i += group_size) {
      sum += p_sum[i];
      sqr += p_sqr[i];
    }

    T* lmem = scratch_.get_pointer().get();
    int total = num_hw_ * chans_per_group_;
    GroupMeanVar<SUB_GROUP_SIZE>(item, &sum, &sqr, total, lmem);

    if (id == 0) {
      int offset = batch_id * num_groups_ + group_id;
      temp_mean_[offset] = sum;
      temp_var_[offset] = sqr;
    }
  }

 private:
  const T* temp_sum_;
  const T* temp_sqr_;
  LocalAcc<T> scratch_;
  T* temp_mean_;
  T* temp_var_;
  int num_hw_;
  int num_channels_;
  int num_groups_;
  int chans_per_group_;
  int scaled_hw_;
};

// Compute sum and square sum of data in each group
template <int SUB_GROUP_SIZE, typename T>
void LaunchMeanFromPartialKernel(const GPUDevice& d, const T* temp_sum,
                                 const T* temp_sqr, T* temp_mean, T* temp_var,
                                 const InputShape& shape, int scaled_hw) {
  auto stream = d.stream();
  auto max_group_size =
      (*stream)
          .get_device()
          .get_info<sycl::info::device::max_work_group_size>();

  size_t group_size = SUB_GROUP_SIZE;
  while (group_size << 1 <= scaled_hw) group_size <<= 1;
  group_size = std::min(group_size, max_group_size);

  // shared local memory size
  size_t lmem_size = group_size / SUB_GROUP_SIZE * 2;

  // Create the range object
  sycl::range<2> global(shape.num_batches, shape.num_groups * group_size);
  sycl::range<2> local(1, group_size);
  sycl::nd_range<2> range(global, local);

  stream->submit([&](sycl::handler& cgh) {
    LocalAcc<T> scratch(sycl::range<1>{lmem_size}, cgh);
    MeanFromPartialKernel<SUB_GROUP_SIZE, T> task(
        temp_sum, temp_sqr, scratch, temp_mean, temp_var, shape, scaled_hw);
    cgh.parallel_for<MeanFromPartialKernel<SUB_GROUP_SIZE, T>>(range, task);
  });
}

// --------------------// NormalizationKernel //-------------------- //

template <bool USE_SCALE, bool USE_CENTER, typename T, typename U,
          int VEC_ON_HW, int VEC_ON_CHAN>
struct NormalizationKernel {
  NormalizationKernel(const T* input, const T* gamma, const T* beta,
                      const U* temp_mean, const U* temp_var, float epsilon,
                      T* output, const InputShape& shape)
      : input_(input),
        gamma_(gamma),
        beta_(beta),
        temp_mean_(temp_mean),
        temp_var_(temp_var),
        epsilon_(epsilon),
        output_(output),
        num_hw_(shape.num_hw),
        num_channels_(shape.num_channels),
        num_groups_(shape.num_groups),
        chans_per_group_(shape.chans_per_group) {}
  void operator()(sycl::nd_item<2> item) const {
    int batch_id = item.get_group(0);
    int gid = item.get_group(1);
    int lid = item.get_local_id(1);
    int group_size = item.get_local_range(1);
    int id = lid + gid * group_size;
    int vec_num_hw = num_hw_ / VEC_ON_HW;
    int vec_chan = num_channels_ / VEC_ON_CHAN;
    if (id >= vec_num_hw * vec_chan) return;

    int batch_offset = batch_id * num_hw_ * num_channels_;
    int hw_id = id / vec_chan;
    int channel_id = id - hw_id * vec_chan;
    int group_id = channel_id / (chans_per_group_ / VEC_ON_CHAN);
    int chans_per_group_id =
        channel_id - group_id * (chans_per_group_ / VEC_ON_CHAN);

    sycl::vec<U, VEC_ON_CHAN> data[VEC_ON_HW], res_data[VEC_ON_HW];

    for (int i = 0; i < VEC_ON_HW; ++i) {
      const T* in_ptr =
          input_ + batch_offset + (hw_id + i * vec_num_hw) * num_channels_ +
          group_id * chans_per_group_ + chans_per_group_id * VEC_ON_CHAN;
      PacketLoad(in_ptr, 0, &data[i]);
    }

    int offset_temp = batch_id * num_groups_ + group_id;

    U mean_data = temp_mean_[offset_temp];
    U inv = sycl::rsqrt(temp_var_[offset_temp] + epsilon_);

    sycl::vec<U, VEC_ON_CHAN> gamma_data{1}, beta_data{0};
    int gamma_offset =
        group_id * chans_per_group_ + chans_per_group_id * VEC_ON_CHAN;
    if constexpr (USE_SCALE) {
      PacketLoad(gamma_, gamma_offset, &gamma_data);
    }
    if constexpr (USE_CENTER) {
      PacketLoad(beta_, gamma_offset, &beta_data);
    }

    for (int i = 0; i < VEC_ON_HW; ++i) {
      for (int j = 0; j < VEC_ON_CHAN; ++j) {
        res_data[i][j] =
            (data[i][j] - mean_data) * inv * gamma_data[j] + beta_data[j];
      }
    }

    for (int i = 0; i < VEC_ON_HW; ++i) {
      T* out_ptr =
          output_ + batch_offset + (hw_id + i * vec_num_hw) * num_channels_ +
          group_id * chans_per_group_ + chans_per_group_id * VEC_ON_CHAN;
      // *(reinterpret_cast<sycl::vec<T, VEC_ON_CHAN>*>(out_ptr)) = res_data[i];
      PacketStore(out_ptr, 0, res_data[i]);
    }
  }

 private:
  const T* input_;
  const T* gamma_;
  const T* beta_;
  const U* temp_mean_;
  const U* temp_var_;
  float epsilon_;
  T* output_;
  int num_hw_;
  int num_channels_;
  int num_groups_;
  int chans_per_group_;
};

// Do group normalization
template <bool USE_SCALE, bool USE_CENTER, typename T, typename U>
void LaunchNormalizationKernel(const GPUDevice& d, const T* input,
                               const T* gamma, const T* beta,
                               const U* temp_mean, const U* temp_var,
                               float epsilon, T* output,
                               const InputShape& shape) {
  auto stream = d.stream();
  auto group_size = (*stream)
                        .get_device()
                        .get_info<sycl::info::device::max_work_group_size>();

  int vec_on_hw = 4;
  while (shape.num_hw % vec_on_hw != 0 && vec_on_hw > 1) {
    vec_on_hw /= 2;
  }

  int vec_on_chan = 4;
  while (shape.chans_per_group % vec_on_chan != 0 && vec_on_chan > 1) {
    vec_on_chan /= 2;
  }

  int num_elems = shape.num_channels * shape.num_hw / vec_on_hw / vec_on_chan;
  int num_wg = (num_elems + group_size - 1) / group_size;

  // Create the range object
  sycl::range<2> global(shape.num_batches, num_wg * group_size);
  sycl::range<2> local(1, group_size);
  sycl::nd_range<2> range(global, local);

  if (vec_on_hw == 4) {
    switch (vec_on_chan) {
#define SUB_KENEL(N)                                                        \
  case N:                                                                   \
    stream->submit([&](sycl::handler& cgh) {                                \
      NormalizationKernel<true, true, T, U, 4, N> task(                     \
          input, gamma, beta, temp_mean, temp_var, epsilon, output, shape); \
      cgh.parallel_for<NormalizationKernel<true, true, T, U, 4, N>>(range,  \
                                                                    task);  \
    });                                                                     \
    break;
      SUB_KENEL(4)
      SUB_KENEL(2)
      SUB_KENEL(1)
#undef SUB_KENEL
    }
  } else if (vec_on_hw == 2) {
    switch (vec_on_chan) {
#define SUB_KENEL(N)                                                        \
  case N:                                                                   \
    stream->submit([&](sycl::handler& cgh) {                                \
      NormalizationKernel<true, true, T, U, 2, N> task(                     \
          input, gamma, beta, temp_mean, temp_var, epsilon, output, shape); \
      cgh.parallel_for<NormalizationKernel<true, true, T, U, 2, N>>(range,  \
                                                                    task);  \
    });                                                                     \
    break;
      SUB_KENEL(4)
      SUB_KENEL(2)
      SUB_KENEL(1)
#undef SUB_KENEL
    }
  } else {
    switch (vec_on_chan) {
#define SUB_KENEL(N)                                                        \
  case N:                                                                   \
    stream->submit([&](sycl::handler& cgh) {                                \
      NormalizationKernel<true, true, T, U, 1, N> task(                     \
          input, gamma, beta, temp_mean, temp_var, epsilon, output, shape); \
      cgh.parallel_for<NormalizationKernel<true, true, T, U, 1, N>>(range,  \
                                                                    task);  \
    });                                                                     \
    break;
      SUB_KENEL(4)
      SUB_KENEL(2)
      SUB_KENEL(1)
#undef SUB_KENEL
    }
  }
}

}  // end namespace impl
}  // end namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_GROUP_NORM_OP_GPU_H_
