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

#ifndef ITEX_CORE_KERNELS_GPU_SOFTMAX_OP_FUNCTOR_H_
#define ITEX_CORE_KERNELS_GPU_SOFTMAX_OP_FUNCTOR_H_

#include <cassert>
#include <limits>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/hw_info.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace impl {

namespace softmax {

template <typename T>
T Inf() {
  return std::numeric_limits<T>::max();
}

template <typename T>
T Div(T a, T b) {
  return a / b;
}

template <typename T>
T Max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
T Exp(T x) {
#if __SYCL_DEVICE_ONLY__
  return sycl::exp(x);
#else
  return ::exp(x);
#endif
}

template <typename T>
T Log(T x) {
#if __SYCL_DEVICE_ONLY__
  return sycl::log(x);
#else
  return ::log(x);
#endif
}

constexpr int kSubGroupSize = 32;

template <typename T>
using __shared__ = sycl::local_accessor<T, 1>;

template <typename T>
struct SumOp {
  T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct MaxOp {
  T operator()(const T& a, const T& b) const { return Max(a, b); }
};

template <template <typename> typename ReductionOp, typename T,
          int workitem_group_width = kSubGroupSize>
T SubGroupAllReduce(const sycl::sub_group& sg, T val) {
  for (int mask = workitem_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, sg.shuffle_xor(val, sycl::id<1>(mask)));
  }
  return val;
}

enum class Algorithm {
  kSoftmax = 0,
  kLogSoftmax = 1,
};

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          int cols_per_workitem, int workitem_group_width, int rows_per_access,
          bool padding, Algorithm algorithm>
struct SoftmaxSubGroupImplKernel {
  SoftmaxSubGroupImplKernel(LOAD device_load, STORE device_store, int32_t rows,
                            int32_t cols)
      : device_load(device_load),
        device_store(device_store),
        rows(rows),
        cols(cols) {}
  [[intel::reqd_sub_group_size(kSubGroupSize)]] void operator()(
      sycl::nd_item<2> id) const {
    const int workgroup_idx = id.get_group(1);
    const int localrange_y = id.get_local_range(0);
    const int workitem_y = id.get_local_id(0);
    const int grouprange_y = id.get_group_range(1);
    const int lane_id = id.get_local_id(1);
    const int num_packs = cols_per_workitem / pack_size;

    sycl::sub_group sg = id.get_sub_group();

    const int global_workitem_group_id =
        workgroup_idx * localrange_y + workitem_y;
    const int num_global_workitem_group = grouprange_y * localrange_y;

    ComputeType buf[rows_per_access][cols_per_workitem];
    for (int32 row = global_workitem_group_id * rows_per_access; row < rows;
         row += num_global_workitem_group * rows_per_access) {
      ComputeType workitem_max[rows_per_access];  // NOLINT(runtime/arrays)
#pragma unroll
      for (int row_id = 0; row_id < rows_per_access; ++row_id) {
        workitem_max[row_id] = -Inf<ComputeType>();
        ComputeType* row_buf = buf[row_id];
#pragma unroll
        for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
          const int col =
              (pack_id * workitem_group_width + lane_id) * pack_size;
          if (!padding || col < cols) {
            device_load.template Load<pack_size>(row_buf + pack_id * pack_size,
                                                 row + row_id, col);
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
              workitem_max[row_id] =
                  Max(workitem_max[row_id], row_buf[pack_id * pack_size + i]);
            }
          } else {
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
              row_buf[pack_id * pack_size + i] = -Inf<ComputeType>();
            }
          }
        }
      }
      ComputeType warp_max[rows_per_access];  // NOLINT(runtime/arrays)
#pragma unroll
      for (int row_id = 0; row_id < rows_per_access; ++row_id) {
        warp_max[row_id] =
            SubGroupAllReduce<MaxOp, ComputeType, workitem_group_width>(
                sg, workitem_max[row_id]);
      }
      ComputeType workitem_sum[rows_per_access];  // NOLINT(runtime/arrays)
#pragma unroll
      for (int row_id = 0; row_id < rows_per_access; ++row_id) {
        workitem_sum[row_id] = 0.f;
        ComputeType* row_buf = buf[row_id];
#pragma unroll
        for (int i = 0; i < cols_per_workitem; ++i) {
          if (row_buf[i] == -Inf<ComputeType>()) break;
          if (algorithm == Algorithm::kSoftmax) {
            row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
            workitem_sum[row_id] += row_buf[i];
          } else if (algorithm == Algorithm::kLogSoftmax) {
            row_buf[i] -= warp_max[row_id];
            workitem_sum[row_id] += Exp(row_buf[i]);
          }
        }
      }
      ComputeType warp_sum[rows_per_access];  // NOLINT(runtime/arrays)
#pragma unroll
      for (int row_id = 0; row_id < rows_per_access; ++row_id) {
        warp_sum[row_id] =
            SubGroupAllReduce<SumOp, ComputeType, workitem_group_width>(
                sg, workitem_sum[row_id]);
      }
#pragma unroll
      for (int row_id = 0; row_id < rows_per_access; ++row_id) {
        ComputeType* row_buf = buf[row_id];
#pragma unroll
        for (int i = 0; i < cols_per_workitem; ++i) {
          if (row_buf[i] == -Inf<ComputeType>()) break;
          if (algorithm == Algorithm::kSoftmax) {
            row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
          } else if (algorithm == Algorithm::kLogSoftmax) {
            row_buf[i] -= Log(warp_sum[row_id]);
          }
        }
#pragma unroll
        for (int i = 0; i < num_packs; ++i) {
          const int col = (i * workitem_group_width + lane_id) * pack_size;
          if (!padding || col < cols) {
            device_store.template Store<pack_size>(row_buf + i * pack_size,
                                                   row + row_id, col);
          }
        }
      }
    }
  }

 private:
  LOAD device_load;
  STORE device_store;
  const int32 rows;
  const int32 cols;
};

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          int cols_per_workitem, int workitem_group_width, int rows_per_access,
          bool padding, Algorithm algorithm>
Status SoftmaxSubGroupImpl(const GPUDevice& device, LOAD device_load,
                           STORE device_store, const int32 rows,
                           const int32 cols, sycl::range<2> global_range,
                           sycl::range<2> local_range) {
  const auto stream = device.stream();

  static_assert(cols_per_workitem % pack_size == 0, "");
  static_assert(workitem_group_width <= kSubGroupSize, "");
  static_assert(kSubGroupSize % workitem_group_width == 0, "");

  assert(cols <= cols_per_workitem * workitem_group_width);

  stream->submit([&](sycl::handler& h) {
    SoftmaxSubGroupImplKernel<LOAD, STORE, ComputeType, pack_size,
                              cols_per_workitem, workitem_group_width,
                              rows_per_access, padding, algorithm>
        task(device_load, device_store, rows, cols);
    h.parallel_for<SoftmaxSubGroupImplKernel<
        LOAD, STORE, ComputeType, pack_size, cols_per_workitem,
        workitem_group_width, rows_per_access, padding, algorithm>>(
        sycl::nd_range<2>(global_range, local_range), task);
  });

  return Status::OK();
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          int cols_per_workitem, int workitem_group_width, int rows_per_access,
          bool padding, Algorithm algorithm>
inline Status LaunchSoftmaxSubGroupImpl(const GPUDevice& device,
                                        LOAD device_load, STORE device_store,
                                        const int32 rows, const int32 cols) {
  constexpr int workgroup_size =
      128;  // 128 is the best choice for all machines
  static_assert(workgroup_size % workitem_group_width == 0, "");
  constexpr int rows_per_block = workgroup_size / workitem_group_width;
  sycl::range<2> local_range(rows_per_block, workitem_group_width);

  const int32 num_blocks = (rows + rows_per_block - 1) / rows_per_block;
  sycl::range<2> global_range(rows_per_block,
                              num_blocks * workitem_group_width);

  auto status = SoftmaxSubGroupImpl<LOAD, STORE, ComputeType, pack_size,
                                    cols_per_workitem, workitem_group_width,
                                    rows_per_access, padding, algorithm>(
      device, device_load, device_store, rows, cols, global_range, local_range);

  return status;
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          int cols_per_workitem, int workitem_group_width, int rows_per_access,
          Algorithm algorithm>
inline Status DispatchSoftmaxSubGroupImplPadding(const GPUDevice& device,
                                                 LOAD device_load,
                                                 STORE device_store,
                                                 const int32 rows,
                                                 const int32 cols) {
  if (cols == cols_per_workitem * workitem_group_width) {
    return LaunchSoftmaxSubGroupImpl<LOAD, STORE, ComputeType, pack_size,
                                     cols_per_workitem, workitem_group_width,
                                     rows_per_access, false, algorithm>(
        device, device_load, device_store, rows, cols);
  } else {
    return LaunchSoftmaxSubGroupImpl<LOAD, STORE, ComputeType, pack_size,
                                     cols_per_workitem, workitem_group_width,
                                     rows_per_access, true, algorithm>(
        device, device_load, device_store, rows, cols);
  }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
typename std::enable_if<pack_size == 1, Status>::type
DispatchSoftmaxSubGroupImplCols(const GPUDevice& device, LOAD device_load,
                                STORE device_store, const int32 rows,
                                const int32 cols) {
  if (cols <= 0) {
    return Status(TF_INVALID_ARGUMENT, "Invalid Value");
  }
#define DEFINE_ONE_ELIF(workitem_group_width)                            \
  if (cols <= (workitem_group_width)*pack_size) {                        \
    if (rows % 2 == 0) {                                                 \
      return DispatchSoftmaxSubGroupImplPadding<                         \
          LOAD, STORE, ComputeType, pack_size, pack_size,                \
          workitem_group_width, 2, algorithm>(device, device_load,       \
                                              device_store, rows, cols); \
    } else {                                                             \
      return DispatchSoftmaxSubGroupImplPadding<                         \
          LOAD, STORE, ComputeType, pack_size, pack_size,                \
          workitem_group_width, 1, algorithm>(device, device_load,       \
                                              device_store, rows, cols); \
    }                                                                    \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                 \
  if (cols <= (col)*kSubGroupSize) {                                         \
    return DispatchSoftmaxSubGroupImplPadding<LOAD, STORE, ComputeType,      \
                                              pack_size, col, kSubGroupSize, \
                                              1, algorithm>(                 \
        device, device_load, device_store, rows, cols);                      \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF

  return Status(TF_INVALID_ARGUMENT, "Invalid Value");
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
typename std::enable_if<pack_size == 2, Status>::type
DispatchSoftmaxSubGroupImplCols(const GPUDevice& device, LOAD device_load,
                                STORE device_store, const int32 rows,
                                const int32 cols) {
  if (cols <= 0) {
    return Status(TF_INVALID_ARGUMENT, "Invalid Value");
  }
#define DEFINE_ONE_ELIF(workitem_group_width)                            \
  if (cols <= (workitem_group_width)*pack_size) {                        \
    if (rows % 2 == 0) {                                                 \
      return DispatchSoftmaxSubGroupImplPadding<                         \
          LOAD, STORE, ComputeType, pack_size, pack_size,                \
          workitem_group_width, 2, algorithm>(device, device_load,       \
                                              device_store, rows, cols); \
    } else {                                                             \
      return DispatchSoftmaxSubGroupImplPadding<                         \
          LOAD, STORE, ComputeType, pack_size, pack_size,                \
          workitem_group_width, 1, algorithm>(device, device_load,       \
                                              device_store, rows, cols); \
    }                                                                    \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                 \
  if (cols <= (col)*kSubGroupSize) {                                         \
    return DispatchSoftmaxSubGroupImplPadding<LOAD, STORE, ComputeType,      \
                                              pack_size, col, kSubGroupSize, \
                                              1, algorithm>(                 \
        device, device_load, device_store, rows, cols);                      \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF

  return Status(TF_INVALID_ARGUMENT, "Invalid Value");
}

template <typename LOAD, typename STORE, typename ComputeType,
          Algorithm algorithm>
struct DispatchSoftmaxSubGroupImplPackSize {
  Status operator()(const GPUDevice& device, LOAD device_load,
                    STORE device_store, const int32 rows, const int32 cols) {
    if (cols % 2 == 0) {
      return DispatchSoftmaxSubGroupImplCols<LOAD, STORE, ComputeType, 2,
                                             algorithm>(
          device, device_load, device_store, rows, cols);
    } else {
      return DispatchSoftmaxSubGroupImplCols<LOAD, STORE, ComputeType, 1,
                                             algorithm>(
          device, device_load, device_store, rows, cols);
    }
  }
};

template <typename LOAD, typename STORE, typename ComputeType,
          Algorithm algorithm>
inline Status DispatchSoftmaxSubGroupImpl(const GPUDevice& device,
                                          LOAD device_load, STORE device_store,
                                          const int32 rows, const int32 cols) {
  return DispatchSoftmaxSubGroupImplPackSize<LOAD, STORE, ComputeType,
                                             algorithm>()(
      device, device_load, device_store, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
struct SoftmaxWorkgroupSMemImplKernel {
  SoftmaxWorkgroupSMemImplKernel(__shared__<unsigned char> scratch, int rows,
                                 int cols, int workgroup_size, int num_packs,
                                 LOAD device_load, STORE device_store)
      :

        scratch(scratch),
        rows(rows),
        cols(cols),
        workgroup_size(workgroup_size),
        num_packs(num_packs),
        device_load(device_load),
        device_store(device_store) {}
  void operator()(sycl::nd_item<1> id) const {
    auto* buf = reinterpret_cast<ComputeType*>(
        ITEXGetLocalAccPointer<unsigned char>(scratch));
    const int local_id = id.get_local_id(0);
    for (int32 row = id.get_group(0); row < rows;
         row += id.get_group_range(0)) {
      ComputeType workitem_max = -Inf<ComputeType>();
      for (int pack_id = local_id; pack_id < num_packs;
           pack_id += workgroup_size) {
        ComputeType pack[pack_size];  // NOLINT(runtime/arrays)
        device_load.template Load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          buf[i * num_packs + pack_id] = pack[i];
          workitem_max = Max(workitem_max, pack[i]);
        }
      }
      const ComputeType row_max =
          sycl::reduce_over_group(id.get_group(), workitem_max,
                                  sycl::ext::oneapi::maximum<ComputeType>());
      ComputeType workitem_sum = 0.f;
      for (int col = local_id; col < cols; col += workgroup_size) {
        if (algorithm == Algorithm::kSoftmax) {
          const ComputeType exp_x = Exp(buf[col] - row_max);
          buf[col] = exp_x;
          workitem_sum += exp_x;
        } else {
          const ComputeType x = buf[col] - row_max;
          buf[col] = x;
          workitem_sum += Exp(x);
        }
      }
      const ComputeType row_sum = sycl::reduce_over_group(
          id.get_group(), workitem_sum, sycl::ext::oneapi::plus<ComputeType>());
      for (int pack_id = local_id; pack_id < num_packs;
           pack_id += workgroup_size) {
        ComputeType pack[pack_size];  // NOLINT(runtime/arrays)
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          if (algorithm == Algorithm::kSoftmax) {
            pack[i] = Div(buf[i * num_packs + pack_id], row_sum);
          } else if (algorithm == Algorithm::kLogSoftmax) {
            pack[i] = buf[i * num_packs + pack_id] - Log(row_sum);
          }
        }
        device_store.template Store<pack_size>(pack, row, pack_id * pack_size);
      }
    }
  }

 private:
  __shared__<unsigned char> scratch;
  int rows;
  int cols;
  int workgroup_size;
  int num_packs;
  LOAD device_load;
  STORE device_store;
};

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
inline Status SoftmaxWorkgroupSMemImpl(const GPUDevice& device,
                                       LOAD device_load, STORE device_store,
                                       const int32 rows, const int32 cols,
                                       sycl::range<1> global_range,
                                       sycl::range<1> local_range,
                                       int workgroup_size) {
  auto stream = device.stream();
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;

  stream->submit([&](sycl::handler& h) {
    __shared__<unsigned char> scratch(
        sycl::range<1>(cols * sizeof(ComputeType)), h);
    SoftmaxWorkgroupSMemImplKernel<LOAD, STORE, ComputeType, pack_size,
                                   algorithm>
        task(scratch, rows, cols, workgroup_size, num_packs, device_load,
             device_store);

    h.parallel_for<SoftmaxWorkgroupSMemImplKernel<LOAD, STORE, ComputeType,
                                                  pack_size, algorithm>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });

  return Status::OK();
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
inline Status LaunchSoftmaxWorkGroupSMemImpl(const GPUDevice& device,
                                             LOAD device_load,
                                             STORE device_store,
                                             const int32 rows,
                                             const int32 cols) {
  int workgroup_size =
      device.stream()
          ->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  sycl::range<1> local_range(workgroup_size);
  int num_wg;
  GetNumWorkGroups(device.stream()->get_device(), workgroup_size, rows, 32,
                   &num_wg);
  sycl::range<1> global_range(num_wg * workgroup_size);

  return SoftmaxWorkgroupSMemImpl<LOAD, STORE, ComputeType, pack_size,
                                  algorithm>(device, device_load, device_store,
                                             rows, cols, global_range,
                                             local_range, workgroup_size);
}

template <typename LOAD, typename STORE, typename ComputeType,
          Algorithm algorithm>
struct DispatchSoftmaxWorkGroupSMemImplPackSize {
  Status operator()(const GPUDevice& device, LOAD device_load,
                    STORE device_store, const int32 rows, const int32 cols) {
    if (cols % 2 == 0) {
      return LaunchSoftmaxWorkGroupSMemImpl<LOAD, STORE, ComputeType, 2,
                                            algorithm>(
          device, device_load, device_store, rows, cols);
    } else {
      return LaunchSoftmaxWorkGroupSMemImpl<LOAD, STORE, ComputeType, 1,
                                            algorithm>(
          device, device_load, device_store, rows, cols);
    }
  }
};

template <typename LOAD, typename STORE, typename ComputeType,
          Algorithm algorithm>
inline Status DispatchSoftmaxWorkGroupSMemImpl(const GPUDevice& device,
                                               LOAD device_load,
                                               STORE device_store,
                                               const int32 rows,
                                               const int32 cols) {
  return DispatchSoftmaxWorkGroupSMemImplPackSize<LOAD, STORE, ComputeType,
                                                  algorithm>()(
      device, device_load, device_store, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
struct SoftmaxWorkGroupUncachedImplKernel {
  SoftmaxWorkGroupUncachedImplKernel(LOAD device_load, STORE device_store,
                                     const int32 rows, const int32 cols,
                                     int workgroup_size)
      :

        device_load(device_load),
        device_store(device_store),
        rows(rows),
        cols(cols),
        workgroup_size(workgroup_size) {}
  void operator()(sycl::nd_item<1> id) const {
    const int local_id = id.get_local_id(0);
    const int num_packs = cols / pack_size;
    for (int32 row = id.get_group(0); row < rows;
         row += id.get_group_range(0)) {
      ComputeType workitem_max = -Inf<ComputeType>();
      for (int pack_id = local_id; pack_id < num_packs;
           pack_id += workgroup_size) {
        ComputeType pack[pack_size];  // NOLINT(runtime/arrays)
        device_load.template Load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          workitem_max = Max(workitem_max, pack[i]);
        }
      }
      const ComputeType row_max =
          sycl::reduce_over_group(id.get_group(), workitem_max,
                                  sycl::ext::oneapi::maximum<ComputeType>());
      ComputeType workitem_sum = 0;
      for (int pack_id = local_id; pack_id < num_packs;
           pack_id += workgroup_size) {
        ComputeType pack[pack_size];  // NOLINT(runtime/arrays)
        device_load.template Load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          workitem_sum += Exp(pack[i] - row_max);
        }
      }
      // TODO(itex): try to reimplement reduce ops
      const ComputeType row_sum = sycl::reduce_over_group(
          id.get_group(), workitem_sum, sycl::ext::oneapi::plus<ComputeType>());
      for (int pack_id = local_id; pack_id < num_packs;
           pack_id += workgroup_size) {
        ComputeType pack[pack_size];  // NOLINT(runtime/arrays)
        device_load.template Load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          if (algorithm == Algorithm::kSoftmax) {
            pack[i] = Div(Exp(pack[i] - row_max), row_sum);
          } else if (algorithm == Algorithm::kLogSoftmax) {
            pack[i] = (pack[i] - row_max) - Log(row_sum);
          }
        }
        device_store.template Store<pack_size>(pack, row, pack_id * pack_size);
      }
    }
  }

 private:
  LOAD device_load;
  STORE device_store;
  const int32 rows;
  const int32 cols;
  int workgroup_size;
};

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
inline Status SoftmaxWorkGroupUncachedImpl(const GPUDevice& device,
                                           LOAD device_load, STORE device_store,
                                           const int32 rows, const int32 cols,
                                           sycl::range<1> global_range,
                                           sycl::range<1> local_range,
                                           int workgroup_size) {
  const auto stream = device.stream();
  assert(cols % pack_size == 0);
  stream->submit([&](sycl::handler& h) {
    SoftmaxWorkGroupUncachedImplKernel<LOAD, STORE, ComputeType, pack_size,
                                       algorithm>
        task(device_load, device_store, rows, cols, workgroup_size);
    h.parallel_for<SoftmaxWorkGroupUncachedImplKernel<LOAD, STORE, ComputeType,
                                                      pack_size, algorithm>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          Algorithm algorithm>
inline Status LaunchSoftmaxWorkGroupUncachedImpl(const GPUDevice& device,
                                                 LOAD device_load,
                                                 STORE device_store,
                                                 const int32 rows,
                                                 const int32 cols) {
  int max_group_size =
      device.stream()
          ->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  sycl::range<1> local_range(max_group_size);
  int num_wg;
  GetNumWorkGroups(device.stream()->get_device(), max_group_size, rows, 32,
                   &num_wg);
  sycl::range<1> global_range(num_wg * max_group_size);

  return SoftmaxWorkGroupUncachedImpl<LOAD, STORE, ComputeType, pack_size,
                                      algorithm>(
      device, device_load, device_store, rows, cols, global_range, local_range,
      max_group_size);
}

template <typename LOAD, typename STORE, typename ComputeType,
          Algorithm algorithm>
struct DispatchSoftmaxWorkGroupUncachedImplPackSize {
  Status operator()(const GPUDevice& device, LOAD device_load,
                    STORE device_store, const int32 rows, const int32 cols) {
    if (cols % 2 == 0) {
      return LaunchSoftmaxWorkGroupUncachedImpl<LOAD, STORE, ComputeType, 2,
                                                algorithm>(
          device, device_load, device_store, rows, cols);
    } else {
      return LaunchSoftmaxWorkGroupUncachedImpl<LOAD, STORE, ComputeType, 1,
                                                algorithm>(
          device, device_load, device_store, rows, cols);
    }
  }
};

template <typename LOAD, typename STORE, typename ComputeType,
          Algorithm algorithm>
inline Status DispatchSoftmaxWorkGroupUncachedImpl(const GPUDevice& device,
                                                   LOAD device_load,
                                                   STORE device_store,
                                                   const int32 rows,
                                                   const int32 cols) {
  return DispatchSoftmaxWorkGroupUncachedImplPackSize<LOAD, STORE, ComputeType,
                                                      algorithm>()(
      device, device_load, device_store, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType>
inline Status DispatchLogSoftmax(const GPUDevice& device, LOAD device_load,
                                 STORE device_store, const int32 num_rows,
                                 const int32 num_cols, bool log) {
  assert(log == true);
  const int32 sharedMemPerBlock =
      device.stream()
          ->get_device()
          .template get_info<sycl::info::device::local_mem_size>();
  auto sycl_device = device.stream()->get_device();
  if (IsXeHPC(&sycl_device) && num_cols <= kSubGroupSize * 32) {
    return DispatchSoftmaxSubGroupImpl<decltype(device_load),
                                       decltype(device_store), ComputeType,
                                       Algorithm::kLogSoftmax>(
        device, device_load, device_store, num_rows, num_cols);
  } else {
    if (sizeof(ComputeType) * num_cols < sharedMemPerBlock / 2) {
      return DispatchSoftmaxWorkGroupSMemImpl<
          decltype(device_load), decltype(device_store), ComputeType,
          Algorithm::kLogSoftmax>(device, device_load, device_store, num_rows,
                                  num_cols);
    } else {
      return DispatchSoftmaxWorkGroupUncachedImpl<
          decltype(device_load), decltype(device_store), ComputeType,
          Algorithm::kLogSoftmax>(device, device_load, device_store, num_rows,
                                  num_cols);
    }
  }
}

// softmax functor still use onednn, this impl just in case
template <typename LOAD, typename STORE, typename ComputeType>
inline Status DispatchSoftmax(const GPUDevice& device, LOAD device_load,
                              STORE device_store, const int32 num_rows,
                              const int32 num_cols, bool log) {
  assert(log == false);
  const int32 sharedMemPerBlock =
      device.stream()
          ->get_device()
          .template get_info<sycl::info::device::local_mem_size>();
  auto sycl_device = device.stream()->get_device();
  if (IsXeHPC(&sycl_device) && num_cols <= kSubGroupSize * 32) {
    return DispatchSoftmaxSubGroupImpl<decltype(device_load),
                                       decltype(device_store), ComputeType,
                                       Algorithm::kSoftmax>(
        device, device_load, device_store, num_rows, num_cols);
  } else {
    if (sizeof(ComputeType) * num_cols < sharedMemPerBlock / 2) {
      return DispatchSoftmaxWorkGroupSMemImpl<decltype(device_load),
                                              decltype(device_store),
                                              ComputeType, Algorithm::kSoftmax>(
          device, device_load, device_store, num_rows, num_cols);
    } else {
      return DispatchSoftmaxWorkGroupUncachedImpl<
          decltype(device_load), decltype(device_store), ComputeType,
          Algorithm::kSoftmax>(device, device_load, device_store, num_rows,
                               num_cols);
    }
  }
}

}  // namespace softmax

}  // end namespace impl

// Functor used by SoftmaxOp to do the computations.
template <typename GPUDevice, typename T>
struct SoftmaxFunctor {
  void operator()(const GPUDevice& device,
                  typename TTypes<T, 2UL>::ConstTensor logits, Tensor* softmax,
                  const bool log) {
    const T* inputs = logits.data();
    T* inputs_ptr = const_cast<T*>(inputs);
    T* outputs = softmax->flat<T>().data();

    const int32 num_cols = logits.dimension(1);
    const int32 num_rows = logits.dimension(0);

    using ComputeType = typename itex::DefaultComputeType<T>::type;

    itex::DirectLoad<T, ComputeType> device_load(inputs_ptr, num_cols);
    itex::DirectStore<ComputeType, T> device_store(outputs, num_cols);

    if (log) {
      auto status = impl::softmax::DispatchLogSoftmax<
          decltype(device_load), decltype(device_store), ComputeType>(
          device, device_load, device_store, num_rows, num_cols, log);
    } else {
      auto status =
          impl::softmax::DispatchSoftmax<decltype(device_load),
                                         decltype(device_store), ComputeType>(
              device, device_load, device_store, num_rows, num_cols, log);
    }
  }
};

template <typename GPUDevice, typename T>
struct AddV2WithSoftmaxFunctor {
  void operator()(const GPUDevice& device, const Tensor& logits_in_tensor,
                  const Tensor& adder_tensor, Tensor* softmax, const bool log) {
    auto logits_shape = logits_in_tensor.shape();

    T* inputs_ptr = const_cast<T*>(logits_in_tensor.flat<T>().data());
    T* adder_ptr = const_cast<T*>(adder_tensor.flat<T>().data());
    T* outputs_ptr = softmax->flat<T>().data();

    using ComputeType = typename itex::DefaultComputeType<T>::type;

    itex::SoftmaxInputShape input_dims;
    input_dims.batch_size = logits_shape.dim_size(0);
    input_dims.num_heads = logits_shape.dim_size(1);
    input_dims.seqlen_from = logits_shape.dim_size(2);
    input_dims.seqlen_to = logits_shape.dim_size(3);
    const int32 num_rows =
        input_dims.batch_size * input_dims.num_heads * input_dims.seqlen_from;
    const int32 num_cols = input_dims.seqlen_to;

    itex::AddMaskLoad<T, ComputeType> device_load(inputs_ptr, adder_ptr,
                                                  input_dims, num_cols);
    itex::DirectStore<ComputeType, T> device_store(outputs_ptr, num_cols);

    auto status =
        impl::softmax::DispatchSoftmax<decltype(device_load),
                                       decltype(device_store), ComputeType>(
            device, device_load, device_store, num_rows, num_cols, log);
  }
};

}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_SOFTMAX_OP_FUNCTOR_H_
