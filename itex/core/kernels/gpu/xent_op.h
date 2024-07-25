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

#ifndef ITEX_CORE_KERNELS_GPU_XENT_OP_H_
#define ITEX_CORE_KERNELS_GPU_XENT_OP_H_

#include <algorithm>

#include "itex/core/kernels/gpu/softmax_op_functor.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/hw_info.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

typedef Eigen::GpuDevice GPUDevice;

namespace impl {

namespace xent {

constexpr int kSubGroupSize = 32;

template <typename T>
struct SumOp {
  T operator()(const T& a, const T& b) const { return a + b; }
};

template <template <typename> typename ReductionOp, typename T,
          int workitem_group_width = kSubGroupSize>
T SubGroupAllReduce(const sycl::sub_group& sg, T val) {
  for (int mask = workitem_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(
        val, sycl::permute_group_by_xor(sg, val, sycl::id<1>(mask)));
  }
  return val;
}

template <typename T>
struct ComputeProbDiffKernel {
  ComputeProbDiffKernel(int32_t elems_cnt, T* labels_in_ptr, T* softmax_out_ptr,
                        T* backprop_out_ptr)
      :

        elems_cnt(elems_cnt),
        labels_in_ptr(labels_in_ptr),
        softmax_out_ptr(softmax_out_ptr),
        backprop_out_ptr(backprop_out_ptr) {}
  [[intel::reqd_sub_group_size(kSubGroupSize)]] void operator()(
      sycl::nd_item<1> id) const {
    for (int32_t i = id.get_global_id(0), step = id.get_global_range(0);
         i < elems_cnt; i += step) {
      backprop_out_ptr[i] = exp(softmax_out_ptr[i]) - labels_in_ptr[i];
    }
  }

 private:
  int32_t elems_cnt;
  T* labels_in_ptr;
  T* softmax_out_ptr;
  T* backprop_out_ptr;
};

template <typename T>
void ComputeProbDiffKernelImpl(const GPUDevice& device, const int32_t elems_cnt,
                               T* labels_in_ptr, T* softmax_out_ptr,
                               T* backprop_out_ptr, sycl::range<1> global_range,
                               sycl::range<1> local_range) {
  auto stream = device.stream();
  stream->submit([&](sycl::handler& h) {
    ComputeProbDiffKernel<T> task(elems_cnt, labels_in_ptr, softmax_out_ptr,
                                  backprop_out_ptr);
    h.parallel_for<ComputeProbDiffKernel<T>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
}

template <typename T>
void LaunchComputeProbDiffKernel(const GPUDevice& device,
                                 const int32_t num_instances,
                                 const int32_t num_classes, T* labels_in_ptr,
                                 T* softmax_out_ptr, T* backprop_out_ptr) {
  int workgroup_size =
      device.stream()
          ->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const int32_t elems_cnt = num_instances * num_classes;
  int num_wg = (elems_cnt + workgroup_size - 1) / workgroup_size;

  sycl::range<1> local_range(workgroup_size);
  sycl::range<1> global_range(num_wg * workgroup_size);
  ComputeProbDiffKernelImpl<T>(device, elems_cnt, labels_in_ptr,
                               softmax_out_ptr, backprop_out_ptr, global_range,
                               local_range);
}

template <typename T>
struct XentWorkItemKernel {
  XentWorkItemKernel(int32_t num_instances, int32_t num_classes,
                     T* labels_in_ptr, T* softmax_out_ptr, T* loss_out_ptr)
      : num_instances(num_instances),
        num_classes(num_classes),
        labels_in_ptr(labels_in_ptr),
        softmax_out_ptr(softmax_out_ptr),
        loss_out_ptr(loss_out_ptr) {}
  [[intel::reqd_sub_group_size(kSubGroupSize)]] void operator()(
      sycl::nd_item<1> id) const {
    for (int32_t row = id.get_global_id(0), step = id.get_global_range(0);
         row < num_instances; row += step) {
      const int row_offset = row * num_classes;
      const T* in_row = softmax_out_ptr + row_offset;
      const T* label_row = labels_in_ptr + row_offset;
      T result = (T)0.f;
#pragma unroll
      for (int col = 0; col < num_classes; col += 1) {
        T label = label_row[col];
        T prob = in_row[col];
        result += -label * (prob);
      }
      loss_out_ptr[row] = result;
    }
  }

 private:
  int32_t num_instances;
  const int32_t num_classes;
  T* labels_in_ptr;
  T* softmax_out_ptr;
  T* loss_out_ptr;
};

template <typename T>
inline Status XentWorkItemImpl(const GPUDevice& device,
                               const int32_t num_instances,
                               const int32_t num_classes, T* labels_in_ptr,
                               T* softmax_out_ptr, T* loss_out_ptr,
                               sycl::range<1> global_range,
                               sycl::range<1> local_range) {
  auto stream = device.stream();
  stream->submit([&](sycl::handler& h) {
    XentWorkItemKernel<T> task(num_instances, num_classes, labels_in_ptr,
                               softmax_out_ptr, loss_out_ptr);
    h.parallel_for<XentWorkItemKernel<T>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

template <typename T>
inline Status LaunchXentWorkItemImpl(const GPUDevice& device,
                                     const int32_t num_instances,
                                     const int32_t num_classes,
                                     T* labels_in_ptr, T* softmax_out_ptr,
                                     T* loss_out_ptr) {
  const int workgroup_size =
      device.stream()
          ->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const int num_wg = (num_instances + workgroup_size - 1) / workgroup_size;
  const sycl::range<1> local_range(workgroup_size);
  sycl::range<1> global_range(num_wg * workgroup_size);

  return XentWorkItemImpl<T>(device, num_instances, num_classes, labels_in_ptr,
                             softmax_out_ptr, loss_out_ptr, global_range,
                             local_range);
}

template <typename T>
inline Status DispatchXentWorkItemImpl(const GPUDevice& device,
                                       const int32_t num_instances,
                                       const int32_t num_classes,
                                       T* labels_in_ptr, T* softmax_out_ptr,
                                       T* loss_out_ptr) {
  return LaunchXentWorkItemImpl<T>(device, num_instances, num_classes,
                                   labels_in_ptr, softmax_out_ptr,
                                   loss_out_ptr);
}

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType, int pack_size, int cols_per_workitem,
          int workitem_group_width, int rows_per_access, bool padding>
struct XentSubGroupKernel {
  XentSubGroupKernel(LOAD device_load, LABEL_LOAD device_label_load,
                     STORE device_store, int64_t rows, int64_t cols)
      : device_load(device_load),
        device_label_load(device_label_load),
        device_store(device_store),
        rows(rows),
        cols(cols) {}
  [[intel::reqd_sub_group_size(kSubGroupSize)]] void operator()(
      sycl::nd_item<2> id) const {
    constexpr int num_packs = cols_per_workitem / pack_size;
    const int workgroup_idx = id.get_group(1);
    const int localrange_y = id.get_local_range(0);
    const int workitem_y = id.get_local_id(0);
    const int grouprange_y = id.get_group_range(1);
    const int lane_id = id.get_local_id(1);

    sycl::sub_group sg = id.get_sub_group();

    const int global_workitem_group_id =
        workgroup_idx * localrange_y + workitem_y;
    const int num_global_workitem_group = grouprange_y * localrange_y;

    ComputeType x_buf[rows_per_access][cols_per_workitem];
    ComputeType y_buf[rows_per_access][cols_per_workitem];

    for (int64_t row = global_workitem_group_id * rows_per_access; row < rows;
         row += num_global_workitem_group * rows_per_access) {
      ComputeType workitem_sum[rows_per_access];  // NOLINT(runtime/arrays)
#pragma unroll
      for (int row_id = 0; row_id < rows_per_access; ++row_id) {
        workitem_sum[row_id] = 0;
        ComputeType* row_x_buf = x_buf[row_id];
        ComputeType* row_y_buf = y_buf[row_id];
#pragma unroll
        for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
          const int col =
              (pack_id * workitem_group_width + lane_id) * pack_size;
          if (!padding || col < cols) {
            device_load.template Load<pack_size>(
                row_x_buf + pack_id * pack_size, row + row_id, col);
            device_label_load.template Load<pack_size>(
                row_y_buf + pack_id * pack_size, row + row_id, col);
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
              workitem_sum[row_id] += (-row_y_buf[pack_id * pack_size + i]) *
                                      (row_x_buf[pack_id * pack_size + i]);
            }
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
        device_store.template Store<rows_per_access>(
            warp_sum, row + row_id * rows_per_access, 0);
      }
    }
  }

 private:
  LOAD device_load;
  LABEL_LOAD device_label_load;
  STORE device_store;
  const int64_t rows;
  const int64_t cols;
};

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType, int pack_size, int cols_per_workitem,
          int workitem_group_width, int rows_per_access, bool padding>
Status XentSubGroupImpl(const GPUDevice& device, LOAD device_load,
                        LABEL_LOAD device_label_load, STORE device_store,
                        const int64_t rows, const int64_t cols,
                        sycl::range<2> global_range,
                        sycl::range<2> local_range) {
  auto stream = device.stream();
  static_assert(cols_per_workitem % pack_size == 0, "");
  static_assert(workitem_group_width <= kSubGroupSize, "");
  static_assert(kSubGroupSize % workitem_group_width == 0, "");
  assert(cols <= cols_per_workitem * workitem_group_width);

  stream->submit([&](sycl::handler& h) {
    XentSubGroupKernel<LOAD, LABEL_LOAD, STORE, ComputeType, pack_size,
                       cols_per_workitem, workitem_group_width, rows_per_access,
                       padding>
        task(device_load, device_label_load, device_store, rows, cols);
    h.parallel_for<XentSubGroupKernel<
        LOAD, LABEL_LOAD, STORE, ComputeType, pack_size, cols_per_workitem,
        workitem_group_width, rows_per_access, padding>>(
        sycl::nd_range<2>(global_range, local_range), task);
  });

  return Status::OK();
}

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType, int pack_size, int cols_per_workitem,
          int workitem_group_width, int rows_per_access, bool padding>
inline Status LaunchXentSubGroupImpl(const GPUDevice& device, LOAD device_load,
                                     LABEL_LOAD device_label_load,
                                     STORE device_store, const int32_t rows,
                                     const int32_t cols) {
  constexpr int workgroup_size = 128;
  static_assert(workgroup_size % workitem_group_width == 0, "");
  constexpr int rows_per_block = workgroup_size / workitem_group_width;
  sycl::range<2> local_range(rows_per_block, workitem_group_width);

  const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
  sycl::range<2> global_range(rows_per_block,
                              num_blocks * workitem_group_width);

  auto status =
      XentSubGroupImpl<LOAD, LABEL_LOAD, STORE, ComputeType, pack_size,
                       cols_per_workitem, workitem_group_width, rows_per_access,
                       padding>(device, device_load, device_label_load,
                                device_store, rows, cols, global_range,
                                local_range);

  return status;
}

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType, int pack_size, int cols_per_workitem,
          int workitem_group_width, int rows_per_access>
inline Status DispatchXentSubGroupImplPadding(
    const GPUDevice& device, LOAD device_load, LABEL_LOAD device_label_load,
    STORE device_store, const int32_t rows, const int32_t cols) {
  if (cols == cols_per_workitem * workitem_group_width) {
    return LaunchXentSubGroupImpl<LOAD, LABEL_LOAD, STORE, ComputeType,
                                  pack_size, cols_per_workitem,
                                  workitem_group_width, rows_per_access, false>(
        device, device_load, device_label_load, device_store, rows, cols);
  } else {
    return LaunchXentSubGroupImpl<LOAD, LABEL_LOAD, STORE, ComputeType,
                                  pack_size, cols_per_workitem,
                                  workitem_group_width, rows_per_access, true>(
        device, device_load, device_label_load, device_store, rows, cols);
  }
}

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, Status>::type
DispatchXentSubGroupImplCols(const GPUDevice& device, LOAD device_load,
                             LABEL_LOAD device_label_load, STORE device_store,
                             const int32_t rows, const int32_t cols) {
  if (cols <= 0) {
    return Status(TF_INVALID_ARGUMENT, "Invalid Value");
  }
#define DEFINE_ONE_ELIF(workitem_group_width)                              \
  if (cols <= (workitem_group_width)*pack_size) {                          \
    {                                                                      \
      return DispatchXentSubGroupImplPadding<                              \
          LOAD, LABEL_LOAD, STORE, ComputeType, pack_size, pack_size,      \
          workitem_group_width, 1>(device, device_load, device_label_load, \
                                   device_store, rows, cols);              \
    }                                                                      \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                               \
  if (cols <= (col)*kSubGroupSize) {                                       \
    return DispatchXentSubGroupImplPadding<LOAD, LABEL_LOAD, STORE,        \
                                           ComputeType, pack_size, col,    \
                                           kSubGroupSize, 1>(              \
        device, device_load, device_label_load, device_store, rows, cols); \
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

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, Status>::type
DispatchXentSubGroupImplCols(const GPUDevice& device, LOAD device_load,
                             LABEL_LOAD device_label_load, STORE device_store,
                             const int32_t rows, const int32_t cols) {
  if (cols <= 0) {
    return Status(TF_INVALID_ARGUMENT, "Invalid Value");
  }
#define DEFINE_ONE_ELIF(workitem_group_width)                              \
  if (cols <= (workitem_group_width)*pack_size) {                          \
    {                                                                      \
      return DispatchXentSubGroupImplPadding<                              \
          LOAD, LABEL_LOAD, STORE, ComputeType, pack_size, pack_size,      \
          workitem_group_width, 1>(device, device_load, device_label_load, \
                                   device_store, rows, cols);              \
    }                                                                      \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                               \
  if (cols <= (col)*kSubGroupSize) {                                       \
    return DispatchXentSubGroupImplPadding<LOAD, LABEL_LOAD, STORE,        \
                                           ComputeType, pack_size, col,    \
                                           kSubGroupSize, 1>(              \
        device, device_load, device_label_load, device_store, rows, cols); \
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

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType>
struct DispatchXentSubGroupImplPackSize {
  Status operator()(const GPUDevice& device, LOAD device_load,
                    LABEL_LOAD device_label_load, STORE device_store,
                    const int32_t rows, const int32_t cols) {
    if (cols % 2 == 0) {
      return DispatchXentSubGroupImplCols<LOAD, LABEL_LOAD, STORE, ComputeType,
                                          2>(
          device, device_load, device_label_load, device_store, rows, cols);
    } else {
      return DispatchXentSubGroupImplCols<LOAD, LABEL_LOAD, STORE, ComputeType,
                                          1>(
          device, device_load, device_label_load, device_store, rows, cols);
    }
  }
};

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType>
inline Status DispatchXentSubGroupImpl(const GPUDevice& device,
                                       LOAD device_load,
                                       LABEL_LOAD device_label_load,
                                       STORE device_store, const int32_t rows,
                                       const int32_t cols) {
  return DispatchXentSubGroupImplPackSize<LOAD, LABEL_LOAD, STORE,
                                          ComputeType>()(
      device, device_load, device_label_load, device_store, rows, cols);
}

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType, int pack_size, int wg_array_size>
struct XentWorkGroupKernel {
  XentWorkGroupKernel(LOAD device_load, LABEL_LOAD device_label_load,
                      STORE device_store, const int32_t rows,
                      const int32_t cols)
      : device_load(device_load),
        device_label_load(device_label_load),
        device_store(device_store),
        rows(rows),
        cols(cols) {}
  [[intel::reqd_sub_group_size(kSubGroupSize)]] void operator()(
      sycl::nd_item<1> id) const {
    ComputeType block_sum[wg_array_size];  // NOLINT(runtime/arrays)
    const int local_id = id.get_local_id(0);
    const int num_packs = cols / pack_size;
    for (int64_t row_id = id.get_group(0); row_id < rows;
         row_id += id.get_group_range(0)) {
      ComputeType workitem_sum = 0;
      for (int pack_id = local_id; pack_id < num_packs;
           pack_id += id.get_local_range(0)) {
        ComputeType x_pack[pack_size];  // NOLINT(runtime/arrays)
        device_load.template Load<pack_size>(x_pack, row_id,
                                             pack_id * pack_size);
        ComputeType y_pack[pack_size];  // NOLINT(runtime/arrays)
        device_label_load.template Load<pack_size>(y_pack, row_id,
                                                   pack_id * pack_size);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          workitem_sum += (-y_pack[i]) * (x_pack[i]);
        }
      }
      block_sum[row_id] = sycl::reduce_over_group(
          id.get_group(), workitem_sum, sycl::ext::oneapi::plus<ComputeType>());

      device_store.template Store<pack_size>(block_sum, row_id, 0);
    }
  }

 private:
  LOAD device_load;
  LABEL_LOAD device_label_load;
  STORE device_store;
  const int32_t rows;
  const int32_t cols;
};

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType, int pack_size, int wg_array_size>
inline Status XentWorkGroupImpl(const GPUDevice& device, LOAD device_load,
                                LABEL_LOAD device_label_load,
                                STORE device_store, const int32_t rows,
                                const int32_t cols, sycl::range<1> global_range,
                                sycl::range<1> local_range) {
  auto stream = device.stream();
  assert(cols % pack_size == 0);
  stream->submit([&](sycl::handler& h) {
    XentWorkGroupKernel<LOAD, LABEL_LOAD, STORE, ComputeType, pack_size,
                        wg_array_size>
        task(device_load, device_label_load, device_store, rows, cols);
    h.parallel_for<XentWorkGroupKernel<LOAD, LABEL_LOAD, STORE, ComputeType,
                                       pack_size, wg_array_size>>(
        sycl::nd_range<1>(global_range, local_range), task);
  });
  return Status::OK();
}

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType, int pack_size>
inline Status LaunchXentWorkGroupImpl(const GPUDevice& device, LOAD device_load,
                                      LABEL_LOAD device_label_load,
                                      STORE device_store, const int32_t rows,
                                      const int32_t cols) {
  const int workgroup_size = 128;
  sycl::range<1> local_range(workgroup_size);
  int num_wg;
  GetNumWorkGroups(device.stream()->get_device(), workgroup_size, rows, 32,
                   &num_wg);
  sycl::range<1> global_range(num_wg * workgroup_size);
  //  wg_array_size is equal to rows/id.get_global_ranges(0),
  //  in this pass, the max value of wg_array_size is 1
  return XentWorkGroupImpl<LOAD, LABEL_LOAD, STORE, ComputeType, pack_size, 1>(
      device, device_load, device_label_load, device_store, rows, cols,
      global_range, local_range);
}

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType>
struct DispatchXentWorkGroupImplPackSize {
  Status operator()(const GPUDevice& device, LOAD device_load,
                    LABEL_LOAD device_label_load, STORE device_store,
                    const int32_t rows, const int32_t cols) {
    if (cols % 2 == 0) {
      return LaunchXentWorkGroupImpl<LOAD, LABEL_LOAD, STORE, ComputeType, 2>(
          device, device_load, device_label_load, device_store, rows, cols);
    } else {
      return LaunchXentWorkGroupImpl<LOAD, LABEL_LOAD, STORE, ComputeType, 1>(
          device, device_load, device_label_load, device_store, rows, cols);
    }
  }
};

template <typename LOAD, typename LABEL_LOAD, typename STORE,
          typename ComputeType>
inline Status DispatchXentWorkGroupImpl(const GPUDevice& device,
                                        LOAD device_load,
                                        LABEL_LOAD device_label_load,
                                        STORE device_store, const int32_t rows,
                                        const int32_t cols) {
  return DispatchXentWorkGroupImplPackSize<LOAD, LABEL_LOAD, STORE,
                                           ComputeType>()(
      device, device_load, device_label_load, device_store, rows, cols);
}

template <typename T>
Status DispatchXent(const GPUDevice& device, const int32_t num_instances,
                    const int32_t num_classes, T* labels_in_ptr,
                    T* softmax_out_ptr, T* loss_out_ptr, T* backprop_out_ptr) {
  LaunchComputeProbDiffKernel<T>(device, num_instances, num_classes,
                                 labels_in_ptr, softmax_out_ptr,
                                 backprop_out_ptr);

  const int32_t MaxCompUnits =
      device.stream()
          ->get_device()
          .template get_info<sycl::info::device::max_compute_units>();
  const int32_t MaxGroupSize =
      device.stream()
          ->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();

  if (num_instances >= MaxCompUnits * MaxGroupSize) {
    //  the num of rows is too large, use one workitem to process one row
    return DispatchXentWorkItemImpl<T>(device, num_instances, num_classes,
                                       labels_in_ptr, softmax_out_ptr,
                                       loss_out_ptr);
  } else {
    using ComputeType = typename itex::DefaultComputeType<T>::type;
    itex::DirectLoad<T, ComputeType> device_load(
        softmax_out_ptr, num_classes);  // [num_instances, num_classes]
    itex::DirectLoad<T, ComputeType> device_label_load(
        labels_in_ptr, num_classes);  // [num_instances, num_classes]
    itex::DirectStore<ComputeType, T> device_store(loss_out_ptr,
                                                   1);  // [num_instances]
    auto sycl_device = device.stream()->get_device();
    if (IsXeHPC(&sycl_device) && num_classes <= 1024) {
      //  use one subgroup to process one row
      return DispatchXentSubGroupImpl<decltype(device_load),
                                      decltype(device_label_load),
                                      decltype(device_store), ComputeType>(
          device, device_load, device_label_load, device_store, num_instances,
          num_classes);
    } else {
      //  the num of cols>1024, use one workgroup to process one row
      return DispatchXentWorkGroupImpl<decltype(device_load),
                                       decltype(device_label_load),
                                       decltype(device_store), ComputeType>(
          device, device_load, device_label_load, device_store, num_instances,
          num_classes);
    }
  }
}

}  // namespace xent

}  // namespace impl

namespace functor {

template <typename GPUDevice, typename T>
struct XentFunctor {
  Status operator()(const GPUDevice& device, const Tensor& logits_in,
                    const Tensor& labels_in, Tensor softmax_temp,
                    Tensor* loss_out, Tensor* back_out) {
    const int32 num_instances = logits_in.shape().dim_size(0);
    const int32 num_classes = logits_in.shape().dim_size(1);

    T* logits_in_ptr = const_cast<T*>(logits_in.flat<T>().data());
    T* labels_in_ptr = const_cast<T*>(labels_in.flat<T>().data());

    T* softmax_out_ptr = softmax_temp.flat<T>().data();
    T* loss_out_ptr = loss_out->flat<T>().data();
    T* back_out_ptr = back_out->flat<T>().data();

    using ComputeType = typename itex::DefaultComputeType<T>::type;

    itex::DirectLoad<T, ComputeType> device_load(logits_in_ptr, num_classes);
    itex::DirectStore<ComputeType, T> device_store(softmax_out_ptr,
                                                   num_classes);

    auto softmax_status =
        impl::softmax::DispatchLogSoftmax<decltype(device_load),
                                          decltype(device_store), ComputeType>(
            device, device_load, device_store, num_instances, num_classes,
            true);
    assert(softmax_status == Status::OK());
    Status runSuccess_xent = impl::xent::DispatchXent<T>(
        device, num_instances, num_classes, labels_in_ptr, softmax_out_ptr,
        loss_out_ptr, back_out_ptr);
    return runSuccess_xent;
  }
};

template <typename GPUDevice, typename T>
struct XentFunctorWithEigen {
  Status operator()(const GPUDevice& g_device,
                    const Eigen::array<Eigen::DenseIndex, 2>& logits_bcast,
                    const Eigen::array<Eigen::DenseIndex, 2>& labels_bcast,
                    typename TTypes<T>::ConstMatrix logits_in,
                    typename TTypes<T>::ConstMatrix labels_in,
                    typename TTypes<T>::Matrix logits_in_b,
                    typename TTypes<T>::Matrix labels_in_b, Tensor softmax_temp,
                    Tensor* loss_out, Tensor* back_out) {
    // one of two inputs is 1D tensor, so do bcast
    logits_in_b.device(g_device) = logits_in.broadcast(logits_bcast);
    labels_in_b.device(g_device) = labels_in.broadcast(labels_bcast);

    const int32 num_instances = logits_in_b.dimension(0);
    const int32 num_classes = logits_in_b.dimension(1);

    T* logits_in_ptr = const_cast<T*>(logits_in_b.data());
    T* labels_in_ptr = const_cast<T*>(labels_in_b.data());

    T* softmax_out_ptr = softmax_temp.flat<T>().data();
    T* loss_out_ptr = loss_out->flat<T>().data();
    T* back_out_ptr = back_out->flat<T>().data();

    using ComputeType = typename itex::DefaultComputeType<T>::type;

    itex::DirectLoad<T, ComputeType> device_load(logits_in_ptr, num_classes);
    itex::DirectStore<ComputeType, T> device_store(softmax_out_ptr,
                                                   num_classes);

    auto softmax_status =
        impl::softmax::DispatchLogSoftmax<decltype(device_load),
                                          decltype(device_store), ComputeType>(
            g_device, device_load, device_store, num_instances, num_classes,
            true);

    assert(softmax_status == Status::OK());
    Status runSuccess_xent = impl::xent::DispatchXent<T>(
        g_device, num_instances, num_classes, labels_in_ptr, softmax_out_ptr,
        loss_out_ptr, back_out_ptr);
    return runSuccess_xent;
  }
};

}  // namespace functor

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_XENT_OP_H_
