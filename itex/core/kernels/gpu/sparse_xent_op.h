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

#ifndef ITEX_CORE_KERNELS_GPU_SPARSE_XENT_OP_H_
#define ITEX_CORE_KERNELS_GPU_SPARSE_XENT_OP_H_

#include <algorithm>

#include "itex/core/kernels/gpu/softmax_op_functor.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/hw_info.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

namespace impl {

namespace sparse_xent {

constexpr int kSubGroupSize = 32;

template <typename T>
struct SumOp {
  T operator()(const T& a, const T& b) const { return a + b; }
};

template <template <typename> typename ReductionOp, typename T,
          int workitem_group_width = kSubGroupSize>
T SubGroupAllReduce(const sycl::sub_group& sg, T val) {
  for (int mask = workitem_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, sg.shuffle_xor(val, sycl::id<1>(mask)));
  }
  return val;
}

template <typename T, typename IndexType>
struct ComputeProbDiffKernel {
  ComputeProbDiffKernel(int32_t num_instances, int32_t num_classes,
                        IndexType* labels_ptr, T* softmax_out_ptr,
                        T* backprop_out_ptr)
      : labels_ptr_(labels_ptr),
        softmax_out_ptr_(softmax_out_ptr),
        backprop_out_ptr_(backprop_out_ptr),
        num_instances_(num_instances),
        num_classes_(num_classes) {}

  [[intel::reqd_sub_group_size(kSubGroupSize)]] void operator()(
      sycl::nd_item<1> id) const {
    const int32_t elems_cnt = num_instances_ * num_classes_;
    for (int32_t i = id.get_global_id(0), step = id.get_global_range(0);
         i < elems_cnt; i += step) {
      const int32_t row_id = i / num_classes_;
      const int32_t col_id = i % num_classes_;
      T subtract = (labels_ptr_[row_id] == col_id) ? T(1.0) : T(0.0);
      backprop_out_ptr_[i] = exp(softmax_out_ptr_[i]) - subtract;
    }
  }

 private:
  IndexType* labels_ptr_;
  T* softmax_out_ptr_;
  T* backprop_out_ptr_;
  int32_t num_instances_;
  int32_t num_classes_;
};

template <typename T, typename IndexType>
void LaunchComputeProbDiffKernel(sycl::queue* stream,
                                 const int32_t num_instances,
                                 const int32_t num_classes,
                                 IndexType* labels_in_ptr, T* softmax_out_ptr,
                                 T* backprop_out_ptr) {
  int workgroup_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const int32_t elems_cnt = num_instances * num_classes;
  int num_wg = (elems_cnt + workgroup_size - 1) / workgroup_size;

  sycl::range<1> local_range(workgroup_size);
  sycl::range<1> global_range(num_wg * workgroup_size);
  ComputeProbDiffKernel<T, IndexType> prop_task(num_instances, num_classes,
                                                labels_in_ptr, softmax_out_ptr,
                                                backprop_out_ptr);

  stream->submit([&](sycl::handler& h) {
    h.parallel_for<ComputeProbDiffKernel<T, IndexType>>(
        sycl::nd_range<1>(global_range, local_range), prop_task);
  });
  return;
}

template <typename T, typename IndexType>
class SparseXentWorkItemKernel {
 public:
  SparseXentWorkItemKernel(int32_t num_instances, int32_t num_classes,
                           IndexType* labels_in_ptr, T* softmax_out_ptr,
                           T* loss_out_ptr)
      : num_instances_(num_instances),
        num_classes_(num_classes),
        labels_in_ptr_(labels_in_ptr),
        softmax_out_ptr_(softmax_out_ptr),
        loss_out_ptr_(loss_out_ptr) {}
  [[intel::reqd_sub_group_size(kSubGroupSize)]] void operator()(
      sycl::nd_item<1> id) const {
    for (int32_t row = id.get_global_id(0), step = id.get_global_range(0);
         row < num_instances_; row += step) {
      const int row_offset = row * num_classes_;
      const T* in_row = softmax_out_ptr_ + row_offset;
      T result = (T)0.0f;
#pragma unroll
      for (int col = 0; col < num_classes_; col += 1) {
        IndexType label = (labels_in_ptr_[row] == col) ? 1 : 0;
        T prob = in_row[col];
        result += (T)label * (-prob);
      }
      loss_out_ptr_[row] = result;
    }
  }

 private:
  const int32_t num_instances_;
  const int32_t num_classes_;
  IndexType* labels_in_ptr_;
  T* softmax_out_ptr_;
  T* loss_out_ptr_;
};

template <typename T, typename IndexType>
inline Status SparseXentWorkItemImpl(
    sycl::queue* stream, const int32_t num_instances, const int32_t num_classes,
    IndexType* labels_in_ptr, T* softmax_out_ptr, T* loss_out_ptr,
    sycl::range<1> global_range, sycl::range<1> local_range) {
  stream->submit([&](sycl::handler& h) {
    SparseXentWorkItemKernel<T, IndexType> workitem_kernel(
        num_instances, num_classes, labels_in_ptr, softmax_out_ptr,
        loss_out_ptr);
    h.parallel_for<SparseXentWorkItemKernel<T, IndexType>>(
        sycl::nd_range<1>(global_range, local_range), workitem_kernel);
  });
  return Status::OK();
}

template <typename T, typename IndexType>
inline Status LaunchSparseXentWorkItemImpl(
    sycl::queue* stream, const int32_t num_instances, const int32_t num_classes,
    IndexType* labels_in_ptr, T* softmax_out_ptr, T* loss_out_ptr) {
  const int workgroup_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const int num_wg = (num_instances + workgroup_size - 1) / workgroup_size;
  sycl::range<1> local_range(workgroup_size);
  sycl::range<1> global_range(num_wg * workgroup_size);

  return SparseXentWorkItemImpl<T, IndexType>(
      stream, num_instances, num_classes, labels_in_ptr, softmax_out_ptr,
      loss_out_ptr, global_range, local_range);
}

template <typename T, typename IndexType>
inline Status DispatchSparseXentWorkItemImpl(
    sycl::queue* stream, const int32_t num_instances, const int32_t num_classes,
    IndexType* labels_in_ptr, T* softmax_out_ptr, T* loss_out_ptr) {
  return LaunchSparseXentWorkItemImpl<T, IndexType>(
      stream, num_instances, num_classes, labels_in_ptr, softmax_out_ptr,
      loss_out_ptr);
}

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType, int pack_size, int cols_per_workitem,
          int workitem_group_width, int rows_per_access, bool padding>
class SparseXentSubGroupKernel {
 public:
  SparseXentSubGroupKernel(LOAD device_load, IndexType* labels_in_ptr,
                           STORE device_store, int32_t rows, int32_t cols,
                           int32_t num_packs)
      : device_load_(device_load),
        labels_in_ptr_(labels_in_ptr),
        device_store_(device_store),
        rows_(rows),
        cols_(cols),
        num_packs_(num_packs) {}
  [[intel::reqd_sub_group_size(kSubGroupSize)]] void operator()(
      sycl::nd_item<2> id) const {
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
    for (int32_t row = global_workitem_group_id * rows_per_access; row < rows_;
         row += num_global_workitem_group * rows_per_access) {
      ComputeType workitem_sum[rows_per_access];  // NOLINT(runtime/arrays)
#pragma unroll
      for (int row_id = 0; row_id < rows_per_access; ++row_id) {
        workitem_sum[row_id] = 0;
        ComputeType* row_x_buf = x_buf[row_id];
#pragma unroll
        for (int pack_id = 0; pack_id < num_packs_; ++pack_id) {
          const int col =
              (pack_id * workitem_group_width + lane_id) * pack_size;
          if (!padding || col < cols_) {
            device_load_.template Load<pack_size>(
                row_x_buf + pack_id * pack_size, row + row_id, col);

            for (int i = 0; i < pack_size; ++i) {
              ComputeType label =
                  (labels_in_ptr_[row + row_id * rows_per_access] == col + i)
                      ? 1
                      : 0;
              workitem_sum[row_id] +=
                  label * (-row_x_buf[pack_id * pack_size + i]);
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
        device_store_.template Store<rows_per_access>(
            warp_sum, row + row_id * rows_per_access, 0);
      }
    }
  }

 private:
  LOAD device_load_;
  IndexType* labels_in_ptr_;
  STORE device_store_;
  const int32_t rows_;
  const int32_t cols_;
  const int32_t num_packs_;
};

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType, int pack_size, int cols_per_workitem,
          int workitem_group_width, int rows_per_access, bool padding>
Status SparseXentSubGroupImpl(sycl::queue* stream, LOAD device_load,
                              IndexType* labels_in_ptr, STORE device_store,
                              const int32_t rows, const int32_t cols,
                              sycl::range<2> global_range,
                              sycl::range<2> local_range) {
  static_assert(cols_per_workitem % pack_size == 0, "");
  static_assert(workitem_group_width <= kSubGroupSize, "");
  static_assert(kSubGroupSize % workitem_group_width == 0, "");
  int32_t num_packs = cols_per_workitem / pack_size;
  assert(cols <= cols_per_workitem * workitem_group_width);

  stream->submit([&](sycl::handler& h) {
    SparseXentSubGroupKernel<LOAD, STORE, ComputeType, IndexType, pack_size,
                             cols_per_workitem, workitem_group_width,
                             rows_per_access, padding>
        subgroup_kernel(device_load, labels_in_ptr, device_store, rows, cols,
                        num_packs);
    h.parallel_for<SparseXentSubGroupKernel<
        LOAD, STORE, ComputeType, IndexType, pack_size, cols_per_workitem,
        workitem_group_width, rows_per_access, padding>>(
        sycl::nd_range<2>(global_range, local_range), subgroup_kernel);
  });

  return Status::OK();
}

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType, int pack_size, int cols_per_workitem,
          int workitem_group_width, int rows_per_access, bool padding>
inline Status LaunchSparseXentSubGroupImpl(
    sycl::queue* stream, LOAD device_load, IndexType* labels_in_ptr,
    STORE device_store, const int32_t rows, const int32_t cols) {
  constexpr int workgroup_size = 128;
  static_assert(workgroup_size % workitem_group_width == 0, "");
  constexpr int rows_per_block = workgroup_size / workitem_group_width;
  sycl::range<2> local_range(rows_per_block, workitem_group_width);

  const int32_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
  sycl::range<2> global_range(rows_per_block,
                              num_blocks * workitem_group_width);

  auto status =
      SparseXentSubGroupImpl<LOAD, STORE, ComputeType, IndexType, pack_size,
                             cols_per_workitem, workitem_group_width,
                             rows_per_access, padding>(
          stream, device_load, labels_in_ptr, device_store, rows, cols,
          global_range, local_range);

  return status;
}

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType, int pack_size, int cols_per_workitem,
          int workitem_group_width, int rows_per_access>
inline Status DispatchSparseXentSubGroupImplPadding(
    sycl::queue* stream, LOAD device_load, IndexType* labels_in_ptr,
    STORE device_store, const int32_t rows, const int32_t cols) {
  if (cols == cols_per_workitem * workitem_group_width) {
    return LaunchSparseXentSubGroupImpl<
        LOAD, STORE, ComputeType, IndexType, pack_size, cols_per_workitem,
        workitem_group_width, rows_per_access, false>(
        stream, device_load, labels_in_ptr, device_store, rows, cols);
  } else {
    return LaunchSparseXentSubGroupImpl<
        LOAD, STORE, ComputeType, IndexType, pack_size, cols_per_workitem,
        workitem_group_width, rows_per_access, true>(
        stream, device_load, labels_in_ptr, device_store, rows, cols);
  }
}

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType, int pack_size>
typename std::enable_if<pack_size == 1, Status>::type
DispatchSparseXentSubGroupImplCols(sycl::queue* stream, LOAD device_load,
                                   IndexType* labels_in_ptr, STORE device_store,
                                   const int32_t rows, const int32_t cols) {
  if (cols <= 0) {
    return Status(TF_INVALID_ARGUMENT, "Invalid Value");
  }
#define DEFINE_ONE_ELIF(workitem_group_width)                          \
  if (cols <= (workitem_group_width)*pack_size) {                      \
    {                                                                  \
      return DispatchSparseXentSubGroupImplPadding<                    \
          LOAD, STORE, ComputeType, IndexType, pack_size, pack_size,   \
          workitem_group_width, 1>(stream, device_load, labels_in_ptr, \
                                   device_store, rows, cols);          \
    }                                                                  \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                \
  if (cols <= (col)*kSubGroupSize) {                                        \
    return DispatchSparseXentSubGroupImplPadding<LOAD, STORE, ComputeType,  \
                                                 IndexType, pack_size, col, \
                                                 kSubGroupSize, 1>(         \
        stream, device_load, labels_in_ptr, device_store, rows, cols);      \
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

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType, int pack_size>
typename std::enable_if<pack_size == 2, Status>::type
DispatchSparseXentSubGroupImplCols(sycl::queue* stream, LOAD device_load,
                                   IndexType* labels_in_ptr, STORE device_store,
                                   const int32_t rows, const int32_t cols) {
  if (cols <= 0) {
    return Status(TF_INVALID_ARGUMENT, "Invalid Value");
  }
#define DEFINE_ONE_ELIF(workitem_group_width)                          \
  if (cols <= (workitem_group_width)*pack_size) {                      \
    {                                                                  \
      return DispatchSparseXentSubGroupImplPadding<                    \
          LOAD, STORE, ComputeType, IndexType, pack_size, pack_size,   \
          workitem_group_width, 1>(stream, device_load, labels_in_ptr, \
                                   device_store, rows, cols);          \
    }                                                                  \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                \
  if (cols <= (col)*kSubGroupSize) {                                        \
    return DispatchSparseXentSubGroupImplPadding<LOAD, STORE, ComputeType,  \
                                                 IndexType, pack_size, col, \
                                                 kSubGroupSize, 1>(         \
        stream, device_load, labels_in_ptr, device_store, rows, cols);      \
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
          typename IndexType>
struct DispatchSparseXentSubGroupImplPackSize {
  Status operator()(sycl::queue* stream, LOAD device_load,
                    IndexType* labels_in_ptr, STORE device_store,
                    const int32_t rows, const int32_t cols) {
    if (cols % 2 == 0) {
      return DispatchSparseXentSubGroupImplCols<LOAD, STORE, ComputeType,
                                                IndexType, 2>(
          stream, device_load, labels_in_ptr, device_store, rows, cols);
    } else {
      return DispatchSparseXentSubGroupImplCols<LOAD, STORE, ComputeType,
                                                IndexType, 1>(
          stream, device_load, labels_in_ptr, device_store, rows, cols);
    }
  }
};

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType>
inline Status DispatchSparseXentSubGroupImpl(
    sycl::queue* stream, LOAD device_load, IndexType* labels_in_ptr,
    STORE device_store, const int32_t rows, const int32_t cols) {
  return DispatchSparseXentSubGroupImplPackSize<LOAD, STORE, ComputeType,
                                                IndexType>()(
      stream, device_load, labels_in_ptr, device_store, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType, int pack_size, int wg_array_size>
class SparseXentWorkGroupKernel {
 public:
  SparseXentWorkGroupKernel(LOAD device_load, IndexType* labels_in_ptr,
                            STORE device_store, int32_t num_packs, int32_t rows)
      : device_load_(device_load),
        labels_in_ptr_(labels_in_ptr),
        device_store_(device_store),
        num_packs_(num_packs),
        rows_(rows) {}
  void operator()(sycl::nd_item<1> id) const {
    ComputeType block_sum[wg_array_size];  // NOLINT(runtime/arrays)
    const int local_id = id.get_local_id(0);
    for (int32_t row_id = id.get_group(0); row_id < rows_;
         row_id += id.get_group_range(0)) {
      ComputeType workitem_sum = 0;
      for (int pack_id = local_id; pack_id < num_packs_;
           pack_id += id.get_local_range(0)) {
        ComputeType x_pack[pack_size];  // NOLINT(runtime/arrays)
        device_load_.template Load<pack_size>(x_pack, row_id,
                                              pack_id * pack_size);

        for (int i = 0; i < pack_size; ++i) {
          ComputeType label =
              (labels_in_ptr_[row_id] == (i + pack_id * pack_size)) ? 1 : 0;
          workitem_sum += label * (-x_pack[i]);
        }
      }
      block_sum[row_id] = sycl::reduce_over_group(
          id.get_group(), workitem_sum, sycl::ext::oneapi::plus<ComputeType>());

      device_store_.template Store<pack_size>(block_sum, row_id, 0);
    }
  }

 private:
  LOAD device_load_;
  IndexType* labels_in_ptr_;
  STORE device_store_;
  const int32_t num_packs_;
  const int32_t rows_;
};

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType, int pack_size, int wg_array_size>
inline Status SparseXentWorkGroupImpl(sycl::queue* stream, LOAD device_load,
                                      IndexType* labels_in_ptr,
                                      STORE device_store, const int32_t rows,
                                      const int32_t cols,
                                      sycl::range<1> global_range,
                                      sycl::range<1> local_range) {
  assert(cols % pack_size == 0);
  int32_t num_packs = cols / pack_size;
  stream->submit([&](sycl::handler& h) {
    SparseXentWorkGroupKernel<LOAD, STORE, ComputeType, IndexType, pack_size,
                              wg_array_size>
        workgroup_kernel(device_load, labels_in_ptr, device_store, num_packs,
                         rows);
    h.parallel_for<SparseXentWorkGroupKernel<
        LOAD, STORE, ComputeType, IndexType, pack_size, wg_array_size>>(
        sycl::nd_range<1>(global_range, local_range), workgroup_kernel);
  });
  return Status::OK();
}

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType, int pack_size>
inline Status LaunchSparseXentWorkGroupImpl(
    sycl::queue* stream, LOAD device_load, IndexType* labels_in_ptr,
    STORE device_store, const int32_t rows, const int32_t cols) {
  const int workgroup_size = 128;
  const int num_wg = (rows + workgroup_size - 1) / workgroup_size;
  sycl::range<1> local_range(workgroup_size);
  sycl::range<1> global_range(num_wg * workgroup_size);
  //  wg_array_size is equal to rows/id.get_global_ranges(0),
  //  in this pass, the max value of wg_array_size is 1
  return SparseXentWorkGroupImpl<LOAD, STORE, ComputeType, IndexType, pack_size,
                                 1>(stream, device_load, labels_in_ptr,
                                    device_store, rows, cols, global_range,
                                    local_range);
}

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType>
struct DispatchSparseXentWorkGroupImplPackSize {
  Status operator()(sycl::queue* stream, LOAD device_load,
                    IndexType* labels_in_ptr, STORE device_store,
                    const int32_t rows, const int32_t cols) {
    if (cols % 2 == 0) {
      return LaunchSparseXentWorkGroupImpl<LOAD, STORE, ComputeType, IndexType,
                                           2>(
          stream, device_load, labels_in_ptr, device_store, rows, cols);
    } else {
      return LaunchSparseXentWorkGroupImpl<LOAD, STORE, ComputeType, IndexType,
                                           1>(
          stream, device_load, labels_in_ptr, device_store, rows, cols);
    }
  }
};

template <typename LOAD, typename STORE, typename ComputeType,
          typename IndexType>
inline Status DispatchSparseXentWorkGroupImpl(
    sycl::queue* stream, LOAD device_load, IndexType* labels_in_ptr,
    STORE device_store, const int32_t rows, const int32_t cols) {
  return DispatchSparseXentWorkGroupImplPackSize<LOAD, STORE, ComputeType,
                                                 IndexType>()(
      stream, device_load, labels_in_ptr, device_store, rows, cols);
}

template <typename T, typename IndexType>
Status DispatchSparseXent(sycl::queue* stream, const int32_t num_instances,
                          const int32_t num_classes, IndexType* labels_in_ptr,
                          T* softmax_out_ptr, T* loss_out_ptr,
                          T* backprop_out_ptr) {
  LaunchComputeProbDiffKernel<T, IndexType>(stream, num_instances, num_classes,
                                            labels_in_ptr, softmax_out_ptr,
                                            backprop_out_ptr);

  const int32_t MaxCompUnits =
      stream->get_device()
          .template get_info<sycl::info::device::max_compute_units>();
  const int32_t MaxGroupSize =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();

  if (num_instances >= MaxCompUnits * MaxGroupSize) {
    //  the num of rows is too large, use one workitem to process one row
    return DispatchSparseXentWorkItemImpl<T, IndexType>(
        stream, num_instances, num_classes, labels_in_ptr, softmax_out_ptr,
        loss_out_ptr);
  } else {
    using ComputeType = typename itex::DefaultComputeType<T>::type;
    itex::DirectLoad<T, ComputeType> device_load(
        softmax_out_ptr, num_classes);  // [num_instances, num_classes]
    itex::DirectStore<ComputeType, T> device_store(loss_out_ptr,
                                                   1);  // [num_instances]
    auto sycl_device = stream->get_device();
    if (IsXeHPC(&sycl_device) && num_classes <= 1024) {
      //  use one subgroup to process one row
      return DispatchSparseXentSubGroupImpl<decltype(device_load),
                                            decltype(device_store), ComputeType,
                                            IndexType>(
          stream, device_load, labels_in_ptr, device_store, num_instances,
          num_classes);
    } else {
      //  the num of cols>1024, use one workgroup to process one row
      return DispatchSparseXentWorkGroupImpl<decltype(device_load),
                                             decltype(device_store),
                                             ComputeType, IndexType>(
          stream, device_load, labels_in_ptr, device_store, num_instances,
          num_classes);
    }
  }
}

}  // namespace sparse_xent

}  // namespace impl

namespace functor {

template <typename GPUDevice, typename T, typename IndexType>
struct SparseXentFunctor {
  Status operator()(const GPUDevice& device, const Tensor& logits_in,
                    const Tensor& labels_in, Tensor softmax_temp,
                    Tensor* loss_out, Tensor* back_out) {
    sycl::queue* stream = device.stream();

    const int num_instances = logits_in.shape().dim_size(0);
    const int num_classes = logits_in.shape().dim_size(1);

    T* logits_in_ptr = const_cast<T*>(logits_in.flat<T>().data());
    IndexType* labels_in_ptr =
        const_cast<IndexType*>(labels_in.flat<IndexType>().data());

    T* softmax_ptr = softmax_temp.flat<T>().data();
    T* loss_out_ptr = loss_out->flat<T>().data();
    T* back_out_ptr = back_out->flat<T>().data();

    using ComputeType = typename itex::DefaultComputeType<T>::type;

    itex::DirectLoad<T, ComputeType> device_load(logits_in_ptr, num_classes);
    itex::DirectStore<ComputeType, T> device_store(softmax_ptr, num_classes);

    auto softmax_status =
        impl::softmax::DispatchLogSoftmax<decltype(device_load),
                                          decltype(device_store), ComputeType>(
            device, device_load, device_store, num_instances, num_classes,
            true);
    assert(softmax_status == Status::OK());
    Status runSuccess_xent =
        impl::sparse_xent::DispatchSparseXent<T, IndexType>(
            stream, num_instances, num_classes, labels_in_ptr, softmax_ptr,
            loss_out_ptr, back_out_ptr);
    return runSuccess_xent;
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_SPARSE_XENT_OP_H_
