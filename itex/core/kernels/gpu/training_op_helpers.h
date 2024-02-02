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

#ifndef ITEX_CORE_KERNELS_GPU_TRAINING_OP_HELPERS_H_
#define ITEX_CORE_KERNELS_GPU_TRAINING_OP_HELPERS_H_

#include <vector>

#include "itex/core/kernels/gpu/dense_update_functor.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/status.h"
#ifdef USING_NEXTPLUGGABLE_DEVICE
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"
#endif
namespace itex {

enum AssignUpdateType {
  Assign = 0,
  Add = 1,
  Sub = 2,
};

// Wrapper struct for TF_VariableInputLockHolder
struct VariableInputLockHolderWrapper {
 public:
  explicit VariableInputLockHolderWrapper(
      TF_VariableInputLockHolder* lockHolder)
      : lockHolder_(lockHolder) {}
  // Must be contructed with an acquired TF_VariableInputLockHolder
  VariableInputLockHolderWrapper() = delete;

  ~VariableInputLockHolderWrapper() {
    TF_ReleaseVariableInputLockHolder(lockHolder_);
  }

 private:
  TF_VariableInputLockHolder* lockHolder_;
};

// TODO(itex): move the DenseAssginWrapper and DenseUpdateWrapper to the
// op_kernel.h, together with EmptyCopyFunctor, after we fix the issue "No
// kernel name provided without -fsycl-unnamed-lambda enabled"

// Wrapper functor DenseUpdate with ASSIGN to C function
template <typename Device, typename T>
void DenseAssignWrapper(TF_OpKernelContext* tf_ctx, TF_Tensor* tf_source,
                        TF_Tensor* tf_dest) {
  OpKernelContext ctx(tf_ctx);
  const Tensor source(tf_source);
  Tensor dest(tf_dest);
  size_t size = dest.shape().num_elements() * DataTypeSize(dest.dtype());

#ifdef USING_NEXTPLUGGABLE_DEVICE
  if (pointer_is_pjrt_tensor(tf_dest)) {
    TF_Status* tf_status = TF_NewStatus();
    PJRT_Buffer* pjrt_c_buffer = TF_GetPjRtCBuffer(tf_dest, tf_status);
    if (pjrt_c_buffer == nullptr) {
      int device_id = TF_GetDeviceId(tf_ctx);
      PJRT_Client* pjrt_c_client = TF_GetPjRtCClient(DEVICE_XPU, tf_status);

      int rank = dest.shape().dims();
      std::vector<int64_t> dimensions(rank);
      for (int d = 0; d < rank; ++d) {
        dimensions[d] = dest.shape().dim_size(d);
      }
      ITEXNpdConfig& npdConfig = ITEXNpdConfig::getNpdConfig();
      if (npdConfig.isXlaAutoJitEnabled()) {
        std::vector<int64_t> layout(rank);
        std::iota(layout.rbegin(), layout.rend(), 0);
        TF_CreatePjRtBuffer(
            tf_dest,
            ITEXCreateSEPjRtBuffer(device_id, DataTypeString(dest.dtype()),
                                   dimensions, layout, pjrt_c_client),
            "XPU", tf_status);
      } else {
        TF_CreatePjRtBuffer(
            tf_dest,
            ITEXCreatePjRtBuffer(device_id, DataTypeString(dest.dtype()),
                                 &dimensions, size, pjrt_c_client),
            "XPU", tf_status);
      }
      TF_DeleteStatus(tf_status);
    }
  }
#endif

  functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
  copy_functor(ctx.eigen_device<Device>(), dest.flat<T>(), source.flat<T>());
}

// Wrapper functor DenseUpdate to C function
template <typename Device, typename T>
void DenseUpdateWrapper(TF_OpKernelContext* tf_ctx, TF_Tensor* tf_source,
                        TF_Tensor* tf_dest, int Op) {
  OpKernelContext ctx(tf_ctx);
  const Tensor source(tf_source);
  Tensor dest(tf_dest);
  size_t size = dest.shape().num_elements() * DataTypeSize(dest.dtype());

#ifdef USING_NEXTPLUGGABLE_DEVICE
  if (pointer_is_pjrt_tensor(tf_dest)) {
    TF_Status* tf_status = TF_NewStatus();
    PJRT_Buffer* pjrt_c_buffer = TF_GetPjRtCBuffer(tf_dest, tf_status);
    if (pjrt_c_buffer == nullptr) {
      int device_id = TF_GetDeviceId(tf_ctx);
      PJRT_Client* pjrt_c_client = TF_GetPjRtCClient(DEVICE_XPU, tf_status);

      int rank = dest.shape().dims();
      std::vector<int64_t> dimensions(rank);
      for (int d = 0; d < rank; ++d) {
        dimensions[d] = dest.shape().dim_size(d);
      }
      ITEXNpdConfig& npdConfig = ITEXNpdConfig::getNpdConfig();
      if (npdConfig.isXlaAutoJitEnabled()) {
        std::vector<int64_t> layout(rank);
        std::iota(layout.rbegin(), layout.rend(), 0);
        TF_CreatePjRtBuffer(
            tf_dest,
            ITEXCreateSEPjRtBuffer(device_id, DataTypeString(dest.dtype()),
                                   dimensions, layout, pjrt_c_client),
            "XPU", tf_status);
      } else {
        TF_CreatePjRtBuffer(
            tf_dest,
            ITEXCreatePjRtBuffer(device_id, DataTypeString(dest.dtype()),
                                 &dimensions, size, pjrt_c_client),
            "XPU", tf_status);
      }
      TF_DeleteStatus(tf_status);
    }
  }
#endif

  if (Op == AssignUpdateType::Assign) {
    functor::DenseUpdate<Device, T, ASSIGN> update_functor;
    update_functor(ctx.eigen_device<Device>(), dest.flat<T>(),
                   source.flat<T>());
  } else if (Op == AssignUpdateType::Add) {
    functor::DenseUpdate<Device, T, ADD> update_functor;
    update_functor(ctx.eigen_device<Device>(), dest.flat<T>(),
                   source.flat<T>());
  } else if (Op == AssignUpdateType::Sub) {
    functor::DenseUpdate<Device, T, SUB> update_functor;
    update_functor(ctx.eigen_device<Device>(), dest.flat<T>(),
                   source.flat<T>());
  }
}

inline VariableInputLockHolderWrapper
MaybeLockVariableInputMutexesInOrderHelper(
    OpKernelContext* ctx, bool do_lock, bool sparse,
    const std::vector<int>& input_ids,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest)) {
  TF_Status* tf_status = TF_NewStatus();
  TF_OpKernelContext* tf_ctx = ctx->Get();
  TF_VariableInputLockHolder* lockHolder = nullptr;
  TF_MaybeLockVariableInputMutexesInOrder(tf_ctx, do_lock, sparse,
                                          input_ids.data(), input_ids.size(),
                                          copyFunc, &lockHolder, tf_status);

  Status status = StatusFromTF_Status(tf_status);
  ITEX_CHECK_OK(status);
  TF_DeleteStatus(tf_status);

  VariableInputLockHolderWrapper lock_wrapper(lockHolder);
  return lock_wrapper;
}

// MaybeLockVariableInputMutexesInOrder is a helper function to acquire mutexes
// in address order to mitigate deadlock.  Returns a structure that, when
// deleted, will release the acquired mutexes. Safe to pass duplicates - will
// only lock each distinct mutex once. If sparse is true will ensure the
// variable gets switched to copy-on-read mode before trying to acquire the
// locks. If do_lock is false, returns immediately for reference variables. For
// resource variables in copy-on-read-mode it will grab a shared lock if do_lock
// is false, exclusive lock otherwise.  Note that this silently doesn't lock
// mutexes for invalid variable references; in all usages this is followed by
// GetInputTensor which will signal a failure.

// MaybeLockVariableInputMutexesInOrder and GetInputTensorFromVariable both have
// two kinds of functions, one has template arguments "Device" and "T", the
// other one doesn't have. The function with template argument can create
// concrete copy functor, which makes it have full functionality. The function
// without template argument is suitable for the situations where the caller
// cannot provide the "T" argument. In those situations, the copy functor is not
// needed at all. Actually the copy functor is only needed to sparse resource
// tensor.
template <typename Device, typename T>
VariableInputLockHolderWrapper MaybeLockVariableInputMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, bool sparse,
    const std::vector<int>& input_ids) {
  return MaybeLockVariableInputMutexesInOrderHelper(
      ctx, do_lock, sparse, input_ids, DenseAssignWrapper<Device, T>);
}

inline VariableInputLockHolderWrapper MaybeLockVariableInputMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, bool sparse,
    const std::vector<int>& input_ids) {
  return MaybeLockVariableInputMutexesInOrderHelper(
      ctx, do_lock, sparse, input_ids, EmptyCopyFunctor);
}

inline Status GetInputTensorFromVariableHelper(
    OpKernelContext* ctx, int input, bool lock_held, bool sparse, Tensor* out,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest)) {
  // TODO(itex): Currently, ITEX actually doesn't support Variant DataType.
  // Add this check when we support such datatype.
  bool is_variant_type = false;
  TF_Status* tf_status = TF_NewStatus();
  TF_OpKernelContext* tf_ctx = ctx->Get();
  TF_Tensor* tf_tensor = nullptr;

  // For ref tensor or dense tensor, the 3th, 4th, 5th arguments are actually
  // useless.
  TF_GetInputTensorFromVariable(tf_ctx, input, lock_held, is_variant_type,
                                sparse, copyFunc, &tf_tensor, tf_status);

  TensorShape shape;
  auto dims = TF_NumDims(tf_tensor);
  for (auto j = 0; j < dims; ++j) {
    shape.AddDim(TF_Dim(tf_tensor, j));
  }

  *out =
      Tensor(static_cast<DataType>(TF_TensorType(tf_tensor)), shape, tf_tensor);

  Status status = StatusFromTF_Status(tf_status);
  ITEX_CHECK_OK(status);
  TF_DeleteStatus(tf_status);
  return status;
}

// This gives you `*out`, a tensor you can update, corresponding to a variable
// passed as input index `input`.  This handles the differences between
// reference and resource variables. For reference variables we can just grab
// the tensor, grabbing the lock if lock_held is False.
//
// For resource variables we, if sparse is true, ensure it's in copy-on-read
// mode, and then, regardless of the value of sparse, ensure its refcount is 1
// (by potentially copying its contents). In this case lock_held is ignored.

// To understand why we have two kinds of functions: w and w/o template
// argument, please turn to the comments in
// "MaybeLockVariableInputMutexesInOrder"
template <typename Device, typename T>
Status GetInputTensorFromVariable(OpKernelContext* ctx, int input,
                                  bool lock_held, bool sparse, Tensor* out) {
  return GetInputTensorFromVariableHelper(ctx, input, lock_held, sparse, out,
                                          DenseAssignWrapper<Device, T>);
}

inline Status GetInputTensorFromVariable(OpKernelContext* ctx, int input,
                                         bool lock_held, bool sparse,
                                         Tensor* out) {
  return GetInputTensorFromVariableHelper(ctx, input, lock_held, sparse, out,
                                          EmptyCopyFunctor);
}

void MaybeForwardRefInputToRefOutput(OpKernelContext* ctx, int input,
                                     int output);

template <typename Device, typename T>
Status AssignVariableHelper(OpKernelContext* ctx, int input_index,
                            int value_index, bool validate_shape) {
  TF_Status* tf_status = TF_NewStatus();
  TF_OpKernelContext* tf_ctx = ctx->Get();
  TF_AssignVariable(tf_ctx, input_index, value_index, validate_shape,
                    DenseAssignWrapper<Device, T>, tf_status);
  Status status = StatusFromTF_Status(tf_status);
  ITEX_CHECK_OK(status);
  TF_DeleteStatus(tf_status);
  return status;
}

template <typename Device, typename T>
Status AssignRefVariableHelper(OpKernelContext* ctx, int input_ref_index,
                               int output_ref_index, int value_index,
                               bool use_exclusive_lock, bool validate_shape) {
  TF_Status* tf_status = TF_NewStatus();
  TF_OpKernelContext* tf_ctx = ctx->Get();
  TF_AssignRefVariable(tf_ctx, input_ref_index, output_ref_index, value_index,
                       use_exclusive_lock, validate_shape,
                       DenseAssignWrapper<Device, T>, tf_status);
  Status status = StatusFromTF_Status(tf_status);
  ITEX_CHECK_OK(status);
  TF_DeleteStatus(tf_status);
  return status;
}

template <typename Device, typename T, DenseUpdateType Op>
Status AssignUpdateVariable(OpKernelContext* ctx, int input_index,
                            int value_index) {
  // TODO(itex): Currently, ITEX actually doesn't support Variant DataType.
  // Add this check when we support such datatype.
  bool is_variant_type = false;
  TF_Status* tf_status = TF_NewStatus();
  TF_OpKernelContext* tf_ctx = ctx->Get();

  int update_type = -1;

  switch (Op) {
    case ASSIGN:
      update_type = AssignUpdateType::Assign;
      break;
    case ADD:
      update_type = AssignUpdateType::Add;
      break;
    case SUB:
      update_type = AssignUpdateType::Sub;
      break;
    default:
      ITEX_CHECK(false);
      break;
  }

  TF_AssignUpdateVariable(tf_ctx, input_index, value_index, update_type,
                          is_variant_type, DenseAssignWrapper<Device, T>,
                          DenseUpdateWrapper<Device, T>, tf_status);
  Status status = StatusFromTF_Status(tf_status);
  ITEX_CHECK_OK(status);
  TF_DeleteStatus(tf_status);
  return status;
}

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_TRAINING_OP_HELPERS_H_
