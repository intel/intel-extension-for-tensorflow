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

#ifndef ITEX_BUILD_JAX
#include "itex/core/utils/op_kernel.h"

#include <iostream>
#include <string>

#include "itex/core/graph/config_util.h"
#ifndef INTEL_CPU_ONLY
#include "itex/core/utils/gpu_resource_mgr_pool.h"
#ifdef USING_NEXTPLUGGABLE_DEVICE
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"
#endif
#endif
#include "itex/core/utils/kernel_def_util.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/padding.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/traceme_encode.h"
#include "protos/node_def.pb.h"
#include "tensorflow/c/tf_tensor.h"

namespace itex {

/* static */ absl::Mutex OpTypeFactory::op_type_factory_mutex_(
    absl::kConstInit);

/* static */ std::map<KernelDefBuilder::KernelCreateFunc, std::string>*
OpTypeFactory::GetOpTypeFactory() {
  static auto op_type_factory =
      std::map<KernelDefBuilder::KernelCreateFunc, std::string>();
  return &op_type_factory;
}

/* static */ void OpTypeFactory::RegisterOpType(
    KernelDefBuilder::KernelCreateFunc func, const std::string& op_type) {
  absl::MutexLock lock(&op_type_factory_mutex_);
  auto* factory = GetOpTypeFactory();
  auto it = factory->find(func);
  if (it != factory->end()) {
    ITEX_CHECK(false) << "Multiple KernelCreateFunc registration";
  }
  factory->emplace(func, op_type);
}

/* static */ absl::string_view OpTypeFactory::GetForKernelCreateFunc(
    KernelDefBuilder::KernelCreateFunc func) {
  absl::MutexLock lock(&op_type_factory_mutex_);
  auto* factory = GetOpTypeFactory();
  auto it = factory->find(func);
  if (it == factory->end()) {
    ITEX_CHECK(false) << "KernelCreateFunc is not registered";
  }
  return it->second;
}

void EmptyCopyFunctor(TF_OpKernelContext* tf_ctx, TF_Tensor* tf_source,
                      TF_Tensor* tf_dest) {
  return;
}

int OpKernelContext::num_inputs() const { return TF_NumInputs(ctx_); }

bool OpKernelContext::input_is_ref(int index) const {
  return TF_IsRefInput(ctx_, index, status_);
}

DataType OpKernelContext::input_dtype(int index) const {
  if (inputs_ != nullptr && inputs_->at(index) != nullptr) {
    return inputs_->at(index)->dtype();
  } else {
    ITEX_CHECK(false)
        << "please call ctx.input_dtype() after calling ctx.input() or "
           "ctx.mutable_input(), due to lack C-API to check whether is ref "
           "tensor or not";
  }
}

// Status OpKernelContext::input_dtype(StringPiece name, DataType* dtype) const;

MemoryType OpKernelContext::input_memory_type(int index) const {
  DataType dtype = input_dtype(index);
  return MTypeFromDType(dtype);
}

int OpKernelContext::num_outputs() const { return TF_NumOutputs(ctx_); }

DataType OpKernelContext::expected_output_dtype(int index) const {
  return static_cast<DataType>(TF_ExpectedOutputDataType(ctx_, index));
}

const Tensor& OpKernelContext::input(int index) const {
  ITEX_CHECK_GE(index, 0);
  ITEX_CHECK_LT(index, num_inputs());
  if (inputs_ == nullptr) {
    inputs_ = new gtl::InlinedVector<std::shared_ptr<Tensor>, 4>(num_inputs());
  }

  if (!inputs_->at(index)) {
    TF_Tensor* tensor = nullptr;
    TF_GetInput(ctx_, index, &tensor, status_);
    TensorShape shape;
    auto dims = TF_NumDims(tensor);
    for (auto j = 0; j < dims; ++j) {
      shape.AddDim(TF_Dim(tensor, j));
    }
    std::shared_ptr<Tensor> ptr = std::make_shared<Tensor>(
        static_cast<DataType>(TF_TensorType(tensor)), shape, tensor);
    inputs_->at(index) = std::move(ptr);
  }
  ITEX_CHECK_NE(inputs_, nullptr);
  return *inputs_->at(index);
}

#ifndef INTEL_CPU_ONLY
ResourceMgr* OpKernelContext::resource_manager() {
  ITEX_GPUStream* itex_gpu_stream = OpKernelContext::GetDeviceStream();
  auto error = GetResourceMgr(itex_gpu_stream, &resource_mgr);
  if (error != ITEX_GPU_SUCCESS) {
    CtxFailure(__FILE__, __LINE__,
               errors::Internal("Error to call GetResourceMgr with error ",
                                ITEX_GPUGetErrorName(error)));
  }
  return resource_mgr;
}
#endif

Status OpKernelContext::input(StringPiece name, const Tensor** tensor) {
  TF_Status* status = TF_NewStatus();
  if (inputsMap_.find(name) == inputsMap_.end()) {
    TF_Tensor* tensor = nullptr;
    TF_GetInputByName(ctx_, std::string(name).c_str(), &tensor, status);
    Status cc_status = StatusFromTF_Status(status);
    if (!cc_status.ok()) {
      return cc_status;
    }

    inputsMap_[name] = std::make_shared<Tensor>(tensor);
  }

  *tensor = inputsMap_[name].get();
  Status cc_status = StatusFromTF_Status(status);
  TF_DeleteStatus(status);
  return cc_status;
}

void* OpKernelContext::tensor_data(int index) {
  TF_Tensor* tensor = nullptr;
  TF_GetInput(ctx_, index, &tensor, status_);
#ifdef USING_NEXTPLUGGABLE_DEVICE
  void* data;
  if (npdConfig_.IfEnableNextPluggableDevice())
    data = tensor_get_raw_data(tensor);
  else
    data = TF_TensorData(tensor);
#else
  void* data = TF_TensorData(tensor);
#endif
  TF_DeleteTensor(tensor);
  return data;
}

bool OpKernelContext::is_input_same(int index, std::vector<int64> shape) {
  TF_Tensor* tensor = nullptr;
  TF_GetInput(ctx_, index, &tensor, status_);
  int dims = TF_NumDims(tensor);
  if (dims != static_cast<int>(shape.size())) {
    TF_DeleteTensor(tensor);
    return false;
  }

  for (int i = 0; i < dims; ++i) {
    if (shape[i] != TF_Dim(tensor, i)) {
      TF_DeleteTensor(tensor);
      return false;
    }
  }

  TF_DeleteTensor(tensor);
  return true;
}

int64_t OpKernelContext::step_id() const { return TF_StepId(ctx_); }

// Status OpKernelContext::set_output(StringPiece name, const Tensor& tensor) {
//   TF_Status* status = TF_NewStatus();

//   TF_SetOutputByName(ctx_, std::string(name).c_str(), tensor.GetTFTensor(),
//                      status);

//   Status cc_status = StatusFromTF_Status(status);
//   TF_DeleteStatus(status);
//   return cc_status;
// }

bool OpKernelContext::ValidateInputsAreSameShape() {
  OpKernelContext ctx(ctx_);

  const size_t kNumInputs = ctx.num_inputs();
  for (size_t i = 1; i < kNumInputs; ++i) {
    if (!ctx.input(0).IsSameSize(ctx.input(i))) {
      ctx.CtxFailure(errors::InvalidArgument(
          "Inputs must have the same size and shape. Input 0: ",
          ctx.input(0).shape().DebugString(), " != input ", std::to_string(i),
          ": ", ctx.input(i).shape().DebugString()));
      return false;
    }
  }

  return true;
}

Status OpKernelContext::forward_input_or_allocate_output(
    gtl::ArraySlice<int> candidate_input_indices, int output_index,
    const TensorShape& output_shape, Tensor** output, int* forwarded_input) {
  ITEX_CHECK_GE(output_index, 0);
  ITEX_CHECK_LT(output_index, num_outputs());
  TF_Tensor* tensor = TF_ForwardInputOrAllocateOutput(
      ctx_, const_cast<int*>(candidate_input_indices.data()),
      candidate_input_indices.size(), output_index,
      output_shape.dim_sizes().data(), output_shape.dims(), forwarded_input,
      status_);
#ifdef USING_NEXTPLUGGABLE_DEVICE
  if (pointer_is_pjrt_tensor(tensor)) {
    PJRT_Buffer* pjrt_c_buffer = TF_GetPjRtCBuffer(tensor, status_);
    if (pjrt_c_buffer == nullptr) {
      int device_id = TF_GetDeviceId(ctx_);
      static PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", status_);
      int rank = output_shape.dims();
      std::vector<int64_t> dimensions(rank);
      for (int d = 0; d < rank; ++d) {
        dimensions[d] = output_shape.dim_size(d);
      }
      DataType out_type =
          static_cast<DataType>(expected_output_dtype(output_index));
      size_t size = output_shape.num_elements() * DataTypeSize(out_type);
      if (npdConfig_.isXlaAutoJitEnabled()) {
        std::vector<int64_t> layout(rank);
        std::iota(layout.rbegin(), layout.rend(), 0);
        TF_CreatePjRtBuffer(
            tensor,
            ITEXCreateSEPjRtBuffer(device_id, DataTypeString(out_type),
                                   dimensions, layout, pjrt_c_client),
            "XPU", status_);
      } else {
        TF_CreatePjRtBuffer(
            tensor,
            ITEXCreatePjRtBuffer(device_id, DataTypeString(out_type),
                                 &dimensions, size, pjrt_c_client),
            "XPU", status_);
      }
    }
  }
#endif

  if (outputs_[output_index] == nullptr) {
    std::shared_ptr<Tensor> ptr = std::make_shared<Tensor>(
        static_cast<DataType>(expected_output_dtype(output_index)),
        output_shape, tensor);
    outputs_[output_index] = std::move(ptr);
  }

  *output = outputs_[output_index].get();
  return StatusFromTF_Status(status_);
}

void OpKernelContext::forward_ref_input_to_ref_output(int input_index,
                                                      int output_index) {
  TF_OpKernelContext_ForwardRefInputToRefOutput(ctx_, input_index,
                                                output_index);
}

Tensor* OpKernelContext::mutable_output(int index) {
  ITEX_DCHECK_GE(index, 0);
  ITEX_DCHECK_LT(index, num_outputs());

  return outputs_[index].get();
}

Tensor& OpKernelContext::mutable_input(int index, bool lock_held) {
  ITEX_CHECK_GE(index, 0);
  ITEX_CHECK_LT(index, num_inputs());
  if (inputs_ == nullptr) {
    inputs_ = new gtl::InlinedVector<std::shared_ptr<Tensor>, 4>(num_inputs());
  }

  if (!inputs_->at(index)) {
    TF_Tensor* tensor = nullptr;
    TF_GetInputTensorFromVariable(
        ctx_, index, lock_held, /* isVariantType unused */ false,
        /* sparse unused */ false, /* copyFunc */ EmptyCopyFunctor, &tensor,
        status_);
    Status s = StatusFromTF_Status(status_);
    ITEX_CHECK_EQ(Status::OK(), s);
    TensorShape shape;
    auto dims = TF_NumDims(tensor);
    for (auto j = 0; j < dims; ++j) {
      shape.AddDim(TF_Dim(tensor, j));
    }
    std::shared_ptr<Tensor> ptr = std::make_shared<Tensor>(
        static_cast<DataType>(TF_TensorType(tensor)), shape, tensor);
    inputs_->at(index) = std::move(ptr);
  }

  ITEX_CHECK_NE(inputs_, nullptr);
  return *inputs_->at(index);
}

Status OpKernelContext::output_list(StringPiece name, OpOutputList* list) {
  // int start, stop;
  // TODO(itex): Get OutputRange before creating OpOutputList.
  // TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  *list = OpOutputList(this, 0, num_outputs() - 1);
  return Status::OK();
}

Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
                                        Tensor** tensor) {
  DataType out_type = static_cast<DataType>(expected_output_dtype(index));
  size_t size = shape.num_elements() * DataTypeSize(out_type);
  TF_Tensor* output =
      TF_AllocateOutput(ctx_, index, static_cast<TF_DataType>(out_type),
                        shape.dim_sizes().data(), shape.dims(), size, status_);
#ifdef USING_NEXTPLUGGABLE_DEVICE
  if (pointer_is_pjrt_tensor(output)) {
    int device_id = TF_GetDeviceId(ctx_);
    static PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", status_);
    int rank = shape.dims();
    std::vector<int64_t> dimensions(rank);
    for (int d = 0; d < rank; ++d) {
      dimensions[d] = shape.dim_size(d);
    }
    if (npdConfig_.isXlaAutoJitEnabled()) {
      std::vector<int64_t> layout(rank);
      std::iota(layout.rbegin(), layout.rend(), 0);
      TF_CreatePjRtBuffer(
          output,
          ITEXCreateSEPjRtBuffer(device_id, DataTypeString(out_type),
                                 dimensions, layout, pjrt_c_client),
          "XPU", status_);
    } else {
      TF_CreatePjRtBuffer(
          output,
          ITEXCreatePjRtBuffer(device_id, DataTypeString(out_type), &dimensions,
                               size, pjrt_c_client),
          "XPU", status_);
    }
  }
#endif

  if (outputs_[index] == nullptr) {
    std::shared_ptr<Tensor> ptr = std::make_shared<Tensor>(
        static_cast<DataType>(expected_output_dtype(index)), shape, output);
    outputs_[index] = std::move(ptr);
  }
  *tensor = outputs_[index].get();

  return StatusFromTF_Status(status_);
}

Status OpKernelContext::allocate_temp(
    DataType type, const TensorShape& shape, Tensor* out_temp,
    AllocatorAttributes allocator_attr,
    const AllocationAttributes& allocation_attr) {
  TF_Tensor* tmp = TF_AllocateTemp(ctx_, static_cast<TF_DataType>(type),
                                   shape.dim_sizes().data(), shape.dims(),
                                   &allocator_attr.plugin_attr(), status_);
#ifdef USING_NEXTPLUGGABLE_DEVICE
  if (pointer_is_pjrt_tensor(tmp)) {
    int device_id = TF_GetDeviceId(ctx_);
    static PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", status_);

    int rank = shape.dims();
    std::vector<int64_t> dimensions(rank);
    for (int d = 0; d < rank; ++d) {
      dimensions[d] = shape.dim_size(d);
    }
    size_t size = shape.num_elements() * DataTypeSize(type);
    if (npdConfig_.isXlaAutoJitEnabled()) {
      std::vector<int64_t> layout(rank);
      std::iota(layout.rbegin(), layout.rend(), 0);
      TF_CreatePjRtBuffer(
          tmp,
          ITEXCreateSEPjRtBuffer(device_id, DataTypeString(type), dimensions,
                                 layout, pjrt_c_client),
          "XPU", status_);
    } else {
      TF_CreatePjRtBuffer(
          tmp,
          ITEXCreatePjRtBuffer(device_id, DataTypeString(type), &dimensions,
                               size, pjrt_c_client),
          "XPU", status_);
    }
  }
#endif

  Tensor t(type, shape, tmp);
  *out_temp = std::move(t);

  return StatusFromTF_Status(status_);
}

Status OpKernelContext::allocate_persistent(DataType type,
                                            const TensorShape& shape,
                                            PersistentTensor* out_persistent,
                                            Tensor** out_tensor,
                                            AllocatorAttributes attr) {
  Tensor persistent;
  TF_ABORT_IF_ERROR(allocate_temp(type, shape, &persistent, attr));

  // TODO(itex): proper use copy for persistent, plugin use move.
  // Investigate the result caused by different implementation.
  *out_persistent = PersistentTensor(std::move(persistent));
  Tensor* allocated = out_persistent->AccessTensor(this);
  if (out_tensor) {
    *out_tensor = allocated;
  }

  return StatusFromTF_Status(status_);
}

void OpKernelContext::set_output(int index, const Tensor& tensor) {
  ITEX_CHECK(index >= 0 && index < num_outputs())
      << " Index out of range while setting output";
  TF_SetOutput(ctx_, index, tensor.GetTFTensor(), status_);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status_)) << " Error while setting output";
  ITEX_CHECK_EQ(outputs_[index], nullptr);
  std::shared_ptr<Tensor> ptr = std::make_shared<Tensor>(tensor);
  outputs_[index] = std::move(ptr);
  return;
}

/// all below CtxFailure will pass back the TF_Status created by plugin.
/// so we need not to delete it, which will be controlled by TF.
void OpKernelContext::CtxFailure(const Status& s) {
  ITEX_VLOG(1) << s;
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}
void OpKernelContext::CtxFailure(const char* file, int line, const Status& s) {
  ITEX_LOG(WARNING) << file << ": " << line << s;
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelContext::CtxFailureWithWarning(const Status& s) {
  ITEX_LOG(WARNING) << s;
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}
void OpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                            const Status& s) {
  ITEX_LOG(WARNING) << file << line << s;
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelContext::SetStatus(const Status& s) {
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

// class OpKernelConstruction -------------------------------------------------
bool OpKernelConstruction::HasAttr(StringPiece attr_name) const {
  // note that StringPiece.data() will return a not nul-terminated char*
  // so will need std::string.c_str()
  std::string name(attr_name.data(), attr_name.size());
  bool ret = TF_OpKernelConstruction_HasAttr(ctx_, name.c_str(), status_);
  return ret;
}

template <>
Status OpKernelConstruction::GetAttr<int32_t>(StringPiece attr_name,
                                              int32_t* value) const {
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrInt32(ctx_, name.c_str(), value, status_);
  return StatusFromTF_Status(status_);
}

template <>
Status OpKernelConstruction::GetAttr<DataType>(StringPiece attr_name,
                                               DataType* value) const {
  TF_DataType type;
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrType(ctx_, name.c_str(), &type, status_);
  *value = static_cast<DataType>(type);
  return StatusFromTF_Status(status_);
}

template <>
Status OpKernelConstruction::GetAttr<int64_t>(StringPiece attr_name,
                                              int64_t* value) const {
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrInt64(ctx_, name.c_str(), value, status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<float>(StringPiece attr_name,
                                            float* value) const {
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrFloat(ctx_, name.c_str(), value, status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<bool>(StringPiece attr_name,
                                           bool* value) const {
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrBool(
      ctx_, name.c_str(), reinterpret_cast<unsigned char*>(value), status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<std::string>(StringPiece attr_name,
                                                  std::string* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;
  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  std::vector<char> val(total_size);
  TF_OpKernelConstruction_GetAttrString(ctx_, name.c_str(), val.data(),
                                        total_size, status_);
  *value = std::string(val.data(), total_size);
  return StatusFromTF_Status(status_);
}

template <>
Status OpKernelConstruction::GetAttr<Padding>(StringPiece attr_name,
                                              Padding* padding) const {
  std::string padding_str;
  auto status = GetAttr("padding", &padding_str);
  if (padding_str == "VALID") {
    *padding = Padding::VALID;
  } else if (padding_str == "SAME") {
    *padding = Padding::SAME;
  } else if (padding_str == "EXPLICIT") {
    *padding = Padding::EXPLICIT;
  } else {
    return errors::InvalidArgument("Unknown padding type: ", padding_str);
  }
  return status;
}

template <>
Status OpKernelConstruction::GetAttr<std::vector<string>>(
    StringPiece attr_name, std::vector<std::string>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;

  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);

  value->resize(list_size);

  std::unique_ptr<void*[]> vals(new void*[list_size]);
  std::unique_ptr<size_t[]> lens(new size_t[list_size]);
  std::unique_ptr<char[]> storage(new char[total_size]);
  size_t storage_size(total_size);
  TF_OpKernelConstruction_GetAttrStringList(
      ctx_, name.c_str(), reinterpret_cast<char**>(vals.get()), lens.get(),
      list_size, storage.get(), storage_size, status_);

  for (int32_t i = 0; i < list_size; ++i) {
    (*value)[i] = string(static_cast<const char*>(vals[i]), lens[i]);
  }

  return StatusFromTF_Status(status_);
}
// TODO(itex): Update if these apis are changed.
template <>
Status OpKernelConstruction::GetAttr<std::vector<int32_t>>(
    StringPiece attr_name, std::vector<int32_t>* value) const {
  Status s;
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;
  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  s = StatusFromTF_Status(status_);
  if (!s.ok()) return s;
  value->resize(list_size);
  TF_OpKernelConstruction_GetAttrInt32List(ctx_, name.c_str(), value->data(),
                                           list_size, status_);
  return StatusFromTF_Status(status_);
}

template <>
Status OpKernelConstruction::GetAttr<std::vector<DataType>>(
    StringPiece attr_name, std::vector<DataType>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;

  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  value->resize(list_size);
  TF_OpKernelConstruction_GetAttrTypeList(
      ctx_, name.c_str(), reinterpret_cast<TF_DataType*>(value->data()),
      list_size, status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<std::vector<int64_t>>(
    StringPiece attr_name, std::vector<int64_t>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;

  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  value->resize(list_size);
  TF_OpKernelConstruction_GetAttrInt64List(ctx_, name.c_str(), value->data(),
                                           list_size, status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<std::vector<float>>(
    StringPiece attr_name, std::vector<float>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;

  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  value->resize(list_size);
  TF_OpKernelConstruction_GetAttrFloatList(ctx_, name.c_str(), value->data(),
                                           list_size, status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<std::vector<bool>>(
    StringPiece attr_name, std::vector<bool>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;

  std::vector<TF_Bool> value_tmp;
  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  value_tmp.resize(list_size);
  TF_OpKernelConstruction_GetAttrBoolList(ctx_, name.c_str(), value_tmp.data(),
                                          list_size, status_);
  value->resize(list_size);
  for (int i = 0; i < list_size; i++)
    (*value)[i] = static_cast<bool>(value_tmp[i]);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<TensorShape>(StringPiece attr_name,
                                                  TensorShape* shape) const {
  int32_t list_size = 0;
  int32_t total_size = 0;
  std::vector<int64_t> shape_list;

  TF_OpKernelConstruction_GetAttrSize(ctx_, "shape", &list_size, &total_size,
                                      status_);
  shape_list.resize(total_size);
  TF_OpKernelConstruction_GetAttrTensorShape(ctx_, "shape", shape_list.data(),
                                             total_size, status_);
  for (auto dim : shape_list) {
    shape->AddDim(dim);
  }
  return StatusFromTF_Status(status_);
}

template <>
Status OpKernelConstruction::GetAttr<Tensor>(StringPiece attr_name,
                                             Tensor* value) const {
  std::string name(attr_name.data(), attr_name.size());
  TF_Tensor* buf = nullptr;

  TF_OpKernelConstruction_GetAttrTensor(ctx_, name.c_str(), &buf, status_);
  *value = Tensor(buf);

  return StatusFromTF_Status(status_);
}

void OpKernelConstruction::CtxFailure(const Status& s) {
  ITEX_VLOG(1) << s;
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelConstruction::CtxFailure(const char* file, int line,
                                      const Status& s) {
  ITEX_LOG(WARNING) << file << ":" << line << s;
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelConstruction::CtxFailureWithWarning(const Status& s) {
  ITEX_LOG(WARNING) << s;
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                 const Status& s) {
  ITEX_LOG(WARNING) << file << ": " << line << s;
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelConstruction::SetStatus(const Status& s) {
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

const char* OpKernelConstruction::OpName() const {
  return TF_OpKernelConstruction_GetName(ctx_).data;
}

OpKernel::OpKernel(OpKernelConstruction* context)
    : op_name(context->OpName()) {}

OpKernel::~OpKernel() {}

string OpKernel::ShapeTraceString(const OpKernelContext& ctx) const {
  int num_inputs = ctx.num_inputs();
  if (num_inputs == 0) return "";
  std::vector<string> tensor_shapes;
  tensor_shapes.reserve(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    if (ctx.input_is_ref(i) || ctx.input(i).GetTFTensor() == nullptr) {
      tensor_shapes.emplace_back();  // Placeholder
      continue;
    }
    DataType input_dtype = ctx.input_dtype(i);
    if (input_dtype == DataType::DT_RESOURCE ||
        input_dtype == DataType::DT_VARIANT || IsRefType(input_dtype)) {
      tensor_shapes.emplace_back();  // Placeholder
      continue;
    }
    tensor_shapes.emplace_back(strings::StrCat(
        DataTypeString(input_dtype), ctx.input(i).shape().DebugString()));
  }
  return strings::StrCat("(", absl::StrJoin(tensor_shapes, ";"), ")");
}

string OpKernel::TraceString(const OpKernelContext& ctx) const {
  string trace_string = TraceMeOp(op_name, op_type);
  string shape = ShapeTraceString(ctx);
  if (!shape.empty()) {
    trace_string = TraceMeEncode(std::move(trace_string), {{"shape", shape}});
  }
  return trace_string;
}

void AsyncOpKernel::Compute(OpKernelContext* context) {
  Notification n;
  this->ComputeAsync(context, [&n]() { n.Notify(); });
  n.WaitForNotification();
}

KernelDefBuilder& KernelDefBuilder::Device(const char* backend) {
  backend_ = std::string(backend);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::HostMemory(const char* host) {
  host_memorys_.push_back(std::string(host));
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Priority(const int32 priority_number) {
  priority_ = priority_number;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::RegisterCreate(KernelCreateFunc func) {
  create_func_ = func;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::RegisterCompute(KernelComputeFunc func) {
  compute_func_ = func;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::RegisterComputeAsync(
    KernelComputeAsyncFunc func) {
  compute_async_func_ = func;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::RegisterDelete(KernelDeleteFunc func) {
  delete_func_ = func;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::KernelClassName(
    const char* kernel_class_name) {
  kernel_class_name_ = std::string(kernel_class_name);
  return *this;
}

Name::Name(const char* op_name) { op_name_ = std::string(op_name); }

void Name::Build(const char* device_name, const char* backend) {
  if (backend != backend_) {
    return;
  }

  ITEX_VLOG(2) << "Register the " << op_name_ << " of " << kernel_class_name_
               << " on the " << backend << " backend "
               << " with device name " << device_name;

  StatusUniquePtr status(TF_NewStatus());
  {
    TF_KernelBuilder* builder = nullptr;
    if (compute_func_) {
      builder = TF_NewKernelBuilder(op_name_.c_str(), device_name, create_func_,
                                    compute_func_, delete_func_);
    } else {
      builder =
          TF_NewAsyncKernelBuilder(op_name_.c_str(), device_name, create_func_,
                                   compute_async_func_, delete_func_);
    }
    OpTypeFactory::RegisterOpType(create_func_, op_name_);
    auto check_type_constraint = [&builder, &status, this](DataType dtype,
                                                           const char* name) {
      auto data_type = static_cast<TF_DataType>(dtype);
      TF_KernelBuilder_TypeConstraint(builder, name, data_type, status.get());
      ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
          << "Error while registering " << kernel_class_name_
          << " kernel with attribute " << name;
    };

    for (size_t i = 0; i < type_constraints_.size(); i++) {
      auto& type_constraint = type_constraints_[i];
      auto& type_value = type_values_[i];
      check_type_constraint(type_value, type_constraint.c_str());
    }

    for (auto const& host_memory : host_memorys_) {
      TF_KernelBuilder_HostMemory(builder, host_memory.c_str());
    }

    if (priority_ > 0) {
      TF_KernelBuilder_Priority(builder, priority_);
    }

    TF_RegisterKernelBuilder(kernel_class_name_.c_str(), builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << kernel_class_name_ << " kernel.";
  }
}

namespace register_kernel {
Registrar::Registrar(std::string key, KernelRegisterFunc func) {
  auto global_registry = GlobalKernelRegistry();
  mutex_lock l(&global_registry->mu);
  global_registry->registry.push_back(std::make_pair(key, func));
}

KernelRegistry* GlobalKernelRegistry() {
  static KernelRegistry global_kernel_registry = KernelRegistry();
  return &global_kernel_registry;
}

void RegisterCPUKernels(const char* device_name) {
  for (auto const& x : GlobalKernelRegistry()->registry) {
    KernelRegisterFunc func = x.second;
    func(device_name, DEVICE_CPU);
  }
}

void RegisterGPUKernels(const char* device_name) {
  for (auto const& x : GlobalKernelRegistry()->registry) {
    KernelRegisterFunc func = x.second;
    func(device_name, DEVICE_GPU);
  }
}

void RegisterDefaultKernels() {
  for (auto const& x : GlobalKernelRegistry()->registry) {
    KernelRegisterFunc func = x.second;
    func(DEVICE_DEFAULT, DEVICE_DEFAULT);
  }
}
}  // namespace register_kernel

static bool ReadSyncFromConfigOrEnv() {
  auto cfg = itex::itex_get_config();
  bool sync_exec_enabled = false;
  if (cfg.debug_options().xpu_force_sync()) {
    sync_exec_enabled = true;
  } else {
    ITEX_CHECK_OK(ReadBoolFromEnvVar("ITEX_SYNC_EXEC",
                                     false, /* default value */
                                     &sync_exec_enabled));
  }

  return sync_exec_enabled;
}

bool IsVerboseEnabled() {
  static std::once_flag verbose_flag;
  static int64 verbose_enabled;
  std::call_once(verbose_flag, [&]() {
    ITEX_CHECK_OK(ReadInt64FromEnvVar("ITEX_VERBOSE", 0, &verbose_enabled));
  });

  return verbose_enabled != 0;
}

bool IsSyncExecEnabled() {
  static std::once_flag sync_exec_flag;
  static bool sync_exec_enabled;
  std::call_once(sync_exec_flag, [&]() {
    sync_exec_enabled = ReadSyncFromConfigOrEnv();
    if (sync_exec_enabled) {
      ITEX_LOG(WARNING) << "Kernels will be executed with sync mode "
                        << "which will be hurt for end-to-end's performance. "
                        << "If this is not intended, please export "
                           "ITEX_SYNC_EXEC=0 or set off for xpu_force_sync.";
    }
  });

  return sync_exec_enabled;
}

namespace {

// Label defaults to empty if not found in NodeDef.
const string& GetKernelLabelAttr(const AttrSlice& node_attrs) {
  static const string& kKernelAttr = *new string("_kernel");
  static const string& kEmptyString = *new string("");

  const AttrValue* attr_value = node_attrs.FindByString(kKernelAttr);
  if (attr_value == nullptr || attr_value->value_case() != AttrValue::kS)
    return kEmptyString;
  else
    return attr_value->s();
}

// TODO(itex): Replace with const Node& version below.
Status FindKernelRegistration(
    const DeviceType& device_type, StringPiece node_name,
    bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    StringPiece node_op, AttrSlice node_attrs, const KernelDef** reg,
    bool* was_attr_mismatch) {
  *reg = nullptr;
  *was_attr_mismatch = false;

  TF_Buffer* kernel_list_buf = TF_NewBuffer();
  TF_Status* tf_status = TF_NewStatus();
  kernel_list_buf =
      TF_GetRegisteredKernelsForOp(string(node_op).c_str(), tf_status);
  Status status = StatusFromTF_Status(tf_status);
  if (!status.ok()) {
    return status;
  }

  KernelList kernel_list;
  kernel_list.ParseFromArray(kernel_list_buf->data, kernel_list_buf->length);

  TF_DeleteBuffer(kernel_list_buf);
  TF_DeleteStatus(tf_status);

  for (const auto& kernel_def : kernel_list.kernel()) {
    // If there is a kernel registered for the op and device_type,
    // check that the attrs match.
    bool match;

    if (kernel_def.device_type() != DeviceTypeString(device_type)) continue;
    const string& label = GetKernelLabelAttr(node_attrs);
    if (label != kernel_def.label()) continue;

    TF_RETURN_IF_ERROR(KernelAttrsMatch(kernel_def, node_attrs, &match));
    if (match) {
      if (*reg != nullptr) {
        if ((*reg)->priority() == kernel_def.priority()) {
          return errors::InvalidArgument(
              "Multiple OpKernel registrations match NodeDef at the same "
              "priority '",
              FormatNodeDefForError(node_name, has_experimental_debug_info,
                                    experimental_debug_info),
              "': '", (*reg)->ShortDebugString(), "' and '",
              kernel_def.ShortDebugString(), "'");
        } else if ((*reg)->priority() > kernel_def.priority()) {
          continue;
        }
        // iter->second's priority is higher than *reg.
      }
      *reg = &kernel_def;
    } else {
      *was_attr_mismatch = true;
    }
  }
  //  Check if no device specific registrations found. If not, try finding a
  //  default kernel.
  //  if (*reg == nullptr &&
  //      !IsSymbolicExecutionDevice(device_type.type_string())) {
  if (*reg == nullptr) {
    // If there is a kernel registered for the op and device_type,
    // check that the attrs match.
    for (const auto& kernel_def : kernel_list.kernel()) {
      if (kernel_def.device_type() != "DEFAULT") continue;
      bool match;
      TF_RETURN_IF_ERROR(KernelAttrsMatch(kernel_def, node_attrs, &match));
      if (match) {
        if (*reg != nullptr) {
          return errors::InvalidArgument(
              "Multiple Default OpKernel registrations match NodeDef '",
              FormatNodeDefForError(node_name, has_experimental_debug_info,
                                    experimental_debug_info),
              "': '", (*reg)->ShortDebugString(), "' and '",
              kernel_def.ShortDebugString(), "'");
        }
        *reg = &kernel_def;
      } else {
        *was_attr_mismatch = true;
      }
    }

    if (*reg != nullptr) {
      ITEX_VLOG(1) << "No device-specific kernels found for NodeDef '"
                   << FormatNodeDefForError(node_name,
                                            has_experimental_debug_info,
                                            experimental_debug_info)
                   << "'"
                   << "Will fall back to a default kernel." << std::endl;
    }
  }
  return Status::OK();
}

Status FindKernelRegistration(const DeviceType& device_type,
                              const NodeDef& node_def, const KernelDef** reg,
                              bool* was_attr_mismatch) {
  return FindKernelRegistration(
      device_type, node_def.name(), node_def.has_experimental_debug_info(),
      node_def.experimental_debug_info(), node_def.op(),
      AttrSlice(&node_def.attr()), reg, was_attr_mismatch);
}

}  // namespace

bool KernelDefAvailable(const DeviceType& device_type,
                        const NodeDef& node_def) {
  const KernelDef* reg = nullptr;
  bool was_attr_mismatch;
  Status result =
      FindKernelRegistration(device_type, node_def, &reg, &was_attr_mismatch);
  return result.ok() && reg != nullptr;
}

// TODO(itex): Change const NodeDef& to const Node&
Status FindKernelDef(
    const DeviceType& device_type, StringPiece node_name,
    bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    StringPiece node_op, StringPiece node_device, AttrSlice node_attrs,
    const KernelDef** def, string* kernel_class_name) {
  const KernelDef* reg = nullptr;
  bool was_attr_mismatch;
  TF_RETURN_IF_ERROR(FindKernelRegistration(
      device_type, node_name, has_experimental_debug_info,
      experimental_debug_info, node_op, node_attrs, &reg, &was_attr_mismatch));
  if (reg == nullptr) {
    const std::string device_str = DeviceTypeString(device_type);
    Status s = errors::NotFound(
        "No registered '", node_op, "' OpKernel for ", device_str,
        " devices compatible with node ",
        FormatNodeDefForError(node_name, has_experimental_debug_info,
                              experimental_debug_info));
    if (was_attr_mismatch) {
      //      errors::AppendToMessage(
      //          &s, " (OpKernel was found, but attributes didn't match) ",
      //          "Requested Attributes: ",
      //          SummarizeAttrsHelper(node_attrs, node_device));
    }
    //    // Do not print kernel registrations for other devices when using _JIT
    //    // devices for compilation.
    //    if (!absl::StrContains(device_str, "JIT")) {
    //      errors::AppendToMessage(
    //          &s, ".  Registered:", KernelsRegisteredForOp(node_op));
    //    }
    return s;
  }
  if (def != nullptr) *def = reg;
  //  if (kernel_class_name != nullptr) *kernel_class_name =
  //  reg->kernel_class_name;
  return Status::OK();
}

Status FindKernelDef(const DeviceType& device_type, const NodeDef& node_def,
                     const KernelDef** def, string* kernel_class_name) {
  return FindKernelDef(
      device_type, node_def.name(), node_def.has_experimental_debug_info(),
      node_def.experimental_debug_info(), node_def.op(), node_def.device(),
      AttrSlice(&node_def.attr()), def, kernel_class_name);
}

// PersistentTensor ----------------------------------------------------------

Tensor* PersistentTensor::AccessTensor(OpKernelConstruction* context) {
  // the caller has to have a valid context
  ITEX_CHECK(context);
  return &tensor_;
}

Tensor* PersistentTensor::AccessTensor(OpKernelContext* context) {
  return &tensor_;
}
void CheckNotInComputeAsync(OpKernelContext* ctx,
                            const char* correct_macro_name) {}

template <>
const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const {
  return eigen_cpu_device();
}

#ifndef INTEL_CPU_ONLY
template <>
const Eigen::GpuDevice& OpKernelContext::eigen_device() const {
  return eigen_gpu_device();
}
#endif  // INTEL_CPU_ONLY

}  // namespace itex
#endif  // ITEX_BUILD_JAX
