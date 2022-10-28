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

#ifndef ITEX_CORE_UTILS_OP_KERNEL_H_
#define ITEX_CORE_UTILS_OP_KERNEL_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "itex/core/utils/allocator.h"
#include "itex/core/utils/annotated_traceme.h"
#include "itex/core/utils/cpu_info.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/kernel_def_util.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/mutex.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/types.h"
#include "protos/node_def.pb.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#ifndef INTEL_CPU_ONLY
#include "itex/core/devices/gpu/eigen_stream_device.h"
#include "itex/core/devices/gpu/gpu_device_plugin.h"
#endif  // INTEL_CPU_ONLY

namespace itex {

class OpKernelContext;
class OpKernelConstruction;
class PersistentTensor;

// Empty function, given to C-API requiring a function pointer
void EmptyCopyFunctor(TF_OpKernelContext* tf_ctx, TF_Tensor* tf_source,
                      TF_Tensor* tf_dest);

template <typename ListType, typename ElementType>
class OpArgIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = ElementType;
  using pointer = ElementType*;
  using const_pointer = const ElementType*;
  using reference = ElementType&;
  using const_reference = const ElementType&;
  using difference_type = ptrdiff_t;

  OpArgIterator(const ListType* list, int i) : list_(list), i_(i) {}

  bool operator==(const OpArgIterator& rhs) {
    ITEX_DCHECK(list_ == rhs.list_);
    return i_ == rhs.i_;
  }

  bool operator!=(const OpArgIterator& rhs) {
    ITEX_DCHECK(list_ == rhs.list_);
    return i_ != rhs.i_;
  }

  OpArgIterator operator++() {  // prefix ++it
    ++i_;
    return *this;
  }

  OpArgIterator operator++(int) {  // postfix it++
    OpArgIterator old_value = *this;
    ++i_;
    return old_value;
  }

  reference operator*() { return (*list_)[i_]; }
  pointer operator->() { return &(*list_)[i_]; }

  const_reference operator*() const { return (*list_)[i_]; }
  const_pointer operator->() const { return &(*list_)[i_]; }

 private:
  const ListType* const list_;
  int i_;
};

// Utility class for representing a list of immutable input tensors
// that are passed to the op as a single named argument.
class OpInputList {
 public:
  typedef OpArgIterator<OpInputList, const Tensor> Iterator;
  OpInputList() : ctx_(nullptr), start_(0), stop_(0) {}
  OpInputList(OpKernelContext* ctx, int start, int stop)
      : ctx_(ctx), start_(start), stop_(stop) {}
  OpInputList& operator=(const OpInputList& other) = default;
  const Tensor& operator[](int i) const;
  int size() const { return stop_ - start_; }
  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

 private:
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;
};

// Utility class for representing a list of mutable ("ref") input tensors
// that are passed to the op as a single named argument.
class OpMutableInputList {
 public:
  typedef OpArgIterator<OpMutableInputList, Tensor*> Iterator;
  OpMutableInputList(OpKernelContext* ctx, int start, int stop)
      : ctx_(ctx), start_(start), stop_(stop) {}
  OpMutableInputList() : ctx_(nullptr), start_(0), stop_(0) {}
  OpMutableInputList& operator=(const OpMutableInputList& other) = default;
  Tensor at(int i, bool lock_held);
  mutex* ref_mutex(int i);
  int size() const { return stop_ - start_; }
  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

 private:
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;
};

// Utility class for representing a list of output tensors that are
// grouped as a single named output.
class OpOutputList {
 public:
  typedef OpArgIterator<OpOutputList, const Tensor*> Iterator;
  OpOutputList() : ctx_(nullptr), start_(0), stop_(0) {}
  OpOutputList(OpKernelContext* ctx, int start, int stop)
      : ctx_(ctx), start_(start), stop_(stop) {}
  OpOutputList& operator=(const OpOutputList& other) = default;
  Tensor* operator[](int i);
  // bool required(int i) const;
  DataType expected_output_dtype(int i) const;
  Status allocate(int i, const TensorShape& shape, Tensor** output);
  void set(int i, const Tensor& tensor);
  void set(int i, Tensor&& tensor);
  // void set_ref(int i, mutex* mu, Tensor* tensor_for_ref);
  int size() const { return stop_ - start_; }
  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

 private:
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;
};

class OpKernelContext {
 public:
  explicit OpKernelContext(TF_OpKernelContext* ctx)
      : ctx_(ctx),
        outputs_(TF_NumOutputs(ctx_)),
        status_(TF_NewStatus()),
        device_(ctx_, status_) {}

  ~OpKernelContext() {
    if (inputs_ != nullptr) {
      delete inputs_;
    }

    TF_DeleteStatus(status_);
    status_ = nullptr;
  }

  int num_inputs() const;  // { return inputs->size(); }

  // TODO(itex): Add TF_InputIsRef C-API to distinguish whether the input is
  // ref tensor or not. Currently, input_dtype is not fully funtional at all !!
  DataType input_dtype(int index) const;
  // Status input_dtype(StringPiece name, DataType* dtype) const;

  MemoryType input_memory_type(int index) const;

  int num_outputs() const;  // { return outputs_.size(); }
  DataType expected_output_dtype(int index) const;

  const Tensor& input(int index) const;

  Status input(StringPiece name, const Tensor** tensor);

  void* tensor_data(int index);

  bool is_input_same(int index, std::vector<int64> shape);

  //  Status input_list(StringPiece name, OpInputList* list);
  //
  //  TF_Mutex* input_ref_mutex(int index);
  //
  //  Tensor mutable_tensor(int index, bool lock_held);
  //
  Tensor& mutable_input(int index, bool lock_held);
  //
  //  Status mutable_input_list(StringPiece name, OPMutableInputList* list);
  //
  // void replace_ref_input(int index, Tensor& tensor, bool lock_held);

  // bool input_is_ref(int index) const;
  //
  //  Status replace_ref_input(StringPiece name, const Tensor& tensor,
  //                           bool lock_held);
  //
  //  void delete_ref_input(int input_index, bool lock_held);
  //
  //  bool has_input(int index) const;

  bool ValidateInputsAreSameShape();

  void forward_ref_input_to_ref_output(int input_index, int output_index);
  //
  //  bool forward_input_to_output_with_shape(int input_index, int output_index,
  //                                          const TensorShape& output_shape,
  //                                          Tensor** output)
  //                                          TF_MUST_USE_RESULT;
  //
  //  Status forward_input_to_output_with_shape(StringPiece input_name,
  //                                            StringPiece output_name,
  //                                            const TensorShape& output_shape,
  //                                            Tensor** output)
  //                                            TF_MUST_USE_RESULT;
  //
  //  std::unique_ptr<Tensor> forward_input(
  //      int input_index, int output_index, DataType output_dtype,
  //      const TensorShape& output_shape, MemoryType output_memory_type,
  //      const AllocatorAttributes& output_attr) TF_MUST_USE_RESULT;

  // Tries to forward one of the inputs given in input_indices to
  // output[output_index]. If none of the given inputs can be forwarded, calls
  // allocate_output() to allocate a new output buffer. The index of the
  // forwarded input will be assign to output argument forwarded_input (if it's
  // not nullptr). If no inputs are forwarded, forwarded_input will be assigned
  // -1.
  Status forward_input_or_allocate_output(
      gtl::ArraySlice<int> candidate_input_indices, int output_index,
      const TensorShape& output_shape, Tensor** output,
      int* forwarded_input = nullptr) TF_MUST_USE_RESULT;
  //  Status forward_input_or_allocate_output(
  //      gtl::ArraySlice<StringPiece> candidate_input_names,
  //      StringPiece output_name, const TensorShape& output_shape,
  //      Tensor** output) TF_MUST_USE_RESULT;

  Status output_list(StringPiece name, OpOutputList* list);

  //  bool output_required(int index) const;

  Status allocate_output(int index, const TensorShape& shape,
                         Tensor** tensor) TF_MUST_USE_RESULT;
  //  Status allocate_output(StringPiece name, const TensorShape& shape,
  //                         Tensor** tensor) TF_MUST_USE_RESULT;

  Status allocate_output(int index, const TensorShape& shape, Tensor** tensor,
                         AllocatorAttributes attr) TF_MUST_USE_RESULT;
  //  Status allocate_output(StringPiece name, const TensorShape& shape,
  //                         Tensor** tensor,
  //                         AllocatorAttributes attr) TF_MUST_USE_RESULT;

  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp, AllocatorAttributes allocator_attr,
                       const AllocationAttributes& allocation_attr);
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp, AllocatorAttributes allocator_attr) {
    return allocate_temp(type, shape, out_temp, allocator_attr,
                         AllocationAttributes());
  }
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp) {
    return allocate_temp(type, shape, out_temp, AllocatorAttributes());
  }

  Status allocate_persistent(DataType type, const TensorShape& shape,
                             PersistentTensor* out_persistent,
                             Tensor** out_tensor, AllocatorAttributes attr);
  Status allocate_persistent(DataType type, const TensorShape& shape,
                             PersistentTensor* out_persistent,
                             Tensor** out_tensor) {
    return allocate_persistent(type, shape, out_persistent, out_tensor,
                               AllocatorAttributes());
  }

  //  Status set_output(StringPiece name, const Tensor& tensor);
  //  Status set_output(StringPiece name, Tensor&& tensor);
  void set_output(int index, const Tensor& tensor);
  //  void set_output(int index, Tensor&& tensor);
  //
  //  Status set_output_ref(StringPiece name, mutex* mu, Tensor*
  //  tensor_for_ref);

  // Status mutable_output(StringPiece name, Tensor** tensor);
  Tensor* mutable_output(int index);

  static const Eigen::ThreadPoolDevice& eigen_cpu_device_singleton() {
    static Eigen::ThreadPool threadpool(port::NumSchedulableCPUs());
    static Eigen::ThreadPoolDevice threadpool_device(
        &threadpool,
        (port::NumSchedulableCPUs() + port::NumHyperthreadsPerCore() - 1) /
            port::NumHyperthreadsPerCore());
    return threadpool_device;
  }

  const Eigen::ThreadPoolDevice& eigen_cpu_device() const {
    // TODO(itex): CPU should get thread pool device from local device:
    // *device()->eigen_cpu_device();
    // This helps to identity NUMA affinity.
    return eigen_cpu_device_singleton();
  }

#ifndef INTEL_CPU_ONLY
  const Eigen::GpuDevice& eigen_gpu_device() const {
    return device_.eigen_gpu_device_;
  }
#endif  // INTEL_CPU_ONLY

  template <typename EigenDeviceType>
  const EigenDeviceType& eigen_device() const;

  void CtxFailure(const Status& s);
  void CtxFailureWithWarning(const Status& s);
  void CtxFailure(const char* file, int line, const Status& s);
  void CtxFailureWithWarning(const char* file, int line, const Status& s);

  const Status status() const { return StatusFromTF_Status(status_); }

  void SetStatus(const Status& s);

  static constexpr int kNeverForward = -2;
  static constexpr int kNoReservation = -1;

#ifndef INTEL_CPU_ONLY
  DPCPPStream* GetDeviceStream() const {
    return TF_GetStream(ctx_, status_)->stream_handle;
  }
#else
  void* GetDeviceStream() { return nullptr; }
#endif  // INTEL_CPU_ONLY

  friend void CheckNotInComputeAsync(OpKernelContext* ctx,
                                     const char* correct_macro_name);

  TF_OpKernelContext* Get() { return ctx_; }

 private:
  OpKernelContext() = delete;
  OpKernelContext(const OpKernelContext&) = delete;
  const OpKernelContext& operator=(const OpKernelContext&) = delete;
  TF_OpKernelContext* ctx_;
  // We use single vector inputs_ to store all kinds of input tensors:
  // normal/ref/resource
  mutable gtl::InlinedVector<std::shared_ptr<Tensor>, 4>* inputs_ = nullptr;
  gtl::InlinedVector<std::shared_ptr<Tensor>, 4> outputs_;
  std::map<StringPiece, std::shared_ptr<Tensor>> inputsMap_;
  TF_Status* status_;
  class InternalDevice {
   public:
#ifndef INTEL_CPU_ONLY
    InternalDevice(TF_OpKernelContext* ctx, TF_Status* status)
        : status_(status),
          stream_(ctx,
                  &(static_cast<SP_Stream_st*>(TF_GetStream(ctx, status_))
                        ->stream_handle),
                  &tmp_tensors_),
          eigen_gpu_device_(&stream_) {}
#else
    InternalDevice(TF_OpKernelContext* ctx, TF_Status* status)
        : status_(status) {}
#endif  // INTEL_CPU_ONLY
    ~InternalDevice() {
      for (auto& tensor : tmp_tensors_) {
        if (tensor != nullptr) {
          TF_DeleteTensor(tensor);
          tensor = nullptr;
        }
      }
    }
    TF_Status* status_;
    gtl::InlinedVector<TF_Tensor*, 4> tmp_tensors_;
#ifndef INTEL_CPU_ONLY
    PluginStreamDevice stream_;
    Eigen::GpuDevice eigen_gpu_device_;
#endif  // INTEL_CPU_ONLY
  };
  InternalDevice device_;
};

template <>
const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const;

#ifndef INTEL_CPU_ONLY
template <>
const Eigen::GpuDevice& OpKernelContext::eigen_device() const;
#endif  // INTEL_CPU_ONLY

class PersistentTensor {
 public:
  PersistentTensor() {}
  // Only right value Tensor is allowed to construct PersistentTensor
  explicit PersistentTensor(Tensor&& tensor) : tensor_(std::move(tensor)) {}

  // Caller does not own the returned Tensor*.
  Tensor* AccessTensor(OpKernelConstruction* context);
  // Caller does not own the returned Tensor*.
  Tensor* AccessTensor(OpKernelContext* context);

  // The check for initialization does not need to access the
  // underlying tensor buffer.
  bool IsInitialized() const { return tensor_.IsInitialized(); }

  int64 NumElements() const { return tensor_.NumElements(); }

  int64 AllocatedBytes() const { return tensor_.AllocatedBytes(); }

 private:
  Tensor tensor_;
  explicit PersistentTensor(const Tensor& tensor) = delete;
};

class OpKernelConstruction {
 public:
  //  OpKernelConstruction(DeviceType device_type, DeviceBase* device,
  //                       Allocator* allocator, FunctionLibraryRuntime* flib,
  //                       ResourceMgr* resource_mgr,
  //                       const std::shared_ptr<const NodeProperties>& props,
  //                       const MemoryTypeSlice& input_memory_types,
  //                       const MemoryTypeSlice& output_memory_types,
  //                       int graph_def_version, Status* status);
  //
  explicit OpKernelConstruction(TF_OpKernelConstruction* ctx)
      : device_type_(DeviceType(DEVICE_GPU)),
        status_(TF_NewStatus()),
        ctx_(ctx) {}

  ~OpKernelConstruction() { TF_DeleteStatus(status_); }

  // User-supplied configuration of this operation.
  // const NodeDef& def() const { return props_->node_def; }

  // For inspecting the inputs to this operation.
  const MemoryTypeSlice& input_memory_types() const {
    return input_memory_types_;
  }

  // For inspecting the outputs expected from this operation.
  const MemoryTypeSlice& output_memory_types() const {
    return output_memory_types_;
  }

  // If expected_inputs == inputs() and expected_outputs == output_types(),
  // returns OK, else returns INVALID_ARGUMENT with an error message.
  // Recommended for Ops with dynamic signatures.
  Status MatchSignature(const DataTypeSlice expected_inputs,
                        const DataTypeSlice expected_outputs);

  // For recording configuration errors during construction.
  const Status status() const { return StatusFromTF_Status(status_); }

  void SetStatus(const Status& s);

  // Look up the attr with name attr_name and set *value to its value.  If no
  // attr with attr_name is found in def(), or the attr does not have
  // a matching type, a non-ok status will be returned.
  template <class T>
  Status GetAttr(StringPiece attr_name, T* value) const;

  // Return true if the attr_name is defined in def().
  bool HasAttr(StringPiece attr_name) const;

  // Return the device type.
  const DeviceType& device_type() const { return device_type_; }

  // If not nullptr, the kernel can instantiate functions defined in
  // the library. E.g.,
  // ITEX_CHECK_NOTNULL(function_library())->Instantiate("Foo", ...).
  // FunctionLibraryRuntime* function_library() const { return flib_; }

  // Shared resources accessible to this kernel.
  // ResourceMgr* resource_manager() const { return resource_mgr_; }

  // The GraphDef version whose behavior we should follow.
  // int graph_def_version() const { return graph_def_version_; }

  // Helper routines for the OP_REQUIRES macros
  void CtxFailure(const Status& s);
  void CtxFailureWithWarning(const Status& s);
  void CtxFailure(const char* file, int line, const Status& s);
  void CtxFailureWithWarning(const char* file, int line, const Status& s);

  // Return the unique name of this op
  const char* OpName() const;

  // Unrecommended functions: these are functions that have some
  // current uses but are not recommended for use, and may go away at
  // some future major version release.

  // May be used, e.g., to get GPU handles, etc.
  //
  // Currently only used to call MakeTensorFromProto() for
  // implementing ConstantOp for every device.  See comments
  // on Device::MakeTensorFromProto for longer-term replacement
  // ideas.
  // DeviceBase* device() const { return device_; }

 private:
  OpKernelConstruction(const OpKernelConstruction&) = delete;
  OpKernelConstruction operator=(const OpKernelConstruction&) = delete;
  const DeviceType device_type_;
  // DeviceBase* const device_;
  // Allocator* allocator_;
  // FunctionLibraryRuntime* flib_;
  // ResourceMgr* const resource_mg;r_;
  // std::shared_ptr<const NodeProperties> props_;
  MemoryTypeSlice input_memory_types_;
  MemoryTypeSlice output_memory_types_;
  // const int graph_def_version_;
  TF_Status* status_;

  TF_OpKernelConstruction* ctx_;

  // Allow access from OpKernel ctor.
  //  friend class OpKernel;
};

class OpKernel {
 public:
  explicit OpKernel(OpKernelConstruction* context);
  virtual ~OpKernel() = 0;
  virtual void Compute(OpKernelContext* context) = 0;

  bool IsLegacyScalar(const TensorShape& shape) const {
    return shape.dims() == 0;
  }
  bool IsLegacyVector(const TensorShape& shape) const {
    return shape.dims() == 1;
  }

  // Accessors.
  const absl::string_view name() const { return op_name; }

  const absl::string_view type() const { return op_type; }

  void set_type(absl::string_view type) { op_type = type; }

  std::string ShapeTraceString(const OpKernelContext& ctx) const;

  std::string TraceString(const OpKernelContext& ctx) const;

 private:
  absl::string_view op_name;
  absl::string_view op_type;
};

class KernelDefBuilder {
 public:
  KernelDefBuilder() { priority_ = 0; }
  virtual ~KernelDefBuilder() {}

  KernelDefBuilder& Device(const char* backend);

  template <typename T>
  KernelDefBuilder& TypeConstraint(const char* type);

  KernelDefBuilder& HostMemory(const char* host);

  KernelDefBuilder& Priority(const int32 priority_number);

  // To be compatible with proper's macro, add set method here.
  KernelDefBuilder& KernelClassName(const char* kernel_class_name);

  virtual void Build(const char* device_name, const char* backend) = 0;

  typedef void* (*KernelCreateFunc)(TF_OpKernelConstruction*);
  typedef void (*KernelComputeFunc)(void*, TF_OpKernelContext*);
  typedef void (*KernelDeleteFunc)(void*);

  KernelDefBuilder& RegisterCreate(KernelCreateFunc func);
  KernelDefBuilder& RegisterCompute(KernelComputeFunc func);
  KernelDefBuilder& RegisterDelete(KernelDeleteFunc func);

 protected:
  std::string backend_;
  int32 priority_;
  std::vector<std::string> type_constraints_;
  std::vector<DataType> type_values_;
  std::vector<std::string> host_memorys_;

  KernelCreateFunc create_func_;
  KernelComputeFunc compute_func_;
  KernelDeleteFunc delete_func_;

  // This is not the same with proper's KernelDefBuilder. Due to this args is
  // used in our Build() but proper will not.
  std::string kernel_class_name_;
};

template <typename T>
KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* type) {
  type_constraints_.push_back(std::string(type));
  type_values_.push_back(itex::DataTypeToEnum<T>::v());
  return *this;
}

class Name : public KernelDefBuilder {
 public:
  explicit Name(const char* name);
  ~Name() {}

  void Build(const char* device_name, const char* backend) override;

 private:
  // The op_name is not the same with kernel_class_name. The op_name names a op
  // but the kernel class name is for implementation. The op_name_ will be past
  // to TF_NewKernelBuilder and kernel_class_name_ will be past to
  // TF_RegisterKernelBuilder.
  std::string op_name_;
};

// If node of node_name, experimental_debug_info, node_op, node_device and
// node_attrs has a corresponding kernel registered on device_type, returns OK
// and fill in the kernel def and kernel_class_name. <def> and
// <kernel_class_name> may be null.
Status FindKernelDef(
    const DeviceType& device_type, StringPiece node_name,
    bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    StringPiece node_op, StringPiece node_device, AttrSlice node_attrs,
    const KernelDef** def, std::string* kernel_class_name);

// If node_def has a corresponding kernel registered on device_type,
// returns OK and fill in the kernel def and kernel_class_name. <def> and
// <kernel_class_name> may be null.
Status FindKernelDef(const DeviceType& device_type, const NodeDef& node_def,
                     const KernelDef** def, std::string* kernel_class_name);

namespace register_kernel {
typedef void (*KernelRegisterFunc)(const char*, const char*);
typedef std::pair<std::string, KernelRegisterFunc> Entry;

struct KernelRegistry {
  mutex mu;
  // This registry will saves all kernels' register function. The key
  // of the map is the specialized kernel class's name. And the value
  // of the map is the register func. For instance, if an kernel has
  // double data types with double device. It will generate 4 entryies
  // in this map. All register function can be called. Because every
  // kernel has its own backend, which is stored in the
  // KernelDefBuilder. So we can filter the proper register function
  // during runtime.
  // Use the vector instead of unordered_multimap for simple and
  // performance.
  std::vector<Entry> registry;
};

KernelRegistry* GlobalKernelRegistry();

void RegisterCPUKernels(const char* device_name);
void RegisterGPUKernels(const char* device_name);
void RegisterDefaultKernels();

class Registrar {
 public:
  Registrar(std::string key, KernelRegisterFunc func);
};
}  // namespace register_kernel

// The synchronize method of dpcpp runtime will be called in this function.
//
// There are 3 methods to implement kernels, including
//   1. Using Eigen APIs.
//   2. Using dpcpp APIs.
//   3. Using OneDNN APIs.
// The triple implementation will use the same stream which is the
// `sycl::queue`. So we can directly call the wait on the stream at the end of
// `Compute`. And the kernel will be executed with sync mode.
//
// We use `ITEX_SYNC_EXEC=1` to unify the user interface (env options) for all 3
// types of implementation.
bool IsSyncExecEnabled();
bool IsVerboseEnabled();
inline void RunOrWaitUntilFinish(OpKernelContext* context, OpKernel* op) {
#ifndef INTEL_CPU_ONLY
  if (IsSyncExecEnabled()) {
    auto start = std::chrono::steady_clock::now();
    op->Compute(context);
    auto stream = context->GetDeviceStream();
    auto error = dpcppStreamSynchronize(stream);
    if (error != DPCPP_SUCCESS) {
      context->CtxFailure(
          __FILE__, __LINE__,
          errors::Internal("Error to call the stream's wait with error ",
                           dpcppGetErrorName(error)));
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    if (IsVerboseEnabled()) {
      ITEX_VLOG(0) << op->type() << "," << op->name() << "," << elapsed;
    }
  } else {
    if (IsVerboseEnabled()) {
      auto start = std::chrono::steady_clock::now();
      op->Compute(context);
      auto end = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      ITEX_VLOG(0) << op->type() << "," << op->name() << "," << elapsed;
    } else {
      op->Compute(context);
    }
  }
#else
  op->Compute(context);
#endif
}

class OpTypeFactory {
 public:
  static void RegisterOpType(KernelDefBuilder::KernelCreateFunc func,
                             const std::string& op_type);
  static absl::string_view GetForKernelCreateFunc(
      KernelDefBuilder::KernelCreateFunc);

 private:
  static std::map<KernelDefBuilder::KernelCreateFunc, std::string>*
  GetOpTypeFactory();
  static absl::Mutex op_type_factory_mutex_;
};

// The macro API is the same with TensorFlow proper. As an example, we can
// use it like below,
//
/*
   #define REGISTER_KERNEL(TYPE)                                        \
     REGISTER_KERNEL_BUILDER(                                           \
         Name("Snapshot").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
         SnapshotOp<GPUDevice, TYPE>)
*/
// But should pay attention that, we will use arguments except the first
// as an global key. If you have same key for double registion, the letter
// will overrite the previous one. One solution is you can define an class
// with the Device argument.
#define REGISTER_KERNEL_BUILDER(kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ_HELP(ctr, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ_HELP(ctr, kernel_builder, ...)         \
  static void* Create_##ctr(TF_OpKernelConstruction* ctx) {                 \
    OpKernelConstruction context(ctx);                                      \
    auto kernel = new __VA_ARGS__(&context);                                \
    absl::string_view op_type =                                             \
        OpTypeFactory::GetForKernelCreateFunc(&Create_##ctr);               \
    kernel->set_type(op_type);                                              \
    return kernel;                                                          \
  }                                                                         \
  static void Delete_##ctr(void* kernel) {                                  \
    if (kernel) {                                                           \
      delete (static_cast<__VA_ARGS__*>(kernel));                           \
    }                                                                       \
  }                                                                         \
  static void Compute_##ctr(void* kernel, TF_OpKernelContext* ctx) {        \
    OpKernelContext context(ctx);                                           \
    auto op = static_cast<__VA_ARGS__*>(kernel);                            \
    ITEX_VLOG(3) << "Executing " << op->name() << " with op type "          \
                 << op->type();                                             \
    AnnotatedTraceMe activity(                                              \
        [op, &context] { return op->TraceString(context); });               \
    RunOrWaitUntilFinish(&context, op);                                     \
  }                                                                         \
  static void Register##ctr(const char* device_name, const char* backend) { \
    kernel_builder.KernelClassName(#__VA_ARGS__)                            \
        .RegisterCreate(&Create_##ctr)                                      \
        .RegisterCompute(&Compute_##ctr)                                    \
        .RegisterDelete(&Delete_##ctr)                                      \
        .Build(device_name, backend);                                       \
  }                                                                         \
  TF_ATTRIBUTE_UNUSED static register_kernel::Registrar const               \
      registrar_body_##ctr##_object(#__VA_ARGS__, &Register##ctr);

#define HostMemoryList1(T0) HostMemory(T0)
#define HostMemoryList2(T0, T1) HostMemory(T0).HostMemory(T1)
#define HostMemoryList3(T0, T1, T2) HostMemoryList2(T0, T1).HostMemory(T2)
#define HostMemoryList4(T0, T1, T2, T3) \
  HostMemoryList2(T0, T1).HostMemoryList2(T2, T3)
#define HostMemoryList5(T0, T1, T2, T3, T4) \
  HostMemoryList3(T0, T1, T2).HostMemoryList2(T3, T4)
#define HostMemoryList6(T0, T1, T2, T3, T4, T5) \
  HostMemoryList3(T0, T1, T2).HostMemoryList3(T3, T4, T5)
#define HostMemoryList7(T0, T1, T2, T3, T4, T5, T6) \
  HostMemoryList4(T0, T1, T2, T3).HostMemoryList3(T4, T5, T6)
#define HostMemoryList8(T0, T1, T2, T3, T4, T5, T6, T7) \
  HostMemoryList4(T0, T1, T2, T3).HostMemoryList4(T4, T5, T6, T7)
#define HostMemoryList9(T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  HostMemoryList5(T0, T1, T2, T3, T4).HostMemoryList4(T5, T6, T7, T8)
#define HostMemoryList10(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) \
  HostMemoryList5(T0, T1, T2, T3, T4).HostMemoryList5(T5, T6, T7, T8, T9)
#define HostMemoryList11(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) \
  HostMemoryList6(T0, T1, T2, T3, T4).HostMemoryList5(T6, T7, T8, T9, T10)
#define HostMemoryList12(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) \
  HostMemoryList6(T0, T1, T2, T3, T4, T5)                                  \
      .HostMemoryList6(T6, T7, T8, T9, T10, T11)

inline const Tensor& OpInputList::operator[](int i) const {
  ITEX_DCHECK_GE(i, 0);
  ITEX_DCHECK_LT(i, stop_ - start_);
  return ctx_->input(start_ + i);
}

inline Tensor* OpOutputList::operator[](int i) {
  ITEX_DCHECK_GE(i, 0);
  ITEX_DCHECK_LT(i, stop_ - start_);
  return ctx_->mutable_output(start_ + i);
}

inline Status OpOutputList::allocate(int i, const TensorShape& shape,
                                     Tensor** output) {
  ITEX_DCHECK_GE(i, 0);
  ITEX_DCHECK_LT(i, stop_ - start_);
  return ctx_->allocate_output(start_ + i, shape, output);
}

inline DataType OpOutputList::expected_output_dtype(int i) const {
  ITEX_DCHECK_GE(i, 0);
  ITEX_DCHECK_LT(i, stop_ - start_);
  return ctx_->expected_output_dtype(start_ + i);
}

inline void OpOutputList::set(int i, const Tensor& tensor) {
  ITEX_DCHECK_GE(i, 0);
  ITEX_DCHECK_LT(i, stop_ - start_);
  ctx_->set_output(start_ + i, tensor);
}

inline void OpOutputList::set(int i, Tensor&& tensor) {
  ITEX_DCHECK_GE(i, 0);
  ITEX_DCHECK_LT(i, stop_ - start_);
  ctx_->set_output(start_ + i, std::move(tensor));
}

inline void CheckNotInComputeAsync(OpKernelConstruction*, const char*) {}
void CheckNotInComputeAsync(OpKernelContext* ctx,
                            const char* correct_macro_name);

}  // namespace itex

#endif  // ITEX_CORE_UTILS_OP_KERNEL_H_
