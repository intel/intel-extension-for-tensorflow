/* Copyright (c) 2022 Intel Corporation

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

#ifndef ITEX_CORE_UTILS_RESOURCE_MGR_H_
#define ITEX_CORE_UTILS_RESOURCE_MGR_H_
#ifndef ITEX_BUILD_JAX
#ifndef INTEL_CPU_ONLY

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/variant.h"
#include "itex/core/utils/common_shape_fns.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/hash.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/macros.h"
#include "itex/core/utils/mutex.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/resource_base.h"
#include "itex/core/utils/resource_handle.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/thread_annotations.h"
#include "itex/core/utils/type_index.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

namespace itex {

// A ResourceMgr instance keeps track of named and typed resources
// grouped into containers.
//
// Each named resource is
// registered with ResourceMgr under a named "container" name. At any
// time, there is at most one instance of a resource given the container
// name, the resource type and the resource name.
//
// All resources for a given container can be dropped by one call of
// Cleanup().

class ResourceMgr {
 public:
  ResourceMgr();
  ~ResourceMgr();

  // Returns the container name for *this.
  const std::string& container() const { return container_name_; }

  // Creates a resource "name" in the "container".  The caller transfers
  // the ownership of one ref on "resource" to *this, regardless of whether this
  // operation succeeds or fails.
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr.
  template <typename T>
  Status Create(const std::string& name, T* resource,
                int64_t current_step_id) TF_MUST_USE_RESULT;

  // Creates a unowned resource "name" in the "container".  The caller does NOT
  // transfer the ownership of any ref on "resource" to *this, regardless of
  // whether this operation succeeds or fails.
  //
  // After the resource is destroyed, lookups from the manager fail.
  // The caller must call this->Delete() on the name to free up the memory
  // entry of the name.
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr.
  template <typename T>
  Status CreateUnowned(const std::string& name, T* resource) TF_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in "*resource" and
  // the caller takes the ownership of one ref on "*resource".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  template <typename T, bool use_dynamic_cast = false>
  Status Lookup(const std::string& name, T** resource) const TF_MUST_USE_RESULT;

  // If the resource manager has a resource matching "handle", returns it in
  // "*resource" and the caller takes the ownership of one ref on "*resource".
  //
  // REQUIRES: resource != nullptr
  Status Lookup(const ResourceHandle& handle,
                ResourceBase** resource) const TF_MUST_USE_RESULT;

  // Similar to Lookup, but looks up multiple resources at once, with only a
  // single lock acquisition.  If containers_and_names[i] is uninitialized
  // then this function does not modify resources[i].
  template <typename T, bool use_dynamic_cast = false>
  Status LookupMany(absl::Span<const string*> names,
                    std::vector<std::unique_ptr<T, core::RefCountDeleter>>*
                        resources) const TF_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in
  // "*resource". Otherwise, invokes creator() to create the resource.
  // The caller takes the ownership of one ref on "*resource".
  //
  // WARNING: creator() must not call any methods on ResourceMgr during its
  // execution, because a non-reentrant lock is held during the creator() call
  // in order to guarantee atomicity of LookupOrCreate().
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  template <typename T, bool use_dynamic_cast = false>
  Status LookupOrCreate(const std::string& name, T** resource,
                        std::function<Status(T**)> creator) TF_MUST_USE_RESULT;

  // Deletes the resource "name" from the "container".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  template <typename T>
  Status Delete(const std::string& name) TF_MUST_USE_RESULT;

  // Deletes the resource pointed by "handle".
  Status Delete(const ResourceHandle& handle) TF_MUST_USE_RESULT;

  // Delete all resources in the container
  void Clear();

  // Returns a text description for all resources.
  std::string DebugString() const;

  template <typename T>
  ResourceHandle MakeResourceHandle(const std::string& name,
                                    const ITEX_GPUDevice& device);

 private:
  typedef std::pair<uint64, StringPiece> Key;
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return Hash64(k.second.data(), k.second.size(), k.first);
    }
  };
  struct KeyEqual {
    bool operator()(const Key& x, const Key& y) const {
      return (x.second == y.second) && (x.first == y.first);
    }
  };
  struct ResourceAndName {
    absl::variant<core::RefCountPtr<ResourceBase>, core::WeakPtr<ResourceBase>>
        resource;
    std::unique_ptr<std::string> name;

    ResourceAndName();
    explicit ResourceAndName(const string& name);
    ResourceAndName(ResourceAndName&& other) noexcept;
    ~ResourceAndName();

    ResourceAndName& operator=(ResourceAndName&&) noexcept;

    // Returns a strong reference to resource, or nullptr if the resource is
    // no longer valid.
    core::RefCountPtr<ResourceBase> GetResource() const;

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(ResourceAndName);
  };
  typedef absl::flat_hash_map<Key, ResourceAndName, KeyHash, KeyEqual>
      Container;

  mutable mutex mu_;
  const std::string container_name_;
  Container* container_ = nullptr;
  int64_t step_id;

  template <typename T, bool use_dynamic_cast = false>
  Status LookupInternal(const std::string& name, T** resource) const
      TF_SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;
  Status LookupInternal(uint64 type_hash_code, const std::string& name,
                        ResourceBase** resource) const
      TF_SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  Status DoCreate(TypeIndex type, const std::string& name,
                  ResourceBase* resource, bool owns_resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  Status DoLookup(TypeIndex type, const std::string& name,
                  ResourceBase** resource) const
      TF_SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;
  Status DoLookup(uint64 type_hash_code, const std::string& type_name,
                  const std::string& resource_name,
                  ResourceBase** resource) const
      TF_SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  Status DoDelete(uint64 type_hash_code, const std::string& resource_name,
                  const std::string& type_name) TF_MUST_USE_RESULT;
  Status DoDelete(TypeIndex type,
                  const std::string& resource_name) TF_MUST_USE_RESULT;

  // Pops the ResourceAndName entry. The entry is moved from the list to
  // the output argument `resource_and_name`.
  Status PopResourceAndName(
      uint64 type_hash_code, const std::string& resource_name,
      const std::string& type_name,
      ResourceAndName& resource_and_name) TF_MUST_USE_RESULT;  // NOLINT
  // Inserts the type name for 'hash_code' into the hash_code to type name map.
  Status InsertDebugTypeName(uint64 hash_code, const std::string& type_name)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  // Returns the type name for the 'hash_code'.
  // Returns "<unknown>" if a resource with such a type was never inserted into
  // the container.
  const char* DebugTypeName(uint64 hash_code) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Map from type hash_code to type name.
  std::unordered_map<uint64, string> debug_type_names_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceMgr);
};

// Makes a resource handle with the specified type for a given container /
// name.
ResourceHandle MakeResourceHandle(
    const string& container, const std::string& name,
    const ITEX_GPUDevice& device, const TypeIndex& type_index,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes = {},
    const absl::optional<ManagedStackTrace>& definition_stack_trace = {})
    TF_MUST_USE_RESULT;

template <typename T>
ResourceHandle MakeResourceHandle(
    OpKernelContext* ctx, const std::string& name,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes = {},
    const absl::optional<ManagedStackTrace>& definition_stack_trace = {}) {
  return MakeResourceHandle(ctx->resource_manager()->container(), name,
                            ctx->GetDeviceStream()->get_device(),
                            TypeIndex::Make<T>(), dtypes_and_shapes,
                            definition_stack_trace);
}

Status MakeResourceHandleToOutput(OpKernelContext* context, int output_index,
                                  const std::string& name,
                                  const TypeIndex& type_index);

// Returns a resource handle from a numbered op input.
const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input);
Status HandleFromInput(OpKernelContext* ctx, StringPiece input,
                       ResourceHandle* handle);

// Create a resource pointed by a given resource handle.
//
// If successful, the caller transfers the ownership of one ref on `resource` to
// `ctx->resource_mgr()`.
template <typename T>
Status CreateResource(OpKernelContext* ctx, const ResourceHandle& p, T* value);

// Looks up a resource pointed by a given resource handle.
//
// If the lookup is successful, the caller takes the ownership of one ref on
// `*value`, and must call its `Unref()` method when it has finished using it.
template <typename T, bool use_dynamic_cast = false>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p, T** value);

// Looks up a resource pointed by a given resource handle.
//
// Prefer usage of LookupResource taking `core::RefCountPtr` to avoid
// requiring the caller to explicitly call `Unref()`.
template <typename T>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      core::RefCountPtr<T>* value);

// Looks up multiple resources pointed by a sequence of resource handles.  If
// p[i] is uninitialized then values[i] is unmodified.
template <typename T>
Status LookupResources(OpKernelContext* ctx, absl::Span<ResourceHandle const> p,
                       std::vector<core::RefCountPtr<T>>* values);

// Looks up or creates a resource.
//
// If successful, the caller takes the ownership of one ref on `*value`, and
// must call its `Unref()` method when it has finished using it. If the
// `creator` is invoked, its reference on the created resource is transferred
// to `ctx->resource_mgr()`.
//
// Prefer usage of LookupOrCreateResource taking `core::RefCountPtr` to avoid
// requiring the caller to explicitly call `Unref()`.
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              T** value, std::function<Status(T**)> creator);

// Looks up or creates a resource.
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              core::RefCountPtr<T>* value,
                              std::function<Status(T**)> creator);

// Destroys a resource pointed by a given resource handle.
template <typename T>
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

// Same as above, but uses the hash code of the type directly.
// The type name information will be missing in the debug output when the
// resource is not present in the container.
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

// Helper for kernels to obtain 'resource' from the
// ctx->resource_manager().
//
// "input_name" specifies the kernel's ref input which gives a string
// tensor with two elements, which specifies the container and
// resource name.
//
// Returns OK if the resource is found and transfers one ref of
// *resource to the caller. Otherwise, returns an error.
template <typename T>
Status GetResourceFromContext(OpKernelContext* ctx,
                              const std::string& input_name, T** resource);

// Implementation details below.

template <typename T>
void CheckDeriveFromResourceBase() {
  static_assert(std::is_base_of<ResourceBase, T>::value,
                "T must derive from ResourceBase");
}

template <typename T>
Status ResourceMgr::Create(const std::string& name, T* resource,
                           int64_t current_step_id) {
  CheckDeriveFromResourceBase<T>();
  ITEX_CHECK(resource != nullptr);
  mutex_lock l(&mu_);
  if (step_id != current_step_id) {
    // clear resources of last step
    if (container_) {
      delete container_;
      container_ = nullptr;
    }
    step_id = current_step_id;
  }

  return DoCreate(TypeIndex::Make<T>(), name, resource,
                  /* owns_resource */ true);
}

template <typename T>
ResourceHandle ResourceMgr::MakeResourceHandle(const std::string& name,
                                               const ITEX_GPUDevice& device) {
  mutex_lock l(&mu_);
  return itex::MakeResourceHandle(container_name_, name, device,
                                  TypeIndex::Make<T>(), {});
}

template <typename T>
Status ResourceMgr::CreateUnowned(const std::string& name, T* resource) {
  CheckDeriveFromResourceBase<T>();
  mutex_lock l(&mu_);
  return DoCreate(TypeIndex::Make<T>(), name, resource,
                  /* owns_resource */ false);
}

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::Lookup(const std::string& name, T** resource) const {
  CheckDeriveFromResourceBase<T>();
  tf_shared_lock l(&mu_);
  return LookupInternal<T, use_dynamic_cast>(name, resource);
}

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupMany(
    absl::Span<const string*> names,
    std::vector<std::unique_ptr<T, core::RefCountDeleter>>* resources) const {
  CheckDeriveFromResourceBase<T>();
  tf_shared_lock l(&mu_);
  resources->resize(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    T* resource;
    Status s = LookupInternal<T, use_dynamic_cast>(*names[i], &resource);
    if (s.ok()) {
      (*resources)[i].reset(resource);
    }
  }
  return Status();
}

// Simple wrapper to allow conditional dynamic / static casts.
template <typename T, bool use_dynamic_cast>
struct TypeCastFunctor {
  static T* Cast(ResourceBase* r) { return static_cast<T*>(r); }
};

template <typename T>
struct TypeCastFunctor<T, true> {
  static T* Cast(ResourceBase* r) { return dynamic_cast<T*>(r); }
};

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupInternal(const std::string& name,
                                   T** resource) const {
  ResourceBase* found = nullptr;
  Status s = DoLookup(TypeIndex::Make<T>(), name, &found);
  if (s.ok()) {
    // It's safe to down cast 'found' to T* since
    // typeid(T).hash_code() is part of the map key.
    *resource = TypeCastFunctor<T, use_dynamic_cast>::Cast(found);
  }
  return s;
}

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupOrCreate(const std::string& name, T** resource,
                                   std::function<Status(T**)> creator) {
  CheckDeriveFromResourceBase<T>();
  *resource = nullptr;
  Status s;
  {
    tf_shared_lock l(&mu_);
    s = LookupInternal<T, use_dynamic_cast>(name, resource);
    if (s.ok()) return s;
  }
  mutex_lock l(&mu_);
  s = LookupInternal<T, use_dynamic_cast>(name, resource);
  if (s.ok()) return s;
  TF_RETURN_IF_ERROR(creator(resource));
  s = DoCreate(TypeIndex::Make<T>(), name, *resource,
               /* owns_resource */ true);
  if (!s.ok()) {
    return errors::Internal("LookupOrCreate failed unexpectedly");
  }
  (*resource)->Ref();
  return s;
}

template <typename T>
Status ResourceMgr::Delete(const std::string& name) {
  CheckDeriveFromResourceBase<T>();
  return DoDelete(TypeIndex::Make<T>(), name);
}

namespace internal {

Status ValidateDevice(OpKernelContext* ctx, const ResourceHandle& p);

template <typename T>
Status ValidateDeviceAndType(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  TF_RETURN_IF_ERROR(p.ValidateType<T>());
  return Status();
}

}  // namespace internal

// Creates the resource pointed at by "p". The caller transfers the ownership of
// one ref on "*value" to the resource manager in "ctx", regardless of whether
// this operation succeeds or fails.
template <typename T>
Status CreateResource(OpKernelContext* ctx, const ResourceHandle& p, T* value) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->Create(p.name(), value);
}

// Finds the resource as "*value" from the handle. If the handle is
// ref-counting, returns the resource owned by the handle. Otherwise, looks up
// the resource matching "p" from resource manager associated with ctx.
// Always returns a new reference to the resource in "*value". The caller shall
// call (*value)->Unref().
template <typename T, bool use_dynamic_cast>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      T** value) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  if (p.IsRefCounting()) {
    TF_ASSIGN_OR_RETURN(*value, p.GetResource<T>());
    // Transfers out a new reference.
    (*value)->Ref();
    return Status();
  }

  return ctx->resource_manager()->Lookup<T, use_dynamic_cast>(p.name(), value);
}

// Finds the resource as "*value" from the handle. This is a type-erased
// variant of LookupResource above.
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      ResourceBase** value);

// If the resource manager in "ctx" has a resource matching "p", returns it in
// "*value".
template <typename T>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      core::RefCountPtr<T>* value) {
  T* raw_ptr = nullptr;
  TF_RETURN_IF_ERROR(LookupResource<T, false>(ctx, p, &raw_ptr));
  value->reset(raw_ptr);

  return Status();
}

// Similar to Lookup, but looks up multiple resources at once, with only a
// single lock acquisition.
template <typename T>
Status LookupResources(OpKernelContext* ctx,
                       absl::Span<ResourceHandle const* const> p,
                       std::vector<core::RefCountPtr<T>>* values) {
  std::vector<const string*> names(p.size());
  for (size_t i = 0; i < p.size(); ++i) {
    TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, *p[i]));
    names[i] = &p[i]->name();
  }
  return ctx->resource_manager()->LookupMany(names, values);
}

// If the resource manager in "ctx" has a resource pointed at by "p", returns
// it in "*value". Otherwise, invokes creator() to create the resource.
// The caller takes the ownership of one ref on "*value".
//
// WARNING: creator() must not call any methods on the resource manager during
// its execution, because a non-reentrant lock is held during the creator() call
// in order to guarantee atomicity of LookupOrCreateResource().
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              T** value, std::function<Status(T**)> creator) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->LookupOrCreate(p.name(), value, creator);
}

// If the resource manager in "ctx" has a resource pointed at by "p", returns
// it in "*value". Otherwise, invokes creator() to create the resource.
//
// WARNING: creator() must not call any methods on the resource manager during
// its execution, because a non-reentrant lock is held during the creator() call
// in order to guarantee atomicity of LookupOrCreateResource().
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              core::RefCountPtr<T>* value,
                              std::function<Status(T**)> creator) {
  T* raw_ptr = nullptr;
  TF_RETURN_IF_ERROR(LookupOrCreateResource<T>(ctx, p, &raw_ptr, creator));
  value->reset(raw_ptr);

  return Status();
}

// Deletes the resource pointed by "p", using the resource manager in "ctx".
template <typename T>
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  // This is a noop because ResourceMgr does not hold a reference.
  // NOTE(feyu): if we can convert all resources handle to ref-counting, then
  // DeleteResource can be removed.
  if (p.IsRefCounting()) {
    return Status();
  }
  return ctx->resource_manager()->Delete<T>(p.name());
}

// Deletes the resource pointed by "p", using the resource manager in "ctx".
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

}  //  end namespace itex

#endif
#endif
#endif  // ITEX_CORE_UTILS_RESOURCE_MGR_H_
