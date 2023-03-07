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
#ifndef ITEX_BUILD_JAX
#ifndef INTEL_CPU_ONLY
#include "itex/core/utils/resource_mgr.h"

#include <algorithm>
#include <atomic>
#include <utility>
#include <vector>

#include "itex/core/utils/demangle.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/gtl/map_util.h"
#include "itex/core/utils/node_def_util.h"
#include "itex/core/utils/scanner.h"
#include "itex/core/utils/stacktrace.h"
#include "itex/core/utils/str_util.h"
#include "itex/core/utils/stringprintf.h"
#include "protos/node_def.pb.h"
namespace itex {

ResourceHandle MakeResourceHandle(
    const string& container, const string& name, const ITEX_GPUDevice& device,
    const TypeIndex& type_index,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes,
    const absl::optional<ManagedStackTrace>& definition_stack_trace) {
  ResourceHandle result;
  result.set_device(device.get_info<sycl::info::device::name>());
  result.set_container(container);
  result.set_definition_stack_trace(definition_stack_trace);
  if (name == ResourceHandle::ANONYMOUS_NAME) {
    result.set_name(
        strings::StrCat("_AnonymousVar", ResourceHandle::GenerateUniqueId()));
  } else {
    result.set_name(name);
  }
  result.set_hash_code(type_index.hash_code());
  result.set_maybe_type_name(type_index.name());
  result.set_dtypes_and_shapes(dtypes_and_shapes);
  return result;
}

Status MakeResourceHandleToOutput(OpKernelContext* context, int output_index,
                                  const string& name,
                                  const TypeIndex& type_index) {
  Tensor* handle;
  TF_RETURN_IF_ERROR(
      context->allocate_output(output_index, TensorShape({}), &handle));
  handle->scalar<ResourceHandle>()() =
      MakeResourceHandle(context->resource_manager()->container(), name,
                         context->GetDeviceStream()->get_device(), type_index);
  return Status();
}

namespace internal {

Status ValidateDevice(OpKernelContext* ctx, const ResourceHandle& p) {
  auto device = ctx->GetDeviceStream()->get_device();
  std::string device_name = device.get_info<sycl::info::device::name>();
  if (device_name != p.device()) {
    return errors::InvalidArgument("Trying to access resource ", p.name(),
                                   " located in device ", p.device(),
                                   " from device ", device_name);
  }
  return Status();
}

}  // end namespace internal

Status ResourceMgr::InsertDebugTypeName(uint64 hash_code,
                                        const string& type_name) {
  auto iter = debug_type_names_.emplace(hash_code, type_name);
  if (iter.first->second != type_name) {
    return errors::AlreadyExists("Duplicate hash code found for type ",
                                 type_name);
  }
  return Status();
}

const char* ResourceMgr::DebugTypeName(uint64 hash_code) const {
  auto type_name_iter = debug_type_names_.find(hash_code);
  if (type_name_iter == debug_type_names_.end()) {
    return "<unknown>";
  } else {
    return type_name_iter->second.c_str();
  }
}

ResourceMgr::ResourceAndName::ResourceAndName() : name(nullptr) {}

ResourceMgr::ResourceAndName::ResourceAndName(const string& name)
    : name(absl::make_unique<string>(name)) {}

core::RefCountPtr<ResourceBase> ResourceMgr::ResourceAndName::GetResource()
    const {
  if (absl::holds_alternative<core::RefCountPtr<ResourceBase>>(resource)) {
    ResourceBase* ptr =
        absl::get<core::RefCountPtr<ResourceBase>>(resource).get();
    ptr->Ref();
    return core::RefCountPtr<ResourceBase>(ptr);
  } else if (absl::holds_alternative<core::WeakPtr<ResourceBase>>(resource)) {
    return absl::get<core::WeakPtr<ResourceBase>>(resource).GetNewRef();
  } else {
    return nullptr;
  }
}

ResourceMgr::ResourceAndName::ResourceAndName(
    ResourceAndName&& other) noexcept {
  name = std::move(other.name);
  resource = std::move(other.resource);
}

ResourceMgr::ResourceAndName::~ResourceAndName() {}

ResourceMgr::ResourceAndName& ResourceMgr::ResourceAndName::operator=(
    ResourceAndName&& other) noexcept {
  name = std::move(other.name);
  resource = std::move(other.resource);
  return *this;
}

ResourceMgr::ResourceMgr() : container_name_("localhost"), step_id(-1) {}

ResourceMgr::~ResourceMgr() { Clear(); }

void ResourceMgr::Clear() {
  // We do the deallocation outside of the lock to avoid a potential deadlock
  // in case any of the destructors access the resource manager.
  {
    mutex_lock l(&mu_);
    if (container_ == nullptr) return;
  }
  container_ = nullptr;
}

string ResourceMgr::DebugString() const {
  mutex_lock l(&mu_);
  struct Line {
    const string* container;
    const string type;
    const string* resource;
    const string detail;
  };
  std::vector<Line> lines;
  for (const auto& q : *container_) {
    const Key& key = q.first;
    const char* type = DebugTypeName(key.first);
    const core::RefCountPtr<ResourceBase> resource = q.second.GetResource();
    Line l{&container_name_, port::Demangle(type), q.second.name.get(),
           resource ? resource->DebugString() : "<nullptr>"};
    lines.push_back(l);
  }

  std::vector<string> text;
  text.reserve(lines.size());
  for (const Line& line : lines) {
    text.push_back(strings::Printf(
        "%-20s | %-40s | %-40s | %-s", line.container->c_str(),
        line.type.c_str(), line.resource->c_str(), line.detail.c_str()));
  }
  std::sort(text.begin(), text.end());
  return absl::StrJoin(text, "\n");
}

Status ResourceMgr::DoCreate(TypeIndex type, const string& name,
                             ResourceBase* resource, bool owns_resource) {
  container_ = [&]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (container_ == nullptr) {
      return std::make_unique<Container>();
    } else {
      return std::move(container_);
    }
  }();

  // NOTE: Separating out the construction of the map key and value so that the
  // key can contain a StringPiece that borrows from the string in the value.
  ResourceAndName resource_and_name(name);

  StringPiece borrowed_name(*resource_and_name.name);

  if (owns_resource) {
    resource_and_name.resource = core::RefCountPtr<ResourceBase>(resource);
  } else {
    auto cleanup_fn = [this, type, borrowed_name]() {
      mutex_lock l(&mu_);
      auto iter = container_->find({type.hash_code(), borrowed_name});
      if (iter != container_->end()) {
        container_->erase(iter);
      }
    };
    resource_and_name.resource =
        core::WeakPtr<ResourceBase>(resource, cleanup_fn);
  }

  Container::value_type key_and_value(Key(type.hash_code(), borrowed_name),
                                      std::move(resource_and_name));
  auto st = container_->insert(std::move(key_and_value));
  if (st.second) {
    TF_RETURN_IF_ERROR(InsertDebugTypeName(type.hash_code(), type.name()));
    return Status();
  }
  return errors::AlreadyExists("Resource ", container_name_, "/", name, "/",
                               type.name());
}

Status ResourceMgr::Lookup(const ResourceHandle& handle,
                           ResourceBase** resource) const {
  tf_shared_lock l(&mu_);
  return DoLookup(handle.hash_code(), /*type_name=*/"ResourceBase",
                  handle.name(), resource);
}

Status ResourceMgr::DoLookup(TypeIndex type, const string& name,
                             ResourceBase** resource) const {
  return DoLookup(type.hash_code(), type.name(), name, resource);
}

Status ResourceMgr::DoLookup(uint64 type_hash_code, const string& type_name,
                             const string& resource_name,
                             ResourceBase** resource) const {
  if (container_ == nullptr) {
    return errors::NotFound("Container ", container_name_,
                            " does not exist. (Could not find resource: ",
                            container_name_, "/", resource_name, ")");
  }
  auto iter = container_->find({type_hash_code, resource_name});
  if (iter == container_->end()) {
    return errors::NotFound("Resource ", container_name_, "/", resource_name,
                            "/", type_name, " does not exist.");
  }
  ResourceBase* ptr = iter->second.GetResource().release();

  if (ptr == nullptr) {
    return errors::NotFound("Resource ", container_name_, "/", resource_name,
                            "/", type_name, " has been destroyed.");
  }
  *resource = ptr;
  return Status();
}

Status ResourceMgr::PopResourceAndName(uint64 type_hash_code,
                                       const string& resource_name,
                                       const string& type_name,
                                       ResourceAndName& resource_and_name) {
  mutex_lock l(&mu_);
  if (container_ == nullptr) {
    return errors::NotFound("Container ", container_name_, " does not exist.");
  }
  auto iter = container_->find({type_hash_code, resource_name});
  if (iter == container_->end()) {
    return errors::NotFound("Resource ", container_name_, "/", resource_name,
                            "/", type_name, " does not exist.");
  }
  std::swap(resource_and_name, iter->second);
  container_->erase(iter);
  return Status();
}

Status ResourceMgr::DoDelete(uint64 type_hash_code, const string& resource_name,
                             const string& type_name) {
  ResourceAndName resource_and_name;
  TF_RETURN_IF_ERROR(PopResourceAndName(type_hash_code, resource_name,
                                        type_name, resource_and_name));

  if (absl::holds_alternative<core::WeakPtr<ResourceBase>>(
          resource_and_name.resource)) {
    return errors::Internal(
        "Cannot delete an unowned Resource ", container_name_, "/",
        resource_name, "/", type_name, " from ResourceMgr. ",
        "This indicates ref-counting ResourceHandle is exposed to weak "
        "ResourceHandle code paths.");
  }
  return Status();
}

Status ResourceMgr::DoDelete(TypeIndex type, const string& resource_name) {
  return DoDelete(type.hash_code(), resource_name, type.name());
}

Status ResourceMgr::Delete(const ResourceHandle& handle) {
  return DoDelete(handle.hash_code(), handle.name(), "<unknown>");
}

const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input) {
  return ctx->input(input).flat<ResourceHandle>()(0);
}

Status HandleFromInput(OpKernelContext* ctx, StringPiece input,
                       ResourceHandle* handle) {
  const Tensor* tensor;
  TF_RETURN_IF_ERROR(ctx->input(input, &tensor));
  *handle = tensor->flat<ResourceHandle>()(0);
  return Status();
}

Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      ResourceBase** value) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  if (p.IsRefCounting()) {
    TF_ASSIGN_OR_RETURN(*value, p.GetResource<ResourceBase>());
    (*value)->Ref();
    return Status();
  }
  return ctx->resource_manager()->Lookup(p, value);
}

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  if (p.IsRefCounting()) {
    return Status();
  }
  return ctx->resource_manager()->Delete(p);
}

}  //  end namespace itex
#endif
#endif  // ITEX_BUILD_JAX
