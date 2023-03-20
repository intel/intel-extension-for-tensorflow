/* Copyright (c) 2023 Intel Corporation

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

#include "itex/core/compiler/xla/stream_executor/sycl/sycl_executor.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#define PATH_MAX MAX_PATH
#else
#include <unistd.h>
#endif
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "itex/core/compiler/xla/stream_executor/kernel_cache_config.h"
#include "itex/core/compiler/xla/stream_executor/lib/env.h"
#include "itex/core/compiler/xla/stream_executor/lib/error.h"
#include "itex/core/compiler/xla/stream_executor/lib/initialize.h"
#include "itex/core/compiler/xla/stream_executor/lib/mathutil.h"
#include "itex/core/compiler/xla/stream_executor/lib/numbers.h"
#include "itex/core/compiler/xla/stream_executor/lib/path.h"
#include "itex/core/compiler/xla/stream_executor/lib/process_state.h"
#include "itex/core/compiler/xla/stream_executor/lib/statusor.h"
#include "itex/core/compiler/xla/stream_executor/platform.h"
#include "itex/core/compiler/xla/stream_executor/platform/logging.h"
#include "itex/core/compiler/xla/stream_executor/platform/port.h"
#include "itex/core/compiler/xla/stream_executor/stream.h"
#include "itex/core/compiler/xla/stream_executor/stream_executor_internal.h"
#include "itex/core/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_event.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_platform_id.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_stream.h"
#include "itex/core/compiler/xla/stream_executor/sycl/sycl_timer.h"
#include "itex/core/compiler/xla/stream_executor/timer.h"
#include "itex/core/devices/gpu/gpu_pool_allocator.h"

// LOG(ERROR) uses a const named ERROR, so a macro with the same name is
// always unwanted. This happens on Windows that defines such a macro.
#undef ERROR

extern bool FLAGS_check_gpu_leaks;
bool FLAGS_prefer_cubin_to_ptx = true;

namespace stream_executor {
namespace gpu {

// Hook that can be used to CUBIN-ate PTX before it is loaded into the driver.
// It has been observed that loading both PTX and cubins into the driver library
// can cause it to crash, but loading only CUBINs avoids those crashes;
// therefore, it's useful to have this hook to hack in uniform CUBIN-ation of
// PTX code.
//
// As this is an implementation-detail workaround, the usage is to declare this
// variable with extern linkage and populate it from another translation unit.
std::function<std::string(const std::string&)> g_cubinate;

static GpuEvent* AsGpuEvent(Event* event) {
  ITEX_DCHECK(event != nullptr);
  return static_cast<GpuEvent*>(event->implementation());
}

// Given a platform-independent timer datatype, returns the internal CUDA
// platform implementation pointer.
static GpuTimer* AsGpuTimer(Timer* timer) {
  ITEX_DCHECK(timer != nullptr);
  return static_cast<GpuTimer*>(timer->implementation());
}

GpuExecutor::~GpuExecutor() {
  ITEX_CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
  ITEX_CHECK(gpu_binary_to_module_.empty())
      << "GpuExecutor has loaded modules.";
}

port::Status GpuExecutor::Init(int device_ordinal,
                               DeviceOptions device_options) {
  device_ordinal_ = device_ordinal;
  device_ = device_ordinal;
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_ordinal);
  sycl_device_ = *device_handle;
  sycl_context_ = ::sycl::context(sycl_device_);
  return port::Status::OK();
}

namespace {
#define L0_SAFE_CALL(call)                      \
  {                                             \
    ze_result_t status = (call);                \
    if (status != 0) {                          \
      ITEX_LOG(FATAL) << "L0 error " << status; \
      exit(1);                                  \
    }                                           \
  }

port::Status LoadLevelzero(const sycl::context& sycl_context,
                           const sycl::device& sycl_device, const char* spir,
                           size_t size, ze_module_handle_t* ze_module) {
  auto ze_device =
      sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(sycl_device);
  auto ze_context =
      sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(sycl_context);

  ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                 nullptr,
                                 ZE_MODULE_FORMAT_IL_SPIRV,
                                 size,
                                 (const uint8_t*)spir,
                                 nullptr,
                                 nullptr};
  L0_SAFE_CALL(
      zeModuleCreate(ze_context, ze_device, &moduleDesc, ze_module, nullptr));
  return port::Status::OK();
}

bool GetModuleFunction(const sycl::context& sycl_context,
                       ze_module_handle_t module, const char* kernel_name,
                       sycl::kernel** sycl_kernel) {
  ITEX_CHECK(module != nullptr && kernel_name != nullptr);
  ze_kernel_handle_t ze_kernel;
  std::string kernel_name_fix = std::string(kernel_name);
  ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                                 kernel_name_fix.c_str()};

#if 1
  bool First = true;
  std::string PINames{""};
  uint32_t Count = 0;
  L0_SAFE_CALL(zeModuleGetKernelNames(module, &Count, nullptr));
  std::unique_ptr<const char*[]> PNames(new const char*[Count]);
  L0_SAFE_CALL(zeModuleGetKernelNames(module, &Count, PNames.get()));
  for (uint32_t I = 0; I < Count; ++I) {
    PINames += (!First ? ";" : "");
    PINames += PNames[I];
    First = false;
  }
  ITEX_VLOG(1) << "Required kernel name: " << kernel_name;
  ITEX_VLOG(1) << "Module has kernel: " << PINames;
#endif
  L0_SAFE_CALL(zeKernelCreate(module, &kernelDesc, &ze_kernel));

  sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>({module},
                                                               sycl_context);
  auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {kernel_bundle, ze_kernel}, sycl_context);
  *sycl_kernel = new sycl::kernel(kernel);
  return true;
}

void UnloadModule(ze_module_handle_t module) {
  if (module) L0_SAFE_CALL(zeModuleDestroy(module));
}

bool GetModuleSymbol(ze_module_handle_t module, const char* symbol_name,
                     size_t* bytes, void** dptr) {
  ITEX_CHECK(module != nullptr && symbol_name != nullptr &&
             (*dptr != nullptr || bytes != nullptr));
  ze_result_t status =
      zeModuleGetGlobalPointer(module, symbol_name, bytes, dptr);
  if (status != ZE_RESULT_SUCCESS) {
    // symbol may not be found in the current module, but it may reside in
    // another module.
    ITEX_VLOG(2) << "failed to get symbol \"" << symbol_name
                 << "\" from module. L0 error: " << status;
    return false;
  }
  return true;
}
#undef L0_SAFE_CALL

}  // namespace

port::Status GpuExecutor::LoadModuleFromSpir(const char* spirv,
                                             const size_t size,
                                             ze_module_handle_t* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[spirv];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(
        LoadLevelzero(sycl_context_, sycl_device_, spirv, size, module));

    module_refcount = 1;
    in_memory_modules_[spirv] = *module;
    ITEX_VLOG(3) << "Loaded SPIRV " << static_cast<const void*>(spirv)
                 << " as module " << *module;
  } else {
    ++module_refcount;
    ITEX_VLOG(3) << "SPIRV " << static_cast<const void*>(spirv)
                 << " is already loaded as module " << *module;
  }
  gpu_binary_to_module_[spirv] = {*module, module_refcount};
  return port::Status::OK();
}

port::Status GpuExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                    KernelBase* kernel) {
  GpuKernel* l0_kernel = AsGpuKernel(kernel);
  ze_module_handle_t module = nullptr;
  string kernelname;

  if (spec.has_cuda_cubin_in_memory()) {
    kernelname = spec.cuda_cubin_in_memory().kernelname();
    const char* spirv = spec.cuda_cubin_in_memory().bytes();
    int size = spec.cuda_cubin_in_memory().size();
    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(LoadModuleFromSpir(spirv, size, &module));
    kernel_to_gpu_binary_[kernel] = spirv;
  } else {
    return port::InternalError("No method of loading SPIR kernel provided");
  }

  ITEX_VLOG(2) << "getting function " << kernelname << " from module "
               << module;
  if (!GetModuleFunction(sycl_context_, module, kernelname.c_str(),
                         l0_kernel->gpu_function_ptr())) {
    return port::InternalError("Failed getting module function");
  }

  // We have to trust the kernel loader spec arity because there doesn't
  // appear to be a way to reflect on the number of expected arguments w/the
  // SPIR API.
  l0_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
  TF_RETURN_IF_ERROR(GetKernelMetadata(l0_kernel, &kernel_metadata));
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(kernelname);
  return port::Status::OK();
}

bool GpuExecutor::UnloadGpuBinary(const void* gpu_binary) {
  auto module_it = gpu_binary_to_module_.find(gpu_binary);
  if (gpu_binary_to_module_.end() == module_it) {
    ITEX_VLOG(3) << "No loaded  SPIR module for " << gpu_binary;
    return false;
  }
  auto& module = module_it->second.first;
  auto& refcount = module_it->second.second;
  ITEX_VLOG(3) << "Found SPIR module " << module << " with refcount "
               << refcount;
  if (--refcount == 0) {
    ITEX_VLOG(3) << "Unloading  SPIR module " << module;
    if (module) zeModuleDestroy(module);
    gpu_binary_to_module_.erase(module_it);
    const char* mem_it = nullptr;
    for (auto x : in_memory_modules_) {
      if (x.second == module) mem_it = x.first;
    }
    if (mem_it != nullptr) {
      in_memory_modules_.erase(mem_it);
      // kernel_index_.erase(mem_it);
    }
  }
  return true;
}

void GpuExecutor::UnloadKernel(const KernelBase* kernel) {
  ITEX_VLOG(3) << "Unloading kernel " << kernel << " : " << kernel->name();

  absl::MutexLock lock{&in_memory_modules_mu_};
  auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
  if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
    ITEX_VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
                 << " has never been loaded.";
    return;  // We've never seen this kernel.
  }
  ITEX_VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
               << " has loaded GPU code " << gpu_binary_it->second;
  UnloadGpuBinary(gpu_binary_it->second);
  kernel_to_gpu_binary_.erase(gpu_binary_it);
}

port::Status GpuExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                     ModuleHandle* module_handle) {
  ze_module_handle_t ze_module = nullptr;
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};

    TF_RETURN_IF_ERROR(LoadModuleFromSpir(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()),
        spec.cuda_cubin_in_memory().size(), &ze_module));
    *module_handle = ModuleHandle(const_cast<void*>(
        static_cast<const void*>(spec.cuda_cubin_in_memory().data())));
    return port::Status::OK();
  } else {
    return port::InternalError("No SPIR binary found");
  }
}

bool GpuExecutor::UnloadModule(ModuleHandle module_handle) {
  const char* gpu_binary = reinterpret_cast<const char*>(module_handle.id());
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(gpu_binary);
}

namespace {
absl::uint128 Fingerprint128(const absl::string_view s) {
  auto fp = itex::Fingerprint128(s);
  return absl::MakeUint128(fp.high64, fp.low64);
}
}  // namespace

port::StatusOr<std::shared_ptr<DeviceMemoryBase>>
GpuExecutor::CreateOrShareConstant(Stream* stream,
                                   const std::vector<uint8_t>& content) {
  absl::MutexLock lock{&shared_constants_mu_};
  // We assume all constants are uniquely identified by this hash. In the
  // (highly unlikely) event of a hash collision, the program will likely crash
  // (because the cached constant that will be returned by mistake is unlikely
  // to have the correct size).
  absl::uint128 fingerprint = Fingerprint128(absl::string_view(
      reinterpret_cast<const char*>(content.data()), content.size()));
  // Must insert nullptr first to get an iterator to the insertion point.
  auto insert_result = shared_constants_.insert(
      {fingerprint, std::weak_ptr<DeviceMemoryBase>()});
  auto it = insert_result.first;
  bool was_already_in_cache = !insert_result.second;
  std::shared_ptr<DeviceMemoryBase> shared_constant;

  if (was_already_in_cache) {
    shared_constant = it->second.lock();
  }

  if (shared_constant == nullptr) {
    // Either the constant wasn't found in the cache, or it was but its
    // weak_ptr had expired.
    DeviceMemoryBase* new_constant =
        new DeviceMemoryBase(Allocate(content.size(), /*memory_space=*/0));
    if (new_constant->opaque() == nullptr) {
      return port::InternalError(absl::StrFormat(
          "Failed to allocate %d bytes for new constant", content.size()));
    }

    port::Status status =
        stream->ThenMemcpy(new_constant, content.data(), content.size())
            .BlockHostUntilDone();
    if (!status.ok()) {
      Deallocate(new_constant);
      status = (port::InternalError(absl::StrFormat(
          "Memcpy to device address %p failed", new_constant->opaque())));
      return status;
    }

    // Capturing 'this' in the custom deleter means this executor must
    // outlive all shared uses of this constant.
    shared_constant = std::shared_ptr<DeviceMemoryBase>(
        new_constant, [this](DeviceMemoryBase* p) {
          Deallocate(p);
          delete p;
        });
    it->second = std::weak_ptr<DeviceMemoryBase>(shared_constant);
  }

  return shared_constant;
}

port::Status GpuExecutor::GetKernelMetadata(GpuKernel* l0_kernel,
                                            KernelMetadata* kernel_metadata) {
  int value = 0;
  // TODO(ITEX): implement this feature in SPIR
  kernel_metadata->set_registers_per_thread(value);

  // TODO(ITEX): implement this feature in SPIR
  kernel_metadata->set_shared_memory_bytes(value);
  return port::Status::OK();
}

port::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const KernelBase& kernel,
                                 const KernelArgsArrayBase& args) {
  ITEX_CHECK_EQ(kernel.Arity(), args.number_of_arguments());
  ITEX_GPUStream* gpu_stream = AsGpuStreamValue(stream);
  // CUstream custream = AsGpuStreamValue(stream);
  const GpuKernel* l0_kernel = AsGpuKernel(&kernel);
  sycl::kernel* l0_func = l0_kernel->AsGpuFunctionHandle();

  // Only perform/print the occupancy check once.  Even just checking to see
  // whether we've done an occupancy check on this kernel before isn't free
  // (because we have to synchronize), so we only do this at -v 2+.
  if (ITEX_VLOG_IS_ON(2)) {
    absl::MutexLock lock(&launched_kernels_mu_);
    if (!launched_kernels_.count(l0_func)) {
      // VlogOccupancyInfo(kernel, thread_dims, block_dims);
      // TODO(rspringer): Remove elements from launched_kernels_...if we ever
      // expose a kernel/module deallocation method.
      launched_kernels_.insert(l0_func);
    }
  }

  std::vector<void*> kernargs;
  KernelArgIterator iter = args.arg_iterator();
  while (iter.has_next()) {
    KernelArg arg = iter.next();
    ITEX_VLOG(2) << "*(arg.address): "
                 << reinterpret_cast<void*>(
                        *static_cast<const uint64_t*>(arg.address));
    kernargs.push_back(
        reinterpret_cast<void*>(*static_cast<const uint64_t*>(arg.address)));
  }

  std::vector<int32_t> sizes(kernargs.size(), sizeof(void*));

  auto sycl_global_range = ::sycl::range<3>(thread_dims.z * block_dims.z,
                                            thread_dims.y * block_dims.y,
                                            thread_dims.x * block_dims.x);
  auto sycl_local_range =
      ::sycl::range<3>(thread_dims.z, thread_dims.y, thread_dims.x);
  sycl::nd_range<3> sycl_nd_range(
      sycl::nd_range<3>(sycl_global_range, sycl_local_range));

  gpu_stream->submit([&](auto& cgh) {
    for (uint32_t i = 0; i < kernargs.size(); i++) {
      cgh.set_arg(i, kernargs.data()[i]);
    }
    cgh.parallel_for(sycl_nd_range, *l0_func);
  });

  return port::Status::OK();
}

DeviceMemoryBase GpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
  ITEX_CHECK_EQ(memory_space, 0);
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_ordinal_);
  std::shared_ptr<itex::BFCAllocator> alloc;
  auto status = ITEX_GPUGetAllocator(device_handle, &alloc);
  ITEX_CHECK(status == ITEX_GPU_SUCCESS)
      << "Failed to get device allocator, device handle: " << device_handle;

  return DeviceMemoryBase(alloc->AllocateRaw(size), size);
}

void* GpuExecutor::GetSubBuffer(DeviceMemoryBase* mem, uint64_t offset_bytes,
                                uint64_t size_bytes) {
  // offset and size are in bytes, so char* works as the pointer type.
  return reinterpret_cast<char*>(mem->opaque()) + offset_bytes;
}

void GpuExecutor::Deallocate(DeviceMemoryBase* mem) {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_ordinal_);
  std::shared_ptr<itex::BFCAllocator> alloc;
  auto status = ITEX_GPUGetAllocator(device_handle, &alloc);
  ITEX_CHECK(status == ITEX_GPU_SUCCESS)
      << "Failed to get device allocator, device handle: " << device_handle;
  alloc->DeallocateRaw(mem->opaque());
}

bool GpuExecutor::HostMemoryRegister(void* location, uint64_t size) {
  return false;
}

bool GpuExecutor::HostMemoryUnregister(void* location) { return false; }

bool GpuExecutor::SynchronizeAllActivity() {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_ordinal_);
  ITEX_GPUCtxSynchronize(device_handle);
  return true;
}

port::Status GpuExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                             uint64_t size) {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_ordinal_);
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    ITEX_GPUMemsetD32(location->opaque(), 0x0, size / 4, device_handle);
  }
  ITEX_GPUMemsetD8(location->opaque(), 0x0, size, device_handle);
  return port::Status::OK();
}

port::Status GpuExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                            int value, uint64_t size) {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_ordinal_);
  ITEX_GPUMemsetD8(location->opaque(), value, size, device_handle);
  return port::Status::OK();
}

port::Status GpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src,
                                            uint64_t size) {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_ordinal_);
  ITEX_GPUMemcpyHtoD(gpu_dst->opaque(), host_src, size, device_handle);
  return port::Status::OK();
}

port::Status GpuExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64_t size) {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_);
  ITEX_GPUMemcpyDtoH(host_dst, gpu_src.opaque(), size, device_handle);
  return port::Status::OK();
}

port::Status GpuExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_);
  ITEX_GPUMemcpyDtoD(gpu_dst->opaque(), gpu_src.opaque(), size, device_handle);
  return port::Status::OK();
}

port::Status GpuExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                  uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    ITEX_GPUMemsetD32Async(location->opaque(), 0x0, size / 4,
                           AsGpuStreamValue(stream));
  } else {
    ITEX_GPUMemsetD8Async(location->opaque(), 0x0, size,
                          AsGpuStreamValue(stream));
  }
  return port::Status::OK();
}

port::Status GpuExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                 uint8_t pattern, uint64_t size) {
  ITEX_VLOG(2) << "enqueueing memset8 operation onto stream " << stream
               << " at location " << location << " with size " << size
               << " and pattern " << std::hex << pattern;
  ITEX_GPUMemsetD8Async(location->opaque(), pattern, size,
                        AsGpuStreamValue(stream));
  return port::Status::OK();
}

port::Status GpuExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                   uint32_t pattern, uint64_t size) {
  ITEX_VLOG(2) << "enqueueing memset32 operation onto stream " << stream
               << " at location " << location << " with size " << size
               << " and pattern " << std::hex << pattern;
  ITEX_CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
             size % 4 == 0);
  ITEX_GPUMemsetD32Async(location->opaque(), pattern, size / 4,
                         AsGpuStreamValue(stream));
  return port::Status::OK();
}

bool GpuExecutor::Memcpy(Stream* stream, void* host_dst,
                         const DeviceMemoryBase& gpu_src, uint64_t size) {
  ITEX_GPUMemcpyDtoHAsync(host_dst, gpu_src.opaque(), size,
                          AsGpuStreamValue(stream));
  return true;
}

bool GpuExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                         const void* host_src, uint64_t size) {
  ITEX_GPUMemcpyHtoDAsync(gpu_dst->opaque(), host_src, size,
                          AsGpuStreamValue(stream));
  return true;
}

bool GpuExecutor::MemcpyDeviceToDevice(Stream* stream,
                                       DeviceMemoryBase* gpu_dst,
                                       const DeviceMemoryBase& gpu_src,
                                       uint64_t size) {
  ITEX_GPUMemcpyDtoDAsync(gpu_dst->opaque(), gpu_src.opaque(), size,
                          AsGpuStreamValue(stream));
  return true;
}

bool GpuExecutor::HostCallback(Stream* stream,
                               std::function<port::Status()> callback) {
  auto callback_ptr = std::function<void()>([callback]() {
    port::Status s = callback();
    if (!s.ok()) {
      ITEX_LOG(WARNING) << "Host callback failed: " << s;
    }
  });

  ITEX_GPUStream* stream_handle = AsGpuStreamValue(stream);
  stream_handle->submit([&](auto& cgh) { cgh.host_task(callback_ptr); });
  return true;
}

port::Status GpuExecutor::AllocateEvent(Event* event) {
  return AsGpuEvent(event)->Init();
}

port::Status GpuExecutor::DeallocateEvent(Event* event) {
  return AsGpuEvent(event)->Destroy();
}

port::Status GpuExecutor::RecordEvent(Stream* stream, Event* event) {
  return AsGpuEvent(event)->Record(AsGpuStream(stream));
}

port::Status GpuExecutor::WaitForEvent(Stream* stream, Event* event) {
  ITEX_GPUStream* stream_handle = AsGpuStreamValue(stream);
  ITEX_GPUEvent* event_handle = AsGpuEvent(event)->gpu_event();
  ITEX_GPUStreamWaitEvent(stream_handle, *event_handle);
  return port::Status::OK();
}

Event::Status GpuExecutor::PollForEventStatus(Event* event) {
  return AsGpuEvent(event)->PollForStatus();
}

bool GpuExecutor::AllocateStream(Stream* stream) {
  absl::MutexLock l(&alive_gpu_streams_mu_);
  bool out = AsGpuStream(stream)->Init();
  alive_gpu_streams_[stream->implementation()->GpuStreamHack()] = stream;
  return out;
}

void GpuExecutor::DeallocateStream(Stream* stream) {
  GpuStream* gpu_stream = AsGpuStream(stream);
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(gpu_stream->GpuStreamHack());
  gpu_stream->Destroy();
}

bool GpuExecutor::AllocateTimer(Timer* timer) {
  return AsGpuTimer(timer)->Init();
}

void GpuExecutor::DeallocateTimer(Timer* timer) {
  AsGpuTimer(timer)->Destroy();
}

bool GpuExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  ITEX_GPUStream* stream_handle1 = AsGpuStreamValue(dependent);
  ITEX_GPUStream* stream_handle2 = AsGpuStreamValue(other);
  ITEX_GPUStreamWaitStream(stream_handle1, stream_handle2);
  return true;
}

bool GpuExecutor::StartTimer(Stream* stream, Timer* timer) {
  return AsGpuTimer(timer)->Start(AsGpuStream(stream));
}

bool GpuExecutor::StopTimer(Stream* stream, Timer* timer) {
  return AsGpuTimer(timer)->Stop(AsGpuStream(stream));
}

port::Status GpuExecutor::BlockHostUntilDone(Stream* stream) {
  ITEX_GPUStream* stream_handle = AsGpuStreamValue(stream);
  stream_handle->wait();
  return port::Status::OK();
}

bool GpuExecutor::CanEnablePeerAccessTo(StreamExecutorInterface* other) {
  return false;
}

port::Status GpuExecutor::EnablePeerAccessTo(StreamExecutorInterface* other) {
  return port::Status::OK();
}

bool GpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  ITEX_GPUDevice* device_handle;
  ITEX_GPUGetDevice(&device_handle, device_);
  *free =
      device_handle->template get_info<sycl::info::device::global_mem_size>();
  *total =
      device_handle->template get_info<sycl::info::device::global_mem_size>();
  return true;
}

bool GpuExecutor::GetSymbol(const std::string& symbol_name,
                            ModuleHandle module_handle, void** mem,
                            size_t* bytes) {
  ITEX_CHECK(static_cast<bool>(module_handle));

  auto lookup_in_module = [&](ze_module_handle_t module) {
    ITEX_CHECK(module != nullptr);
    return GetModuleSymbol(module, symbol_name.c_str(), bytes, mem);
  };

  {  // give limited scope to mutex_lock
    absl::MutexLock lock{&in_memory_modules_mu_};
    auto it = gpu_binary_to_module_.find(module_handle.id());
    ITEX_CHECK(it != gpu_binary_to_module_.end());
    return lookup_in_module(it->second.first);
  }

  ITEX_LOG(INFO) << "Failed to find symbol: " << symbol_name;
  return false;
}

std::unique_ptr<internal::EventInterface>
GpuExecutor::CreateEventImplementation() {
  return std::unique_ptr<internal::EventInterface>(new GpuEvent(this));
}

std::unique_ptr<internal::KernelInterface>
GpuExecutor::CreateKernelImplementation() {
  return std::unique_ptr<internal::KernelInterface>(new GpuKernel());
}

std::unique_ptr<internal::StreamInterface>
GpuExecutor::GetStreamImplementation() {
  return std::unique_ptr<internal::StreamInterface>(new GpuStream(this));
}

std::unique_ptr<internal::TimerInterface>
GpuExecutor::GetTimerImplementation() {
  return std::unique_ptr<internal::TimerInterface>(new GpuTimer(this));
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
GpuExecutor::CreateDeviceDescription(int device_ordinal) {
  internal::DeviceDescriptionBuilder builder;
  builder.set_device_vendor("INTEL Corporation");
  return builder.Build();
}

}  // namespace gpu

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(cuda_gpu_executor, {});
