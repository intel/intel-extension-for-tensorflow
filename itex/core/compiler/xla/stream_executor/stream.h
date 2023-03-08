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

// The Stream is used in conjunction with the StreamExecutor "parent" to
// perform actions with a linear stream of dependencies. Dependencies can also
// be created between Streams to do task management (i.e. limit which tasks
// can be performed concurrently and specify what task dependencies exist).

#ifndef ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_STREAM_H_
#define ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_STREAM_H_

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "itex/core/compiler/xla/stream_executor/device_memory.h"
#include "itex/core/compiler/xla/stream_executor/event.h"
#include "itex/core/compiler/xla/stream_executor/kernel.h"
#include "itex/core/compiler/xla/stream_executor/launch_dim.h"
#include "itex/core/compiler/xla/stream_executor/lib/array_slice.h"
#include "itex/core/compiler/xla/stream_executor/platform/port.h"
#include "itex/core/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "itex/core/compiler/xla/stream_executor/temporary_memory_manager.h"

namespace stream_executor {

namespace host {
class HostBlas;
class HostFft;
class HostRng;
class HostTimer;
}  // namespace host

namespace ocl {
class CLBlas;
}  // namespace ocl

namespace internal {
class StreamInterface;
}  // namespace internal

class DeviceMemoryBase;
template <typename ElemT>
class DeviceMemory;

class Timer;

namespace dnn {
class BatchDescriptor;
class FilterDescriptor;
class ConvolutionDescriptor;
class ProfileResult;
class AlgorithmDesc;
}  // namespace dnn

class StreamExecutor;
class ScratchAllocator;

namespace detail {

// Helper class to prevent a template function argument from being deduced. This
// is identical to std::type_identity in C++20.
template <typename T>
struct NonDeduced {
  using type = T;
};
template <typename T>
using NonDeducedType = typename NonDeduced<T>::type;

// Helper to return if `T` is the same type as `First` or any or `Rest`.
template <typename T>
constexpr bool is_any_of() {
  return false;
}

template <typename T, typename First, typename... Rest>
constexpr bool is_any_of() {
  return std::is_same_v<T, First> || is_any_of<T, Rest...>();
}

}  // namespace detail

// Convert a type to the corresponding QuantizedActivationMode.
template <typename ElementType>
struct Quantization;

// Represents a stream of dependent computations on a GPU device.
//
// The operations within a stream execute linearly and asynchronously until
// BlockHostUntilDone() is invoked, which synchronously joins host code with
// the execution of the stream.
//
// If any given operation fails when entraining work for the stream, ok() will
// indicate that an error has occurred. After initialization, once a stream is
// !ok(), it will never be ok().
//
// Thread-safe post-initialization.
class Stream {
 public:
  // Instantiate a stream tied to parent as a platform executor. Work
  // entrained onto this stream will be launched/managed on that
  // StreamExecutor's platform.
  explicit Stream(StreamExecutor* parent);

  // Deallocates any stream resources that the parent StreamExecutor has
  // bestowed
  // upon this object.
  ~Stream();

  // Returns whether any errors have occurred while entraining work for this
  // stream.
  bool ok() const { return !InErrorState(); }

  // Retrieves execution status back into the stream from the underlying
  // implementation without blocking the stream.
  //
  // Normally, Stream::BlockHostUntilDone is used to get execution status.
  // However, some devices use out-of-band mechnanisms to ensure their streams
  // have finished on-device work, without needing to block the streams. (These
  // devices should also override AllowsSyncOnCompletion to return false.) For
  // these devices, this method can be used after work is finished to retrieve
  // execution status.
  port::Status RefreshStatus();  // TF_LOCKS_EXCLUDED(mu_);

  // Initialize the stream. This must be performed before entraining any other
  // operations.
  Stream& Init();  // TF_LOCKS_EXCLUDED(mu_);

  // Initializes timer t via the StreamExecutor.
  Stream& InitTimer(Timer* t);

  // Convenience wrapper around Init() and InitTimer().
  Stream& InitWithTimer(Timer* t);

  // Get or create a sub-stream from this stream. If there is any sub-stream in
  // the pool that can be reused then just return this sub-stream.  Otherwise
  // create a new sub-stream.
  //
  // TODO(b/112196569): The semantics of failed sub-streams is error-prone.
  Stream* GetOrCreateSubStream();  // TF_LOCKS_EXCLUDED(mu_);

  // Return the sub-stream back to the host stream so that it can be reused
  // later. Sub-streams that are !ok() will not be reused.
  //
  // TODO(b/112196569): The semantics of failed sub-streams is error-prone.
  void ReturnSubStream(Stream* sub_stream);  // TF_LOCKS_EXCLUDED(mu_);

  // Allocate temporary memories. The stream will deallocate them when blocked
  // or destroyed.
  template <typename T>
  port::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>>
  AllocateTemporaryArray(uint64_t element_count);

  // Entrains onto the stream of operations: a kernel launch with the given
  // (variadic) parameters for the invocation. These arguments can be things
  // like DeviceMemory or primitive types such as int. What arguments you may
  // pass to a given kernel are noted as the template parameters to the
  // TypedKernel type that the machocc compiler generates.
  //
  // Template parameters:
  //  Params...   The type list of formal parameters that the typed kernel
  //              expects, which is matched against Args...
  //  Args...     The deduced type list for passed actual arguments
  //
  // Implementation: A compile-time compatibility check is performed that has
  // some leniency versus an exact parameter pack match -- for example,
  // `const DeviceMemory<T>` is considered "pack compatible" with a
  // `const DeviceMemory<T>&` formal parameter; in part, because we don't have
  // perfect forwarding support without rvalue references. It also attempts to
  // spit out helpful static_assert error traces with information as to the
  // argument number and types that were mismatched.
  template <typename... Params, typename... Args>
  port::Status ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                          const TypedKernel<Params...>& kernel, Args... args);

  // Record a "start" event for the interval timer at this point in the
  // stream's execution (relative to the previously and subsequently enqueued
  // items in the stream's execution). Streams may be started/stopped multiple
  // times.
  Stream& ThenStartTimer(Timer* t);

  // Record a "stop" event for the interval timer at this point in the
  // stream's execution. See also Stream::ThenStartTimer.
  Stream& ThenStopTimer(Timer* t);

  // TODO(leary) If work is added to the stream that is being depended upon,
  //              then what? Have to describe what happens.
  template <typename... Params>
  Stream& ThenWaitFor(Stream* other, Params... more_streams) {
    return ThenWaitFor(more_streams...).ThenWaitFor(other);
  }

  // Create a dependency for this stream's next work on the other stream
  // completing. Does not take ownership of other, and other must not be
  // null.
  //
  // Checks that a stream does not wait for itself, and it is up to the
  // user to guarantee that a stream does not come to wait on itself in a
  // cyclic manner; in that case, behavior is undefined.
  //
  // N.B. Base recursion case for the variadic ThenWaitFor.
  Stream& ThenWaitFor(Stream* other);

  // Waits for all streams values in others.
  // Checks that there is no shallow circular wait (i.e. that "this" is not in
  // others)
  template <typename P>
  Stream& ThenWaitFor(P others) {
    for (auto& stream : *others) {
      CHECK_NE(stream.get(), this);
      ThenWaitFor(stream.get());
    }
    return *this;
  }

  // Waits for an event object to be set.
  // Note that ThenRecordEvent must have been called on the event before
  // you call this function; otherwise the event will be considered complete
  // and this wait will do nothing.
  Stream& ThenWaitFor(Event* event);

  // Inserts the specified event into the end of this stream. Once the stream
  // has processed all events prior to the insertion point, the event will be
  // marked as completed.
  // The stream does not take ownership of event - meaning that event's lifetime
  // must extend past the point at which it is marked complete!
  Stream& ThenRecordEvent(Event* event);

  // Entrain onto the stream: a memcpy to a host destination from a GPU source
  // of the given target size. host_dst must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate or otherwise allocated and
  // then registered with StreamExecutor::HostMemoryRegister.
  Stream& ThenMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                     uint64_t size);

  // Entrain onto the stream: a memcpy to a GPU destination from a host source
  // of the given target size. host_src must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate or otherwise allocated and
  // then registered with StreamExecutor::HostMemoryRegister.
  Stream& ThenMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                     uint64_t size);

  // Alternative interface for memcpying from device to host that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  Stream& ThenMemcpyD2H(const DeviceMemory<T>& gpu_src,
                        port::MutableArraySlice<T> host_dst) {
    auto host_size = host_dst.size() * sizeof(T);
    ITEX_CHECK(gpu_src.size() == 0 || host_size >= gpu_src.size());
    return ThenMemcpy(host_dst.begin(), gpu_src, host_size);
  }

  // Alternative interface for memcpying from host to device that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  Stream& ThenMemcpyH2D(port::ArraySlice<T> host_src,  // non-absl ok
                        DeviceMemory<T>* gpu_dst) {
    auto host_size = host_src.size() * sizeof(T);
    ITEX_CHECK(gpu_dst->size() == 0 || gpu_dst->size() >= host_size);
    return ThenMemcpy(gpu_dst, host_src.begin(), host_size);
  }

  // Entrain onto the stream: a memcpy to a GPU destination from a GPU source
  // of the given target size. gpu_src/dst must be pointers to GPU memory and
  // peer access must be enabled between their owning StreamExecutors.
  Stream& ThenMemcpy(DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src,
                     uint64_t size);

  // Calls to the device-to-device copy overload of ThenMemcpy -- useful for
  // ensuring that the host pointer isn't getting confused accidentally with a
  // device pointer if you're not doing metaprogramming against the API.
  Stream& ThenMemcpyD2D(DeviceMemoryBase* gpu_dst,
                        const DeviceMemoryBase& gpu_src, uint64_t size) {
    return ThenMemcpy(gpu_dst, gpu_src, size);
  }

  // Entrain onto the stream: a memset of zero at a GPU location of size bytes.
  // The location must not be null.
  Stream& ThenMemZero(DeviceMemoryBase* location, uint64_t size);

  // Entrain onto the stream: a memset of a 32-bit pattern at a GPU location of
  // size bytes, where bytes must be evenly 32-bit sized (i.e. evenly divisible
  // by 4). The location must not be null.
  Stream& ThenMemset32(DeviceMemoryBase* location, uint32_t pattern,
                       uint64_t size);

  // (Synchronously) block the host code waiting for the operations
  // entrained on the stream (enqueued to this point in program
  // execution) to complete.
  //
  // Returns an OK status if the blocking was successful and the stream is ok().
  // Otherwise returns an error describing why the blocking failed.
  port::Status BlockHostUntilDone();  // TF_LOCKS_EXCLUDED(mu_);

  // Warning! This method interacts with internal threads in
  // sometimes-unpredictable ways and is intended for GPU-Executor-internal
  // use
  // only. Please check with a member of the FASTR team before making use of
  // this method.
  //
  // Entrains onto the stream a function to be executed on the host at some
  // point in the future.
  // Async host callbacks DO NOT block the stream as device functions (or as
  // synchronous host callbacks). No synchronization is possible with
  // asynchronous callbacks; they are strictly fire-and-forget.
  // This method is private due to the potential for undefined behavior with
  // synchronization using OpenCL user events.
  // The ONLY lifetime guarantee in these calls is that the StreamExecutor
  // parameter will still be valid - this Stream may not be!
  // Any callbacks requiring device API calls must use this method.
  Stream& ThenEnqueueOnBackgroundThread(
      std::function<void(StreamExecutor*)> task);

  // Returns the (opaque) platform-specific backing object. Ownership is not
  // transferred to the caller.
  internal::StreamInterface* implementation() { return implementation_.get(); }

  // Entrains onto the stream a callback to the host (from the device).
  // Behaves as ThenDoHostCallbackWithStatus below, but the callback should
  // never fail or its failure is inconsequential.
  //
  // This is kept for backward compatibility. Future code should use
  // ThenDoHostCallbackWithStatus and explicitly return a success status.
  // TODO(b/112125301): Eventually remove this method.
  Stream& ThenDoHostCallback(std::function<void()> callback);

  // Entrains onto the stream a callback to the host (from the device).
  // Host callbacks block/occupy the stream just as device functions
  // (execute one at a time, block later stream operations).
  // Whether the callback return status affects the result of BlockHostUntilDone
  // is platform-dependent.
  //
  // Behavior is undefined when synchronizing using OpenCL user events.
  // Behavior is undefined if host callbacks call device routines or insert
  // them into any stream.
  //
  // On certain platforms, ThenDoHostCallback is expected to have significant
  // negative effects on performance.
  Stream& ThenDoHostCallbackWithStatus(std::function<port::Status()> callback);

  // Runs the given callback after the next call to BlockHostUntilDone on this
  // stream (or after the Stream does BlockHostUntilDone in its destructor).
  // This can act as a faster alternative to ThenDoHostCallbackWithStatus for
  // some use cases.
  Stream& ThenRunAfterNextBlockHostUntilDone(std::function<void()> callback);

  // Returns the StreamExecutor (parent object) associated with this stream.
  StreamExecutor* parent() const {
    ITEX_CHECK(parent_ != nullptr);
    return parent_;
  }

  //
  CudaComputeCapability GetCudaComputeCapability() const {
    return parent()->GetDeviceDescription().cuda_compute_capability();
  }

  RocmComputeCapability GetRocmComputeCapability() const {
    return parent()->GetDeviceDescription().rocm_compute_capability();
  }
  // Returns the (internal usage) temporary-memory-allocation manager associated
  // with this stream.
  internal::TemporaryMemoryManager* temporary_memory_manager();

  // Returns a debugging string "[stream=0x...,impl=0x...]".
  std::string DebugStreamPointers() const;

 private:
  friend class host::HostBlas;  // for parent_.
  friend class host::HostFft;   // for parent_.
  friend class host::HostRng;   // for parent_.
  template <typename... Args>
  friend struct ThenBlasImpl;  // for implementing ThenBlasXXX.
  friend class ocl::CLBlas;    // for parent_.

  bool InErrorState() const {  // TF_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock lock(&mu_);
    return !status_.ok();
  }

  // Sets the error state if operation_retcode is false.
  // This is a useful shorthand for many stream routines.
  void CheckError(bool operation_retcode);  // TF_LOCKS_EXCLUDED(mu_);

  // Checks the status and logs the error message, if any.
  void CheckStatus(port::Status status);  // TF_LOCKS_EXCLUDED(mu_);

  void SetError() { CheckError(false /* = operation_retcode */); }

  void SetErrorAndLogNoDnnSupport() {
    SetError();
    ITEX_LOG(WARNING)
        << "attempting to perform DNN operation using StreamExecutor "
           "without DNN support";
  }

  // Runs the set of callbacks that are intended to run after
  // BlockHostUntilDone.
  void RunAfterBlockHostUntilDoneCallbacks();

  // The StreamExecutor that supports the operation of this stream.
  StreamExecutor* parent_;

  // The platform-dependent implementation that the StreamExecutor interface
  // delegates to.
  std::unique_ptr<internal::StreamInterface> implementation_;

  // mutex that guards the allocation / error state flags.
  // Mutable so that it can be obtained via const reader lock.
  mutable absl::Mutex mu_;

  // Whether Init() was successfully called to allocate this stream on the
  // underlying platform. It simply flips from 0 to 1 with a sanity check.
  // See StreamExecutor::AllocateStream.
  bool allocated_ ABSL_GUARDED_BY(mu_);

  // The last error (if any) of all method calls.
  port::Status status_ ABSL_GUARDED_BY(mu_);

  // Sub-streams that are generated from this stream. Each element has a pointer
  // to sub-stream and a boolean value indicating if this substream is ready to
  // be reused.
  std::vector<std::pair<std::unique_ptr<Stream>, bool>> sub_streams_
      ABSL_GUARDED_BY(mu_);

  // Streams can allocate temporary memories to help with work they enqueue
  // (e.g. for scratch memory spaces). This member tracks those allocations and
  // notes when they can be reclaimed -- reclamation is attempted when
  // BlockHostUntilDone() is called.
  internal::TemporaryMemoryManager temporary_memory_manager_;

  // Callbacks enqueued to be run after the next call to BlockHostUntilDone().
  std::vector<std::function<void()>> after_block_host_until_done_callbacks_
      ABSL_GUARDED_BY(mu_);

  // Non-extended BLAS interface requires alpha/beta to be floats when input
  // type is Eigen::half. However, for consistency purposes it is convenient
  // for the interface to accept Eigen::half.
  template <typename T>
  void UpcastHalfToFloat(void** alpha_ptr, void** beta_ptr,
                         float* alpha_storage, float* beta_storage) {
    if (std::is_same<T, Eigen::half>::value) {
      *alpha_storage =
          static_cast<float>(*reinterpret_cast<Eigen::half*>(*alpha_ptr));
      *beta_storage =
          static_cast<float>(*reinterpret_cast<Eigen::half*>(*beta_ptr));
      *alpha_ptr = alpha_storage;
      *beta_ptr = beta_storage;
    } else if (std::is_same<T, Eigen::bfloat16>::value) {
      *alpha_storage =
          static_cast<float>(*reinterpret_cast<Eigen::bfloat16*>(*alpha_ptr));
      *beta_storage =
          static_cast<float>(*reinterpret_cast<Eigen::bfloat16*>(*beta_ptr));
      *alpha_ptr = alpha_storage;
      *beta_ptr = beta_storage;
    }
  }

  SE_DISALLOW_COPY_AND_ASSIGN(Stream);
};

////////////
// Inlines

template <typename... Params, typename... Args>
inline port::Status Stream::ThenLaunch(ThreadDim thread_dims,
                                       BlockDim block_dims,
                                       const TypedKernel<Params...>& kernel,
                                       Args... args) {
  KernelInvocationChecker<std::tuple<Params...>,
                          std::tuple<Args...>>::CheckAllStaticAssert();

  // This is the core that allows type-safe kernel launching.
  // Since the platforms take kernel arguments as tuples of (void *, size),
  // we pack the variadic parameters passed as ...args into the desired
  // tuple form and pass that packed form to the StreamExecutor::Launch()
  // implementation.
  KernelArgsArray<sizeof...(args)> kernel_args;
  kernel.PackParams(&kernel_args, args...);
  TF_RETURN_IF_ERROR(
      parent_->Launch(this, thread_dims, block_dims, kernel, kernel_args));
  return itex::Status::OK();
}

template <typename T>
inline port::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>>
Stream::AllocateTemporaryArray(uint64_t element_count) {
  return temporary_memory_manager_.AllocateArray<T>(element_count);
}

inline internal::TemporaryMemoryManager* Stream::temporary_memory_manager() {
  return &temporary_memory_manager_;
}

}  // namespace stream_executor

#endif  // ITEX_CORE_COMPILER_XLA_STREAM_EXECUTOR_STREAM_H_
