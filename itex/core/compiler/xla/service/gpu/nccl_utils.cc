/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/compiler/xla/service/gpu/nccl_utils.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "itex/core/compiler/xla/debug_options_flags.h"
#include "itex/core/compiler/xla/service/global_device_id.h"
#include "itex/core/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "itex/core/compiler/xla/status_macros.h"
#include "itex/core/compiler/xla/statusor.h"
#include "itex/core/utils/env.h"

namespace itex_xla {
namespace gpu {

bool IsGlobalNcclConfig() {
  static const bool global_nccl_config = std::getenv("NCCL_COMM_ID") != nullptr;
  return global_nccl_config;
}

bool IsNcclLaunchModeParallel() {
  static const bool is_launch_mode_parallel =
      absl::string_view(std::getenv("NCCL_LAUNCH_MODE")) == "PARALLEL";
  return is_launch_mode_parallel;
}

#if ITEX_USE_CCL
ccl::reduction ToNcclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ccl::reduction::sum;
    case ReductionKind::PRODUCT:
      return ccl::reduction::prod;
    case ReductionKind::MIN:
      return ccl::reduction::min;
    case ReductionKind::MAX:
      return ccl::reduction::max;
  }
}
#endif  // ITEX_USE_CCL

namespace {
#if ITEX_USE_CCL
StatusOr<ccl::datatype> ToNcclDataType(PrimitiveType element_type) {
  switch (element_type) {
    case S8:
      return ccl::datatype::int8;
    case PRED:
    case U8:
      return ccl::datatype::uint8;
    case S32:
      return ccl::datatype::int32;
    case U32:
      return ccl::datatype::uint32;
    case S64:
      return ccl::datatype::int64;
    case U64:
      return ccl::datatype::uint64;
    case F16:
      return ccl::datatype::float16;
    case F32:
    case C64:
      return ccl::datatype::float32;
    case F64:
    case C128:
      return ccl::datatype::float64;
    case BF16:
      return ccl::datatype::bfloat16;
    default:
      return itex::errors::InvalidArgument(absl::StrFormat(
          "Unsupported data type: %s", PrimitiveType_Name(element_type)));
  }
}

StatusOr<ccl::kvs::address_type> ToNcclUniqueId(const std::string& id_str) {
  // static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES,
  //               "NCCL_UNIQUE_ID_BYTES");

  // TF_RET_CHECK(id_str.size() == NCCL_UNIQUE_ID_BYTES);
  ccl::kvs::address_type id;
  // absl::c_copy(id_str, id.internal);
  std::copy(id_str.begin(), id_str.end(), id.data());
  return id;
}
#else
StatusOr<std::string> ToNcclUniqueId(const std::string& id_str) {
  return id_str;
}
#endif  // ITEX_USE_CCL

template <typename K, typename V>
class ThreadSafeMap {
 public:
  V& operator[](const K& key) {
    absl::MutexLock lock(&mutex_);
    std::unique_ptr<V>& value = map_[key];
    if (value == nullptr) value = std::make_unique<V>();
    return *value;
  }

  void ForEachValue(const std::function<void(V&)>& fn) {
    absl::MutexLock lock(&mutex_);
    for (const auto& it : map_) fn(*it.second);
  }

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<K, std::unique_ptr<V>> map_ ABSL_GUARDED_BY(mutex_);
};

StatusOr<std::string> LocalNcclUniqueIdCallback(const NcclCliqueKey&) {
#if ITEX_USE_CCL
  // ccl::kvs::address_type id;
  // XLA_CUDA_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  auto id = ccl::create_main_kvs()->get_address();
  return std::string(id.begin(), id.end());
#else
  return std::string("");
#endif  // ITEX_USE_CCL
}

void WaitAndLogIfStuck(absl::Mutex& mutex, const absl::Condition& condition) {
  constexpr absl::Duration kTimeout = absl::Seconds(10);
  if (mutex.AwaitWithTimeout(condition, kTimeout)) {
    return;
  }

  ITEX_LOG(ERROR) << "This thread has been waiting for "
                  << absl::ToInt64Seconds(kTimeout) << "s and may be stuck:";

  int64_t termination_timeout = itex_xla::GetDebugOptionsFromFlags()
                                    .xla_gpu_nccl_termination_timeout_seconds();
  // infinite timeout is equivalent to await call without timeout.
  absl::Duration kTerminationTimeout = termination_timeout >= 0
                                           ? absl::Seconds(termination_timeout)
                                           : absl::InfiniteDuration();

  if (mutex.AwaitWithTimeout(condition, kTerminationTimeout)) {
    ITEX_LOG(ERROR) << "Thread is unstuck! Warning above was a false-positive. "
                       "Perhaps the timeout is too short.";
    return;
  }
  ITEX_LOG(ERROR)
      << "Termination timeout of " << termination_timeout
      << " seconds exceeded. Exiting to ensure a consistent program state.";
  std::exit(42);
}

// A rendezvous for a group of threads.
//
// The group of threads identifies itself with a key that must be unique to the
// the group. When all threads have arrived at the rendezvous, one thread
// executes the given function and all threads received the result.
// TODO(cjfj): Replace XLA rendezvous code with this simpler implementation.
template <typename R, typename K>
std::shared_ptr<R> Rendezvous(const K& key, size_t num_threads,
                              const std::function<R()>& fn) {
  // Fast-path (DO NOT REMOVE: the logic below doesn't work for single thread).
  if (num_threads == 1) return std::make_shared<R>(fn());

  struct State {
    absl::Mutex mutex;
    size_t num_threads_arrived ABSL_GUARDED_BY(mutex) = 0;
    std::shared_ptr<R> result ABSL_GUARDED_BY(mutex);
  };

  static auto& states = *new ThreadSafeMap<K, State>;
  State& state = states[key];

  absl::MutexLock lock(&state.mutex);
  ++state.num_threads_arrived;

  std::shared_ptr<R> result;
  if (state.num_threads_arrived == num_threads) {
    // Last thread to arrive executes the function.
    ITEX_CHECK(state.result == nullptr);
    result = std::make_shared<R>(fn());
    state.result = result;
    state.num_threads_arrived = 0;
  } else {
    absl::Condition result_ready(
        +[](std::shared_ptr<R>* ptr) { return ptr->get() != nullptr; },
        &state.result);
    WaitAndLogIfStuck(state.mutex, result_ready);

    // There is one use of the result in the shared state, plus one use for each
    // thread that has already retrieved the result.
    if (state.result.use_count() < num_threads) {
      result = state.result;
    } else {
      // Last thread to retrieve the result takes the result from the state,
      // allowing the other threads to exit the function.
      return std::move(state.result);
    }
  }

  // Wait for all threads to have retrieved the result. Without this, a thread
  // could duplicate or delete its copy of the result, invalidating the use
  // count logic above.
  absl::Condition result_taken(
      +[](std::shared_ptr<R>* ptr) { return ptr->get() == nullptr; },
      &state.result);
  WaitAndLogIfStuck(state.mutex, result_taken);
  return result;
}

struct NcclCliqueState {
#if ITEX_USE_CCL
  ccl::kvs::address_type unique_id;
#else
  std::string unique_id;
#endif  // ITEX_USE_CCL
  int64_t run_id = -1;
};

using NcclClique = Lockable<NcclCliqueState>;

std::shared_ptr<StatusOr<NcclClique::Lock>> AcquireNcclClique(
    RunId run_id, OpId op_id, NcclCliqueKey clique_key,
    const NcclUniqueIdCallback& unique_id_callback,
    size_t num_local_participants) {
  static auto& cliques = *new ThreadSafeMap<NcclCliqueKey, NcclClique>;

  auto rendezvous_key = std::make_tuple(run_id, op_id, std::move(clique_key));

  return Rendezvous<StatusOr<NcclClique::Lock>>(
      rendezvous_key, num_local_participants,
      [&]() -> StatusOr<NcclClique::Lock> {
        const NcclCliqueKey& clique_key = std::get<2>(rendezvous_key);
        NcclClique::Lock clique = cliques[clique_key].Acquire();
        if (clique->run_id < 0) {
          // set run_is as unique_id
          // TF_ASSIGN_OR_RETURN(std::string id,
          // unique_id_callback(clique_key));
          std::string id = run_id.ToString();
          TF_ASSIGN_OR_RETURN(clique->unique_id, ToNcclUniqueId(id));
        }
        // If multiple executable are running simultaneously while using
        // multiple hosts, it is possible that different executables could
        // acquire the same clique on different hosts. We protect against this
        // by checking that the run ID increases monotonically.
        bool is_local = clique_key.devices().size() == num_local_participants;
        TF_RET_CHECK(is_local || (run_id.ToInt() >= clique->run_id));
        clique->run_id = run_id.ToInt();
        return clique;
      });
}

/*
void CheckNcclAsyncError(NcclComm& lockable_comm) {
  ccl::communicator* comm = *lockable_comm.Acquire();
  if (comm == nullptr) return;

  Status status = [comm] {
    ncclResult_t async_err;
    XLA_CUDA_RETURN_IF_ERROR(ncclCommGetAsyncError(comm, &async_err));
    if (async_err != ncclSuccess) {
      ITEX_LOG(ERROR) << "Aborting communicator: " << comm
                 << " due to async NCCL error: "
                 << ncclGetErrorString(async_err);
      XLA_CUDA_RETURN_IF_ERROR(ncclCommAbort(comm));
    }
    return XLA_CUDA_STATUS(async_err);
  }();

  if (!status.ok()) ITEX_LOG(ERROR) << status.ToString();
}
*/
}  // namespace

#if ITEX_USE_CCL
StatusOr<std::pair<ccl::datatype, int>> ToNcclDataTypeAndCountMultiplier(
    PrimitiveType element_type) {
  TF_ASSIGN_OR_RETURN(ccl::datatype dtype, ToNcclDataType(element_type));
  bool is_complex = primitive_util::IsComplexType(element_type);
  return std::make_pair(dtype, is_complex ? 2 : 1);
}
#endif  // ITEX_USE_CCL

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices) {
  if (local_devices == nullptr) return participants.size();

  return absl::c_count_if(participants, [&](const GlobalDeviceId& device_id) {
    return absl::c_linear_search(*local_devices, device_id);
  });
}

StatusOr<const NcclUniqueIdCallback*> GetNcclUniqueIdCallback(
    const NcclUniqueIdCallback* unique_id_callback, bool is_local) {
  if (unique_id_callback != nullptr) return unique_id_callback;

  TF_RET_CHECK(is_local || IsGlobalNcclConfig())
      << "If non-local devices are taking part of a collective API on "
         "GPU, the nccl_unique_id_callback must be provided by the client.";

  static NcclUniqueIdCallback local_callback(LocalNcclUniqueIdCallback);
  return &local_callback;
}

StatusOr<NcclComm::Lock> AcquireNcclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    size_t num_local_participants,
    const NcclUniqueIdCallback& unique_id_callback, int rank,
    se::Stream* stream) {
#if ITEX_USE_CCL
  // Ensure that this group of threads have exclusive access to the clique to
  // prevent threads from different groups locking communicators in the clique.
  NcclCliqueKey clique_key(std::move(participants));
  std::shared_ptr<StatusOr<NcclClique::Lock>> clique = AcquireNcclClique(
      run_id, op_id, clique_key, unique_id_callback, num_local_participants);

  if (!clique->ok()) return clique->status();

  auto comm_key = std::make_pair(std::move(clique_key), rank);
  static auto& comms = *new ThreadSafeMap<decltype(comm_key), NcclComm>;
  /*
  // Launch a thread that periodically checks all NCCL communicators for
  // asynchronous errors. If an asynchronous error is observed, the communicator
  // is aborted and an error message logged.
  static auto check_async_error_thread =
      itex::Env::Default()->StartThread(
          itex::ThreadOptions(), "nccl_async_error_thread", [&] {
            while (true) {
              absl::SleepFor(absl::Seconds(30));
              comms.ForEachValue(CheckNcclAsyncError);
            }
          });
  (void)check_async_error_thread;  // Silence unused variable warning.
  */
  NcclComm::Lock comm = comms[comm_key].Acquire();
  if (*comm == nullptr) {
    int nranks = comm_key.first.devices().size();
    auto id = ccl::create_kvs((**clique)->unique_id);
    auto queue = stream->stream_handle;
    auto ccl_stream = ccl::create_stream(*queue);
    auto ccl_device = ccl::create_device(queue->get_device());
    auto ccl_context = ccl::create_context(queue->get_context());
    auto ccl_comm =
        ccl::create_communicator(nranks, rank, ccl_device, ccl_context, id);
    // TODO(ITEX): Is it safe?
    *(comm.get()) = new ccl::communicator(std::move(ccl_comm));
  }
#else
  // Ensure that this group of threads have exclusive access to the clique to
  // prevent threads from different groups locking communicators in the clique.
  NcclCliqueKey clique_key(std::move(participants));
  std::shared_ptr<StatusOr<NcclClique::Lock>> clique = AcquireNcclClique(
      run_id, op_id, clique_key, unique_id_callback, num_local_participants);

  if (!clique->ok()) return clique->status();

  auto comm_key = std::make_pair(std::move(clique_key), rank);
  static auto& comms = *new ThreadSafeMap<decltype(comm_key), NcclComm>;

  NcclComm::Lock comm = comms[comm_key].Acquire();
  if (*comm == nullptr) {
    int nranks = comm_key.first.devices().size();
    // auto queue = stream->stream_handle;
    *(comm.get()) = new ccl::communicator(nranks, rank, (**clique)->unique_id);
  }
#endif  // ITEX_USE_CCL
  return comm;
}

}  // namespace gpu
}  // namespace itex_xla
