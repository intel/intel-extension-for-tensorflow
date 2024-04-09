/* Copyright (c) 2023 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_GPU_COLLECTIVE_OPS_H_
#define ITEX_CORE_KERNELS_GPU_COLLECTIVE_OPS_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/mutex.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/strcat.h"
#include "itex/core/utils/types.h"

namespace itex {

constexpr int MAX_RANK_SIZE = 16;
// After tuned, we found 8 has best performance and XeLink bandwidth.
constexpr size_t VecBytes = 8;

enum class ReductionOp { SUM = 0, MIN = 1, MAX = 2, PROD = 3 };

enum class CollectiveType {
  kAllReduce = 0,
  kBroadcast = 1,
  kReduce = 2,
  kAllGather = 3,
  kReduceScatter = 4,
  kAllToAll = 5
};

// CollectiveManager is used to make the asynchronous communicator calls and to
// manage the per-device streams used for communication.
class CollectiveManager {
 public:
  typedef std::function<void(Status)> DoneCallback;
  CollectiveManager() {}
  ~CollectiveManager();

  static CollectiveManager* instance();

  struct Participant;
  struct Context;
  struct Collective;

  // Adds one participant to an allreduce Collective.
  void AddToAllReduce(std::unique_ptr<Participant> participant,
                      const Context& context, ReductionOp reduction_op);

 private:
  ITEX_GPUStream* GetCommStream(Participant* participant);
  ITEX_GPUStream* GetCommStream(Collective* collective);
  // Adds a participant device to the local `Collective` instance corresponding
  // to `collective_key`.  Launches the `Collective` if it is ready, which it
  // checks by calling `CheckReady()`.
  void AddParticipant(std::unique_ptr<Participant> participant,
                      const Context& context, CollectiveType collective_type,
                      ReductionOp reduction_op);

  // Assumes a `collective_key` corresponds to a `collective`.
  // If `collective` is ready to run, removes it from the `collectives_` map and
  // returns true.  Otherwise returns false.
  // A collective is ready to run when all local participants have called AddTo*
  // function`.
  bool CheckReady(const string& collective_key, Collective* collective)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Run <collective>.  This calls takes ownership of <collective>.
  void RunCollective(Collective* collective);

  Status RunAllReduce(Collective* collective);

  mutex mu_;

  // Maps key to collectives currently being assembled or run.
  absl::flat_hash_map<string, Collective*> collectives_ TF_GUARDED_BY(mu_);

  // Maps a device to the communication streams that make up its collective.
  absl::flat_hash_map<int, ITEX_GPUStream*> device_to_comm_streams_
      TF_GUARDED_BY(mu_);

  Status status_ TF_GUARDED_BY(mu_);

  CollectiveManager(const CollectiveManager&) = delete;
  void operator=(const CollectiveManager&) = delete;
};

// A participant in a Collective.
struct CollectiveManager::Participant {
  Participant(ITEX_GPUStream* tensor_stream, int gpu_device_id,
              const Tensor* input, Tensor* output, DoneCallback done_callback)
      : tensor_stream(tensor_stream),
        comm_stream(nullptr),
        gpu_device_id(gpu_device_id),
        input(input),
        output(output),
        done_callback(std::move(done_callback)) {
    ITEX_DCHECK(tensor_stream != nullptr);
  }

  // `tensor_stream` is the stream that should be waited on to ensure
  // `input`'s data is available on the GPU for the communication stream to
  // access. It is also the stream that will use the produced data;
  // `done_callback` is not called until the next kernel launched on `stream`
  // would see the data. Owned by the caller, who must keep it live until
  // `done_callback` is called.
  ITEX_GPUStream* const tensor_stream;

  ITEX_GPUStream* comm_stream;

  const int gpu_device_id;

  // Owned by the caller, who must keep it live until `done_callback` is
  // called. Is NULL for participants that only receive data.
  const Tensor* input;

  // Owned by the caller, who must keep it live until `done_callback` is
  // called. Is NULL for participants that only send data.
  Tensor* output;

  // The callback which is called at the completion of the operation.
  // When called, `output` has been set to the result of the operation. (note:
  // the stream may not yet have been synced)
  DoneCallback done_callback;
};

// Data that provides context for the collective operation, including the
// operation key, number of participants.
struct CollectiveManager::Context {
  Context(const string& collective_key, int num_local_devices, int root_rank)
      : collective_key(collective_key),
        num_local_devices(num_local_devices),
        root_rank(root_rank) {}

  // Unique key for this collective instance
  const string collective_key;

  // Devices local to this node
  int num_local_devices;

  // Rank of root.
  int root_rank;
};

// A `Collective` encapsulates state for a collective instance at one node.
// Typically, an instance in TensorFlow context would be defined by a collective
// group and the (step, frame iteration) for that execution.
// For each collective instance there will be one `Collective` object per node.
// For example, a collective that runs on a single node with 4 GPUs would
// have a single `Collective` per step.
struct CollectiveManager::Collective {
  Collective(const string& collective_key_in, DataType data_type_in,
             CollectiveType type_in, ReductionOp reduction_op_in,
             int num_local_devices_in)
      : collective_key(collective_key_in),
        data_type(data_type_in),
        type(type_in),
        reduction_op(reduction_op_in),
        num_local_devices(num_local_devices_in) {
    participants.reserve(num_local_devices_in);
  }

  const string collective_key;
  const DataType data_type;
  const CollectiveType type;
  const ReductionOp reduction_op;
  const int num_local_devices;

  // All collective participants.
  // Adding values in this vector is guarded by the mutex of the containing
  // CollectiveManager.
  std::vector<std::unique_ptr<Participant>> participants;

  // For collective types that have a root (e.g. the root of broadcast is the
  // sender), this is the rank of the root.
  int root_rank = -1;

  // How many participants have been registered so far. The Collective is
  // eligible for running with <available_participants> == num_local_devices.
  // Guarded by the mutex of the containing Communicator.
  int available_participants = 0;

  Status status;
};

CollectiveManager::~CollectiveManager() {
  for (auto& it : device_to_comm_streams_) {
    delete it.second;
  }
}

CollectiveManager* CollectiveManager::instance() {
  static CollectiveManager* instance = new CollectiveManager();
  return instance;
}

bool CollectiveManager::CheckReady(const string& collective_key,
                                   Collective* collective) {
  if (collective->available_participants == collective->num_local_devices) {
    // Ownership transferred to callee.
    collectives_.erase(collective_key);
    return true;
  }
  return false;
}

ITEX_GPUStream* CollectiveManager::GetCommStream(Participant* participant) {
  if (participant->comm_stream) return participant->comm_stream;
  mutex_lock l(&mu_);
  ITEX_GPUStream* comm_stream = nullptr;
  auto it = device_to_comm_streams_.find(participant->gpu_device_id);
  if (it == device_to_comm_streams_.end()) {
    auto tensor_stream = participant->tensor_stream;
    auto property_list = sycl::property_list{sycl::property::queue::in_order()};
    comm_stream =
        new ITEX_GPUStream(tensor_stream->get_context(),
                           tensor_stream->get_device(), property_list);
    device_to_comm_streams_.emplace(participant->gpu_device_id, comm_stream);
  } else {
    comm_stream = it->second;
  }
  participant->comm_stream = comm_stream;
  return comm_stream;
}

void CollectiveManager::AddToAllReduce(std::unique_ptr<Participant> participant,
                                       const Context& context,
                                       ReductionOp reduction_op) {
  AddParticipant(std::move(participant), context, CollectiveType::kAllReduce,
                 reduction_op);
}

void CollectiveManager::AddParticipant(std::unique_ptr<Participant> participant,
                                       const Context& context,
                                       CollectiveType collective_type,
                                       ReductionOp reduction_op) {
  Collective* to_run = nullptr;
  DataType data_type;
  Status manager_status;
  if (participant->input != nullptr) {
    data_type = participant->input->dtype();
  } else {
    data_type = participant->output->dtype();
  }
  {
    mutex_lock l(&mu_);
    manager_status = status_;
    if (manager_status.ok()) {
      auto collective_it = collectives_.find(context.collective_key);
      Collective* collective = nullptr;
      if (collective_it == collectives_.end()) {
        collective =
            new Collective(context.collective_key, data_type, collective_type,
                           reduction_op, context.num_local_devices);
        collectives_.emplace(context.collective_key, collective);
      } else {
        collective = collective_it->second;
      }

      if (collective->status.ok() && collective->data_type != data_type) {
        collective->status = errors::InvalidArgument(
            "Collective previously initialized with datatype ",
            DataTypeString(collective->data_type), " but now got datatype ",
            DataTypeString(data_type));
      }
      if (collective->status.ok() && collective->reduction_op != reduction_op) {
        collective->status = errors::InvalidArgument(
            "Collective previously initialized with reduction_op ",
            static_cast<int>(collective->reduction_op),
            " but now got reduction_op ", static_cast<int>(reduction_op));
      }
      if (collective->status.ok() && collective->type != collective_type) {
        collective->status = errors::InvalidArgument(
            "Collective previously initialized with type ",
            static_cast<int>(collective->type), " but now got type ",
            static_cast<int>(collective_type));
      }
      if (collective->status.ok() &&
          collective->num_local_devices != context.num_local_devices) {
        collective->status = errors::InvalidArgument(
            "Collective previously initialized with num_local_devices ",
            collective->num_local_devices, " but now got ",
            context.num_local_devices);
      }
      if (collective->status.ok() &&
          collective->participants.size() >= collective->num_local_devices) {
        collective->status = errors::InvalidArgument(
            "Collective expected ", collective->num_local_devices,
            " participants but now has ", collective->participants.size(),
            " with one more participant being added");
      }
      if (collective->status.ok() && collective->root_rank >= 0 &&
          context.root_rank >= 0 &&
          collective->root_rank != context.root_rank) {
        collective->status = errors::InvalidArgument(
            "Collective ", collective->collective_key,
            " already has root_rank ", collective->root_rank,
            " but new participant has root_rank ", context.root_rank);
      }

      if (context.root_rank >= 0) {
        collective->root_rank = context.root_rank;
      }

      collective->participants.emplace_back(std::move(participant));
      ++collective->available_participants;

      if (CheckReady(context.collective_key, collective)) {
        to_run = collective;
      }
    }
  }
  if (!manager_status.ok()) {
    participant->done_callback(manager_status);
    return;
  }
  if (to_run != nullptr) RunCollective(to_run);
}

template <typename T, typename Func, bool PartialStore>
struct AllReduceKernel;
template <typename T, typename Func, typename AccT = T>
void LaunchAllReduceKernel(ITEX_GPUStream* stream, size_t element_count,
                           const std::vector<const void*>& inputs,
                           const std::vector<void*>& outputs, int rank,
                           int reduction_size) {
  constexpr size_t VecSize = VecBytes / sizeof(T);
  size_t vec_count = element_count / VecSize;
  size_t vec_tail_element_count = element_count % VecSize;
  size_t total_vec_count = vec_count + (vec_tail_element_count > 0 ? 1 : 0);

  // Each rank allreduces a sub slice of the tensors. Last rank
  // also allreduce the tail vectors of the tensor.
  size_t slice_vec_count = total_vec_count / reduction_size;
  size_t tail_vec_count = total_vec_count % reduction_size;
  size_t local_vec_count =
      slice_vec_count + ((rank == (reduction_size - 1)) ? tail_vec_count : 0);

  if (local_vec_count == 0) return;

  auto device = stream->get_device();
  size_t group_size =
      device.template get_info<sycl::info::device::max_work_group_size>();

  // set max_workitems = HW_workgroup_num * max_workgroup_size
  int num_max_concurrent_workitem =
      device.template get_info<sycl::ext::intel::info::device::gpu_slices>() *
      device.template get_info<
          sycl::ext::intel::info::device::gpu_subslices_per_slice>() *
      group_size;
  int num_workitem = local_vec_count <= num_max_concurrent_workitem
                         ? local_vec_count
                         : num_max_concurrent_workitem;
  size_t num_vec_per_workitem = local_vec_count / num_workitem;
  size_t num_tail_vec = local_vec_count % num_workitem;

  int num_workgroup = (num_workitem + group_size - 1) / group_size;

  if (reduction_size <= MAX_RANK_SIZE) {
    stream->submit([&](sycl::handler& cgh) {
      T* in_ptr[MAX_RANK_SIZE];
      T* out_ptr[MAX_RANK_SIZE];

      for (int i = 0; i < reduction_size; ++i) {
        in_ptr[i] = static_cast<T*>(const_cast<void*>(inputs[i])) +
                    rank * slice_vec_count * VecSize;
        out_ptr[i] =
            static_cast<T*>(outputs[i]) + rank * slice_vec_count * VecSize;
      }

      // Last rank may need to process the tail elements which can't form a
      // full vector and need partial block store.
      if (rank != (reduction_size - 1) || vec_tail_element_count == 0) {
        cgh.parallel_for<AllReduceKernel<T, Func, false>>(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
              const int index = item.get_global_linear_id();
              if (index >= num_workitem) return;

              for (size_t n = 0; n < num_vec_per_workitem; ++n) {
                size_t offset = (num_workitem * n + index) * VecSize;
                AlignedVector<AccT, VecSize, Func> result;
                result.Load(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                    &(in_ptr[0][offset])));
                for (int i = 1; i < reduction_size; ++i)
                  result.Accumulate(
                      *reinterpret_cast<AlignedVector<T, VecSize>*>(
                          &(in_ptr[i][offset])));
                for (int i = 0; i < reduction_size; ++i)
                  result.Store(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                      &(out_ptr[i][offset])));
              }

              if (index < num_tail_vec) {
                size_t offset =
                    (num_workitem * num_vec_per_workitem + index) * VecSize;
                AlignedVector<AccT, VecSize, Func> result;
                result.Load(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                    &(in_ptr[0][offset])));
                for (int i = 1; i < reduction_size; ++i)
                  result.Accumulate(
                      *reinterpret_cast<AlignedVector<T, VecSize>*>(
                          &(in_ptr[i][offset])));
                for (int i = 0; i < reduction_size; ++i)
                  result.Store(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                      &(out_ptr[i][offset])));
              }
            });
      } else {
        cgh.parallel_for<AllReduceKernel<T, Func, true>>(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
              const int index = item.get_global_linear_id();
              if (index >= num_workitem) return;

              for (size_t n = 0; n < num_vec_per_workitem; ++n) {
                size_t offset = (num_workitem * n + index) * VecSize;
                AlignedVector<AccT, VecSize, Func> result;
                result.Load(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                    &(in_ptr[0][offset])));
                for (int i = 1; i < reduction_size; ++i)
                  result.Accumulate(
                      *reinterpret_cast<AlignedVector<T, VecSize>*>(
                          &(in_ptr[i][offset])));

                if (local_vec_count > num_workitem ||
                    index != (num_workitem - 1)) {
                  for (int i = 0; i < reduction_size; ++i)
                    result.Store(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                        &(out_ptr[i][offset])));
                } else {  // the last workitem may process a partial vector
                  for (int i = 0; i < reduction_size; ++i)
                    result.PartialStore(
                        *reinterpret_cast<AlignedVector<T, VecSize>*>(
                            &(out_ptr[i][offset])),
                        vec_tail_element_count);
                }
              }

              if (index < num_tail_vec) {
                size_t offset =
                    (num_workitem * num_vec_per_workitem + index) * VecSize;
                AlignedVector<AccT, VecSize, Func> result;
                result.Load(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                    &(in_ptr[0][offset])));
                for (int i = 1; i < reduction_size; ++i)
                  result.Accumulate(
                      *reinterpret_cast<AlignedVector<T, VecSize>*>(
                          &(in_ptr[i][offset])));

                if (index != num_tail_vec - 1) {
                  for (int i = 0; i < reduction_size; ++i)
                    result.Store(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                        &(out_ptr[i][offset])));
                } else {  // the last workitem may process a partial vector
                  for (int i = 0; i < reduction_size; ++i)
                    result.PartialStore(
                        *reinterpret_cast<AlignedVector<T, VecSize>*>(
                            &(out_ptr[i][offset])),
                        vec_tail_element_count);
                }
              }
            });
      }
    });
  } else {
    ITEX_LOG(FATAL) << "Reduction size " << reduction_size
                    << " is not supported in AllReduce.";
  }
}

Status CollectiveManager::RunAllReduce(Collective* collective) {
  DataType data_type = collective->data_type;
  ReductionOp reduction_op = collective->reduction_op;
  auto num_elements = collective->participants[0]->input->NumElements();
  std::vector<const void*> inputs;
  std::vector<void*> outputs;
  for (auto& participant : collective->participants) {
    if (num_elements != participant->input->NumElements()) {
      return errors::InvalidArgument(
          "Collective Allreduce require the tensors have the same number of "
          "elements, "
          "one rank has NumElements: ",
          num_elements,
          " while another rank"
          "has NumElements: ",
          participant->input->NumElements());
    }
    inputs.push_back(participant->input->data());
    outputs.push_back(participant->output->data());
  }

  std::vector<ITEX_GPUStream*> comm_streams;
  std::vector<sycl::event> begin_events;
  std::vector<sycl::event> end_events;
  int reduction_size = collective->participants.size();
  for (int i = 0; i < reduction_size; ++i) {
    auto comm_stream = GetCommStream(collective->participants[i].get());
    comm_streams.push_back(comm_stream);

    // TODO(intel): use barrier instead of wait once barrier bug is fixed.
    comm_stream->wait();
    // auto begin_event = comm_stream->ext_oneapi_submit_barrier();
    // begin_events.push_back(begin_event);
  }

  for (int i = 0; i < reduction_size; ++i) {
    auto comm_stream = comm_streams[i];
    // comm_stream->ext_oneapi_submit_barrier(begin_events);

    if (reduction_op == ReductionOp::SUM) {
      switch (data_type) {
        case DT_BFLOAT16:
          LaunchAllReduceKernel<Eigen::bfloat16, sycl::plus<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_HALF:
          LaunchAllReduceKernel<Eigen::half, sycl::plus<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_FLOAT:
          LaunchAllReduceKernel<float, sycl::plus<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_INT32:
          LaunchAllReduceKernel<int, sycl::plus<int>, int>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        default:
          return errors::InvalidArgument(
              "Collective Allreduce unsupports datatype ",
              DataTypeString(data_type));
      }
    } else if (reduction_op == ReductionOp::MIN) {
      switch (data_type) {
        case DT_BFLOAT16:
          LaunchAllReduceKernel<Eigen::bfloat16, sycl::minimum<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_HALF:
          LaunchAllReduceKernel<Eigen::half, sycl::minimum<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_FLOAT:
          LaunchAllReduceKernel<float, sycl::minimum<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_INT32:
          LaunchAllReduceKernel<int, sycl::minimum<int>, int>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        default:
          return errors::InvalidArgument(
              "Collective Allreduce unsupports datatype ",
              DataTypeString(data_type));
      }
    } else if (reduction_op == ReductionOp::MAX) {
      switch (data_type) {
        case DT_BFLOAT16:
          LaunchAllReduceKernel<Eigen::bfloat16, sycl::maximum<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_HALF:
          LaunchAllReduceKernel<Eigen::half, sycl::maximum<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_FLOAT:
          LaunchAllReduceKernel<float, sycl::maximum<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_INT32:
          LaunchAllReduceKernel<int, sycl::maximum<int>, int>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        default:
          return errors::InvalidArgument(
              "Collective Allreduce unsupports datatype ",
              DataTypeString(data_type));
      }
    } else if (reduction_op == ReductionOp::PROD) {
      switch (data_type) {
        case DT_BFLOAT16:
          LaunchAllReduceKernel<Eigen::bfloat16, sycl::multiplies<float>,
                                float>(comm_stream, num_elements, inputs,
                                       outputs, i, reduction_size);
          break;
        case DT_HALF:
          LaunchAllReduceKernel<Eigen::half, sycl::multiplies<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_FLOAT:
          LaunchAllReduceKernel<float, sycl::multiplies<float>, float>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        case DT_INT32:
          LaunchAllReduceKernel<int, sycl::multiplies<int>, int>(
              comm_stream, num_elements, inputs, outputs, i, reduction_size);
          break;
        default:
          return errors::InvalidArgument(
              "Collective Allreduce unsupports datatype ",
              DataTypeString(data_type));
      }
    } else {
      return errors::InvalidArgument(
          "Collective Allreduce unsupports ReductionOp yet!");
    }
    // auto event = comm_stream->ext_oneapi_submit_barrier();
    // end_events.push_back(event);
  }

  for (int i = 0; i < reduction_size; ++i) {
    // TODO(intel): use barrier instead of wait once barrier bug is fixed.
    comm_streams[i]->wait();
    // comm_streams[i]->ext_oneapi_submit_barrier(end_events);
  }
  return Status::OK();
}

void CollectiveManager::RunCollective(Collective* collective) {
  Status status = collective->status;
  for (int i = 0; status.ok() && i < collective->num_local_devices; ++i) {
    Participant* participant = collective->participants[i].get();
    ITEX_GPUStream* comm_stream = GetCommStream(participant);
    ITEX_DCHECK(comm_stream != nullptr);

    if (participant->input != nullptr) {
      // Wait to ensure that the kernel that produces the data in the input
      // tensor has finished running before the collective kernel runs on the
      // communication stream.
      comm_stream->ext_oneapi_submit_barrier(
          {participant->tensor_stream->ext_oneapi_submit_barrier()});
    }
  }

  if (status.ok() && collective->type == CollectiveType::kBroadcast &&
      collective->root_rank < 0) {
    status = errors::InvalidArgument("Root rank not indicated for collective ",
                                     collective->collective_key);
  }

  if (!status.ok()) {
    for (int i = 0; i < collective->num_local_devices; ++i) {
      collective->participants[i]->done_callback(status);
    }
    delete collective;
    return;
  }

  switch (collective->type) {
    case CollectiveType::kAllReduce: {
      status = RunAllReduce(collective);
      break;
    }
    default:
      status =
          errors::InvalidArgument("ITEX does not support collective ",
                                  static_cast<int>(collective->type), " yet!");
  }

  for (int i = 0; status.ok() && i < collective->num_local_devices; ++i) {
    Participant* participant = collective->participants[i].get();
    ITEX_GPUStream* comm_stream = GetCommStream(participant);
    participant->tensor_stream->ext_oneapi_submit_barrier(
        {comm_stream->ext_oneapi_submit_barrier()});
  }

  for (int i = 0; i < collective->num_local_devices; ++i) {
    collective->participants[i]->done_callback(status);
  }
  delete collective;
}

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_COLLECTIVE_OPS_H_
