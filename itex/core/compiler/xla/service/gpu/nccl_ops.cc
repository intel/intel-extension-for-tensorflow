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

#include "itex/core/compiler/xla/service/gpu/nccl_ops.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if !ITEX_USE_CCL
namespace itex_xla {
namespace gpu {

namespace {
struct Participant {
  Participant(ITEX_GPUStream* stream, const void* send, void* recv, int rank)
      : stream(stream), send(send), recv(recv), rank(rank) {}
  ITEX_GPUStream* stream;
  const void* send;
  void* recv;
  int rank;
};

struct AlltoAllParticipant {
  ITEX_GPUStream* stream;
  std::vector<const void*> send;
  std::vector<void*> recv;
  int rank;
};

struct PermuteParticipant {
  ITEX_GPUStream* stream;
  const void* send;
  void* recv;
  absl::optional<int64_t> send_id;
  absl::optional<int64_t> recv_id;
  int rank;
};

struct Manager {
  static Manager& instance() {
    static Manager m;
    return m;
  }

  itex::mutex mu;
  itex::condition_variable cv;
  // The order should be: (r0p0, r1p0), (r0p1, r1p1)
  std::unordered_map<std::string, std::vector<Participant>> collectives
      TF_GUARDED_BY(mu);
  std::unordered_map<std::string, std::vector<AlltoAllParticipant>>
      alltoall_collectives TF_GUARDED_BY(mu);
  std::unordered_map<std::string, std::vector<PermuteParticipant>>
      permute_collectives TF_GUARDED_BY(mu);
};

template <typename T, typename Func, int size>
struct AllReduceKernel;

template <typename T, typename Func, typename AccT = T>
void allreduce_dpcpp(ITEX_GPUStream* stream, int tensor_size,
                     std::vector<Participant>& participants,
                     int reduction_size) {
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<cl::sycl::info::device::max_work_group_size>();
  auto num_workgroup = (tensor_size + group_size - 1) / group_size;

  if (reduction_size == 2) {
    stream->submit([&](cl::sycl::handler& cgh) {
      auto in0_ptr = static_cast<const T*>(participants[0].send);
      auto in1_ptr = static_cast<const T*>(participants[1].send);
      auto out0_ptr = static_cast<T*>(participants[0].recv);
      auto out1_ptr = static_cast<T*>(participants[1].recv);

      cgh.parallel_for<AllReduceKernel<T, Func, 2>>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(group_size * num_workgroup),
                                cl::sycl::range<1>(group_size)),
          [=](cl::sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            out0_ptr[index] =
                T(Func()(AccT(in0_ptr[index]), AccT(in1_ptr[index])));
            out1_ptr[index] = out0_ptr[index];
          });
    });
  } else if (reduction_size == 3) {
    stream->submit([&](cl::sycl::handler& cgh) {
      auto in0_ptr = static_cast<const T*>(participants[0].send);
      auto in1_ptr = static_cast<const T*>(participants[1].send);
      auto in2_ptr = static_cast<const T*>(participants[2].send);
      auto out0_ptr = static_cast<T*>(participants[0].recv);
      auto out1_ptr = static_cast<T*>(participants[1].recv);
      auto out2_ptr = static_cast<T*>(participants[2].recv);

      cgh.parallel_for<AllReduceKernel<T, Func, 3>>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(group_size * num_workgroup),
                                cl::sycl::range<1>(group_size)),
          [=](cl::sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            out0_ptr[index] =
                T(Func()(Func()(AccT(in0_ptr[index]), AccT(in1_ptr[index])),
                         AccT(in2_ptr[index])));
            out1_ptr[index] = out0_ptr[index];
            out2_ptr[index] = out0_ptr[index];
          });
    });
  } else if (reduction_size == 4) {
    stream->submit([&](cl::sycl::handler& cgh) {
      auto in0_ptr = static_cast<const T*>(participants[0].send);
      auto in1_ptr = static_cast<const T*>(participants[1].send);
      auto in2_ptr = static_cast<const T*>(participants[2].send);
      auto in3_ptr = static_cast<const T*>(participants[3].send);
      auto out0_ptr = static_cast<T*>(participants[0].recv);
      auto out1_ptr = static_cast<T*>(participants[1].recv);
      auto out2_ptr = static_cast<T*>(participants[2].recv);
      auto out3_ptr = static_cast<T*>(participants[3].recv);

      cgh.parallel_for<AllReduceKernel<T, Func, 4>>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(group_size * num_workgroup),
                                cl::sycl::range<1>(group_size)),
          [=](cl::sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            out0_ptr[index] = T(Func()(
                Func()(Func()(AccT(in0_ptr[index]), AccT(in1_ptr[index])),
                       AccT(in2_ptr[index])),
                AccT(in3_ptr[index])));
            out1_ptr[index] = out0_ptr[index];
            out2_ptr[index] = out0_ptr[index];
            out3_ptr[index] = out0_ptr[index];
          });
    });
  } else {
    ITEX_LOG(FATAL) << "Reduction size " << reduction_size
                    << " is not supported in AllReduce.";
  }
}

template <typename T, int size>
struct AllGatherKernel;

template <typename T>
void allgather_dpcpp(ITEX_GPUStream* stream, int tensor_size,
                     std::vector<Participant>& participants,
                     int reduction_size) {
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<cl::sycl::info::device::max_work_group_size>();
  auto num_workgroup = (tensor_size + group_size - 1) / group_size;

  if (reduction_size == 2) {
    stream->submit([&](cl::sycl::handler& cgh) {
      auto in0_ptr = static_cast<const T*>(participants[0].send);
      auto in1_ptr = static_cast<const T*>(participants[1].send);
      auto out0_ptr = static_cast<T*>(participants[0].recv);
      auto out1_ptr = static_cast<T*>(participants[1].recv);

      cgh.parallel_for<AllGatherKernel<T, 2>>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(group_size * num_workgroup),
                                cl::sycl::range<1>(group_size)),
          [=](cl::sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            out0_ptr[index] = in0_ptr[index];
            out0_ptr[index + tensor_size] = in1_ptr[index];
            out1_ptr[index] = in0_ptr[index];
            out1_ptr[index + tensor_size] = in1_ptr[index];
          });
    });
  } else {
    ITEX_LOG(FATAL) << "Reduction size " << reduction_size
                    << " is not supported in AllGather.";
  }
}

template <typename T, int size>
struct AllToAllKernel;

template <typename T>
void alltoall_dpcpp(ITEX_GPUStream* stream, int tensor_size,
                    std::vector<AlltoAllParticipant>& participants,
                    int reduction_size) {
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<cl::sycl::info::device::max_work_group_size>();
  auto num_workgroup = (tensor_size + group_size - 1) / group_size;

  // Process: send vec -> rev vec
  // P0: (s0, s1) -> (s0, s2)
  // p1: (s2, s3) -> (s1, s3)
  if (reduction_size == 2) {
    stream->submit([&](cl::sycl::handler& cgh) {
      auto s0_ptr = static_cast<const T*>(participants[0].send[0]);
      auto s1_ptr = static_cast<const T*>(participants[0].send[1]);
      auto s2_ptr = static_cast<const T*>(participants[1].send[0]);
      auto s3_ptr = static_cast<const T*>(participants[1].send[1]);
      auto r0_ptr = static_cast<T*>(participants[0].recv[0]);
      auto r1_ptr = static_cast<T*>(participants[0].recv[1]);
      auto r2_ptr = static_cast<T*>(participants[1].recv[0]);
      auto r3_ptr = static_cast<T*>(participants[1].recv[1]);

      cgh.parallel_for<AllToAllKernel<T, 2>>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(group_size * num_workgroup),
                                cl::sycl::range<1>(group_size)),
          [=](cl::sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            r0_ptr[index] = s0_ptr[index];
            r1_ptr[index] = s2_ptr[index];
            r2_ptr[index] = s1_ptr[index];
            r3_ptr[index] = s3_ptr[index];
          });
    });
  } else {
    ITEX_LOG(FATAL) << "Reduction size " << reduction_size
                    << " is not supported in AllGather.";
  }
}

template <typename T, typename Func, int size>
struct ReduceScatterKernel;

template <typename T, typename Func>
void reducescatter_dpcpp(ITEX_GPUStream* stream, int tensor_size,
                         std::vector<Participant>& participants,
                         int reduction_size) {
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<cl::sycl::info::device::max_work_group_size>();
  // tensor_size: output tensor size
  auto num_workgroup = (tensor_size + group_size - 1) / group_size;

  if (reduction_size == 2) {
    stream->submit([&](cl::sycl::handler& cgh) {
      auto in0_ptr = static_cast<const T*>(participants[0].send);
      auto in1_ptr = static_cast<const T*>(participants[1].send);
      auto out0_ptr = static_cast<T*>(participants[0].recv);
      auto out1_ptr = static_cast<T*>(participants[1].recv);

      cgh.parallel_for<ReduceScatterKernel<T, Func, 2>>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(group_size * num_workgroup),
                                cl::sycl::range<1>(group_size)),
          [=](cl::sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            out0_ptr[index] = Func()(in0_ptr[index], in1_ptr[index]);
            out1_ptr[index] = Func()(in0_ptr[index + tensor_size],
                                     in1_ptr[index + tensor_size]);
          });
    });
  } else if (reduction_size == 3) {
    stream->submit([&](cl::sycl::handler& cgh) {
      auto in0_ptr = static_cast<const T*>(participants[0].send);
      auto in1_ptr = static_cast<const T*>(participants[1].send);
      auto in2_ptr = static_cast<const T*>(participants[2].send);
      auto out0_ptr = static_cast<T*>(participants[0].recv);
      auto out1_ptr = static_cast<T*>(participants[1].recv);
      auto out2_ptr = static_cast<T*>(participants[2].recv);

      cgh.parallel_for<ReduceScatterKernel<T, Func, 3>>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(group_size * num_workgroup),
                                cl::sycl::range<1>(group_size)),
          [=](cl::sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            out0_ptr[index] =
                Func()(Func()(in0_ptr[index], in1_ptr[index]), in2_ptr[index]);
            out1_ptr[index] = Func()(Func()(in0_ptr[index + tensor_size],
                                            in1_ptr[index + tensor_size]),
                                     in2_ptr[index + tensor_size]);
            out2_ptr[index] = Func()(Func()(in0_ptr[index + 2 * tensor_size],
                                            in1_ptr[index + 2 * tensor_size]),
                                     in2_ptr[index + 2 * tensor_size]);
          });
    });
  } else if (reduction_size == 4) {
    stream->submit([&](cl::sycl::handler& cgh) {
      auto in0_ptr = static_cast<const T*>(participants[0].send);
      auto in1_ptr = static_cast<const T*>(participants[1].send);
      auto in2_ptr = static_cast<const T*>(participants[2].send);
      auto in3_ptr = static_cast<const T*>(participants[3].send);
      auto out0_ptr = static_cast<T*>(participants[0].recv);
      auto out1_ptr = static_cast<T*>(participants[1].recv);
      auto out2_ptr = static_cast<T*>(participants[2].recv);
      auto out3_ptr = static_cast<T*>(participants[3].recv);

      cgh.parallel_for<ReduceScatterKernel<T, Func, 4>>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(group_size * num_workgroup),
                                cl::sycl::range<1>(group_size)),
          [=](cl::sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            out0_ptr[index] = Func()(
                Func()(Func()(in0_ptr[index], in1_ptr[index]), in2_ptr[index]),
                in3_ptr[index]);
            out1_ptr[index] =
                Func()(Func()(Func()(in0_ptr[index + tensor_size],
                                     in1_ptr[index + tensor_size]),
                              in2_ptr[index + tensor_size]),
                       in3_ptr[index]);
            out2_ptr[index] =
                Func()(Func()(Func()(in0_ptr[index + 2 * tensor_size],
                                     in1_ptr[index + 2 * tensor_size]),
                              in2_ptr[index + 2 * tensor_size]),
                       in3_ptr[index]);
            out3_ptr[index] =
                Func()(Func()(Func()(in0_ptr[index + 3 * tensor_size],
                                     in1_ptr[index + 3 * tensor_size]),
                              in2_ptr[index + 3 * tensor_size]),
                       in3_ptr[index]);
          });
    });
  } else {
    ITEX_LOG(FATAL) << "Reduction size " << reduction_size
                    << " is not supported in AllReduce.";
  }
}

template <typename T, int size>
struct CollectivePermuteKernel;

template <typename T>
void permute_dpcpp(ITEX_GPUStream* stream, int tensor_size,
                   std::vector<PermuteParticipant>& participants,
                   int reduction_size) {
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<cl::sycl::info::device::max_work_group_size>();
  auto num_workgroup = (tensor_size + group_size - 1) / group_size;

  if (reduction_size == 2) {
    for (int i = 0; i < 2; i++)
      if (participants[i].send_id)
        stream->memcpy(participants[i].recv,
                       (const void*)participants[*participants[i].send_id].send,
                       tensor_size * sizeof(T));

  } else {
    ITEX_LOG(FATAL) << "Reduction size " << reduction_size
                    << " is not supported in AllReduce.";
  }
}

template <class T>
void stream_wait_streamlist(ITEX_GPUStream* stream, const std::vector<T>& p) {
  std::vector<ITEX_GPUEvent> event_list;
  for (int i = 1; i < p.size(); i++) {
    ITEX_GPUEvent event = p[i].stream->ext_oneapi_submit_barrier();
    event_list.push_back(event);
  }
  stream->ext_oneapi_submit_barrier(event_list);
}

template <class T>
void streamlist_wait_stream(ITEX_GPUStream* stream, const std::vector<T>& p) {
  ITEX_GPUEvent event = stream->ext_oneapi_submit_barrier();

  const std::vector<ITEX_GPUEvent> event_list{event};
  for (int i = 1; i < p.size(); i++) {
    p[i].stream->ext_oneapi_submit_barrier(event_list);
  }
}
}  // namespace

void itex_allreduce(const void* send_buffer, void* recv_buffer,
                    int element_count, PrimitiveType dtype,
                    ReductionKind reduction_kind, ITEX_GPUStream* gpu_stream,
                    ncclComm_t comm) {
  std::vector<Participant> p;
  {
    itex::mutex_lock l(&(Manager::instance().mu));
    if (Manager::instance().collectives.find(comm->id) ==
        Manager::instance().collectives.end()) {
      std::vector<Participant> participants;
      participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      Manager::instance().collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().collectives[comm->id].push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      p = Manager::instance().collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(&l);
    } else {
      Manager::instance().collectives.erase(comm->id);
      std::sort(p.begin(), p.end(),
                [](const Participant& a, const Participant& b) -> bool {
                  return a.rank < b.rank;
                });

      ITEX_GPUStream* stream = p[0].stream;
      stream_wait_streamlist(stream, p);

      if (reduction_kind == ReductionKind::SUM) {
        if (dtype == PRED)
          allreduce_dpcpp<bool, sycl::plus<bool>>(stream, element_count, p,
                                                  comm->nranks);
        else if (dtype == F32)
          allreduce_dpcpp<float, sycl::plus<float>>(stream, element_count, p,
                                                    comm->nranks);
        else if (dtype == F64)
          allreduce_dpcpp<double, sycl::plus<double>>(stream, element_count, p,
                                                      comm->nranks);
        else if (dtype == S32)
          allreduce_dpcpp<int32_t, sycl::plus<int32_t>>(stream, element_count,
                                                        p, comm->nranks);
        else if (dtype == S64)
          allreduce_dpcpp<int64_t, sycl::plus<int64_t>>(stream, element_count,
                                                        p, comm->nranks);
        else if (dtype == C64)
          allreduce_dpcpp<std::complex<float>, sycl::plus<std::complex<float>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C128)
          allreduce_dpcpp<std::complex<double>,
                          sycl::plus<std::complex<double>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          allreduce_dpcpp<Eigen::bfloat16, sycl::plus<float>, float>(
              stream, element_count, p, comm->nranks);

        else
          ITEX_LOG(FATAL) << "PrimitiveType "
                          << primitive_util::LowercasePrimitiveTypeName(dtype)
                          << " is not supported in AllReduce.";
      } else if (reduction_kind == ReductionKind::PRODUCT) {
        if (dtype == PRED)
          allreduce_dpcpp<bool, sycl::multiplies<bool>>(stream, element_count,
                                                        p, comm->nranks);
        else if (dtype == F32)
          allreduce_dpcpp<float, sycl::multiplies<float>>(stream, element_count,
                                                          p, comm->nranks);
        else if (dtype == F64)
          allreduce_dpcpp<double, sycl::multiplies<double>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S32)
          allreduce_dpcpp<int32_t, sycl::multiplies<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          allreduce_dpcpp<int64_t, sycl::multiplies<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C64)
          allreduce_dpcpp<std::complex<float>,
                          sycl::multiplies<std::complex<float>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C128)
          allreduce_dpcpp<std::complex<double>,
                          sycl::multiplies<std::complex<double>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          allreduce_dpcpp<Eigen::bfloat16, sycl::multiplies<float>, float>(
              stream, element_count, p, comm->nranks);

        else
          ITEX_LOG(FATAL) << "PrimitiveType "
                          << primitive_util::LowercasePrimitiveTypeName(dtype)
                          << " is not supported in AllReduce.";
      } else if (reduction_kind == ReductionKind::MIN) {
        if (dtype == PRED)
          allreduce_dpcpp<bool, sycl::minimum<bool>>(stream, element_count, p,
                                                     comm->nranks);
        else if (dtype == F32)
          allreduce_dpcpp<float, sycl::minimum<float>>(stream, element_count, p,
                                                       comm->nranks);
        else if (dtype == F64)
          allreduce_dpcpp<double, sycl::minimum<double>>(stream, element_count,
                                                         p, comm->nranks);
        else if (dtype == S32)
          allreduce_dpcpp<int32_t, sycl::minimum<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          allreduce_dpcpp<int64_t, sycl::minimum<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          allreduce_dpcpp<Eigen::bfloat16, sycl::minimum<float>, float>(
              stream, element_count, p, comm->nranks);

        else
          ITEX_LOG(FATAL) << "PrimitiveType "
                          << primitive_util::LowercasePrimitiveTypeName(dtype)
                          << " is not supported in AllReduce.";
      } else if (reduction_kind == ReductionKind::MAX) {
        if (dtype == PRED)
          allreduce_dpcpp<bool, sycl::maximum<bool>>(stream, element_count, p,
                                                     comm->nranks);
        else if (dtype == F32)
          allreduce_dpcpp<float, sycl::maximum<float>>(stream, element_count, p,
                                                       comm->nranks);
        else if (dtype == F64)
          allreduce_dpcpp<double, sycl::maximum<double>>(stream, element_count,
                                                         p, comm->nranks);
        else if (dtype == S32)
          allreduce_dpcpp<int32_t, sycl::maximum<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          allreduce_dpcpp<int64_t, sycl::maximum<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          allreduce_dpcpp<Eigen::bfloat16, sycl::maximum<float>, float>(
              stream, element_count, p, comm->nranks);

        else
          ITEX_LOG(FATAL) << "PrimitiveType "
                          << primitive_util::LowercasePrimitiveTypeName(dtype)
                          << " is not supported in AllReduce.";
      } else {
        ITEX_LOG(FATAL) << "ReductionKind " << static_cast<int>(reduction_kind)
                        << " is not supported in AllReduce.";
      }

      streamlist_wait_stream(stream, p);
      Manager::instance().cv.notify_all();
    }
  }
}

void itex_allgather(const void* send_buffer, void* recv_buffer,
                    int element_count, PrimitiveType dtype,
                    ITEX_GPUStream* gpu_stream, ncclComm_t comm) {
  std::vector<Participant> p;
  {
    itex::mutex_lock l(&(Manager::instance().mu));
    if (Manager::instance().collectives.find(comm->id) ==
        Manager::instance().collectives.end()) {
      std::vector<Participant> participants;
      participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      Manager::instance().collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().collectives[comm->id].push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      p = Manager::instance().collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(&l);
    } else {
      Manager::instance().collectives.erase(comm->id);
      std::sort(p.begin(), p.end(),
                [](const Participant& a, const Participant& b) -> bool {
                  return a.rank < b.rank;
                });

      ITEX_GPUStream* stream = p[0].stream;
      stream_wait_streamlist(stream, p);
      if (dtype == PRED)
        allgather_dpcpp<bool>(stream, element_count, p, comm->nranks);
      else if (dtype == F32)
        allgather_dpcpp<float>(stream, element_count, p, comm->nranks);
      else if (dtype == F64)
        allgather_dpcpp<double>(stream, element_count, p, comm->nranks);
      else if (dtype == S32)
        allgather_dpcpp<int32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == S64)
        allgather_dpcpp<int64_t>(stream, element_count, p, comm->nranks);
      else
        ITEX_LOG(FATAL) << "PrimitiveType "
                        << primitive_util::LowercasePrimitiveTypeName(dtype)
                        << " is not supported in AllGather.";
      streamlist_wait_stream(stream, p);

      Manager::instance().cv.notify_all();
    }
  }
}

void itex_alltoall(std::vector<const void*> send_buffers,
                   std::vector<void*> recv_buffers, int element_count,
                   PrimitiveType dtype, ITEX_GPUStream* gpu_stream,
                   ncclComm_t comm) {
  std::vector<AlltoAllParticipant> p;
  {
    itex::mutex_lock l(&(Manager::instance().mu));
    if (Manager::instance().alltoall_collectives.find(comm->id) ==
        Manager::instance().alltoall_collectives.end()) {
      std::vector<AlltoAllParticipant> participants;
      participants.push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      Manager::instance().alltoall_collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().alltoall_collectives[comm->id].push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      p = Manager::instance().alltoall_collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(&l);
    } else {
      Manager::instance().alltoall_collectives.erase(comm->id);
      std::sort(
          p.begin(), p.end(),
          [](const AlltoAllParticipant& a,
             const AlltoAllParticipant& b) -> bool { return a.rank < b.rank; });

      ITEX_GPUStream* stream = p[0].stream;
      stream_wait_streamlist(stream, p);
      if (dtype == PRED)
        alltoall_dpcpp<bool>(stream, element_count, p, comm->nranks);
      else if (dtype == F32)
        alltoall_dpcpp<float>(stream, element_count, p, comm->nranks);
      else if (dtype == F64)
        alltoall_dpcpp<double>(stream, element_count, p, comm->nranks);
      else if (dtype == S32)
        alltoall_dpcpp<int32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == S64)
        alltoall_dpcpp<int64_t>(stream, element_count, p, comm->nranks);
      else
        ITEX_LOG(FATAL) << "PrimitiveType "
                        << primitive_util::LowercasePrimitiveTypeName(dtype)
                        << " is not supported in AllToAll.";
      streamlist_wait_stream(stream, p);

      Manager::instance().cv.notify_all();
    }
  }
}

void itex_reduce_scatter(const void* send_buffer, void* recv_buffer,
                         int element_count, PrimitiveType dtype,
                         ReductionKind reduction_kind,
                         ITEX_GPUStream* gpu_stream, ncclComm_t comm) {
  std::vector<Participant> p;
  {
    itex::mutex_lock l(&(Manager::instance().mu));
    if (Manager::instance().collectives.find(comm->id) ==
        Manager::instance().collectives.end()) {
      std::vector<Participant> participants;
      participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      Manager::instance().collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().collectives[comm->id].push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      p = Manager::instance().collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(&l);
    } else {
      Manager::instance().collectives.erase(comm->id);
      std::sort(p.begin(), p.end(),
                [](const Participant& a, const Participant& b) -> bool {
                  return a.rank < b.rank;
                });

      ITEX_GPUStream* stream = p[0].stream;
      stream_wait_streamlist(stream, p);

      if (reduction_kind == ReductionKind::SUM) {
        if (dtype == PRED)
          reducescatter_dpcpp<bool, sycl::plus<bool>>(stream, element_count, p,
                                                      comm->nranks);
        else if (dtype == F32)
          reducescatter_dpcpp<float, sycl::plus<float>>(stream, element_count,
                                                        p, comm->nranks);
        else if (dtype == F64)
          reducescatter_dpcpp<double, sycl::plus<double>>(stream, element_count,
                                                          p, comm->nranks);
        else if (dtype == S32)
          reducescatter_dpcpp<int32_t, sycl::plus<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          reducescatter_dpcpp<int64_t, sycl::plus<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C64)
          reducescatter_dpcpp<std::complex<float>,
                              sycl::plus<std::complex<float>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C128)
          reducescatter_dpcpp<std::complex<double>,
                              sycl::plus<std::complex<double>>>(
              stream, element_count, p, comm->nranks);
        else
          ITEX_LOG(FATAL) << "PrimitiveType "
                          << primitive_util::LowercasePrimitiveTypeName(dtype)
                          << " is not supported in AllReduce.";
      } else if (reduction_kind == ReductionKind::PRODUCT) {
        if (dtype == PRED)
          reducescatter_dpcpp<bool, sycl::multiplies<bool>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F32)
          reducescatter_dpcpp<float, sycl::multiplies<float>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F64)
          reducescatter_dpcpp<double, sycl::multiplies<double>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S32)
          reducescatter_dpcpp<int32_t, sycl::multiplies<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          reducescatter_dpcpp<int64_t, sycl::multiplies<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C64)
          reducescatter_dpcpp<std::complex<float>,
                              sycl::multiplies<std::complex<float>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C128)
          reducescatter_dpcpp<std::complex<double>,
                              sycl::multiplies<std::complex<double>>>(
              stream, element_count, p, comm->nranks);
        else
          ITEX_LOG(FATAL) << "PrimitiveType "
                          << primitive_util::LowercasePrimitiveTypeName(dtype)
                          << " is not supported in AllReduce.";
      } else if (reduction_kind == ReductionKind::MIN) {
        if (dtype == PRED)
          reducescatter_dpcpp<bool, sycl::minimum<bool>>(stream, element_count,
                                                         p, comm->nranks);
        else if (dtype == F32)
          reducescatter_dpcpp<float, sycl::minimum<float>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F64)
          reducescatter_dpcpp<double, sycl::minimum<double>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S32)
          reducescatter_dpcpp<int32_t, sycl::minimum<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          reducescatter_dpcpp<int64_t, sycl::minimum<int64_t>>(
              stream, element_count, p, comm->nranks);
        else
          ITEX_LOG(FATAL) << "PrimitiveType "
                          << primitive_util::LowercasePrimitiveTypeName(dtype)
                          << " is not supported in AllReduce.";
      } else if (reduction_kind == ReductionKind::MAX) {
        if (dtype == PRED)
          reducescatter_dpcpp<bool, sycl::maximum<bool>>(stream, element_count,
                                                         p, comm->nranks);
        else if (dtype == F32)
          reducescatter_dpcpp<float, sycl::maximum<float>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F64)
          reducescatter_dpcpp<double, sycl::maximum<double>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S32)
          reducescatter_dpcpp<int32_t, sycl::maximum<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          reducescatter_dpcpp<int64_t, sycl::maximum<int64_t>>(
              stream, element_count, p, comm->nranks);
        else
          ITEX_LOG(FATAL) << "PrimitiveType "
                          << primitive_util::LowercasePrimitiveTypeName(dtype)
                          << " is not supported in AllReduce.";
      } else {
        ITEX_LOG(FATAL) << "ReductionKind " << static_cast<int>(reduction_kind)
                        << " is not supported in AllReduce.";
      }

      streamlist_wait_stream(stream, p);
      Manager::instance().cv.notify_all();
    }
  }
}

void itex_collective_permute(const void* send_buffer, void* recv_buffer,
                             int element_count, PrimitiveType dtype,
                             const absl::optional<int64_t>& source_id,
                             const absl::optional<int64_t>& target_id,
                             ITEX_GPUStream* gpu_stream, ncclComm_t comm) {
  std::vector<PermuteParticipant> p;
  {
    itex::mutex_lock l(&(Manager::instance().mu));
    if (Manager::instance().permute_collectives.find(comm->id) ==
        Manager::instance().permute_collectives.end()) {
      std::vector<PermuteParticipant> participants;
      participants.push_back({gpu_stream, send_buffer, recv_buffer, source_id,
                              target_id, comm->rank});
      Manager::instance().permute_collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().permute_collectives[comm->id].push_back(
          {gpu_stream, send_buffer, recv_buffer, source_id, target_id,
           comm->rank});
      p = Manager::instance().permute_collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(&l);
    } else {
      Manager::instance().permute_collectives.erase(comm->id);
      std::sort(
          p.begin(), p.end(),
          [](const PermuteParticipant& a, const PermuteParticipant& b) -> bool {
            return a.rank < b.rank;
          });

      ITEX_GPUStream* stream = p[0].stream;
      stream_wait_streamlist(stream, p);
      if (dtype == PRED)
        permute_dpcpp<bool>(stream, element_count, p, comm->nranks);
      else if (dtype == F32)
        permute_dpcpp<float>(stream, element_count, p, comm->nranks);
      else if (dtype == F64)
        permute_dpcpp<double>(stream, element_count, p, comm->nranks);
      else if (dtype == S32)
        permute_dpcpp<int32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == S64)
        permute_dpcpp<int64_t>(stream, element_count, p, comm->nranks);
      else
        ITEX_LOG(FATAL) << "PrimitiveType "
                        << primitive_util::LowercasePrimitiveTypeName(dtype)
                        << " is not supported in AllToAll.";
      streamlist_wait_stream(stream, p);

      Manager::instance().cv.notify_all();
    }
  }
}

}  // namespace gpu
}  // namespace itex_xla
#endif  // ITEX_USE_CCL
