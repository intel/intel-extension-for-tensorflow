/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/devices/gpu/gpu_info.h"

#include <map>

namespace itex {

#ifndef INTEL_CPU_ONLY
DeviceInfo* GetDeviceInfo(ITEX_GPUStream* stream) {
  static thread_local std::map<ITEX_GPUStream*, DeviceInfo*> stream_device_map;
  auto iter = stream_device_map.find(stream);
  if (iter != stream_device_map.end()) return iter->second;
  auto device_ = stream->get_device();
  auto context_ = stream->get_context();
  auto max_work_group_size_ =
      device_.template get_info<sycl::info::device::max_work_group_size>();
  // init device property
  gpuDeviceProp_t device_prop_;
  device_prop_.multiProcessorCount =
      device_.template get_info<sycl::info::device::max_compute_units>();
  device_prop_.maxThreadsPerBlock =
      device_.template get_info<sycl::info::device::max_work_group_size>();
  device_prop_.maxThreadsPerMultiProcessor =
      device_.template get_info<sycl::info::device::max_work_group_size>();
  device_prop_.sharedMemPerBlock =
      device_.template get_info<sycl::info::device::local_mem_size>();
  device_prop_.major = 0;
  device_prop_.minor = 0;
  device_prop_.local_mem_type =
      device_.template get_info<sycl::info::device::local_mem_type>();

  stream_device_map[stream] =
      new DeviceInfo(device_prop_, device_, context_, max_work_group_size_);
  return stream_device_map[stream];
}
#endif

}  // namespace itex
