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

#ifndef ITEX_CORE_DEVICES_ALLOCATOR_H_
#define ITEX_CORE_DEVICES_ALLOCATOR_H_

#include <string>

namespace itex {

class Allocator {
 public:
  Allocator() = default;
  virtual ~Allocator() = default;

  // Return a string identifying this allocator
  virtual std::string Name() = 0;

  // Return an uninitialized block of memory that is "num_bytes" bytes
  // in size.
  virtual void* AllocateRaw(size_t num_bytes) = 0;

  // Deallocate a block of memory pointer to by "ptr"
  // REQUIRES: "ptr" was previously returned by a call to AllocateRaw
  virtual void DeallocateRaw(void* ptr) = 0;
};

}  // namespace itex
#endif  // ITEX_CORE_DEVICES_ALLOCATOR_H_
