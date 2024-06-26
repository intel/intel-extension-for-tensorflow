/* Copyright (c) 2023 Intel Corporation

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_UTILS_NUMA_H_
#define ITEX_CORE_UTILS_NUMA_H_

#include "itex/core/utils/platform.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace port {

// Returns true iff NUMA functions are supported.
bool NUMAEnabled();

// Returns the number of NUMA nodes present with respect to CPU operations.
// Typically this will be the number of sockets where some RAM has greater
// affinity with one socket than another.
int NUMANumNodes();

static const int kNUMANoAffinity = -1;

// If possible sets affinity of the current thread to the specified NUMA node.
// If node == kNUMANoAffinity removes affinity to any particular node.
void NUMASetThreadNodeAffinity(int node);

// Returns NUMA node affinity of the current thread, kNUMANoAffinity if none.
int NUMAGetThreadNodeAffinity();

// Like AlignedMalloc, but allocates memory with affinity to the specified NUMA
// node.
//
// Notes:
//  1. node must be >= 0 and < NUMANumNodes.
//  1. minimum_alignment must a factor of system page size, the memory
//     returned will be page-aligned.
//  2. This function is likely significantly slower than AlignedMalloc
//     and should not be used for lots of small allocations.  It makes more
//     sense as a backing allocator for BFCAllocator, PoolAllocator, or similar.
void* NUMAMalloc(int node, size_t size, int minimum_alignment);

// Memory allocated by NUMAMalloc must be freed via NUMAFree.
void NUMAFree(void* ptr, size_t size);

// Returns NUMA node affinity of memory address, kNUMANoAffinity if none.
int NUMAGetMemAffinity(const void* ptr);

}  // namespace port
}  // namespace itex
#endif  // ITEX_CORE_UTILS_NUMA_H_
