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

#ifndef ITEX_CORE_DEVICES_BFC_ALLOCATOR_H_
#define ITEX_CORE_DEVICES_BFC_ALLOCATOR_H_

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "itex/core/devices/allocator.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/hw_info.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/mutex.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"

namespace itex {

// Currently, the default strategy of itex custom device allocator is BFC
class BFCAllocator : public Allocator {
 public:
  explicit BFCAllocator(ITEX_GPUDevice* device);
  ~BFCAllocator() override;
  void* AllocateRaw(size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  string Name() override { return "itex_device_bfc"; }

 private:
  ITEX_GPUDevice* device_;
  size_t memory_limit_;
  static constexpr size_t kMinAllocationBits = 8;
  static constexpr size_t kMinAllocationSize = 1 << kMinAllocationBits;
  typedef int BinNum;
  static constexpr int kInvalidBinNum = -1;
  // The following means that the largest bin'd chunk size is 256 << 21 = 512MB.
  static constexpr int kNumBins = 21;

  struct Bin;

  // A ChunkHandle is an index into the chunks_ vector in BFCAllocator
  // kInvalidChunkHandle means an invalid chunk
  typedef size_t ChunkHandle;
  static constexpr ChunkHandle kInvalidChunkHandle = SIZE_MAX;

  // A Chunk points to a piece of memory that's either entirely free or entirely
  // in use by one user memory allocation.
  //
  // An AllocationRegion's memory is split up into one or more disjoint Chunks,
  // which together cover the whole region without gaps.  Chunks participate in
  // a doubly-linked list, and the prev/next pointers point to the physically
  // adjacent chunks.
  //
  // Since a chunk cannot be partially in use, we may need to split a free chunk
  // in order to service a user allocation.  We always merge adjacent free
  // chunks.
  //
  // Chunks contain information about whether they are in use or whether they
  // are free, and contain a pointer to the bin they are in.
  struct Chunk {
    size_t size = 0;  // Full size of buffer.

    // We sometimes give chunks that are larger than needed to reduce
    // fragmentation.  requested_size keeps track of what the client
    // actually wanted so we can understand whether our splitting
    // strategy is efficient.
    size_t requested_size = 0;

    // allocation_id is set to -1 when the chunk is not in use. It is assigned a
    // value greater than zero before the chunk is returned from
    // AllocateRaw, and this value is unique among values assigned by
    // the parent allocator.
    int64_t allocation_id = -1;
    void* ptr = nullptr;  // pointer to granted subbuffer.

    // If not kInvalidChunkHandle, the memory referred to by 'prev' is directly
    // preceding the memory used by this chunk.  E.g., It should start
    // at 'ptr - prev->size'
    ChunkHandle prev = kInvalidChunkHandle;

    // If not kInvalidChunkHandle, the memory referred to by 'next' is directly
    // following the memory used by this chunk.  E.g., It should be at
    // 'ptr + size'
    ChunkHandle next = kInvalidChunkHandle;

    // What bin are we in?
    BinNum bin_num = kInvalidBinNum;

    bool in_use() const { return allocation_id != -1; }
  };  // struct Chunk

  // A Bin is a collection of similar-sized free chunks.
  // Allocated chunks are never in a Bin.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    class ChunkComparator {
     public:
      explicit ChunkComparator(BFCAllocator* allocator)
          : allocator_(allocator) {}
      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const ChunkHandle ha, const ChunkHandle hb) const {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      BFCAllocator* allocator_;  // The parent allocator
    };

    typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
    // List of free chunks within the bin, sorted by chunk size.
    // Chunk * not owned.
    FreeChunkSet free_chunks;
    Bin(BFCAllocator* allocator, size_t bs)
        : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };  // struck Bin

  // BFCAllocator allocates memory into a collection of disjoint
  // AllocationRegions.  Each AllocationRegion corresponds to one call to
  // SubAllocator::Alloc().  (Actually, if a subsequent call to
  // SubAllocator::Alloc() returns another region immediately adjacent to the
  // last, it will be used to extend the first AllocationRegion, not create a
  // separate one.)
  //
  // An AllocationRegion contains one or more Chunks, covering all of its
  // memory.  Its primary job is to map pointers to ChunkHandles.
  //
  // This class is thread-compatible.
  class AllocationRegion {
   public:
    AllocationRegion(void* ptr, size_t memory_size)
        : ptr_(ptr),
          memory_size_(memory_size),
          end_ptr_(
              static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)) {
      ITEX_DCHECK_EQ(0, memory_size % kMinAllocationSize);
      const size_t n_handles =
          (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_.resize(n_handles, kInvalidChunkHandle);
    }

    AllocationRegion() = default;
    AllocationRegion(AllocationRegion&& other) { Swap(&other); }
    AllocationRegion& operator=(AllocationRegion&& other) {
      Swap(&other);
      return *this;
    }

    void* ptr() const { return ptr_; }
    void* end_ptr() const { return end_ptr_; }
    size_t memory_size() const { return memory_size_; }
    void extend(size_t size) {
      memory_size_ += size;
      ITEX_DCHECK_EQ(0, memory_size_ % kMinAllocationSize);

      end_ptr_ = static_cast<void*>(static_cast<char*>(end_ptr_) + size);
      const size_t n_handles =
          (memory_size_ + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_.resize(n_handles, kInvalidChunkHandle);
    }
    ChunkHandle get_handle(const void* p) const {
      return handles_[IndexFor(p)];
    }
    void set_handle(const void* p, ChunkHandle h) { handles_[IndexFor(p)] = h; }
    void erase(const void* p) { set_handle(p, kInvalidChunkHandle); }

   private:
    void Swap(AllocationRegion* other) {
      std::swap(ptr_, other->ptr_);
      std::swap(memory_size_, other->memory_size_);
      std::swap(end_ptr_, other->end_ptr_);
      std::swap(handles_, other->handles_);
    }

    size_t IndexFor(const void* p) const {
      std::uintptr_t p_int = reinterpret_cast<std::uintptr_t>(p);
      std::uintptr_t base_int = reinterpret_cast<std::uintptr_t>(ptr_);
      ITEX_DCHECK_GE(p_int, base_int);
      ITEX_DCHECK_LT(p_int, base_int + memory_size_);
      return static_cast<size_t>(((p_int - base_int) >> kMinAllocationBits));
    }

    // Metadata about the allocation region.
    void* ptr_ = nullptr;
    size_t memory_size_ = 0;
    void* end_ptr_ = nullptr;

    // Array of size "memory_size / kMinAllocationSize".  It is
    // indexed by (p-base) / kMinAllocationSize, contains ChunkHandle
    // for the memory allocation represented by "p"
    std::vector<ChunkHandle> handles_;

    TF_DISALLOW_COPY_AND_ASSIGN(AllocationRegion);
  };  // class AllocationRegion

  // RegionManager aggregates one or more "AllocationRegions" and provides
  // a layer of indirection from pointers to the underlying ChunkHandle,
  // allowing allocation across multiple discontiguous memory regions.
  //
  // This class is thread-compatible.
  class RegionManager {
   public:
    RegionManager() {}
    ~RegionManager() {}

    void AddAllocationRegion(void* ptr, size_t memory_size) {
      // Insert sorted by end_ptr.
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      regions_.insert(entry, AllocationRegion(ptr, memory_size));
    }

    // Adds an alloation region for the given ptr and size, potentially
    // extending a region if ptr matches the end_ptr of an existing region.
    // If a region is extended, returns a pointer to the extended region so that
    // the BFC allocator can reason about chunkification.
    AllocationRegion* AddOrExtendAllocationRegion(void* ptr,
                                                  size_t memory_size) {
      // Insert sorted by end_ptr.
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      // Check if can be coalesced with preceding region.
      if (entry != regions_.begin()) {
        auto preceding_region = entry - 1;
        if (preceding_region->end_ptr() == ptr) {
          preceding_region->extend(memory_size);
          return &*preceding_region;
        }
      }
      regions_.insert(entry, AllocationRegion(ptr, memory_size));
      return nullptr;
    }

    std::vector<AllocationRegion>::iterator RemoveAllocationRegion(
        std::vector<AllocationRegion>::iterator it) {
      return regions_.erase(it);
    }

    ChunkHandle get_handle(const void* p) const {
      return RegionFor(p)->get_handle(p);
    }

    void set_handle(const void* p, ChunkHandle h) {
      return MutableRegionFor(p)->set_handle(p, h);
    }
    void erase(const void* p) { return MutableRegionFor(p)->erase(p); }

    const std::vector<AllocationRegion>& regions() const { return regions_; }

   private:
    static bool Comparator(const void* ptr, const AllocationRegion& other) {
      return ptr < other.end_ptr();
    }

    AllocationRegion* MutableRegionFor(const void* p) {
      return const_cast<AllocationRegion*>(RegionFor(p));
    }

    const AllocationRegion* RegionFor(const void* p) const {
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), p, &Comparator);

      if (entry != regions_.end()) {
        return &(*entry);
      }

      ITEX_LOG(FATAL) << "Could not find Region for " << p;
      return nullptr;
    }

   private:
    std::vector<AllocationRegion> regions_;
  };  // class RegionManager

  ChunkHandle AllocateChunk() TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void DeallocateChunk(ChunkHandle h) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  Chunk* ChunkFromHandle(ChunkHandle h) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);
  const Chunk* ChunkFromHandle(ChunkHandle h) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Returns a pointer to an underlying allocated chunk of size
  // 'rounded_bytes'.
  void* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes)
      TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Returns 'bytes' rounded up to the next highest kMinAllocationSize.
  static size_t RoundedBytes(size_t bytes);

  // Map from bin size to Bin
  Bin* BinFromIndex(BinNum index) {
    return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
  }

  size_t BinNumToSize(BinNum index) {
    return static_cast<size_t>(256) << index;
  }

  Bin* BinForSize(size_t bytes) { return BinFromIndex(BinNumForSize(bytes)); }

  BinNum BinNumForSize(size_t bytes) {
    uint64 v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
    int b = std::min(kNumBins - 1, Log2FloorNonZero(v));
    return b;
  }

  // Returns floor(log2(n)).
  inline int Log2FloorNonZero(uint64 n) {
#if defined(__GNUC__)
    return 63 ^ __builtin_clzll(n);
#else
    return Log2FloorNonZeroSlow(n);
#endif
  }

  inline int Log2FloorNonZeroSlow(uint64 n) {
    int r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r - 1;
  }

  // Removes the free chunk pointed to by 'c' from the set free_chunks.
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c)
      TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Splits the chunk specified by 'h' into two chunks, one at least
  // of size 'num_bytes'.
  void SplitChunk(ChunkHandle h, size_t num_bytes)
      TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Adds the chunk 'h' to the proper free bin.
  void InsertFreeChunkIntoBin(ChunkHandle h) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Try to add a new memory region that can satisfy an allocation of
  // 'rounded_bytes' bytes.  Returns true on success and false on
  // failure.
  bool Extend(size_t rounded_bytes) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Extend a large memory(almost all device memory)
  bool ExtendLarge(size_t rounded_bytes) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);
  // Extend size near the required.
  bool ExtendSmall(size_t rounded_bytes) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  ChunkHandle TryToCoalesce(ChunkHandle h) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Removes a free chunk from the bin.
  void RemoveFreeChunkFromBin(ChunkHandle h) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Merges the two chunk handles.  Requires that the chunks are
  // contiguous in their allocation.
  void Merge(ChunkHandle h, ChunkHandle h2) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Removes the chunk metadata represented by 'h'.
  void DeleteChunk(ChunkHandle h) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  int64 AllocMode();

  char bins_space_[sizeof(Bin) * kNumBins];
  mutable mutex lock_;
  RegionManager region_manager_ TF_GUARDED_BY(lock_);

  // Pointer to head of linked list of free Chunks
  ChunkHandle free_chunks_list_ TF_GUARDED_BY(lock_);

  // The size of the current region allocation.
  size_t curr_region_allocation_bytes_;

  // The total number of allocated bytes by the allocator.
  size_t total_region_allocated_bytes_ = 0;

  std::vector<Chunk> chunks_ TF_GUARDED_BY(lock_);
  TF_DISALLOW_COPY_AND_ASSIGN(BFCAllocator);

  size_t LimitAlloc();
  size_t GetLimitAlloc();
};  // class BFCAllocator

}  // namespace itex
#endif  // ITEX_CORE_DEVICES_BFC_ALLOCATOR_H_
