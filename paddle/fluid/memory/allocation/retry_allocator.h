// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>              // NOLINT
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <utility>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

class RetryAllocator : public Allocator {
 public:
  RetryAllocator(std::shared_ptr<Allocator> allocator, size_t retry_ms)
      : underlying_allocator_(std::move(allocator)), retry_time_(retry_ms) {
    PADDLE_ENFORCE_NOT_NULL(
        underlying_allocator_,
        platform::errors::InvalidArgument(
            "Underlying allocator of RetryAllocator is NULL"));
    PADDLE_ENFORCE_EQ(
        underlying_allocator_->IsAllocThreadSafe(),
        true,
        platform::errors::PreconditionNotMet(
            "Underlying allocator of RetryAllocator is not thread-safe"));
  }

  bool IsAllocThreadSafe() const override { return true; }
  // return real used, total is in alloc
  size_t GetTotalMemInfo(size_t *total, size_t *available) {
    return underlying_allocator_->GetTotalMemInfo(total, available);
  }

 protected:
  void FreeImpl(phi::Allocation* allocation) override;
  phi::Allocation* AllocateImpl(size_t size) override;
  uint64_t ReleaseImpl(const platform::Place& place) override {
    return underlying_allocator_->Release(place);
  }

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
  std::chrono::milliseconds retry_time_;
  std::mutex mutex_;
  std::condition_variable cv_;

  std::atomic<size_t> waited_allocate_size_{0};
};
using Allocation = phi::Allocation;
class SampleAllocator : public Allocator {
  /**
   * Descriptor for device memory allocations
   */
  struct BlockDescriptor {
    Allocation *d_ptr;  // Device pointer
    size_t bytes;       // Size of allocation in bytes
    size_t used;        // Real used
    unsigned int bin;   // Bin enumeration

    explicit BlockDescriptor(Allocation *ptr);
    BlockDescriptor();
    static bool ptrcompare(const BlockDescriptor &a, const BlockDescriptor &b);
    static bool sizecompare(const BlockDescriptor &a, const BlockDescriptor &b);
  };
  // BlockDescriptor comparator function interface
  typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);
  /// Set type for cached blocks (ordered by size)
  typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;
  /// Set type for live blocks (ordered by ptr)
  typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

 public:
  // Total Bytes
  struct TotalBytes {
    size_t free = 0;
    size_t live = 0;
    size_t used = 0;
  };
  explicit SampleAllocator(std::shared_ptr<Allocator> allocator);
  bool IsAllocThreadSafe() const override { return true; }
  void GetMemInfo(TotalBytes *info);
  // return real used, total is in alloc
  size_t GetTotalMemInfo(size_t *total, size_t *available) {
    *total = cached_bytes_.live + cached_bytes_.free;
    *available = cached_bytes_.free;
    return cached_bytes_.used;
  }

 protected:
  void FreeImpl(Allocation *allocation) override;
  Allocation *AllocateImpl(size_t size) override;
  uint64_t ReleaseImpl(const platform::Place &place) override {
    FreeAllCache();
    return allocator_->Release(place);
  }
  void FreeAllCache(void);

 private:
  std::shared_ptr<Allocator> allocator_;
  std::mutex mutex_;

  unsigned int bin_growth_;  /// Geometric growth factor for bin-sizes
  unsigned int min_bin_;     /// Minimum bin enumeration
  size_t min_bin_bytes_;
  size_t max_bin_bytes_;

  TotalBytes cached_bytes_;  /// Map of device ordinal to aggregate cached bytes
                             /// on that device
  CachedBlocks
      cached_blocks_;  /// Set of cached device allocations available for reuse
  BusyBlocks live_blocks_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
