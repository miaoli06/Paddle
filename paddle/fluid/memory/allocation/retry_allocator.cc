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

#include "paddle/fluid/memory/allocation/retry_allocator.h"

#include "glog/logging.h"

PADDLE_DEFINE_EXPORTED_int32(sample_max_bin_bytes, 2048, "sample max bytes in pool MB");
PADDLE_DEFINE_EXPORTED_int32(sample_bin_growth, 2, "sample growth memory by bin");
PADDLE_DEFINE_EXPORTED_int32(sample_min_bin, 8, "sample min bin number");
PADDLE_DEFINE_EXPORTED_bool(sample_debug_info, false, "sample print debug info");

namespace paddle {
namespace memory {
namespace allocation {

class WaitedAllocateSizeGuard {
 public:
  WaitedAllocateSizeGuard(std::atomic<size_t>* waited_size,
                          size_t requested_size)
      : waited_size_(waited_size), requested_size_(requested_size) {
    waited_size_->fetch_add(requested_size_,
                            std::memory_order::memory_order_relaxed);
  }

  ~WaitedAllocateSizeGuard() {
    waited_size_->fetch_sub(requested_size_,
                            std::memory_order::memory_order_relaxed);
  }

 private:
  std::atomic<size_t>* waited_size_;
  size_t requested_size_;
};

void RetryAllocator::FreeImpl(phi::Allocation* allocation) {
  // Delete underlying allocation first.
  size_t size = allocation->size();
  underlying_allocator_->Free(allocation);
  if (UNLIKELY(waited_allocate_size_)) {
    VLOG(10) << "Free " << size
             << " bytes and notify all waited threads, "
                "where waited_allocate_size_ = "
             << waited_allocate_size_;
    cv_.notify_all();
  }
}

phi::Allocation* RetryAllocator::AllocateImpl(size_t size) {
  auto alloc_func = [&, this]() {
    return underlying_allocator_->Allocate(size).release();
  };
  // In fact, we can unify the code of allocation success and failure
  // But it would add lock even when allocation success at the first time
  try {
    return alloc_func();
  } catch (BadAlloc&) {
    {
      WaitedAllocateSizeGuard guard(&waited_allocate_size_, size);
      VLOG(10) << "Allocation failed when allocating " << size
               << " bytes, waited_allocate_size_ = " << waited_allocate_size_;
      // We can just write allocation retry inside the predicate function of
      // wait_until. But it needs to acquire the lock when executing predicate
      // function. For better performance, we use loop here
      auto end_time = std::chrono::high_resolution_clock::now() + retry_time_;
      auto wait_until = [&, this] {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_until(lock, end_time);
      };

      size_t retry_time = 0;
      while (wait_until() != std::cv_status::timeout) {
        try {
          return alloc_func();
        } catch (BadAlloc&) {
          // do nothing when it is not timeout
          ++retry_time;
          VLOG(10) << "Allocation failed when retrying " << retry_time
                   << " times when allocating " << size
                   << " bytes. Wait still.";
        } catch (...) {
          throw;
        }
      }
    }
    VLOG(10) << "Allocation failed because of timeout when allocating " << size
             << " bytes.";
    return alloc_func();  // If timeout, try last allocation request.
  } catch (...) {
    throw;
  }
}

static const unsigned int INVALID_BIN = (unsigned int)-1;
SampleAllocator::BlockDescriptor::BlockDescriptor(phi::Allocation* ptr)
    : d_ptr(ptr), bytes(0), used(0), bin(INVALID_BIN) {}
SampleAllocator::BlockDescriptor::BlockDescriptor()
    : d_ptr(NULL), bytes(0), used(0), bin(INVALID_BIN) {}
bool SampleAllocator::BlockDescriptor::ptrcompare(const BlockDescriptor& a,
                                                  const BlockDescriptor& b) {
  return (a.d_ptr < b.d_ptr);
}
bool SampleAllocator::BlockDescriptor::sizecompare(const BlockDescriptor& a,
                                                   const BlockDescriptor& b) {
  return (a.bytes < b.bytes);
}
SampleAllocator::SampleAllocator(std::shared_ptr<Allocator> allocator)
    : allocator_(std::move(allocator)),
      bin_growth_(FLAGS_sample_bin_growth),
      min_bin_(FLAGS_sample_min_bin),
      min_bin_bytes_(pow(bin_growth_, min_bin_)),
      max_bin_bytes_(FLAGS_sample_max_bin_bytes * 1024 * 1024),
      cached_blocks_(BlockDescriptor::sizecompare),
      live_blocks_(BlockDescriptor::ptrcompare) {
  PADDLE_ENFORCE_NOT_NULL(
      allocator_, platform::errors::InvalidArgument(
                      "Underlying allocator of SampleAllocator is NULL"));
  VLOG(0) << "SampleAllocator init";
}
void SampleAllocator::FreeImpl(phi::Allocation* allocation) {
  if (allocation == NULL) {
    return;
  }
  bool recached = false;
  BlockDescriptor search_key(allocation);

  mutex_.lock();
  auto block_itr = live_blocks_.find(search_key);
  if (block_itr != live_blocks_.end()) {
    search_key = *block_itr;
    live_blocks_.erase(block_itr);
    cached_bytes_.live -= search_key.bytes;
    cached_bytes_.used -= search_key.used;
    if (search_key.bin != INVALID_BIN) {
      recached = true;
      cached_blocks_.insert(search_key);
      cached_bytes_.free += search_key.bytes;
    }
  }
  mutex_.unlock();

  if (!recached) {
    allocator_->Free(allocation);
  }

  if (FLAGS_sample_debug_info && search_key.bin == INVALID_BIN) {
    VLOG(0) << "pool total: " << (cached_bytes_.live >> 20)
            << "MB, used: " << (cached_bytes_.used >> 20) << "MB, free"
            << (cached_bytes_.free >> 20)
            << "MB, free big memory: " << search_key.bytes << " bytes";
  }
}
// alloc memory
phi::Allocation* SampleAllocator::AllocateImpl(size_t bytes) {
  // Create a block descriptor for the requested allocation
  bool found = false;
  BlockDescriptor search_key;
  search_key.used = bytes;
  if (bytes > max_bin_bytes_) {
    search_key.bin = INVALID_BIN;
    search_key.bytes = bytes;
  } else {
    if (bytes < min_bin_bytes_) {
      search_key.bin = min_bin_;
      search_key.bytes = min_bin_bytes_;
    } else {
      search_key.bin = 0;
      search_key.bytes = 1;
      while (search_key.bytes < bytes) {
        search_key.bytes *= bin_growth_;
        ++search_key.bin;
      }
    }
    mutex_.lock();
    auto block_itr = cached_blocks_.lower_bound(search_key);
    if ((block_itr != cached_blocks_.end()) &&
        (block_itr->bin == search_key.bin)) {
      found = true;
      search_key = *block_itr;
      search_key.used = bytes;
      live_blocks_.insert(search_key);
      // Remove from free blocks
      cached_bytes_.free -= search_key.bytes;
      cached_bytes_.live += search_key.bytes;
      cached_bytes_.used += search_key.used;
      cached_blocks_.erase(block_itr);
    }
    mutex_.unlock();
  }
  if (!found) {
    try {
      search_key.d_ptr = allocator_->Allocate(search_key.bytes).release();
    } catch (BadAlloc&) {
      // release all free cache
      FreeAllCache();
      // cuda malloc
      search_key.d_ptr = allocator_->Allocate(search_key.bytes).release();
    } catch (...) {
      throw;
    }
    mutex_.lock();
    live_blocks_.insert(search_key);
    cached_bytes_.live += search_key.bytes;
    cached_bytes_.used += search_key.used;
    mutex_.unlock();

    if (FLAGS_sample_debug_info && search_key.bin == INVALID_BIN) {
      VLOG(0) << "pool total: " << (cached_bytes_.live >> 20)
              << "MB, used: " << (cached_bytes_.used >> 20) << "MB, free"
              << (cached_bytes_.free >> 20)
              << "MB, cuda alloc big memory: " << bytes << " bytes";
    }
  }
  return search_key.d_ptr;
}
void SampleAllocator::FreeAllCache(void) {
  mutex_.lock();
  if (cached_blocks_.empty()) {
    mutex_.unlock();
    return;
  }
  while (!cached_blocks_.empty()) {
    auto begin = cached_blocks_.begin();
    allocator_->Free(begin->d_ptr);
    cached_bytes_.free -= begin->bytes;
    cached_blocks_.erase(begin);
  }
  mutex_.unlock();
}

void SampleAllocator::GetMemInfo(TotalBytes* info) { *info = cached_bytes_; }

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
