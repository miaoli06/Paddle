// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_BOX_PS
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace framework {
const int CUDA_NUM_THREADS = platform::PADDLE_CUDA_NUM_THREADS;
#define GET_BLOCK(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define CUDA_BLOCK(N) GET_BLOCK(N), CUDA_NUM_THREADS, 0

const int max_gpu_thread_num = 64 * 2048 * 5;

struct EmbedxQuantOp {
  __forceinline__ __device__ void copy(float* dest, const float* src,
                                       const int& idx,
                                       const float& scale) const {
    *dest = *(reinterpret_cast<const int16_t*>(src) + idx) * scale;
  }
};
struct EmbedxNormalOp {
  __forceinline__ __device__ void copy(float* dest, const float* src,
                                       const int& idx,
                                       const float& /**scale*/) const {
    *dest = src[idx];
  }
};
struct ExpandPushGetOp {
  __forceinline__ __device__ float get(float* expand, const int& row,
                                       const int& expand_id,
                                       const int& /**hidden*/,
                                       const int& expand_dim) const {
    return expand[row * expand_dim + expand_id];
  }
};
struct ExpandPushEmdGetOp {
  __forceinline__ __device__ float get(float* expand, const int& row,
                                       const int& expand_id, const int& hidden,
                                       const int& expand_dim) const {
    return expand[row * (hidden + expand_dim) + hidden + expand_id];
  }
};
template <typename T>
__forceinline__ __device__ const T& get_byfloat(const float* src) {
  return (*reinterpret_cast<const T*>(src));
}
template <typename T>
__forceinline__ __device__ void set_byfloat(float* dest, const T& val) {
  (*reinterpret_cast<T*>(dest)) = val;
}
template <typename TEmbedxOp>
__global__ void PullCopy(const TEmbedxOp& op,
                         const boxps::FeaturePullOffset* info,
                         const int pull_float_num, const int64_t total_len,
                         float** dest, const float* src, const int hidden,
                         int* total_dims, const int64_t* slot_lens,
                         const int slot_num, const int* key2slot,
                         const float scale, const int cvm_offset,
                         const int skip_offset, const uint32_t* restore_idx) {
  // only process no expand data
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    int pos = (restore_idx != nullptr) ? restore_idx[i] : i;
    const float* src_val = &src[pos * pull_float_num];
    float* dest_ptr = dest[x] + y * hidden;
    for (int k = 0; k < cvm_offset; ++k) {
      dest_ptr[k] = src_val[info->show + k + skip_offset];
    }
    // embedx
    int dim_size = hidden - cvm_offset;
    int embedx_size = get_byfloat<uint32_t>(&src_val[info->embedx_size]);
    total_dims[i] = static_cast<int>(embedx_size > 0);

    for (int k = 0; k < embedx_size; ++k) {
      op.copy(&dest_ptr[cvm_offset + k], &src_val[info->embedx], k, scale);
    }
    for (int k = embedx_size; k < dim_size; ++k) {
      dest_ptr[cvm_offset + k] = 0;
    }
  }  // end kernel loop
}

template <typename TEmbedxOp>
__global__ void PullCopyEx(const TEmbedxOp& op,
                           const boxps::FeaturePullOffset* info,
                           const int pull_float_num, const int64_t total_len,
                           float** dest, const float* src, const int hidden,
                           int* total_dims, const int64_t* slot_lens,
                           const int slot_num, const int* key2slot,
                           const float scale, const int cvm_offset,
                           const int skip_offset, const uint32_t* restore_idx) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / hidden;
    int col = i % hidden;
    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    int pos = (restore_idx != nullptr) ? restore_idx[idx] : idx;
    const float* src_val = &src[pos * pull_float_num];
    if (col == 0) {
      int embedx_size = get_byfloat<uint32_t>(&src_val[info->embedx_size]);
      total_dims[idx] = static_cast<int>(embedx_size > 0);
    }

    float* dest_ptr = dest[x] + y * hidden;
    if (col < cvm_offset) {
      dest_ptr[col] = src_val[info->show + col + skip_offset];
    } else {
      int embedx_size = get_byfloat<uint32_t>(&src_val[info->embedx_size]);
      int embedx_id = col - cvm_offset;
      if (embedx_id < embedx_size) {
        op.copy(&dest_ptr[col], &src_val[info->embedx], embedx_id, scale);
      } else {
        dest_ptr[col] = 0;
      }
    }
  }  // end kernel loop
}
//================================== support nncross
//================================
template <typename TEmbedxOp>
__global__ void PullCopyNNCross(
    const TEmbedxOp& op, const boxps::FeaturePullOffset* info,
    const int pull_float_num, const int64_t total_len, float** dest,
    const float* src, const int max_cols_num, const int hidden,
    const int expand_dim, int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const float scale,
    const int cvm_offset, const int skip_offset, const uint32_t* restore_idx) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / max_cols_num;
    int col = i % max_cols_num;
    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    int pos = (restore_idx != nullptr) ? restore_idx[idx] : idx;
    const float* src_val = &src[pos * pull_float_num];
    if (col == 0) {
      // embedx flags + expand flags   && *(keys[x] + y) != 0  && *(keys[x] + y)
      total_dims[idx] =
          (get_byfloat<uint32_t>(&src_val[info->embedx_size]) > 0) |
          ((get_byfloat<uint32_t>(&src_val[info->expand_size]) > 0) << 1);
    }
    if (col < cvm_offset) {  // cvm offset
      if (dest[x] != 0) {
        float* dest_ptr = dest[x] + y * hidden;
        dest_ptr[col] = src_val[info->show + col + skip_offset];
      }
    } else if (col < hidden) {  // embedx
      if (dest[x] == 0) {
        return;
      }
      float* dest_ptr = dest[x] + y * hidden;
      int embedx_size = get_byfloat<uint32_t>(&src_val[info->embedx_size]);
      int embedx_id = col - cvm_offset;
      if (embedx_id < embedx_size) {
        op.copy(&dest_ptr[col], &src_val[info->embedx], embedx_id, scale);
      } else {
        dest_ptr[col] = 0;
      }
    } else {  // expand
      if (dest[x + slot_num] == 0) {
        return;
      }
      float* dest_ptr = dest[x + slot_num] + y * expand_dim;
      int expand_id = col - hidden;
      int expand_size = get_byfloat<uint32_t>(&src_val[info->expand_size]);
      if (expand_id < expand_size) {
        op.copy(&dest_ptr[expand_id], &src_val[info->expand], expand_id, scale);
      } else {
        dest_ptr[expand_id] = 0;
      }
    }
  }  // end kernel loop
}

template <typename TEmbedxOp>
__global__ void PullCopyNNCrossWithEmb(
    const TEmbedxOp& op, const boxps::FeaturePullOffset* info,
    const int pull_float_num, const int64_t total_len, float** dest,
    const float* src, const int max_cols_num, const int hidden,
    const int expand_dim, int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const float scale,
    const int cvm_offset, const int skip_offset, const uint32_t* restore_idx) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / max_cols_num;
    int col = i % max_cols_num;
    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    int pos = (restore_idx != nullptr) ? restore_idx[idx] : idx;
    const float* src_val = &src[pos * pull_float_num];
    if (col == 0) {
      // embedx flags + expand flags   && *(keys[x] + y) != 0  && *(keys[x] + y)
      total_dims[idx] =
          (get_byfloat<uint32_t>(&src_val[info->embedx_size]) > 0) |
          ((get_byfloat<uint32_t>(&src_val[info->expand_size]) > 0) << 1);
    }

    if (col < cvm_offset) {  // cvm offset
      if (dest[x] != 0) {
        float* dest_ptr = dest[x] + y * hidden;
        dest_ptr[col] = src_val[info->show + col + skip_offset];
      }
      if (dest[x + slot_num] != 0) {
        float* dest_ptr = dest[x + slot_num] + y * (hidden + expand_dim);
        dest_ptr[col] = src_val[info->show + col + skip_offset];
      }
    } else if (col < hidden) {  // embedx
      if (dest[x] != 0) {
        float* dest_ptr = dest[x] + y * hidden;
        int embedx_size = get_byfloat<uint32_t>(&src_val[info->embedx_size]);
        int embedx_id = col - cvm_offset;
        if (embedx_id < embedx_size) {
          op.copy(&dest_ptr[col], &src_val[info->embedx], embedx_id, scale);
        } else {
          dest_ptr[col] = 0;
        }
      }
      if (dest[x + slot_num] != 0) {
        float* dest_ptr = dest[x + slot_num] + y * (hidden + expand_dim);
        int embedx_size = get_byfloat<uint32_t>(&src_val[info->embedx_size]);
        int embedx_id = col - cvm_offset;
        if (embedx_id < embedx_size) {
          op.copy(&dest_ptr[col], &src_val[info->embedx], embedx_id, scale);
        } else {
          dest_ptr[col] = 0;
        }
      }
    } else {  // expand
      if (dest[x + slot_num] == 0) {
        return;
      }
      float* dest_ptr = dest[x + slot_num] + y * (hidden + expand_dim);
      int expand_id = col - hidden;
      int expand_size = get_byfloat<uint32_t>(&src_val[info->expand_size]);
      if (expand_id < expand_size) {
        op.copy(&dest_ptr[col], &src_val[info->expand], expand_id, scale);
      } else {
        dest_ptr[col] = 0;
      }
    }
  }  // end kernel loop
}
//========================== feature var pull ========================
template <typename TEmbedxOp>
__global__ void PullCopyVariable(
    const TEmbedxOp& op, const boxps::FeaturePullOffset* info,
    const int pull_float_num, const int64_t total_len, float** dest,
    const float* src, const int max_cols_num, const int embedx_dim,
    const int expand_dim, int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const float scale,
    const int cvm_offset, const int skip_offset, const uint32_t* restore_idx) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / max_cols_num;
    int col = i % max_cols_num;
    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    int pos = (restore_idx != nullptr) ? restore_idx[idx] : idx;
    const float* src_val = &src[pos * pull_float_num];
    if (col == 0) {
      // embedx flags + expand flags   && *(keys[x] + y) != 0  && *(keys[x] + y)
      int embedx_size = get_byfloat<uint32_t>(&src_val[info->embedx_size]);
      total_dims[idx] =
          ((embedx_size == expand_dim) << 1) | (embedx_size == embedx_dim);
    }

    float* dest_ptr = 0;
    if (col < cvm_offset) {  // cvm offset
      if (dest[x + slot_num] != 0) {
        dest_ptr = dest[x + slot_num] + y * (expand_dim + cvm_offset);
      } else {
        dest_ptr = dest[x] + y * (embedx_dim + cvm_offset);
      }
      assert(dest_ptr != 0);
      dest_ptr[col] = src_val[info->show + col + skip_offset];
    } else {
      int embedx_id = col - cvm_offset;
      int embedx_size = get_byfloat<uint32_t>(&src_val[info->embedx_size]);
      if (dest[x + slot_num] != 0) {  // expand
        dest_ptr = dest[x + slot_num] + y * (expand_dim + cvm_offset);
        if (embedx_id < embedx_size) {
          op.copy(&dest_ptr[col], &src_val[info->embedx], embedx_id, scale);
        } else {
          dest_ptr[col] = 0;
        }
      } else if (dest[x] != 0 && embedx_id < embedx_dim) {  // embedx
        dest_ptr = dest[x] + y * (embedx_dim + cvm_offset);
        if (embedx_id < embedx_size) {
          op.copy(&dest_ptr[col], &src_val[info->embedx], embedx_id, scale);
        } else {
          dest_ptr[col] = 0;
        }
      }
    }
  }  // end kernel loop
}
//==========================  end ==================================
__global__ void CopyKeysKernel(const int total_len, uint64_t** src_keys,
                               uint64_t* dest_total_keys,
                               const int64_t* slot_lens, const int slot_num,
                               int* key2slots) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < slot_lens[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    key2slots[i] = low;
    int y = i - slot_lens[low];
    dest_total_keys[i] = src_keys[low][y];
  }
}
__global__ void PushCopy(const boxps::FeaturePushOffset* info,
                         const int push_float_num, const int64_t total_len,
                         float* dest, float** src, const int hidden, int bs,
                         const int* slot_vector, const int* total_dims,
                         const int64_t* slot_lens, const int slot_num,
                         const int* key2slot, const int cvm_offset,
                         const int skip_offset) {
  // only process no expand
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    float* dest_val = &dest[push_float_num * i];
    set_byfloat<int>(&dest_val[info->slot], slot_vector[x]);

    float* optr = reinterpret_cast<float*>(&dest_val[info->show]);
    float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = 1.0;
    }
    for (int k = 0; k < cvm_offset; ++k) {
      optr[k + skip_offset] = src_val[k];  // support variable length
    }
    for (int k = 0; k < info->embed_num; ++k) {
      dest_val[info->embed_g + k] *= -1. * bs;
    }
    if (total_dims[i] & 0x01) {
      for (int j = 0; j < hidden - cvm_offset; ++j) {
        dest_val[info->embedx_g + j] = src_val[cvm_offset + j] * -1. * bs;
      }
    } else {
      for (int j = 0; j < hidden - cvm_offset; ++j) {
        dest_val[info->embedx_g + j] = 0;
      }
    }
  }
}
//================================== base push ================================
__global__ void PushCopyEx(const boxps::FeaturePushOffset* info,
                           const int push_float_num, const int64_t total_len,
                           float* dest, float** src, const int hidden,
                           const int bs, const int* slot_vector,
                           const int* total_dims, const int64_t* slot_lens,
                           const int slot_num, const int* key2slot,
                           const int cvm_offset, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / push_float_num;
    int col = i % push_float_num;
    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    if (col == 0) {  // slot
      set_byfloat<int>(&dest[i], slot_vector[x]);
    } else if (col < cvm_offset + 1 + skip_offset) {  // cvm
      int cvm_id = col - 1 - skip_offset;
      if (col < skip_offset + 1) {
        dest[i] = 1.0;
      } else if (col < info->embed_g) {
        dest[i] = *(src[x] + y * hidden + cvm_id);  // support variable length
      } else {                                      // embed_g
        dest[i] = *(src[x] + y * hidden + cvm_id) * -1. * bs;
      }
    } else {
      // embedx
      int col_id = col - 1 - skip_offset;
      if ((total_dims[idx] & 0x01)) {
        dest[i] = *(src[x] + y * hidden + col_id) * -1. * bs;
      } else {
        dest[i] = 0;
      }
    }
  }
}
__global__ void PushMergeCopy(const boxps::FeaturePushOffset* info,
                              const int push_float_num, const int64_t total_len,
                              float* dest, float** src, const int hidden,
                              const int bs, const int* slot_vector,
                              const int* total_dims, const int64_t* slot_lens,
                              const int slot_num, const int* key2slot,
                              const int cvm_offset, const uint32_t* d_sort_idx,
                              const uint32_t* d_sort_offset,
                              const uint32_t* d_sort_cnt,
                              const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / push_float_num;
    int col = i % push_float_num;

    const uint32_t& start = d_sort_offset[idx];
    const uint32_t& count = d_sort_cnt[idx];
    const uint32_t& pos = d_sort_idx[start];

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];

    if (col == 0) {  // slot
      set_byfloat<int>(&dest[i], slot_vector[x]);
    } else if (col < cvm_offset + 1 + skip_offset) {  // cvm
      if (col < skip_offset + 1) {
        dest[i] = count;
      } else {
        double val = 0;
        int cvm_id = col - 1 - skip_offset;
        // merge same key in diffent slot id
        for (uint32_t j = 0; j < count; ++j) {
          const uint32_t& pos = d_sort_idx[start + j];
          const int& x = key2slot[pos];
          y = pos - slot_lens[x];
          val += *(src[x] + y * hidden + cvm_id);
        }
        if (col < info->embed_g) {  // cvm
          dest[i] = val;
        } else {  // embed_g
          dest[i] = val * -1. * bs;
        }
      }
    } else {
      // embedx
      double val = 0;
      int col_id = col - 1 - skip_offset;
      // merge same key in diffent slot id
      for (uint32_t j = 0; j < count; ++j) {
        const uint32_t& pos = d_sort_idx[start + j];
        const int& x = key2slot[pos];
        if ((total_dims[pos] & 0x01)) {
          y = pos - slot_lens[x];
          val += *(src[x] + y * hidden + col_id);
        }
      }
      dest[i] = val * -1. * bs;
    }
  }
}
__global__ void PushMergeCopyAtomic(
    const boxps::FeaturePushOffset* info, const int push_float_num,
    const int64_t total_len, float* dest, float** src, const int hidden,
    const int bs, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const uint32_t* d_restore_idx,
    const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / push_float_num;
    int col = i % push_float_num;

    const int& x = key2slot[idx];
    int y = idx - slot_lens[x];

    float* dest_ptr = &dest[d_restore_idx[idx] * push_float_num];
    if (col == 0) {  // slot
      set_byfloat<int>(&dest_ptr[col], slot_vector[x]);
    } else if (col < cvm_offset + 1 + skip_offset) {  // cvm
      if (col < skip_offset + 1) {
        paddle::platform::CudaAtomicAdd(&dest_ptr[col], 1.0);
      } else {
        float& val = *(src[x] + y * hidden + col - 1 - skip_offset);
        if (col < info->embed_g) {  // cvm
          paddle::platform::CudaAtomicAdd(&dest_ptr[col], val);
        } else {  // embed_g
          paddle::platform::CudaAtomicAdd(&dest_ptr[col], val * -1. * bs);
        }
      }
    } else {
      // embedx
      if ((total_dims[idx] & 0x01)) {
        float& val = *(src[x] + y * hidden + col - 1 - skip_offset);
        paddle::platform::CudaAtomicAdd(&dest_ptr[col], val * -1. * bs);
      }
    }
  }
}
//============================== expand nncross ===============================
template <typename TExpandPushGetOp>
__global__ void PushCopyNNCross(
    const TExpandPushGetOp& op, const boxps::FeaturePushOffset* info,
    const int push_float_num, const int64_t total_len, float* dest, float** src,
    const int hidden, const int expand_dim, const int bs,
    const int* slot_vector, const int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int cvm_offset,
    const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / push_float_num;
    int col = i % push_float_num;
    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    if (col == 0) {  // slot
      set_byfloat<int>(&dest[i], slot_vector[x]);
    } else if (col < cvm_offset + 1 + skip_offset) {  // cvm
      if (col < skip_offset + 1) {                    // skip
        dest[i] = 1.0;
      } else {
        if (src[x] != 0) {
          int cvm_id = col - 1 - skip_offset;
          if (col < info->embed_g) {  // cvm
            dest[i] =
                *(src[x] + y * hidden + cvm_id);  // support variable length
          } else {                                // embed_g
            dest[i] = *(src[x] + y * hidden + cvm_id) * -1. * bs;
          }
        } else {
          if (col == info->show) {  // show
            dest[i] = 1;
          } else {  // other
            dest[i] = 0;
          }
        }
      }
    } else {
      int col_id = col - 1 - skip_offset;
      if (col_id < hidden) {  // embedx
        if ((total_dims[idx] & 0x01) && src[x] != 0) {
          dest[i] = *(src[x] + y * hidden + col_id) * -1. * bs;
        } else {
          dest[i] = 0;
        }
      } else {  // expand
        int expand_id = col_id - hidden;
        if ((total_dims[idx] & 0x02) && src[x + slot_num] != 0) {
          dest[i] =
              op.get(src[x + slot_num], y, expand_id, hidden, expand_dim) *
              -1. * bs;
        } else {
          dest[i] = 0;
        }
      }
    }
  }
}
template <typename TExpandPushGetOp>
__global__ void PushMergeCopyNNCross(
    const TExpandPushGetOp& op, const boxps::FeaturePushOffset* info,
    const int push_float_num, const int64_t total_len, float* dest, float** src,
    const int hidden, const int expand_dim, const int bs,
    const int* slot_vector, const int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int cvm_offset,
    const uint32_t* d_sort_idx, const uint32_t* d_sort_offset,
    const uint32_t* d_sort_cnt, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / push_float_num;
    int col = i % push_float_num;

    const uint32_t& start = d_sort_offset[idx];
    const uint32_t& count = d_sort_cnt[idx];
    const uint32_t& pos = d_sort_idx[start];

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];

    if (col == 0) {  // slot
      set_byfloat<int>(&dest[i], slot_vector[x]);
    } else if (col < cvm_offset + 1 + skip_offset) {  // cvm
      if (col < skip_offset + 1) {                    // skip
        dest[i] = count;
      } else {
        if (src[x] != 0) {
          int cvm_id = col - 1 - skip_offset;
          double val = 0.0;
          // merge same key in diffent slot id
          for (uint32_t j = 0; j < count; ++j) {
            const uint32_t& pos = d_sort_idx[start + j];
            const int& x = key2slot[pos];
            if (src[x] == 0) {
              continue;
            }
            y = pos - slot_lens[x];
            val += *(src[x] + y * hidden + cvm_id);
          }
          if (col < info->embed_g) {  // cvm
            dest[i] = val;            // support variable length
          } else {                    // embed_g
            dest[i] = val * -1. * bs;
          }
        } else {
          if (col == info->show) {  // show
            dest[i] = count;
          } else {  // other
            dest[i] = 0;
          }
        }
      }
    } else {
      int col_id = col - 1 - skip_offset;
      if (col_id < hidden) {  // embedx
        if ((total_dims[idx] & 0x01) && src[x] != 0) {
          double val = 0.0;
          for (uint32_t j = 0; j < count; ++j) {
            const uint32_t& pos = d_sort_idx[start + j];
            const int& x = key2slot[pos];
            if ((total_dims[pos] & 0x01) && src[x] != 0) {
              y = pos - slot_lens[x];
              val += *(src[x] + y * hidden + col_id);
            }
          }
          dest[i] = val * -1. * bs;
        } else {
          dest[i] = 0;
        }
      } else {  // expand
        int expand_id = col_id - hidden;
        if ((total_dims[idx] & 0x02) && src[x + slot_num] != 0) {
          double val = 0.0;
          for (uint32_t j = 0; j < count; ++j) {
            const uint32_t& pos = d_sort_idx[start + j];
            const int& x = key2slot[pos];
            if ((total_dims[pos] & 0x02) && src[x + slot_num] != 0) {
              y = pos - slot_lens[x];
              val +=
                  op.get(src[x + slot_num], y, expand_id, hidden, expand_dim);
            }
          }
          dest[i] = val * -1. * bs;
        } else {
          dest[i] = 0;
        }
      }
    }
  }
}
template <typename TExpandPushGetOp>
__global__ void PushMergeCopyNNCrossAtomic(
    const TExpandPushGetOp& op, const boxps::FeaturePushOffset* info,
    const int push_float_num, const int64_t total_len, float* dest, float** src,
    const int hidden, const int expand_dim, const int bs,
    const int* slot_vector, const int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int cvm_offset,
    const uint32_t* d_restore_idx, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / push_float_num;
    int col = i % push_float_num;

    const int& x = key2slot[idx];
    int y = idx - slot_lens[x];

    float* dest_ptr = &dest[d_restore_idx[idx] * push_float_num];
    if (col == 0) {  // slot
      set_byfloat<int>(&dest_ptr[col], slot_vector[x]);
    } else if (col < cvm_offset + 1 + skip_offset) {  // cvm
      if (col < skip_offset + 1) {
        paddle::platform::CudaAtomicAdd(&dest_ptr[col], 1.0);
      } else {
        if (src[x] != 0) {
          float& val = *(src[x] + y * hidden + col - 1 - skip_offset);
          if (col < info->embed_g) {  // cvm
            paddle::platform::CudaAtomicAdd(&dest_ptr[col], val);
          } else {  // embed_g
            paddle::platform::CudaAtomicAdd(&dest_ptr[col], val * -1. * bs);
          }
        } else if (col == info->show) {
          paddle::platform::CudaAtomicAdd(&dest_ptr[col], 1.0);
        }
      }
    } else {
      // embedx
      int col_id = col - 1 - skip_offset;
      if (col_id < hidden) {  // embedx
        if ((total_dims[idx] & 0x01) && src[x] != 0) {
          float& val = *(src[x] + y * hidden + col_id);
          paddle::platform::CudaAtomicAdd(&dest_ptr[col], val * -1. * bs);
        }
      } else {  // expand
        int expand_id = col_id - hidden;
        if ((total_dims[idx] & 0x02) && src[x + slot_num] != 0) {
          float val =
              op.get(src[x + slot_num], y, expand_id, hidden, expand_dim);
          paddle::platform::CudaAtomicAdd(&dest_ptr[col], val * -1. * bs);
        }
      }
    }
  }
}
//========================== feature variable push ============================
__global__ void PushCopyVariable(
    const boxps::FeaturePushOffset* info, const int push_float_num,
    const int64_t total_len, float* dest, float** src, const int hidden,
    const int expand_hidden, const int bs, const int* slot_vector,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / push_float_num;
    int col = i % push_float_num;
    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    if (col == 0) {  // slot
      set_byfloat<int>(&dest[i], slot_vector[x]);
    } else if (col < cvm_offset + 1 + skip_offset) {  // cvm
      if (col < skip_offset + 1) {                    // skip
        dest[i] = 1.0;
      } else {
        float* src_val = 0;
        if (src[x + slot_num] != 0) {
          src_val =
              reinterpret_cast<float*>(src[x + slot_num] + y * expand_hidden);
        } else {
          src_val = reinterpret_cast<float*>(src[x] + y * hidden);
        }
        int cvm_id = col - 1 - skip_offset;
        if (col < info->embed_g) {    // cvm
          dest[i] = src_val[cvm_id];  // support variable length
        } else {                      // embed_g
          dest[i] = src_val[cvm_id] * -1. * bs;
        }
      }
    } else {
      int col_id = col - 1 - skip_offset;
      if ((total_dims[idx] & 0x01) && src[x] != 0 &&
          col_id < hidden) {  // embedx
        dest[i] = *(src[x] + y * hidden + col_id) * -1. * bs;
      } else if ((total_dims[idx] & 0x02) &&
                 src[x + slot_num] != 0) {  // expand
        dest[i] = *(src[x] + y * expand_hidden + col_id) * -1. * bs;
      } else {
        dest[i] = 0;
      }
    }
  }
}
__global__ void PushMergeCopyVariable(
    const boxps::FeaturePushOffset* info, const int push_float_num,
    const int64_t total_len, float* dest, float** src, const int hidden,
    const int expand_hidden, const int bs, const int* slot_vector,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_cnt,
    const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / push_float_num;
    int col = i % push_float_num;

    const uint32_t& start = d_sort_offset[idx];
    const uint32_t& count = d_sort_cnt[idx];
    const uint32_t& pos = d_sort_idx[start];

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];

    if (col == 0) {
      set_byfloat<int>(&dest[i], slot_vector[x]);
    } else if (col < cvm_offset + 1 + skip_offset) {  // cvm
      if (col < skip_offset + 1) {                    // skip
        dest[i] = count;
      } else {
        int cvm_id = col - 1 - skip_offset;
        double val = 0.0;
        float* src_val = nullptr;
        // merge same key in diffent slot id
        for (uint32_t j = 0; j < count; ++j) {
          const uint32_t& pos = d_sort_idx[start + j];
          const int& x = key2slot[pos];
          y = pos - slot_lens[x];
          if (src[x + slot_num] != 0) {
            src_val =
                reinterpret_cast<float*>(src[x + slot_num] + y * expand_hidden);
          } else {
            src_val = reinterpret_cast<float*>(src[x] + y * hidden);
          }
          assert(src_val != 0);
          val += src_val[cvm_id];
        }
        if (col < info->embed_g) {  // cvm
          dest[i] = val;            // support variable length
        } else {                    // embed_g
          dest[i] = val * -1. * bs;
        }
      }
    } else {
      double val = 0.0;
      int col_id = col - 1 - skip_offset;
      // merge same key in diffent slot id
      for (uint32_t j = 0; j < count; ++j) {
        const uint32_t& pos = d_sort_idx[start + j];
        const int& x = key2slot[pos];
        y = pos - slot_lens[x];
        if ((total_dims[pos] & 0x02) && src[x + slot_num] != 0) {  // expand
          val += *(src[x + slot_num] + y * expand_hidden + col_id);
        } else if ((total_dims[pos] & 0x01) && src[x] != 0 &&
                   col_id < hidden) {  // embedx
          val += *(src[x] + y * hidden + col_id);
        }
      }
      dest[i] = val * -1. * bs;
    }
  }
}
__global__ void PushMergeCopyVariableAtomic(
    const boxps::FeaturePushOffset* info, const int push_float_num,
    const int64_t total_len, float* dest, float** src, const int hidden,
    const int expand_hidden, const int bs, const int* slot_vector,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const uint32_t* d_restore_idx,
    const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / push_float_num;
    int col = i % push_float_num;
    const int& x = key2slot[idx];
    int y = idx - slot_lens[x];

    float* dest_ptr = &dest[d_restore_idx[idx] * push_float_num];
    if (col == 0) {  // slot
      set_byfloat<int>(&dest_ptr[col], slot_vector[x]);
    } else if (col < cvm_offset + 1 + skip_offset) {  // cvm
      if (col < skip_offset + 1) {
        paddle::platform::CudaAtomicAdd(&dest_ptr[col], 1.0);
      } else {
        int cvm_id = col - 1 - skip_offset;
        float* src_val = nullptr;
        if (src[x + slot_num] != 0) {
          src_val =
              reinterpret_cast<float*>(src[x + slot_num] + y * expand_hidden);
        } else {
          src_val = reinterpret_cast<float*>(src[x] + y * hidden);
        }
        if (col < info->embed_g) {  // cvm
          paddle::platform::CudaAtomicAdd(&dest_ptr[col], src_val[cvm_id]);
        } else {  // embed_g
          paddle::platform::CudaAtomicAdd(&dest_ptr[col],
                                          src_val[cvm_id] * -1. * bs);
        }
      }
    } else {
      // embedx
      int col_id = col - 1 - skip_offset;
      if ((total_dims[idx] & 0x02) && src[x + slot_num] != 0) {  // expand
        float& val = *(src[x + slot_num] + y * expand_hidden + col_id);
        paddle::platform::CudaAtomicAdd(&dest_ptr[col], val * -1. * bs);
      } else if ((total_dims[idx] & 0x01) && src[x] != 0 &&
                 col_id < hidden) {  // embedx
        float& val = *(src[x] + y * hidden + col_id);
        paddle::platform::CudaAtomicAdd(&dest_ptr[col], val * -1. * bs);
      }
    }
  }
}
template <typename TEmbedxOp>
inline void FeaturePullCopy(
    const TEmbedxOp& op, const boxps::FeaturePullOffset* info,
    const size_t& pull_float_num, cudaStream_t stream, uint64_t** gpu_keys,
    float** gpu_values, void* src, const int hidden_size,
    const size_t embedx_dim, const size_t total_length, int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const float scale, const int cvm_offset, const uint32_t* gpu_restore_idx,
    const int skip_offset) {
  const float* pull_val = reinterpret_cast<const float*>(src);
  if (total_length > max_gpu_thread_num) {
    PullCopy<<<CUDA_BLOCK(total_length), stream>>>(
        op, info, pull_float_num, total_length, gpu_values, pull_val,
        hidden_size, total_dims, slot_lens, slot_num, key2slot, scale,
        cvm_offset, skip_offset, gpu_restore_idx);
  } else {
    const size_t N = total_length * hidden_size;
    PullCopyEx<<<CUDA_BLOCK(N), stream>>>(
        op, info, pull_float_num, N, gpu_values, pull_val, hidden_size,
        total_dims, slot_lens, slot_num, key2slot, scale, cvm_offset,
        skip_offset, gpu_restore_idx);
  }
}

template <typename TEmbedxOp>
inline void FeaturePullCopyNNCross(
    const TEmbedxOp& op, const boxps::FeaturePullOffset* info,
    const size_t& pull_float_num, cudaStream_t stream, uint64_t** gpu_keys,
    float** gpu_values, void* src, const int hidden_size,
    const size_t embedx_dim, const size_t expand_dim, const size_t total_length,
    int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const float scale, const int cvm_offset,
    const uint32_t* gpu_restore_idx, const int skip_offset, bool expand_only) {
  const float* pull_val = reinterpret_cast<const float*>(src);
  const size_t N = total_length * (hidden_size + expand_dim);
  if (expand_only) {
    PullCopyNNCross<<<CUDA_BLOCK(N), stream>>>(
        op, info, pull_float_num, N, gpu_values, pull_val,
        (hidden_size + expand_dim), hidden_size, expand_dim, total_dims,
        slot_lens, slot_num, key2slot, scale, cvm_offset, skip_offset,
        gpu_restore_idx);
  } else {
    PullCopyNNCrossWithEmb<<<CUDA_BLOCK(N), stream>>>(
        op, info, pull_float_num, N, gpu_values, pull_val,
        (hidden_size + expand_dim), hidden_size, expand_dim, total_dims,
        slot_lens, slot_num, key2slot, scale, cvm_offset, skip_offset,
        gpu_restore_idx);
  }
}

template <typename TEmbedxOp>
inline void FeaturePullCopyVariable(
    const TEmbedxOp& op, const boxps::FeaturePullOffset* info,
    const size_t& pull_float_num, cudaStream_t stream, uint64_t** gpu_keys,
    float** gpu_values, void* src, const int hidden_size,
    const size_t embedx_dim, const size_t expand_dim, const int total_length,
    int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const float scale, const int cvm_offset,
    const uint32_t* gpu_restore_idx, const int skip_offset) {
  const float* pull_val = reinterpret_cast<const float*>(src);
  int max_cols_num =
      std::max(hidden_size, static_cast<int>(expand_dim + cvm_offset));
  const size_t N = total_length * max_cols_num;
  PullCopyVariable<<<CUDA_BLOCK(N), stream>>>(
      op, info, pull_float_num, N, gpu_values, pull_val, max_cols_num,
      embedx_dim, expand_dim, total_dims, slot_lens, slot_num, key2slot, scale,
      cvm_offset, skip_offset, gpu_restore_idx);
}

void BoxWrapper::CopyForPull(
    const paddle::platform::Place& place, uint64_t** gpu_keys,
    float** gpu_values, void* total_values_gpu,
    boxps::FeaturePullOffset* pull_offset, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int hidden_size,
    const int expand_embed_dim, const int64_t total_length, int* total_dims,
    const int skip_offset, bool expand_only, const uint32_t* gpu_restore_idx) {
  auto stream = dynamic_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();
  const int cvm_offset = cvm_offset_ - skip_offset;
  if (pull_info_.is_quant) {
    EmbedxQuantOp op;
    if (expand_embed_dim > 0 && pull_info_.expand_size > 0) {  // nncross
      FeaturePullCopyNNCross(op, pull_offset, pull_float_num_, stream, gpu_keys,
                             gpu_values, total_values_gpu, hidden_size,
                             embedx_dim_, expand_embed_dim_, total_length,
                             total_dims, slot_lens, slot_num, key2slot,
                             pull_embedx_scale_, cvm_offset, gpu_restore_idx,
                             skip_offset, expand_only);
    } else if (pull_info_.expand_size < 0 &&
               expand_embed_dim == cvm_offset + expand_embed_dim_ &&
               hidden_size == cvm_offset + embedx_dim_) {  // var
      FeaturePullCopyVariable(
          op, pull_offset, pull_float_num_, stream, gpu_keys, gpu_values,
          total_values_gpu, hidden_size, embedx_dim_, expand_embed_dim_,
          total_length, total_dims, slot_lens, slot_num, key2slot,
          pull_embedx_scale_, cvm_offset, gpu_restore_idx, skip_offset);
    } else {
      // normal and adam
      FeaturePullCopy(op, pull_offset, pull_float_num_, stream, gpu_keys,
                      gpu_values, total_values_gpu, hidden_size, embedx_dim_,
                      total_length, total_dims, slot_lens, slot_num, key2slot,
                      pull_embedx_scale_, cvm_offset, gpu_restore_idx,
                      skip_offset);
    }
  } else {
    EmbedxNormalOp op;
    if (expand_embed_dim > 0 && pull_info_.expand_size > 0) {  // nncross
      FeaturePullCopyNNCross(op, pull_offset, pull_float_num_, stream, gpu_keys,
                             gpu_values, total_values_gpu, hidden_size,
                             embedx_dim_, expand_embed_dim_, total_length,
                             total_dims, slot_lens, slot_num, key2slot,
                             pull_embedx_scale_, cvm_offset, gpu_restore_idx,
                             skip_offset, expand_only);
    } else if (pull_info_.expand_size < 0 &&
               expand_embed_dim == cvm_offset + expand_embed_dim_ &&
               hidden_size == cvm_offset + embedx_dim_) {  // var
      FeaturePullCopyVariable(
          op, pull_offset, pull_float_num_, stream, gpu_keys, gpu_values,
          total_values_gpu, hidden_size, embedx_dim_, expand_embed_dim_,
          total_length, total_dims, slot_lens, slot_num, key2slot,
          pull_embedx_scale_, cvm_offset, gpu_restore_idx, skip_offset);
    } else {
      // normal and adam
      FeaturePullCopy(op, pull_offset, pull_float_num_, stream, gpu_keys,
                      gpu_values, total_values_gpu, hidden_size, embedx_dim_,
                      total_length, total_dims, slot_lens, slot_num, key2slot,
                      pull_embedx_scale_, cvm_offset, gpu_restore_idx,
                      skip_offset);
    }
  }
  cudaStreamSynchronize(stream);
}

void BoxWrapper::CopyKeys(const paddle::platform::Place& place,
                          uint64_t** origin_keys, uint64_t* total_keys,
                          const int64_t* slot_lens, int slot_num, int total_len,
                          int* key2slot) {
  auto stream = dynamic_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();
  CopyKeysKernel<<<CUDA_BLOCK(total_len), stream>>>(
      total_len, origin_keys, total_keys, slot_lens, slot_num, key2slot);
  cudaStreamSynchronize(stream);
}

inline void FeaturePushCopy(
    const boxps::FeaturePushOffset* info, const size_t& push_float_num,
    cudaStream_t stream, const int64_t& total_length,
    const int64_t& dedup_length, void* dest, float** grad_values,
    const int hidden_size, const int embedx_dim, const int batch_size,
    const int* slot_vector, const int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int cvm_offset,
    const uint32_t* d_sort_idx, const uint32_t* d_sort_offset,
    const uint32_t* d_sort_cnt, const uint32_t* d_restore_idx,
    const int skip_offset) {
  float* push_grad_values = reinterpret_cast<float*>(dest);
  if (d_sort_idx != nullptr) {
    if (total_length < dedup_length * 2) {
      const int64_t N = dedup_length * push_float_num;
      PushMergeCopy<<<CUDA_BLOCK(N), stream>>>(
          info, push_float_num, N, push_grad_values, grad_values, hidden_size,
          batch_size, slot_vector, total_dims, slot_lens, slot_num, key2slot,
          cvm_offset, d_sort_idx, d_sort_offset, d_sort_cnt, skip_offset);
    } else {
      const int64_t N = total_length * push_float_num;
      cudaMemsetAsync(push_grad_values, 0,
                      dedup_length * push_float_num * sizeof(float), stream);
      PushMergeCopyAtomic<<<CUDA_BLOCK(N), stream>>>(
          info, push_float_num, N, push_grad_values, grad_values, hidden_size,
          batch_size, slot_vector, total_dims, slot_lens, slot_num, key2slot,
          cvm_offset, d_restore_idx, skip_offset);
    }
  } else {
    if (total_length > max_gpu_thread_num) {
      PushCopy<<<CUDA_BLOCK(total_length), stream>>>(
          info, push_float_num, total_length, push_grad_values, grad_values,
          hidden_size, batch_size, slot_vector, total_dims, slot_lens, slot_num,
          key2slot, cvm_offset, skip_offset);
    } else {
      const int64_t N = total_length * push_float_num;
      PushCopyEx<<<CUDA_BLOCK(N), stream>>>(
          info, push_float_num, N, push_grad_values, grad_values, hidden_size,
          batch_size, slot_vector, total_dims, slot_lens, slot_num, key2slot,
          cvm_offset, skip_offset);
    }
  }
}
inline void FeaturePushCopyNNCross(
    const boxps::FeaturePushOffset* info, const size_t& push_float_num,
    cudaStream_t stream, const int64_t& total_length,
    const int64_t& dedup_length, void* dest, float** grad_values,
    const int hidden_size, const int embedx_dim, const int expand_dim,
    const int batch_size, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_cnt,
    const uint32_t* d_restore_idx, const int skip_offset, bool expand_only) {
  float* push_grad_values = reinterpret_cast<float*>(dest);
  if (expand_only) {
    ExpandPushGetOp op;
    if (d_sort_idx != nullptr) {
      if (total_length < dedup_length * 2) {
        const int64_t N = dedup_length * push_float_num;
        PushMergeCopyNNCross<<<CUDA_BLOCK(N), stream>>>(
            op, info, push_float_num, N, push_grad_values, grad_values,
            hidden_size, expand_dim, batch_size, slot_vector, total_dims,
            slot_lens, slot_num, key2slot, cvm_offset, d_sort_idx,
            d_sort_offset, d_sort_cnt, skip_offset);
      } else {
        const int64_t N = total_length * push_float_num;
        cudaMemsetAsync(push_grad_values, 0,
                        dedup_length * push_float_num * sizeof(float), stream);
        PushMergeCopyNNCrossAtomic<<<CUDA_BLOCK(N), stream>>>(
            op, info, push_float_num, N, push_grad_values, grad_values,
            hidden_size, expand_dim, batch_size, slot_vector, total_dims,
            slot_lens, slot_num, key2slot, cvm_offset, d_restore_idx,
            skip_offset);
      }
    } else {
      const int64_t N = total_length * push_float_num;
      PushCopyNNCross<<<CUDA_BLOCK(N), stream>>>(
          op, info, push_float_num, N, push_grad_values, grad_values,
          hidden_size, expand_dim, batch_size, slot_vector, total_dims,
          slot_lens, slot_num, key2slot, cvm_offset, skip_offset);
    }
  } else {
    ExpandPushEmdGetOp op;
    if (d_sort_idx != nullptr) {
      if (total_length < dedup_length * 2) {
        const int64_t N = dedup_length * push_float_num;
        PushMergeCopyNNCross<<<CUDA_BLOCK(N), stream>>>(
            op, info, push_float_num, N, push_grad_values, grad_values,
            hidden_size, expand_dim, batch_size, slot_vector, total_dims,
            slot_lens, slot_num, key2slot, cvm_offset, d_sort_idx,
            d_sort_offset, d_sort_cnt, skip_offset);
      } else {
        const int64_t N = total_length * push_float_num;
        cudaMemsetAsync(push_grad_values, 0,
                        dedup_length * push_float_num * sizeof(float), stream);
        PushMergeCopyNNCrossAtomic<<<CUDA_BLOCK(N), stream>>>(
            op, info, push_float_num, N, push_grad_values, grad_values,
            hidden_size, expand_dim, batch_size, slot_vector, total_dims,
            slot_lens, slot_num, key2slot, cvm_offset, d_restore_idx,
            skip_offset);
      }
    } else {
      const int64_t N = total_length * push_float_num;
      PushCopyNNCross<<<CUDA_BLOCK(N), stream>>>(
          op, info, push_float_num, N, push_grad_values, grad_values,
          hidden_size, expand_dim, batch_size, slot_vector, total_dims,
          slot_lens, slot_num, key2slot, cvm_offset, skip_offset);
    }
  }
}
inline void FeaturePushCopyVariable(
    const boxps::FeaturePushOffset* info, const size_t& push_float_num,
    cudaStream_t stream, const int64_t& total_length,
    const int64_t& dedup_length, void* dest, float** grad_values,
    const int hidden_size, const int embedx_dim, const int expand_dim,
    const int batch_size, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_cnt,
    const uint32_t* d_restore_idx, const int skip_offset) {
  float* push_grad_values = reinterpret_cast<float*>(dest);
  if (d_sort_idx != nullptr) {
    if (total_length < dedup_length * 2) {
      const int64_t N = dedup_length * push_float_num;
      PushMergeCopyVariable<<<CUDA_BLOCK(N), stream>>>(
          info, push_float_num, N, push_grad_values, grad_values, hidden_size,
          expand_dim + cvm_offset, batch_size, slot_vector, total_dims,
          slot_lens, slot_num, key2slot, cvm_offset, d_sort_idx, d_sort_offset,
          d_sort_cnt, skip_offset);
    } else {
      const int64_t N = total_length * push_float_num;
      cudaMemsetAsync(push_grad_values, 0,
                      dedup_length * push_float_num * sizeof(float), stream);
      PushMergeCopyVariableAtomic<<<CUDA_BLOCK(N), stream>>>(
          info, push_float_num, N, push_grad_values, grad_values, hidden_size,
          expand_dim + cvm_offset, batch_size, slot_vector, total_dims,
          slot_lens, slot_num, key2slot, cvm_offset, d_restore_idx,
          skip_offset);
    }
  } else {
    const int64_t N = total_length * push_float_num;
    PushCopyVariable<<<CUDA_BLOCK(N), stream>>>(
        info, push_float_num, N, push_grad_values, grad_values, hidden_size,
        expand_dim + cvm_offset, batch_size, slot_vector, total_dims, slot_lens,
        slot_num, key2slot, cvm_offset, skip_offset);
  }
}
void BoxWrapper::CopyForPush(
    const paddle::platform::Place& place, float** grad_values,
    void* total_grad_values_gpu, boxps::FeaturePushOffset* push_offset,
    const int64_t total_length, const int64_t dedup_length,
    const int* d_slot_vector, const int64_t* slot_lens, const int slot_num,
    const int hidden_size, const int expand_embed_dim, const int batch_size,
    const int* total_dims, const int* key2slot, const int skip_offset,
    bool expand_only, const uint32_t* gpu_sort_idx,
    const uint32_t* gpu_sort_offset, const uint32_t* gpu_sort_lens,
    const uint32_t* gpu_restore_idx) {
  auto stream = dynamic_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();
  const int cvm_offset = cvm_offset_ - skip_offset;
  if (expand_embed_dim > 0 && pull_info_.expand_size > 0) {  // nncross
    FeaturePushCopyNNCross(
        push_offset, push_float_num_, stream, total_length, dedup_length,
        total_grad_values_gpu, grad_values, hidden_size, embedx_dim_,
        expand_embed_dim_, batch_size, d_slot_vector, total_dims, slot_lens,
        slot_num, key2slot, cvm_offset, gpu_sort_idx, gpu_sort_offset,
        gpu_sort_lens, gpu_restore_idx, skip_offset, expand_only);

  } else if (pull_info_.expand_size < 0 &&
             expand_embed_dim == cvm_offset + expand_embed_dim_ &&
             hidden_size == cvm_offset + embedx_dim_) {  // var
    FeaturePushCopyVariable(
        push_offset, push_float_num_, stream, total_length, dedup_length,
        total_grad_values_gpu, grad_values, hidden_size, embedx_dim_,
        expand_embed_dim_, batch_size, d_slot_vector, total_dims, slot_lens,
        slot_num, key2slot, cvm_offset, gpu_sort_idx, gpu_sort_offset,
        gpu_sort_lens, gpu_restore_idx, skip_offset);
  } else {
    FeaturePushCopy(push_offset, push_float_num_, stream, total_length,
                    dedup_length, total_grad_values_gpu, grad_values,
                    hidden_size, embedx_dim_, batch_size, d_slot_vector,
                    total_dims, slot_lens, slot_num, key2slot, cvm_offset,
                    gpu_sort_idx, gpu_sort_offset, gpu_sort_lens,
                    gpu_restore_idx, skip_offset);
  }
  cudaStreamSynchronize(stream);
}

__global__ void pull_cache_value_kernel(int len, int dim, uint64_t* key,
                                        float* val, float* table) {
  CUDA_KERNEL_LOOP(i, len) { val[i] = table[key[i / dim] * dim + i % dim]; }
}

void GpuReplicaCache::PullCacheValue(uint64_t* d_keys, float* d_vals, int num,
                                     int gpu_id) {
  auto place = platform::CUDAPlace(gpu_id);
  auto stream = dynamic_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();
  int len = emb_dim_ * num;
  pull_cache_value_kernel<<<CUDA_BLOCK(len), stream>>>(len, emb_dim_, d_keys,
                                                       d_vals, d_embs_[gpu_id]);
}

}  // end namespace framework
}  // end namespace paddle
#endif
