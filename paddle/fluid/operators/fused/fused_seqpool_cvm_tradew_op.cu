//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/fluid/operators/fused/fused_seqpool_cvm_tradew_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

#define GET_BLOCK(N) \
  ((N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS)

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// normal
template <typename T>
__global__ void FusedSeqpoolTradeWKernelNormal(
    const size_t N, T **input_values, T **seqpool_output_values,
    size_t **lods_values, const int batch_size, const int hidden_size,
    const int embedding_size, const float pad_value, const int cvm_offset,
    const int trade_num) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    double val = pad_value;
    if (offset < cvm_offset) {
      for (auto k = start; k < end; ++k) {
        val += *(input_values[x] + k * hidden_size + offset);
      }
    } else {
      for (auto k = start; k < end; ++k) {
        val += *(input_values[x] + k * hidden_size + trade_num + offset);
      }
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}

// normal
template <typename T>
__global__ void FusedSeqpoolTradeWKernelWithTradeId(
    const size_t N, T **input_values, T **seqpool_output_values,
    size_t **lods_values, const int batch_size, const int hidden_size,
    const int embedding_size, const float pad_value, const int cvm_offset,
    const int trade_id, const int trade_num) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    double val = pad_value;
    if (offset < cvm_offset) {
      for (auto k = start; k < end; ++k) {
        val += *(input_values[x] + k * hidden_size + offset);
      }
    } else {
      for (auto k = start; k < end; ++k) {
        val += (*(input_values[x] + k * hidden_size + trade_num + offset)) *
               (*(input_values[x] + k * hidden_size + cvm_offset + trade_id));
      }
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}

// join need show click input
template <typename T>
__global__ void FusedCVMTradeWKernelWithCVM(const size_t N, T **output_values,
                                            T **seqpool_output_values,
                                            const int batch_size,
                                            const int embedding_size,
                                            const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    if (offset == 0) {         // show
      *(output_values[x] + y * embedding_size) =
          log(*(seqpool_output_values[x] + y * embedding_size) + 1);
    } else if (offset == 1) {  // click
      *(output_values[x] + y * embedding_size + offset) =
          log(*(seqpool_output_values[x] + y * embedding_size + 1) + 1) -
          log(*(seqpool_output_values[x] + y * embedding_size) + 1);
    } else {
      *(output_values[x] + y * embedding_size + offset) =
          *(seqpool_output_values[x] + y * embedding_size + offset);
    }
  }
}
// update not need show click input
template <typename T>
__global__ void FusedCVMTradeWKernelNoCVM(const size_t N, T **output_values,
                                          T **seqpool_output_values,
                                          const int batch_size,
                                          const int no_cvm_embedding_size,
                                          const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / no_cvm_embedding_size;
    int offset = i % no_cvm_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    // no cvm
    *(output_values[x] + y * no_cvm_embedding_size + offset) =
        *(seqpool_output_values[x] + y * (no_cvm_embedding_size + cvm_offset) +
          offset + cvm_offset);
  }
}

template <typename T>
inline void FusedSeqpoolCVMTradeW(const paddle::platform::Place &place,
                                  const std::vector<const T *> &input_data,
                                  const std::vector<T *> &output_data,
                                  const std::vector<T *> &seqpool_output_data,
                                  std::vector<const size_t *> lods,
                                  const int batch_size, const int slot_num,
                                  const int embedding_size,
                                  const float padding_value, const bool use_cvm,
                                  const int cvm_offset, const int trade_id,
                                  const int trade_num) {
  auto stream = dynamic_cast<phi::GPUContext*>(
                 platform::DeviceContextPool::Instance().Get(place))
                 ->stream();
    
  size_t total_ptr_len = input_data.size() + output_data.size() +
                         seqpool_output_data.size() + lods.size();
  auto temp_ptr = memory::AllocShared(place, total_ptr_len * sizeof(void *));
  void *ptr = temp_ptr->ptr();

  T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
  cudaMemcpyAsync(gpu_input_values, input_data.data(),
                  input_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);
  T **gpu_output_values =
      reinterpret_cast<T **>(&gpu_input_values[input_data.size()]);
  cudaMemcpyAsync(gpu_output_values, output_data.data(),
                  output_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);
  T **gpu_seqpool_output_values =
      reinterpret_cast<T **>(&gpu_output_values[output_data.size()]);
  cudaMemcpyAsync(gpu_seqpool_output_values, seqpool_output_data.data(),
                  seqpool_output_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice, stream);
  size_t **lods_values = reinterpret_cast<size_t **>(
      &gpu_seqpool_output_values[seqpool_output_data.size()]);
  cudaMemcpyAsync(lods_values, lods.data(), lods.size() * sizeof(size_t *),
                  cudaMemcpyHostToDevice, stream);

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  //
  if (trade_id >= 0) {
    FusedSeqpoolTradeWKernelWithTradeId<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                          0, stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size + trade_num, embedding_size, padding_value, cvm_offset,
        trade_id, trade_num);
  } else {
    FusedSeqpoolTradeWKernelNormal<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                     stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size + trade_num, embedding_size, padding_value, cvm_offset,
        trade_num);
  }
  // second log
  if (use_cvm) {
    FusedCVMTradeWKernelWithCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                  stream>>>(
        N, gpu_output_values, gpu_seqpool_output_values, batch_size,
        embedding_size, cvm_offset);
  } else {
    // not need show click input
    N = static_cast<size_t>(batch_size * slot_num *
                            (embedding_size - cvm_offset));
    FusedCVMTradeWKernelNoCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                stream>>>(
        N, gpu_output_values, gpu_seqpool_output_values, batch_size,
        (embedding_size - cvm_offset), cvm_offset);
  }
}

template <typename T>
class FusedSeqpoolCVMTradeWCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<LoDTensor>("X");
    auto outputs = ctx.MultiOutput<Tensor>("Out");

    const auto slot_size = inputs.size();
    std::vector<const float *> input_data(slot_size);
    std::vector<const size_t *> lods_data(slot_size);
    std::vector<T *> output_data(slot_size);

    std::vector<LoDTensor> seqpool_outputs(slot_size);
    std::vector<T *> seqpool_output_data(slot_size);

    auto padding_value = ctx.Attr<float>("pad_value");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    const int trade_id = ctx.Attr<int>("trade_id");
    const int trade_num = ctx.Attr<int>("trade_num");

    PADDLE_ENFORCE_GE(inputs[0]->dims()[0], 0, "batch ins zero");
    int embedding_size = inputs[0]->numel() / inputs[0]->dims()[0] - trade_num;
    PADDLE_ENFORCE_GE(embedding_size, 0, "embedx size is less trade num");
    
    framework::GPULodVector gpu_lods[slot_size];
    auto place = ctx.GetPlace();
    
    int batch_size = -1;
    for (size_t i = 0; i < slot_size; ++i) {
      const auto *input = inputs[i];
      CHECK(input->lod().size() == 1);
      auto lod_data = input->lod()[0];
      int cur_batch = lod_data.size() - 1;
      if (batch_size == -1) {
        batch_size = cur_batch;
      } else {
        CHECK(batch_size == cur_batch) << "batch: " << batch_size
                                       << ", current: " << cur_batch;
      }
      input_data[i] = reinterpret_cast<const T *>(input->data<T>());
      auto *output = outputs[i];
      if (use_cvm) {
        output->Resize({batch_size, embedding_size});
      } else {
        output->Resize({batch_size, embedding_size - cvm_offset});
      }
      output_data[i] =
          reinterpret_cast<T *>(output->mutable_data<T>(ctx.GetPlace()));
      lods_data[i] = gpu_lods[i].mutable_data<size_t>(place, lod_data);

      seqpool_output_data[i] =
          reinterpret_cast<T *>(seqpool_outputs[i].mutable_data<T>(
              {batch_size, embedding_size}, ctx.GetPlace()));
    }
    FusedSeqpoolCVMTradeW(ctx.GetPlace(), input_data, output_data,
                          seqpool_output_data, lods_data, batch_size, slot_size,
                          embedding_size, padding_value, use_cvm, cvm_offset,
                          trade_id, trade_num);
  }
};
// join grad
template <typename T>
__global__ void FusedSeqpoolCVMTradeWGradKernelNoTradeId(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int hidden_num,
    const int embedding_size, const int cvm_offset, const int trade_num,
    const int skip_off) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / hidden_num;
    int offset = i % hidden_num;  // embedx offset
    int x = key / batch_size;     // slot id
    int y = key % batch_size;     // ins id

    T val = 0.0;
    if (offset < cvm_offset) {
      val = *(cvm_values[x] + y * cvm_offset + offset);
    } else if (offset >= cvm_offset + trade_num) {
      val = *(out_grads_values[x] + y * embedding_size + offset - skip_off);
    }
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * hidden_num + offset) = val;
    }
  }
}
// join grad
template <typename T>
__global__ void FusedSeqpoolCVMTradeWGradKernel(
    const size_t N, T **out_grads_values, T **input_values, T **in_grads_values,
    T **cvm_values, size_t **lods_values, const int batch_size,
    const int hidden_num, const int embedding_size, const int cvm_offset,
    const int trade_id, const int trade_num, const int skip_off,
    const int embedx_off) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / hidden_num;
    int offset = i % hidden_num;  // embedx offset
    int x = key / batch_size;     // slot id
    int y = key % batch_size;     // ins id

    if (offset < cvm_offset) {
      //      T &val = *(cvm_values[x] + y * cvm_offset + offset);
      auto &start = *(lods_values[x] + y);
      auto &end = *(lods_values[x] + y + 1);
      for (auto k = start; k < end; ++k) {
        // trade not need set show click grad
        *(in_grads_values[x] + k * hidden_num + offset) = 0.0;
      }
    } else if (offset < cvm_offset + trade_num) {
      auto &start = *(lods_values[x] + y);
      auto &end = *(lods_values[x] + y + 1);
      if (trade_id == offset - cvm_offset) {
        double sum_val = 0.0;
        for (auto k = start; k < end; ++k) {
          sum_val = 0.0;
          T *in_ptr = input_values[x] + k * hidden_num + cvm_offset + trade_num;
          T *g_ptr = out_grads_values[x] + y * embedding_size + embedx_off;
          for (int j = 0; j < (embedding_size - embedx_off); ++j) {
            sum_val += g_ptr[j] * in_ptr[j];
          }
          *(in_grads_values[x] + k * hidden_num + offset) = sum_val;
        }
      } else {
        for (auto k = start; k < end; ++k) {
          *(in_grads_values[x] + k * hidden_num + offset) = 0.0;
        }
      }
    } else {
      T &val = *(out_grads_values[x] + y * embedding_size + offset - skip_off);
      auto &start = *(lods_values[x] + y);
      auto &end = *(lods_values[x] + y + 1);
      for (auto k = start; k < end; ++k) {
        *(in_grads_values[x] + k * hidden_num + offset) =
            val * (*(input_values[x] + k * hidden_num + cvm_offset + trade_id));
      }
    }
  }
}
template <typename T>
inline void FusedSeqpoolCVMTradeWGrad(
    const paddle::platform::Place &place,
    const std::vector<const T *> &out_grads_data,
    const std::vector<const T *> &input_data,
    const std::vector<T *> &in_grads_data,
    const std::vector<const T *> &cvm_data,
    const std::vector<const size_t *> &lods, const int batch_size,
    const int slot_num, const int embedding_size, const bool use_cvm,
    const int cvm_offset, const int trade_id, const int trade_num) {
  auto stream = dynamic_cast<phi::GPUContext*>(
                 platform::DeviceContextPool::Instance().Get(place))
                 ->stream();
  size_t total_ptr_len = input_data.size() + out_grads_data.size() +
                         in_grads_data.size() + cvm_data.size() + lods.size();
  auto temp_ptr = memory::AllocShared(place, total_ptr_len * sizeof(void *));
  T **gpu_out_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
  cudaMemcpyAsync(gpu_out_grads_values, out_grads_data.data(),
                  out_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);
  T **gpu_in_values =
      reinterpret_cast<T **>(&gpu_out_grads_values[out_grads_data.size()]);
  cudaMemcpyAsync(gpu_in_values, input_data.data(),
                  input_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  T **gpu_in_grads_values =
      reinterpret_cast<T **>(&gpu_in_values[input_data.size()]);
  cudaMemcpyAsync(gpu_in_grads_values, in_grads_data.data(),
                  in_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  T **gpu_cvm_values =
      reinterpret_cast<T **>(&gpu_in_grads_values[in_grads_data.size()]);
  cudaMemcpyAsync(gpu_cvm_values, cvm_data.data(),
                  cvm_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  size_t **lods_values =
      reinterpret_cast<size_t **>(&gpu_cvm_values[cvm_data.size()]);
  cudaMemcpyAsync(lods_values, lods.data(), lods.size() * sizeof(size_t *),
                  cudaMemcpyHostToDevice, stream);

  int hidden_num = embedding_size + trade_num;
  size_t N = static_cast<size_t>(batch_size * slot_num * hidden_num);
  if (use_cvm) {
    // join grad
    if (trade_id >= 0) {
      FusedSeqpoolCVMTradeWGradKernel<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                        0, stream>>>(
          N, gpu_out_grads_values, gpu_in_values, gpu_in_grads_values,
          gpu_cvm_values, lods_values, batch_size, hidden_num, embedding_size,
          cvm_offset, trade_id, trade_num, trade_num, cvm_offset);
    } else {
      FusedSeqpoolCVMTradeWGradKernelNoTradeId<<<
          GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
          lods_values, batch_size, hidden_num, embedding_size, cvm_offset,
          trade_num, trade_num);
    }
  } else {
    // update grad
    if (trade_id >= 0) {
      FusedSeqpoolCVMTradeWGradKernel<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                        0, stream>>>(
          N, gpu_out_grads_values, gpu_in_values, gpu_in_grads_values,
          gpu_cvm_values, lods_values, batch_size, hidden_num,
          embedding_size - cvm_offset, cvm_offset, trade_id, trade_num,
          trade_num + cvm_offset, 0);
    } else {
      FusedSeqpoolCVMTradeWGradKernelNoTradeId<<<
          GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
          lods_values, batch_size, hidden_num, embedding_size - cvm_offset,
          cvm_offset, trade_num, trade_num + cvm_offset);
    }
  }
}

template <typename T>
class FusedSeqpoolCVMTradeWGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_grads = ctx.MultiInput<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
    auto *cvm = ctx.Input<LoDTensor>("CVM");
    auto inputs = ctx.MultiInput<LoDTensor>("X");

    std::string pooltype = ctx.Attr<std::string>("pooltype");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    const int trade_id = ctx.Attr<int>("trade_id");
    const int trade_num = ctx.Attr<int>("trade_num");

    const auto slot_size = in_grads.size();
    std::vector<const T *> out_grads_data(slot_size);
    std::vector<T *> in_grads_data(slot_size);
    std::vector<const T *> cvm_data(slot_size);
    std::vector<const size_t *> lods_data(slot_size);
    std::vector<const T *> input_data(slot_size);
    
    framework::GPULodVector gpu_lods[slot_size];
    auto place = ctx.GetPlace();

    int embedding_size =
        in_grads[0]->numel() / in_grads[0]->dims()[0] - trade_num;
    int batch_size = -1;
    for (size_t i = 0; i < slot_size; ++i) {
      auto *in_grad = in_grads[i];
      auto lod_data = in_grad->lod()[0];
      int cur_batch = lod_data.size() - 1;
      if (batch_size == -1) {
        batch_size = cur_batch;
      } else {
        CHECK(batch_size == cur_batch) << "batch: " << batch_size
                                       << ", current: " << cur_batch;
      }
      input_data[i] = reinterpret_cast<const T *>(inputs[i]->data<T>());
      auto *out_grad = out_grads[i];
      out_grads_data[i] = reinterpret_cast<const T *>(out_grad->data<T>());

      in_grads_data[i] =
          reinterpret_cast<T *>(in_grad->mutable_data<T>(place));
      lods_data[i] = gpu_lods[i].mutable_data<size_t>(place, lod_data);
      cvm_data[i] = reinterpret_cast<const T *>(cvm->data<T>());
    }
    FusedSeqpoolCVMTradeWGrad(place, out_grads_data, input_data,
                              in_grads_data, cvm_data, lods_data, batch_size,
                              slot_size, embedding_size, use_cvm, cvm_offset,
                              trade_id, trade_num);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_tradew,
                        ops::FusedSeqpoolCVMTradeWCUDAKernel<float>);

REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_tradew_grad,
                        ops::FusedSeqpoolCVMTradeWGradCUDAKernel<float>);
