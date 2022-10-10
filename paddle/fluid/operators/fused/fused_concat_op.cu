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
#include "paddle/fluid/operators/fused/fused_concat_op.h"
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
__global__ void FusedSeqpoolConcatKernel(const size_t N, T **input_values,
                                         T **output_values, const int *idxs,
                                         const int *ptr_idxs, const int *dims,
                                         const int batch_size,
                                         const int total_cols,
                                         const int x_num) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / total_cols;
    int offset = i % total_cols;  // cols id
    int x = key / batch_size;     // slot id
    int y = key % batch_size;     // rows id

    *(output_values[x] + y * total_cols + offset) =
        *(input_values[x * x_num + ptr_idxs[offset]] + y * dims[offset] +
          idxs[offset]);
  }
}
template <typename T>
class FusedSeqpoolConcatCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto outputs = ctx.MultiOutput<Tensor>("Out");

    const int x_num = 2;
    const std::string input_names[] = {"X1", "X2"};
    std::vector<std::vector<const LoDTensor *>> x_inputs(x_num);
    for (int k = 0; k < x_num; ++k) {
      x_inputs[k] = ctx.MultiInput<LoDTensor>(input_names[k]);
    }

    const int total_cols = ctx.Attr<int>("output_dim");
    const std::vector<int> idxs = ctx.Attr<std::vector<int>>("output_idx");
    CHECK(idxs.size() == static_cast<size_t>(3 * total_cols))
        << "total idxs len error: " << idxs.size()
        << ", total cols:" << total_cols;

    const int slot_size = static_cast<int>(x_inputs[0].size());
    std::vector<const T *> input_data(slot_size * x_num);
    std::vector<T *> output_data(slot_size);

    int batch_size = x_inputs[0][0]->dims()[0];
    for (int i = 0; i < slot_size; ++i) {
      for (int k = 0; k < x_num; ++k) {
        const auto *input = x_inputs[k][i];
        CHECK(batch_size == input->dims()[0])
            << "batch: " << batch_size << ", current: " << input->dims()[0];
        input_data[i * x_num + k] =
            reinterpret_cast<const T *>(input->data<T>());
      }
      auto *output = outputs[i];
      output->Resize({batch_size, total_cols});
      output_data[i] = reinterpret_cast<T *>(output->mutable_data<T>(place));
    }

    auto stream = dynamic_cast<phi::GPUContext*>(
                 platform::DeviceContextPool::Instance().Get(place))
                 ->stream();

    size_t total_ptr_len = input_data.size() + output_data.size();
    auto temp_ptr = memory::AllocShared(
        place, total_ptr_len * sizeof(void *) + sizeof(int) * idxs.size());
    T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
    cudaMemcpyAsync(gpu_input_values, input_data.data(),
                    input_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                    stream);
    T **gpu_output_values =
        reinterpret_cast<T **>(&gpu_input_values[input_data.size()]);
    cudaMemcpyAsync(gpu_output_values, output_data.data(),
                    output_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                    stream);
    int *gpu_idxs =
        reinterpret_cast<int *>(&gpu_output_values[output_data.size()]);
    cudaMemcpyAsync(gpu_idxs, idxs.data(), idxs.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    size_t N = static_cast<size_t>(batch_size * slot_size * total_cols);
    FusedSeqpoolConcatKernel<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                               stream>>>(
        N, gpu_input_values, gpu_output_values, gpu_idxs, &gpu_idxs[total_cols],
        &gpu_idxs[total_cols * 2], batch_size, total_cols, x_num);
  }
};

template <typename T>
__global__ void FusedSeqpoolSplitKernel(const size_t N, T **out_grads_values,
                                        T **in_grads_values, const int *idxs,
                                        const int *ptr_idxs, const int *dims,
                                        const int batch_size,
                                        const int total_cols, const int x_num) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / total_cols;
    int offset = i % total_cols;  // cols id
    int x = key / batch_size;     // slot id
    int y = key % batch_size;     // rows id

    *(in_grads_values[x * x_num + ptr_idxs[offset]] + y * dims[offset] +
      idxs[offset]) = *(out_grads_values[x] + y * total_cols + offset);
  }
}
template <typename T>
class FusedSeqpoolConcatGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto out_grads = ctx.MultiInput<LoDTensor>(framework::GradVarName("Out"));

    const int x_num = 2;
    const std::string input_names[] = {"X1", "X2"};
    std::vector<std::vector<LoDTensor *>> x_input_grads(x_num);
    for (int k = 0; k < x_num; ++k) {
      x_input_grads[k] =
          ctx.MultiOutput<LoDTensor>(framework::GradVarName(input_names[k]));
    }

    const int total_cols = ctx.Attr<int>("output_dim");
    const std::vector<int> idxs = ctx.Attr<std::vector<int>>("output_idx");
    CHECK(idxs.size() == static_cast<size_t>(3 * total_cols));

    const int slot_size = static_cast<int>(x_input_grads[0].size());
    std::vector<const T *> out_grads_data(slot_size);
    std::vector<T *> in_grads_data(slot_size * x_num);

    int batch_size = out_grads[0]->dims()[0];
    for (int i = 0; i < slot_size; ++i) {
      for (int k = 0; k < x_num; ++k) {
        auto *in_grad = x_input_grads[k][i];
        CHECK(batch_size == in_grad->dims()[0])
            << "batch: " << batch_size << ", current: " << in_grad->dims()[0];
        in_grads_data[i * x_num + k] =
            reinterpret_cast<T *>(in_grad->mutable_data<T>(place));
      }
      auto *out_grad = out_grads[i];
      out_grads_data[i] = reinterpret_cast<const T *>(out_grad->data<T>());
    }

    auto stream = dynamic_cast<phi::GPUContext*>(
                 platform::DeviceContextPool::Instance().Get(place))
                 ->stream();
    size_t total_ptr_len = out_grads_data.size() + in_grads_data.size();
    auto temp_ptr = memory::AllocShared(
        place, total_ptr_len * sizeof(void *) + sizeof(int) * idxs.size());
    T **gpu_out_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
    cudaMemcpyAsync(gpu_out_grads_values, out_grads_data.data(),
                    out_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                    stream);
    T **gpu_in_grads_values =
        reinterpret_cast<T **>(&gpu_out_grads_values[out_grads_data.size()]);
    cudaMemcpyAsync(gpu_in_grads_values, in_grads_data.data(),
                    in_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                    stream);
    int *gpu_idxs =
        reinterpret_cast<int *>(&gpu_in_grads_values[in_grads_data.size()]);
    cudaMemcpyAsync(gpu_idxs, idxs.data(), idxs.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    size_t N = static_cast<size_t>(batch_size * slot_size * total_cols);
    // update grad
    FusedSeqpoolSplitKernel<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                              stream>>>(
        N, gpu_out_grads_values, gpu_in_grads_values, gpu_idxs,
        &gpu_idxs[total_cols], &gpu_idxs[total_cols * 2], batch_size,
        total_cols, x_num);
  }
};

//================================== normal equal dim concat
//===========================
// equal dim concat
template <typename T>
__global__ void FusedColsConcatKernel(const size_t N, T **input_values,
                                      T *output_values, const int offset,
                                      const int dim_size, const int length,
                                      const int total_cols) {
  CUDA_KERNEL_LOOP(i, N) {
    int x = i % total_cols;
    int y = i / total_cols;
    output_values[i] =
        *(input_values[x / length] + y * dim_size + offset + (x % length));
  }
}
template <typename T>
class FusedConcatCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto output = ctx.Output<Tensor>("Out");
    auto inputs = ctx.MultiInput<LoDTensor>("X");

    const int length = ctx.Attr<int>("length");
    const int offset = ctx.Attr<int>("offset");

    const int x_num = static_cast<int>(inputs.size());
    const int total_cols = x_num * length;
    std::vector<const T *> input_data(x_num);

    int dim_size = inputs[0]->dims()[1];
    int batch_size = inputs[0]->dims()[0];
    for (int k = 0; k < x_num; ++k) {
      const auto *input = inputs[k];
      CHECK(batch_size == input->dims()[0])
          << "batch: " << batch_size << ", current: " << input->dims()[0];
      input_data[k] = reinterpret_cast<const T *>(input->data<T>());
    }
    output->Resize({batch_size, total_cols});
    T *gpu_out_value = reinterpret_cast<T *>(output->mutable_data<T>(place));

    auto stream = dynamic_cast<phi::GPUContext*>(
                 platform::DeviceContextPool::Instance().Get(place))
                 ->stream();

    auto temp_ptr = memory::AllocShared(place, x_num * sizeof(void *));
    T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
    cudaMemcpyAsync(gpu_input_values, input_data.data(), x_num * sizeof(T *),
                    cudaMemcpyHostToDevice, stream);
    size_t N = static_cast<size_t>(batch_size * total_cols);
    FusedColsConcatKernel<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        N, gpu_input_values, gpu_out_value, offset, dim_size, length,
        total_cols);
  }
};

// split equal dim concat
template <typename T>
__global__ void FusedColsSplitKernel(const size_t N, const T *input_values,
                                     T **output_values, const int offset,
                                     const int dim_size, const int length,
                                     const int total_cols) {
  CUDA_KERNEL_LOOP(i, N) {
    int x = i % total_cols;  // cols
    int y = i / total_cols;  // rows
    *(output_values[x / length] + y * dim_size + offset + (x % length)) =
        input_values[i];
  }
}
template <typename T>
class FusedConcatGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto out_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));

    const int length = ctx.Attr<int>("length");
    const int offset = ctx.Attr<int>("offset");
    const int x_num = static_cast<int>(in_grads.size());
    const int total_cols = x_num * length;
    std::vector<T *> in_grads_data(x_num);

    int batch_size = out_grad->dims()[0];
    int dim_size = in_grads[0]->dims()[1];
    for (int k = 0; k < x_num; ++k) {
      auto *in_grad = in_grads[k];
      CHECK(batch_size == in_grad->dims()[0])
          << "batch: " << batch_size << ", current: " << in_grad->dims()[0];
      in_grads_data[k] = reinterpret_cast<T *>(in_grad->mutable_data<T>(place));
    }
    const T *gpu_out_grad = reinterpret_cast<const T *>(out_grad->data<T>());

    auto stream = dynamic_cast<phi::GPUContext*>(
                 platform::DeviceContextPool::Instance().Get(place))
                 ->stream();
    auto temp_ptr = memory::AllocShared(place, x_num * sizeof(void *));
    T **gpu_in_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
    cudaMemcpyAsync(gpu_in_grads_values, in_grads_data.data(),
                    x_num * sizeof(T *), cudaMemcpyHostToDevice, stream);

    size_t N = static_cast<size_t>(batch_size * total_cols);
    // update grad
    FusedColsSplitKernel<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        N, gpu_out_grad, gpu_in_grads_values, offset, dim_size, length,
        total_cols);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_seqpool_concat,
                        ops::FusedSeqpoolConcatCUDAKernel<float>);
REGISTER_OP_CUDA_KERNEL(fused_seqpool_concat_grad,
                        ops::FusedSeqpoolConcatGradCUDAKernel<float>);

REGISTER_OP_CUDA_KERNEL(fused_concat, ops::FusedConcatCUDAKernel<float>);
REGISTER_OP_CUDA_KERNEL(fused_concat_grad,
                        ops::FusedConcatGradCUDAKernel<float>);
