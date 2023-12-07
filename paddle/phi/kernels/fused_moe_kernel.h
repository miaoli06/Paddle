// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/collective/global_gather_op.h"
#include "paddle/fluid/operators/collective/global_scatter_op.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/operators/layer_norm_kernel.cu.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/bmm_kernel.h"
#include "paddle/phi/kernels/cum_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/index_select_kernel.h"
#include "paddle/phi/kernels/number_count_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/scatter_kernel.h"
#include "paddle/phi/kernels/top_k_kernel.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11040)
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#endif
namespace phi {
using Tensor = DenseTensor;
namespace framework = paddle::framework;
namespace platform = paddle::platform;

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
static ncclComm_t GetNCCLCommRanks(const Place& place,
                                   const int ring_id,
                                   int* nranks) {
  static auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  if (map->has(ring_id)) {
    auto pg_nccl =
        static_cast<paddle::distributed::ProcessGroupNCCL*>(map->get(ring_id));
    *nranks = pg_nccl->GetSize();
    return pg_nccl->NCCLComm(place);
  } else {
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    *nranks = comm->nranks();
    return comm->comm();
  }
}
#endif

template <typename T>
static void AllReduce(phi::DenseTensor& tensor,  // NOLINT
                      const int ring_id,
                      const int count,
                      const phi::GPUContext& ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto place = ctx.GetPlace();
  int nranks = 0;
  auto comm = GetNCCLCommRanks(place, ring_id, &nranks);

  auto dtype =
      platform::ToNCCLDataType(framework::TransToProtoVarType(tensor.dtype()));
  int64_t numel = tensor.numel();
  const void* sendbuff = tensor.data<T>();
  void* recvbuff = tensor.mutable_data<T>(place);
  auto stream = ctx.stream();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
      sendbuff, recvbuff, count, dtype, ncclSum, comm, stream));
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename T>
static void AllToAll(Tensor& tensor,  // NOLINT
                     Tensor& out,
                     const int ring_id,
                     const phi::GPUContext& ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto place = ctx.GetPlace();
  int nranks = 0;
  auto comm = GetNCCLCommRanks(place, ring_id, &nranks);

  auto dtype =
      platform::ToNCCLDataType(framework::TransToProtoVarType(tensor.dtype()));
  int64_t send_numel = tensor.numel();  // send_numel

  auto stream = ctx.stream();

  framework::DDim x_dims = tensor.dims();
  framework::DDim out_dims(x_dims);
  PADDLE_ENFORCE_EQ(
      x_dims[0] % nranks,
      0,
      platform::errors::InvalidArgument(
          "The first dimension size (%d) of the input tensor must be "
          "divisible by the number of ranks (%d).",
          x_dims[0],
          nranks));
  auto send_buf = tensor.data<T>();
  auto recv_buf = out.mutable_data<T>(out_dims, place);
  size_t offset = 0;
  send_numel /= nranks;
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  for (auto i = 0; i < nranks; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
        send_buf + offset, send_numel, dtype, i, comm, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
        recv_buf + offset, send_numel, dtype, i, comm, stream));
    offset += send_numel;
  }
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename T>
static void AllGather(Tensor& tensor,  // NOLINT
                      Tensor& out,
                      const int ring_id,
                      const phi::GPUContext& ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto place = ctx.GetPlace();
  int nranks = 0;
  auto comm = GetNCCLCommRanks(place, ring_id, &nranks);
  auto dtype =
      platform::ToNCCLDataType(framework::TransToProtoVarType(tensor.dtype()));
  int64_t numel = tensor.numel();
  auto stream = ctx.stream();
  auto out_dims = tensor.dims();
  out_dims[0] *= nranks;
  out.mutable_data<T>(out_dims, place);
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      tensor.data<T>(), out.data<T>(), numel, dtype, comm, stream));
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename T>
void GlobalScatterFunctor(const phi::GPUContext& ctx,
                          const framework::Tensor* x,
                          const framework::Tensor* local_count,
                          const framework::Tensor* global_count,
                          int ring_id,
                          bool use_calc_stream,
                          framework::Tensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
  const int64_t* cpu_local_count_data;
  const int64_t* cpu_global_count_data;
  framework::Tensor cpu_local_count;
  if (platform::is_cpu_place(local_count->place())) {
    cpu_local_count_data = local_count->data<int64_t>();
  } else {
    framework::TensorCopy(*local_count, platform::CPUPlace(), &cpu_local_count);
    cpu_local_count_data = cpu_local_count.data<int64_t>();
  }
  auto global_count_len = 0;
  framework::Tensor cpu_global_count;
  if (platform::is_cpu_place(global_count->place())) {
    cpu_global_count_data = global_count->data<int64_t>();
    global_count_len = global_count->numel();
  } else {
    framework::TensorCopy(
        *global_count, platform::CPUPlace(), &cpu_global_count);
    cpu_global_count_data = cpu_global_count.data<int64_t>();
    global_count_len = cpu_global_count.numel();
  }
  ctx.Wait();

  ncclDataType_t dtype =
      platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      platform::errors::InvalidArgument(
          "The ring_id (%d) for global scatter op must be non-negative.",
          ring_id));

  auto place = ctx.GetPlace();
  int nranks = 0;
  auto comm = GetNCCLCommRanks(place, ring_id, &nranks);

  gpuStream_t stream = ctx.stream();

  auto in_feat = x->dims()[1];
  auto n_expert = local_count->dims()[0] / nranks;
  std::vector<int64_t> expert_ptr(n_expert * nranks);
  expert_ptr[0] = 0;
  auto tot_experts = n_expert * nranks;
  for (auto i = 1; i < tot_experts; ++i) {
    expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
  }

  auto recv_ptr = 0;
  auto send_buf = x->data<T>();
  auto recv_buf = out->data<T>();

  for (auto i = 0; i < n_expert; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto j = 0; j < nranks; ++j) {
      int idx = i + j * n_expert;
      if (cpu_local_count_data[idx]) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::ncclSend(send_buf + expert_ptr[idx] * in_feat,
                                        cpu_local_count_data[idx] * in_feat,
                                        dtype,
                                        j,
                                        comm,
                                        stream));
      }
      if (cpu_global_count_data[idx]) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::ncclRecv(recv_buf + recv_ptr * in_feat,
                                        cpu_global_count_data[idx] * in_feat,
                                        dtype,
                                        j,
                                        comm,
                                        stream));
        recv_ptr += cpu_global_count_data[idx];
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  }

#else
  PADDLE_THROW(
      platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
  PADDLE_THROW(
      platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
}

template <typename T>
void GlobalScatterProcessGroupFunctor(const phi::GPUContext& ctx,
                                      const framework::Tensor* x,
                                      const framework::Tensor* local_count,
                                      const framework::Tensor* global_count,
                                      int ring_id,
                                      bool use_calc_stream,
                                      framework::Tensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
  const int64_t* cpu_local_count_data;
  const int64_t* cpu_global_count_data;
  framework::Tensor cpu_local_count;
  if (platform::is_cpu_place(local_count->place())) {
    cpu_local_count_data = local_count->data<int64_t>();
  } else {
    framework::TensorCopy(*local_count, platform::CPUPlace(), &cpu_local_count);
    cpu_local_count_data = cpu_local_count.data<int64_t>();
  }
  auto global_count_len = 0;
  framework::Tensor cpu_global_count;
  if (platform::is_cpu_place(global_count->place())) {
    cpu_global_count_data = global_count->data<int64_t>();
    global_count_len = global_count->numel();
  } else {
    framework::TensorCopy(
        *global_count, platform::CPUPlace(), &cpu_global_count);
    cpu_global_count_data = cpu_global_count.data<int64_t>();
    global_count_len = cpu_global_count.numel();
  }
  ctx.Wait();

  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      platform::errors::InvalidArgument(
          "The ring_id (%d) for global scatter op must be non-negative.",
          ring_id));

  auto place = ctx.GetPlace();
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  paddle::distributed::ProcessGroup* pg = map->get(ring_id);
  int nranks = pg->GetSize();
  auto in_feat = x->dims()[1];
  auto n_expert = local_count->dims()[0] / nranks;

  std::vector<int64_t> expert_ptr(n_expert * nranks);
  expert_ptr[0] = 0;
  auto tot_experts = n_expert * nranks;
  for (auto i = 1; i < tot_experts; ++i) {
    expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
  }

  auto recv_ptr = 0;
  for (auto i = 0; i < n_expert; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto j = 0; j < nranks; ++j) {
      int idx = i + j * n_expert;
      if (cpu_local_count_data[idx]) {
        phi::DenseTensor tmp = *x;
        pg->Send_Partial(tmp,
                         j,
                         expert_ptr[idx] * in_feat,
                         cpu_local_count_data[idx] * in_feat);
      }
      if (cpu_global_count_data[idx]) {
        pg->Recv_Partial(
            *out, j, recv_ptr * in_feat, cpu_global_count_data[idx] * in_feat);
        recv_ptr += cpu_global_count_data[idx];
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  }

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

#else
  PADDLE_THROW(
      platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
  PADDLE_THROW(
      platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
}

template <typename T>
void GlobalGatherFunctor(const phi::GPUContext& ctx,
                         const framework::Tensor* x,
                         const framework::Tensor* local_count,
                         const framework::Tensor* global_count,
                         int ring_id,
                         bool use_calc_stream,
                         framework::Tensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
  const int64_t* cpu_local_count_data;
  const int64_t* cpu_global_count_data;
  auto local_count_len = 0;

  framework::Tensor cpu_local_count;
  if (platform::is_cpu_place(local_count->place())) {
    cpu_local_count_data = local_count->data<int64_t>();
    local_count_len = local_count->numel();
  } else {
    framework::TensorCopy(*local_count, platform::CPUPlace(), &cpu_local_count);
    cpu_local_count_data = cpu_local_count.data<int64_t>();
    local_count_len = cpu_local_count.numel();
  }

  framework::Tensor cpu_global_count;
  if (platform::is_cpu_place(global_count->place())) {
    cpu_global_count_data = global_count->data<int64_t>();
  } else {
    framework::TensorCopy(
        *global_count, platform::CPUPlace(), &cpu_global_count);
    cpu_global_count_data = cpu_global_count.data<int64_t>();
  }
  ctx.Wait();

  ncclDataType_t dtype =
      platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      platform::errors::InvalidArgument(
          "The ring_id (%d) for global gather op must be non-negative.",
          ring_id));

  auto place = ctx.GetPlace();
  int nranks = 0;
  auto comm = GetNCCLCommRanks(place, ring_id, &nranks);

  gpuStream_t stream = ctx.stream();

  auto in_feat = x->dims()[1];
  auto n_expert = local_count->dims()[0] / nranks;

  std::vector<int64_t> expert_ptr(n_expert * nranks);
  expert_ptr[0] = 0;
  auto tot_experts = n_expert * nranks;
  for (auto i = 1; i < tot_experts; ++i) {
    expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
  }
  auto send_ptr = 0;
  auto send_buf = x->data<T>();
  auto recv_buf = out->data<T>();

  for (auto i = 0; i < n_expert; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto j = 0; j < nranks; ++j) {
      int idx = i + j * n_expert;
      if (cpu_global_count_data[idx]) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::ncclSend(send_buf + send_ptr * in_feat,
                                        cpu_global_count_data[idx] * in_feat,
                                        dtype,
                                        j,
                                        comm,
                                        stream));
        send_ptr += cpu_global_count_data[idx];
      }
      if (cpu_local_count_data[idx]) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::ncclRecv(recv_buf + expert_ptr[idx] * in_feat,
                                        cpu_local_count_data[idx] * in_feat,
                                        dtype,
                                        j,
                                        comm,
                                        stream));
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  }

#else
  PADDLE_THROW(
      platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
  PADDLE_THROW(
      platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
}

template <typename T>
void GlobalGatherProcessGroupFunctor(const phi::GPUContext& ctx,
                                     const framework::Tensor* x,
                                     const framework::Tensor* local_count,
                                     const framework::Tensor* global_count,
                                     int ring_id,
                                     bool use_calc_stream,
                                     framework::Tensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
  const int64_t* cpu_local_count_data;
  const int64_t* cpu_global_count_data;
  auto local_count_len = 0;

  framework::Tensor cpu_local_count;
  if (platform::is_cpu_place(local_count->place())) {
    cpu_local_count_data = local_count->data<int64_t>();
    local_count_len = local_count->numel();
  } else {
    framework::TensorCopy(*local_count, platform::CPUPlace(), &cpu_local_count);
    cpu_local_count_data = cpu_local_count.data<int64_t>();
    local_count_len = cpu_local_count.numel();
  }

  framework::Tensor cpu_global_count;
  if (platform::is_cpu_place(global_count->place())) {
    cpu_global_count_data = global_count->data<int64_t>();
  } else {
    framework::TensorCopy(
        *global_count, platform::CPUPlace(), &cpu_global_count);
    cpu_global_count_data = cpu_global_count.data<int64_t>();
  }
  ctx.Wait();

  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      platform::errors::InvalidArgument(
          "The ring_id (%d) for global gather op must be non-negative.",
          ring_id));
  auto place = ctx.GetPlace();
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  int nranks = pg->GetSize();
  auto in_feat = x->dims()[1];
  auto n_expert = local_count->dims()[0] / nranks;

  std::vector<int64_t> expert_ptr(n_expert * nranks);
  expert_ptr[0] = 0;
  auto tot_experts = n_expert * nranks;
  for (auto i = 1; i < tot_experts; ++i) {
    expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
  }
  auto send_ptr = 0;

  for (auto i = 0; i < n_expert; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto j = 0; j < nranks; ++j) {
      int idx = i + j * n_expert;
      if (cpu_global_count_data[idx]) {
        phi::DenseTensor tmp = *x;
        pg->Send_Partial(
            tmp, j, send_ptr * in_feat, cpu_global_count_data[idx] * in_feat);
        send_ptr += cpu_global_count_data[idx];
      }
      if (cpu_local_count_data[idx]) {
        pg->Recv_Partial(*out,
                         j,
                         expert_ptr[idx] * in_feat,
                         cpu_local_count_data[idx] * in_feat);
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  }

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

#else
  PADDLE_THROW(
      platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
  PADDLE_THROW(
      platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
}

template <typename T>
void MatMulAndAddGelu(const phi::GPUContext& dev_ctx,
                      const framework::Tensor* weight,
                      const framework::Tensor* input,
                      const framework::Tensor* bias,
                      bool istransA,
                      bool istransB,
                      bool compute_bias,
                      framework::Tensor* output) {
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11040)
  phi::funcs::LinearWithCublasLt<T>::Run(
      dev_ctx,
      input,   // x
      weight,  // y
      output,  // out
      ((compute_bias) ? static_cast<const void*>(bias->data<T>())
                      : nullptr),  // bias
      nullptr,
      input->dims()[0],   // M  bsz_seq
      weight->dims()[1],  // N  output_size
      input->dims()[1],   // K  input_size
      istransA,
      istransB,
      ((compute_bias) ? phi::funcs::MatmulFusedType::kMatmulBiasGelu
                      : phi::funcs::MatmulFusedType::kMatmulGelu));
#endif
}

template <typename T>
void MatMulAndAdd(const phi::GPUContext& dev_ctx,
                  const framework::Tensor* weight,
                  const framework::Tensor* input,
                  const framework::Tensor* bias,
                  bool istransA,
                  bool istransB,
                  bool compute_bias,
                  framework::Tensor* output,
                  framework::Tensor* bias_out) {
#if (CUDA_VERSION >= 11040)
  if (compute_bias) {
    phi::funcs::LinearWithCublasLt<T>::Run(
        dev_ctx,
        input,                                      // x
        weight,                                     // y
        bias_out,                                   // out
        static_cast<const void*>(bias->data<T>()),  // bias
        nullptr,
        input->dims()[0],   // M   bsz_seq
        weight->dims()[1],  // N   output_size
        input->dims()[1],   // K   input_size
        istransA,
        istransB,
        phi::funcs::MatmulFusedType::kMatmulBias);
    return;
  }
#endif
  // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
  // here: (transa, transb): nt, input * weight.
  CBLAS_TRANSPOSE transA = istransA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = istransB ? CblasTrans : CblasNoTrans;
  T alpha = static_cast<T>(1.0);
  T beta = static_cast<T>(0.0);

  // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
  auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
  blas.GEMM(transA,
            transB,
            input->dims()[0],
            weight->dims()[1],
            input->dims()[1],
            alpha,
            input->data<T>(),
            weight->data<T>(),
            beta,
            output->data<T>());
  if (compute_bias) {
    // bias_out = output + bias
    std::vector<const framework::Tensor*> ins = {output, bias};
    std::vector<framework::Tensor*> outs = {bias_out};
    phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, -1, phi::funcs::AddFunctor<T>());
  }
}

template <typename T, typename DeviceContext>
void FusedMoeKernel(const DeviceContext& context,
                    const DenseTensor& x,
                    const DenseTensor& gate_weight,
                    const DenseTensor& gate_bias,
                    const DenseTensor& ln_scale,
                    const DenseTensor& ln_bias,
                    const std::vector<const DenseTensor*>& experts_weight1,
                    const std::vector<const DenseTensor*>& experts_bias1,
                    const std::vector<const DenseTensor*>& experts_weight2,
                    const std::vector<const DenseTensor*>& experts_bias2,
                    bool pre_layer_norm,
                    float ln_epsilon,
                    int topk,
                    int mp_size,
                    int mp_rank,
                    int num_expert,
                    int world_size,
                    int moe_ring_id,
                    bool approximate,
                    DenseTensor* out);

}  // namespace phi
