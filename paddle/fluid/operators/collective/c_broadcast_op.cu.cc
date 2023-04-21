/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_broadcast_op.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/phi/api/include/tensor.h"
#include "paddle/fluid/framework/data_type_transform.h"

namespace paddle {
namespace operators {

template <typename T>
class CBroadcastOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int rid = ctx.Attr<int>("ring_id");
    const auto& place = ctx.GetPlace();
    ctx.device_context().Alloc<T>(out);

    int root = ctx.Attr<int>("root");

    gpuStream_t stream = ctx.cuda_device_context().stream();
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (comm_context_manager.Has(rid)) {
      auto* comm_context = static_cast<phi::distributed::NCCLCommContext*>(
          comm_context_manager.Get(rid));
      comm_context->Broadcast(out, *x, root, stream);
    } else {
      // NOTE(liyurui): This will be removed after moving this operator to phi.
      int numel = x->numel();
      auto out_dtype = out->dtype();
      ncclDataType_t dtype =
          platform::ToNCCLDataType(framework::TransToProtoVarType(out_dtype));
      auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
      if (root == comm->rank()) {
        auto in_dtype = x->dtype();
        if (in_dtype != out_dtype) {
          // convert to float16 tensor
          auto in_kernel_type =
             phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, in_dtype);
          auto out_kernel_type =
             phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, out_dtype);
          framework::TransDataType(
             in_kernel_type, out_kernel_type, *x, out);

          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclBcast(
              out->data<T>(),
              numel,
              dtype,
              root,
              comm->comm(),
              stream));
          VLOG(3) << "rank " << comm->rank() << " invoke Bcast. cast sent "
                  << x->numel();
        } else {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclBcast(
              reinterpret_cast<void*>(const_cast<T*>(x->data<T>())),
              numel,
              dtype,
              root,
              comm->comm(),
              stream));
          VLOG(3) << "rank " << comm->rank() << " invoke Bcast. sent "
                  << x->numel();
          if (out != x) {
            framework::TensorCopy(
                *static_cast<const phi::DenseTensor*>(x),
                place,
                *platform::DeviceContextPool::Instance().Get(place),
                static_cast<phi::DenseTensor*>(out));
          }
        }
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclBcast(
            out->data<T>(), numel, dtype, root, comm->comm(), stream));
        VLOG(3) << "rank " << comm->rank() << " invoke Bcast. received "
                << phi::product(out->dims());
      }
    }

    out->set_lod(x->lod());
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(c_broadcast,
                        ops::CBroadcastOpCUDAKernel<float>,
                        ops::CBroadcastOpCUDAKernel<double>,
#if NCCL_VERSION_CODE >= 21000
                        ops::CBroadcastOpCUDAKernel<plat::bfloat16>,
#endif
                        ops::CBroadcastOpCUDAKernel<int>,
                        ops::CBroadcastOpCUDAKernel<int64_t>,
                        ops::CBroadcastOpCUDAKernel<plat::float16>);
