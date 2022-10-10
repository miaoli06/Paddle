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

#pragma once
#include <memory>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = phi::DenseTensor;
//=============== tensor vector concat part to tensor vector ===================
template <typename T>
class FusedSeqpoolConcatOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_THROW(
        "Unimplemented CPU kernel for FusedConcatOp, only support GPU "
        "now.");
  }
};

template <typename T>
class FusedSeqpoolConcatGradOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_THROW(
        "Unimplemented CPU kernel for FusedConcatGradOp, only support GPU "
        "now.");
  }
};

//==================== tensor vector concat to one tensor =====================
template <typename T>
class FusedConcatOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto output = ctx.Output<Tensor>("Out");
    auto inputs = ctx.MultiInput<LoDTensor>("X");

    const int length = ctx.Attr<int>("length");
    const int offset = ctx.Attr<int>("offset");
    const int x_num = static_cast<int>(inputs.size());
    const int total_cols = x_num * length;

    int dim_size = inputs[0]->dims()[1];
    int batch_size = inputs[0]->dims()[0];
    T *out_value_ptr = reinterpret_cast<T *>(output->mutable_data<T>(place));
    for (int k = 0; k < x_num; ++k) {
      const auto *input = inputs[k];
      CHECK(batch_size == input->dims()[0])
          << "batch: " << batch_size << ", current: " << input->dims()[0];
      const T *input_ptr = reinterpret_cast<const T *>(input->data<T>());
      for (int y = 0; y < batch_size; ++y) {
        int pos = y * total_cols + k * length;
        for (int i = 0; i < length; ++i) {
          out_value_ptr[pos + i] = input_ptr[y * dim_size + i + offset];
        }
      }
    }
  }
};

template <typename T>
class FusedConcatGradOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto out_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));

    const int length = ctx.Attr<int>("length");
    const int offset = ctx.Attr<int>("offset");
    const int x_num = static_cast<int>(in_grads.size());
    const int total_cols = x_num * length;

    const T *out_grad_ptr = reinterpret_cast<const T *>(out_grad->data<T>());
    int batch_size = out_grad->dims()[0];
    int dim_size = in_grads[0]->dims()[1];
    for (int k = 0; k < x_num; ++k) {
      auto *in_grad = in_grads[k];
      CHECK(batch_size == in_grad->dims()[0])
          << "batch: " << batch_size << ", current: " << in_grad->dims()[0];
      T *in_grad_ptr = reinterpret_cast<T *>(in_grad->mutable_data<T>(place));
      for (int y = 0; y < batch_size; ++y) {
        int pos = y * total_cols + k * length;
        for (int i = 0; i < length; ++i) {
          in_grad_ptr[y * dim_size + i + offset] = out_grad_ptr[pos + i];
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
