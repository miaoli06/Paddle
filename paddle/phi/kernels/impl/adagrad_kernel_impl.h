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

#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/kernels/adagrad_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename DeviceContext, typename T>
struct SparseAdagradFunctor {
  void operator()(const DeviceContext& context,
                  const phi::SelectedRows& grad,
                  const DenseTensor& learning_rate,
                  T epsilon,
                  DenseTensor* moment,
                  DenseTensor* param);
};

template <typename DeviceContext, typename T>
phi::SelectedRows SquareSelectedRows(const DeviceContext& context,
                                     const phi::SelectedRows& input) {
  phi::SelectedRows out;
  out.set_rows(input.rows());
  out.set_height(input.height());
  out.mutable_value()->Resize(input.value().dims());
  context.template Alloc<T>(out.mutable_value());
  auto e_out = EigenVector<T>::Flatten(*(out.mutable_value()));
  auto e_in = EigenVector<T>::Flatten(input.value());
  e_out.device(*context.eigen_device()) = e_in.square();
  return out;
}

template <typename T, typename Context>
void AdagradDenseKernel(const Context& ctx,
                        const DenseTensor& param_t,
                        const DenseTensor& grad_t,
                        const DenseTensor& moment_t,
                        const DenseTensor& learning_rate,
                        float epsilon_t,
                        DenseTensor* param_out_tensor,
                        DenseTensor* moment_out_tensor) {
  ctx.template Alloc<T>(param_out_tensor);
  ctx.template Alloc<T>(moment_out_tensor);

  T epsilon = static_cast<T>(epsilon_t);

  auto param = EigenVector<T>::Flatten(param_t);

  auto grad = EigenVector<T>::Flatten(grad_t);

  auto moment = EigenVector<T>::Flatten(moment_t);

  auto param_out = EigenVector<T>::Flatten(*param_out_tensor);
  auto moment_out = EigenVector<T>::Flatten(*moment_out_tensor);
  auto place = *ctx.eigen_device();

  moment_out.device(place) = moment + grad * grad;
  Eigen::DSizes<int, 1> m_dsize(moment_out_tensor->numel());
  if (paddle::platform::is_cpu_place(ctx.GetPlace())) {
    auto* lr = learning_rate.data<T>();
    param_out.device(place) =
        param - lr[0] * grad / (moment_out.sqrt() + epsilon);
  } else {
    auto lr = EigenVector<T>::Flatten(learning_rate);
    param_out.device(place) =
        param - lr.broadcast(m_dsize) * grad / (moment_out.sqrt() + epsilon);
  }
}

template <typename T, typename Context>
void AdagradSparseKernel(const Context& ctx,
                         const DenseTensor& param_t,
                         const SelectedRows& grad_t,
                         const DenseTensor& moment_t,
                         const DenseTensor& learning_rate,
                         float epsilon_t,
                         DenseTensor* param_out,
                         DenseTensor* moment_out) {
  auto* param_out_tensor = param_out;
  auto* moment_out_tensor = moment_out;

  ctx.template Alloc<T>(param_out_tensor);
  ctx.template Alloc<T>(moment_out_tensor);

  T epsilon = static_cast<T>(epsilon_t);

  auto* param_tensor = &param_t;
  PADDLE_ENFORCE_EQ(param_tensor,
                    param_out_tensor,
                    phi::errors::InvalidArgument(
                        "the input tensor not euqal with output tensor"));

  auto* moment_tensor = &moment_t;
  PADDLE_ENFORCE_EQ(moment_tensor,
                    moment_out_tensor,
                    phi::errors::InvalidArgument(
                        "the input moment not eual with output moment"));

  SparseAdagradFunctor<Context, T> functor;
  functor(
      ctx, grad_t, learning_rate, epsilon, moment_out_tensor, param_out_tensor);
}

}  // namespace phi
