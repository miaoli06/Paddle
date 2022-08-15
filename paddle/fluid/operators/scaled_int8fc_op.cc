/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/scaled_int8fc_op.h"
#include <string>

namespace paddle {
namespace operators {

class ScaledINT8FCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "ScaledINT8FCOp");
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "ScaledINT8FCOp");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "ScaledINT8FCOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ScaledINT8FCOp");

    auto input_dims = ctx->GetInputDim("Input");
    auto w_dims = ctx->GetInputDim("W");

    int feature_dim = input_dims[1];
    PADDLE_ENFORCE_EQ(feature_dim, w_dims[0],
                      platform::errors::InvalidArgument(
                          "Input.dim[1] and W.dim[0] of ScaledINT8FCOp "
                          "should be same."));

    auto bias_dims = ctx->GetInputDim("Bias");
    //PADDLE_ENFORCE_EQ(bias_dims[1], w_dims[1],
    //                  platform::errors::InvalidArgument(
    //                      "Bias.dim[1] should be same as W.dim[1]."));
    PADDLE_ENFORCE_EQ(bias_dims[0], w_dims[1],
                      platform::errors::InvalidArgument(
                          "Bias.dim[1] should be same as W.dim[1]."));

    ctx->SetOutputDim("Out", {input_dims[0], w_dims[1]});
    ctx->ShareLoD("Input", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class ScaledINT8FCGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::InvalidArgument("Input should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("W"), true,
        platform::errors::InvalidArgument("Input(W) should not be null"));

    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
    ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));
    ctx->SetOutputDim(framework::GradVarName("Bias"), ctx->GetInputDim("Bias"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class ScaledINT8FCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) Input tensor of scaled_int8fc_op operator.");
    AddInput("W", "(Tensor) Input tensor of scaled_int8fc_op operator.");
    AddInput("Bias", "(Tensor) Input tensor of scaled_int8fc_op operator.");
    AddAttr<float>("input_scale_factor", "(float) the scale factor of input tensor");
    AddAttr<float>("bias_scale_factor", "(float) the scale factor of bias tensor");
    AddAttr<float>("grad_scale_factor", "(float) the scale factor of gradient tensor");
    
    AddAttr<float>("expand_factor", "(float) the input first mul expand_factor then cast to int8");
    AddAttr<float>("clip_factor", "(float) the max value for casting to int8");
    
    AddAttr<float>("weight_expand_factor", "(float) the input first mul expand_factor then cast to int8");
    AddAttr<float>("weight_clip_factor", "(float) the max value for casting to int8");
    
    AddAttr<float>("int8_range", "(float) the range of int8");

    AddOutput("Out", "Output tensor of scaled_int8fc_op operator.");
    AddComment(R"DOC(
ScaledFC Operator.
Notice: It currently supports GPU device.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
  }
};

template <typename T>
class ScaledINT8FCGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("scaled_int8fc_grad");

    op->SetInput("Input", this->Input("Input"));
    op->SetInput("W", this->Input("W"));
    op->SetInput("Bias", this->Input("Bias"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle
using CPUCtx = phi::CPUContext;
namespace ops = paddle::operators;
REGISTER_OPERATOR(scaled_int8fc, ops::ScaledINT8FCOp, ops::ScaledINT8FCOpMaker,
                  ops::ScaledINT8FCGradOpMaker<paddle::framework::OpDesc>,
                  ops::ScaledINT8FCGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(scaled_int8fc_grad, ops::ScaledINT8FCGradOp);

REGISTER_OP_CPU_KERNEL(
    scaled_int8fc, ops::ScaledINT8FCKernel<CPUCtx, float>,
    ops::ScaledINT8FCKernel<CPUCtx, double>);
