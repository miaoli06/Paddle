/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class MeanOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "mean");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "mean");
    ctx->SetOutputDim("Out", {1});
  }
};

class MeanOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of mean op");
    AddOutput("Out", "(Tensor) The output of mean op");
    AddComment(R"DOC(
Mean Operator calculates the mean of all elements in X.

)DOC");
  }
};

class MeanOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class MeanGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class MeanGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("mean_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(MeanGradNoNeedBufferVarsInferer, "X");

//================================ mask mean ==================================
class MaskMeanOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "mask_mean");
    OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "mask_mean");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "mask_mean");
    OP_INOUT_CHECK(ctx->HasOutput("Num"), "Output", "Num", "mask_mean");
    ctx->SetOutputDim("Out", {1});
    ctx->SetOutputDim("Num", {1});
  }
};

class MaskMeanOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of mean op");
    AddInput("Mask", "(Tensor) The input of mean op");
    AddOutput("Out", "(Tensor) The output of mean op");
    AddOutput("Num", "(Tensor) The output of mean op");
    AddComment(R"DOC(
Mean Operator calculates the mean of all elements in X.

)DOC");
  }
};

class MaskMeanOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class MaskMeanGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class MaskMeanGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("mask_mean_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Mask", this->Input("Mask"));
    grad_op->SetInput("Num", this->Output("Num"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(MaskMeanGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(mean,
                            MeanInferShapeFunctor,
                            PD_INFER_META(phi::MeanAllInferMeta));
REGISTER_OPERATOR(mean,
                  ops::MeanOp,
                  ops::MeanOpMaker,
                  ops::MeanOpInferVarType,
                  ops::MeanGradMaker<paddle::framework::OpDesc>,
                  ops::MeanGradMaker<paddle::imperative::OpBase>,
                  MeanInferShapeFunctor);

REGISTER_OPERATOR(mean_grad,
                  ops::MeanGradOp,
                  ops::MeanGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(
    mean, ops::MeanKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MeanKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    mean_grad, ops::MeanGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MeanGradKernel<paddle::platform::CPUDeviceContext, double>);

// mask mean
REGISTER_OPERATOR(mask_mean, ops::MaskMeanOp, ops::MaskMeanOpMaker,
                  ops::MaskMeanOpInferVarType,
                  ops::MaskMeanGradMaker<paddle::framework::OpDesc>,
                  ops::MaskMeanGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(mask_mean_grad, ops::MaskMeanGradOp,
                  ops::MaskMeanGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(
    mask_mean, ops::MaskMeanKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MaskMeanKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    mask_mean_grad,
    ops::MaskMeanGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MaskMeanGradKernel<paddle::platform::CPUDeviceContext, double>);
