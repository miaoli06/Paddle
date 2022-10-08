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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/operators/batch_size_like.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class FillConstantBatchSizeLikeOp : public BatchSizeLikeOp {
 protected:
  using BatchSizeLikeOp::BatchSizeLikeOp;
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kernel_type = framework::OpKernelType(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.device_context());
    if (ctx.Attr<bool>("force_cpu")) {
      kernel_type.place_ = platform::CPUPlace();
    }
    return kernel_type;
  }
};

class FillConstantBatchSizeLikeOpMaker : public BatchSizeLikeOpMaker {
 protected:
  void Apply() override {
    AddAttr<int>(
        "dtype",
        "It could be numpy.dtype. Output data type. Default is float32")
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<float>("value", "default 0. The value to be filled")
        .SetDefault(0.0f);
    AddAttr<std::string>("str_value", "default empty. The value to be filled")
        .SetDefault("");
    AddAttr<bool>("force_cpu",
                  "(bool, default false) Force fill output variable to cpu "
                  "memory. Otherwise, fill output variable to the running "
                  "device")
        .SetDefault(false);
    AddComment(R"DOC(
This function creates a tensor of specified *shape*, *dtype* and batch size,
and initializes this with a constant supplied in *value*. The batch size is
obtained from the `input` tensor.

)DOC");
  }
};

template <typename T>
class FillConstantBatchSizeLikeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
//    auto data_type =
//        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto float_value = ctx.Attr<float>("value");
    auto str_value = ctx.Attr<std::string>("str_value");

    auto *out = ctx.Output<framework::LoDTensor>("Out");
    auto *in = ctx.Input<framework::LoDTensor>("Input");
    if (in->lod().size() && ctx.Attr<int>("input_dim_idx") == 0) {
      // set the correct batch size for the LoDTensor.
      auto odims = out->dims();
      int output_dim_idx = ctx.Attr<int>("output_dim_idx");
      odims[output_dim_idx] = static_cast<int>(in->lod().back().size()) - 1;
      out->mutable_data<T>(odims, ctx.GetPlace());
    }

    T value;
    if (str_value.empty()) {
      value = static_cast<T>(float_value);
    } else {
      // handle NaN/Inf first, which cannot be read from stream.
      if (str_value == "inf") {
        value = static_cast<T>(std::numeric_limits<double>::infinity());
      } else if (str_value == "-inf") {
        value = static_cast<T>(-std::numeric_limits<double>::infinity());
      } else if (str_value == "nan") {
        value = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
      } else {
        std::stringstream convert_stream(str_value);
        if (std::is_same<int64_t, T>::value) {
          int64_t tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        } else {
          double tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        }
      }
    }
    out->mutable_data<T>(ctx.GetPlace());
    auto &dev_ctx = ctx.template device_context<phi::DeviceContext>();
    phi::funcs::set_constant(dev_ctx, out, static_cast<T>(value));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(fill_constant_batch_size_like,
                            FillConstantBatchSizeLikeInferShapeFunctor,
                            PD_INFER_META(phi::FullBatchSizeLikeInferMeta));
REGISTER_OPERATOR(
    fill_constant_batch_size_like,
    ops::FillConstantBatchSizeLikeOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::FillConstantBatchSizeLikeOpMaker,
    ops::BatchSizeLikeNoNeedBufferVarsInferer,
    FillConstantBatchSizeLikeInferShapeFunctor);

#ifdef PADDLE_WITH_XPU
REGISTER_OP_XPU_KERNEL(
    fill_constant_batch_size_like,
    ops::FillConstantBatchSizeLikeOpKernel<float>,
    ops::FillConstantBatchSizeLikeOpKernel<double>,
    ops::FillConstantBatchSizeLikeOpKernel<uint8_t>,
    ops::FillConstantBatchSizeLikeOpKernel<int16_t>,
    ops::FillConstantBatchSizeLikeOpKernel<int>,
    ops::FillConstantBatchSizeLikeOpKernel<int64_t>,
    ops::FillConstantBatchSizeLikeOpKernel<bool>,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::float16>);
#endif
