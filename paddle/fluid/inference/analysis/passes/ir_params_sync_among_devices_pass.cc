// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/passes/ir_params_sync_among_devices_pass.h"

#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"

DECLARE_bool(enable_opt_infer_offload);
DECLARE_bool(enable_opt_infer_debug_mode);
PADDLE_DEFINE_EXPORTED_bool(enable_offload_pinned_memory,
                            true,
                            "enable infer used pinned memory ");
PADDLE_DEFINE_EXPORTED_int64(infer_offload_min_param_size,
                             0,
                             "enable infer offload min param size MB");

extern std::multiset<std::string> g_persistable_vars_;
namespace paddle {
namespace inference {
namespace analysis {

static void tensor2pinned(paddle::framework::LoDTensor *self_tensor) {
  if (!FLAGS_enable_offload_pinned_memory) {
    return;
  }
#if defined(PADDLE_WITH_CUDA)
  const size_t need_allocate_size = self_tensor->memory_size();
  void *data_ptr = self_tensor->data();
  void *host_buffer = NULL;
  cudaHostAlloc(reinterpret_cast<void **>(&host_buffer),
                need_allocate_size,
                cudaHostAllocDefault);
  memcpy(host_buffer, data_ptr, need_allocate_size);
  std::shared_ptr<memory::allocation::Allocation> holder =
      std::make_shared<memory::allocation::Allocation>(
          host_buffer, need_allocate_size, platform::CPUPlace());
  self_tensor->ResetHolderWithType(holder, self_tensor->dtype());
#endif
}

#ifdef PADDLE_WITH_ASCEND_CL
void IrParamsSyncAmongDevicesPass::CopyParamsToNpu(Argument *argument) {
  if (!argument->use_npu()) return;

  auto &graph = argument->main_graph();
  std::vector<std::string> repetitive_params;

  if (graph.Has(framework::ir::kRepetitiveParamAttr))
    repetitive_params = graph.Get<std::vector<std::string>>(
        framework::ir::kRepetitiveParamAttr);

  LOG(INFO) << "Sync params from CPU to NPU";

  PADDLE_ENFORCE_EQ(argument->npu_device_id_valid(),
                    true,
                    platform::errors::PreconditionNotMet(
                        "The npu_device_id field should be valid"));
  platform::Place place = platform::NPUPlace(argument->npu_device_id());
  auto *scope = argument->scope_ptr();
  std::vector<std::string> all_vars = scope->LocalVarNames();

  for (auto &var_name : all_vars) {
    auto *var = scope->FindLocalVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::PreconditionNotMet("The var should not be nullptr"));

    if (var->IsType<framework::LoDTensor>() ||
        var->IsType<framework::Tensor>()) {
      auto *t = var->GetMutable<framework::LoDTensor>();

      platform::CPUPlace cpu_place;
      framework::LoDTensor temp_tensor;
      temp_tensor.Resize(t->dims());
      temp_tensor.mutable_data<float>(cpu_place);

      paddle::framework::TensorCopySync(*t, cpu_place, &temp_tensor);
      t->clear();
      paddle::framework::TensorCopySync(temp_tensor, place, t);
    }
  }
}

#else

void IrParamsSyncAmongDevicesPass::CopyParamsToGpu(Argument *argument) {
  // The parameters are on the cpu, therefore, synchronization is not necessary.
  if (!argument->use_gpu()) return;

  auto &graph = argument->main_graph();
  std::vector<std::string> repetitive_params;

  if (graph.Has(framework::ir::kRepetitiveParamAttr))
    repetitive_params = graph.Get<std::vector<std::string>>(
        framework::ir::kRepetitiveParamAttr);

  LOG(INFO) << "Sync params from CPU to GPU";

  PADDLE_ENFORCE_EQ(argument->gpu_device_id_valid(),
                    true,
                    platform::errors::PreconditionNotMet(
                        "The gpu_device_id field should be valid"));
  platform::Place place = platform::CUDAPlace(argument->gpu_device_id());
  auto *scope = argument->scope_ptr();
  std::vector<std::string> all_vars = scope->LocalVarNames();
  LOG(INFO) << "scope=" << scope;
  // We get all the vars from local_scope instead of the ProgramDesc.
  // Because there exists the case that new parameter variables are not added to
  // the program in the analysis pass.
  bool reserve_cpu_weights = false;
  bool with_dynamic_shape = false;
  if (argument->Has("max_input_shape") && argument->Has("min_input_shape") &&
      argument->Has("optim_input_shape")) {
    with_dynamic_shape = (argument->max_input_shape().size() > 0 &&
                          argument->min_input_shape().size() > 0 &&
                          argument->optim_input_shape().size() > 0);
  }
  with_dynamic_shape =
      with_dynamic_shape || (argument->Has("tensorrt_tuned_dynamic_shape") &&
                             argument->tensorrt_tuned_dynamic_shape());
  if (with_dynamic_shape) {
    reserve_cpu_weights = true;
  }
  int total_persistable_var = 0;
  int offload_var = 0;
  size_t offload_size = 0;
  size_t offload_memory_size = FLAGS_infer_offload_min_param_size * 1024 * 1024;
  std::unordered_set<std::string> visited;
  for (auto *node : paddle::framework::ir::TopologySortOperations(graph)) {
    if (!node->IsOp()) continue;
    if (node->Op()->Type() == "feed" || node->Op()->Type() == "fetch") continue;
    for (auto *var_node : node->inputs) {
      if (!var_node->Var()->Persistable()) continue;
      auto var_name = var_node->Var()->Name();
      if (std::count(
              repetitive_params.begin(), repetitive_params.end(), var_name)) {
        if (!reserve_cpu_weights) {
          scope->EraseVars({var_name});
        }
        continue;
      }
      if (visited.count(var_name)) continue;
      visited.insert(var_name);
      auto *var = scope->FindLocalVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(var,
                              platform::errors::PreconditionNotMet(
                                  "The var should not be nullptr"));
      ++total_persistable_var;
      if (var->IsType<framework::LoDTensor>() ||
          var->IsType<framework::Tensor>()) {
        auto *t = var->GetMutable<framework::LoDTensor>();
        auto var_data_type = var_node->Var()->GetDataType();
        VLOG(3) << "var_name is " << var_name << ", data type is "
                << var_data_type << ", place is" << t->place()
                << ", IsInitialized is " << var->IsInitialized()
                << ", memory size=" << t->numel();
        if (FLAGS_enable_opt_infer_offload
            && t->memory_size() > offload_memory_size) {
          tensor2pinned(t);
          g_persistable_vars_.insert(var_name);
          if (FLAGS_enable_opt_infer_debug_mode) {
            VLOG(0) << "var_name is " << var_name << ", data type is "
                    << var_data_type << ", place is" << t->place()
                    << ", IsInitialized is " << var->IsInitialized()
                    << ", numel size=" << t->numel();
          }
          offload_size += t->memory_size();
          ++offload_var;
        } else {
          platform::CPUPlace cpu_place;
          framework::LoDTensor temp_tensor;
          temp_tensor.Resize(t->dims());
          paddle::framework::TensorCopySync(*t, cpu_place, &temp_tensor);
          t->clear();
          paddle::framework::TensorCopySync(temp_tensor, place, t);
        }
      }
    }
  }
  VLOG(0) << "total_persistable_var count=" << total_persistable_var
          << ", offload var count=" << offload_var
          << ", offload memory=" << offload_size / 1024.0 / 1024.0 << "MB";
}

#endif

void IrParamsSyncAmongDevicesPass::RunImpl(Argument *argument) {
  PADDLE_ENFORCE_EQ(
      argument->scope_valid(),
      true,
      platform::errors::PreconditionNotMet("The scope field should be valid"));

#ifdef PADDLE_WITH_ASCEND_CL
  if (!argument->use_npu_valid()) return;
  CopyParamsToNpu(argument);
#else
  if (!argument->use_gpu_valid()) return;
  CopyParamsToGpu(argument);
#endif
}

std::string IrParamsSyncAmongDevicesPass::repr() const {
  return "ir_params_sync_among_devices_pass";
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
