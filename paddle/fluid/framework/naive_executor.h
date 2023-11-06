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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {

/*
 * Simple, intuitive and effective. Only single thread is supported, and
 * currently designed for inference.
 */
class ProgramDesc;
class Scope;
class GarbageCollector;
class NaiveExecutor {
  struct OffLoadVarInfo {
    std::vector<std::string> persistable_inputs;
    size_t total_param_len = 0;
    void CopyInputs(const Scope* root,
                    const platform::Place& place,
                    Scope* scope) {
      total_param_len = 0;
      for (auto& name : persistable_inputs) {
        auto src_var = root->FindLocalVar(name);
        CHECK(src_var != nullptr) << "name=" << name << " is nullptr";
        auto& src_tensor = src_var->Get<framework::LoDTensor>();
        CHECK(src_tensor.IsInitialized())
            << "name=" << name << " not IsInitialized";
        auto dest_var = scope->FindLocalVar(name);
        CHECK(dest_var != nullptr) << "dest name=" << name << " is nullptr";
        auto* dest_tensor = dest_var->GetMutable<framework::LoDTensor>();
        paddle::framework::TensorCopy(src_tensor, place, dest_tensor);
        total_param_len += src_tensor.memory_size();
      }
    }
    void GCInputsVar(Scope* scope) {
      for (auto& name : persistable_inputs) {
        auto var = scope->FindLocalVar(name);
        if (var == nullptr) {
          continue;
        }
        var->GetMutable<LoDTensor>()->MoveMemoryHolder().reset();
      }
    }
  };

 public:
  explicit NaiveExecutor(const platform::Place& place) : place_(place) {}

  ~NaiveExecutor();

  // Create child scope.
  // Create variables.
  // @with_feed_fetch_ops: whether to work with the feed and fetch operators.
  void Prepare(Scope* scope,
               const ProgramDesc& program_desc,
               int block_id,
               bool with_feed_fetch_ops);
  // Create variables before head.
  // Create parameters if persistable is ture, or create the temporary variables
  // instead.
  void CreateVariables(const ProgramDesc& desc,
                       int block_id,
                       bool persistable,
                       Scope* scope);

  // Run all the operators.
  void Run();

  // Get an tensor to operating directly, without the need for feed_ops.
  LoDTensor* FindTensor(const std::string& name);

  Scope* scope() { return scope_; }

  void CleanFeedFetchOps();

  void ResetTrtOps(int num);
  void AddSkipVars(const std::vector<std::string>& skip_vars);
  void SetRunByExecutor(bool executor) {
    run_by_executor_ = executor;
  }

 protected:
  void CreateOps(const ProgramDesc& desc,
                 int block_id,
                 bool with_feed_fetch_ops);
  void RunDebug();
  void RunNormal();
  void RunOffLoad();

 private:
  const platform::Place place_;
  // Catch the required resource to avoid recreate.
  std::vector<std::unique_ptr<OperatorBase>> ops_;
  // op gc vars
  std::vector<std::string> skip_vars_;
  // gc vars
  std::unordered_map<const OperatorBase *, std::vector<std::string>> unused_vars_;
  // offload vars
  std::unordered_map<const OperatorBase*, OffLoadVarInfo> offload_vars_;
  // scope
  Scope* scope_;
  // root scope
  const Scope* root_scope_;
  // executor
  bool run_by_executor_ = false;
  // gc
  GarbageCollector *gc_ = nullptr;
};

}  // namespace framework
}  // namespace paddle
