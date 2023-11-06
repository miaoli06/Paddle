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

#include "paddle/fluid/framework/naive_executor.h"

#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/denormal.h"
#include "paddle/fluid/platform/timer.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#if PADDLE_WITH_TENSORRT
#include "paddle/fluid/operators/tensorrt/tensorrt_engine_op.h"
#endif
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/string/string_helper.h"

PADDLE_DEFINE_EXPORTED_bool(enable_opt_infer_gc_var,
                            false,
                            "enable opt infer gc var");
PADDLE_DEFINE_EXPORTED_bool(enable_opt_infer_offload,
                            false,
                            "enable opt infer offload");
PADDLE_DEFINE_EXPORTED_bool(enable_opt_infer_debug_mode,
                            false,
                            "enable opt infer offload");

std::multiset<std::string> g_persistable_vars_;

namespace paddle {
namespace framework {
void NaiveExecutor::Prepare(Scope *scope,
                            const ProgramDesc &program_desc,
                            int block_id,
                            bool with_feed_fetch_ops) {
  if (!scope) {
    scope_ = new framework::Scope;
  } else {
    scope_ = scope;
  }
  root_scope_ = scope;
  while (root_scope_->parent()) {
    root_scope_ = root_scope_->parent();
  }

  gc_ = nullptr;
  int64_t max_memory_size = GetEagerDeletionThreshold();
  if (FLAGS_enable_opt_infer_gc_var && max_memory_size >= 0) {
    auto gc = CreateGarbageCollector(place_, max_memory_size);
    gc_ = gc.release();
  }

  CreateOps(program_desc, block_id, with_feed_fetch_ops);
  // debug print info
  if (FLAGS_enable_opt_infer_debug_mode) {
    VLOG(0) << "NaiveExecutor init with scope " << scope
            << ", root scope=" << root_scope_
            << ", op count=" << ops_.size()
            << ", enable gc=" << FLAGS_enable_opt_infer_gc_var
            << ", gc op count=" << unused_vars_.size()
            << ", enable offload=" << FLAGS_enable_opt_infer_offload
            << ", offload op count=" << offload_vars_.size();
  }
}
inline void GetOpParam(const std::unique_ptr<OperatorBase> &op,
                       const Scope *scope,
                       size_t *in_param,
                       size_t *out_param) {
  for (auto &obj : op->Inputs()) {
    for (auto &name : obj.second) {
      auto var = scope->FindVar(name);
      if (var == nullptr) {
        continue;
      }
      if (var->IsType<LoDTensor>()) {
        auto gc_tensor = var->GetMutable<LoDTensor>();
        *in_param += gc_tensor->memory_size();
      } else if (var->IsType<phi::SelectedRows>()) {
        auto gc_tensor = var->GetMutable<phi::SelectedRows>()->mutable_value();
        *in_param += gc_tensor->memory_size();
      } else if (var->IsType<LoDTensorArray>()) {
        auto *tensor_arr = var->GetMutable<LoDTensorArray>();
        for (auto &t : *tensor_arr) {
          *in_param += t.memory_size();
        }
      }
    }
  }
  for (auto &obj : op->Outputs()) {
    for (auto &name : obj.second) {
      auto var = scope->FindVar(name);
      if (var == nullptr) {
        continue;
      }
      if (var->IsType<LoDTensor>()) {
        auto gc_tensor = var->GetMutable<LoDTensor>();
        *out_param += gc_tensor->memory_size();
      } else if (var->IsType<phi::SelectedRows>()) {
        auto gc_tensor = var->GetMutable<phi::SelectedRows>()->mutable_value();
        *out_param += gc_tensor->memory_size();
      } else if (var->IsType<LoDTensorArray>()) {
        auto *tensor_arr = var->GetMutable<LoDTensorArray>();
        for (auto &t : *tensor_arr) {
          *out_param += t.memory_size();
        }
      }
    }
  }
}
void NaiveExecutor::RunDebug() {
  platform::Timer tm;
  tm.Start();
  platform::Timer cp_tm;
  platform::Timer gc_tm;
  platform::ScopedFlushDenormal flush;

  size_t copy_data_len = 0;
  size_t total_in_param = 0;
  size_t total_out_param = 0;

  const platform::DeviceContext *dev_ctx =
      platform::DeviceContextPool::Instance().Get(place_);
  for (auto &op : ops_) {
    VLOG(4) << std::this_thread::get_id() << " run "
            << op->DebugStringEx(scope_) << " on scope " << scope_;
    auto it = offload_vars_.find(op.get());
    if (it != offload_vars_.end()) {
      if (op->Type() != "while") {
        cp_tm.Resume();
        it->second.CopyInputs(root_scope_, place_, scope_);
        dev_ctx->Wait();
        cp_tm.Pause();
      }
    }
    op->SetIsCalledByExecutor(run_by_executor_);
    op->Run(*scope_, place_);

    size_t in_param = 0;
    size_t out_param = 0;
    GetOpParam(op, scope_, &in_param, &out_param);

    if (it != offload_vars_.end()) {
      dev_ctx->Wait();
      gc_tm.Resume();
      it->second.GCInputsVar(scope_);
      gc_tm.Pause();
      copy_data_len += it->second.total_param_len;
    }
    // gc input and output var
    if (gc_) {
      gc_tm.Resume();
      DeleteUnusedTensors(*scope_, op.get(), unused_vars_, gc_);
      gc_tm.Pause();
    }
    VLOG(1) << "op name=" << op->Type() << ", input len=" << in_param
            << ", output len=" << out_param;
    total_in_param += in_param;
    total_out_param += out_param;
  }
  dev_ctx->Wait();
  tm.Pause();

  VLOG(0) << "exec total span=" << tm.ElapsedSec()
          << ", copy span=" << cp_tm.ElapsedSec()
          << ", gc span=" << gc_tm.ElapsedSec()
          << ", copy memory=" << copy_data_len / 1024.0 / 1024.0 << "MB"
          << ", total input=" << total_in_param / 1024.0 / 1024.0 << "MB"
          << ", total output=" << total_out_param / 1024.0 / 1024.0 << "MB";
}
void NaiveExecutor::RunNormal() {
  platform::ScopedFlushDenormal flush;
  // old executor
  for (auto &op : ops_) {
    VLOG(4) << std::this_thread::get_id() << " run "
            << op->DebugStringEx(scope_) << " on scope " << scope_;
    op->SetIsCalledByExecutor(run_by_executor_);
    op->Run(*scope_, place_);
    // gc input and output var
    if (gc_) {
      DeleteUnusedTensors(*scope_, op.get(), unused_vars_, gc_);
    }
  }
}
void NaiveExecutor::RunOffLoad() {
  platform::ScopedFlushDenormal flush;
  for (auto &op : ops_) {
    VLOG(4) << std::this_thread::get_id() << " run "
            << op->DebugStringEx(scope_) << " on scope " << scope_;
    auto it = offload_vars_.find(op.get());
    if (op->Type() != "while" && it != offload_vars_.end()) {
      it->second.CopyInputs(root_scope_, place_, scope_);
    }
    op->SetIsCalledByExecutor(run_by_executor_);
    op->Run(*scope_, place_);
    if (it != offload_vars_.end()) {
      it->second.GCInputsVar(scope_);
    }
    // gc input and output var
    if (gc_) {
      DeleteUnusedTensors(*scope_, op.get(), unused_vars_, gc_);
    }
  }
}
void NaiveExecutor::Run() {
#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
  platform::RegisterModelLayout(ops_, place_);
#endif
  if (FLAGS_enable_opt_infer_debug_mode) {
    RunDebug();
  } else if (FLAGS_enable_opt_infer_offload) {
    RunOffLoad();
  } else {
    RunNormal();
  }
  if (!run_by_executor_) {
    return;
  }
  if (gc_) {
    gc_->DirectClearCallback([this](){
      scope_->DropKids();
    });
  } else {
    platform::DeviceContextPool::Instance().Get(place_)->Wait();
    scope_->DropKids();
  }
}
void NaiveExecutor::CreateVariables(const ProgramDesc &desc,
                                    int block_id,
                                    bool persistable,
                                    Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(scope,
                          platform::errors::InvalidArgument(
                              "The Scope to hold variables is nullptr."));

  auto &global_block = desc.Block(block_id);

  const auto *anc = scope;
  PADDLE_ENFORCE_NE(
      anc->parent(),
      anc,
      platform::errors::InvalidArgument("Input scope should be child scope."));
  while (anc->parent()) {
    anc = anc->parent();
  }
  VLOG(1) << "execute root scope=" << root_scope_;

  int num_vars = 0;
  for (auto &var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }
    num_vars++;

    if (persistable == var->Persistable()) {
      if (persistable) {
        if (!anc->FindVar(var->Name())) {
          auto *ptr = const_cast<Scope *>(anc)->Var(var->Name());
          VLOG(3) << scope << " Create persistable variable " << var->Name()
                  << ", which pointer is " << ptr;
          InitializeVariable(ptr, var->GetType());
        }
      } else {
        auto *ptr = const_cast<Scope *>(scope)->Var(var->Name());
        VLOG(3) << scope << " Create variable " << var->Name()
                << ", which pointer is " << ptr;
        InitializeVariable(ptr, var->GetType());
      }
    }
  }
  VLOG(1) << "naive executor create " << num_vars << " vars";
}
void NaiveExecutor::CreateOps(const ProgramDesc &desc,
                              int block_id,
                              bool with_feed_fetch_ops) {
  auto &global_block = desc.Block(block_id);
  // create op
  auto ops_desc = global_block.AllOps();
  for (const auto &op_desc : desc.Block(block_id).AllOps()) {
    // gc var
    if (gc_) {
      if (op_desc->Type() == "feed" || op_desc->Type() == "fetch") {
        for (auto &o : op_desc->Inputs()) {
          skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
        }
        for (auto &o : op_desc->Outputs()) {
          skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
        }
      }
    }
    if (!with_feed_fetch_ops &&
        (op_desc->Type() == "feed" || op_desc->Type() == "fetch")) {
      LOG(INFO) << "---  skip [" << op_desc->Input("X")[0] << "], "
                << op_desc->Type() << " -> " << op_desc->Output("Out")[0];
      continue;
    }
    ops_.emplace_back(OpRegistry::CreateOp(*op_desc));
    // offload
    if (FLAGS_enable_opt_infer_offload) {
      auto &op = ops_.back();
      // offload
      for (auto &o : op->Inputs()) {
        for (auto &name : o.second) {
          if (g_persistable_vars_.find(name) == g_persistable_vars_.end()) {
            continue;
          }
          auto dest_var = scope_->Var(name);  // init local var
          CHECK(dest_var != nullptr);
          offload_vars_[op.get()].persistable_inputs.push_back(name);
        }
      }
    }
  }
  if (!gc_) {
    return;
  }
  // get used
  unused_vars_ = GetUnusedVars(global_block, ops_, skip_vars_);
}

LoDTensor *NaiveExecutor::FindTensor(const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(scope_,
                          platform::errors::PreconditionNotMet(
                              "Need to init scope in NaiveExecutor firstly."));
  auto *var = scope_->FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(
      var,
      platform::errors::NotFound("No variable [%s] in current scope.", name));
  auto *tensor = const_cast<LoDTensor *>(&var->Get<LoDTensor>());
  return tensor;
}

void NaiveExecutor::CleanFeedFetchOps() {
  std::vector<std::unique_ptr<OperatorBase>> ops;
  for (auto &op : ops_) {
    if (op->Type() != "feed" && op->Type() != "fetch") {
      ops.emplace_back(std::move(op));
    }
  }
  ops_.swap(ops);
}
void NaiveExecutor::AddSkipVars(const std::vector<std::string> &skip_vars) {
  if (skip_vars.empty()) {
    return;
  }
  skip_vars_.insert(skip_vars_.end(), skip_vars.begin(), skip_vars.end());
}

NaiveExecutor::~NaiveExecutor() {
#ifdef PADDLE_WITH_MKLDNN
  // Clear mkl-dnn cache,
  // this is needed to have mkl-dnn unit tests working
  platform::ClearMKLDNNCache(place_, this);
#endif
  if (gc_) {
    delete gc_;
    gc_ = nullptr;
  }
}

void NaiveExecutor::ResetTrtOps(int num) {
#if PADDLE_WITH_TENSORRT
  for (auto &op : ops_) {
    if (op->Type() == "tensorrt_engine") {
      operators::TensorRTEngineOp *trtop =
          dynamic_cast<operators::TensorRTEngineOp *>(op.get());
      if (!trtop) return;
      std::string engine_key = trtop->Attr<std::string>("engine_key");
      int engine_predictor_id = trtop->Attr<int>("predictor_id");
      std::string engine_name =
          engine_key + std::to_string(engine_predictor_id);
      operators::TensorRTEngine *trt_engine = nullptr;
      // can't get trt engine if int8 calibration table data process.
      if (paddle::inference::Singleton<
              inference::tensorrt::TRTEngineManager>::Global()
              .Has(engine_name)) {
        trt_engine = paddle::inference::Singleton<
                         inference::tensorrt::TRTEngineManager>::Global()
                         .Get(engine_name);
      }
      if (trt_engine && trt_engine->with_dynamic_shape()) {
        LOG(INFO) << "rebuild trt engine, this may cost a lot of time!";
        trt_engine->ResetContext();
        trt_engine->ClearTensorMap();
        trt_engine->SetProfileNum(num);
        auto *anc = scope_->parent();
        while (anc && anc->parent()) {
          anc = anc->parent();
        }
        if (anc == nullptr) {
          anc = scope_;
        }
        trtop->PrepareTRTEngine(*anc, trt_engine);
      }
    }
  }
#endif
}

}  // namespace framework
}  // namespace paddle
