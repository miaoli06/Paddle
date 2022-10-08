/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/device_worker.h"

#include <chrono>
#include "paddle/fluid/framework/convert_utils.h"
#ifdef PADDLE_WITH_BOX_PS
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#endif
DECLARE_bool(lineid_have_extend_info);
DECLARE_bool(dump_filed_same_as_aibox);

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {

class Scope;

void DeviceWorker::SetRootScope(Scope* root_scope) { root_scope_ = root_scope; }

void DeviceWorker::SetDataFeed(DataFeed* data_feed) {
  device_reader_ = data_feed;
}
template<class... ARGS>
void format_string_append(std::string* str,
    const char* fmt, ARGS && ... args) { // use VA_ARGS may be better ?
    int len = snprintf(NULL, 0, fmt, args...);
    PADDLE_ENFORCE(len >= 0, "format args length error");
    size_t oldlen = str->length();
    str->resize(oldlen + len + 1);
    PADDLE_ENFORCE(snprintf(&(*str)[oldlen], (size_t)len + 1, fmt, args...) == len);
    str->resize(oldlen + len);
}
template <typename T>
void PrintLodTensorType(Tensor* tensor,
                        int64_t start,
                        int64_t end,
                        std::string& out_val,
                        char separator = ',',
                        bool need_leading_separator = true) {
  if (start >= end) return;
  const T *ptr = tensor->data<T>();
  if (!need_leading_separator) {
    out_val += std::to_string(ptr[start]);
    start++;
  }
  for (int64_t i = start; i < end; i++) {
    out_val += separator;
    out_val += std::to_string(ptr[i]);
  }
}

#define FLOAT_EPS 1e-8
#define MAX_NUM_BUFF_SIZE 40
void PrintLodTensorFloatType(Tensor* tensor,
                               int64_t start,
                               int64_t end,
                               std::string& out_val,
                               char separator,
                               bool need_leading_separator) {
  if (start >= end) {
    return;
  }

  const float *ptr = tensor->data<float>();
  if (!need_leading_separator) {
    format_string_append(&out_val, "%.9g", ptr[start]);
    ++start;
  }
  for (int64_t i = start; i < end; i++) {
    const float &n = ptr[i];
    if (n > -FLOAT_EPS && n < FLOAT_EPS) {
      format_string_append(&out_val, "%c0", separator);
    } else {
      format_string_append(&out_val, "%c%.9g", separator, n);
    }
  }
}
void PrintLodTensorIntType(Tensor* tensor,
                           int64_t start,
                           int64_t end,
                           std::string& out_val,
                           char separator = ',',
                           bool need_leading_separator = true) {
  if (start >= end){
    return;
  }
  const int64_t *ptr = tensor->data<int64_t>();
  if (!need_leading_separator) {
    format_string_append(&out_val, "%lu", static_cast<uint64_t>(ptr[start]));
    start++;
  }
  for (int64_t i = start; i < end; i++) {
    format_string_append(&out_val, "%c%lu", separator, static_cast<uint64_t>(ptr[i]));
  }
}
std::string PrintLodTensor(Tensor* tensor,
                           int64_t start,
                           int64_t end,
                           char separator,
                           bool need_leading_separator) {
  std::string out_val;
  PrintLodTensor(tensor, start, end,
      out_val, separator, need_leading_separator);
  return out_val;
}
void PrintLodTensor(Tensor* tensor,
                    int64_t start,
                    int64_t end,
                    std::string& out_val,
                    char separator,
                    bool need_leading_separator) {
  auto dtype = framework::TransToProtoVarType(tensor->dtype());
  if (dtype == proto::VarType::FP32) {
    PrintLodTensorFloatType(
        tensor, start, end, out_val, separator, need_leading_separator);
  } else if (dtype == proto::VarType::INT64) {
    PrintLodTensorIntType(
        tensor, start, end, out_val, separator, need_leading_separator);
  } else if (dtype == proto::VarType::FP64) {
    PrintLodTensorType<double>(
        tensor, start, end, out_val, separator, need_leading_separator);
  } else {
    out_val += "unsupported type";
  }
}
std::pair<int64_t, int64_t> GetTensorBound(LoDTensor* tensor, int index) {
  auto& dims = tensor->dims();
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    return {lod[index] * dims[1], lod[index + 1] * dims[1]};
  } else {
    return {index * dims[1], (index + 1) * dims[1]};
  }
}
bool CheckValidOutput(LoDTensor* tensor, size_t batch_size) {
  auto& dims = tensor->dims();
  if (dims.size() != 2) return false;
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    if (lod.size() != batch_size + 1) {
      return false;
    }
  } else {
    if (dims[0] != static_cast<int>(batch_size)) {
      return false;
    }
  }
  return true;
}

void DeviceWorker::DumpParam(const Scope& scope, const int batch_id) {
  std::ostringstream os;
  for (auto& param : *dump_param_) {
    os.str("");
    Variable* var = scope.FindVar(param);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    framework::LoDTensor cpu_tensor;
    if (platform::is_gpu_place(tensor->place())) {
      TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
      tensor = &cpu_tensor;
    }
    int64_t len = tensor->numel();
    os << "(" << batch_id << "," << param << ")"
       << PrintLodTensor(tensor, 0, len);
    writer_ << os.str();
  }
}

void DeviceWorker::InitRandomDumpConfig(const TrainerDesc& desc) {
  bool is_dump_in_simple_mode = desc.is_dump_in_simple_mode();
  if (is_dump_in_simple_mode) {
    dump_mode_ = 3;
    return;
  }
  bool enable_random_dump = desc.enable_random_dump();
  if (!enable_random_dump) {
    dump_mode_ = 0;
  } else {
    if (desc.random_with_lineid()) {
      dump_mode_ = 1;
    } else {
      dump_mode_ = 2;
    }
  }
  dump_interval_ = desc.dump_interval();
}

void DeviceWorker::DumpField(const Scope& scope,
                             int dump_mode,
                             int dump_interval) {  // dump_mode: 0: no random,
                                                   // 1: random with insid hash,
                                                   // 2: random with random
  // 3: simple mode using multi-threads, for gpugraphps-mode
  auto start1 = std::chrono::steady_clock::now();

  size_t batch_size = device_reader_->GetCurBatchSize();
  auto& ins_id_vec = device_reader_->GetInsIdVec();
  auto& ins_content_vec = device_reader_->GetInsContentVec();
  if (dump_mode_ == 3) {
    batch_size = std::string::npos;
    bool has_valid_batch = false;
    for (auto& field : *dump_fields_) {
      Variable* var = scope.FindVar(field);
      if (var == nullptr) {
        VLOG(0) << "Note: field[" << field
                << "] cannot be find in scope, so it was skipped.";
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (!tensor->IsInitialized()) {
        VLOG(0) << "Note: field[" << field
                << "] is not initialized, so it was skipped.";
        continue;
      }
      auto& dims = tensor->dims();
      if (dims.size() == 2 && dims[0] > 0) {
        batch_size = std::min(batch_size, static_cast<size_t>(dims[0]));
        has_valid_batch = true;
      }
    }
    if (!has_valid_batch) return;
  } else if (ins_id_vec.size() > 0) {
    batch_size = ins_id_vec.size();
  }
  std::vector<std::string> ars(batch_size);
  if (dump_mode_ == 3) {
    if (dump_fields_ == NULL || (*dump_fields_).size() == 0) {
      return;
    }
    auto set_output_str = [&, this](
                              size_t begin, size_t end, LoDTensor* tensor) {
      std::pair<int64_t, int64_t> bound;
      auto& dims = tensor->dims();
      for (size_t i = begin; i < end; ++i) {
        bound = {i * dims[1], (i + 1) * dims[1]};
        // auto bound = GetTensorBound(tensor, i);
        if (ars[i].size() > 0) ars[i] += "\t";
        PrintLodTensor(tensor, bound.first, bound.second, ars[i], ' ', false);
      }
    };
    std::vector<std::thread> threads(tensor_iterator_thread_num);
    for (auto& field : *dump_fields_) {
      Variable* var = scope.FindVar(field);
      if (var == nullptr) {
        VLOG(0) << "Note: field[" << field
                << "] cannot be find in scope, so it was skipped.";
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (!tensor->IsInitialized()) {
        VLOG(0) << "Note: field[" << field
                << "] is not initialized, so it was skipped.";
        continue;
      }
      framework::LoDTensor cpu_tensor;
      if (platform::is_gpu_place(tensor->place())) {
        TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
        cpu_tensor.set_lod(tensor->lod());
        tensor = &cpu_tensor;
      }
      auto& dims = tensor->dims();
      if (dims.size() != 2 || dims[0] <= 0) {
        VLOG(0) << "Note: field[" << field
                << "] cannot pass check, so it was "
                   "skipped. Maybe the dimension is "
                   "wrong ";
        VLOG(0) << dims.size() << " " << dims[0] << " * " << dims[1];
        continue;
      }
      size_t acutal_thread_num =
          std::min((size_t)batch_size, tensor_iterator_thread_num);
      for (size_t i = 0; i < acutal_thread_num; i++) {
        size_t average_size = batch_size / acutal_thread_num;
        size_t begin =
            average_size * i + std::min(batch_size % acutal_thread_num, i);
        size_t end =
            begin + average_size + (i < batch_size % acutal_thread_num ? 1 : 0);
        threads[i] = std::thread(set_output_str, begin, end, tensor);
      }
      for (size_t i = 0; i < acutal_thread_num; i++) threads[i].join();
    }
    auto end1 = std::chrono::steady_clock::now();
    auto tt =
        std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    VLOG(1) << "writing a batch takes " << tt.count() << " us";

    size_t acutal_thread_num =
        std::min((size_t)batch_size, tensor_iterator_thread_num);
    for (size_t i = 0; i < acutal_thread_num; i++) {
      size_t average_size = batch_size / acutal_thread_num;
      size_t begin =
          average_size * i + std::min(batch_size % acutal_thread_num, i);
      size_t end =
          begin + average_size + (i < batch_size % acutal_thread_num ? 1 : 0);
      for (size_t j = begin + 1; j < end; j++) {
        if (ars[begin].size() > 0 && ars[j].size() > 0) ars[begin] += "\n";
        ars[begin] += ars[j];
      }
      if (ars[begin].size() > 0) writer_ << ars[begin];
    }
    return;
  }
  std::vector<bool> hit(batch_size, false);
  std::default_random_engine engine(0);
  std::uniform_int_distribution<size_t> dist(0U, INT_MAX);
  for (size_t i = 0; i < batch_size; i++) {
    size_t r = 0;
    if (dump_mode == 1) {
      r = XXH64(ins_id_vec[i].data(), ins_id_vec[i].length(), 0);
    } else if (dump_mode == 2) {
      r = dist(engine);
    }
    if (r % dump_interval != 0) {
      continue;
    }
    hit[i] = true;
  }  // dump_mode = 0
  for (size_t i = 0; i < ins_id_vec.size(); i++) {
    if (!hit[i]) {
      continue;
    }
    ars[i] += ins_id_vec[i];
    ars[i] += "\t" + ins_content_vec[i];
  }
  for (auto& field : *dump_fields_) {
    Variable* var = scope.FindVar(field);
    if (var == nullptr) {
      VLOG(0) << "Note: field[" << field
              << "] cannot be find in scope, so it was skipped.";
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (!tensor->IsInitialized()) {
      VLOG(0) << "Note: field[" << field
              << "] is not initialized, so it was skipped.";
      continue;
    }
    framework::LoDTensor cpu_tensor;
    if (platform::is_gpu_place(tensor->place())) {
      TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
      cpu_tensor.set_lod(tensor->lod());
      tensor = &cpu_tensor;
    }
    if (!CheckValidOutput(tensor, batch_size)) {
      VLOG(0) << "Note: field[" << field
              << "] cannot pass check, so it was "
                 "skipped. Maybe the dimension is "
                 "wrong ";
      continue;
    }
    for (size_t i = 0; i < batch_size; ++i) {
      if (!hit[i]) {
        continue;
      }
      auto bound = GetTensorBound(tensor, i);
      ars[i] += "\t" + field + ":" + std::to_string(bound.second - bound.first);
      ars[i] += PrintLodTensor(tensor, bound.first, bound.second);
    }
  }

  // #pragma omp parallel for
  for (size_t i = 0; i < ars.size(); i++) {
    if (ars[i].length() == 0) {
      continue;
    }
    writer_ << ars[i];
  }
}
inline void GetLodBound(const LoD& lod, const int64_t &dim, const int &index,
    std::pair<int64_t, int64_t> *bound) {
  if (lod.size() != 0) {
    bound->first = lod[0][index] * dim;
    bound->second = lod[0][index + 1] * dim;
  } else {
    bound->first = index * dim;
    bound->second = (index + 1) * dim;
  }
}
template<typename T, typename C>
void PrintLodTensorFmtType(Tensor* tensor,
                           const int64_t &start,
                           const int64_t &end,
                           const char *fmt,
                           std::string* out_val) {
  if (start >= end){
    return;
  }
  const T *ptr = tensor->data<T>();
  for (int64_t i = start; i < end; i++) {
    format_string_append(out_val, fmt, static_cast<C>(ptr[i]));
  }
}
void PrintLodTensor(Tensor* tensor, const int64_t &start, const int64_t &end, std::string* out) {
  auto dtype = framework::TransToProtoVarType(tensor->dtype());
  if (dtype == proto::VarType::FP32) {
    PrintLodTensorFmtType<float, float>(tensor, start, end, ":%.9g", out);
  } else if (dtype == proto::VarType::INT64) {
    PrintLodTensorFmtType<int64_t, uint64_t>(tensor, start, end, ":%lu", out);
  } else if (dtype == proto::VarType::FP64) {
    PrintLodTensorFmtType<double, double>(tensor, start, end, ":%.9g", out);
  } else if (dtype == proto::VarType::INT32) {
    PrintLodTensorFmtType<int, int>(tensor, start, end, ":%d", out);
  } else if (dtype == proto::VarType::INT16) {
    PrintLodTensorFmtType<int16_t, int16_t>(tensor, start, end, ":%d", out);
  } else {
    out->append("unsupported type");
  }
}
void DeviceWorker::DumpParamBoxPS(const Scope& scope, const int batch_id) {
  size_t field_num = dump_fields_->size();
  std::vector<std::string> ars(field_num);

  // thread process fields
#ifdef PADDLE_WITH_BOX_PS
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  box_ptr->ExecuteFunc(platform::CPUPlace(),
#else
  parallel_run_dynamic(
#endif
      field_num, [this, &scope, batch_id, &ars](const size_t &id) {
    auto &name = (*dump_param_)[id];
    Variable* var = scope.FindVar(name);
    if (var == nullptr) {
      return;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    framework::LoDTensor cpu_tensor;
    if (!platform::is_cpu_place(tensor->place())) {
      TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
      tensor = &cpu_tensor;
    }
    auto &s = ars[id];
    format_string_append(&s, "(%d,%s)", batch_id, name.c_str());
    int64_t len = tensor->numel();
    PrintLodTensor(tensor, 0, len, &s);
  });

  // write to channel
  for (auto &s : ars) {
    if (s.empty()) {
      continue;
    }
    writer_ << s;
  }
}
void DeviceWorker::DumpFieldBoxPS(const Scope& scope, int dump_mode,
                             int dump_interval) {  // dump_mode: 0: no random,
                                                   // 1: random with insid hash,
                                                   // 2: random with random
                                                   // number
  size_t batch_size = device_reader_->GetCurBatchSize();
  size_t field_num = dump_fields_->size();
  std::vector<int64_t> dims(field_num, 0);
  std::vector<framework::LoDTensor> cpu_tensors(field_num);
  std::vector<const LoD *> lods(field_num, nullptr);
  // copy fields
#ifdef PADDLE_WITH_BOX_PS
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  box_ptr->ExecuteFunc(platform::CPUPlace(),
#else
  parallel_run_dynamic(
#endif
      field_num, [this, &dims, &cpu_tensors, &lods, &scope, batch_size](const size_t &i) {
    auto& field = (*dump_fields_)[i];
    Variable* var = scope.FindVar(field);
    if (var == nullptr) {
      VLOG(3) << "Note: field[" << field
              << "] cannot be find in scope, so it was skipped.";
      return;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (!tensor->IsInitialized()) {
      VLOG(3) << "Note: field[" << field
              << "] is not initialized, so it was skipped.";
      return;
    }
    if (!CheckValidOutput(tensor, batch_size)) {
//      VLOG(0) << "Note: field[" << field << "] cannot pass check, so it was "
//                                            "skipped. Maybe the dimension is "
//                                            "wrong ";
      return;
    }
    dims[i] = tensor->dims()[1];
    lods[i] = (&tensor->lod());
    if (!platform::is_cpu_place(tensor->place())) {
      TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensors[i]);
    } else {
      cpu_tensors[i] = *tensor;
    }
  });

  // dump data
  std::default_random_engine engine(0);
  std::uniform_int_distribution<size_t> dist(0U, INT_MAX);
  // need dump check
  auto need_dump_func = [this, &dist, &engine, dump_mode, dump_interval](
      const std::string &lineid) {
    size_t r = 0;
    if (dump_mode == 1) {
      r = XXH64(lineid.data(), lineid.length(), 0);
    } else if (dump_mode == 2) {
      r = dist(engine);
    }
    if (r % dump_interval != 0) {
      return false;
    }
    return true;
  };

  std::atomic<size_t> line_cnt{0};
  std::atomic<size_t> num_cnt{0};

  std::vector<std::string> ars(batch_size);
#ifdef PADDLE_WITH_BOX_PS
  box_ptr->ExecuteFunc(platform::CPUPlace(),
#else
  // dump data
  parallel_run_dynamic(
#endif
      batch_size, [this, &ars, &dims, &cpu_tensors,
         &lods, &need_dump_func, field_num, &line_cnt, &num_cnt](const size_t &i) {
    const std::string& lineid = device_reader_->GetLineId(i);
    if (!need_dump_func(lineid)) {
      return;
    }

    ++line_cnt;

    thread_local std::pair<int64_t, int64_t> bound;
    auto &s = ars[i];
    size_t pos = 0;
    if (FLAGS_lineid_have_extend_info) {
      pos = lineid.find(" ");
      if (pos != std::string::npos) {
        s.append(&lineid[0], pos);
      } else {
        s.append(lineid);
      }
    } else {
      s.append(lineid);
    }

    size_t num = 0;
    for (size_t k = 0; k < field_num; ++k) {
      auto &lod = lods[k];
      if (lod == nullptr) {
        continue;
      }
      auto &field = (*dump_fields_)[k];
      s.append("\t", 1);
      GetLodBound(*lod, dims[k], i, &bound);

      num += (bound.second - bound.first);
      if (FLAGS_dump_filed_same_as_aibox) {
        size_t ext_pos = field.find(".");
        if (ext_pos != std::string::npos) {
          s.append(&field[0], ext_pos);
        } else {
          s.append(field);
        }
      } else {
        format_string_append(&s, "%s:%ld", field.c_str(), bound.second - bound.first);
      }
      PrintLodTensor(&cpu_tensors[k], bound.first, bound.second, &s);
    }
    num_cnt += num;

    // append extends tag info
    if (pos > 0) {
      s.append("\t", 1);
      s.append(&lineid[pos + 1], lineid.length() - pos - 1);
    }
  });
  // write to channel
  for (auto &s : ars) {
    if (s.empty()) {
      continue;
    }
    writer_ << s;
  }
}

}  // namespace framework
}  // namespace paddle
