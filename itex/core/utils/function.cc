/* Copyright (c) 2021-2022 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "itex/core/utils/function.h"

#include "itex/core/utils/tf_buffer.h"

namespace itex {

FunctionLibraryDefinition::FunctionDefAndOpRegistration::
    FunctionDefAndOpRegistration(const FunctionDef& fdef_in)
    : fdef(fdef_in) {}

FunctionLibraryDefinition::FunctionLibraryDefinition(const GraphDef& g_def)
    : function_defs_(g_def.library().function_size()) {
  TF_Buffer* g_buf = TF_NewBuffer();
  TF_ABORT_IF_ERROR(MessageToBuffer(g_def, g_buf));

  TF_Status* status = TF_NewStatus();
  func_ = TF_NewFunctionLibraryDefinition(g_buf, status);
  TF_DeleteBuffer(g_buf);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status))
      << " Error while creating FunctionLibraryDefinition";
  TF_DeleteStatus(status);

  // Initialization function_defs_ and func_grad_.
  const FunctionDefLibrary& def_lib = g_def.library();
  for (const auto& fdef : def_lib.function()) {
    // The latter function definition wins.
    auto& ptr = function_defs_[fdef.signature().name()];
    ptr.reset(new FunctionDefAndOpRegistration(fdef));
  }
  for (const auto& grad : def_lib.gradient()) {
    func_grad_[grad.function_name()] = grad.gradient_func();
  }
}

FunctionLibraryDefinition::~FunctionLibraryDefinition() {
  TF_DeleteFunctionLibraryDefinition(func_);
}

Status FunctionLibraryDefinition::LookUpOpDef(const std::string& op_type_name,
                                              OpDef* op_def) const {
  TF_Buffer* buf = TF_NewBuffer();
  TF_Status* tf_status = TF_NewStatus();
  TF_LookUpOpDef(func_, op_type_name.c_str(), buf, tf_status);
  TF_ABORT_IF_ERROR(BufferToMessage(buf, *op_def));
  TF_DeleteBuffer(buf);
  Status status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  return status;
}

const FunctionDef* FunctionLibraryDefinition::Find(const string& func) const {
  tf_shared_lock l(&mu_);
  auto result = FindHelper(func);
  if (result) {
    return &result->fdef;
  } else {
    return nullptr;
  }
}

std::shared_ptr<FunctionLibraryDefinition::FunctionDefAndOpRegistration>
FunctionLibraryDefinition::FindHelper(const string& func) const {
  auto iter = function_defs_.find(func);
  if (iter == function_defs_.end()) {
    return nullptr;
  } else {
    return iter->second;
  }
}

}  // namespace itex
