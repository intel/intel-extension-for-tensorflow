/* Copyright (c) 2022 Intel Corporation

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

#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
  using namespace tensorflow;       // NOLINT(build/namespaces)
  using namespace tensorflow::ops;  // NOLINT(build/namespaces)

  TF_Status* status = TF_NewStatus();
#ifdef ITEX_CPU_CC
  string xpu_lib_path = "libitex_cpu_cc.so";
#else
  string xpu_lib_path = "libitex_gpu_cc.so";
#endif
  TF_LoadPluggableDeviceLibrary(xpu_lib_path.c_str(), status);
  TF_Code code = TF_GetCode(status);
  if (code == TF_OK) {
    LOG(INFO) << "intel-extension-for-tensorflow load successfully!";
  } else {
    string status_msg(TF_Message(status));
    LOG(FATAL)
        << "Could not load intel-extension-for-tensorflow, please check! "
        << status_msg;
  }

  Scope root = Scope::NewRootScope();
  auto X = Variable(root, {5, 2}, DataType::DT_FLOAT);
  auto assign_x =
      Assign(root, X, RandomNormal(root, {5, 2}, DataType::DT_FLOAT));
  auto Y = Variable(root, {2, 3}, DataType::DT_FLOAT);
  auto assign_y =
      Assign(root, Y, RandomNormal(root, {2, 3}, DataType::DT_FLOAT));
  auto Z = Const(root, 2.f, {5, 3});
  auto V = MatMul(root, assign_x, assign_y);
  auto VZ = Add(root, V, Z);

  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch VZ
  TF_CHECK_OK(session.Run({VZ}, &outputs));
  LOG(INFO) << "Output:\n" << outputs[0].matrix<float>();
  return 0;
}
