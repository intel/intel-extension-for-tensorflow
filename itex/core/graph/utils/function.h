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

#ifndef ITEX_CORE_GRAPH_UTILS_FUNCTION_H_
#define ITEX_CORE_GRAPH_UTILS_FUNCTION_H_

#include <memory>
#include <string>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/gtl/flatmap.h"
#include "itex/core/utils/mutex.h"
#include "itex/core/utils/status.h"
#include "protos/graph.pb.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/grappler/grappler.h"

namespace itex {
namespace graph {

class FunctionLibraryDefinition {
 public:
  explicit FunctionLibraryDefinition(const GraphDef& g_def);
  ~FunctionLibraryDefinition();
  Status LookUpOpDef(const std::string& op_type_name, OpDef* op_def) const;
  const FunctionDef* Find(const std::string& func) const TF_LOCKS_EXCLUDED(mu_);

 private:
  TF_FunctionLibraryDefinition* func_;

  struct FunctionDefAndOpRegistration {
    explicit FunctionDefAndOpRegistration(const FunctionDef& fdef_in);

    const FunctionDef fdef;
  };

  std::shared_ptr<FunctionDefAndOpRegistration> FindHelper(
      const string& func) const TF_SHARED_LOCKS_REQUIRED(mu_);

  mutable mutex mu_;
  gtl::FlatMap<string, std::shared_ptr<FunctionDefAndOpRegistration>>
      function_defs_ TF_GUARDED_BY(mu_);
  gtl::FlatMap<string, string> func_grad_ TF_GUARDED_BY(mu_);
};

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_UTILS_FUNCTION_H_
