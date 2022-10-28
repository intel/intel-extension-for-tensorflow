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

#include "itex/core/graph/utils/grappler_item.h"

#include <memory>

namespace itex {
namespace graph {

GrapplerItem::GrapplerItem(const TF_GrapplerItem* tf_item) {
  TF_Status* status = TF_NewStatus();
  item_ = const_cast<TF_GrapplerItem*>(tf_item);

  int num_values = 0;
  size_t storage_size = 0;
  TF_GetFetchNodesListSize(item_, &num_values, &storage_size, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status))
      << " Error for TF_GetFetchNodesListSize";
  fetch.resize(num_values);

  std::unique_ptr<char*[]> values(new char*[num_values]);
  std::unique_ptr<size_t[]> lens(new size_t[num_values]);
  std::unique_ptr<char[]> storage(new char[storage_size]);
  TF_GetFetchNodesList(
      item_, reinterpret_cast<char**>(values.get()), lens.get(), num_values,
      reinterpret_cast<void*>(storage.get()), storage_size, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status)) << " Error for TF_GetFetchNodesList";

  for (int32_t i = 0; i < num_values; ++i) {
    fetch[i] = string(values[i], lens[i]);
  }

  TF_DeleteStatus(status);
}

std::unordered_set<string> GrapplerItem::NodesToPreserve() const {
  TF_Status* status = TF_NewStatus();
  int num_values = 0;
  size_t storage_size = 0;
  std::unordered_set<string> nodes;
  TF_GetNodesToPreserveListSize(item_, &num_values, &storage_size, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status))
      << " Error for TF_GetNodesToPreserveListSize";

  std::unique_ptr<char*[]> values(new char*[num_values]);
  std::unique_ptr<size_t[]> lens(new size_t[num_values]);
  std::unique_ptr<char[]> storage(new char[storage_size]);
  TF_GetNodesToPreserveList(
      item_, reinterpret_cast<char**>(values.get()), lens.get(), num_values,
      reinterpret_cast<void*>(storage.get()), storage_size, status);
  ITEX_CHECK_EQ(TF_OK, TF_GetCode(status))
      << " Error for TF_GetNodesToPreserveList";

  for (int32_t i = 0; i < num_values; ++i) {
    nodes.insert(string(values[i], lens[i]));
  }
  TF_DeleteStatus(status);
  return nodes;
}
}  // namespace graph
}  // namespace itex
