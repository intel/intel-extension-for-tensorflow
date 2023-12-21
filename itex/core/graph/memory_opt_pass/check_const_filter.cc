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

#include "itex/core/graph/memory_opt_pass/check_const_filter.h"

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/graph/optimizer_config.h"
#include "itex/core/graph/utils/layout_utils.h"
#include "itex/core/graph/utils/op_types.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/op_def_util.h"

namespace itex {
namespace graph {

static const std::vector<int> GetConstFilterCheckList(
    const std::string& op_name) {
  static std::unordered_map<std::string, std::vector<int>>
      op_const_checklist_map = {{"GRUBlockCell", {2, 3, 4, 5}},
                                {"_ITEXGRUCell", {2, 3, 4, 5}},
                                {"_ITEXAUGRUCell", {3, 4, 5, 6}},
                                {"_ITEXForwardGRU", {2, 3, 4, 5}},
                                {"_ITEXForwardAUGRU", {3, 4, 5, 6}},
                                {"_default", {1}}};

  if (op_const_checklist_map.find(op_name) == op_const_checklist_map.end()) {
    return op_const_checklist_map.at("_default");
  }
  return op_const_checklist_map.at(op_name);
}

bool IsUnchangingVariable(const utils::MutableNodeView* node_view) {
  const NodeDef* node_def = node_view->node();
  if (IsCast(*node_def) &&
      IsReadVariableOp(*(node_view->GetRegularFanin(0).node_view()->node())) &&
      GetOptimizerConfigFlags().enable_optimize_aggressive)
    return true;

  if (!GetOptimizerConfigFlags().enable_optimize_aggressive ||
      !IsReadVariableOp(*node_def))
    return false;

  auto* arg_node_view = node_view->GetRegularFanin(0).node_view();
  auto* arg_node_def = arg_node_view->node();

  if (IsEnter(*arg_node_def)) {
    arg_node_view = arg_node_view->GetRegularFanin(0).node_view();
    arg_node_def = arg_node_view->node();
  }

  if (!IsArg(*arg_node_def)) return false;

  if (arg_node_view->NumRegularFanouts() == 1) return true;

  // Since _Arg Fanouts number > 1, attempt to read variable inside while loop
  //         _Arg
  //         /  |
  //     Enter ReadVariable
  //       |
  //  ReadVariable
  for (const auto& fanout_i : arg_node_view->GetRegularFanouts()) {
    for (const auto fanout : fanout_i) {
      if (!IsEnter(*(fanout.node_view()->node())) &&
          !IsReadVariableOp(*(fanout.node_view()->node())))
        return false;
    }
  }

  return true;
}

void CheckConstFilter(const utils::MutableNodeView* node_view,
                      const std::unordered_set<string>& nodes_to_preserve) {
  const NodeDef* node_def = node_view->node();
  const OpDef op_def = GetOpDef(*node_def);
  bool is_filter_const = false;

  // Skip if has no attr `is_filter_const`.
  // Note: NodeDef may not have default attr, need to check OpDef either.
  if (!TryGetNodeAttr(AttrSlice(*node_def), "is_filter_const",
                      &is_filter_const))
    if (FindAttr("is_filter_const", op_def) == nullptr) return;

  // Skip if weight is already marked as const.
  if (is_filter_const == true) return;

  auto checklist = GetConstFilterCheckList(node_view->node()->op());

  is_filter_const = true;
  for (int index = 0; index < static_cast<int>(checklist.size()); ++index) {
    const auto* filter_node_view =
        node_view->GetRegularFanin(checklist[index]).node_view();
    const NodeDef* filter_node_def = filter_node_view->node();

    // Do not set const filter attr if filter is feed node.
    if (nodes_to_preserve.count(filter_node_def->name()) > 0) {
      is_filter_const = false;
      break;
    }

    if (IsConstant(*filter_node_def)) continue;

    // Find variables that will not change from ReadVariables & _Arg
    if (!IsUnchangingVariable(filter_node_view)) {
      is_filter_const = false;
      break;
    }
  }

  auto* new_attr = node_view->node()->mutable_attr();
  SetAttrValue(is_filter_const, &(*new_attr)["is_filter_const"]);
}

}  // namespace graph
}  // namespace itex
