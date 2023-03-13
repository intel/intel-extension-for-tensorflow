/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/graph/utils/pattern_utils.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <utility>

#include "absl/container/flat_hash_set.h"

namespace itex {
namespace graph {
namespace utils {

inline const bool IsCommutativeOp(const string& op) {
  // TODO(itex): Add more ops to this list if needed.
  static const auto commutative_ops =
      absl::flat_hash_set<string>({"Add", "AddV2", "Mul"});
  return commutative_ops.contains(op);
}

// `expected` are op names in the pattern and they could be wildcard `*` or
// some registered op in tensorflow. `input` are real op names in the
// computation graph.
// Look further if any input is wildcard `*`.
inline bool NeedSwap(string input_0, string expected_0, string input_1,
                     string expected_1) {
  // Do not swap if the original order can be matched.
  // TODO(itex): Continue swapping even it's matched for further optimization.
  if (input_0 == expected_0 && input_1 == expected_1) return false;

  if (input_0 == expected_1 && input_1 == expected_0) return true;

  if (expected_0 == "*" && input_0 == expected_1) return true;

  if (expected_1 == "*" && input_1 == expected_0) return true;

  return false;
}

inline bool HasUndeterminedSameTypeInputs(const OpTypePattern& pattern) {
  // Only support 1 undetermined same type of inputs
  int n_input_num = 0;
  for (auto const& child : pattern.children) {
    if (child.op != "*" && absl::EndsWith(child.op, "*")) {
      n_input_num++;
    }
  }
  ITEX_CHECK(n_input_num <= 1);

  return n_input_num == 1;
}

inline std::vector<int> GetChildrenIndices(const OpTypePattern& pattern,
                                           int size) {
  std::vector<int> indices(size);
  bool has_n_input = HasUndeterminedSameTypeInputs(pattern);
  if (has_n_input) {
    int start = 0;
    for (int i = 0; i < pattern.children.size(); i++) {
      if (absl::EndsWith(pattern.children[i].op, "*")) {
        start = i;
        break;
      }
    }

    for (int i = 0; i < start; i++) {
      indices[i] = i;
    }

    int num = size - (pattern.children.size() - 1);
    std::fill(indices.begin() + start, indices.begin() + start + num, start);

    for (int i = start + num, j = 1; i < size; i++, j++) {
      indices[i] = start + j;
    }
  } else {
    std::iota(indices.begin(), indices.end(), 0);
  }
  return indices;
}

// A subgraph pattern syntax implicitly defines a DAG having a single root. We
// traverse the syntax DAG in DFS manner. This function finds a match for
// current root of the pattern with the current node and recursively matches
// children subpatterns with the children of current node.
template <>
bool SubGraphMatcher<MatchingDirection::kFollowInputs>::DoesOpTypePatternMatch(
    const OpTypePattern& pattern, MutableNodeView* node_view,
    NodeViewMatch* match, bool fanin_checking) {
  // Currently no control inputs and outputs are allowed.
  // But there's a situation that if a node is remained with controlling
  // fanins, we can continue to the matching.
  if (node_view->NumControllingFanins() > 0 &&
      (fanin_checking || pattern.node_status != NodeStatus::kRemain)) {
    ITEX_VLOG(3) << pattern.op << "[" << pattern.label
                 << "] failed due to controlling fanins";
    return false;
  }

  if (node_view->NumControlledFanouts() > 0) {
    ITEX_VLOG(3) << pattern.op << "[" << pattern.label
                 << "] failed due to controlled fanouts";
    return false;
  }

  ITEX_VLOG(3) << pattern.op << "[" << pattern.label << "] vs. "
               << node_view->node()->op() << "[" << node_view->node()->name()
               << "]";

  bool op_type_matched = false;
  if (pattern.op == "*") {
    op_type_matched = true;
  } else {
    // The op field string of current pattern might express an op among multiple
    // op types (mutually exclusive) separated by '|'.
    std::vector<string> op_list = str_util::Split(pattern.op, '|');
    for (const string& op : op_list) {
      auto current_op = op;
      if (absl::EndsWith(op, "*")) {
        current_op.pop_back();
      }
      if (node_view->node()->op() == current_op) {
        op_type_matched = true;
        break;
      }
    }
  }
  if (op_type_matched) {
    // If op type matches and current node is visited first time, insert current
    // node to node_label_to_index_ map with the current label as the key.
    // Multiple occurances of same label in the pattern syntax indicates that
    // the same node needs to be visited for each of such occurances. Hence
    // subsequent visits should find the corresponding label in the map as a key
    // and the current node should be the value for that key.
    auto label = pattern.label;
    if (absl::EndsWith(label, "*")) {
      label.pop_back();  // Delete the last star

      int count = 0;
      for (const auto& item : node_label_to_index_) {
        if (absl::StartsWith(item.first, label)) {
          count++;
        }
      }

      label += std::to_string(count);
    }

    if (node_label_to_index_.find(label) == node_label_to_index_.end()) {
      node_label_to_index_[label] = node_view->node_index();
      // Bookkeeping
      matched_node_indices_.insert(node_view->node_index());
      if (pattern.node_status == NodeStatus::kRemove) {
        remove_node_indices_.insert(node_view->node_index());
      }
    } else if (node_label_to_index_[label] != node_view->node_index()) {
      ITEX_VLOG(3) << "The label (" << label << ") "
                   << node_label_to_index_[label] << " can't match "
                   << node_view->node_index();
      auto name1 =
          graph_view_->GetNode(node_label_to_index_[label])->node()->name();
      auto name2 = node_view->node()->name();
      ITEX_VLOG(3) << "The exsiting name is " << name1;
      ITEX_VLOG(3) << "Current name is " << name2;
      return false;  // label constraint could not be satisfied.
    } else {
      ITEX_DCHECK(node_label_to_index_[label] == node_view->node_index());
    }
  } else {
    ITEX_VLOG(3) << "The op type is not match " << node_view->node()->op()
                 << " vs. " << pattern.op;
    return false;
  }
  // Current root of the pattern syntax is matched with the current node.
  match->node_view = node_view;

  // Go for matching child subpattern.
  if (!pattern.children.empty()) {
    // Currently only direction toward inputs is implemented.
    auto has_n_inputs = HasUndeterminedSameTypeInputs(pattern);
    auto graph_children = node_view->GetRegularFanins();
    auto num_children = graph_children.size();
    if (!has_n_inputs && num_children != pattern.children.size()) {
      ITEX_VLOG(3) << "The " << pattern.label
                   << "'s children size is not consistent (" << num_children
                   << " vs. " << pattern.children.size() << ")";
      return false;
    } else {
      // A pattern is a graph that we would like to match with a subgraph of
      // a tensorflow computation graph. We travese both pattern-graph and the
      // given graph in DFS manner and try to find one-to-one mapping between
      // the nodes. However, commutative binary ops (e.g., Add, AddV2, Mul
      // etc.) in the computation graph can have their inputs in different order
      // than the pattern syntax graph. To allow such input permutation in a
      // limited manner, we employ a heuristic of looking one level ahead in
      // both graphs, whether visiting the right child of pattern is likely to
      // match left child of the given graph. In that case, we simply swap the
      // left subtree with right subtree of pattern syntax graph and continue
      // matching children of pattern with the children of given computation
      // graph. Note, we do not change anything in the computation graph during
      // pattern matching, only the pattern graph is changed. By looking ahead
      // one step in case of commutative ops, we keep the time comlexity of
      // pattern matching linear. Since it is only a heuristic and we look only
      // one step ahead it is not guranteed that all possible permutations will
      // be matched. For example, when both the input ops to the commutative op
      // are same, we cannot anticipate which of the permutation is likely to
      // match unless we look two level down the graphs.
      std::vector<int> pattern_child_indices =
          GetChildrenIndices(pattern, num_children);

      string op_name = pattern.op;
      if (IsCommutativeOp(op_name) && num_children == 2) {
        MutableNodeView* graph_child0_node_view =
            graph_view_->GetNode(graph_children[0].node_index());
        MutableNodeView* graph_child1_node_view =
            graph_view_->GetNode(graph_children[1].node_index());
        if (NeedSwap(graph_child0_node_view->GetOp(), pattern.children[0].op,
                     graph_child1_node_view->GetOp(), pattern.children[1].op))
          std::swap(pattern_child_indices[0], pattern_child_indices[1]);
      }

      for (size_t i = 0; i < num_children; ++i) {
        auto child_node_index = graph_children[i].node_index();
        // TODO(mdfaijul): Is it guaranted that GetNode will reuturn non
        // null pointer.
        MutableNodeView* child_node_view =
            graph_view_->GetNode(child_node_index);
        const OpTypePattern& child_pattern =
            pattern.children[pattern_child_indices[i]];
        match->children.push_back(NodeViewMatch());
        NodeViewMatch* child_match = &(match->children.back());
        if (!DoesOpTypePatternMatch(child_pattern, child_node_view, child_match,
                                    fanin_checking)) {
          return false;
        }
      }
    }
  }
  return true;
}

// Current implementation supports pattern maching toward node's inputs only.
template <>
bool SubGraphMatcher<MatchingDirection::kFollowInputs>::GetMatchedNodes(
    const OpTypePattern& pattern,
    const std::unordered_set<string>& nodes_to_preserve,
    MutableNodeView* node_view, std::map<string, int>* matched_nodes_map,
    std::set<int>* remove_node_indices, bool fanin_checking) {
  bool found_match = false;
  match_.reset(new NodeViewMatch());
  if (DoesOpTypePatternMatch(pattern, node_view, match_.get(),
                             fanin_checking)) {
    if (IsSafeNodesToRemove(nodes_to_preserve)) {
      found_match = true;
      *matched_nodes_map = this->node_label_to_index_;
      *remove_node_indices = this->remove_node_indices_;
    } else {
      ITEX_VLOG(3) << "Some nodes in preserve set";
    }
  } else {
    found_match = false;
  }

  // Clear all bookkeeping data
  match_->Clear();
  match_.reset(nullptr);
  matched_node_indices_.clear();
  node_label_to_index_.clear();
  remove_node_indices_.clear();

  return found_match;
}

static std::string DumpPatternHelper(const OpTypePattern& pattern) {
  std::string record = std::string(pattern.label);
  record += " [label=\"{";
  record += pattern.label;
  record += "|";
  record += pattern.op;
  record += "|";
  switch (pattern.node_status) {
    case NodeStatus::kRemain:
      record += "Remain";
      break;
    case NodeStatus::kRemove:
      record += "Remove";
      break;
    case NodeStatus::kReplace:
      record += "Replace";
      break;
  }
  record += "\\l}\"]\n";

  for (auto const& child : pattern.children) {
    record += DumpPatternHelper(child);
  }

  record += pattern.label + " -> {";
  for (auto const& child : pattern.children) {
    record += child.label;
    record += " ";
  }
  record += "} [dir=back]\n";

  return record;
}

void DumpPattern(const OpTypePattern& pattern, std::string path) {
  std::string header = "digraph Pattern {\n";
  header.append("rankdir=BT\n");
  header.append("node [shape=record]\n");
  std::string body = DumpPatternHelper(pattern);
  std::string tail = "}";
  std::ofstream result(path);
  result << header << body << tail << std::endl;
  result.close();
}

}  // namespace utils
}  // namespace graph
}  // namespace itex
