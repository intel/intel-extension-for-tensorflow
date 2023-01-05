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

#include "itex/core/graph/remapper/fusion.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "itex/core/graph/utils/pattern_utils.h"
#include "itex/core/graph/utils/utils.h"

namespace itex {
namespace graph {

using utils::NodeStatus;
using utils::OpTypePattern;

static int NumNodesHelper(const OpTypePattern& pattern) {
  int count = 1;
  for (auto const& child : pattern.children) {
    count += NumNodesHelper(child);
  }
  return count;
}

static Fusion::Labels FilterLabels(const OpTypePattern& pattern,
                                   const NodeStatus status) {
  Fusion::Labels labels;
  if (pattern.node_status == status) {
    labels.push_back(pattern.label);
  }

  for (auto const& child : pattern.children) {
    auto current = FilterLabels(child, status);
    labels.insert(labels.end(), current.begin(), current.end());
  }

  return labels;
}

Fusion::InternalPattern::InternalPattern(OpTypePattern&& pattern) {
  info = std::move(pattern);
  labels_of_replace = FilterLabels(info, NodeStatus::kReplace);
  num_nodes = NumNodesHelper(info);
}

int Fusion::NumNodes() const { return pattern_.num_nodes; }

std::string Fusion::Key() { return pattern_.info.op; }

FusionMgr& FusionMgr::GetInstance() {
  static FusionMgr fusioners;
  return fusioners;
}

void FusionMgr::Sort() {
  for (auto& [key, value] : map_) {
    std::sort(value.begin(), value.end(),
              [](const Fusion* left, const Fusion* right) {
                return left->NumNodes() > right->NumNodes();
              });
  }
}

void FusionMgr::AddFusion(const std::string& key, Fusion* fusion) {
  map_[key].push_back(fusion);
}

std::vector<Fusion*>& FusionMgr::GetFusions(const std::string& key) {
  if (map_.count(key) != 0) {
    return map_[key];
  }

  static auto empty_vector = std::vector<Fusion*>();
  return empty_vector;
}

MatchedProperties FillProperties(utils::MutableGraphView* graph_view,
                                 utils::MutableNodeView* node_view,
                                 const Fusion::InternalPattern& pattern,
                                 bool fanin_checking) {
  using utils::MatchingDirection;
  using utils::SubGraphMatcher;
  MatchedProperties properties;
  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(graph_view);
  graph_matcher.GetMatchedNodes(pattern.info, {}, node_view, &properties.map,
                                &properties.deleted, fanin_checking);
  if (!properties.Empty()) {
    for (auto const& label : pattern.labels_of_replace) {
      properties.invalidated.insert(properties.map.at(label));
    }
  }

  return properties;
}

std::vector<OpInfo_TensorProperties> GetOutputProperties(RemapperContext* ctx,
                                                         int index) {
  std::vector<OpInfo_TensorProperties> properties;

  NodeDef* node_def = ctx->graph_view.GetNode(index)->node();
  Status s = ctx->GetGraphProperties().GetOutputProperties(node_def->name(),
                                                           &properties);

  if (!s.ok()) {
    ITEX_VLOG(1) << "Have not found the output properties for "
                 << node_def->name();
  }

  return properties;
}

Status LaunchPatternMatcher(RemapperContext* ctx, int index,
                            std::vector<bool>* invalidated,
                            std::vector<bool>* deleted, bool is_full) {
  auto* node = ctx->graph_view.GetNode(index)->node();

  for (auto const& fusion : FusionMgr::GetInstance().GetFusions(node->op())) {
    if (!is_full && !fusion->is_partial) continue;
    ITEX_VLOG(3) << "Start to run fusion pass: " << fusion->Name();
    auto properties = fusion->Check(ctx, index);
    if (!properties.Empty()) {
      Status status = fusion->Update(ctx, properties);

      for (auto const& index : properties.invalidated) {
        invalidated->at(index) = true;
      }

      for (auto const& index : properties.deleted) {
        RemoveAllRegularFanin(ctx, index);
        deleted->at(index) = true;
      }

      ITEX_VLOG(3) << "Succeed to match fusion pass: " << fusion->Name();
      return status;
    }
    ITEX_VLOG(3) << "Failed to match fusion pass: " << fusion->Name();
  }

  return Status::OK();
}
}  // namespace graph
}  // namespace itex
