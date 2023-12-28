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

#ifndef ITEX_CORE_GRAPH_MEMORY_OPT_PASS_MEMORY_OPT_PASS_H_
#define ITEX_CORE_GRAPH_MEMORY_OPT_PASS_MEMORY_OPT_PASS_H_

#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "itex/core/graph/memory_opt_pass/check_const_filter.h"
#include "itex/core/graph/memory_opt_pass/weight_prepack.h"
#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/graph/utils/grappler_item.h"
#include "itex/core/graph/utils/node_type_attr_map.h"
#include "itex/core/graph/utils/utils.h"
#include "itex/core/utils/node_def_util.h"
#include "protos/graph.pb.h"

namespace itex {
namespace graph {

using utils::MutableNodeView;

typedef struct {
  bool is_inplace;
  bool is_instack;
  bool is_visited;
  // queries to this node
  // std::vector<int> query;
  // required conditions for safe inplace
  // std::map<int, bool> cond;
} SearchInfo;

struct MemoryOptContext {
  explicit MemoryOptContext(const GrapplerItem& item, GraphDef* g_def,
                            Status* status)
      : graph_view(g_def, status), nodes_to_preserve(item.NodesToPreserve()) {
    TF_ABORT_IF_ERROR(node_type_map.Init(*g_def));
  }

  utils::MutableGraphView graph_view;
  std::unordered_set<string> nodes_to_preserve;
  NodeTypeAttrMap node_type_map;
};

// Return the port of input tensor may be forwarded
std::vector<int> GetCandidateForwardPort(const MutableNodeView* node_view);

bool IsInPreserveSet(const MemoryOptContext* ctx, const NodeDef* node);

bool IsOnSameDevice(const MutableNodeView* node_view_x,
                    const MutableNodeView* node_view_y);

// Return out port id `cur_node_view` get fanin from `tgt_node_view`
int GetOutPort(const MutableNodeView* cur_node_view,
               const MutableNodeView* tgt_node_view);

// Return the static reference count of target buffer referenced by current node
int GetStaticRefCount(const MutableNodeView* node_view, const int forward_port);

void CheckDependence(MemoryOptContext* ctx, const MutableNodeView* node_view,
                     const int forward_port);

void DetectUnvisitedNode(MemoryOptContext* ctx,
                         const MutableNodeView* node_view);

void InplaceInference(MemoryOptContext* ctx, const MutableNodeView* node_view);

void StaticInplaceOpt(MemoryOptContext* ctx, const char* device_name);

void WeightCacheOpt(MemoryOptContext* ctx);

Status RunMemoryOptPass(OptimizerContext* opt_ctx, const GrapplerItem& item,
                        const GraphDef& graph_def, GraphDef* optimized_graph);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_MEMORY_OPT_PASS_MEMORY_OPT_PASS_H_
