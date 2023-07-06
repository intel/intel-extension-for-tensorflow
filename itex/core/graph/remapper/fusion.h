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

#ifndef ITEX_CORE_GRAPH_REMAPPER_FUSION_H_
#define ITEX_CORE_GRAPH_REMAPPER_FUSION_H_

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/graph/remapper/remapper.h"
#include "itex/core/graph/utils/pattern_utils.h"

namespace itex {
namespace graph {

struct MatchedProperties {
  typedef std::map<std::string, int> NodesMap;
  typedef std::set<int> NodeIndices;

  // The output of SubGraphMatcher, whose key is label and value is node index
  // in graph.
  NodesMap map;

  // The node index which will be override, which are NodeStatus::kReplace node
  // in pattern.
  NodeIndices invalidated;

  // The output of SubGraphMatcher, which is the node index will be deleted.
  NodeIndices deleted;

  // To check whether the graph is matched.
  bool Empty() { return map.empty(); }

  // Helper function for false checking.
  MatchedProperties& ToEmpty() {
    map.clear();
    invalidated.clear();
    deleted.clear();

    return *this;
  }

  const NodeDef* GetNode(utils::MutableGraphView* graph_view,
                         const char* name) const {
    auto node_name = this->map.at(name);
    auto* node_view = graph_view->GetNode(node_name);

    if (node_view) {
      return node_view->node();
    }

    ITEX_VLOG(FATAL) << "Has not found the node " << node_name
                     << " with pattern " << name;
    return nullptr;
  }
};

class Fusion {
 public:
  typedef std::vector<std::string> Labels;

  struct InternalPattern {
    InternalPattern() = default;
    explicit InternalPattern(utils::OpTypePattern&& pattern_graph);

    utils::OpTypePattern info;

    // This two members come from `info` and will be used very often.
    Labels labels_of_replace;
    int num_nodes;
  };

  Fusion() {}
  virtual ~Fusion() {}

  // Check with the saved pattern. If failed, the result will be an empty.
  virtual MatchedProperties Check(RemapperContext* ctx,
                                  const int node_index) const = 0;

  // Based on the result of Check to update mutable graph in RemapperContext.
  virtual Status Update(RemapperContext* ctx /** in and out **/,
                        const MatchedProperties& properties) const = 0;

  // The fusion name, such as sigmoid-with-mul.
  virtual std::string Name() = 0;

  // The output node op of pattern graph.
  std::string Key();

  // The nodes number in graph, including Any node.
  int NumNodes() const;

  inline bool IsPartial() const { return is_partial_; }

 protected:
  InternalPattern pattern_;

  // Set it as true only if need this fusion before oneDNN Graph.
  bool is_partial_ = false;
};

class FusionMgr {
 public:
  static FusionMgr& GetInstance();

  FusionMgr(const FusionMgr& other) = delete;
  void operator=(const FusionMgr& other) = delete;

  // Maybe multiple fusion pattern with the same output op. We should use the
  // sort to determine the priority. Currently, we just use the nodes number as
  // the priority, which means, the more nodes, the higher priority.
  void Sort();

  // Add a fusion to global, the key must be the output node op. For instance,
  // the AnyInput -> Sigmoid -> Mul, the key will be Mul.
  void AddFusion(const std::string& key, Fusion* fusion);

  // Based on the node op, get all relevant fusions.
  std::vector<Fusion*>& GetFusions(const std::string& key);

 private:
  FusionMgr() {}

  // Main structure of FusionManager. The key is pattern graph's output node op
  // (not label). And the value will be multiple fusions.
  std::unordered_map<std::string, std::vector<Fusion*>> map_;
};

template <typename T>
class FusionRegistrar {
 public:
  FusionRegistrar() {
    fusion_ = new T();
    std::vector<std::string> keys = absl::StrSplit(fusion_->Key(), "|");
    for (auto const& key : keys) {
      FusionMgr::GetInstance().AddFusion(key, fusion_);
      ITEX_VLOG(1) << "Register fusion " << fusion_->Name() << " with " << key;
    }
  }

  ~FusionRegistrar() { delete fusion_; }

 private:
  Fusion* fusion_;
};

#define REGISTER_FUSION(klass) REGISTER_FUSION_UNIQ_HELPER(__COUNTER__, klass)

#define REGISTER_FUSION_UNIQ_HELPER(ctr, klass) \
  REGISTER_FUSION_UNIQ_HELP(ctr, klass)

#define REGISTER_FUSION_UNIQ_HELP(ctr, klass) \
  static FusionRegistrar<klass> const fusion_##ctr;

// Helper function to compatiable with existing SubGraphMatcher.
MatchedProperties FillProperties(utils::MutableGraphView* graph_view,
                                 utils::MutableNodeView* node_view,
                                 const Fusion::InternalPattern& pattern,
                                 bool fanin_checking = true);

// Helper function to get output properties from graph.
std::vector<OpInfo_TensorProperties> GetOutputProperties(RemapperContext* ctx,
                                                         int index);

// Helper function to compatiable current remapper for loop.
// Will change the content of invalidated and deleted.
// Use the pointer output instead of reference to make cpplint happy.
Status LaunchPatternMatcher(RemapperContext* ctx, int index,
                            std::vector<bool>* invalidated,
                            std::vector<bool>* deleted, bool is_full = true);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_REMAPPER_FUSION_H_
