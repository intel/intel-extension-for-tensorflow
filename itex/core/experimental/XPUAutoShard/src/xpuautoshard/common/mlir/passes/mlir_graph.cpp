/* Copyright (c) 2023 Intel Corporation

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

#include "xpuautoshard/common/mlir/passes/mlir_graph.h"

#include <queue>
#include <string>
#include <unordered_map>
#include <utility>

#include "xpuautoshard/common/mlir/passes/pass_utils.h"

namespace mlir {
namespace hs {

std::vector<int64_t> MLIRValueDesc::getConstVecInt64() const {
  return getConstantIntArrayForValue(v_);
}

std::vector<int64_t> MLIROpDesc::getAttrVecInt64(
    const std::string& attr_name) const {
  auto&& int_vec = getIntArrayAttr(op_, attr_name);
  assert(!int_vec.empty() && "Cannot find op attribute of integer vector type");
  return int_vec;
}

as::BreadthFirstGraphIterRangeRef MLIRGraph::getBreadthFirstIterRange() {
  std::vector<Operation*> bf_ordered_vec;
  std::queue<Operation*> ready_queue;
  for (Region& region : root_op_->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        ready_queue.push(&op);
      }
    }
  }
  std::unordered_map<Operation*, size_t> num_operands_visited;
  while (!ready_queue.empty()) {
    Operation* op = ready_queue.front();
    ready_queue.pop();
    bf_ordered_vec.push_back(op);
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        if (num_operands_visited.find(user) == num_operands_visited.end()) {
          num_operands_visited[user] = 0;
        }
        if (++num_operands_visited[user] == user->getNumOperands()) {
          ready_queue.push(user);
        }
      }
    }
  }
  return as::makeRef<BreadthFirstIterRangeImpl, as::BreadthFirstGraphIterRange>(
      std::move(bf_ordered_vec));
}

as::OpDescRef MLIRGraph::BreadFirstIterImpl::operator*() {
  return mlirOpToOpDesc(*iter_);
}

as::BreadthFirstGraphIterator MLIRGraph::BreadthFirstIterRangeImpl::begin() {
  return as::BreadthFirstGraphIterator(
      as::makeRef<BreadFirstIterImpl, as::BreadthFirstGraphIterator::Impl>(
          bf_ordered_vec_.begin()));
}

as::BreadthFirstGraphIterator MLIRGraph::BreadthFirstIterRangeImpl::end() {
  return as::BreadthFirstGraphIterator(
      as::makeRef<BreadFirstIterImpl, as::BreadthFirstGraphIterator::Impl>(
          bf_ordered_vec_.end()));
}

}  // namespace hs
}  // namespace mlir
