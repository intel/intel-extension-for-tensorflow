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

#include "xpuautoshard/tensorflow/passes/tfg_to_hs.h"

#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "itex/core/ir/dialect.h"
#include "llvm/ADT/BitVector.h"
#include "xpuautoshard/tensorflow/macro.h"

namespace mlir {
namespace hs {

using as::ShardingProperty;

namespace {

bool isInputOp(Operation* op) { return op->getNumOperands() == 0; }

bool isResourceTensor(mlir::Type t) {
  if (auto tensor_ty = t.dyn_cast<mlir::TensorType>()) {
    return tensor_ty.getElementType().isa<mlir::tfg::ResourceType>();
  }
  return false;
}

bool isResourceOp(Operation* op) {
  return std::any_of(op->getOperandTypes().begin(), op->getOperandTypes().end(),
                     isResourceTensor);
}

bool isBlackListOp(Operation* op) {
  auto op_name = op->getName().getStringRef();
  return op_name == "tfg.StringFormat" || op_name == "tfg.PrintV2" ||
         op_name == "tfg.UnsortedSegmentSum";
}

bool isReturnOp(Operation* op) {
  auto op_name = op->getName().getStringRef();
  return op_name == "tfg._Retval" || op_name == "tfg._DeviceRetval";
}

bool opNeedUnshard(Operation* op) {
  return isInputOp(op) || isResourceOp(op) || isReturnOp(op) ||
         isBlackListOp(op);
}

void insertShardOpAfter(Operation* op, Value result, ShardingPropertyAttr hsp) {
  assert(!result.use_empty());
  // XXX: should we make builder a member variable for better efficiency?
  OpBuilder builder(op);
  auto hs_op =
      builder.create<ShardOp>(op->getLoc(), result.getType(), result, hsp);
  op->moveBefore(hs_op);
  result.replaceAllUsesExcept(hs_op->getResult(0), hs_op);
}

void insertUnshardOpAfter(Operation* op, Value result) {
  OpBuilder builder(op);
  auto unshard_op =
      builder.create<UnshardOp>(op->getLoc(), result.getType(), result);
  op->moveBefore(unshard_op);
  result.replaceAllUsesExcept(unshard_op->getResult(0), unshard_op);
}

void insertUnshardOpBefore(Operation* op) {
  OpBuilder builder(op);
  for (auto operand : op->getOperands()) {
    auto unshard_op =
        builder.create<UnshardOp>(op->getLoc(), operand.getType(), operand);
    op->replaceUsesOfWith(operand, unshard_op->getResult(0));
  }
}

/**
 * @brief FusedBatchNormV3 doesn't use mean and variance as inputs during
 * training but sometimes the TF frontend would generate ReadVariable on
 * mean and variance from another FusedBatchNormV3 and insert a control
 * dependency on the AssignVariable of the FusedBatchNormV3. This is wrong and
 * might cause performance hit too. It would also cause unnecessary cyclic
 * dependency after we add control dependencies among sharding stages.
 * https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fused-batch-norm-v3
 *
 * For example, the "FusedBatchNormV3 (2)" below shouldn't do ReadVariable for
 * its mean but should read an empty tensor instead. FusedBatchNormV3 (1) ->
 * mean -> AssignVariable -> control -> ReadVariable -> FusedBatchNormV3 (2)
 *
 * This function removes the control edge between AssignVariable and
 * ReadVariable to avoid the unnecessary cyclic dependency.
 *
 * @param root
 */
void fixFusedBatchNormV3Training(Operation* root) {
  auto fix_graph_op = [&](mlir::tfg::GraphOp* graph_op) {
    auto& region = graph_op->getRegion();
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (op.getName().getStringRef() == "tfg.FusedBatchNormV3" &&
            op.getNumOperands() >= 5 &&
            op.getOperand(3).getDefiningOp()->getName().getStringRef() ==
                "tfg.ReadVariableOp" &&
            op.getOperand(4).getDefiningOp()->getName().getStringRef() ==
                "tfg.ReadVariableOp") {
          auto drop_control_edges = [&](Operation* read_variable) {
            std::vector<Value> control_edges;
            llvm::BitVector indices(read_variable->getNumOperands());
            for (auto indexed_operand :
                 llvm::enumerate(read_variable->getOperands())) {
              if (indexed_operand.value().getType().isa<tfg::ControlType>()) {
                indices.set(indexed_operand.index());
                control_edges.push_back(indexed_operand.value());
              }
            }
            read_variable->eraseOperands(indices);
            for (auto control_edge : control_edges) {
              control_edge.dropAllUses();
            }
          };
          drop_control_edges(op.getOperand(3).getDefiningOp());
          drop_control_edges(op.getOperand(4).getDefiningOp());
        }
      }
    }
  };
  if (auto graph_op = llvm::dyn_cast<mlir::tfg::GraphOp>(root)) {
    fix_graph_op(&graph_op);
    return;
  }
  for (Region& region : root->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (auto graph_op = llvm::dyn_cast<mlir::tfg::GraphOp>(op)) {
          fix_graph_op(&graph_op);
        }
      }
    }
  }
}

void op_erase_dfs(std::unordered_map<Operation*, bool>* visited,
                  Operation* op) {
  if (op == nullptr) return;
  if (visited->at(op) == true) return;
  if (op->use_empty() == false) return;
  std::vector<Operation*> input_ops;
  for (auto operand : op->getOperands()) {
    Operation* operation = operand.getDefiningOp();
    input_ops.push_back(operation);
  }
  op->erase();
  visited->at(op) = true;
  for (auto operation : input_ops) {
    op_erase_dfs(visited, operation);
  }
}

bool isPreserveNode(Operation* op) {
  // Preserve Nodes are a collection of nodes that must be reserved. The OP
  //  of these nodes mainly includes the following four types:
  //  1. Update_OP
  //  2. Assign_OP
  //  3. Communication_OP
  //  4. Retval_OP
  std::string op_name = op->getName().getStringRef().str();
  if (kUpdateOP.find(op_name) != kUpdateOP.end()) return true;
  if (kAssignOP.find(op_name) != kAssignOP.end()) return true;
  if (kCommunicationOP.find(op_name) != kCommunicationOP.end()) return true;
  if (kRetvalOP.find(op_name) != kRetvalOP.end()) return true;
  return false;
}

}  // anonymous namespace

TFGtoHSConversion::~TFGtoHSConversion() = default;

TFGtoHSConversion::TFGtoHSConversion(const as::DeviceInfo& device_info,
                                     const as::ShardingConfig& sharding_config)
    : device_info_(device_info), sharding_config_(sharding_config) {}

void TFGtoHSConversion::DeadNodePrune(mlir::tfg::GraphOp* graph_op) {
  // Starting from `Preserve Nodes`, BFS search the TFG, and prune the
  // unreachable nodes.
  std::queue<Operation*> preserve_queue;
  std::unordered_map<Operation*, bool> visited;
  std::vector<Operation*> all_nodes;
  for (Block& block : graph_op->getRegion().getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (isPreserveNode(&op)) {
        preserve_queue.push(&op);
        visited[&op] = true;
      }
      all_nodes.push_back(&op);
    }
  }
  while (!preserve_queue.empty()) {
    Operation* op = preserve_queue.front();
    preserve_queue.pop();
    for (auto operand : op->getOperands()) {
      Operation* operation = operand.getDefiningOp();
      if (visited.count(operation) == 0) {
        visited[operation] = true;
        preserve_queue.push(operation);
      }
    }
  }
  for (auto&& op : all_nodes) {
    if (visited[op] == false) {
      op_erase_dfs(&visited, op);
    }
  }
}

void TFGtoHSConversion::addUnshardToGraphOutputs(mlir::Block* block) {
  std::vector<Operation*> end_ops;
  // for all the graph outputs, i.e. values without use or non TFG ops, add
  // unshard op after them
  // TODO(itex): handle non TFG ops
  for (Operation& op : block->getOperations()) {
    for (auto result : op.getResults()) {
      if (result.use_empty()) {
        end_ops.push_back(&op);
        break;
      }
    }
  }
  for (auto op : end_ops) {
    for (auto result : op->getResults()) {
      if (result.use_empty()) {
        insertUnshardOpAfter(op, result);
      }
    }
  }
}

bool TFGtoHSConversion::cancelShardUnshardPair(mlir::Block* block) {
  bool changed = false;
  std::vector<std::pair<Operation*, Operation*>> pairs;
  for (Operation& op : block->getOperations()) {
    if (auto shard_op = llvm::dyn_cast<ShardOp>(op)) {
      if (shard_op.getResult().hasOneUse()) {
        for (auto user : shard_op.getResult().getUsers()) {
          if (llvm::dyn_cast<UnshardOp>(user)) {
            pairs.emplace_back(std::make_pair(&op, user));
            break;
          }
        }
      }
    }
  }
  for (auto pair : pairs) {
    pair.first->getResult(0).replaceAllUsesWith(pair.first->getOperand(0));
    pair.first->erase();
    pair.second->getResult(0).replaceAllUsesWith(pair.second->getOperand(0));
    pair.second->erase();
    changed = true;
  }
  return changed;
}

/**
 * @brief Insert PostOp for all tensor outputs of it.
 *
 * @param op The framework op
 */
void TFGtoHSConversion::handleGenericOp(Operation* op) {
  for (auto indexed_result : llvm::enumerate(op->getResults())) {
    op->setAttr(HSDialect::getHspAttrKeyFor(indexed_result.index()),
                ShardingPropertyAttr::get(
                    op->getContext(),
                    createShardingPropertyForType(
                        indexed_result.value().getType(), device_info_)));
  }
}

// void TFGtoHSConversion::mergeReadResourceOP(mlir::tfg::GraphOp* graph_op) {
//   // This function is designed to merge ReadResourceOP(So far only
//   `tfg.ReadVariableOp` is included)
//   // that are functionally equivalent, thereby can be merged into one to
//   reduce the number of cross-device
//   // control edges on the multi-Device.
//   // If two ReadResourceOPs are equivalent, the following requirements must
//   be met:
//   //    1. The two OPs use the same resource value.
//   //    2. The two OPs have the same control edge.
//   auto get_resource_from_readvar = []( mlir::Operation* op ) ->
//   mlir::Operation* {
//     // Find the real resouce_op. Some other nodes(eg: `tfg.Identity`) may
//     exist between ReadResourceOP and resouce_op. TF_ASSERT(
//     op->getName().getStringRef() == "tfg.ReadVariableOp", "Expect this is
//     ReadVariableOp OP" ); TF_ASSERT( op->getNumOperands() != 0, "Expect
//     ReadVariableOp OP has operand." ); mlir::Operation* resource_op = op;
//     while( resource_op->getNumOperands() != 0 ){
//       resource_op = resource_op->getOperand(0).getDefiningOp();
//     }
//     // Assume the results of `resource_op` need to be resource tensor except
//     for the control edge. TF_ASSERT( std::all_of(
//     resource_op->getResultTypes().begin(),
//                             resource_op->getResultTypes().end() - 1,
//                             isResourceTensor ), "Expect the last OP we get
//                             from ReadResourceOPs is resource Op");
//     return resource_op;
//   };
//   std::unordered_map<Operation*, std::vector<mlir::Operation*>>
//   resource_readvars_mp; auto& region = graph_op->getRegion(); for (Block
//   &block : region.getBlocks()) {
//     for (Operation& op : block.getOperations()) {
//       if( op.getName().getStringRef() == "tfg.ReadVariableOp" ) {
//         mlir::Operation* resource_op = get_resource_from_readvar( &op );
//         resource_readvars_mp[ resource_op ].push_back( &op );
//       }
//     }
//   }
//   auto same_control_edges = []( mlir::Operation* read_op_1, mlir::Operation*
//   read_op_2 ) -> bool {
//     auto num_1 = read_op_1->getNumOperands();
//     auto num_2 = read_op_2->getNumOperands();
//     if( num_1 != num_2 ) {
//       return false;
//     }
//     TF_ASSERT( read_op_1->getNumOperands() >= 1, "Expect ReadVariableOp OP
//     has at least one operand" ); std::vector<mlir::Value> operands; for( auto
//     i = 1; i < num_1; i++ ) {
//       operands.push_back( read_op_1->getOperand(i) );
//     }
//     for( auto i = 1; i < num_2; i++ ) {
//       if( std::find( operands.begin(), operands.end(),
//       read_op_2->getOperand(i) )
//       != operands.end() ) {
//         return false;
//       }
//     }
//     return true;
//   };
//   for( auto& resource_readvars_pair : resource_readvars_mp ){
//     auto&& readvars_vec = resource_readvars_pair.second;
//     size_t readvar_num = readvars_vec.size();
//     for( int16_t  i = readvar_num - 1; i >= 0; i-- ) {
//       for( size_t  j = 0; j < i; j++ ) {
//         if( same_control_edges( readvars_vec[i], readvars_vec[j] ) ) {
//           TF_ASSERT( readvars_vec[i]->getNumResults() == 2, "Expect
//           ReadVariableOp OP has two results" ); TF_ASSERT(
//           readvars_vec[j]->getNumResults() == 2, "Expect ReadVariableOp OP
//           has two results" );
//           readvars_vec[i]->getResult(0).replaceAllUsesWith(readvars_vec[j]->getResult(0));
//           readvars_vec[i]->getResult(1).replaceAllUsesWith(readvars_vec[j]->getResult(1));
//           auto erase_duplicate_operand = [&]( Operation* op ) -> void {
//             size_t id = 1;
//             while( id != op->getNumOperands() ){
//               Value pre = op->getOperand( id - 1 );
//               Value now = op->getOperand( id );
//               if( pre == now ){
//                 op->eraseOperand( id );
//               }else{
//                 id++;
//               }
//             }
//           };
//           for( auto user : readvars_vec[j]->getResult(0).getUsers() ) {
//             erase_duplicate_operand( user );
//           }
//           for( auto user : readvars_vec[j]->getResult(1).getUsers() ) {
//             erase_duplicate_operand( user );
//           }
//           TF_ASSERT( readvars_vec[i]->use_empty(), "Expect no uses before
//           incorporating ReadVariableOp OP" ); readvars_vec[i]->erase();
//           break;
//         }
//       }
//     }
//   }
// }

void TFGtoHSConversion::handleGraphOp(mlir::tfg::GraphOp* graph_op) {
  // The merge ReadVariableOp will be included by Tensorflow's Dependency
  // Optimizer. So temporarily disable the optimization in AutoShard.
  // mergeReadResourceOP( graph_op );
  graph_op->getOperation()->setAttr(
      HSDialect::getDeviceInfoAttrKey(),
      DeviceInfoAttr::get(graph_op->getContext(), device_info_));
  auto& region = graph_op->getRegion();
  for (Block& block : region.getBlocks()) {
    std::vector<mlir::Operation*> tfg_ops;
    for (Operation& op : block.getOperations()) {
      // Ignore ops not under tfg.
      if (op.getName().getDialectNamespace() != "tfg") {
        continue;
      }
      tfg_ops.push_back(&op);
    }
    // TODO(itex): Mark ops with a supporting device set,
    //       e.g., some ops can only run on host and some devices only support a
    //       limited set of ops.
    for (auto op : tfg_ops) {
      for (auto result : op->getResults()) {
        // For input ops: tfg.Placeholder and tfg.Const, insert hs.shard_op on
        // their defined tensors.
        if (opNeedUnshard(op)) {
          auto prop =
              createShardingPropertyForType(result.getType(), device_info_);
          // Do single split only for scalar (0-rank) tensors or
          // rank=-1 meaning unranked tensor types or non tensor types (e.g.,
          // control type)
          if (rank(result.getType()) <= 0) {
            prop->splitSingleOnly();
          }
          insertShardOpAfter(op, result,
                             ShardingPropertyAttr::get(op->getContext(), prop));
        }
      }
      if (opNeedUnshard(op)) {
        // Insert an unshard op on each operand to make sure shard and unshard
        // in pairs.
        insertUnshardOpBefore(op);
      } else {
        handleGenericOp(op);
      }
    }

    addUnshardToGraphOutputs(&block);

    // Final cleanup: Neighboring pairs of ShardOp and UnshardOp are cancelled.
    bool changed = false;
    do {
      changed = cancelShardUnshardPair(&block);
    } while (changed);
  }
}

void TFGtoHSConversion::runOnOperation() {
  auto this_op = getOperation();
  fixFusedBatchNormV3Training(this_op);
  if (auto graph_op = llvm::dyn_cast<mlir::tfg::GraphOp>(*this_op)) {
    if (checkGraphShapeInferenceException(&graph_op) == true) {
      ITEX_LOG(WARNING)
          << "Graph has shape inference exception, skip tfg_to_hs pass.";
      return;
    }
    if (sharding_config_.isNeedDeadNodePrune()) {
      DeadNodePrune(&graph_op);
    }
    handleGraphOp(&graph_op);
    return;
  }
  // TODO(itex): handle further nested sub-graphs
  for (Region& region : this_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (auto graph_op = llvm::dyn_cast<mlir::tfg::GraphOp>(op)) {
          if (checkGraphShapeInferenceException(&graph_op) == true) {
            ITEX_LOG(WARNING)
                << "Graph has shape inference exception, skip tfg_to_hs pass.";
            return;
          }
          if (sharding_config_.isNeedDeadNodePrune()) {
            DeadNodePrune(&graph_op);
          }
          handleGraphOp(&graph_op);
        }
      }
    }
  }
}

}  // namespace hs
}  // namespace mlir
