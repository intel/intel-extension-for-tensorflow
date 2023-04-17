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

#include "xpuautoshard/tensorflow/passes/type_inference.h"

#include <bitset>
#include <queue>
#include <unordered_map>
#include <vector>

#include "itex/core/utils/status.h"
#include "protos/op_performance_data.pb.h"
#include "xpuautoshard/common/mlir/passes/pass_utils.h"
#include "xpuautoshard/tensorflow/macro.h"

namespace mlir {
namespace tfg {

using itex::OpInfo_TensorProperties;
using itex::Status;
using mlir::hs::getConstantIntArrayForValue;
using mlir::hs::getRankedTensorType;
using mlir::hs::kDeviceAttr;
using mlir::hs::kMlirNameAttr;

TypeInference::~TypeInference() = default;

bool TypeInference::forwardOp(Operation* op) {
  bool changed = false;
  auto node_name =
      op->getAttrOfType<StringAttr>(kMlirNameAttr).getValue().str();
  std::vector<OpInfo_TensorProperties> out_props;
  Status status = properties_->GetOutputProperties(node_name, &out_props);
  if (!status.ok()) {
    ITEX_LOG(WARNING) << "Error get ouput properties from GraphProperties"
                      << status;
  }
  for (size_t id = 0; id < out_props.size(); id++) {
    if (out_props[id].shape().unknown_rank()) {
      continue;
    }
    std::vector<int64_t> shape;
    int64_t rank = out_props[id].shape().dim_size();
    for (size_t dim = 0; dim < rank; dim++) {
      shape.push_back(out_props[id].shape().dim(dim).size());
    }
    auto tensor_type = op->getResult(id).getType().dyn_cast<TensorType>();
    TF_ASSERT(tensor_type, "Expect result of OP to be of tensor type");
    auto element_type = tensor_type.getElementType();
    auto type = op->getResult(id).getType();
    type = getRankedTensorType(rank, element_type, shape);
    if (op->getResult(id).getType() != type) {
      op->getResult(id).setType(type);
      changed = true;
    }
  }
  return changed;
}

/**
 * The value of some Const OP may be [-1], and the specific value will only
 * be infered at runtime. But when this kind of Const OP has multiple users
 * and will derive different values at runtime, this will cause problems
 * for AutoShard. Therefore, in order to eliminate these implicit inference,
 * this type of Const OP will be copied here to give each user a one-to-one
 * Const OP that does not interfere with each other.
 */
void TypeInference::eliminateConstValueImplicitInfer() {
  // 1. Get Const OP set with [-1] value.
  std::vector<Operation*> eliminate_const_ops;
  auto graph_op = getOperation();
  auto& region = graph_op.getRegion();
  for (Block& block : region.getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (op.getName().getStringRef() != "tfg.Const") {
        continue;
      }
      auto value_vec = getConstantIntArrayForValue(op.getResult(0));
      if (value_vec.size() != 1 || value_vec[0] != -1) {
        continue;
      }
      eliminate_const_ops.push_back(&op);
    }
  }
  auto copy_suffix = "_eliminate_const_implicit_";
  for (auto op : eliminate_const_ops) {
    if (op->getUsers().empty()) {
      continue;
    }
    auto result = op->getResult(0);
    auto copy_id = 1;
    // 2. Get the user set of this Const OP.
    std::vector<Operation*> users;
    for (auto user : result.getUsers()) {
      users.push_back(user);
    }
    for (auto user : users) {
      // 3. If user is Reshape OP, set the shape of Reshape OP result as the
      // value of the copy Const OP. Default `const_values` is [-1].
      std::vector<int32_t> const_values = {-1};
      if (user->getName().getStringRef() == "tfg.Reshape") {
        auto reshape_type = user->getResult(0).getType().dyn_cast<TensorType>();
        TF_ASSERT(reshape_type,
                  "Expect Reshape OP result of OP to be of tensor type");
        const_values.assign(reshape_type.getShape().begin(),
                            reshape_type.getShape().end());
      }
      // 4. Copy this Const OP for every user.
      Operation* copy_const_op;
      {
        OperationState op_state(op->getLoc(), "tfg.Const");
        OpBuilder builder(op);
        // Set dtype and value attr.
        auto element_type = builder.getI32Type();
        op_state.addAttribute("dtype", TypeAttr::get(element_type));
        // Set shape.
        // Keep consistent with the original rank and shape.
        // If the user is the Reshape OP, it will be consistent with the output
        // shape of the Reshape OP.
        auto shaped_type =
            getRankedTensorType(/*rank =*/{}, element_type, /*shape =*/{});
        if (auto output_ranked_type =
                op->getResult(0).getType().dyn_cast<mlir::RankedTensorType>()) {
          shaped_type = getRankedTensorType(
              /*rank =*/output_ranked_type.getRank(), element_type,
              /*shape =*/output_ranked_type.getShape());
        }
        if (const_values != std::vector<int32_t>{-1}) {
          shaped_type =
              getRankedTensorType(/*rank =*/1, element_type, /*shape =*/{1});
        }
        op_state.types.push_back(shaped_type);
        // Set value.
        op_state.addAttribute(
            "value",
            DenseElementsAttr::get(
                shaped_type, llvm::ArrayRef(/*const value*/ const_values)));
        // Build and set _mlir_name attr.
        auto ori_op_name =
            op->getAttrOfType<StringAttr>(kMlirNameAttr).getValue().str();
        auto mlir_name = ori_op_name + copy_suffix + std::to_string(copy_id++);
        op_state.addAttribute(kMlirNameAttr, builder.getStringAttr(mlir_name));
        // Build and set device attr.
        auto device_name =
            op->getAttrOfType<StringAttr>(kDeviceAttr).getValue().str();
        op_state.addAttribute(kDeviceAttr, builder.getStringAttr(device_name));
        // Output control edge
        op_state.types.push_back(
            mlir::tfg::ControlType::get(builder.getContext()));
        // Build Copy OP.
        copy_const_op = builder.create(op_state);
      }
      // 5. Replace original Const OP with copy operation value.
      user->replaceUsesOfWith(result, copy_const_op->getResult(0));
    }
    // 6. this op is replaced with copy op, need to be erased.
    TF_ASSERT(op->getUses().empty(), "Expect no use to erase this const op");
    op->erase();
  }
}

void TypeInference::runOnOperation() {
  auto graph_op = getOperation();
  auto& region = graph_op.getRegion();
  // TODO(itex): add backward pass for better efficiency
  bool changed;
  do {
    changed = false;
    std::queue<mlir::Operation*> ready_queue;
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (op.getNumOperands() == 0) {
          ready_queue.push(&op);
        }
      }
    }
    // assuming DAG
    std::unordered_map<Operation*, size_t> num_operands_visited;
    while (!ready_queue.empty()) {
      Operation* op = ready_queue.front();
      ready_queue.pop();
      changed |= forwardOp(op);
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
  } while (changed);

  // Check shape inference for exceptions.
  if (auto this_graph_op = llvm::dyn_cast<mlir::tfg::GraphOp>(*graph_op)) {
    if (mlir::hs::checkGraphShapeInferenceException(&this_graph_op) == true) {
      ITEX_LOG(WARNING)
          << "Graph has shape inference exception, skip type inference pass.";
      return;
    }
  }

  eliminateConstValueImplicitInfer();
}

}  // namespace tfg
}  // namespace mlir
