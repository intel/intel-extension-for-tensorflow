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

#include "xpuautoshard/tensorflow/passes/hs_to_tfg.h"

#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "itex/core/ir/dialect.h"
#include "xpuautoshard/common/mlir/passes/pass_utils.h"
#include "xpuautoshard/tensorflow/macro.h"

namespace mlir {
namespace hs {

using as::AllReduceL2PostOp;
using as::AllReduceMaxPostOp;
using as::AllReduceMinPostOp;
using as::AllReduceProdPostOp;
using as::AllReduceSumPostOp;
using as::Device;
using as::DeviceId;
using as::ShapeSlicePostOp;
using as::ShardDescGroup;
using as::ShardingPropertyGrouper;
using as::ShardingPropertyRef;
using as::SlicePostOp;
using as::SplitSpec;
using as::WeightedScalePostOp;

namespace {

// TODO(itex): move the util functions to tfg_pass_utils.h

std::string getHostDeviceName() { return "CPU:0"; }

/**
 * @brief Some TF ops can only run on the host device. Then we have
 * to place them on the host device only regardless of their preferred
 * device placement. This function returns true on these ops.
 *
 * @param op
 * @return true
 * @return false
 */
bool isHostOnlyOp(Operation* op) {
  // Assume host only ops include op set with `CPU` device.
  // As much as possible, ops will be assigned to `GPU` after Itex optimization.
  // An op assigned to a CPU is determined not to run on another device, eg:
  // GPU, XPU.
  if (op->hasAttrOfType<StringAttr>(kDeviceAttr)) {
    auto device_name =
        op->getAttrOfType<StringAttr>(kDeviceAttr).getValue().str();
    if (device_name.find("CPU") != std::string::npos) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Guess device name for value according to the device property in TFG
 * op-> Check its defining op first, then users. If device property not found,
 * CPU is assumed.
 *
 * @param v
 * @return std::string
 */
std::string getDeviceNameForValue(Value v) {
  std::string device_name;
  auto defining_op = v.getDefiningOp();
  if (defining_op->hasAttrOfType<StringAttr>(kDeviceAttr)) {
    device_name =
        defining_op->getAttrOfType<StringAttr>(kDeviceAttr).getValue().str();
  } else {
    for (auto user : v.getUsers()) {
      if (user->hasAttrOfType<StringAttr>(kDeviceAttr)) {
        device_name =
            user->getAttrOfType<StringAttr>(kDeviceAttr).getValue().str();
        if (!device_name.empty()) {
          break;
        }
      }
    }
  }
  if (!device_name.empty()) {
    return device_name;
  } else {
    return getHostDeviceName();
  }
}

std::string getMlirDeviceName(const std::string& device_name) {
  if (device_name.find("/") != std::string::npos) {
    return device_name;
  } else {
    return std::string(DEVICE_PREFIX) + device_name;
  }
}

/**
 * @brief Get the control dependency on the op that
 * defines the given value `v`.
 *
 * @param v
 * @return Value
 */
Value getControlEdgeForValue(Value v) {
  auto defining_op = v.getDefiningOp();
  for (auto result : defining_op->getResults()) {
    if (result.getType().dyn_cast<tfg::ControlType>()) {
      return result;
    }
  }
  return Value();
}

/**
 * @brief Find the first split values that the given `v` depend on in the data
 * flow. These values are defined by tfg.SplitV that shares the same use with a
 * Shardop->. Input values are marked as split in multistage, so the OP defining
 * the value must contain operands that are also marked as split. In general,
 * the operand of the input value can come from kCommunicationOP (Assume that
 * tfg.SplitV must come before the Communication OP in the topological order.
 * Communication OP will make split value into single split value), but there
 * must be an operand from ShardOP.
 *
 * @param v
 * @return std::vector<Value>
 */
std::vector<Value> findEntrySplitValues(Value v) {
  // TODO(itex): simplify the following logic with a data flow visitor
  // the hsp propagator can also be replaced with data flow visitor
  std::vector<Value> communication_values;
  auto find_entry_split_value =
      [&communication_values](
          const std::vector<Value>& values,
          bool communication_op_stop) -> std::vector<Value> {
    // Search the operand values from the input values through the BFS
    // algorithm:
    //    1. If the searched value comes from `kCommunicationOP`, it indicates
    //    that
    //      this value is single split only, and this value does not belong to
    //      `entry_values`, so the operand values are no longer searched. If
    //      `communication_op_stop` is true, the operand value will no longer be
    //      searched, and the value will be recorded in `communication_values`.
    //    2. If the searched value comes from the two layers of tfg.SplitV
    //    converted
    //      by shardOP, record the value in entry_values.
    std::vector<Value> entry_values;
    std::queue<Value> values_to_visit;
    std::unordered_set<Operation*> visited_ops;
    for (auto value : values) {
      values_to_visit.push(value);
    }
    while (!values_to_visit.empty()) {
      Value v = values_to_visit.front();
      Operation* op = v.getDefiningOp();
      values_to_visit.pop();
      if (visited_ops.find(op) != visited_ops.end()) {
        continue;
      }
      visited_ops.insert(op);
      bool found = false;
      if (op->getName().getStringRef() == "tfg.SplitV") {
        if (op->getNumOperands() != 0) {
          Operation* pre_first_op = op->getOperand(0).getDefiningOp();
          if (pre_first_op->getName().getStringRef() == "tfg.SplitV") {
            for (auto user : pre_first_op->getOperand(0).getUsers()) {
              if (auto shard_op = llvm::dyn_cast<ShardOp>(user)) {
                found = true;
                break;
              }
            }
          }
        }
      }
      if (kCommunicationOP.find(op->getName().getStringRef().str()) !=
              kCommunicationOP.end() ||
          op->getName().getStringRef() == "tfg.ConcatV2") {
        if (communication_op_stop) {
          for (auto operand : op->getOperands()) {
            // Control edge value input is not considered
            if (!operand.getType().isa<tfg::ControlType>()) {
              communication_values.push_back(operand);
            }
          }
        }
        continue;
      }
      if (found) {
        entry_values.push_back(v);
      }
      if (!found) {
        for (auto operand : op->getOperands()) {
          // we only track data dependency, not control dependency
          if (!operand.getType().isa<tfg::ControlType>()) {
            values_to_visit.push(operand);
          }
        }
      }
    }
    return entry_values;
  };

  std::vector<Value> all_entry_values =
      find_entry_split_value({v}, /*communication_op_stop*/ true);
  std::vector<Value> comm_entry_values = find_entry_split_value(
      communication_values, /*communication_op_stop*/ false);
  std::vector<Value> final_entry_values;
  // Among the entry values quoted by the input value, exclude the entry values
  // from `kCommunicationOP`, and the rest are the required `entry_values`.
  for (auto value : all_entry_values) {
    if (std::find(comm_entry_values.begin(), comm_entry_values.end(), value) ==
        comm_entry_values.end()) {
      final_entry_values.push_back(value);
    }
  }
  // The input value must have an operand from ShardOP.
  // Therefore, `final_entry_values` should not be empty.
  if (final_entry_values.size() == 0) {
    llvm::outs() << "Srror: entry_values not found\n";
  }
  return final_entry_values;
}

/**
 * @brief Add control dependency `control_edge` to all the ops that use `v` by
 * adding `control_edge` as the input of these ops, making sure that `v` is
 * consumed after `control_edge` is available.
 *
 * @param v
 * @param control_edge
 */
void setControlEdgeForValue(Value v, Value control_edge) {
  // Sdd control dependency to the users of `Shardop->` value.
  auto define_op = v.getDefiningOp();
  std::unordered_set<Operation*> qualified_ops;
  if (define_op->getName().getStringRef() == "tfg.SplitV") {
    for (auto op : v.getUsers()) {
      if (std::find(op->operand_begin(), op->operand_end(), control_edge) ==
          op->operand_end()) {
        qualified_ops.insert(op);
      }
    }
  }
  for (auto op : qualified_ops) {
    op->insertOperands(op->getNumOperands(), control_edge);
  }
}

/**
 * @brief Check if dest is reachable from src, assuming we
 * are working on DAG without cycles.
 *
 * @param src
 * @param dest
 * @return true
 * @return false
 */
bool reachable(Value src, Value dest) {
  std::queue<Operation*> ops;
  std::unordered_map<Operation*, Value> visited_ops;
  auto push_users = [&](Value v) {
    for (auto op : v.getUsers()) {
      if (visited_ops.find(op) == visited_ops.end()) {
        ops.push(op);
        visited_ops.insert({op, v});
      }
    }
  };
  push_users(src);
  while (!ops.empty()) {
    auto op = ops.front();
    ops.pop();
    for (auto result : op->getResults()) {
      if (result == dest) {
        return true;
      }
      push_users(result);
    }
  }
  return false;
}

/**
 * @brief Mark the end of a multi-stage region, e.g.,
 * add control dependencies among multiple stages belonging to the same device
 *
 * @param begin
 * @param end
 */
void multiStageJoin(ShardedValue::const_iterator begin,
                    ShardedValue::const_iterator end) {
  for (auto offset = begin; offset < end - 1; offset++) {
    Value control_edge = getControlEdgeForValue(*offset);
    std::vector<Value> entry_values = findEntrySplitValues(*(offset + 1));
    for (auto entry_value : entry_values) {
      // Only when the `control_edge` cannot be reached from the `entry_value`,
      // the stage can be joined, otherwise a cycle will be formed.
      // When the connection cannot be made through the control edge, it will
      // skip.
      // TODO(itex): A better and efficient algorithm join stage is needed.
      if (reachable(entry_value, control_edge) == false) {
        setControlEdgeForValue(entry_value, control_edge);
      }
    }
  }
}

}  // anonymous namespace

HStoTFGConversion::~HStoTFGConversion() = default;

std::string HStoTFGConversion::getDeviceName(DeviceId id) {
  if (id == Device::CPU_HOST_ID) {
    return getHostDeviceName();
  } else {
    return current_device_info_.getDevice(id).getName();
  }
}

std::string HStoTFGConversion::getNextCollectiveKey() {
  return std::to_string(collective_id_++);
}

std::string HStoTFGConversion::getNextOpNameIdStr() {
  return std::to_string(op_name_id_++);
}

std::string HStoTFGConversion::createShardedTfgOpNameFor(
    const std::string& device_name, const std::string& op_name) {
  std::string encoded_device_name(device_name);
  // if the device name is already in full path encoding, e.g.,
  // "/job:localhost/replica:0/task:0/device:GPU:0", we find the
  // short name, i.e., "GPU:0"
  auto slash_rpos = encoded_device_name.rfind("/device:");
  if (slash_rpos != std::string::npos) {
    encoded_device_name =
        encoded_device_name.substr(slash_rpos + std::string("/device:").size());
  }
  // op name should not contain colon, replace them with underscores.
  std::string::size_type pos;
  while ((pos = encoded_device_name.find(":")) != std::string::npos) {
    encoded_device_name.replace(pos, 1, "_");
  }
  const std::string key(encoded_device_name + op_name);
  if (device_op_to_id_map_.find(key) == device_op_to_id_map_.end()) {
    device_op_to_id_map_[key] = 0;
  }
  if (created_op_names_.find(op_name) != created_op_names_.end()) {
    return "AutoShard/" + encoded_device_name + "/" + op_name + "/" +
           std::to_string(device_op_to_id_map_[key]++);
  } else {
    created_op_names_.insert(op_name);
    return op_name;
  }
}

void HStoTFGConversion::setCurrentDeviceInfo(mlir::tfg::GraphOp* graph_op) {
  auto device_info_attr =
      graph_op->getOperation()->getAttrOfType<DeviceInfoAttr>(
          HSDialect::getDeviceInfoAttrKey());
  current_device_info_ = device_info_attr.getDeviceInfo();
}

Operation* HStoTFGConversion::buildTfgOp(
    OpBuilder* builder, Location loc, const std::string& op_name,
    Type* tensor_type, const std::string& device_name,
    const std::function<void(OperationState* op_state)> op_state_init,
    bool need_output_control_edge) {
  OperationState op_state(loc, op_name);
  op_state_init(&op_state);
  if (tensor_type) {
    if (auto shaped_type = tensor_type->dyn_cast<ShapedType>()) {
      op_state.addAttribute("T", TypeAttr::get(shaped_type.getElementType()));
    }
  }
  // add _mlir_name attribute if the op_state_init not defined it yet.
  // the default name would be the op name without "tfg." prefix but with
  // "_AutoShard_#n" suffix to avoid name clashing.
  Attribute mlir_name_attr;
  for (auto attr : op_state.attributes) {
    if (attr.getName().str() == kMlirNameAttr) {
      mlir_name_attr = attr.getValue();
      break;
    }
  }
  std::string mlir_name;
  if (mlir_name_attr) {
    op_state.attributes.erase(kMlirNameAttr);
    mlir_name = mlir_name_attr.dyn_cast<StringAttr>().getValue().str();
  } else {
    if (op_name.find("tfg.") == 0) {
      mlir_name = op_name.substr(4) + "_AutoShard_" + getNextOpNameIdStr();
    }
  }
  op_state.addAttribute(kMlirNameAttr,
                        builder->getStringAttr(
                            createShardedTfgOpNameFor(device_name, mlir_name)));
  op_state.addAttribute(kDeviceAttr,
                        builder->getStringAttr(getMlirDeviceName(device_name)));
  if (need_output_control_edge) {
    op_state.types.push_back(
        mlir::tfg::ControlType::get(builder->getContext()));
  }
  return builder->create(op_state);
}

template <>
Value HStoTFGConversion::buildConstantTensor(OpBuilder* builder, Location loc,
                                             const std::string& device_name,
                                             std::vector<int64_t> tensor,
                                             std::vector<int64_t> shape) {
  // NOTE: int64_t constants seems not working with some TF ops like SplitV or
  // Concat. So, we use int32_t here.
  std::vector<int32_t> int32_vec;
  for (auto i64 : tensor) {
    int32_vec.push_back(i64);
  }
  return buildTfgOp(
             builder, loc, "tfg.Const", nullptr, device_name,
             [&](OperationState* const_op_state) -> void {
               auto element_type = builder->getI32Type();
               auto shaped_type =
                   getRankedTensorType(shape.size(), element_type, shape);
               const_op_state->types.push_back(shaped_type);
               const_op_state->addAttribute("dtype",
                                            TypeAttr::get(element_type));
               const_op_state->addAttribute(
                   "value", DenseElementsAttr::get(
                                shaped_type, llvm::makeArrayRef(int32_vec)));
             })
      ->getResult(0);
}

template <>
Value HStoTFGConversion::buildConstantTensor(OpBuilder* builder, Location loc,
                                             const std::string& device_name,
                                             std::vector<float> tensor,
                                             std::vector<int64_t> shape) {
  return buildTfgOp(
             builder, loc, "tfg.Const", nullptr, device_name,
             [&](OperationState* const_op_state) -> void {
               auto element_type = builder->getF32Type();
               auto shaped_type =
                   getRankedTensorType(shape.size(), element_type, shape);
               const_op_state->types.push_back(shaped_type);
               const_op_state->addAttribute("dtype",
                                            TypeAttr::get(element_type));
               const_op_state->addAttribute(
                   "value", DenseElementsAttr::get(shaped_type,
                                                   llvm::makeArrayRef(tensor)));
             })
      ->getResult(0);
}

Value HStoTFGConversion::buildConstant(OpBuilder* builder, Location loc,
                                       const std::string& device_name,
                                       int64_t v) {
  return buildConstantTensor(builder, loc, device_name,
                             std::vector<int64_t>(1, v));
}

Value HStoTFGConversion::buildConstant(OpBuilder* builder, Location loc,
                                       const std::string& device_name,
                                       size_t v) {
  return buildConstantTensor(builder, loc, device_name,
                             std::vector<int64_t>(1, v));
}

Value HStoTFGConversion::buildConstant(OpBuilder* builder, Location loc,
                                       const std::string& device_name,
                                       float v) {
  return buildConstantTensor(builder, loc, device_name,
                             std::vector<float>(1, v));
}

std::vector<Value> HStoTFGConversion::buildFrameworkOpOperands(
    OpBuilder* builder, Location loc,
    const std::vector<ShardedValue>& sharded_operands,
    const ShardDescGroup& shard_desc_group,
    std::unordered_map<std::string, Value>& device_noop_mp) {
  // For each operand of lower framework OP on the current device:
  //  1. If it is data dependent, add it to the operand.
  //  2. If the control edge from the current device, add it to the operand.
  //  3. If there is only one control edge from another device, add it to the
  //  operand.
  //  4. If there are more than one control edges from another device,
  //     a. when you can't use last optimizer result, insert a `tfg.NoOp`
  //     connecting all control
  //        edges on another device, record the NoOp into `device_noop_mp`
  //        according to the device information, and add the NoOP's control edge
  //        to the operand.
  //     b. when you can use last optimizer result, it indicates that the
  //     operands of the current
  //        lower framework OP are the same as the last lower framework OP in
  //        the neighborhood. You can directly use NoOPs in `device_noop_mp`
  //        without rebuilding it again.
  auto current_device =
      getMlirDeviceName(getDeviceName(shard_desc_group.getDeviceId()));
  std::vector<Value> new_operands;
  std::unordered_map<std::string, std::vector<Value>> control_edges_per_device;
  for (size_t operand_idx = 0; operand_idx < sharded_operands.size();
       operand_idx++) {
    auto operand =
        sharded_operands[operand_idx][shard_desc_group[operand_idx].getNum()];
    auto operand_device = getDeviceNameForValue(operand);
    // Optimize the cross-device control edge operand by adding NoOp that meet
    // the following requirements:
    // 1. Operand is control edge.
    // 2. Operand and OP are not on the same device.
    // 3. The number of operands from the same device is greater than one.
    if (current_device == operand_device ||
        !(operand.getType().dyn_cast<tfg::ControlType>())) {
      new_operands.emplace_back(operand);
    } else {
      control_edges_per_device[operand_device].push_back(operand);
    }
  }
  for (auto& control_edges : control_edges_per_device) {
    auto operand_device = control_edges.first;
    TF_ASSERT(
        operand_device != current_device,
        "Except get control edges in other device ranther than op device");
    if (control_edges.second.size() == 1) {
      new_operands.push_back(control_edges.second[0]);
      continue;
    }
    if (device_noop_mp.count(operand_device) == 0) {
      Operation* noop_op =
          buildTfgOp(builder, loc, "tfg.NoOp", nullptr, operand_device,
                     [&](OperationState* noop) -> void {
                       for (auto operand : control_edges.second) {
                         noop->operands.push_back(operand);
                       }
                     });
      device_noop_mp[operand_device] = noop_op->getResult(0);
    }
    new_operands.push_back(device_noop_mp[operand_device]);
  }
  return new_operands;
}

Operation* HStoTFGConversion::buildFrameworkOp(
    OpBuilder* builder, Operation* template_op,
    const std::vector<Value>& operands, const std::vector<Type>& result_types,
    const std::string& device_name) {
  return buildTfgOp(
      builder, template_op->getLoc(),
      template_op->getName().getStringRef().str(), nullptr,
      isHostOnlyOp(template_op) ? getHostDeviceName() : device_name,
      [&](OperationState* op_state) -> void {
        op_state->addTypes(result_types);
        op_state->addOperands(operands);
        // TODO(itex): For some ops, attrs might be different after sharding
        // need to revise further.
        for (auto named_attr : template_op->getAttrs()) {
          // device attr will be overridden, no need to copy
          if (named_attr.getName() != kDeviceAttr &&
              !isHsAttr(named_attr.getName().str())) {
            op_state->addAttribute(named_attr.getName(), named_attr.getValue());
          }
        }
      },
      /*need_output_control_edge=*/false);
}

ShardedValue HStoTFGConversion::lowerShardOp(ShardOp* shard_op) {
  auto&& prop = shard_op->getHspAttr()
                    .dyn_cast<ShardingPropertyAttr>()
                    .getShardingProperty();
  TF_ASSERT(prop->isInitialized(),
            "Expect sharding property to be initialized but not!");
  auto shard_descs = prop->getShardDescriptors();
  OpBuilder builder(shard_op->getOperation());
  ShardedValue sharded_result;
  if (prop->isSplitSingleOnly()) {
    // single split only, each device shares the same value
    sharded_result =
        ShardedValue(prop->getNumLogicalShards(), shard_op->getOperand());
  } else {
    // TODO(itex): support halo split, currently assume plain split.
    // TODO(itex): support multiple split dims, currently assume
    // only single split on batch size dim.
    TF_ASSERT(prop->numMultiSplitDims() == 1,
              "Support maximum 1 split dim for now");
    Type t = shard_op->getOperand().getType();
    auto location = shard_op->getLoc();
    auto num_shards_per_device = prop->getNumLogicalShardsPerdevice();
    auto split_spec = shard_descs[0].getSplitSpec(-1);
    std::string split_op_name = "";
    if (split_spec.getType() == SplitSpec::SplitType::SIZE) {
      split_op_name = "tfg.SplitV";
    } else if (split_spec.getType() == SplitSpec::SplitType::RATIO) {
      split_op_name = "tfg.Split";
    } else {
      TF_ASSERT(
          false,
          "only support `SplitType::SIZE` and `SplitType::RATIO` for now");
    }

    // 1. The first split is on device level
    auto device_name_first_level =
        getDeviceNameForValue(shard_op->getOperand());
    auto splitv_op_first_level = buildTfgOp(
        &builder, location, split_op_name, &t, device_name_first_level,
        [&](OperationState* splitv) -> void {
          size_t num_splits = prop->getNumDevices();
          // add result types: num_splits tensors
          auto shaped_type = t.dyn_cast<ShapedType>();
          TF_ASSERT(shaped_type, "Expect shaped type for splitting");
          for (size_t shard_desc_id = 0, device_id = 0;
               shard_desc_id < shard_descs.size() && device_id < num_splits;
               shard_desc_id += num_shards_per_device[device_id], device_id++) {
            // fetch the first shard_descs of every device
            splitv->types.push_back(
                getRankedTensorType(shard_descs[shard_desc_id].getRank(),
                                    shaped_type.getElementType(),
                                    shard_descs[shard_desc_id].getShape()));
          }
          // add operands: operand of shard_op, split sizes (const 1D tensor),
          // num_splits (const scalar)
          // Build split_dim const value
          auto split_dim_value =
              buildConstant(&builder, location, device_name_first_level,
                            shard_descs[0].getSplitDim(-1));
          if (split_op_name == "tfg.Split") {
            // add `tfg.Split` operand: split_dim
            splitv->operands.push_back(split_dim_value);
          }
          // add `tfg.SplitV` and `tfg.Split` operand: value
          splitv->operands.push_back(shard_op->getOperand());
          if (split_op_name == "tfg.SplitV") {
            // add `tfg.SplitV` operand: size_splits
            std::vector<int64_t> split_sizes(num_splits);
            for (size_t shard_desc_id = 0, device_id = 0;
                 shard_desc_id < shard_descs.size() && device_id < num_splits;
                 shard_desc_id += num_shards_per_device[device_id],
                        device_id++) {
              // get split sizes of every shard
              split_spec = shard_descs[shard_desc_id].getSplitSpec(-1);
              auto beg_offset = split_spec.getSizes().begin();
              split_sizes[device_id] = std::accumulate(
                  beg_offset, beg_offset + num_shards_per_device[device_id], 0);
            }
            splitv->operands.push_back(buildConstantTensor(
                &builder, location, device_name_first_level, split_sizes,
                std::vector<int64_t>(1, split_sizes.size())));
            // add `tfg.SplitV` operand: split_dim
            splitv->operands.push_back(split_dim_value);
            // add `tfg.SplitV` attributes:
            //   Tlen: DT_INT32
            splitv->addAttribute("Tlen", TypeAttr::get(builder.getI32Type()));
          }
          // else if (split_spec.getType() == SplitSpec::SplitType::RATIO) {
          //   // Split by ratio, compute shape dynamically
          //   std::vector<int64_t> split_sizes;
          //   for (auto this_shard_desc : shard_descs) {
          //     split_sizes.push_back(
          //         this_shard_desc.getShape()[this_shard_desc.getSplitDim(-1)]);
          //   }
          //   splitv.operands.push_back(buildConstantTensor(
          //       builder, shard_op->getLoc(), device_name_first_level,
          //       split_sizes,
          //       std::vector<int64_t>(1, split_spec.getRatios().size())));
          // }

          // add `tfg.SplitV` and `tfg.Split` attributes:
          //   num_split: integer value
          splitv->addAttribute("num_split",
                               builder.getI32IntegerAttr(num_splits));
        });
    TF_ASSERT(
        splitv_op_first_level->getResults().size() - 1 == prop->getNumDevices(),
        "Expect split number is device number");

    // 2. The second split is on stage level
    std::vector<Value> results;
    size_t current_device_stage_begin_id = 0;
    for (size_t i = 0; i < num_shards_per_device.size(); i++) {
      auto split_result_first_level = splitv_op_first_level->getResult(i);
      size_t stage_sum = num_shards_per_device[i];
      TF_ASSERT(stage_sum > 0,
                "Expect stage number in every device gather than 0");
      if (stage_sum == 1) {
        results.push_back(split_result_first_level);
        continue;
      }
      auto device_name_second_level = getDeviceName(prop->getDeviceIds()[i]);
      // In order to support the type of splitV, temporarily allocate the second
      // layer of splitV to the CPU.
      auto splitv_op_second_level = buildTfgOp(
          &builder, location, split_op_name, &t, device_name_second_level,
          [&](OperationState* splitv) -> void {
            // add result types: num_splits tensors
            auto shaped_type = t.dyn_cast<ShapedType>();
            TF_ASSERT(shaped_type, "Expect shaped type for splitting");
            for (auto offset = 0; offset < stage_sum; offset++) {
              auto&& shard_desc =
                  shard_descs[current_device_stage_begin_id + offset];
              splitv->types.push_back(getRankedTensorType(
                  shard_desc.getRank(), shaped_type.getElementType(),
                  shard_desc.getShape()));
            }
            // add operands: operand of shard_op, split sizes (const 1D tensor),
            // num_splits (const scalar)
            // Build split_dim const value
            auto split_dim_value = buildConstant(
                &builder, location, device_name_second_level,
                shard_descs[current_device_stage_begin_id].getSplitDim(-1));
            if (split_op_name == "tfg.Split") {
              // add `tfg.Split` operand: split_dim
              splitv->operands.push_back(split_dim_value);
            }
            // add `tfg.SplitV` and `tfg.Split` operand: value
            splitv->operands.push_back(split_result_first_level);
            if (split_op_name == "tfg.SplitV") {
              auto&& split_spec = shard_descs[0].getSplitSpec(-1);
              if (split_spec.getType() == SplitSpec::SplitType::SIZE) {
                std::vector<int64_t> split_sizes;
                for (auto offset = 0; offset < stage_sum; offset++) {
                  size_t size_id = current_device_stage_begin_id + offset;
                  split_sizes.push_back(split_spec.getSizes()[size_id]);
                }
                splitv->operands.push_back(buildConstantTensor(
                    &builder, location, device_name_second_level, split_sizes,
                    std::vector<int64_t>(1, split_sizes.size())));
              } else {
                // TODO(itex): support split by ratio, compute shape dynamically
                TF_ASSERT(false,
                          "only support split by concrete shape for now");
              }
              // add `tfg.SplitV` operand: split_dim
              splitv->operands.push_back(split_dim_value);
              // add `tfg.SplitV` attributes:
              //   Tlen: DT_INT32
              splitv->addAttribute("Tlen", TypeAttr::get(builder.getI32Type()));
            }
            // add `tfg.SplitV` and `tfg.Split` attributes:
            //   num_split: integer value
            splitv->addAttribute("num_split",
                                 builder.getI32IntegerAttr(stage_sum));
          });
      current_device_stage_begin_id += stage_sum;
      auto&& result_num = splitv_op_second_level->getNumResults();
      for (auto j = 0; j < result_num - 1; j++) {
        results.push_back(splitv_op_second_level->getResult(j));
      }
    }
    TF_ASSERT(
        results.size() == prop->getNumLogicalShards(),
        "Expect the results of `tfg.splitV` correspond to the property shards");
    // Add Identity after SplitV to place tensor on the right device explicitly
    size_t shard_num = 0;
    Operation* last_identity_op = nullptr;
    for (size_t i = 0; i < num_shards_per_device.size(); i++) {
      for (size_t shard_num_per_device = 0;
           shard_num_per_device < num_shards_per_device[i];
           shard_num_per_device++, shard_num++) {
        Type result_type = results[shard_num].getType();
        Operation* identity_op =
            buildTfgOp(&builder, location, "tfg.Identity", &result_type,
                       getDeviceName(prop->getDeviceIds()[i]),
                       [&](OperationState* identity) -> void {
                         identity->types.push_back(result_type);
                         identity->operands.push_back(results[shard_num]);
                       });
        results[shard_num] = identity_op->getResult(0);
        if (last_identity_op) {
          identity_op->insertOperands(
              identity_op->getNumOperands(),
              getControlEdgeForValue(last_identity_op->getResult(0)));
        }
        last_identity_op = identity_op;
      }
      last_identity_op = nullptr;
    }
    results.push_back(splitv_op_first_level->getResult(prop->getNumDevices()));
    for (auto result : results) {
      sharded_result.push_back(result);
    }
  }
  std::vector<ShardedValue> sharded_results({std::move(sharded_result)});
  std::vector<ShardedValue> control_edges;
  std::vector<DeviceId> dev_placements;
  for (auto&& shard_desc : shard_descs) {
    dev_placements.push_back(shard_desc.getDeviceId());
  }
  handlePostOps(&builder, shard_op->getOperation(), sharded_results,
                control_edges, ShardingPropertyRefVec({std::move(prop)}),
                dev_placements);
  return sharded_results[0];
}

void HStoTFGConversion::lowerUnshardOp(UnshardOp* unshard_op,
                                       const ShardedValue& sharded_operand) {
  auto prop = getShardingPropertyForValue(unshard_op->getOperand());
  OpBuilder builder(unshard_op->getOperation());
  auto device_name = getDeviceNameForValue(unshard_op->getResult());
  auto shard_descs = prop->getShardDescriptors();
  if (prop->isSplitSingleOnly()) {
    Type t = unshard_op->getOperand().getType();
    // default (single shard or no user of unshard_op): don't insert new op by
    // default
    Value new_result = sharded_operand[0];
    if (sharded_operand.size() > 1 && !unshard_op->getResult().use_empty()) {
      if (t.dyn_cast<tfg::ControlType>()) {
        // Note that the number of control edge might be bigger than the number
        // of shard descriptors. The number of shard descriptors equals to the
        // number of devices since control edge is single split only while
        // control edges could be output from ops sharded with multiple stages.
        TF_ASSERT(shard_descs.size() <= sharded_operand.size(),
                  "Expect number of shard descriptors no larger than number of "
                  "operand shard with control edge");
        // in the case of control edge input, we don't need to
        // explicitly output a new control edge
        Operation* noop_op = buildTfgOp(
            &builder, unshard_op->getLoc(), "tfg.NoOp", &t, device_name,
            [&](OperationState* noop) -> void {
              noop->types.push_back(t);
              noop->operands.push_back(sharded_operand[0]);
              for (size_t i = 0; i < sharded_operand.size(); i++) {
                // for other shards, we should make control dependencies
                if (i != 0) {
                  auto control_edge =
                      getControlEdgeForValue(sharded_operand[i]);
                  TF_ASSERT(control_edge, "Could not find control edge");
                  noop->operands.push_back(control_edge);
                }
              }
            },
            /*need_output_control_edge=*/false);
        new_result = noop_op->getResult(0);
      } else {
        // TODO(itex): here, we assume data are replicate when it is single
        // split only but content could be different with single split only
        // case. We should mark HSPs with "replicate" in the auto sharding
        // pass and assert the "replicate" property here. We propbably
        // don't have to handle non-replicate case here, e.g., get the shape
        // of a split tensor and then unshard on it.
        TF_ASSERT(shard_descs.size() == sharded_operand.size(),
                  "Expect number of shard descriptors matches number of "
                  "operand shard");
        int64_t operand_id = -1;
        for (size_t i = 0; i < shard_descs.size(); i++) {
          // prefer to use the sharded operand having the same device of the
          // result is placed on
          if (getDeviceName(shard_descs[i].getDeviceId()) == device_name) {
            operand_id = i;
            break;
          }
        }
        // otherwise, use the first operand
        if (operand_id < 0) {
          operand_id = 0;
        }
        new_result = sharded_operand[operand_id];
      }
    }
    unshard_op->getResult().replaceAllUsesWith(new_result);
  } else {  // TODO(itex): support halo split, currently assume plain split
    TF_ASSERT(
        sharded_operand.size() == prop->getNumLogicalShards(),
        "Unshard expects the same number of operands as the number of shards");
    // TODO(itex): support multiple split dims, currently assume
    // only single split on batch size dim.
    TF_ASSERT(prop->numMultiSplitDims() == 1,
              "Support maximum 1 split dim for now");
    Type t = unshard_op->getOperand().getType();
    auto concat_op = buildTfgOp(
        &builder, unshard_op->getLoc(), "tfg.ConcatV2", &t, device_name,
        [&](OperationState* concat) -> void {
          size_t num_splits = shard_descs.size();
          // add result types: concatenated tensor
          concat->types.push_back(t);
          // add operands: tensors to concat, axis dim
          for (auto operand : sharded_operand) {
            concat->operands.push_back(operand);
          }
          concat->operands.push_back(
              buildConstant(&builder, unshard_op->getLoc(), device_name,
                            shard_descs[0].getSplitDim(-1)));
          // add attributes:
          //   Tidx: DT_INT32
          //   N: number of tensors
          concat->addAttribute("Tidx", TypeAttr::get(builder.getI32Type()));
          concat->addAttribute("N", builder.getI32IntegerAttr(num_splits));
        });
    unshard_op->getResult().replaceAllUsesWith(concat_op->getResult(0));
  }
}

ShardedValue HStoTFGConversion::lowerReshardOp(
    ReshardOp* reshard_op, const ShardedValue& sharded_operand) {
  // TODO(itex): support ReshardOp lowering
  return ShardedValue();
}

std::vector<ShardedValue> HStoTFGConversion::lowerFrameworkOp(
    Operation* framework_op,
    const std::vector<ShardedValue>& sharded_operands) {
  TF_ASSERT(sharded_operands.size() == framework_op->getNumOperands(),
            "Expect same number of sharded operands as original operands");
  ShardingPropertyRefVec props;
  for (auto operand : framework_op->getOperands()) {
    props.emplace_back(getShardingPropertyForValue(operand));
  }
  for (auto result : framework_op->getResults()) {
    props.emplace_back(getShardingPropertyForValue(result));
  }
  OpBuilder builder(framework_op);
  std::vector<ShardedValue> sharded_results(framework_op->getNumResults());
  ShardingPropertyGrouper prop_grouper(props);
  std::vector<ShardDescGroup> grouped_shard_descs =
      prop_grouper.getShardDescriptorGroups();
  std::vector<DeviceId> dev_placements;
  std::unordered_map<std::string, Value> device_noop_mp;
  auto have_same_control_edge_compared_last_group = [&](size_t id) -> bool {
    // Checks whether the two current framework OP and the last lower framework
    // OP in the neighborhood contain the same control edges in operands.
    if (id <= 0) return true;
    auto&& group_now = grouped_shard_descs[id];
    auto&& group_pre = grouped_shard_descs[id - 1];
    for (size_t operand_idx = 0; operand_idx < sharded_operands.size();
         operand_idx++) {
      auto& operand_now =
          sharded_operands[operand_idx][group_now[operand_idx].getNum()];
      auto& operand_pre =
          sharded_operands[operand_idx][group_pre[operand_idx].getNum()];
      bool is_control_now = operand_now.getType().isa<tfg::ControlType>();
      bool is_control_pre = operand_now.getType().isa<tfg::ControlType>();
      // The operands at the corresponding positions should be of the same type.
      if (is_control_now ^ is_control_pre) {
        return false;
      }
      if (is_control_now && operand_now != operand_pre) {
        return false;
      }
    }
    return true;
  };
  for (auto id = 0; id < grouped_shard_descs.size(); id++) {
    bool use_last_optimizer_noop_flag =
        have_same_control_edge_compared_last_group(id);
    auto&& shard_desc_group = grouped_shard_descs[id];
    auto device_name = getDeviceName(shard_desc_group.getDeviceId());
    if (!use_last_optimizer_noop_flag) {
      device_noop_mp.clear();
    }
    std::vector<Value> new_operands = buildFrameworkOpOperands(
        &builder, framework_op->getLoc(), sharded_operands, shard_desc_group,
        device_noop_mp);
    dev_placements.push_back(shard_desc_group.getDeviceId());
    std::vector<Type> result_types;
    for (size_t result_idx = 0; result_idx < framework_op->getNumResults();
         result_idx++) {
      Type result_type = framework_op->getResult(result_idx).getType();
      auto&& shard_desc =
          shard_desc_group[result_idx + sharded_operands.size()];
      if (auto ranked_type = result_type.dyn_cast<RankedTensorType>()) {
        result_types.push_back(getRankedTensorType(shard_desc.getRank(),
                                                   ranked_type.getElementType(),
                                                   shard_desc.getShape()));
      } else {
        result_types.push_back(result_type);
      }
    }
    Operation* new_framework_op = buildFrameworkOp(
        &builder, framework_op, new_operands, result_types, device_name);
    for (auto indexed_result :
         llvm::enumerate(new_framework_op->getResults())) {
      sharded_results[indexed_result.index()].push_back(indexed_result.value());
    }
  }
  if (grouped_shard_descs.size() == 1) {
    return sharded_results;
  }
  // process post ops, excluding control edge (assuming it is the
  // last result).
  ShardedValue control_edge =
      sharded_results[framework_op->getNumResults() - 1];
  std::vector<ShardedValue> sharded_control_edges(
      framework_op->getNumResults() - 1, control_edge);
  handlePostOps(&builder, framework_op, sharded_results, sharded_control_edges,
                {props.begin() + sharded_operands.size(), props.end() - 1},
                dev_placements);
  // FIXME: the handling of control edge is tricky, an op may
  // have both multi-split type or single split type results with different
  // size of ShardedValue when the split is multi-stage. For simplicity,
  // we output the control edge of one of the post-op if any exists.
  // A complete implementation should output the control edge collected
  // from all the post-ops.
  sharded_results[framework_op->getNumResults() - 1] =
      sharded_control_edges.size() > 0 ? sharded_control_edges[0]
                                       : control_edge;
  return sharded_results;
}

void HStoTFGConversion::handlePostOps(
    OpBuilder* builder, Operation* op,
    std::vector<ShardedValue>& sharded_results,
    std::vector<ShardedValue>& sharded_control_edges,
    const std::vector<ShardingPropertyRef>& result_props,
    const std::vector<DeviceId>& dev_placements) {
  for (int result_idx = 0; result_idx < result_props.size(); result_idx++) {
    auto&& result_prop = result_props[result_idx];
    auto&& post_ops = result_prop->getPostOps();
    for (auto&& post_op : post_ops) {
      if (dynamic_cast<as::AllReduceSumPostOp*>(post_op.get())) {
        std::tie(sharded_results[result_idx],
                 sharded_control_edges[result_idx]) =
            buildAllReduceSumPostOp(
                builder, op->getLoc(), sharded_results[result_idx],
                sharded_control_edges[result_idx], result_prop, dev_placements);
      } else if (dynamic_cast<as::AllReduceMaxPostOp*>(post_op.get())) {
        std::tie(sharded_results[result_idx],
                 sharded_control_edges[result_idx]) =
            buildAllReduceMaxPostOp(
                builder, op->getLoc(), sharded_results[result_idx],
                sharded_control_edges[result_idx], result_prop, dev_placements);
      } else if (dynamic_cast<as::AllReduceMinPostOp*>(post_op.get())) {
        std::tie(sharded_results[result_idx],
                 sharded_control_edges[result_idx]) =
            buildAllReduceMinPostOp(
                builder, op->getLoc(), sharded_results[result_idx],
                sharded_control_edges[result_idx], result_prop, dev_placements);
      } else if (dynamic_cast<as::AllReduceProdPostOp*>(post_op.get())) {
        std::tie(sharded_results[result_idx],
                 sharded_control_edges[result_idx]) =
            buildAllReduceProdPostOp(
                builder, op->getLoc(), sharded_results[result_idx],
                sharded_control_edges[result_idx], result_prop, dev_placements);
      } else if (dynamic_cast<as::AllReduceL2PostOp*>(post_op.get())) {
        std::tie(sharded_results[result_idx],
                 sharded_control_edges[result_idx]) =
            buildAllReduceL2PostOp(
                builder, op->getLoc(), sharded_results[result_idx],
                sharded_control_edges[result_idx], result_prop, dev_placements);
      } else if (auto weighted_scale =
                     dynamic_cast<WeightedScalePostOp*>(post_op.get())) {
        std::tie(sharded_results[result_idx],
                 sharded_control_edges[result_idx]) =
            buildWeightedScalePostOp(
                builder, op->getLoc(), sharded_results[result_idx],
                sharded_control_edges[result_idx], dev_placements,
                weighted_scale->getWeights());
      } else if (dynamic_cast<SlicePostOp*>(post_op.get())) {
        std::tie(sharded_results[result_idx],
                 sharded_control_edges[result_idx]) =
            buildSlicePostOp(builder, op->getLoc(), sharded_results[result_idx],
                             sharded_control_edges[result_idx], result_prop,
                             dev_placements);
      } else if (auto shape_split =
                     dynamic_cast<ShapeSlicePostOp*>(post_op.get())) {
        TF_ASSERT(result_prop->isShapeTensor(), "Expect a shape tensor");
        TF_ASSERT(shape_split->getSplitDims().size() == 1,
                  "Expect exactly 1 multiple split dim");
        for (size_t i = 0; i < sharded_results[result_idx].size(); i++) {
          auto&& result = sharded_results[result_idx][i];
          auto define_op_name =
              result.getDefiningOp()->getName().getStringRef();
          bool can_replace_const = true;
          auto&& split_spec = shape_split->getSplitSpecs()[0];
          std::vector<int64_t> int_array;
          if (define_op_name == "tfg.Const") {
            int_array = getConstantIntArrayForValue(result);
          } else if (define_op_name == "tfg.Shape") {
            int_array = getShapeIntArrayForValue(result);
          } else {
            can_replace_const = false;
          }
          // TODO(itex): Now only supports slice on the batch size dim.
          // More dimensions need to be supported.
          auto split_dim = shape_split->getSplitDims()[0];

          auto device_name =
              getDeviceNameForValue(sharded_results[result_idx][i]);
          if (can_replace_const) {
            // Build Const OP replace slice shape tensor.
            TF_ASSERT(
                split_dim < int_array.size(),
                "Expect split dim resides in the rank of tensor under split");
            if (split_spec.getType() == SplitSpec::SplitType::SIZE) {
              auto total_size = split_spec.getTotalSize();
              int_array[split_dim] =
                  int_array[split_dim] * split_spec.getSizes()[i] / total_size;
            } else if (split_spec.getType() == SplitSpec::SplitType::RATIO) {
              // TODO(itex): Calculate and check the shape of each shard by
              // ratio Assume the shape can be divided by ratio
              int_array[split_dim] = static_cast<int>(
                  int_array[split_dim] * split_spec.getRatios()[i]);
            } else {
              TF_ASSERT(false, "Only support size split and ratio split");
            }
            sharded_results[result_idx][i] =
                buildConstantTensor(builder, op->getLoc(), device_name,
                                    int_array, {(int64_t)int_array.size()});
          } else {
            // Inserd a Mul OP to calculate the slice shape tensor.
            // Abtain the slice ratio of shape.
            std::vector<float> slice_ratio(split_spec.size(), 1.0);
            TF_ASSERT(
                split_dim < slice_ratio.size(),
                "Expect split dim resides in the rank of tensor under split");
            if (split_spec.getType() == SplitSpec::SplitType::SIZE) {
              auto total_size = split_spec.getTotalSize();
              slice_ratio[split_dim] = split_spec.getSizes()[i] / total_size;
            } else if (split_spec.getType() == SplitSpec::SplitType::RATIO) {
              // TODO(itex): Calculate and check the shape of each shard
              // by ratio.
              // Assume the shape can be divided by ratio
              slice_ratio[split_dim] = split_spec.getRatios()[i];
            } else {
              TF_ASSERT(false, "Only support size split and ratio split");
            }
            // Build Const OP as a multiplier.
            // TIP: float type
            auto slice_ratio_value =
                buildConstantTensor(builder, op->getLoc(), device_name,
                                    slice_ratio, {(int64_t)slice_ratio.size()});

            // Build Mul OP
            Type t = slice_ratio_value.getType();
            auto mul_op =
                buildTfgOp(builder, op->getLoc(), "tfg.Mul", &t, device_name,
                           [&](OperationState* op_state) -> void {
                             op_state->types.push_back(t);
                             op_state->operands.push_back(result);
                             op_state->operands.push_back(slice_ratio_value);
                           });
            // Update sharded_results[result_idx][i].
            sharded_results[result_idx][i] = mul_op->getResult(0);
          }
        }
      } else {
        TF_ASSERT(false, "unsupported post op");
      }
    }
  }
}

std::pair<ShardedValue, ShardedValue>
HStoTFGConversion::buildAllReduceSumPostOp(
    OpBuilder* builder, Location loc, const ShardedValue& sharded_value,
    const ShardedValue& in_control_edge, ShardingPropertyRef result_prop,
    const std::vector<DeviceId>& dev_placements) {
  ShardedValue sharded_result;
  ShardedValue control_edge;
  std::tie(sharded_result, control_edge) = addIntraDeviceReducePostOp(
      "tfg.AddN", builder, loc, sharded_value, in_control_edge, dev_placements);
  std::tie(sharded_result, control_edge) = addInterDeviceReducePostOp(
      "tfg.AddN", builder, loc, sharded_result, control_edge, result_prop);
  return std::make_pair(sharded_result, control_edge);
}

std::pair<ShardedValue, ShardedValue>
HStoTFGConversion::buildAllReduceMaxPostOp(
    OpBuilder* builder, Location loc, const ShardedValue& sharded_value,
    const ShardedValue& in_control_edge, ShardingPropertyRef result_prop,
    const std::vector<DeviceId>& dev_placements) {
  ShardedValue sharded_result;
  ShardedValue control_edge;
  std::tie(sharded_result, control_edge) =
      addIntraDeviceReducePostOp("tfg.Maximum", builder, loc, sharded_value,
                                 in_control_edge, dev_placements);
  std::tie(sharded_result, control_edge) = addInterDeviceReducePostOp(
      "tfg.Maximum", builder, loc, sharded_result, control_edge, result_prop);
  return std::make_pair(sharded_result, control_edge);
}

std::pair<ShardedValue, ShardedValue>
HStoTFGConversion::buildAllReduceMinPostOp(
    OpBuilder* builder, Location loc, const ShardedValue& sharded_value,
    const ShardedValue& in_control_edge, ShardingPropertyRef result_prop,
    const std::vector<DeviceId>& dev_placements) {
  ShardedValue sharded_result;
  ShardedValue control_edge;
  std::tie(sharded_result, control_edge) =
      addIntraDeviceReducePostOp("tfg.Minimum", builder, loc, sharded_value,
                                 in_control_edge, dev_placements);
  std::tie(sharded_result, control_edge) = addInterDeviceReducePostOp(
      "tfg.Minimum", builder, loc, sharded_result, control_edge, result_prop);
  return std::make_pair(sharded_result, control_edge);
}

std::pair<ShardedValue, ShardedValue>
HStoTFGConversion::buildAllReduceProdPostOp(
    OpBuilder* builder, Location loc, const ShardedValue& sharded_value,
    const ShardedValue& in_control_edge, ShardingPropertyRef result_prop,
    const std::vector<DeviceId>& dev_placements) {
  ShardedValue sharded_result;
  ShardedValue control_edge;
  std::tie(sharded_result, control_edge) = addIntraDeviceReducePostOp(
      "tfg.Mul", builder, loc, sharded_value, in_control_edge, dev_placements);
  std::tie(sharded_result, control_edge) = addInterDeviceReducePostOp(
      "tfg.Mul", builder, loc, sharded_result, control_edge, result_prop);
  return std::make_pair(sharded_result, control_edge);
}

std::pair<ShardedValue, ShardedValue> HStoTFGConversion::buildAllReduceL2PostOp(
    OpBuilder* builder, Location loc, const ShardedValue& sharded_value,
    const ShardedValue& in_control_edge, ShardingPropertyRef result_prop,
    const std::vector<DeviceId>& dev_placements) {
  ShardedValue sharded_result;
  ShardedValue control_edge;
  std::tie(sharded_result, control_edge) =
      addIntraDeviceReducePostOp("tfg.AddV2", builder, loc, sharded_value,
                                 in_control_edge, dev_placements);
  std::tie(sharded_result, control_edge) = addInterDeviceReducePostOp(
      "tfg.AddN", builder, loc, sharded_result, control_edge, result_prop);
  return std::make_pair(sharded_result, control_edge);
}

std::pair<ShardedValue, ShardedValue>
HStoTFGConversion::buildWeightedScalePostOp(
    OpBuilder* builder, Location loc, const ShardedValue& sharded_value,
    const ShardedValue& in_control_edge,
    const std::vector<DeviceId>& dev_placements,
    const WeightedScalePostOp& weighted_scale_post_op) {
  ShardedValue sharded_result;
  ShardedValue control_edge;
  std::tie(sharded_result, control_edge) =
      addWeightedScaleToPostOp(builder, loc, sharded_value, in_control_edge,
                               dev_placements, weighted_scale_post_op);
  return std::make_pair(sharded_result, control_edge);
}

std::pair<ShardedValue, ShardedValue> HStoTFGConversion::buildSlicePostOp(
    OpBuilder* builder, Location loc, const ShardedValue& sharded_value,
    const ShardedValue& in_control_edge, as::ShardingPropertyRef prop,
    const std::vector<as::DeviceId>& dev_placements) {
  // TODO(itex): support halo split, currently assume plain split
  // TODO(itex): support multiple split dims, currently assume
  // only single split on batch size dim.
  TF_ASSERT(prop->numMultiSplitDims() == 1,
            "Support maximum 1 split dim for now");
  TF_ASSERT(sharded_value.size() == dev_placements.size(),
            "Expect the number of device placements matching number of tensor "
            "shards");
  ShardedValue sharded_result;
  auto shard_descs = prop->getShardDescriptors();
  ShardedValue control_edge;
  TF_ASSERT(shard_descs.size() == sharded_value.size(),
            "Expect same number of sharded values for sharding descriptors");
  Type t = sharded_value[0].getType();
  int64_t offset = 0;  // offset into the slice
  for (size_t operand_idx = 0; operand_idx < shard_descs.size();
       operand_idx++) {
    auto&& shard_desc = shard_descs[operand_idx];
    auto split_type = shard_desc.getSplitSpec(-1).getType();
    size_t shard_num = 0;
    int64_t this_offset = 0;
    if (split_type == SplitSpec::SplitType::SIZE) {
      shard_num = shard_desc.getSplitSpec(-1).getSizes().size();
      this_offset = shard_desc.getSplitSpec(-1).getSizes()[shard_desc.getNum()];
    } else if (split_type == SplitSpec::SplitType::RATIO) {
      // Split by ratio, compute shape dynamically
      shard_num = shard_desc.getSplitSpec(-1).getRatios().size();
      auto&& split_shape = shard_desc.getShape();
      auto split_dim_size = split_shape[shard_desc.getSplitDim(-1)];
      // TODO(itex): Calculate and check the shape of each shard by ratio
      // Assume the shape can be divided by ratio
      this_offset += split_dim_size;
    } else {
      TF_ASSERT(false, "Only spliting by concrete sizes is supported");
    }
    TF_ASSERT(shard_desc.getNum() < shard_num, "Shard num out of range");
    auto&& device_name = getDeviceName(dev_placements[operand_idx]);
    // Build the slice op for the shard
    auto slice_op = buildTfgOp(
        builder, loc, "tfg.StridedSlice", &t, device_name,
        [&](OperationState* slice) -> void {
          // add result type: the sliced tensor
          auto shaped_type = t.dyn_cast<ShapedType>();
          slice->types.push_back(getRankedTensorType(
              shard_desc.getRank(), shaped_type.getElementType(),
              shard_desc.getShape()));
          // add operand: the tensor to slice, the begin, end and strides const
          // tensors according to the tensor split
          slice->operands.push_back(sharded_value[operand_idx]);
          int64_t split_dim = shard_desc.getSplitDim(-1);
          TF_ASSERT(split_dim < shard_desc.getRank(), "Invalid split dim");
          std::vector<int64_t> begin(shard_desc.getRank(), 0);
          std::vector<int64_t> end(shard_desc.getRank(), 0);
          begin[split_dim] = offset;
          offset += this_offset;
          end[split_dim] = offset;
          slice->operands.push_back(buildConstantTensor(
              builder, loc, device_name, begin,
              std::vector<int64_t>({shard_desc.getRank()})));
          slice->operands.push_back(buildConstantTensor(
              builder, loc, device_name, end,
              std::vector<int64_t>({shard_desc.getRank()})));
          slice->operands.push_back(buildConstantTensor(
              builder, loc, device_name,
              std::vector<int64_t>(shard_desc.getRank(), 1),
              std::vector<int64_t>({shard_desc.getRank()})));
          // add attributes: begin_mask and end_mask to ignore non-split dims
          // (i.e. mask non-split dim bits)
          unsigned begin_mask = 0, end_mask = 0;
          for (int64_t dim = 0; dim < shard_desc.getRank(); dim++) {
            if (dim != split_dim) {
              begin_mask |= 1 << dim;
              end_mask |= 1 << dim;
            }
          }
          slice->addAttribute("begin_mask",
                              builder->getI32IntegerAttr(begin_mask));
          slice->addAttribute("end_mask", builder->getI32IntegerAttr(end_mask));
          // add attributes: index type
          slice->addAttribute("Index", TypeAttr::get(builder->getI32Type()));
          // add attributes: optional attributes
          slice->addAttribute("ellipsis_mask", builder->getI32IntegerAttr(0));
          slice->addAttribute("new_axis_mask", builder->getI32IntegerAttr(0));
          slice->addAttribute("shrink_axis_mask",
                              builder->getI32IntegerAttr(0));
        });
    sharded_result.push_back(slice_op->getResult(0));
    control_edge.push_back(slice_op->getResult(1));
  }
  return std::make_pair(sharded_result, control_edge);
}

std::pair<ShardedValue, ShardedValue>
HStoTFGConversion::addInterDeviceReducePostOp(
    const std::string& inter_op_name, OpBuilder* builder, Location loc,
    const ShardedValue& sharded_value, const ShardedValue& in_control_edge,
    ShardingPropertyRef prop) {
  TF_ASSERT(sharded_value.size() == prop->getDeviceIds().size(),
            "Expect number of shards matches the number of devices");
  auto num_logical_shards_per_device = prop->getNumLogicalShardsPerdevice();
  ShardedValue reduced_value;
  ShardedValue control_edge;
  if (sharded_value.size() == 1) {
    for (size_t i = 0; i < num_logical_shards_per_device[0]; i++) {
      reduced_value.push_back(sharded_value[0]);
      if (!in_control_edge.empty()) {
        control_edge.push_back(in_control_edge[0]);
      }
    }
  } else {
    Operation* inter_op = nullptr;
    // By default, the reduce is performed on the first device.
    std::string device_name = getDeviceName(prop->getDeviceIds()[0]);
    Type t = sharded_value[0].getType();
    if (inter_op_name == "tfg.AddN") {
      // The operand num of `inter_op_name` is N(>2)
      // (Temporarily only contains AddN OP)
      inter_op = buildTfgOp(
          builder, loc, "tfg.AddN", &t, device_name,
          [&](OperationState* op_state) -> void {
            op_state->types.push_back(t);
            for (auto operand : llvm::enumerate(sharded_value)) {
              op_state->operands.push_back(operand.value());
            }
            op_state->addAttribute(
                "N", builder->getI64IntegerAttr(sharded_value.size()));
          });
    } else {
      // The operand num of inter_op_name is 2.
      // Add `inter_op_name` OP two by two.
      auto x_operand = *(sharded_value.begin());
      for (auto i = 1; i < sharded_value.size(); i++) {
        auto y_operand = sharded_value[i];
        Type t = y_operand.getType();
        inter_op = buildTfgOp(builder, loc, inter_op_name, &t, device_name,
                              [&](OperationState* op_state) -> void {
                                op_state->types.push_back(t);
                                op_state->operands.push_back(x_operand);
                                op_state->operands.push_back(y_operand);
                              });
        x_operand = inter_op->getResult(0);
      }
    }
    // The result of reduce is passed to each stage on each device.
    for (auto indexed_value : llvm::enumerate(sharded_value)) {
      for (size_t j = 0;
           j < num_logical_shards_per_device[indexed_value.index()]; j++) {
        reduced_value.push_back(inter_op->getResult(0));
        control_edge.push_back(inter_op->getResult(1));
      }
    }
  }
  return {reduced_value, control_edge};
}

std::pair<ShardedValue, ShardedValue>
HStoTFGConversion::addWeightedScaleToPostOp(
    OpBuilder* builder, Location loc, const ShardedValue& sharded_value,
    const ShardedValue& in_control_edge,
    const std::vector<DeviceId>& dev_placements,
    const WeightedScalePostOp& weighted_scale_post_op) {
  TF_ASSERT(
      sharded_value.size() == dev_placements.size() &&
          sharded_value.size() == weighted_scale_post_op.getNumShards(),
      "Expect number of shards match for Value, weights and dev_placements");
  ShardedValue averaged_value;
  ShardedValue control_edge;
  size_t tensor_size = weighted_scale_post_op.getTensorSize();
  for (auto indexed_value : llvm::enumerate(sharded_value)) {
    std::string device_name =
        getDeviceName(dev_placements[indexed_value.index()]);
    Type t = indexed_value.value().getType();
    // get weights for this shard
    std::vector<float> per_shard_weights;
    for (size_t i = 0; i < tensor_size; i++) {
      per_shard_weights.push_back(
          weighted_scale_post_op
              .getWeights()[i * sharded_value.size() + indexed_value.index()]);
    }
    // build tfg.Mul with the per-shard weights
    Operation* mul_op =
        buildTfgOp(builder, loc, "tfg.Mul", &t, device_name,
                   [&](OperationState* mul_op_state) {
                     mul_op_state->types.push_back(t);
                     mul_op_state->operands.push_back(indexed_value.value());
                     mul_op_state->operands.push_back(buildConstantTensor(
                         builder, loc, device_name, per_shard_weights,
                         weighted_scale_post_op.getTensorShape()));
                   });
    averaged_value.push_back(mul_op->getResult(0));
    control_edge.push_back(mul_op->getResult(1));
  }
  return {averaged_value, control_edge};
}

std::pair<ShardedValue, ShardedValue>
HStoTFGConversion::addIntraDeviceReducePostOp(
    const std::string& intra_op_name, OpBuilder* builder, Location loc,
    const ShardedValue& sharded_value, const ShardedValue& in_control_edge,
    const std::vector<DeviceId>& dev_placements) {
  TF_ASSERT(sharded_value.size() == dev_placements.size(),
            "Expect number of shards match for Value and Input Desc");
  ShardedValue result_value;
  ShardedValue control_edge;
  auto handle_shards_with_same_device = [&](int64_t offset_begin,
                                            int64_t offset_end) {
    if (offset_end - offset_begin == 1) {
      result_value.push_back(sharded_value[offset_begin]);
      if (!in_control_edge.empty()) {
        control_edge.push_back(in_control_edge[offset_begin]);
      }
      return;
    }
    std::string device_name = getDeviceName(dev_placements[offset_begin]);
    Type t = sharded_value[offset_begin].getType();
    Operation* intra_op = nullptr;
    if (intra_op_name == "tfg.AddN") {
      // The operand num of `intra_op_name` is N(>2)
      // (Temporarily only contains AddN OP)
      intra_op = buildTfgOp(
          builder, loc, "tfg.AddN", &t, device_name,
          [&](OperationState* op_state) -> void {
            op_state->types.push_back(t);
            for (auto offset = offset_begin; offset < offset_end; offset++) {
              op_state->operands.push_back(sharded_value[offset]);
            }
            op_state->addAttribute(
                "N", builder->getI64IntegerAttr(offset_end - offset_begin));
          });
    } else {
      // The operand num of intra_op_name is 2.
      // Add `intra_op_name` OP two by two.
      auto x_operand = sharded_value[offset_begin];
      for (auto offset = offset_begin + 1; offset < offset_end; offset++) {
        auto y_operand = sharded_value[offset];
        intra_op = buildTfgOp(builder, loc, intra_op_name, &t, device_name,
                              [&](OperationState* op_state) -> void {
                                op_state->types.push_back(t);
                                op_state->operands.push_back(x_operand);
                                op_state->operands.push_back(y_operand);
                              });
        x_operand = intra_op->getResult(0);
      }
    }
    result_value.push_back(intra_op->getResult(0));
    control_edge.push_back(intra_op->getResult(1));
    if (sharding_config_.isUseMultiStageJoin()) {
      multiStageJoin(sharded_value.begin() + offset_begin,
                     sharded_value.begin() + offset_end);
    }
  };
  auto iter_begin = dev_placements.begin();
  for (auto iter = iter_begin; iter != dev_placements.end(); iter++) {
    if (*iter != *iter_begin) {
      handle_shards_with_same_device(iter_begin - dev_placements.begin(),
                                     iter - dev_placements.begin());
      iter_begin = iter;
    }
  }
  handle_shards_with_same_device(iter_begin - dev_placements.begin(),
                                 dev_placements.end() - dev_placements.begin());
  return {result_value, control_edge};
}

void HStoTFGConversion::handleGraphOp(mlir::tfg::GraphOp* graph_op) {
  if (!isRootOp(graph_op->getOperation())) return;
  // Save DeviceInfo in the current context and cleanup DeviceInfoAttr from
  // graph_op
  setCurrentDeviceInfo(graph_op);
  graph_op->getOperation()->removeAttr(HSDialect::getDeviceInfoAttrKey());
  // forward lowering
  std::queue<Operation*> ready_queue;
  for (Block& block : graph_op->getRegion().getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (llvm::dyn_cast<ShardOp>(op)) {
        ready_queue.push(&op);
      }
    }
  }
  using ::llvm::hash_value;
  std::unordered_map<Operation*, size_t> num_operands_visited;
  llvm::MapVector<Value, ShardedValue> value_to_sharded_map;
  while (!ready_queue.empty()) {
    Operation* op = ready_queue.front();
    ready_queue.pop();
    if (auto shard_op = llvm::dyn_cast<ShardOp>(op)) {
      auto sharded_result = lowerShardOp(&shard_op);
      value_to_sharded_map[shard_op.getResult()] = sharded_result;
    } else if (auto unshard_op = llvm::dyn_cast<UnshardOp>(op)) {
      auto operand = unshard_op.getOperand();
      lowerUnshardOp(&unshard_op, value_to_sharded_map[operand]);
      continue;  // end of the sharding region, no need to add it users
    } else if (auto reshard_op = llvm::dyn_cast<ReshardOp>(op)) {
      auto operand = reshard_op.getOperand();
      auto sharded_result =
          lowerReshardOp(&reshard_op, value_to_sharded_map[operand]);
      value_to_sharded_map[reshard_op.getResult()] = sharded_result;
    } else {
      TF_ASSERT(
          isFrameworkOp(op) && op->getDialect()->getNamespace() ==
                                   tfg::TFGraphDialect::getDialectNamespace(),
          "Unexpected op type in the sharding sub-graph");
      std::vector<ShardedValue> sharded_operands;
      for (auto&& operand : op->getOperands()) {
        auto&& sharded_operand = value_to_sharded_map[operand];
        sharded_operands.push_back(sharded_operand);
      }
      auto sharded_results = lowerFrameworkOp(op, sharded_operands);
      TF_ASSERT(sharded_results.size() == op->getNumResults(),
                "Expect same number of results from original and sharded op");
      for (size_t i = 0; i < op->getNumResults(); i++) {
        value_to_sharded_map[op->getResult(i)] = sharded_results[i];
      }
    }
    // add users to the ready queue
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
  // backward erasing lowered ops
  for (Block& block : graph_op->getRegion().getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (llvm::dyn_cast<UnshardOp>(op)) {
        ready_queue.push(&op);
      }
    }
  }
  std::unordered_set<Operation*> visited_ops;
  while (!ready_queue.empty()) {
    Operation* op = ready_queue.front();
    ready_queue.pop();
    if (visited_ops.find(op) != visited_ops.end()) {
      continue;
    }
    visited_ops.insert(op);
    std::vector<Operation*> defining_ops;
    for (auto operand : op->getOperands()) {
      defining_ops.push_back(operand.getDefiningOp());
    }
    TF_ASSERT(op->use_empty(), "Expect no uses before erasing lowered ops");
    op->erase();
    for (auto defining_op : defining_ops) {
      if (defining_op->use_empty()) {
        ready_queue.push(defining_op);
      }
    }
  }
  avoidPruningOfCommOps(graph_op);

  // TODO(itex): add control edge for multi-iteration runs

  // clean up noops
}

void HStoTFGConversion::avoidPruningOfCommOps(mlir::tfg::GraphOp* graph_op) {
  // Collect all the communication ops from the devices other than the main
  // device. Add NoOp on each device and link the comm ops on the same device to
  // the corresponding NoOp with control edges. Add control edge from the NoOps
  // to the first user of the last comm op on the main device.
  //
  // Assumptions of the algorithm:
  // 1. All comm ops work on the same set of devices tracked by
  // current_device_info_.
  // 2. The first device in current_device_info_ is the main device.
  // 3. On the main device, the comm ops with the largest name number is
  // `last_communication_op`.
  //    Because the names are in topological order, the last_op will be the last
  //    comm ops executed on that branch on the computation graph. This
  //    assumption can avoid deadlock between comm ops.
  // 4. The comm ops on the main device will all have a data-dependent user
  // named `next_op`,
  //    and will not be pruned.

  std::unordered_map<std::string, std::vector<Value>> device_noop_control_edges;
  auto devices = *(current_device_info_.getDevices().begin());
  std::string main_device = getMlirDeviceName(devices.getName());
  Operation* last_communication_op = nullptr;
  for (Block& block : graph_op->getRegion().getBlocks()) {
    for (Operation& op : block.getOperations()) {
      // 1. get all comm ops in other device
      if (kCommunicationOP.find(op.getName().getStringRef().str()) !=
          kCommunicationOP.end()) {
        TF_ASSERT(op.getResults().size() == 2, "Expect have two result");
        auto result = op.getResult(1);
        auto device_name = getDeviceNameForValue(result);
        // 2. update `last_communication_op ` with the largest name number.
        if (device_name == main_device) {
          if (last_communication_op == nullptr) {
            last_communication_op = &op;
          } else {
            auto now_value_name =
                op.getAttrOfType<StringAttr>(kMlirNameAttr).getValue().str();
            auto max_value_name =
                last_communication_op->getAttrOfType<StringAttr>(kMlirNameAttr)
                    .getValue()
                    .str();
            last_communication_op =
                now_value_name >= max_value_name ? &op : last_communication_op;
          }
          continue;
        }
        // 3. get control edge set from comm ops on each other devices.
        auto control_edge = getControlEdgeForValue(result);
        TF_ASSERT(control_edge, "Could not find control edge");
        device_noop_control_edges[device_name].push_back(control_edge);
      }
    }
  }
  if (device_noop_control_edges.size() == 0) {
    return;
  }
  // 4. build `NoOp` for each other device connect all comm ops.
  std::vector<Value> noop_values;
  OpBuilder builder(last_communication_op);
  for (auto& vec : device_noop_control_edges) {
    auto device_name = vec.first;
    Type t = (vec.second)[0].getType();
    Operation* noop_op = buildTfgOp(
        &builder, last_communication_op->getLoc(), "tfg.NoOp", &t, device_name,
        [&](OperationState* noop) -> void {
          noop->types.push_back(t);
          auto&& operator_vec = device_noop_control_edges[device_name];
          noop->operands.insert(noop->operands.end(), operator_vec.begin(),
                                operator_vec.end());
        });
    noop_values.push_back(noop_op->getResult(0));
  }
  // 5. Make `NoOp` ops in other devices connect to the `next_op` in main_device
  auto&& users = last_communication_op->getUsers();
  TF_ASSERT(users.empty() == false,
            "Expect `last_communication_op` has at least one user in the main "
            "device");
  bool connect_flag = false;
  for (auto next_op : users) {
    auto operands_num = next_op->getNumOperands();
    next_op->insertOperands(operands_num, noop_values);
    if (operands_num + noop_values.size() == next_op->getNumOperands()) {
      connect_flag = true;
    }
  }
  TF_ASSERT(connect_flag,
            "Expect `noop_values` insert into the operands of the one user of "
            "`last_communication_op`");
}

bool checkCycleDFS(
    Operation* op,
    std::unordered_map<Operation*, size_t>& op_state) {  // NOLINT
  const auto IN_QUEUE = 1;   // It means the OP is in the DFS processing queue.
  const auto OUT_CYCLE = 2;  // It means the OP is not on the cycle.
  op_state[op] = IN_QUEUE;
  for (auto result : op->getResults()) {
    for (auto user : result.getUsers()) {
      if (op_state[user] == IN_QUEUE) {
        llvm::outs()
            << "There is a cycle after auto shard. Print cycle path : \n";
        llvm::outs() << "** " << *user << "\n";
        return true;
      }
      if (op_state[user] == OUT_CYCLE) {
        continue;
      }
      if (checkCycleDFS(user, op_state)) {
        llvm::outs() << "** " << *op << "\n";
        return true;
      }
    }
  }
  op_state[op] = OUT_CYCLE;
  return false;
}

void checkGraphCycle(mlir::tfg::GraphOp* graph_op) {
  std::unordered_map<Operation*, size_t> op_state;
  for (Block& block : graph_op->getRegion().getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (op.getNumOperands() == 0) {
        if (checkCycleDFS(&op, op_state)) {
          return;
        }
      }
    }
  }
  llvm::outs() << "auto shard cycle check over\n";
}

void HStoTFGConversion::runOnOperation() {
  auto this_op = getOperation();
  if (auto graph_op = llvm::dyn_cast<mlir::tfg::GraphOp>(*this_op)) {
    if (checkGraphShapeInferenceException(&graph_op) == true) {
      ITEX_LOG(WARNING)
          << "Graph has shape inference exception, skip hs_to_tfg pass.";
      return;
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
                << "Graph has shape inference exception, skip hs_to_tfg pass.";
            return;
          }
          handleGraphOp(&graph_op);
          checkGraphCycle(&graph_op);
        }
      }
    }
  }
}
}  // namespace hs
}  // namespace mlir
