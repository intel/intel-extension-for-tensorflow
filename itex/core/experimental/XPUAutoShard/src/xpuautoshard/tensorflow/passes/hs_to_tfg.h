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

#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "itex/core/ir/ops.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "xpuautoshard/common/config.h"
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/sharding_property_grouper.h"

namespace mlir {
namespace hs {
using ShardedValue = std::vector<Value>;

struct HStoTFGConversion
    : public PassWrapper<HStoTFGConversion, OperationPass<> > {
  explicit HStoTFGConversion(const as::ShardingConfig& sharding_config)
      : sharding_config_(sharding_config), collective_id_(0), op_name_id_(0) {}
  HStoTFGConversion(const HStoTFGConversion& pass) = default;
  ~HStoTFGConversion() override;
  void runOnOperation() override;

 private:
  as::ShardingConfig sharding_config_;
  as::DeviceInfo current_device_info_;
  int64_t collective_id_;
  int64_t op_name_id_;
  std::unordered_map<std::string, int64_t> device_op_to_id_map_;
  // NOTE: TF expects the names of some op types unchanged. We don't want to
  // hard-code those rules due to the complexity. Therefore, we track the first
  // time the corresponding op_name is created and use the original op_name
  // on the first creation.
  std::unordered_set<std::string> created_op_names_;

  void handleGraphOp(mlir::tfg::GraphOp* graph_op);

  ShardedValue lowerShardOp(ShardOp* shard_op);

  void lowerUnshardOp(UnshardOp* unshard_op,
                      const ShardedValue& sharded_operand);

  ShardedValue lowerReshardOp(ReshardOp* reshard_op,
                              const ShardedValue& sharded_operand);

  std::vector<ShardedValue> lowerFrameworkOp(
      Operation* tfg_op, const std::vector<ShardedValue>& sharded_operands);

  void setCurrentDeviceInfo(mlir::tfg::GraphOp* graph_op);

  std::string getDeviceName(as::DeviceId id);

  std::string createShardedTfgOpNameFor(const std::string& device_name,
                                        const std::string& op_name);

  std::string getNextOpNameIdStr();

  std::string getNextCollectiveKey();

  /**
   * The function avoids the pruning of cross-device communication ops (e.g.,
   * all-reduce) by inserting control dependencies to the "primary" device from
   * other devices in an efficient manner (with as less cross-device control
   * edges as possible). The related communication ops are those yielding the
   * same result on all ranks, e.g., all-reduce, all-gather, broadcast etc. but
   * not all-to-all, scatter, gather etc. The reason why we have to add
   * additional control dependency is that there are inherent dependencies among
   * these ops and because they yield the same result, we originally only
   * connect the result from the primary device to the main trunk of the graph
   * while not the remaining ones which will be pruned by the
   * dead-code-elimination if control dependencies are not added.
   */
  void avoidPruningOfCommOps(mlir::tfg::GraphOp* graph_op);

  /**
   * @brief Handle the post ops of an op given the sharded results
   * `sharded_results` on corresponding devices `dev_palcements`, control edges,
   * `sharded_control_edges` and sharding properties in `result_props`. The new
   * results and control edges are updated to `sharded_results` and
   * `sharded_control_edges` after the call.
   *
   * @param builder
   * @param op
   * @param sharded_results
   * @param sharded_control_edges
   * @param result_props
   * @param dev_placements
   */
  void handlePostOps(
      OpBuilder* builder, Operation* op,
      std::vector<ShardedValue>& sharded_results,        // NOLINT
      std::vector<ShardedValue>& sharded_control_edges,  // NOLINT
      const std::vector<as::ShardingPropertyRef>& result_props,
      const std::vector<as::DeviceId>& dev_placements);

  std::pair<ShardedValue, ShardedValue> addInterDeviceReducePostOp(
      const std::string& inter_op_name, OpBuilder* builder, const Location loc,
      const ShardedValue& sharded_value, const ShardedValue& control_edge,
      as::ShardingPropertyRef prop);

  std::pair<ShardedValue, ShardedValue> addWeightedScaleToPostOp(
      OpBuilder* builder, const Location loc, const ShardedValue& sharded_value,
      const ShardedValue& control_edge,
      const std::vector<as::DeviceId>& dev_placements,
      const as::WeightedScalePostOp& weighted_scale_post_op);

  std::pair<ShardedValue, ShardedValue> addIntraDeviceReducePostOp(
      const std::string& intra_op_name, OpBuilder* builder, const Location loc,
      const ShardedValue& sharded_value, const ShardedValue& control_edge,
      const std::vector<as::DeviceId>& dev_placements);

  std::pair<ShardedValue, ShardedValue> buildAllReduceSumPostOp(
      OpBuilder* builder, const Location loc, const ShardedValue& sharded_value,
      const ShardedValue& control_edge, as::ShardingPropertyRef result_prop,
      const std::vector<as::DeviceId>& dev_placements);

  std::pair<ShardedValue, ShardedValue> buildAllReduceMaxPostOp(
      OpBuilder* builder, const Location loc, const ShardedValue& sharded_value,
      const ShardedValue& control_edge, as::ShardingPropertyRef result_prop,
      const std::vector<as::DeviceId>& dev_placements);

  std::pair<ShardedValue, ShardedValue> buildAllReduceMinPostOp(
      OpBuilder* builder, const Location loc, const ShardedValue& sharded_value,
      const ShardedValue& control_edge, as::ShardingPropertyRef result_prop,
      const std::vector<as::DeviceId>& dev_placements);

  std::pair<ShardedValue, ShardedValue> buildAllReduceProdPostOp(
      OpBuilder* builder, const Location loc, const ShardedValue& sharded_value,
      const ShardedValue& control_edge, as::ShardingPropertyRef result_prop,
      const std::vector<as::DeviceId>& dev_placements);

  std::pair<ShardedValue, ShardedValue> buildAllReduceL2PostOp(
      OpBuilder* builder, const Location loc, const ShardedValue& sharded_value,
      const ShardedValue& control_edge, as::ShardingPropertyRef result_prop,
      const std::vector<as::DeviceId>& dev_placements);

  std::pair<ShardedValue, ShardedValue> buildWeightedScalePostOp(
      OpBuilder* builder, const Location loc, const ShardedValue& sharded_value,
      const ShardedValue& control_edge,
      const std::vector<as::DeviceId>& dev_placements,
      const as::WeightedScalePostOp& weighted_scale_post_op);

  std::pair<ShardedValue, ShardedValue> buildSlicePostOp(
      OpBuilder* builder, const Location loc, const ShardedValue& sharded_value,
      const ShardedValue& control_edge, as::ShardingPropertyRef result_prop,
      const std::vector<as::DeviceId>& dev_placements);

  Operation* buildTfgOp(
      OpBuilder* builder, const Location loc, const std::string& op_name,
      Type* tensor_type, const std::string& device_name,
      const std::function<void(OperationState* op_state)> op_state_init,
      bool need_output_control_edge = true);

  Value buildConstant(OpBuilder* builder, const Location loc,
                      const std::string& device_name, int64_t v);
  Value buildConstant(OpBuilder* builder, const Location loc,
                      const std::string& device_name, size_t v);
  Value buildConstant(OpBuilder* builder, const Location loc,
                      const std::string& device_name, float v);

  template <typename T>
  Value buildConstantTensor(
      OpBuilder* builder, const Location loc, const std::string& device_name,
      std::vector<T> tensor,
      std::vector<int64_t> shape = std::vector<int64_t>());

  /**
   * @brief This function is used to build lower framework OP operands on the
   * curent device and optimize the number of across-device control edges by
   * inserting NoOp in the operands.The optimization of the current framework OP
   * needs to refer to the optimization result of the previous framework OP.
   *
   * @param builder
   * @param loc
   * @param sharded_operands
   * @param shard_desc_group  A group of ShardDescs corresponding to a group of
   * sharded tensors.
   * @param use_noop_mp_flag  Indicates whether the result recorded can be used
   * with `device_noop_mp`. Optimization results in `device_noop_mp` can be
   * shared only if all control edges of the current lower framework OP and the
   * last lower framework OP in the neighborhood are the same.
   * @param device_noop_mp  It is a hash table that records the NoOp results of
   * building and optimizing the operands of the last lower framework OP in the
   * neighborhood.This hash table is referenced and updated when building the
   * operands of the current lower framework OP. Key: device mlir name Value:
   * indicates the control edge of a NoOp. This NoOp is connected to all control
   *                               edges on a device related to a Lower
   * Framework OP.
   */
  std::vector<Value> buildFrameworkOpOperands(
      OpBuilder* builder, const Location loc,
      const std::vector<ShardedValue>& sharded_operands,
      const as::ShardDescGroup& shard_desc_group,
      std::unordered_map<std::string, Value>& device_noop_mp);  // NOLINT

  Operation* buildFrameworkOp(OpBuilder* builder, Operation* template_op,
                              const std::vector<Value>& operands,
                              const std::vector<Type>& result_types,
                              const std::string& device_name);
};
}  // namespace hs
}  // namespace mlir
