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

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "itex/core/ir/ops.h"
#include "itex/core/utils/logging.h"
#include "xpuautoshard/common/config.h"
#include "xpuautoshard/common/graph.h"
#include "xpuautoshard/common/hsp_tuner.h"
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/op_desc.h"
#include "xpuautoshard/common/sharding_property.h"

namespace mlir {
namespace hs {

namespace {  // NOLINT

using as::DeviceInfo;
using as::ShardingProperty;
using as::ShardingPropertyRefVec;
using OPSet = std::set<std::string>;

static const OPSet kUpdateOP = {"tfg.ApplyGradientDescent",
                                "tfg.ApplyProximalGradientDescent",
                                "tfg.ApplyAdadelta",
                                "tfg.ApplyAdagrad",
                                "tfg.ApplyProximalAdagrad",
                                "tfg.ApplyAdagradDA",
                                "tfg.ApplyFtrl",
                                "tfg.ApplyMomentum",
                                "tfg.ApplyAdam",
                                "tfg.ApplyRMSProp",
                                "tfg.ApplyCenteredRMSProp",
                                "tfg.ResourceApplyGradientDescent",
                                "tfg.ResourceApplyAdam",
                                "tfg.ResourceApplyKerasMomentum"};

static const OPSet kAssignOP = {"tfg.Assign",
                                "tfg.AssignVariableOp",
                                "tfg.AssignAddVariableOp",
                                "tfg.AssignSubVariableOp",
                                "tfg.AssignAdd",
                                "tfg.AssignSub"};

static const OPSet kCommunicationOP = {"tfg.CCLAllReduce", "tfg.NcclAllReduce"};

static const OPSet kRetvalOP = {"tfg._Retval"};

static const OPSet kResourceOP = {"tfg.ResourceGather"};

// // Generally, the sharding property of this type of OP is usually single
// split only.
static const OPSet kConstSingleSplitOP = {"tfg.Const", "tfg.VarHandleOp",
                                          "tfg.ReadVariableOp"};

static constexpr char kDeviceAttr[] = "_mlir_device";
static constexpr char kMlirNameAttr[] = "_mlir_name";
static constexpr char DEVICE_PREFIX[] =
    "/job:localhost/replica:0/task:0/device:";

/**
 * @brief Check if a given `op` is a root for auto sharding pass to process. A
 * root op is a top-level DAG that auto sharding works on. It could contain
 * multiple sub-graphs that start with ShardOp and end with Unshardop-> The root
 * op has the DeviceInfo attribute that describes the device info needed for
 * auto sharding. Different root op can have different device info to work with.
 *
 * @param op
 * @return true
 * @return false
 */
bool isRootOp(Operation* op) {
  bool has_device_info =
      op->hasAttrOfType<DeviceInfoAttr>(HSDialect::getDeviceInfoAttrKey());
  return has_device_info && op->hasTrait<OpTrait::OneRegion>() &&
         op->hasTrait<OpTrait::HasOnlyGraphRegion>();
}

/**
 * @brief Check if the given `op` is a framework op that autosharding pass
 * recognizes.
 *
 * @param op
 * @return true
 * @return false
 */
bool isFrameworkOp(Operation* op) {
  // exclude HS ops first
  if (op->getDialect()->getNamespace() == "hs") {
    return false;
  }
  for (auto indexed_result : ::llvm::enumerate(op->getResults())) {
    if (op->hasAttrOfType<ShardingPropertyAttr>(
            HSDialect::getHspAttrKeyFor(indexed_result.index()))) {
      return true;
    }
  }
  return false;
}

std::pair<int64_t, std::vector<int64_t>> rankAndShapes(
    mlir::RankedTensorType ranked_tensor_type) {
  int64_t rank = ranked_tensor_type.getRank();
  std::vector<int64_t> shapes(rank);
  for (unsigned int dim = 0; dim < rank; dim++) {
    if (!ranked_tensor_type.isDynamicDim(dim)) {
      shapes[dim] = ranked_tensor_type.getDimSize(dim);
    } else {
      shapes[dim] = as::DYNAMIC_DIM_SIZE;
    }
  }
  return std::make_pair(rank, shapes);
}

std::pair<int64_t, std::vector<int64_t>> rankAndShapes(mlir::Type type) {
  if (auto ranked_tensor_type = type.dyn_cast<mlir::RankedTensorType>()) {
    return rankAndShapes(ranked_tensor_type);
  }
  return std::make_pair(as::UNRANKED, std::vector<int64_t>());
}

int64_t rank(mlir::Type type) {
  auto rank_and_shapes = rankAndShapes(type);
  return rank_and_shapes.first;
}

size_t numShardedResults(Operation* op) {
  size_t num_sharded_results = 0;
  for (auto attr : op->getAttrs()) {
    if (attr.getValue().isa<ShardingPropertyAttr>()) {
      num_sharded_results++;
    }
  }
  return num_sharded_results;
}

as::DataType getElementTypeFromMlirType(Type type) {
  using as::DataType;
  auto data_type_from_mlir_type = [](Type t) -> DataType {
    if (t.isa<Float64Type>()) {
      return DataType::FLOAT64;
    } else if (t.isa<Float32Type>()) {
      return DataType::FLOAT32;
    } else if (t.isa<Float16Type>()) {
      return DataType::FLOAT16;
    } else if (t.isa<BFloat16Type>()) {
      return DataType::BFLOAT16;
    } else if (t.isa<IntegerType>()) {
      return DataType::INTEGER;
    }
    return DataType::UNKNOWN;
  };
  DataType element_type = DataType::UNKNOWN;
  if (auto shaped_type = type.dyn_cast<ShapedType>()) {
    element_type = data_type_from_mlir_type(shaped_type.getElementType());
  }
  return element_type;
}

as::ShardingPropertyRef createShardingPropertyForType(
    Type type, const DeviceInfo& device_info) {
  auto rank_and_shapes = rankAndShapes(type);
  return as::makeRef<ShardingProperty>(
      device_info, getElementTypeFromMlirType(type), rank_and_shapes.first,
      rank_and_shapes.second);
}

ShardingPropertyAttr getShardingPropertyAttr(Operation* op,
                                             unsigned int result_id = 0) {
  if (auto shard_op = llvm::dyn_cast<ShardOp>(op)) {
    return shard_op.getHsp();
  } else if (auto reshard_op = llvm::dyn_cast<ReshardOp>(op)) {
    return reshard_op.getHsp();
  } else {
    return op->getAttrOfType<ShardingPropertyAttr>(
        HSDialect::getHspAttrKeyFor(result_id));
  }
}

void setShardingPropertyAttr(Operation* op,
                             const ShardingPropertyAttr& prop_attr,
                             unsigned int result_id = 0) {
  if (auto shard_op = llvm::dyn_cast<ShardOp>(op)) {
    shard_op.setHspAttr(prop_attr);
  } else if (auto reshard_op = llvm::dyn_cast<ReshardOp>(op)) {
    reshard_op.setHspAttr(prop_attr);
  } else {
    op->setAttr(HSDialect::getHspAttrKeyFor(result_id), prop_attr);
  }
}

as::ShardingPropertyRef getShardingPropertyForValue(Value value) {
  auto defining_op = value.getDefiningOp();
  for (unsigned int i = 0; i < defining_op->getNumResults(); i++) {
    auto op_result = defining_op->getResult(i);
    if (op_result == value) {
      auto hsp =
          getShardingPropertyAttr(defining_op, op_result.getResultNumber());
      assert(hsp && "Value should have sharding property attribute");
      return hsp.getShardingProperty();
    }
  }
  assert(false &&
         "Should not reach here, value should be in the results of the "
         "defining op");
  return as::makeRef<as::ShardingProperty>(
      as::DeviceInfo(/*add_cpu_host=*/false));
}

as::ShardingPropertyRef getShardingPropertyForValue(OpResult result) {
  auto defining_op = result.getDefiningOp();
  auto hsp = getShardingPropertyAttr(defining_op, result.getResultNumber());
  assert(hsp && "Value should have sharding property attribute");
  return hsp.getShardingProperty();
}

void setShardingPropertyForValue(Value value, as::ShardingPropertyRef prop) {
  auto defining_op = value.getDefiningOp();
  for (unsigned int i = 0; i < defining_op->getNumResults(); i++) {
    auto op_result = defining_op->getResult(i);
    if (op_result == value) {
      setShardingPropertyAttr(
          defining_op,
          ShardingPropertyAttr::get(defining_op->getContext(), prop),
          op_result.getResultNumber());
      return;
    }
  }
  assert(false &&
         "Should not reach here, value should be in the results of the "
         "defining op");
}

void setShardingPropertyForValue(OpResult result,
                                 as::ShardingPropertyRef prop) {
  auto defining_op = result.getDefiningOp();
  auto&& prop_attr = ShardingPropertyAttr::get(defining_op->getContext(), prop);
  if (auto shard_op = llvm::dyn_cast<ShardOp>(defining_op)) {
    shard_op.setHspAttr(prop_attr);
  } else if (auto reshard_op = llvm::dyn_cast<ReshardOp>(defining_op)) {
    reshard_op.setHspAttr(prop_attr);
  } else {
    defining_op->setAttr(HSDialect::getHspAttrKeyFor(result.getResultNumber()),
                         prop_attr);
  }
}

RankedTensorType getRankedTensorType(
    int64_t rank, Type element_type,
    const std::vector<int64_t> sizes = std::vector<int64_t>()) {
  llvm::SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
  for (size_t i = 0; i < sizes.size(); i++) {
    shape[i] = sizes[i];
  }
  return mlir::RankedTensorType::get(shape, element_type);
}

bool isHsAttr(const std::string& attr_name) {
  return attr_name.find("hs.") == 0;
}

std::vector<int64_t> getIntArrayAttr(Operation* op,
                                     const std::string attr_name) {
  std::vector<int64_t> rst;
  if (op->hasAttrOfType<DenseElementsAttr>(attr_name)) {
    auto dense_attr = op->getAttrOfType<DenseElementsAttr>(attr_name);
    if (auto elem_type =
            dense_attr.getElementType().dyn_cast<mlir::IntegerType>()) {
      if (elem_type.getWidth() == 32) {
        for (auto value : dense_attr.getValues<int32_t>()) {
          rst.push_back(value);
        }
      } else if (elem_type.getWidth() == 64) {
        for (auto value : dense_attr.getValues<int64_t>()) {
          rst.push_back(value);
        }
      }
    }
  } else if (op->hasAttrOfType<ArrayAttr>(attr_name)) {
    auto array_attr = op->getAttrOfType<ArrayAttr>(attr_name);
    for (auto element : array_attr) {
      if (auto int_element = element.dyn_cast<mlir::IntegerAttr>()) {
        rst.push_back(int_element.getInt());
      }
    }
  } else if (op->hasAttrOfType<IntegerAttr>(attr_name)) {
    auto int_attr = op->getAttrOfType<IntegerAttr>(attr_name);
    rst.push_back(int_attr.getInt());
  } else if (op->hasAttrOfType<BoolAttr>(attr_name)) {
    auto bool_attr = op->getAttrOfType<BoolAttr>(attr_name);
    rst.push_back(bool_attr.getValue());
  }
  return rst;
}

std::vector<int64_t> getConstantIntArrayForValue(Value v) {
  std::vector<int64_t> rst;
  if (auto shard_op = llvm::dyn_cast<ShardOp>(v.getDefiningOp())) {
    v = shard_op.getOperand();
  }
  if (v.getDefiningOp()->getName().getStringRef() == "tfg.Const") {
    return getIntArrayAttr(v.getDefiningOp(), "value");
  }
  return std::vector<int64_t>();
}

std::vector<int64_t> getShapeIntArrayForValue(Value v) {
  bool get_int_array = false;
  std::vector<int64_t> rst;
  if (v.getDefiningOp()->getName().getStringRef() == "tfg.Shape") {
    for (auto user : v.getUsers()) {
      // Assume the users of tfg.Shape include tfg.Reshape.
      // TODO(itex): Other OP maybe infer the output value of tfg.Shape.
      if (user->getName().getStringRef() != "tfg.Reshape") {
        continue;
      }
      for (auto result : user->getResults()) {
        auto prop = getShardingPropertyForValue(result);
        if (!get_int_array) {
          rst = prop->getShape();
          get_int_array = true;
        } else {
          assert(rst == prop->getShape() &&
                 "Expect all tfg.Reshape uses have same output shape");
        }
      }
    }
    return rst;
  }
  return std::vector<int64_t>();
}

std::pair<ShardingPropertyRefVec, ShardingPropertyRefVec>
getShardingPropertiesForOp(Operation* op) {
  ShardingPropertyRefVec input_hsps;
  for (auto operand : op->getOperands()) {
    input_hsps.push_back(getShardingPropertyForValue(operand));
  }
  ShardingPropertyRefVec output_hsps;
  for (auto result : op->getResults()) {
    output_hsps.push_back(getShardingPropertyForValue(result));
  }
  return {input_hsps, output_hsps};
}

void setShardingPropertiesForOp(Operation* op,
                                const ShardingPropertyRefVec& input_hsps,
                                const ShardingPropertyRefVec& output_hsps) {
  for (size_t i = 0; i < input_hsps.size(); i++) {
    setShardingPropertyForValue(op->getOperand(i), input_hsps[i]);
  }
  for (size_t i = 0; i < output_hsps.size(); i++) {
    setShardingPropertyForValue(op->getResult(i), output_hsps[i]);
  }
}

bool checkGraphShapeInferenceException(mlir::tfg::GraphOp* graph_op) {
  // Check whether there is a tensor type result that cannot derive rank
  // during shape inference. When this happens, autoshard will not be
  // implemented.
  // But there are some OP collections whose shapes cannot be deduced by
  // tensorflow, but they do not affect autoshard, so they are not abnormal.
  std::set<std::string> exclude_ops = {"tfg.FusedBatchNormV3"};
  for (Block& block : graph_op->getRegion().getBlocks()) {
    for (Operation& op : block.getOperations()) {
      // Skip some special OPs.
      if (exclude_ops.count(op.getName().getStringRef().str()) != 0) {
        continue;
      }
      // Skip some OPs which may be affected vy `exclude_ops` operand.
      bool operand_is_exclude = false;
      for (auto operand : op.getOperands()) {
        auto operand_op = operand.getDefiningOp();
        if (exclude_ops.count(operand_op->getName().getStringRef().str()) !=
            0) {
          operand_is_exclude = true;
          break;
        }
      }
      if (operand_is_exclude) {
        continue;
      }
      // Only focus on output of tensor type.
      if (llvm::dyn_cast<UnshardOp>(op) || llvm::dyn_cast<ShardOp>(op)) {
        continue;
      }
      for (auto result_id = 0; result_id < op.getNumResults() - 1;
           result_id++) {
        auto type = op.getResult(result_id).getType();
        if (type.dyn_cast<tfg::ControlType>()) {
          continue;
        }
        if (type.isa<mlir::TensorType>() == false) {
          continue;
        }
        auto tensor_ty = type.dyn_cast<mlir::TensorType>();
        if (tensor_ty.getElementType().isa<mlir::tfg::ResourceType>()) {
          continue;
        }
        if (type.isa<mlir::UnrankedTensorType>()) {
          ITEX_LOG(WARNING) << "Shape inference exception: "
                            << op.getName().getStringRef().str();
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace

/**
 * @brief Wrap an MLIR op with OpDesc. The OpDesc references but doesn't
 * own the `op`.
 *
 * @param op
 * @return HsOp
 */
as::OpDescRef mlirOpToOpDesc(Operation* op);

/**
 * @brief Wrap an MLIR graph `root_op` with the Graph. The Graph
 * references but doesn't own the `root_op`.
 *
 * @param op
 * @return as::GraphHandleRef
 */
as::GraphRef mlirGraphToGraphHandle(Operation* root_op);

/**
 * @brief Create a new HspTuner object for MLIR graph
 *
 * @param graph The graph handle, expect MLIR graph
 * @param device_info
 * @param sharding_config
 * @return as::HspTunerRef
 */
as::HspTunerRef createHspTunerMlir(as::GraphRef graph,
                                   const as::DeviceInfo& device_info,
                                   const as::ShardingConfig& sharding_config);

/**
 * @brief Set the Annotation into Graph object
 *
 * @param graph
 * @param annot
 */
void setAnnotationToGraph(as::GraphRef graph, as::HspAnnotationRef annot);

}  // namespace hs
}  // namespace mlir
