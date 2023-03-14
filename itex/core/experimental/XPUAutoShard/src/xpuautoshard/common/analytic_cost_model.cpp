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

#include "xpuautoshard/common/analytic_cost_model.h"

#include <cmath>
#include <vector>

namespace as {

#define RETURN_IF_OPERAND_DYNAMIC(op_desc, operand_num)         \
  do {                                                          \
    if (!(op_desc)->getOperand(operand_num).isConcreteDims()) { \
      return makeRef<QualitativeComputeCharacteristics>(        \
          (op_desc)->getOperand(operand_num).getElementType()); \
    }                                                           \
  } while (0)

#define RETURN_IF_RESULT_DYNAMIC(op_desc, result_num)         \
  do {                                                        \
    if (!(op_desc)->getResult(result_num).isConcreteDims()) { \
      return makeRef<QualitativeComputeCharacteristics>(      \
          (op_desc)->getResult(result_num).getElementType()); \
    }                                                         \
  } while (0)

#define RETURN_IF_OPERAND_DYNAMIC_TRAIN(op_desc, operand_num)         \
  do {                                                                \
    if (!(op_desc)->getOperand(operand_num).isConcreteDims()) {       \
      return makeRef<QualitativeComputeCharacteristics>(              \
          (op_desc)->getOperand(operand_num).getElementType(), true); \
    }                                                                 \
  } while (0)

#define RETURN_IF_RESULT_DYNAMIC_TRAIN(op_desc, result_num)         \
  do {                                                              \
    if (!(op_desc)->getResult(result_num).isConcreteDims()) {       \
      return makeRef<QualitativeComputeCharacteristics>(            \
          (op_desc)->getResult(result_num).getElementType(), true); \
    }                                                               \
  } while (0)

QuantitativeComputeCharacteristics&
QuantitativeComputeCharacteristics::operator+=(
    const QuantitativeComputeCharacteristics& rhs) {
  num_float_ops_ += rhs.num_float_ops_;
  num_bfloat16_ops_ += rhs.num_bfloat16_ops_;
  num_float16_ops_ += rhs.num_float16_ops_;
  num_int8_ops_ += rhs.num_int8_ops_;
  memory_load_bytes_ += rhs.memory_load_bytes_;
  memory_store_bytes_ += rhs.memory_store_bytes_;
  return *this;
}

QualitativeComputeCharacteristics::QualitativeComputeCharacteristics(
    DataType dtype, bool maybe_training)
    : has_bfloat16_(false),
      has_float16_(false),
      has_int8_(false),
      maybe_training_(maybe_training) {
  switch (dtype) {
    case DataType::BFLOAT16:
      has_bfloat16_ = true;
      break;
    case DataType::FLOAT16:
      has_float16_ = true;
      break;
    default:
      // TODO(itex) support int8
      break;
  }
}

QualitativeComputeCharacteristics::QualitativeComputeCharacteristics(
    const QuantitativeComputeCharacteristics& quan_comp_ch)
    : QualitativeComputeCharacteristics() {
  // TODO(itex): support initiate maybe_training_
  has_bfloat16_ = quan_comp_ch.getNumBfloat16Ops() > 0;
  has_float16_ = quan_comp_ch.getNumFloat16Ops() > 0;
  has_int8_ = quan_comp_ch.getNumInt8Ops() > 0;
}

QualitativeComputeCharacteristics&
QualitativeComputeCharacteristics::operator+=(
    const QualitativeComputeCharacteristics& rhs) {
  maybe_training_ |= rhs.maybe_training_;
  has_bfloat16_ |= rhs.has_bfloat16_;
  has_float16_ |= rhs.has_float16_;
  has_int8_ |= rhs.has_int8_;
  return *this;
}

ComputeCharacterizerRef AnalyticCostModel::createComputeCharacterizer() {
  return makeRef<AnalyticComputeCharacterizer, ComputeCharacterizer>();
}

CostModel::TimeCost AnalyticCostModel::evaluateTimeQualitative(
    QualitativeComputeCharacteristicsRef comp_ch) {
  // estimate with normalized hardware capability (NHC) when the shapes are
  // unknown and the analytic compute characteristics cannot be decided.
  // Training:  TFLOPS^0.4 * Memory_Bandwidth^0.5 * L1^0.025 * LLC^0.075
  // Inference: TFLOPS^0.5 * Memory_Bandwidth^0.3 * L1^0.05 * LLC^0.15
  float tops_weight = comp_ch->maybeTraining() ? 0.4 : 0.5;
  float mem_weight = comp_ch->maybeTraining() ? 0.5 : 0.3;
  // TODO(itex): model cache capability
  // float l1_weight = comp_ch->maybeTraining() ? 0.025 : 0.05;
  // float llc_weight = comp_ch->maybeTraining() ? 0.075 : 0.15;

  float tops = device_cap_.getFloatOPS();
  if (comp_ch->hasBfloat16Compute()) {
    tops = device_cap_.getBfloat16OPS();
  } else if (comp_ch->hasFloat16Compute()) {
    tops = device_cap_.getFloat16OPS();
  } else if (comp_ch->hasInt8Compute()) {
    tops = device_cap_.getInt8OPS();
  }
  // cost is reciprocal of capability
  return std::pow(tops, -tops_weight) *
         std::pow(device_cap_.getMemBandwidthTriad(), -mem_weight);
}

CostModel::TimeCost AnalyticCostModel::evaluateTime(
    ComputeCharacteristicsRef comp_ch) {
  if (auto qual_comp_ch =
          downcastRef<QualitativeComputeCharacteristics>(comp_ch)) {
    return evaluateTimeQualitative(qual_comp_ch);
  } else {
    auto&& quan_comp_ch =
        downcastRef<QuantitativeComputeCharacteristics>(comp_ch);
    assert(quan_comp_ch && "Expect QuantitativeComputeCharacteristics");
    return (evaluateTimeFloat(quan_comp_ch) +
            evaluateTimeBfloat16(quan_comp_ch) +
            evaluateTimeFloat16(quan_comp_ch) + evaluateTimeInt8(quan_comp_ch) +
            evaluateTimeMemory(quan_comp_ch));
  }
}

CostModel::TimeCost AnalyticCostModel::evaluateTimeFloat(
    QuantitativeComputeCharacteristicsRef comp_ch) {
  return comp_ch->getNumFloatOps() / device_cap_.getFloatOPS();
}
CostModel::TimeCost AnalyticCostModel::evaluateTimeBfloat16(
    QuantitativeComputeCharacteristicsRef comp_ch) {
  return comp_ch->getNumBfloat16Ops() / device_cap_.getBfloat16OPS();
}

CostModel::TimeCost AnalyticCostModel::evaluateTimeFloat16(
    QuantitativeComputeCharacteristicsRef comp_ch) {
  return comp_ch->getNumFloat16Ops() / device_cap_.getFloat16OPS();
}
CostModel::TimeCost AnalyticCostModel::evaluateTimeInt8(
    QuantitativeComputeCharacteristicsRef comp_ch) {
  return comp_ch->getNumInt8Ops() / device_cap_.getInt8OPS();
}
CostModel::TimeCost AnalyticCostModel::evaluateTimeMemory(
    QuantitativeComputeCharacteristicsRef comp_ch) {
  return (comp_ch->getMemoryLoadBytes() + comp_ch->getMemoryStoreBytes()) /
         device_cap_.getMemBandwidthTriad();
}

ComputeCharacteristicsRef AnalyticComputeCharacterizer::characterize(
    GraphRef graph) {
  auto&& graph_quan_ch = makeRef<QuantitativeComputeCharacteristics>();
  auto&& graph_qual_ch = makeRef<QualitativeComputeCharacteristics>();
  bool has_qual = false;
  auto&& bf_traversal = graph->getBreadthFirstIterRange();
  for (OpDescRef op_desc : *bf_traversal) {
    auto&& op_ch = characterize(op_desc);
    if (auto quan_op_ch =
            downcastRef<QuantitativeComputeCharacteristics>(op_ch)) {
      *graph_quan_ch += *quan_op_ch;
      *graph_qual_ch += *quan_op_ch;
    } else if (auto qual_op_ch =
                   downcastRef<QualitativeComputeCharacteristics>(op_ch)) {
      *graph_qual_ch += *qual_op_ch;
      has_qual = true;
    } else {
      // return unknown characteristics if any of the op reports unknown.
      assert(isRef<UnknownComputeCharacteristics>(op_ch));
      return op_ch;
    }
  }
  return has_qual ? (ComputeCharacteristicsRef)graph_qual_ch
                  : (ComputeCharacteristicsRef)graph_quan_ch;
}

ComputeCharacteristicsRef AnalyticComputeCharacterizer::characterize(
    OpDescRef op_desc) {
  if (op_desc->getName() == "tfg.Conv2D") {
    return characterizeConvForward(op_desc);
  } else if (op_desc->getName() == "tfg.Conv2DBackpropInput") {
    return characterizeConvBackwardData(op_desc);
  } else if (op_desc->getName() == "tfg.ConvBackwardWeight") {
    return characterizeConvBackwardWeight(op_desc);
  } else if (op_desc->getName() == "tfg.MatMul") {
    return characterizeMatMul(op_desc);
  } else if (op_desc->getName() == "tfg.ShapeN" ||
             op_desc->getName() == "tfg.Shape" ||
             op_desc->getName() == "tfg.Size" ||
             op_desc->getName() == "tfg.Squeeze" ||
             op_desc->getName() == "tfg.ExpandDims" ||
             op_desc->getName().find("tfg.", 0) !=
                 0  // not start with tfg namespace
  ) {
    // zero
    return makeRef<QuantitativeComputeCharacteristics>();
  } else {
    return characterizeMemoryOp(op_desc);
  }
}

ComputeCharacteristicsRef AnalyticComputeCharacterizer::characterizeMemoryOp(
    OpDescRef op_desc) {
  auto analytic_ch = makeRef<QuantitativeComputeCharacteristics>();
  for (size_t i = 0; i < op_desc->getNumOperands(); i++) {
    if (op_desc->getOperand(i).isRanked()) {
      RETURN_IF_OPERAND_DYNAMIC(op_desc, i);
      addMemoryOps(analytic_ch, op_desc->getOperand(i).getElementType(),
                   op_desc->getOperand(i).getNumElements(), true);
    }
  }
  for (size_t i = 0; i < op_desc->getNumResults(); i++) {
    if (op_desc->getResult(i).isRanked()) {
      RETURN_IF_RESULT_DYNAMIC(op_desc, i);
      addMemoryOps(analytic_ch, op_desc->getResult(i).getElementType(),
                   op_desc->getResult(i).getNumElements(), true);
    }
  }
  return analytic_ch;
}

void AnalyticComputeCharacterizer::addMemoryOps(
    QuantitativeComputeCharacteristicsRef analytic_ch, DataType dtype,
    float ops, bool is_load) {
  switch (dtype) {
    case DataType::FLOAT32:
      if (is_load) {
        analytic_ch->setMemoryLoadBytes(analytic_ch->getMemoryLoadBytes() +
                                        ops * 4);
      } else {
        analytic_ch->setMemoryStoreBytes(analytic_ch->getMemoryStoreBytes() +
                                         ops * 4);
      }
      break;
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
      if (is_load) {
        analytic_ch->setMemoryLoadBytes(analytic_ch->getMemoryLoadBytes() +
                                        ops * 2);
      } else {
        analytic_ch->setMemoryStoreBytes(analytic_ch->getMemoryStoreBytes() +
                                         ops * 2);
      }
      break;
    default:
      // TODO(itex): support int8
      break;
  }
}

void AnalyticComputeCharacterizer::addMultiplyAddComputeOps(
    QuantitativeComputeCharacteristicsRef analytic_ch, DataType dtype,
    float ops) {
  switch (dtype) {
    case DataType::FLOAT32:
      analytic_ch->setNumFloatOps(ops);
      break;
    case DataType::FLOAT16:
      analytic_ch->setNumFloat16Ops(ops);
      break;
    case DataType::BFLOAT16:
      analytic_ch->setNumBfloat16Ops(ops);
      break;
    default:
      // TODO(itex): support int8
      break;
  }
}

float AnalyticComputeCharacterizer::getConvOps(
    int64_t bs, int64_t ic, int64_t oc, const std::vector<int64_t>& kernel_dims,
    const std::vector<int64_t>& out_dims) {
  float ops = 2.0f;
  for (size_t i = 0; i < kernel_dims.size(); i++) {
    ops *= kernel_dims[i];
  }
  for (size_t i = 0; i < out_dims.size(); i++) {
    ops *= out_dims[i];
  }
  return ops * bs * ic * oc;
}

ComputeCharacteristicsRef AnalyticComputeCharacterizer::characterizeConvForward(
    OpDescRef op_desc) {
  RETURN_IF_OPERAND_DYNAMIC(op_desc, 0);
  RETURN_IF_OPERAND_DYNAMIC(op_desc, 1);
  RETURN_IF_RESULT_DYNAMIC(op_desc, 0);
  if (!op_desc->getOperand(0).isConcreteDims() ||
      !op_desc->getOperand(1).isConcreteDims() ||
      !op_desc->getResult(0).isConcreteDims()) {
    return UnknownComputeCharacteristics::get();
  }
  // TODO(itex): assume channels last for now
  int64_t rank = op_desc->getOperand(0).getRank();
  DataType dtype = op_desc->getOperand(0).getElementType();
  float ops =
      getConvOps(op_desc->getOperand(0).getDimSize(0),         // bs
                 op_desc->getOperand(0).getDimSize(rank - 1),  // ic
                 op_desc->getOperand(1).getDimSize(rank - 1),  // oc
                 {                                             // kernel dims
                  op_desc->getOperand(1).getDimSize(1),
                  (rank > 3 ? op_desc->getOperand(1).getDimSize(2) : 1),
                  (rank > 4 ? op_desc->getOperand(1).getDimSize(3) : 1)},
                 {// spatial dims
                  op_desc->getResult(0).getDimSize(1),
                  (rank > 3 ? op_desc->getResult(0).getDimSize(1) : 1),
                  (rank > 4 ? op_desc->getResult(0).getDimSize(1) : 1)});
  auto analytic_ch = makeRef<QuantitativeComputeCharacteristics>();
  addMultiplyAddComputeOps(analytic_ch, dtype, ops);
  return analytic_ch;
}

ComputeCharacteristicsRef
AnalyticComputeCharacterizer::characterizeConvBackwardData(OpDescRef op_desc) {
  RETURN_IF_OPERAND_DYNAMIC_TRAIN(op_desc, 1);
  RETURN_IF_OPERAND_DYNAMIC_TRAIN(op_desc, 2);
  RETURN_IF_RESULT_DYNAMIC_TRAIN(op_desc, 0);
  int64_t rank = op_desc->getOperand(2).getRank();
  DataType dtype = op_desc->getOperand(2).getElementType();
  float ops =
      getConvOps(op_desc->getOperand(2).getDimSize(0),         // bs
                 op_desc->getOperand(2).getDimSize(rank - 1),  // ic
                 op_desc->getResult(0).getDimSize(rank - 1),   // oc
                 {                                             // kernel dims
                  op_desc->getOperand(1).getDimSize(1),
                  (rank > 3 ? op_desc->getOperand(1).getDimSize(2) : 1),
                  (rank > 4 ? op_desc->getOperand(1).getDimSize(3) : 1)},
                 {// out dims
                  op_desc->getResult(0).getDimSize(1),
                  (rank > 3 ? op_desc->getResult(0).getDimSize(1) : 1),
                  (rank > 4 ? op_desc->getResult(0).getDimSize(1) : 1)});
  auto analytic_ch = makeRef<QuantitativeComputeCharacteristics>();
  addMultiplyAddComputeOps(analytic_ch, dtype, ops);
  return analytic_ch;
}

ComputeCharacteristicsRef
AnalyticComputeCharacterizer::characterizeConvBackwardWeight(
    OpDescRef op_desc) {
  RETURN_IF_OPERAND_DYNAMIC_TRAIN(op_desc, 0);
  RETURN_IF_OPERAND_DYNAMIC_TRAIN(op_desc, 2);
  RETURN_IF_RESULT_DYNAMIC_TRAIN(op_desc, 0);
  int64_t rank = op_desc->getOperand(0).getRank();
  DataType dtype = op_desc->getOperand(0).getElementType();
  float ops =
      getConvOps(op_desc->getOperand(0).getDimSize(0),         // bs
                 op_desc->getOperand(0).getDimSize(rank - 1),  // ic
                 op_desc->getOperand(2).getDimSize(rank - 1),  // oc
                 {                                             // kernel dims
                  op_desc->getResult(0).getDimSize(1),
                  (rank > 3 ? op_desc->getResult(0).getDimSize(2) : 1),
                  (rank > 4 ? op_desc->getResult(0).getDimSize(3) : 1)},
                 {// out dims
                  op_desc->getOperand(2).getDimSize(1),
                  (rank > 3 ? op_desc->getOperand(2).getDimSize(1) : 1),
                  (rank > 4 ? op_desc->getOperand(2).getDimSize(1) : 1)});
  auto analytic_ch = makeRef<QuantitativeComputeCharacteristics>();
  addMultiplyAddComputeOps(analytic_ch, dtype, ops);
  return analytic_ch;
}

ComputeCharacteristicsRef AnalyticComputeCharacterizer::characterizeMatMul(
    OpDescRef op_desc) {
  RETURN_IF_OPERAND_DYNAMIC(op_desc, 0);
  RETURN_IF_OPERAND_DYNAMIC(op_desc, 1);
  RETURN_IF_RESULT_DYNAMIC(op_desc, 0);
  int64_t m_dim = op_desc->getAttrBool("transpose_a") ? 1 : 0;
  int64_t k_dim = op_desc->getAttrBool("transpose_a") ? 0 : 1;
  int64_t n_dim = op_desc->getAttrBool("transpose_b") ? 0 : 1;
  DataType dtype = op_desc->getOperand(0).getElementType();
  auto analytic_ch = makeRef<QuantitativeComputeCharacteristics>();
  addMultiplyAddComputeOps(analytic_ch, dtype, 2.0f * m_dim * k_dim * n_dim);
  return analytic_ch;
}
}  // namespace as
