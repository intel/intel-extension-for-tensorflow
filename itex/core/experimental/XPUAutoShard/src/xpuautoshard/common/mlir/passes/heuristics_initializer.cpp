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

#include "xpuautoshard/common/mlir/passes/heuristics_initializer.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "xpuautoshard/common/analytic_cost_model.h"
#include "xpuautoshard/common/device_info.h"
#include "xpuautoshard/common/hsp_inference/hsp_inference.h"
#include "xpuautoshard/common/hsp_inference/hsp_inference_utils.h"
#include "xpuautoshard/common/mlir/passes/pass_utils.h"
#include "xpuautoshard/common/sharding_property.h"

namespace mlir {
namespace hs {

using as::ShardingProperty;
using as::SplitSpec;

namespace {

bool has_resource_user(Value value) {
  // Check whether a kResourceOP exits after this value.
  std::queue<Operation*> op_que;
  std::map<Operation*, bool> visited;
  for (auto user : value.getUsers()) {
    op_que.push(user);
    visited[user] = true;
  }
  while (!op_que.empty()) {
    Operation* op = op_que.front();
    op_que.pop();
    if (kResourceOP.find(op->getName().getStringRef().str()) !=
        kResourceOP.end()) {
      return true;
    }
    if (op->getName().getStringRef().str() == "hs.unshard_op") {
      for (auto user : op->getResult(0).getUsers()) {
        if (kResourceOP.find(user->getName().getStringRef().str()) !=
            kResourceOP.end()) {
          return true;
        }
      }
      continue;
    }
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        if (visited[user] == true) {
          continue;
        }
        op_que.push(user);
        visited[user] = true;
      }
    }
  }
  return false;
}

std::vector<Value> getBatchSplitValue(Operation* op,
                                      bool require_concrete_batchsize) {
  std::vector<Value> values;
  auto op_name = op->getName().getStringRef().str();
  if (kResourceOP.find(op_name) != kResourceOP.end()) {
    // Only split on the result of kResourceOP->Shard.
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        for (auto value : user->getResults()) {
          if (auto ranked_tensor_type =
                  value.getType().dyn_cast<mlir::RankedTensorType>()) {
            values.push_back(value);
          }
        }
      }
    }
  } else if (op_name.find("Conv2D") != std::string::npos) {
    // Split on the first operand of Conv2D type OP.
    if (op->getOperand(0).getType().dyn_cast<mlir::RankedTensorType>()) {
      values.push_back(op->getOperand(0));
    }
  } else if (op_name == "tfg.IteratorGetNext") {
    // Split on the result of IteratorGetNext->Shard.
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        for (auto value : user->getResults()) {
          // kResourceOP cannot be used across devices, if
          // IteratorGetNext->Shard contains kResourceOP, split will not be
          // performed.
          if (has_resource_user(value)) {
            continue;
          }
          if (auto ranked_tensor_type =
                  value.getType().dyn_cast<mlir::RankedTensorType>()) {
            values.push_back(value);
          }
        }
      }
    }
  }
  return values;
}

std::vector<size_t> numStagesByDevice(const DeviceInfo& device_info) {
  float min_score = std::numeric_limits<float>::max();
  size_t num_stages_of_min_score = 0;
  for (auto device : device_info.getDevices()) {
    if (min_score > device.getScore()) {
      min_score = device.getScore();
      num_stages_of_min_score = device.getNumStages();
    }
  }
  if (num_stages_of_min_score == 0) {
    // we assume the smallest score is 1 stage if not set.
    num_stages_of_min_score = 1;
  }
  std::vector<size_t> num_stages_by_device;
  for (auto device : device_info.getDevices()) {
    if (device.getNumStages() > 0) {  // follow configured deviced num_stages
      num_stages_by_device.push_back(device.getNumStages());
    } else {  // otherwise, make it a multiply of num_stages_of_min_score
      num_stages_by_device.push_back(
          size_t(std::round(device.getScore() / min_score)) *
          num_stages_of_min_score);
    }
  }
  return num_stages_by_device;
}

int64_t getLocalBatchSize(int64_t global_bs, float ratio, int64_t grain_size) {
  return int64_t(std::round(global_bs * ratio)) / grain_size * grain_size;
}

}  // anonymous namespace

bool HeuristicsInitializer::tryBatchSplitValue(
    Value value, const std::vector<float>& ratios) {
  auto prop = annot_->getShardingPropertyForValue(value);
  if (prop->isSplitAt(0)) {
    return false;
  }

  auto ranked_tensor_type = value.getType().dyn_cast<mlir::RankedTensorType>();
  assert(ranked_tensor_type &&
         "Expect batch split candidate to be a ranked tensor");
  std::pair<int64_t, std::vector<int64_t>> rank_and_shapes =
      rankAndShapes(ranked_tensor_type);
  int64_t batchsize = rank_and_shapes.second[0];  // assume batch dim is 0
  assert(batchsize != 0 && "Batch size shouldn't be zero");
  bool success;
  if (batchsize < 0) {
    // Support dynamic batch size, split by ratio
    std::vector<float> split_ratio;
    std::vector<int64_t> stage_offsets;
    if (heuristics_config_.isMultiStageEnabled()) {
      auto num_stages_by_device = numStagesByDevice(device_info_);
      int64_t num_stages_acc = 0;
      for (auto num_stages : num_stages_by_device) {
        num_stages_acc += num_stages;
        stage_offsets.push_back(num_stages_acc);
      }
      for (size_t i = 0; i < num_stages_by_device.size(); i++) {
        size_t num_stages = num_stages_by_device[i];
        for (size_t stage_num = 0; stage_num < num_stages; stage_num++) {
          // TODO(itex): Calculate and check the bs of each shard by ratio
          // Assume the shape can be divided by ratio
          split_ratio.push_back(ratios[i] / num_stages);
        }
      }
    } else {
      for (auto ratio : ratios) {
        // TODO(itex): Calculate and check the bs of each shard by ratio
        // Assume the shape can be divided by ratio
        split_ratio.push_back(ratio);
      }
    }
    success = prop->splitAt(
        0, SplitSpec::buildFromRatios(*prop, 0, split_ratio, stage_offsets));
    assert(success && "Failed to ratio split at the batch dim!");
  } else {
    // Support concrete batch size, split by size
    std::vector<int64_t> stage_offsets;
    int64_t batchsize_left = batchsize;
    std::vector<int64_t> split_sizes;
    if (heuristics_config_.isMultiStageEnabled()) {
      auto num_stages_by_device = numStagesByDevice(device_info_);
      auto total_num_stages = std::accumulate(num_stages_by_device.begin(),
                                              num_stages_by_device.end(), 0);
      int64_t num_stages_acc = 0;
      for (auto num_stages : num_stages_by_device) {
        num_stages_acc += num_stages;
        stage_offsets.push_back(num_stages_acc);
      }
      for (size_t i = 0; i < num_stages_by_device.size(); i++) {
        size_t num_stages = num_stages_by_device[i];
        auto bs_per_stage =
            getLocalBatchSize(batchsize, (1.0f / num_stages) * ratios[i],
                              heuristics_config_.getBatchGrainSize());
        for (size_t stage_num = 0; stage_num < num_stages; stage_num++) {
          auto my_batchsize = std::min(bs_per_stage, batchsize_left);
          batchsize_left -= my_batchsize;
          if (i == num_stages_by_device.size() - 1 &&
              stage_num == num_stages - 1) {
            my_batchsize += batchsize_left;
          }
          if (my_batchsize == 0) {
            llvm::errs() << "Batch size too small to split: bs = " << batchsize
                         << ", num_devices = " << device_info_.getNumDevices()
                         << ", total_num_stages = " << total_num_stages
                         << ", bs_per_stage = " << bs_per_stage
                         << ", batch_grain_size = "
                         << heuristics_config_.getBatchGrainSize() << "\n";
            return false;
          } else if (my_batchsize % heuristics_config_.getBatchGrainSize()) {
            llvm::errs() << "Unable to meet batch grain size constraints: bs = "
                         << batchsize
                         << ", num_devices = " << device_info_.getNumDevices()
                         << ", total_num_stages = " << total_num_stages
                         << ", bs_per_stage = " << bs_per_stage
                         << ", batch_grain_size = "
                         << heuristics_config_.getBatchGrainSize() << "\n";
            return false;
          }
          split_sizes.push_back(my_batchsize);
        }
      }
    } else {
      for (auto ratio : ratios) {
        int64_t my_batchsize = getLocalBatchSize(
            batchsize, ratio, heuristics_config_.getBatchGrainSize());
        my_batchsize = std::min(my_batchsize, batchsize_left);
        batchsize_left -= my_batchsize;
        if (my_batchsize == 0) {
          llvm::errs() << "Batch size too small to split: bs = " << batchsize
                       << ", num_devices = " << device_info_.getNumDevices()
                       << ", batch_grain_size = "
                       << heuristics_config_.getBatchGrainSize() << "\n";
          return false;
        } else if (my_batchsize % heuristics_config_.getBatchGrainSize()) {
          llvm::errs() << "Unable to meet batch grain size constraints: bs = "
                       << batchsize
                       << ", num_devices = " << device_info_.getNumDevices()
                       << ", batch_grain_size = "
                       << heuristics_config_.getBatchGrainSize() << "\n";
          return false;
        }
        split_sizes.push_back(my_batchsize);
      }
    }
    // TODO(itex): construct split spec with multi stage
    success = prop->splitAt(
        0, SplitSpec::buildFromSizes(*prop, 0, split_sizes, stage_offsets));
    assert(success && "Failed to size split at the batch dim!");
  }
  return success;
}

bool HeuristicsInitializer::tryBatchSplit(Operation* root_op) {
  auto&& ratios = getSplitRatios(root_op);
  auto do_batch_split = [&](bool require_concrete_batchsize) -> bool {
    for (Region& region : root_op->getRegions()) {
      for (Block& block : region.getBlocks()) {
        for (Operation& op : block.getOperations()) {
          std::vector<Value> values =
              getBatchSplitValue(&op, require_concrete_batchsize);
          for (auto value : values) {
            if (tryBatchSplitValue(value, ratios)) {
              return true;
            }
          }
        }
      }
    }
    return false;
  };
  // require concrete batch size for now
  if (do_batch_split(true)) {
    return true;
  }
  // uncomment code below to split on dynamic batch size
  // if (do_batch_split(false)) {
  //   return true;
  // }
  return false;
}

std::vector<float> HeuristicsInitializer::getSplitRatios(Operation* root_op) {
  std::vector<float> scores;
  std::vector<float> ratios;
  float total_score = 0;
  for (auto device : device_info_.getDevices()) {
    float score = device.getScore();
    if (score < 0) {
      auto cost_model =
          as::makeRef<as::AnalyticCostModel>(device.getComputeCapability());
      auto compute_characterizer = cost_model->createComputeCharacterizer();
      auto time_cost = cost_model->evaluateTime(
          compute_characterizer->characterize(mlirGraphToGraphHandle(root_op)));
      score = 1 / time_cost;
    }
    scores.push_back(score);
    total_score += score;
  }
  for (auto score : scores) {
    ratios.push_back(score / total_score);
  }
  return ratios;
}

bool HeuristicsInitializer::tryPropSingleSplitOnlyInputs(Operation* root_op) {
  bool changed = false;
  // Check the pattern:
  //   "Reshape" that expands rank -> (uninit hsp) -> "Tile" -> (split hsp)
  // The uninit hsp is most likely single split only. Do that.
  for (Region& region : root_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (!isFrameworkOp(&op)) {
          continue;
        }
        ShardingPropertyRefVec input_hsps;
        ShardingPropertyRefVec output_hsps;
        std::tie(input_hsps, output_hsps) =
            annot_->getShardingPropertiesForOp(&op);
        if (as::utils::inferWithAllInputsSingleSplitOnly(input_hsps,
                                                         output_hsps)) {
          changed = true;
        }
      }
    }
  }
  return changed;
}

bool HeuristicsInitializer::tryInitReshapeMatmul(Operation* root_op) {
  bool changed = false;
  // Reshape(n-D tensor, *) -> 2-D tensor -> MatMul, where n>2 and n-D tensor
  // shape unknown, propagate SplitSpec of dim 0 of n-D tensor to dim 0 of 2-D
  // tensor assuming the dim 0 is batch dim
  for (Region& region : root_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (  // check pattern first
            op.getName().getStringRef() == "tfg.Reshape" &&
            !op.getResult(0).use_empty() &&
            op.getResult(0).getUsers().begin()->getName().getStringRef() ==
                "tfg.MatMul") {
          auto ranked_input =
              op.getOperand(0).getType().dyn_cast<mlir::RankedTensorType>();
          auto ranked_output =
              op.getResult(0).getType().dyn_cast<mlir::RankedTensorType>();
          if (  // rank expanded
              ranked_input && ranked_output && ranked_input.getRank() > 2 &&
              ranked_output.getRank() == 2) {
            auto prop_operand =
                annot_->getShardingPropertyForValue(op.getOperand(0));
            auto prop_result =
                annot_->getShardingPropertyForValue(op.getResult(0));
            auto&& split_spec = prop_operand->getSplitSpec(0);
            if (split_spec.isInitialized() && !prop_result->isInitialized() &&
                prop_result->splitAt(0, split_spec)) {
              changed = true;
            }
          }
        }
      }
    }
  }
  return changed;
}

bool HeuristicsInitializer::tryInitReshape(Operation* root_op) {
  bool changed = false;
  // Some heuristics to initialize reshape ops, which will be
  // generalized and moved to HSP inference after complete shape
  // propagation and const prop are applied
  for (Region& region : root_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (op.getName().getStringRef() == "tfg.Reshape") {
          bool op_changed = false;
          ShardingPropertyRefVec input_hsps;
          ShardingPropertyRefVec output_hsps;
          std::tie(input_hsps, output_hsps) =
              annot_->getShardingPropertiesForOp(&op);
          if (  // expand dim case from rank 1 to 2 ranks
              input_hsps[0]->getRank() == 1 && output_hsps[0]->getRank() == 2) {
            if (!input_hsps[0]->isInitialized() &&
                output_hsps[0]->isInitialized()) {
              if (output_hsps[0]->isSplitSingleOnly()) {
                input_hsps[0]->splitSingleOnly();
                op_changed = true;
              } else {
                for (auto&& split_spec : output_hsps[0]->getSplitSpecs()) {
                  if (split_spec.getType() != SplitSpec::SplitType::SINGLE &&
                      input_hsps[0]->splitAt(0, split_spec)) {
                    op_changed = true;
                  }
                }
              }
            }
          } else if (input_hsps[0]->getRank() == 2 &&
                     output_hsps[0]->getRank() == 1) {
            // squeeze dim case from 2 to 1
            if (!output_hsps[0]->isInitialized() &&
                input_hsps[0]->isInitialized()) {
              if (input_hsps[0]->isSplitSingleOnly()) {
                output_hsps[0]->splitSingleOnly();
                op_changed = true;
              } else {
                for (auto&& split_spec : input_hsps[0]->getSplitSpecs()) {
                  if (split_spec.getType() != SplitSpec::SplitType::SINGLE &&
                      output_hsps[0]->splitAt(0, split_spec)) {
                    op_changed = true;
                  }
                }
              }
            }
          } else if (input_hsps[0]->getRank() == output_hsps[0]->getRank()) {
            // same rank, assume no shape change.
            // FIXME: check shape and make more general.
            op_changed = as::utils::elementwiseInfer(input_hsps, output_hsps, 1,
                                                     1, false);
          }
          if (op_changed) {
            changed = true;
          }
        }
      }
    }
  }
  return changed;
}

bool HeuristicsInitializer::trySingleSplitOnlyForShardOp(Operation* root_op,
                                                         bool only_const_flag) {
  bool changed = false;
  for (Region& region : root_op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        if (auto shard_op = mlir::dyn_cast<ShardOp>(op)) {
          auto define_op = shard_op->getOperand(0).getDefiningOp();
          if (only_const_flag &&
              std::find(kConstSingleSplitOP.begin(), kConstSingleSplitOP.end(),
                        define_op->getName().getStringRef()) ==
                  kConstSingleSplitOP.end()) {
            continue;
          }
          if (!shard_op.getHsp().isInitialized()) {
            auto hsps = annot_->getResultHsps(shard_op.getOperation());
            assert(hsps[0]->splitSingleOnly());
            changed = true;
          }
        }
      }
    }
  }
  return changed;
}

bool HeuristicsInitializer::initSome(Operation* root_op) {
  bool changed = false;
  changed = tryBatchSplit(root_op);

  if (!changed) {
    changed = tryInitReshapeMatmul(root_op);
  }
  if (!changed) {
    changed = tryInitReshape(root_op);
  }
  // Do single split only for `kConstSingleSplitOP`
  // In order to avoid transmission interruption caused by the inability of some
  // op sharding property to be transmitted. If not, it will cause the batch
  // size mismatch of some op operands.
  if (!changed) {
    changed = trySingleSplitOnlyForShardOp(root_op, /*only_const_falg*/ true);
  }
  // Do single split only for all remaining ShardOp
  if (!changed) {
    changed = trySingleSplitOnlyForShardOp(root_op, /*only_const_falg*/ false);
  }
  // Mark output as single split only if all inputs are single split only
  // HSP inference rules do not force this for all ops, e.g., Tile
  // Do it here.
  if (!changed) {
    changed = tryPropSingleSplitOnlyInputs(root_op);
  }
  return changed;
}

}  // namespace hs
}  // namespace mlir
