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

#include <vector>

#include "xpuautoshard/common/hsp_inference/dim_group.h"
#include "xpuautoshard/common/op_desc.h"
#include "xpuautoshard/common/sharding_property.h"

namespace as {
namespace utils {

/**
 * @brief Align the SplitSpec along all the dims in the given `dim_group`.
 *
 * @param input_props
 * @param output_props
 * @param dim_group
 * @return true
 * @return false
 */
bool inferWithIdenticalDimGroup(const ShardingPropertyRefVec& input_props,
                                const ShardingPropertyRefVec& output_props,
                                const DimGroup& dim_group);

bool inferWithContractingDimGroup(const ShardingPropertyRefVec& input_props,
                                  const ShardingPropertyRefVec& output_props,
                                  const DimGroup& dim_group);

bool inferDefault(const ShardingPropertyRefVec& input_props,
                  const ShardingPropertyRefVec& output_props);

bool inferWithAllInputsSingleSplitOnly(
    const ShardingPropertyRefVec& input_props,
    const ShardingPropertyRefVec& output_props);

bool inferWithAllSingleSplitOnlyExceptOne(
    const ShardingPropertyRefVec& input_props,
    const ShardingPropertyRefVec& output_props);

bool inferWithDimGroups(const ShardingPropertyRefVec& input_props,
                        const ShardingPropertyRefVec& output_props,
                        const std::vector<as::DimGroup>& dim_groups);

/**
 * @brief The default inference rule for propagating shape or size tensor
 * property. Usually operations on a shape or size tensor are unary ops or
 * binary ops. Hence, we do the inference per the following rules:
 * 1. For unary op, we infer from both directions, i.e. input -> output and
 * output -> input.
 * 2. For binary op, we infer only from input to output, i.e., input[0] ->
 * output, input[1] -> output
 *
 * @param input_props
 * @param output_props
 * @param is_unary True if it is for unary op, False if it is binary
 * @return true
 * @return false
 */
bool inferShapeOrSizeTensorForUnaryOrBinary(
    const ShardingPropertyRefVec& input_props,
    const ShardingPropertyRefVec& output_props, bool is_unary = true);

bool trySplitSingleOnly(ShardingPropertyRef prop, int64_t dim = -1);

/**
 * @brief Slice the shape on a shape tensor with given split specs if
 * applicable.
 *
 * This handles the case where a concrete shape (from a constant op) is used as
 * an input to an op that works on tensors with multiple splits and expects the
 * concrete shape values also correspond to the split tensors. For example, a
 * reshape tensor describes an unsplit tensor while reshape operation operates
 * on a multi-split tensor. This happens when the model is given inputs with
 * concrete shapes and the reshape tensor is specialized with concrete values
 * corresponding to the single device graph. A shape mismatch happens and we
 * should adjust the reshape size to make the shape matched. We mark the
 * "reshape" with ShapeSlicePostOp for the lowering pass to fill the gap.
 *
 * As a concrete example, the Reshape operates on a tensor with [64, 10, 10] and
 * the reshape is [64, 100]. The tensor is split into two shards with [32, 10,
 * 10] each while the reshape is still [64, 100]. We should change [64, 100] to
 * [32, 100] for each shard to match the sharded scenario.
 *
 * We assume the following pre-conditions to be satisfied before we add
 * ShapeSlicePostOp:
 * 1. The `prop` annotates a shape tensor and describes a single-split-only
 * tensor. This is necessary since this creates a shape mismatch.
 * 2. The `prop` doesn't have a post-op attached.
 *
 * For simplicity, we also assume the following pre-conditions to hold:
 * 1. shape[split_dims] have concrete sizes.
 * 2. split_specs all have SplitSpec::SplitType::SIZE types. The total size
 * matches corresonding shape[dim].
 *
 * @param prop The sharding property of the shape tensor
 * @param shape The unsliced shape
 * @param split_specs Describes how the tensor taking the shape to slice is
 * split. With this, we decide how the shape is sliced.
 * @param split_dims Describe the dims corresponding to the split_specs. The
 * following invaraint should hold:
 *                   1. split_specs.size() == split_dims.size()
 *                   2. all split_dim in split_dims < shape.size()
 * @return true
 * @return false
 */
bool trySliceShape(ShardingPropertyRef prop, const std::vector<int64_t>& shape,
                   const std::vector<SplitSpec>& split_specs,
                   const std::vector<int64_t>& split_dims);

bool isSliceShape(ShardingPropertyRef prop);

bool elementwiseInfer(const ShardingPropertyRefVec& input_props,
                      const ShardingPropertyRefVec& output_props,
                      unsigned num_inputs, unsigned num_outputs,
                      bool broadcastable = true,
                      const std::vector<int64_t>& exclude_dims = {});

bool contractionInfer(const ShardingPropertyRefVec& input_props,
                      const ShardingPropertyRefVec& output_props,
                      const std::vector<int64_t>& contraction_dims,
                      ContractionType contraction_type, bool keep_dims = false);

bool squeezeInfer(const ShardingPropertyRefVec& input_props,
                  const ShardingPropertyRefVec& output_props,
                  const std::vector<int64_t>& squeeze_dims);

bool packInfer(const ShardingPropertyRefVec& input_props,
               const ShardingPropertyRefVec& output_props, int64_t pack_dim);

}  // namespace utils
}  // namespace as
