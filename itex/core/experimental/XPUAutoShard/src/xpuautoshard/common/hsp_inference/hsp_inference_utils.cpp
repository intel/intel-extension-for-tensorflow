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

#include "xpuautoshard/common/hsp_inference/hsp_inference_utils.h"

#include <algorithm>
#include <memory>

#include "xpuautoshard/common/hsp_exception.h"

namespace as {
namespace utils {

bool inferWithIdenticalDimGroup(const ShardingPropertyRefVec& input_props,
                                const ShardingPropertyRefVec& output_props,
                                const DimGroup& dim_group) {
  auto is_equivalent = [](const SplitSpec& split_spec_1,
                          const SplitSpec& split_spec_2) -> bool {
    if (split_spec_1.getType() != split_spec_2.getType()) {
      return false;
    }
    auto&& sizes_vec_1 = split_spec_1.getSizes();
    auto&& sizes_vec_2 = split_spec_2.getSizes();
    if (sizes_vec_1.size() != sizes_vec_2.size()) {
      return false;
    }
    auto sum_1 = split_spec_1.getTotalSize();
    auto sum_2 = split_spec_2.getTotalSize();
    for (auto id = 0; id < sizes_vec_1.size(); id++) {
      if (sizes_vec_1[id] / sum_1 != sizes_vec_2[id] / sum_2) {
        return false;
      }
    }
    return true;
  };
  bool changed = false;
  ShardingPropertyRef splitProp = nullptr;
  SplitSpec split_spec;
  DimItem dim_item;
  unsigned num_initialized = 0;
  for (auto this_dim_item : dim_group) {
    auto& prop = this_dim_item.isInput() ? input_props[this_dim_item.getNum()]
                                         : output_props[this_dim_item.getNum()];
    if (prop->isSplitAt(this_dim_item.getDim())) {
      num_initialized++;
      auto this_split_spec = prop->getSplitSpec(this_dim_item.getDim());
      if (splitProp == nullptr ||
          (split_spec.getType() == SplitSpec::SplitType::SINGLE &&
           this_split_spec.getType() != SplitSpec::SplitType::SINGLE)) {
        splitProp = prop;
        split_spec = this_split_spec;
        dim_item = this_dim_item;
      } else if (  // Allow single split paired with other split types
          split_spec.getType() != SplitSpec::SplitType::SINGLE &&
          this_split_spec.getType() != SplitSpec::SplitType::SINGLE &&
          is_equivalent(split_spec, this_split_spec) == false) {
        throw SplitSpecMismatchException(dim_item.isInput()
                                             ? *input_props[dim_item.getNum()]
                                             : *output_props[dim_item.getNum()],
                                         dim_item, *prop, this_dim_item);
      }
    }
  }
  if (  // Align the the split spec if any dim is initialized while some dim not
      0 < num_initialized && num_initialized < dim_group.size() &&
      (  // Only align single split if there is only one dim not initialized
          split_spec.getType() != SplitSpec::SplitType::SINGLE ||
          dim_group.size() - num_initialized == 1)) {
    for (auto this_dim_item : dim_group) {
      auto& prop = this_dim_item.isInput()
                       ? input_props[this_dim_item.getNum()]
                       : output_props[this_dim_item.getNum()];
      if (!prop->isSplitAt(this_dim_item.getDim())) {
        assert(splitProp != nullptr);
        bool success = prop->splitAt(this_dim_item.getDim(), split_spec);
        if (!success) {
          throw SplitAtException(*prop, this_dim_item, split_spec);
        }
        changed = true;
      }
    }
  }
  return changed;
}

bool inferWithContractingDimGroup(const ShardingPropertyRefVec& input_props,
                                  const ShardingPropertyRefVec& output_props,
                                  const DimGroup& dim_group) {
  assert(dim_group.getType() == DimGroupType::CONTRACTING &&
         "Expect contracting DimGroup");
  bool changed = false;
  // Apply contraction with ALL_REDUCE_SUM post-op. We assume the contraction is
  // applied to the first output and there is only one output.
  auto&& dim_item = *dim_group.begin();
  if (input_props[dim_item.getNum()]->isInitialized() &&
      input_props[dim_item.getNum()]->isSplitAt(dim_item.getDim()) &&
      input_props[dim_item.getNum()]
              ->getSplitSpec(dim_item.getDim())
              .getType() != SplitSpec::SplitType::SINGLE &&
      output_props[dim_group.getContractionOutputNum()]->getNumPostOps() == 0) {
    // TODO(itex): only support splitting on one dim only,
    // make sure other dims are single split
    bool single_dim_split = true;
    auto&& split_specs = input_props[dim_item.getNum()]->getSplitSpecs();
    for (int64_t dim = 0; dim < split_specs.size(); dim++) {
      if (dim != dim_item.getDim() &&
          split_specs[dim].getType() != SplitSpec::SplitType::SINGLE) {
        single_dim_split = false;
        break;
      }
    }
    if (single_dim_split) {
      if (dim_group.getContractionType() == ContractionType::SUM) {
        output_props[dim_group.getContractionOutputNum()]->appendPostOp(
            std::make_unique<AllReduceSumPostOp>());
      } else if (dim_group.getContractionType() == ContractionType::Max) {
        output_props[dim_group.getContractionOutputNum()]->appendPostOp(
            std::make_unique<AllReduceMaxPostOp>());
      } else if (dim_group.getContractionType() == ContractionType::Min) {
        output_props[dim_group.getContractionOutputNum()]->appendPostOp(
            std::make_unique<AllReduceMinPostOp>());
      } else if (dim_group.getContractionType() == ContractionType::Prod) {
        output_props[dim_group.getContractionOutputNum()]->appendPostOp(
            std::make_unique<AllReduceProdPostOp>());
      } else if (dim_group.getContractionType() == ContractionType::L2) {
        output_props[dim_group.getContractionOutputNum()]->appendPostOp(
            std::make_unique<AllReduceL2PostOp>());
      } else {
        assert(dim_group.getContractionType() == ContractionType::MEAN &&
               "Expect contraction type is mean");
        auto shard_descs =
            input_props[dim_item.getNum()]->getShardDescriptors();
        std::vector<float> weights;
        for (auto&& shard_desc : shard_descs) {
          weights.push_back(shard_desc.getRatio());
        }
        output_props[dim_group.getContractionOutputNum()]->appendPostOp(
            std::make_unique<WeightedScalePostOp>(weights));
        output_props[dim_group.getContractionOutputNum()]->appendPostOp(
            std::make_unique<AllReduceSumPostOp>());
      }
      changed = true;
    }
  }
  return changed;
}

bool inferWithAllInputsSingleSplitOnly(
    const ShardingPropertyRefVec& input_props,
    const ShardingPropertyRefVec& output_props) {
  bool changed = false;
  // Conservatively work with single split only HSPs when
  // all input HSPs are single split only with the same device,
  // then output HSPs follow
  auto all_single_split_only = std::all_of(
      input_props.begin(), input_props.end(),
      [](ShardingPropertyRef prop) { return prop->isSplitSingleOnly(); });
  auto same_device_set = input_props.size() > 0 &&
                         std::all_of(input_props.begin() + 1, input_props.end(),
                                     [&input_props](ShardingPropertyRef prop) {
                                       return input_props[0]->getDeviceIds() ==
                                              prop->getDeviceIds();
                                     });
  if (all_single_split_only && same_device_set) {
    for (auto& output_prop : output_props) {
      if (!output_prop->isInitialized()) {
        assert(output_prop->splitSingleOnly());
        changed = true;
      }
    }
  }
  return changed;
}

bool inferDefault(const ShardingPropertyRefVec& input_props,
                  const ShardingPropertyRefVec& output_props) {
  return inferWithAllInputsSingleSplitOnly(input_props, output_props);
}

bool inferShapeOrSizeTensorForUnaryOrBinary(
    const ShardingPropertyRefVec& input_props,
    const ShardingPropertyRefVec& output_props, bool is_unary) {
  bool changed = false;
  if (is_unary) {
    assert(input_props.size() >= 1 && output_props.size() >= 1 &&
           "Need at least 1 input and 1 output hsps to infer shape or size "
           "tensor for unary op");
    changed |= input_props[0]->setShapeOrSizeTensorWith(*output_props[0]);
    changed |= output_props[0]->setShapeOrSizeTensorWith(*input_props[0]);
  } else {
    assert(input_props.size() >= 1 && output_props.size() >= 1 &&
           "Need at least 1 input and 1 output hsps to infer shape or size "
           "tensor for binary op");
    changed |= output_props[0]->setShapeOrSizeTensorWith(*input_props[0]);
    changed |= output_props[0]->setShapeOrSizeTensorWith(*input_props[1]);
  }
  return changed;
}

bool inferWithDimGroups(const ShardingPropertyRefVec& input_props,
                        const ShardingPropertyRefVec& output_props,
                        const std::vector<DimGroup>& dim_groups) {
  bool changed = false;
  for (auto dim_group : dim_groups) {
    switch (dim_group.getType()) {
      case DimGroupType::IDENTICAL:
        changed |=
            inferWithIdenticalDimGroup(input_props, output_props, dim_group);
        break;
      case DimGroupType::CONTRACTING:
        changed |=
            inferWithContractingDimGroup(input_props, output_props, dim_group);
        break;
      case DimGroupType::WINDOWED:
        // TODO(itex): Handle windowed contraction dims, for halo split
        break;
      case DimGroupType::BROADCASTING:
        // TODO(itex): Handle broadcasting
        break;
      default:
        break;
    }
  }
  return changed;
}

bool trySplitSingleOnly(ShardingPropertyRef prop, int64_t dim) {
  bool changed = false;
  if (!prop->isInitialized()) {
    if (dim < 0) {
      if (!prop->isInitialized()) {
        if (!prop->splitSingleOnly()) {
          return changed;
        }
        changed = true;
      }
    } else {
      if (!prop->isSplitAt(dim)) {
        assert(prop->splitSingleAt(dim));
        changed = true;
      }
    }
  }
  return changed;
}

bool elementwiseInfer(const ShardingPropertyRefVec& input_props,
                      const ShardingPropertyRefVec& output_props,
                      unsigned num_inputs, unsigned num_outputs,
                      bool broadcastable,
                      const std::vector<int64_t>& exclude_dims) {
  bool changed = false;
  assert(input_props.size() >= num_inputs &&
         "Number of operands less than expected for Elementwise op");
  assert(output_props.size() >= num_outputs &&
         "Number of results less than expected for Elementwise op");

  int64_t input_rank = UNRANKED;
  int64_t output_rank = UNRANKED;
  for (unsigned int num = 0; num < num_inputs; num++) {
    if (input_rank < 0) {
      input_rank = input_props[num]->getRank();
    } else {
      if (broadcastable) {
        input_rank = std::max(input_rank, input_props[num]->getRank());
      } else {
        assert(input_rank == input_props[num]->getRank() &&
               "Expect rank for all input/output of non-broadcastable "
               "Elementwise op should be equal");
      }
    }
  }
  for (unsigned int num = 0; num < num_outputs; num++) {
    if (output_rank < 0) {
      output_rank = output_props[num]->getRank();
    } else {
      assert(output_rank == output_props[num]->getRank() &&
             "Expect rank for all output of Elementwise op should be equal");
    }
  }
  int64_t rank = UNRANKED;
  if (input_rank >= 0 && output_rank >= 0) {
    assert(input_rank == output_rank &&
           "Expect input and output ranks equivalent for Elementwise op");
    rank = input_rank;
  } else {
    rank = input_rank >= 0 ? input_rank : output_rank;
  }
  if (rank == UNRANKED) {
    for (int64_t num = 0; num < num_inputs; num++) {
      trySplitSingleOnly(input_props[num]);
    }
    for (int64_t num = 0; num < num_outputs; num++) {
      trySplitSingleOnly(output_props[num]);
    }
  }

  std::vector<DimGroup> dim_groups;
  for (int64_t i = 0; i < rank; i++) {
    std::vector<DimItem> dim_items;
    bool is_exclude_dim = (std::find(exclude_dims.begin(), exclude_dims.end(),
                                     i) != exclude_dims.end());
    for (int64_t num = 0; num < num_inputs; num++) {
      if (!broadcastable || rank - i <= input_props[num]->getRank()) {
        if (!is_exclude_dim) {
          dim_items.emplace_back(
              DimItem(/* is_input = */ true, /* num = */ num,
                      /* dim = */ i - rank + input_props[num]->getRank()));
        }
        // NOTE: The split rule is not enforced on the excluded dims for now
      }
    }
    for (int64_t num = 0; num < num_outputs; num++) {
      if (!is_exclude_dim) {
        dim_items.emplace_back(
            DimItem(/* is_input = */ false, /* num = */ num, /* dim = */ i));
      }
      // NOTE: The split rule is not enforced on the excluded dims for now
    }
    dim_groups.emplace_back(DimGroup::create(dim_items));
  }
  changed |= inferWithDimGroups(input_props, output_props, dim_groups);
  if (num_inputs == 1 &&
      num_outputs == 1) {  // infer shape tensor property for unary op
    changed |= inferShapeOrSizeTensorForUnaryOrBinary(input_props, output_props,
                                                      /*is_unary*/ true);
  }
  if (num_inputs == 2 &&
      num_outputs == 1) {  // infer shape tensor property for binary op
    changed |= inferShapeOrSizeTensorForUnaryOrBinary(input_props, output_props,
                                                      /*is_unary*/ false);
  }
  return changed;
}

bool contractionInfer(const ShardingPropertyRefVec& input_props,
                      const ShardingPropertyRefVec& output_props,
                      const std::vector<int64_t>& contraction_dims,
                      ContractionType contraction_type, bool keep_dims) {
  bool changed = inferWithAllInputsSingleSplitOnly(input_props, output_props);
  if (contraction_dims.size() == 0) {
    // we were not able to infer the reduction dims from the frontend
    return changed;
  }
  std::vector<DimGroup> dim_groups;
  std::vector<int64_t> identical_input_dims;   // non-reduction input dims
  std::vector<int64_t> identical_output_dims;  // non-reduction output dims
  // check if this dim is non-reduction dim
  auto is_contraction_dim = [&](int64_t dim) -> bool {
    for (auto contraction_dim : contraction_dims) {
      if (dim == (contraction_dim + input_props[0]->getRank()) %
                     input_props[0]->getRank()) {
        return true;
      }
    }
    return false;
  };
  // Get input dims of identical dim groups.
  for (int64_t dim = 0; dim < input_props[0]->getRank(); dim++) {
    if (is_contraction_dim(dim) == false) {
      identical_input_dims.push_back(dim);
    }
  }
  // Get output dims of identical dim groups.
  for (int64_t dim = 0; dim < output_props[0]->getRank(); dim++) {
    // When dim meets`keep_dim = true` and is in conctraction dims set, the dim
    // will be set single split only.
    if (keep_dims && is_contraction_dim(dim)) {
      changed |= trySplitSingleOnly(output_props[0], dim);
    } else {
      identical_output_dims.push_back(dim);
    }
  }
  // add identical dim groups according to non-reduction dims
  assert(identical_input_dims.size() == identical_output_dims.size() &&
         "Number of non-reduction dims should match");
  for (int64_t id = 0; id < identical_input_dims.size(); id++) {
    dim_groups.emplace_back(
        DimGroup::create({{/* is_input = */ true, /* num = */ 0,
                           /* dim = */ identical_input_dims[id]},
                          {/* is_input = */ false, /* num = */ 0,
                           /* dim = */ identical_output_dims[id]}}));
  }
  // add contraction dims as contraction a dim group
  if (input_props[0]->isInitialized() && contraction_dims.size() > 0) {
    std::vector<int64_t> split_dims;
    for (auto contraction_dim : contraction_dims) {
      int64_t dim = (contraction_dim + input_props[0]->getRank()) %
                    input_props[0]->getRank();
      if (input_props[0]->getSplitSpec(dim).getType() !=
          SplitSpec::SplitType::SINGLE) {
        split_dims.push_back(dim);
      }
    }
    if (split_dims.size() > 0) {
      std::vector<DimItem> dim_items;
      for (auto split_dim : split_dims) {
        dim_items.emplace_back(DimItem(/* is_input = */ true, /* num = */ 0,
                                       /* dim = */ split_dim));
      }
      dim_groups.emplace_back(
          DimGroup::createContracting(dim_items,
                                      /*output_num=*/0,
                                      /*contraction_type=*/contraction_type));
    }
  }
  changed |= inferWithDimGroups(input_props, output_props, dim_groups);
  return changed;
}

bool squeezeInfer(const ShardingPropertyRefVec& input_props,
                  const ShardingPropertyRefVec& output_props,
                  const std::vector<int64_t>& squeeze_dims) {
  assert(input_props[0]->getRank() - squeeze_dims.size() ==
             output_props[0]->getRank() &&
         "Incorrect ranks of squeeze input and output tensors");
  bool changed = false;
  std::vector<DimGroup> dim_groups;
  for (int64_t input_dim = 0, output_dim = 0;
       input_dim < input_props[0]->getRank(); input_dim++) {
    bool is_squeeze_dim = false;
    for (auto squeeze_dim : squeeze_dims) {
      if (input_dim == squeeze_dim) {
        is_squeeze_dim = true;
        break;
      }
    }
    if (is_squeeze_dim) {
      // squeeze dims are all ones and only single split is possible
      changed |= input_props[0]->splitSingleAt(input_dim);
      continue;
    }
    dim_groups.emplace_back(DimGroup::create(
        {{/* is_input = */ true, /* num = */ 0, /* dim = */ input_dim},
         {/* is_input = */ false, /* num = */ 0, /* dim = */ output_dim}}));
    output_dim++;
  }
  changed |= inferWithDimGroups(input_props, output_props, dim_groups);
  return changed;
}

bool packInfer(const ShardingPropertyRefVec& input_props,
               const ShardingPropertyRefVec& output_props, int64_t pack_dim) {
  size_t N = std::max(input_props.size(), output_props.size() - 1);
  size_t input_rank = input_props[0]->getRank();
  size_t output_rank = output_props[0]->getRank();
  bool is_pack = input_rank < output_rank ? true : false;
  size_t rank = std::max(input_rank, output_rank);
  std::vector<DimGroup> dim_groups;
  for (size_t dim = 0; dim < rank; dim++) {
    std::vector<DimItem> dim_items;
    for (size_t num = 0; num < N; num++) {
      if (dim != pack_dim) {
        dim_items.emplace_back(
            DimItem(/* is_input = */ is_pack, /* num = */ num,
                    /* dim = */ dim > pack_dim ? dim - 1 : dim));
      }
    }
    // The split rule is not enforced on the axis dim for now
    // TODO(itex): need add split rule on axis dim
    dim_items.emplace_back(
        DimItem(/* is_input = */ !is_pack, /* num = */ 0, /* dim = */ dim));
    dim_groups.emplace_back(DimGroup::create(dim_items));
  }
  bool changed = false;
  changed |= utils::inferWithDimGroups(input_props, output_props, dim_groups);
  return changed;
}

bool isSliceShape(ShardingPropertyRef prop) {
  if (prop->isShapeTensor() && prop->isShapeTensorSingleSplitOnly() &&
      prop->getNumPostOps() != 0) {
    // Check whether `ShapeSlicePostOp` has been added.
    // A maximum of one `ShapeSlicePostOp` exists
    auto&& post_ops = prop->getPostOps();
    for (auto& post_op : post_ops) {
      if (post_op->getName() == typeid(ShapeSlicePostOp).name()) {
        return true;
      } else {
        return false;
      }
    }
  }
  return false;
}

bool trySliceShape(ShardingPropertyRef prop, const std::vector<int64_t>& shape,
                   const std::vector<SplitSpec>& split_specs,
                   const std::vector<int64_t>& split_dims) {
  if (!prop->isShapeTensor() || !prop->isShapeTensorSingleSplitOnly()) {
    return false;
  }
  if (prop->getNumPostOps() != 0) {
    return false;
  }
  assert(split_specs.size() == split_dims.size());
  bool is_concrete = (shape.size() != 0);
  is_concrete &=
      std::all_of(shape.begin(), shape.end(),
                  [&](const int64_t& dim_size) { return dim_size >= 0; });
  // Slice concrete shape with size split
  // split_specs are of size type.
  if (is_concrete && std::any_of(split_specs.begin(), split_specs.end(),
                                 [&](const SplitSpec& split_spec) {
                                   return split_spec.getType() ==
                                          SplitSpec::SplitType::SIZE;
                                 })) {
    if (prop->getShape()[0] != shape.size()) {
      return false;
    }
    assert(std::all_of(split_dims.begin(), split_dims.end(),
                       [&](int64_t dim) { return dim <= shape.size(); }));
    for (size_t i = 0; i < split_specs.size(); i++) {
      auto total_size = split_specs[i].getTotalSize();
      for (auto& size : split_specs[i].getSizes()) {
        if (shape[split_dims[i]] * size % total_size != 0) {
          return false;
        }
      }
    }
    prop->appendPostOp(
        std::make_unique<ShapeSlicePostOp>(split_dims, split_specs));
    return true;
  } else {
    // Slice dynamic shape with ratio split
    // split_specs are of ratio type.
    if (std::any_of(split_specs.begin(), split_specs.end(),
                    [&](const SplitSpec& split_spec) {
                      return split_spec.getType() ==
                             SplitSpec::SplitType::RATIO;
                    })) {
      // TODO(itex): Add check of `shape` with dynamic dims and ratio split
      prop->appendPostOp(
          std::make_unique<ShapeSlicePostOp>(split_dims, split_specs));
      return true;
    }
  }
  return false;
}

}  // namespace utils
}  // namespace as
