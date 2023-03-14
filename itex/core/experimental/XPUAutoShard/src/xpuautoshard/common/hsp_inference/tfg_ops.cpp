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

#include <limits>

#include "xpuautoshard/common/hsp_exception.h"
#include "xpuautoshard/common/hsp_inference/dim_group.h"
#include "xpuautoshard/common/hsp_inference/hsp_inference.h"
#include "xpuautoshard/common/hsp_inference/hsp_inference_factory.h"
#include "xpuautoshard/common/hsp_inference/hsp_inference_utils.h"
#include "xpuautoshard/common/op_desc.h"

namespace as {

namespace {

bool ConvForwardInfer(const OpDesc& op_desc,
                      const as::ShardingPropertyRefVec& input_props,
                      const as::ShardingPropertyRefVec& output_props) {
  int64_t num_spatial_dims = op_desc.getOperand(0).getRank() - 2;
  std::vector<DimGroup> dim_groups_;
  // TODO(itex): Support tensor format (NHWC vs. NCHW, TF vs. PT)
  // Assume TF NHWC now
  // NXI * XIO -> NXO
  // Identical Group for N
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ 0},  // dim_item for input data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ 0},  // dim_item for output data
  }));
  // Contraction Group for I
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ num_spatial_dims + 1},  // dim_item for input data
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ num_spatial_dims},  // dim_item for input weight
  }));
  // Identical Group for O
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ num_spatial_dims + 1},  // dim_item for input weight
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ num_spatial_dims + 1},  // dim_item for output data
  }));

  bool changed = false;
  // TODO(itex): Assume TF NHWC now, weight is KhKwIO handle other formats
  // TODO(itex): Assume KhKw single split for now
  changed |= utils::trySplitSingleOnly(input_props[1], 0);
  changed |= utils::trySplitSingleOnly(input_props[1], 1);
  changed |= utils::inferWithDimGroups(input_props, output_props, dim_groups_);
  return changed;
}

bool ConvBackwardDataInfer(const OpDesc& op_desc,
                           const as::ShardingPropertyRefVec& input_props,
                           const as::ShardingPropertyRefVec& output_props) {
  int64_t num_spatial_dims = op_desc.getOperand(1).getRank() - 2;
  std::vector<DimGroup> dim_groups_;
  // TODO(itex): Support tensor format (NHWC vs. NCHW, TF vs. PT)
  // Assume TF NHWC now, the first operand is input size
  // NXO * XIO -> NXI
  // Identical Group for N
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 2,
       /* dim = */ 0},  // dim_item for input data gradient
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ 0},  // dim_item for output data
  }));
  // Contraction Group for O
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 2,
       /* dim = */ num_spatial_dims + 1},  // dim_item for input data gradient
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ num_spatial_dims + 1},  // dim_item for input weight
  }));
  // Identical Group for I
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ num_spatial_dims},  // dim_item for input weight
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ num_spatial_dims + 1},  // dim_item for output data
  }));
  bool changed =
      utils::inferWithDimGroups(input_props, output_props, dim_groups_);
  // mark the first input as shape tensor
  if (input_props[2]->isInitialized() && !input_props[0]->isShapeTensor()) {
    changed |= input_props[0]->setShapeTensorSingleSplitOnly();
  }
  // slice on batch size
  if (utils::isSliceShape(input_props[0]) == false) {
    changed |= utils::trySliceShape(input_props[0],
                                    op_desc.getOperand(0).getConstVecInt64(),
                                    {input_props[2]->getSplitSpecs()[0]}, {0});
  }
  return changed;
}

bool ConvBackwardWeightInfer(const OpDesc& op_desc,
                             const as::ShardingPropertyRefVec& input_props,
                             const as::ShardingPropertyRefVec& output_props) {
  int64_t num_spatial_dims = op_desc.getOperand(0).getRank() - 2;
  std::vector<DimGroup> dim_groups_;
  // TODO(itex): Support tensor format (NHWC vs. NCHW, TF vs. PT)
  // Assume TF NHWC now
  // NXI * NXO -> XIO
  // Contraction Group for N
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ 0},  // dim_item for input data
      {/* is_input = */ true, /* num = */ 2,
       /* dim = */ 0},  // dim_item for gradient data
  }));
  // Identical Group for I
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ num_spatial_dims + 1},  // dim_item for input data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ num_spatial_dims},  // dim_item for output weight gradient
  }));
  // Identical Group for O
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 2,
       /* dim = */ num_spatial_dims + 1},  // dim_item for gradient data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ num_spatial_dims +
           1},  // dim_item for output weight gradient
  }));
  bool changed = false;
  // TODO(itex): Assume TF NHWC now, weight is KhKwIO handle other formats
  // TODO(itex): Assume KhKw single split for now
  changed |= utils::trySplitSingleOnly(output_props[0], 0);
  changed |= utils::trySplitSingleOnly(output_props[0], 1);
  changed |= utils::inferWithDimGroups(input_props, output_props, dim_groups_);
  return changed;
}

bool BatchNormInfer(const OpDesc& op_desc,
                    const as::ShardingPropertyRefVec& input_props,
                    const as::ShardingPropertyRefVec& output_props) {
  bool changed = false;
  // Assuming element-wise on the data input/output, true for all
  // dims except for the batch dim. Should mark batch dim as
  // normalization instead. But usually we do split at the batch dim,
  // we treat it as element-wise for batch-split to be possible.
  // This impacts the statistcal batch size.
  changed |= utils::elementwiseInfer(input_props, output_props, 1, 1);
  // for simplicity, we force statistics and weights to single split only
  for (int i = 1; i < input_props.size(); i++) {
    changed |= utils::trySplitSingleOnly(input_props[i]);
  }
  // Exclude control edge output.
  for (int i = 1; i < output_props.size() - 1; i++) {
    changed |= utils::trySplitSingleOnly(output_props[i]);
  }
  // We assume `batch_mean` and `batch_variance` are single split only. To
  // achieve higher performance, they service the current device, and no longer
  // need to run allReduce across multiple devices.
  //
  // changed |= utils::inferWithDimGroups(input_props, output_props, {
  //   DimGroup::createContracting(
  //     {
  //       { /* is_input = */ true, /* num = */ 0, /* dim = */ 0 }, //
  //       contraction on batch-dim
  //     },
  //     /*output_num=*/1, // batch_mean
  //     /*contraction_type=*/ContractionType::MEAN
  //   ),
  //   DimGroup::createContracting(
  //     {
  //       { /* is_input = */ true, /* num = */ 0, /* dim = */ 0 }, //
  //       contraction on batch-dim
  //     },
  //     /*output_num=*/2, // batch_variance
  //     /*contraction_type=*/ContractionType::MEAN
  //   )
  // });
  return changed;
}

bool BatchNormGradInfer(const OpDesc& op_desc,
                        const as::ShardingPropertyRefVec& input_props,
                        const as::ShardingPropertyRefVec& output_props) {
  bool changed = false;
  changed |= utils::elementwiseInfer(input_props, output_props, 2, 1);
  // for simplicity, we force statistics and weights to single split
  for (int i = 2; i < input_props.size(); i++) {
    changed |= utils::trySplitSingleOnly(input_props[i]);
  }
  // Exclude control edge output.
  for (int i = 1; i < output_props.size() - 1; i++) {
    changed |= utils::trySplitSingleOnly(output_props[i]);
  }
  changed |= utils::inferWithDimGroups(
      input_props, output_props,
      {DimGroup::createContracting(
           {
               {/* is_input = */ true, /* num = */ 0,
                /* dim = */ 0},  // contraction on batch-dim
           },
           /*output_num=*/1,  // scale
           /*contraction_type=*/ContractionType::SUM),
       DimGroup::createContracting(
           {
               {/* is_input = */ true, /* num = */ 0,
                /* dim = */ 0},  // contraction on batch-dim
           },
           /*output_num=*/2,  // shift
           /*contraction_type=*/ContractionType::SUM)});
  return changed;
}

bool BiasAddInfer(const OpDesc& op_desc,
                  const as::ShardingPropertyRefVec& input_props,
                  const as::ShardingPropertyRefVec& output_props) {
  bool changed = false;
  changed |= utils::trySplitSingleOnly(input_props[1]);
  changed |= utils::elementwiseInfer(input_props, output_props, 1, 1);
  return changed;
}

bool BiasAddGradInfer(const OpDesc& op_desc,
                      const as::ShardingPropertyRefVec& input_props,
                      const as::ShardingPropertyRefVec& output_props) {
  int64_t channel_dim = -1;
  if (input_props[0]->getRank() >= 1 && output_props[0]->getRank() == 1) {
    channel_dim = input_props[0]->getRank() -
                  1;  // TODO(itex): assume TF NHWC, suppor other formats
  }
  if (channel_dim >= 0) {
    std::vector<DimGroup> dim_groups;
    dim_groups.emplace_back(DimGroup::create({
        {/* is_input = */ true, /* num = */ 0,
         /* dim = */ channel_dim},  // dim_item for input data
        {/* is_input = */ false, /* num = */ 0,
         /* dim = */ 0},  // dim_item for output data
    }));
    // Add other dims to the contracting group
    std::vector<DimItem> dim_items;
    for (size_t i = 0; i < input_props.size(); i++) {
      if (i != channel_dim) {
        dim_items.emplace_back(DimItem(
            /* is_input = */ true, /* num = */ 0,
            /* dim = */ i));  // dim_item for input data
      }
    }
    if (dim_items.size() > 0) {
      dim_groups.emplace_back(DimGroup::create(dim_items));
    }
    return utils::inferWithDimGroups(input_props, output_props, dim_groups);
  }
  return false;
}

bool MatMulInfer(const OpDesc& op_desc,
                 const as::ShardingPropertyRefVec& input_props,
                 const as::ShardingPropertyRefVec& output_props) {
  auto op_name = op_desc.getName();
  auto transpose_a = "";
  auto transpose_b = "";
  if (op_name == "tfg.MatMul") {
    transpose_a = "transpose_a";
    transpose_b = "transpose_b";
  } else if (op_name == "tfg.BatchMatMulV2") {
    transpose_a = "adj_x";
    transpose_b = "adj_y";
  }
  int64_t m_dim = op_desc.getAttrBool(transpose_a) ? 1 : 2;
  int64_t a_k_dim = op_desc.getAttrBool(transpose_a) ? 2 : 1;
  int64_t b_k_dim = op_desc.getAttrBool(transpose_b) ? 1 : 2;
  int64_t n_dim = op_desc.getAttrBool(transpose_b) ? 2 : 1;
  auto rank = op_desc.getOperand(0).getRank();
  assert(rank == op_desc.getResult(0).getRank() &&
         "Expect input and output of MatMul has same rank");
  std::vector<DimGroup> dim_groups_;
  // XMK * XKN -> XMN
  // Identical Group for M
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ rank - m_dim},  // dim_item for input data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ rank - 2},  // dim_item for output data
  }));
  // Contraction Group for K
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ rank - a_k_dim},  // dim_item for input data
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ rank - b_k_dim},  // dim_item for input weight
  }));
  // Identical Group for N
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ rank - n_dim},  // dim_item for input weight
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ rank - 1},  // dim_item for output data
  }));
  // Identical Group for X
  for (auto i = 0; i < rank - 2; i++) {
    dim_groups_.emplace_back(DimGroup::create({
        {/* is_input = */ true, /* num = */ 0,
         /* dim = */ i},  // dim_item for input weight
        {/* is_input = */ true, /* num = */ 1,
         /* dim = */ i},  // dim_item for input weight
        {/* is_input = */ false, /* num = */ 0,
         /* dim = */ i},  // dim_item for output data
    }));
  }
  return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
}

bool MaxPoolInfer(const OpDesc& op_desc,
                  const as::ShardingPropertyRefVec& input_props,
                  const as::ShardingPropertyRefVec& output_props) {
  int64_t num_spatial_dims = op_desc.getOperand(0).getRank() - 2;
  std::vector<DimGroup> dim_groups_;
  // TODO(itex): Support tensor format (NHWC vs. NCHW, TF vs. PT)
  // TODO(itex): Support windowed contraction on spatial dims
  // Assume TF NHWC now
  // NXC -> NXC
  // Identical Group for N
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ 0},  // dim_item for input data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ 0},  // dim_item for output data
  }));
  // Identical Group for C
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ num_spatial_dims + 1},  // dim_item for input data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ num_spatial_dims + 1},  // dim_item for output data
  }));
  return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
}

bool MaxPoolGradInfer(const OpDesc& op_desc,
                      const as::ShardingPropertyRefVec& input_props,
                      const as::ShardingPropertyRefVec& output_props) {
  int64_t num_spatial_dims = op_desc.getOperand(0).getRank() - 2;
  std::vector<DimGroup> dim_groups_;
  // TODO(itex): Support tensor format (NHWC vs. NCHW, TF vs. PT)
  // TODO(itex): Support windowed contraction on spatial dims
  // Assume TF NHWC now
  // NXC, NXC, NXC -> NXC
  // Identical Group for N
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ 0},  // dim_item for orig input data
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ 0},  // dim_item for orig output data
      {/* is_input = */ true, /* num = */ 2,
       /* dim = */ 0},  // dim_item for grad output data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ 0},  // dim_item for grad input data
  }));
  // Identical Group for C
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ num_spatial_dims + 1},  // dim_item for orig input data
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ num_spatial_dims + 1},  // dim_item for orig output data
      {/* is_input = */ true, /* num = */ 2,
       /* dim = */ num_spatial_dims + 1},  // dim_item for grad output data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ num_spatial_dims + 1},  // dim_item for grad input data
  }));
  return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
}

bool CastInfer(const OpDesc& op_desc,
               const as::ShardingPropertyRefVec& input_props,
               const as::ShardingPropertyRefVec& output_props) {
  // cast is a unary by nature
  bool changed = utils::elementwiseInfer(input_props, output_props, 1, 1);
  // When there is a cast from a shape tensor to a floating point tensor,
  // we assume the downstream ops would compute some activation values by
  // treating the shape values as activation values. In this case, we should
  // make sure the shape values match those of the unsharded graph, not those
  // shape values of the sharded tensors. We add the post-op to align the casted
  // values across shards.
  // A common example of it is diving the activation by the batch size where
  // the batch size is retrieved with a Shape op, casted to floating point and
  // then dividing the activation data. From the tensorflow graph, it would be
  // something like:
  //   SparseSoftmaxCrossEntropyWithLogits -> Shape -> StridedSlices -> Cast ->
  //   RealDiv/Recirocal+Mul
  // or
  //   Mean -> Size -> Cast -> DivNoNan
  auto add_split_spec_to_weights = [](const SplitSpec& split_spec,
                                      std::vector<float>& weights,
                                      size_t num_splits) {
    float epsilon = std::numeric_limits<float>::lowest();
    if (split_spec.getType() == SplitSpec::SplitType::SIZE) {
      size_t total_size = std::accumulate(split_spec.getSizes().begin(),
                                          split_spec.getSizes().end(), 0);
      for (auto size : split_spec.getSizes()) {
        float size_float = size != 0 ? static_cast<float>(size) : epsilon;
        weights.push_back(static_cast<float>(total_size) / size_float);
      }
    } else if (split_spec.getType() == SplitSpec::SplitType::RATIO) {
      for (auto ratio : split_spec.getRatios()) {
        if (ratio < epsilon) {
          ratio = epsilon;
        }
        weights.push_back(1 / ratio);
      }
    } else {
      for (size_t i = 0; i < num_splits; i++) {
        weights.push_back(1.0f);
      }
    }
  };
  if (input_props[0]->isInitialized() && output_props[0]->isInitialized() &&
      input_props[0]->isShapeTensor() &&
      output_props[0]->isElementFloatingPointType() &&
      output_props[0]->getNumPostOps() == 0) {
    int64_t tensor_rank = input_props[0]->getShape()[0];
    // we only support shape split at the a single dim, check that
    size_t num_shape_split_dims = 0;
    size_t num_splits = 1;
    for (auto&& shape_split_spec : input_props[0]->getShapeSplitSpecs()) {
      if (shape_split_spec.getType() == SplitSpec::SplitType::SIZE ||
          shape_split_spec.getType() == SplitSpec::SplitType::RATIO) {
        num_splits *= shape_split_spec.size();
        num_shape_split_dims++;
      }
    }
    if (num_shape_split_dims > 1) {
      throw HspUnimplementedException(
          "Only support number of shape split dims at most one but got "
          "multiple",
          *input_props[0]);
    } else if (num_shape_split_dims == 0) {
      return changed;
    }
    std::vector<float> weights;
    for (auto&& shape_split_spec : input_props[0]->getShapeSplitSpecs()) {
      add_split_spec_to_weights(shape_split_spec, weights, num_splits);
    }
    output_props[0]->appendPostOp(std::make_unique<WeightedScalePostOp>(
        weights,
        std::vector<int64_t>(1, tensor_rank)));  // tensor_rank x num_splits
    changed = true;
  } else if (input_props[0]->isInitialized() &&
             output_props[0]->isInitialized() &&
             input_props[0]->isSizeTensor() &&
             output_props[0]->isElementFloatingPointType() &&
             input_props[0]->getSizeSplitSpec().size() > 1 &&
             output_props[0]->getNumPostOps() == 0) {
    std::vector<float> weights;
    add_split_spec_to_weights(
        input_props[0]->getSizeSplitSpec(), weights,
        /*num_splits*/ input_props[0]->getSizeSplitSpec().size());
    output_props[0]->appendPostOp(
        std::make_unique<WeightedScalePostOp>(weights));
    changed = true;
  }
  return changed;
}

bool SoftmaxInfer(const OpDesc& op_desc,
                  const as::ShardingPropertyRefVec& input_props,
                  const as::ShardingPropertyRefVec& output_props) {
  std::vector<DimGroup> dim_groups_;
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ 0},  // dim_item for input data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ 0},  // dim_item for output data
  }));
  return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
}

bool SoftmaxCrossEntropyInfer(const OpDesc& op_desc,
                              const as::ShardingPropertyRefVec& input_props,
                              const as::ShardingPropertyRefVec& output_props) {
  std::vector<DimGroup> dim_groups_;
  // TODO(itex): only map the batch dim now.
  // Need to handle the normalization dim.
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ 0},  // dim_item for input data
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ 0},  // dim_item for label
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ 0},  // dim_item for loss
      {/* is_input = */ false, /* num = */ 1,
       /* dim = */ 0},  // dim_item for grad output
  }));
  return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
}

bool UnaryInfer(const OpDesc& op_desc,
                const as::ShardingPropertyRefVec& input_props,
                const as::ShardingPropertyRefVec& output_props) {
  return utils::elementwiseInfer(input_props, output_props, 1, 1);
}

bool BinaryInfer(const OpDesc& op_desc,
                 const as::ShardingPropertyRefVec& input_props,
                 const as::ShardingPropertyRefVec& output_props) {
  return utils::elementwiseInfer(input_props, output_props, 2, 1);
}

bool NnaryInfer(const OpDesc& op_desc,
                const as::ShardingPropertyRefVec& input_props,
                const as::ShardingPropertyRefVec& output_props) {
  if (op_desc.getName() == "tfg.SelectV2") {
    return utils::elementwiseInfer(input_props, output_props, 3, 1);
  }
  auto N = op_desc.getAttrInt64("N");
  if (op_desc.getName() == "tfg.Concat") {
    std::vector<int64_t> exclude_dims;
    exclude_dims.push_back(op_desc.getAttrInt64("concat_dim"));
    return utils::elementwiseInfer(input_props, output_props, N, 1, false,
                                   exclude_dims);
  } else {
    return utils::elementwiseInfer(input_props, output_props, N, 1);
  }
}

bool IdentityNInfer(const OpDesc& op_desc,
                    const as::ShardingPropertyRefVec& input_props,
                    const as::ShardingPropertyRefVec& output_props) {
  assert(input_props.size() == output_props.size() - 1 &&
         "Expect input elementwise output");
  bool changed = false;
  for (auto i = 0; i < input_props.size(); i++) {
    ShardingPropertyRefVec input_tmp = {input_props[i]};
    ShardingPropertyRefVec output_tmp = {output_props[i]};
    changed |= utils::elementwiseInfer(input_tmp, output_tmp, 1, 1);
  }
  return changed;
}

bool ReductionInfer(const OpDesc& op_desc,
                    const as::ShardingPropertyRefVec& input_props,
                    const as::ShardingPropertyRefVec& output_props) {
  std::vector<int64_t> contraction_dims;
  bool keep_dims = false;
  if (op_desc.getName() == "tfg.Mean" || op_desc.getName() == "tfg.Sum" ||
      op_desc.getName() == "tfg.ArgMax" || op_desc.getName() == "tfg.Max" ||
      op_desc.getName() == "tfg.Min" || op_desc.getName() == "tfg.Prod") {
    if (op_desc.hasAttr("keep_dims")) {
      keep_dims = op_desc.getAttrBool("keep_dims");
    }
    contraction_dims = op_desc.getOperand(1).getConstVecInt64();
  } else {
    assert(op_desc.getName() == "tfg.L2Loss" && "Expect op name tfg.L2Loss");
    // TODO(itex): it is half of L2, should pass 1/2 too.
    int64_t rank = op_desc.getOperand(0).getRank();
    for (int64_t dim = 0; dim < rank; dim++) {
      contraction_dims.push_back(dim);
    }
  }
  ContractionType contraction_type;
  if (op_desc.getName() == "tfg.Mean") {
    contraction_type = ContractionType::MEAN;
  } else if (op_desc.getName() == "tfg.Sum") {
    contraction_type = ContractionType::SUM;
  } else if (op_desc.getName() == "tfg.ArgMax") {
    contraction_type = ContractionType::ARGMAX;
  } else if (op_desc.getName() == "tfg.Max") {
    contraction_type = ContractionType::Max;
  } else if (op_desc.getName() == "tfg.Min") {
    contraction_type = ContractionType::Min;
  } else if (op_desc.getName() == "tfg.Prod") {
    contraction_type = ContractionType::Prod;
  } else {
    assert(op_desc.getName() == "tfg.L2Loss" && "Expect op name tfg.L2Loss");
    contraction_type = ContractionType::L2;
  }
  return utils::contractionInfer(input_props, output_props, contraction_dims,
                                 contraction_type, keep_dims);
}

bool ExpandDimsInfer(const OpDesc& op_desc,
                     const as::ShardingPropertyRefVec& input_props,
                     const as::ShardingPropertyRefVec& output_props) {
  auto&& axis_vec = op_desc.getOperand(1).getConstVecInt64();
  if (axis_vec.size() == 0) {
    // TODO(itex): Assum default inference when axis is variable
    return utils::inferDefault(input_props, output_props);
  }
  auto rank = op_desc.getOperand(0).getRank();
  auto axis = (axis_vec[0] + rank + 1) % (rank + 1);
  std::vector<DimGroup> dim_groups;
  for (auto dim = 0; dim < rank; dim++) {
    auto offset = 0;
    if (dim >= axis) {
      offset = 1;
    }
    dim_groups.emplace_back(DimGroup::create({
        {/* is_input = */ true, /* num = */ 0,
         /* dim = */ dim},  // dim_item for input data
        {/* is_input = */ false, /* num = */ 0,
         /* dim = */ dim + offset},  // dim_item for output data
    }));
  }
  output_props[0]->splitSingleAt(axis);
  return utils::inferWithDimGroups(input_props, output_props, dim_groups);
}

bool ShapeInfer(const OpDesc& op_desc,
                const as::ShardingPropertyRefVec& input_props,
                const as::ShardingPropertyRefVec& output_props) {
  bool changed = false;
  // Get shape on one or multiple tensors
  for (size_t i = 0; i < output_props.size(); i++) {
    changed |= utils::trySplitSingleOnly(output_props[i]);
    if (output_props[i]->isElementIntegerType() &&
        input_props[i]->isInitialized()) {
      // mark the output prop as a shape tensor
      changed |=
          output_props[i]->setShapeTensor(input_props[i]->getSplitSpecs());
    }
  }
  return changed;
}

bool SizeInfer(const OpDesc& op_desc,
               const as::ShardingPropertyRefVec& input_props,
               const as::ShardingPropertyRefVec& output_props) {
  bool changed = false;
  changed |= utils::trySplitSingleOnly(output_props[0]);
  changed |= output_props[0]->setSizeTensor(*input_props[0]);
  return changed;
}

bool TransposeInfer(const OpDesc& op_desc,
                    const as::ShardingPropertyRefVec& input_props,
                    const as::ShardingPropertyRefVec& output_props) {
  std::vector<DimGroup> dim_groups;
  auto perm = op_desc.getOperand(1).getConstVecInt64();
  auto rank = op_desc.getOperand(0).getRank();
  assert(rank == perm.size());
  assert(rank == op_desc.getResult(0).getRank());
  for (auto dim = 0; dim < rank; dim++) {
    dim_groups.emplace_back(DimGroup::create({
        {/* is_input = */ true, /* num = */ 0,
         /* dim = */ perm[dim]},  // dim_item for input data
        {/* is_input = */ false, /* num = */ 0,
         /* dim = */ dim},  // dim_item for output data
    }));
  }
  return utils::inferWithDimGroups(input_props, output_props, dim_groups);
}

bool ReshapeInfer(const OpDesc& op_desc,
                  const as::ShardingPropertyRefVec& input_props,
                  const as::ShardingPropertyRefVec& output_props) {
  auto get_reshape_vec = [&]() {
    auto&& reshape_vec = op_desc.getOperand(1).getConstVecInt64();
    if (reshape_vec.size() == 0) {
      // shape value doesn't cone from tfg.Const OP.
      // TODO(itex): abtain shape value from other OP. The shape of tfg.
      // Reshape result equal the shape input when concrete input.
      return reshape_vec;
    }
    assert(reshape_vec.size() == op_desc.getResult(0).getRank());
    // Infer shape value from output when shape const value is [-1].
    // The specific value will only be infered at runtime, and output shape
    // of the OP must be [1]. So, take the output shape as the Const OP value.
    if (reshape_vec.size() == 1 && reshape_vec[0] == -1) {
      reshape_vec[0] = op_desc.getResult(0).getDimSize(0);
    }
    return reshape_vec;
  };
  bool changed = false;
  as::ShardingPropertyRef input_reshape = nullptr;
  as::ShardingPropertyRef input_data = nullptr;
  if (op_desc.getName() == "tfg.StridedSliceGrad") {
    input_data = input_props[4];
    input_reshape = input_props[0];
  } else if (op_desc.getName() == "tfg.Reshape" ||
             op_desc.getName() == "tfg.BroadcastTo") {
    input_data = input_props[0];
    input_reshape = input_props[1];
  } else if (op_desc.getName() == "tfg.InvertPermutation") {
    // TODO(itex): need to process input_reshape
    input_data = input_props[0];
  }
  if (input_reshape && input_reshape->isInitialized() &&
      !input_reshape->isShapeTensor()) {
    changed |= input_reshape->setShapeTensorSingleSplitOnly();
  }
  // For simplicity, we assume the following conditions:
  // 1. input 0 has only 1 multi-split dim with split by size.
  // 2. leading dims before the split dim should be concrete.
  // 3. leading dims before the split dim should match the reshape.
  if (input_data && input_data->isInitialized() &&
      !input_data->isSplitSingleOnly() &&
      input_data->numMultiSplitDims() == 1) {
    for (int64_t dim = 0; dim < input_data->getRank(); dim++) {
      auto&& split_spec = input_data->getSplitSpec(dim);
      assert(split_spec.isInitialized());
      if (split_spec.getType() != SplitSpec::SplitType::SINGLE) {
        auto&& reshape_vec = get_reshape_vec();
        // leading dims should match and corresponding reshape size should be
        // concrete
        bool leading_dims_match = true;
        for (size_t leading_dim = 0; leading_dim < dim; leading_dim++) {
          if (reshape_vec[leading_dim] < 0 ||
              op_desc.getOperand(0).getDimSize(leading_dim) !=
                  reshape_vec[leading_dim]) {
            leading_dims_match = false;
            break;
          }
        }
        bool slice_shape_flag = utils::isSliceShape(input_reshape);
        if (slice_shape_flag == false) {
          slice_shape_flag |= utils::trySliceShape(input_reshape, reshape_vec,
                                                   {split_spec}, {dim});
        }
        if (leading_dims_match && slice_shape_flag) {
          if (!output_props[0]->isInitialized()) {
            for (size_t output_dim = 0; output_dim < output_props[0]->getRank();
                 output_dim++) {
              if (output_dim == dim) {
                output_props[0]->splitAt(output_dim, split_spec);
              } else {
                output_props[0]->splitSingleAt(output_dim);
              }
            }
            changed = true;
          }
        }
        break;
      }
    }
  } else if (output_props[0]->isInitialized() &&
             !output_props[0]->isSplitSingleOnly() &&
             output_props[0]->numMultiSplitDims() == 1) {
    for (int64_t dim = 0; dim < output_props[0]->getRank(); dim++) {
      auto&& split_spec = output_props[0]->getSplitSpec(dim);
      assert(split_spec.isInitialized());
      if (split_spec.getType() != SplitSpec::SplitType::SINGLE) {
        auto&& reshape_vec = get_reshape_vec();
        // leading dims should match and corresponding reshape size should be
        // concrete
        bool leading_dims_match = true;
        for (size_t leading_dim = 0; leading_dim < dim; leading_dim++) {
          if (reshape_vec[leading_dim] < 0 ||
              op_desc.getOperand(0).getDimSize(leading_dim) !=
                  reshape_vec[leading_dim]) {
            leading_dims_match = false;
            break;
          }
        }
        bool slice_shape_flag = utils::isSliceShape(input_reshape);
        if (slice_shape_flag == false) {
          slice_shape_flag |= utils::trySliceShape(input_reshape, reshape_vec,
                                                   {split_spec}, {dim});
        }
        if (leading_dims_match && slice_shape_flag) {
          if (!input_data->isInitialized()) {
            for (size_t input_dim = 0; input_dim < input_data->getRank();
                 input_dim++) {
              if (input_dim == dim && (!input_data->isInitialized() ||
                                       input_data->getSplitSpec(input_dim)
                                               .isSizeSplitConvertEquivalently(
                                                   split_spec) == false)) {
                changed |= input_data->splitAt(input_dim, split_spec);
              }
              if (input_dim != dim &&
                  (input_data->isInitialized() ||
                   !input_data->isSingleSplitAt(input_dim))) {
                changed |= input_data->splitSingleAt(input_dim);
              }
            }
          }
        }
        break;
      }
    }
  } else {
    changed |= utils::inferDefault(input_props, output_props);
  }
  return changed;
}

bool SqueezeInfer(const OpDesc& op_desc,
                  const as::ShardingPropertyRefVec& input_props,
                  const as::ShardingPropertyRefVec& output_props) {
  std::string axis_attr_name;
  if (op_desc.hasAttr("axis")) {
    axis_attr_name = "axis";
  } else if (op_desc.hasAttr("squeeze_dims")) {
    axis_attr_name = "squeeze_dims";
  }
  auto&& operand_desc = op_desc.getOperand(0);
  std::vector<int64_t> squeeze_dims;
  if (!axis_attr_name.empty()) {
    squeeze_dims = op_desc.getAttrVecInt64(axis_attr_name);
    // make sure all dims are positive
    for (size_t i = 0; i < squeeze_dims.size(); i++) {
      squeeze_dims[i] =
          (squeeze_dims[i] + operand_desc.getRank()) % operand_desc.getRank();
    }
  } else {
    for (int64_t idx = 0; idx < operand_desc.getRank(); idx++) {
      if (!operand_desc.isDynamicDim(idx) &&
          operand_desc.getDimSize(idx) == 1) {
        squeeze_dims.push_back(idx);
      }
    }
  }
  return utils::squeezeInfer(input_props, output_props, squeeze_dims);
}

bool PadInfer(const OpDesc& op_desc,
              const as::ShardingPropertyRefVec& input_props,
              const as::ShardingPropertyRefVec& output_props) {
  // TODO(itex): add padding params
  std::vector<std::pair<int64_t, int64_t>> padding_params;
  for (int64_t dim = 0; dim < op_desc.getOperand(0).getRank(); dim++) {
    padding_params.emplace_back(std::make_pair(0, 0));
  }
  if (padding_params.size() > 0) {
    assert(padding_params.size() == input_props[0]->getRank() &&
           "Size of padding parameters should match input rank");
    std::vector<DimGroup> dim_groups;
    // zero padding as identical group
    for (int64_t dim = 0; dim < padding_params.size(); dim++) {
      if (padding_params[dim].first == 0 && padding_params[dim].second == 0) {
        dim_groups.emplace_back(DimGroup::create({
            {/* is_input = */ true, /* num = */ 0,
             /* dim = */ dim},  // dim_item for input data
            {/* is_input = */ false, /* num = */ 0,
             /* dim = */ dim},  // dim_item for output data
        }));
      }
    }
    return utils::inferWithDimGroups(input_props, output_props, dim_groups);
  }
  return utils::inferDefault(input_props, output_props);
}

bool StridedSliceInfer(const OpDesc& op_desc,
                       const as::ShardingPropertyRefVec& input_props,
                       const as::ShardingPropertyRefVec& output_props) {
  // Assume only support split on batch dim.
  // TODO(itex): support split on multi-dims.
  if (op_desc.getName() == "tfg.StridedSlice") {
    // Check ellipsis_mask and new_axis_mask are all zero, and shrink_axis_mask
    // is 1
    if (op_desc.getAttrInt64("ellipsis_mask") != 0 ||
        op_desc.getAttrInt64("new_axis_mask") != 0) {
      return false;
    }
  }
  // Get the begin, end, and stride on the batch dim.
  int64_t begin_batch_dim = 0;
  int64_t end_batch_dim = 0;
  int64_t stride_batch_dim = 1;
  auto begin_param = op_desc.getOperand(1).getConstVecInt64();
  auto end_param = op_desc.getOperand(2).getConstVecInt64();
  std::vector<int64_t> strides;
  begin_batch_dim = begin_param[0];
  end_batch_dim = end_param[0];
  if (op_desc.getNumOperands() > 3) {
    strides = op_desc.getOperand(3).getConstVecInt64();
  }
  stride_batch_dim = (strides.size() == 0 ? 1 : strides[0]);
  // If (begin_mask & (1<<0)) == 1, set the minimum value of the interval on the
  // batch dim.
  if (op_desc.getAttrInt64("begin_mask") & 1) {
    begin_batch_dim = 0;
  }
  // If (end_mask & (1<<0)) == 1, set the maximum value of the interval on the
  // batch dim.
  if (op_desc.getAttrInt64("end_mask") & 1) {
    end_batch_dim = op_desc.getOperand(0).getDimSize(0) - 1;
  }
  // TODO(itex): Support stride_batch_dim < 0,
  // begin_batch_dim and end_batch_dim need become negative.
  auto&& operand_desc = op_desc.getOperand(0);
  if (operand_desc.getRank() == 1 &&
      std::abs(end_batch_dim - begin_batch_dim) <= stride_batch_dim) {
    // Pass the sliced dim as params for shape tensor that extracts a single dim
    // with dim shrinking. This is used to compute the batch averaging of loss.
    auto selected_dim = begin_batch_dim;
    if (input_props[0]->isShapeTensor()) {
      return output_props[0]->setSizeTensor(
          input_props[0]->getShapeSplitSpecs()[selected_dim]);
    }
    return false;
  }
  // Check that no slicing is performed on the batch dim.
  if (begin_batch_dim != 0 ||
      (end_batch_dim != 0 ||
       end_batch_dim != op_desc.getOperand(0).getDimSize(0)) -
          1 ||
      stride_batch_dim != 1) {
    return false;
  }
  auto rank = op_desc.getOperand(0).getRank();
  std::vector<DimGroup> dim_groups_;
  for (auto dim = 0; dim < rank; dim++) {
    dim_groups_.emplace_back(DimGroup::create({
        {/* is_input = */ true, /* num = */ 0,
         /* dim = */ dim},  // dim_item for input data
        {/* is_input = */ false, /* num = */ 0,
         /* dim = */ dim},  // dim_item for output data
    }));
  }
  return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
}

bool TileInfer(const OpDesc& op_desc,
               const as::ShardingPropertyRefVec& input_props,
               const as::ShardingPropertyRefVec& output_props) {
  if (!input_props[0]->isInitialized() &&
      output_props[0]->isSplitSingleOnly()) {
    // Backward single split only from output to the input not vice versa since
    // it is possible that a single split only input becomes multi-split after
    // Tile op
    return utils::trySplitSingleOnly(input_props[0]);
  } else if (
      // If single split input becomes multi-split output, we insert a split
      // post op
      input_props[0]->isInitialized() && output_props[0]->isInitialized() &&
      input_props[0]->isSplitSingleOnly() &&
      !output_props[0]->isSplitSingleOnly() &&
      output_props[0]->getNumPostOps() == 0) {
    for (int64_t dim = 0; dim < output_props[0]->getRank(); dim++) {
      auto&& split_spec = output_props[0]->getSplitSpecs()[dim];
      if (split_spec.getType() != as::SplitSpec::SplitType::SINGLE) {
        output_props[0]->appendPostOp(std::make_unique<as::SlicePostOp>());
      }
    }
    return true;
  } else if (
      // If multi-split input, identical group for data
      input_props[0]->isInitialized() && !input_props[0]->isSplitSingleOnly() &&
      input_props[0]->numMultiSplitDims() == 1 &&
      !output_props[0]->isInitialized()) {
    auto rank = op_desc.getOperand(0).getRank();
    std::vector<DimGroup> dim_groups_;
    for (auto i = 0; i < rank; i++) {
      dim_groups_.emplace_back(DimGroup::create({
          {/* is_input = */ true, /* num = */ 0,
           /* dim = */ i},  // dim_item for input data
          {/* is_input = */ false, /* num = */ 0,
           /* dim = */ i},  // dim_item for output data
      }));
    }
    return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
  }
  return false;
}

bool PackInfer(const OpDesc& op_desc,
               const as::ShardingPropertyRefVec& input_props,
               const as::ShardingPropertyRefVec& output_props) {
  // Pack:   [N * ABC] (axis = 1) ->  ANBC
  // Unpack: ANBC (axis = 1) ->  [N * ABC]
  int64_t pack_dim = op_desc.getAttrInt64("axis");
  return utils::packInfer(input_props, output_props, pack_dim);
}

bool UniqueInfer(const OpDesc& op_desc,
                 const as::ShardingPropertyRefVec& input_props,
                 const as::ShardingPropertyRefVec& output_props) {
  // A -> B,C
  // TODO(itex): If input is shard, add post op to unshard.
  // TODO(itex): Offset needs to be added to maintain the unshard output.
  // Assume single split for now
  bool changed = false;
  changed |= utils::trySplitSingleOnly(input_props[0], 0);
  changed |= utils::trySplitSingleOnly(output_props[0], 0);
  changed |= utils::trySplitSingleOnly(output_props[1], 0);
  return changed;
}

bool WhereInfer(const OpDesc& op_desc,
                const as::ShardingPropertyRefVec& input_props,
                const as::ShardingPropertyRefVec& output_props) {
  // Where op returns the coordinates of true elements in tensor input.
  // TODO(itex): if split, it may need add offset and post op.
  // Assume single split now.
  bool changed = false;
  changed |= utils::trySplitSingleOnly(input_props[0]);
  changed |= utils::trySplitSingleOnly(output_props[0], 0);
  changed |= utils::trySplitSingleOnly(output_props[1], 1);
  return changed;
}

bool EinsumInfer(const OpDesc& op_desc,
                 const as::ShardingPropertyRefVec& input_props,
                 const as::ShardingPropertyRefVec& output_props) {
  // BIJ, BJK, "bij,bjk->bik" -> BIK
  std::string equation = op_desc.getAttrString("equation");
  auto equal_sign_pos = equation.find("->");
  assert(equal_sign_pos != std::string::npos &&
         "Expect equation attribute of `Einsum` has `->`");
  auto operation_desc_str = equation.substr(0, equal_sign_pos);
  auto result_desc_str = equation.substr(equal_sign_pos + 2, equation.size());
  // `result_desc_str` can be empty.
  assert(operation_desc_str != "" &&
         "Expect equation attribute of `Einsum` has describtion of operand");
  std::set<char> all_symbols;
  auto handle_desc_by_separator =
      [&all_symbols](std::string describtion,
                     std::string separator =
                         ",") -> std::vector<std::unordered_map<char, size_t>> {
    std::vector<std::unordered_map<char, size_t>> symbols_mp_vec;
    size_t separator_pos = 0;
    while (true) {
      separator_pos = std::min(describtion.size(), describtion.find(separator));
      std::unordered_map<char, size_t> symbols_mp;
      for (size_t dim = 0; dim < separator_pos; dim++) {
        symbols_mp[describtion[dim]] = dim;
        all_symbols.insert(describtion[dim]);
      }
      symbols_mp_vec.push_back(symbols_mp);
      if (separator_pos == describtion.size()) {
        break;
      }
      describtion = describtion.substr(separator_pos + 1, describtion.size());
    }
    return symbols_mp_vec;
  };
  std::vector<std::unordered_map<char, size_t>> operand_symbols_vec,
      result_symbols_vec;
  operand_symbols_vec = handle_desc_by_separator(operation_desc_str);
  result_symbols_vec = handle_desc_by_separator(result_desc_str);
  assert(operand_symbols_vec.size() == op_desc.getAttrInt64("N") &&
         "Expect equation attribute of `Einsum` has `N` operation describtion");
  std::vector<DimGroup> dim_groups_;
  for (auto symbol : all_symbols) {
    std::vector<DimItem> dim_items;
    for (size_t operand_id = 0; operand_id < operand_symbols_vec.size();
         operand_id++) {
      std::unordered_map<char, size_t> operand_symbols_mp =
          operand_symbols_vec[operand_id];
      if (operand_symbols_mp.count(symbol) != 0) {
        dim_items.push_back(DimItem(
            /* is_input = */ true, /* num = */ operand_id,
            /* dim = */ operand_symbols_mp[symbol]));
      }
    }
    for (size_t result_id = 0; result_id < result_symbols_vec.size();
         result_id++) {
      std::unordered_map<char, size_t> result_symbols_mp =
          result_symbols_vec[result_id];
      if (result_symbols_mp.count(symbol) != 0) {
        dim_items.push_back(DimItem(
            /* is_input = */ false, /* num = */ result_id,
            /* dim = */ result_symbols_mp[symbol]));  // dim_item for input data
      }
    }
    if (dim_items.size() > 0) {
      dim_groups_.emplace_back(DimGroup::create(dim_items));
    }
  }
  return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
}

bool OneHotInfer(const OpDesc& op_desc,
                 const as::ShardingPropertyRefVec& input_props,
                 const as::ShardingPropertyRefVec& output_props) {
  auto axis = op_desc.getAttrInt64("axis");
  int64_t rank = op_desc.getOperand(0).getRank();
  axis = axis == -1 ? rank : axis;
  bool changed = false;
  // Assume single split on new added dim
  changed |= utils::trySplitSingleOnly(output_props[0], axis);
  std::vector<DimGroup> dim_groups_;
  // Identical group for other dims
  for (auto dim = 0; dim < rank; dim++) {
    dim_groups_.emplace_back(DimGroup::create({
        {/* is_input = */ true, /* num = */ 0,
         /* dim = */ dim},  // dim_item for input data
        {/* is_input = */ false, /* num = */ 0,
         /* dim = */ dim >= axis ? dim + 1 : dim},  // dim_item for output data
    }));
  }
  changed |= utils::inferWithDimGroups(input_props, output_props, dim_groups_);
  return changed;
}

bool BroadcastGradientArgsInfer(
    const OpDesc& op_desc, const as::ShardingPropertyRefVec& input_props,
    const as::ShardingPropertyRefVec& output_props) {
  // shape1, shape2 -> pos_vec1, pos_vec2
  auto operand_num = op_desc.getNumOperands();
  bool changed = false;
  for (auto id = 0; id < operand_num; id++) {
    if (input_props[id]->isInitialized() && !input_props[id]->isShapeTensor()) {
      changed |= input_props[id]->setShapeTensorSingleSplitOnly();
    }
    // Assume single split on output
    changed |= utils::trySplitSingleOnly(output_props[id], 0);
  }
  return changed;
}

bool FillInfer(const OpDesc& op_desc,
               const as::ShardingPropertyRefVec& input_props,
               const as::ShardingPropertyRefVec& output_props) {
  // shape_tensor[A,B,C] -> ABC
  bool changed = false;
  // Mark the input as shape tensor.
  if (input_props[0]->isInitialized() && !input_props[0]->isShapeTensor()) {
    changed |= input_props[0]->setShapeTensorSingleSplitOnly();
  }
  // If batch split on output, mark input as ShapeSlicePostOp.
  // Aassume only split on batch dim.
  // TODO(itex): split on other dims.
  if (output_props[0]->isInitialized() &&
      !output_props[0]->isSplitSingleOnly()) {
    auto&& shape_vec = op_desc.getOperand(0).getConstVecInt64();
    assert(shape_vec.size() == op_desc.getResult(0).getRank());
    if (utils::isSliceShape(input_props[0]) == false) {
      changed |= utils::trySliceShape(input_props[0], shape_vec,
                                      {output_props[0]->getSplitSpec(0)}, {0});
    }
  }
  // Output will not be handled by User, so must processing it.
  return changed;
}

bool GatherV2Infer(const OpDesc& op_desc,
                   const as::ShardingPropertyRefVec& input_props,
                   const as::ShardingPropertyRefVec& output_props) {
  // ABCD  XYZ  axis = 1 -> AXYZCD
  auto rank_0 = op_desc.getOperand(0).getRank();
  auto rank_1 = op_desc.getOperand(1).getRank();
  auto axis = (op_desc.getOperand(2).getConstVecInt64()[0] + rank_0) % rank_0;
  std::vector<DimGroup> dim_groups_;
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 0,
       /* dim = */ axis},  // dim_item for input data
  }));
  for (auto dim = 0; dim < rank_0 + rank_1 - 1; dim++) {
    auto input_num = 0;
    auto input_dim = dim;
    if (dim >= axis && dim < axis + rank_1) {
      input_num = 1;
      input_num = dim - axis;
    } else if (dim >= axis + rank_1) {
      input_dim = dim - rank_1 + 1;
    }
    dim_groups_.emplace_back(DimGroup::create({
        {/* is_input = */ true, /* num = */ input_num,
         /* dim = */ input_dim},  // dim_item for input data
        {/* is_input = */ false, /* num = */ 0,
         /* dim = */ dim},  // dim_item for output data
    }));
  }
  return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
}

bool ResourceGatherInfer(const OpDesc& op_desc,
                         const as::ShardingPropertyRefVec& input_props,
                         const as::ShardingPropertyRefVec& output_props) {
  // resource, indices(0-D or 1-D) -> output(1-D)
  auto rank = op_desc.getOperand(1).getRank();
  if (rank == 0) {
    return utils::trySplitSingleOnly(output_props[0], 0);
  }
  std::vector<DimGroup> dim_groups_;
  dim_groups_.emplace_back(DimGroup::create({
      {/* is_input = */ true, /* num = */ 1,
       /* dim = */ 0},  // dim_item for input data
      {/* is_input = */ false, /* num = */ 0,
       /* dim = */ 0},  // dim_item for output data
  }));
  return utils::inferWithDimGroups(input_props, output_props, dim_groups_);
}

bool QuantizedMaxPoolInfer(const OpDesc& op_desc,
                           const as::ShardingPropertyRefVec& input_props,
                           const as::ShardingPropertyRefVec& output_props) {
  bool changed = MaxPoolInfer(op_desc, input_props, output_props);
  changed |= utils::trySplitSingleOnly(input_props[1]);
  changed |= utils::trySplitSingleOnly(input_props[2]);
  changed |= utils::trySplitSingleOnly(output_props[1]);
  changed |= utils::trySplitSingleOnly(output_props[2]);
  return changed;
}

bool QuantizedConv2DInfer(const OpDesc& op_desc,
                          const as::ShardingPropertyRefVec& input_props,
                          const as::ShardingPropertyRefVec& output_props) {
  bool changed = false;
  // Quantize inference
  changed |=
      utils::trySplitSingleOnly(input_props[3]);  // single split on min_input
  changed |=
      utils::trySplitSingleOnly(input_props[4]);  // single split on max_input

  // Conv2D inference
  changed |= ConvForwardInfer(op_desc, input_props, output_props);
  // TODO(itex): Assume TF NHWC now, weight is KhKwIO handle other formats
  // TODO(itex): Assume KhKw single split for now
  changed |= utils::trySplitSingleOnly(
      input_props[1], 0);  // single split on the 0th dim of filter
  changed |= utils::trySplitSingleOnly(
      input_props[1], 1);  // single split on the 1th dim of filter
  changed |=
      utils::trySplitSingleOnly(input_props[5]);  // single split on min_filter
  changed |=
      utils::trySplitSingleOnly(input_props[6]);  // single split on max_filter
  changed |= utils::trySplitSingleOnly(output_props[1]);
  changed |= utils::trySplitSingleOnly(output_props[2]);

  // BiasAdd inference with elementwise
  changed |= utils::trySplitSingleOnly(input_props[2]);  // single split on bias
  ShardingPropertyRefVec bias_input_prop = {input_props[0], input_props[2]};
  ShardingPropertyRefVec bias_output_prop = {output_props[0]};
  changed |= utils::elementwiseInfer(bias_input_prop, bias_output_prop, 1,
                                     1);  // elementwise infer bias

  // Relu inference
  if (op_desc.getName().find("Relu") != std::string::npos) {
    changed |= utils::elementwiseInfer(input_props, output_props, 1, 1);
  }

  // Sum inference
  if (op_desc.getName().find("Sum") != std::string::npos) {
    ShardingPropertyRefVec summand_input_prop = {input_props[0],
                                                 input_props[9]};
    ShardingPropertyRefVec summand_output_prop = {output_props[0]};
    changed |= utils::elementwiseInfer(summand_input_prop, summand_output_prop,
                                       1, 1);  // elementwise infer summand
    changed |= utils::trySplitSingleOnly(
        input_props[10]);  // single split on min_summand
    changed |= utils::trySplitSingleOnly(
        input_props[11]);  // single split on max_summand
  }

  // Requantize inference
  if (op_desc.getName().find("Requantize") != std::string::npos) {
    changed |= utils::trySplitSingleOnly(
        input_props[7]);  // single split on min_freezed_output
    changed |= utils::trySplitSingleOnly(
        input_props[8]);  // single split on max_freezed_output
    changed |= utils::elementwiseInfer(input_props, output_props, 1, 1);
  }

  return changed;
}

}  // anonymous namespace

DEFINE_AND_REGISTER_HSP_INFERENCE(ConvForward, "tfg.Conv2D");
DEFINE_AND_REGISTER_HSP_INFERENCE(ConvBackwardData, "tfg.Conv2DBackpropInput");
DEFINE_AND_REGISTER_HSP_INFERENCE(ConvBackwardWeight,
                                  "tfg.Conv2DBackpropFilter");
DEFINE_AND_REGISTER_HSP_INFERENCE(BatchNorm, "tfg.FusedBatchNormV3")
+ "tfg._FusedBatchNormEx" + "tfg.LayerNorm";
DEFINE_AND_REGISTER_HSP_INFERENCE(BatchNormGrad, "tfg.FusedBatchNormGradV3")
+ "tfg.LayerNormGrad";
DEFINE_AND_REGISTER_HSP_INFERENCE(BiasAdd, "tfg.BiasAdd");
DEFINE_AND_REGISTER_HSP_INFERENCE(BiasAddGrad, "tfg.BiasAddGrad");
DEFINE_AND_REGISTER_HSP_INFERENCE(MatMul, "tfg.MatMul")
+ "tfg.BatchMatMulV2";
DEFINE_AND_REGISTER_HSP_INFERENCE(MaxPool, "tfg.MaxPool");
DEFINE_AND_REGISTER_HSP_INFERENCE(MaxPoolGrad, "tfg.MaxPoolGrad");
DEFINE_AND_REGISTER_HSP_INFERENCE(Cast, "tfg.Cast");
DEFINE_AND_REGISTER_HSP_INFERENCE(Softmax, "tfg.Softmax")
+ "tfg.LogSoftmax";
DEFINE_AND_REGISTER_HSP_INFERENCE(SoftmaxCrossEntropy,
                                  "tfg.SoftmaxCrossEntropyWithLogits")
+ "tfg.SparseSoftmaxCrossEntropyWithLogits";

DEFINE_AND_REGISTER_HSP_INFERENCE(Unary, "tfg.Neg")
+ "tfg.Relu" +
    "tfg.Gelu" + "tfg.Identity" + "tfg.Reciprocal" + "tfg.Sqrt" + "tfg.Rsqrt" +
    "tfg.Abs" + "tfg.ZerosLike" + "tfg.Exp" + "tfg.Log" + "tfg.Square" +
    "tfg.Erf" + "tfg.Tanh" + "tfg.Sigmoid" + "tfg.IsFinite" + "tfg.LogicalNot" +
    "tfg.Cos" + "tfg.Acos" + "tfg.Cosh" + "tfg.Acosh" + "tfg.Angle" +
    "tfg.Asin" + "tfg.Asinh" + "tfg.Atan" + "tfg.Atan2" + "tfg.Atanh" +
    "tfg.Ceil" + "tfg.Conj" + "tfg.Floor" + "tfg.IsNan" + "tfg.Log1p" +
    "tfg.Round" + "tfg.Sign" + "tfg.Sin" + "tfg.Relu6" +
    "tfg.Dequantize";  // consider elementwise 3, 1

DEFINE_AND_REGISTER_HSP_INFERENCE(Binary, "tfg.SigmoidGrad")
// don't support broadcasting
+ "tfg.DivNoNan" +
    "tfg.Div" + "tfg.Mod" + "tfg.RealDiv" + "tfg.ReluGrad" + "tfg.GeluGrad" +
    "tfg.AddV2" + "tfg.Add" + "tfg.Equal" + "tfg.TanhGrad" + "tfg.RsqrtGrad" +
    "tfg.SqrtGrad" + "tfg.Pow" + "tfg.BitwiseAnd" + "tfg.BitwiseOr" +
    "tfg.BitwiseXor" + "tfg.LeftShift" +
    "tfg.RightShift"
    // support broadcasting
    + "tfg.Greater" + "tfg.GreaterEqual" + "tfg.SquaredDifference" +
    "tfg.Maximum" + "tfg.Minimum" + "tfg.FloorDiv" + "tfg.Sub" +
    "tfg.FloorMod" + "tfg.NotEqual" + "tfg.Less" + "tfg.LessEqual" +
    "tfg.LogicalAnd" + "tfg.Mul" + "tfg.MulNoNan" + "tfg.LogicalOr" +
    "tfg.TruncateMod" + "tfg.Select";

DEFINE_AND_REGISTER_HSP_INFERENCE(Nnary, "tfg.AddN")
+ "tfg.Concat" + "tfg.SelectV2";

DEFINE_AND_REGISTER_HSP_INFERENCE(IdentityN, "tfg.IdentityN");

DEFINE_AND_REGISTER_HSP_INFERENCE(Reduction, "tfg.Mean")
+ "tfg.Sum" + "tfg.ArgMax" + "tfg.L2Loss" + "tfg.Max" + "tfg.Min" + "tfg.Prod";

DEFINE_AND_REGISTER_HSP_INFERENCE(Shape, "tfg.ShapeN")
+ "tfg.Shape";

DEFINE_AND_REGISTER_HSP_INFERENCE(Size, "tfg.Size");

DEFINE_AND_REGISTER_HSP_INFERENCE(Reshape, "tfg.Reshape")
+ "tfg.BroadcastTo" + "tfg.InvertPermutation" + "tfg.StridedSliceGrad";

DEFINE_AND_REGISTER_HSP_INFERENCE(Transpose, "tfg.Transpose");

DEFINE_AND_REGISTER_HSP_INFERENCE(Squeeze, "tfg.Squeeze");
DEFINE_AND_REGISTER_HSP_INFERENCE(Pad, "tfg.Pad");
DEFINE_AND_REGISTER_HSP_INFERENCE(StridedSlice, "tfg.StridedSlice")
+ "tfg.Slice";
DEFINE_AND_REGISTER_HSP_INFERENCE(Tile, "tfg.Tile");
DEFINE_AND_REGISTER_HSP_INFERENCE(Unique, "tfg.Unique");
DEFINE_AND_REGISTER_HSP_INFERENCE(Fill, "tfg.Fill")
+ "tfg.RandomUniform";
DEFINE_AND_REGISTER_HSP_INFERENCE(Pack, "tfg.Pack")
+ "tfg.Unpack";
DEFINE_AND_REGISTER_HSP_INFERENCE(ExpandDims, "tfg.ExpandDims");
DEFINE_AND_REGISTER_HSP_INFERENCE(Where, "tfg.Where");
DEFINE_AND_REGISTER_HSP_INFERENCE(Einsum, "tfg.Einsum");
DEFINE_AND_REGISTER_HSP_INFERENCE(OneHot, "tfg.OneHot");
DEFINE_AND_REGISTER_HSP_INFERENCE(BroadcastGradientArgs,
                                  "tfg.BroadcastGradientArgs");
DEFINE_AND_REGISTER_HSP_INFERENCE(GatherV2, "tfg.GatherV2");
DEFINE_AND_REGISTER_HSP_INFERENCE(ResourceGather, "tfg.ResourceGather");
DEFINE_AND_REGISTER_HSP_INFERENCE(QuantizedMaxPool, "tfg.QuantizedMaxPool");
DEFINE_AND_REGISTER_HSP_INFERENCE(QuantizedConv2D,
                                  "tfg.QuantizedConv2DWithBiasAndRequantize")
+ "tfg.QuantizedConv2DWithBiasAndReluAndRequantize" +
    "tfg.QuantizedConv2DWithBiasSumAndReluAndRequantize" +
    "tfg.QuantizedConv2DWithBiasSignedSumAndReluAndRequantize";

}  // namespace as
