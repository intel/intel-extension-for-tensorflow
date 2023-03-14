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

#include "xpuautoshard/common/sharding_property.h"

#include <assert.h>

#include <algorithm>

namespace as {

constexpr int64_t ShardingProperty::UNRANKED;
constexpr int64_t ShardingProperty::DYNAMIC_DIM_SIZE;

ShardingProperty::ShardingProperty(const DeviceInfo& device_info,
                                   DataType element_type, int64_t rank,
                                   const std::vector<int64_t>& shapes)
    : element_type_(element_type),
      rank_(rank),
      shape_(shapes),
      is_shape_or_size_tensor_(false) {
  for (auto&& device : device_info.getDevices()) {
    device_ids_.push_back(device.getId());
  }
  if (isRanked()) {
    split_specs_.resize(rank_);
    // If we only have one device with single stage, every dim would be marked
    // as single split
    if (device_ids_.size() == 1 &&
        device_info.getDevice(device_ids_[0]).getNumStages() == 1) {
      for (size_t i = 0; i < split_specs_.size(); i++) {
        split_specs_[i] = SplitSpec::buildSingleSplit(*this, i);
      }
    }
  }
}

ShardingProperty::ShardingProperty(const ShardingProperty& rhs)
    : element_type_(rhs.element_type_),
      rank_(rhs.rank_),
      shape_(rhs.shape_),
      device_ids_(rhs.device_ids_),
      split_specs_(rhs.split_specs_),
      ordered_split_dims_(rhs.ordered_split_dims_),
      single_split_only_stage_offsets_(rhs.single_split_only_stage_offsets_),
      is_shape_or_size_tensor_(rhs.is_shape_or_size_tensor_),
      shape_split_specs_(rhs.shape_split_specs_),
      size_split_spec_(rhs.size_split_spec_) {
  for (auto&& post_op : rhs.post_ops_) {
    post_ops_.emplace_back(std::unique_ptr<PostOp>(post_op->clone()));
  }
}

bool ShardingProperty::operator==(const ShardingProperty& rhs) const {
  bool eq = this->element_type_ == rhs.element_type_ &&
            this->rank_ == rhs.rank_ && this->device_ids_ == rhs.device_ids_ &&
            this->shape_ == rhs.shape_ &&
            this->split_specs_ == rhs.split_specs_ &&
            this->ordered_split_dims_ == rhs.ordered_split_dims_ &&
            this->post_ops_.size() == rhs.post_ops_.size() &&
            this->single_split_only_stage_offsets_ ==
                rhs.single_split_only_stage_offsets_ &&
            this->is_shape_or_size_tensor_ == rhs.is_shape_or_size_tensor_ &&
            this->shape_split_specs_ == rhs.shape_split_specs_ &&
            this->size_split_spec_ == rhs.size_split_spec_;
  if (eq) {
    for (size_t i = 0; i < this->post_ops_.size(); i++) {
      if (*this->post_ops_[i] != *rhs.post_ops_[i]) {
        return false;
      }
    }
  }
  return eq;
}

bool ShardingProperty::splitSingleOnly(
    const std::vector<int64_t>& stage_offsets) {
  bool changed = false;
  if (getRank() > 0) {
    for (int64_t i = 0; i < rank_; i++) {
      changed |= splitSingleAt(i);
    }
  }
  if (single_split_only_stage_offsets_.empty() &&
      single_split_only_stage_offsets_ != stage_offsets) {
    single_split_only_stage_offsets_ = stage_offsets;
    changed = true;
  }
  return changed;
}

bool ShardingProperty::splitSingleAt(int64_t dim) {
  return splitAt(dim, SplitSpec::buildSingleSplit(*this, dim));
}

bool ShardingProperty::isSplitSingleOnly() const {
  return isInitialized() &&
         (!isRanked() || std::all_of(split_specs_.begin(), split_specs_.end(),
                                     [](const SplitSpec& split_spec) {
                                       return split_spec.isInitialized() &&
                                              split_spec.size() == 1;
                                     }));
}

bool ShardingProperty::splitAt(int64_t dim, const SplitSpec& split_spec) {
  assert(dim >= 0);
  if (!isRanked() || isSplitAt(dim) || !isDevicesAssigned()) {
    return false;
  }
  // For now, we only support split > 1 on one of the dims and number of splits
  // equals to the number of stages.
  // TODO(itex): support split > 1 on more than one dim or less than number of
  // stages. Construct `new_split_spec` based on the same proportion of
  // `split_spec` and current dim size.
  SplitSpec new_split_spec;
  auto size_sum = split_spec.getTotalSize();
  if (shape_.size() <= dim || shape_[dim] <= 0 || size_sum <= 0) {
    new_split_spec = split_spec;
  } else {
    std::vector<int64_t> new_sizes;
    for (auto size : split_spec.getSizes()) {
      assert(shape_[dim] * size % size_sum == 0 &&
             "Expect split proportionally on the dim");
      // TODO(itex): If it cannot be split proportionally, the whole module
      // will not be split and the original module will be returned.
      new_sizes.push_back(shape_[dim] * size / size_sum);
    }
    new_split_spec = SplitSpec::buildFromSizes(*this, dim, new_sizes,
                                               split_spec.getStageOffsets());
  }
  split_specs_[dim] = new_split_spec;
  if (split_spec.size() > 1) {
    ordered_split_dims_.push_back(dim);
    if (isCompleteSplitAt(dim)) {
      // For a complete split on `dim`, mark all other dims as single split
      for (size_t i = 0; i < split_specs_.size(); i++) {
        if (i != dim) {
          if (!split_specs_[i].isInitialized()) {
            split_specs_[i] = SplitSpec::buildSingleSplit(*this, i);
          } else {
            assert(split_specs_[i].getType() == SplitSpec::SplitType::SINGLE &&
                   "Expect single split on other dims");
          }
        }
      }
    }
  }
  return true;
}

bool ShardingProperty::isSplitAt(int64_t dim) const {
  if (!isRanked() || isScalar()) {
    return false;
  }
  return split_specs_[dim].isInitialized();
}

bool ShardingProperty::isSingleSplitAt(int64_t dim) const {
  if (isInitialized() == false) {
    return false;
  }
  auto split_spec = getSplitSpec(dim);
  if (split_spec.isInitialized() == false) {
    return false;
  }
  return split_spec.getType() == SplitSpec::SplitType::SINGLE;
}

bool ShardingProperty::isCompleteSplitAt(int64_t dim) const {
  if (!isSplitAt(dim)) {
    return false;
  }
  return split_specs_[dim].size() >= getNumLogicalShards();
}

SplitSpec ShardingProperty::getSplitSpec(int64_t dim) const {
  assert(isRanked() && !isScalar());
  return split_specs_[dim];
}

bool ShardingProperty::isInitialized() const {
  // we assume stage numbers initialized when devices are assigned
  return isDevicesAssigned() &&
         (!isRanked() || std::all_of(split_specs_.begin(), split_specs_.end(),
                                     [](const SplitSpec& split_spec) {
                                       return split_spec.isInitialized();
                                     }));
}

DeviceId ShardingProperty::getDeviceIdPerShardNum(size_t shard_num) const {
  std::vector<int64_t> stage_offsets = getStageOffsets();
  assert(!device_ids_.empty());
  assert(!stage_offsets.empty() || shard_num < device_ids_.size());
  DeviceId device_id =
      device_ids_[stage_offsets.empty() ? shard_num : device_ids_.size() - 1];
  for (size_t i = 0; i < stage_offsets.size(); i++) {
    if (shard_num < stage_offsets[i]) {
      device_id = device_ids_[i];
      break;
    }
  }
  return device_id;
}

std::vector<ShardDesc> ShardingProperty::getShardDescriptors() const {
  std::vector<ShardDesc> shard_descs;
  size_t num_logical_shards = getNumLogicalShards();
  for (size_t shard_num = 0; shard_num < num_logical_shards; shard_num++) {
    std::vector<int64_t> shape(getShape());
    std::vector<SplitSpec> split_specs;
    // fill split_specs
    for (auto split_dim : ordered_split_dims_) {
      split_specs.push_back(split_specs_[split_dim]);
    }
    // compute shape and ratio
    float ratio = 1.0;
    // A tensor is split by the dims in ordered_split_dims_ in the order
    // that forms a tree. Each leaf node corresponds to a shard. With the
    // shard_num of a leaf node, the following loop identifies the all
    // corresponding SplitSpecs in the split tree and computes the device
    // placement, shape and ratio of the shard accordingly.
    size_t split_indexing = shard_num;
    for (int64_t j = ordered_split_dims_.size() - 1; j >= 0; j--) {
      auto split_dim = ordered_split_dims_[j];
      auto&& split_spec = split_specs_[split_dim];
      size_t split_num = split_indexing % split_spec.size();
      split_indexing /= split_spec.size();
      SplitSpec::SplitType split_type = split_spec.getType();
      // compute the ratio
      if (split_type == SplitSpec::SplitType::SIZE) {
        auto&& sizes = split_spec.getSizes();
        size_t total = std::accumulate(sizes.begin(), sizes.end(), 0);
        ratio *= (1.0f * split_spec.getSizes()[split_num]) / total;
      } else if (split_type == SplitSpec::SplitType::RATIO) {
        ratio *= split_spec.getRatios()[split_num];
      }
      if (isDynamicSize(shape[split_dim])) {
        continue;
      }
      // compute the shape
      if (split_spec.getType() == SplitSpec::SplitType::SIZE) {
        shape[split_dim] = split_spec.getSizes()[split_num];
      } else if (split_spec.getType() == SplitSpec::SplitType::RATIO) {
        // TODO(itex): Calculate the shape of each shard by ratio
        // Assume the shape can be divided by ratio
        shape[split_dim] = static_cast<int>(shape[split_dim] *
                                            split_spec.getRatios()[split_num]);
      } else {
        assert(false && "Expect split with concrete sizes");
      }
    }
    ShardDesc desc(shard_num, getRank(), shape,
                   getDeviceIdPerShardNum(shard_num), ordered_split_dims_,
                   split_specs, ratio);
    shard_descs.emplace_back(desc);
  }
  return shard_descs;
}

std::set<DeviceId> ShardingProperty::getDeviceSetBySpec(
    int64_t dim, const SplitSpec& split_spec, size_t split_num) const {
  assert(split_spec.isInitialized());
  std::set<DeviceId> device_set;
  if (split_spec.getType() == SplitSpec::SplitType::SINGLE) {
    device_set.insert(device_ids_.begin(), device_ids_.end());
  } else if (split_spec.isMultiStages()) {
    assert(split_spec.getStageOffsets().size() == device_ids_.size());
    for (size_t i = 0; i < device_ids_.size(); i++) {
      if (split_num < split_spec.getStageOffsets()[i]) {
        device_set.insert(device_ids_[i]);
        break;
      }
    }
  } else if (split_spec.size() == device_ids_.size()) {
    device_set.insert(device_ids_[split_num]);
  } else {
    assert(split_spec.size() < device_ids_.size());
    // TODO(itex): return the device set according to the "dim" in the
    // ordered_split_dims_
    assert(false && "Not implemented yet");
  }
  return device_set;
}

std::vector<int64_t> ShardingProperty::getStageOffsets() const {
  if (isSplitSingleOnly()) {
    return single_split_only_stage_offsets_;
  }
  for (size_t i = 0; i < split_specs_.size(); i++) {
    if (split_specs_[i].isInitialized() &&
        !split_specs_[i].getStageOffsets().empty()) {
      // TODO(itex): combine stage offsets from multiple non-single split specs
      // currently, we assume at most one non-single split spec
      return split_specs_[i].getStageOffsets();
    }
  }
  return std::vector<int64_t>();
}

size_t ShardingProperty::getNumLogicalShards() const {
  auto&& num_shards_per_device = getNumLogicalShardsPerdevice();
  return std::accumulate(num_shards_per_device.begin(),
                         num_shards_per_device.end(), 0);
}

std::vector<size_t> ShardingProperty::getNumLogicalShardsPerdevice() const {
  std::vector<int64_t> offsets = getStageOffsets();
  if (offsets.empty()) {
    return std::vector<size_t>(device_ids_.size(), 1);
  }
  std::vector<size_t> num_shards_per_device;
  int64_t prev_offset = 0;
  for (size_t i = 0; i < offsets.size(); i++) {
    num_shards_per_device.push_back(offsets[i] - prev_offset);
    prev_offset = offsets[i];
  }
  return num_shards_per_device;
}

bool ShardingProperty::isShapeTensor() const {
  return is_shape_or_size_tensor_ && getRank() == 1;
}

bool ShardingProperty::isSizeTensor() const {
  return is_shape_or_size_tensor_ && getRank() == 0;
}

bool ShardingProperty::maybeShapeTensor() const {
  return getRank() == 1 && getShape()[0] >= 0 && isElementIntegerType() &&
         isSplitSingleOnly();
}

bool ShardingProperty::maybeSizeTensor() const {
  return getRank() == 0 && isElementIntegerType();
}

bool ShardingProperty::setShapeTensor(
    const std::vector<SplitSpec>& split_specs) {
  if (isShapeTensor() || !maybeShapeTensor() ||
      getShape()[0] != split_specs.size()) {
    return false;
  }
  shape_split_specs_ = split_specs;
  is_shape_or_size_tensor_ = true;
  return true;
}

bool ShardingProperty::setShapeOrSizeTensorWith(const ShardingProperty& rhs) {
  if (rhs.isShapeTensor()) {
    return setShapeTensor(rhs.getShapeSplitSpecs());
  } else if (rhs.isSizeTensor()) {
    return setSizeTensor(rhs.getSizeSplitSpec());
  }
  return false;
}

bool ShardingProperty::setSizeTensor(const SplitSpec& split_spec) {
  if (isSizeTensor() || !maybeSizeTensor()) {
    return false;
  }
  size_split_spec_ = split_spec;
  is_shape_or_size_tensor_ = true;
  return true;
}

bool ShardingProperty::setSizeTensor(const ShardingProperty& rhs) {
  if (isSizeTensor() || !maybeSizeTensor() || !rhs.isInitialized()) {
    return false;
  }
  if (rhs.getNumSplits() > 1) {
    auto shard_descs = rhs.getShardDescriptors();
    std::vector<float> ratios;
    for (auto&& shard_desc : shard_descs) {
      ratios.push_back(shard_desc.getRatio());
    }
    size_split_spec_ = SplitSpec::buildFromRatios(*this, -1, ratios);
  } else {
    size_split_spec_ = SplitSpec::buildSingleSplit(*this, /*dim*/ -1);
  }
  is_shape_or_size_tensor_ = true;
  return true;
}

bool ShardingProperty::setShapeTensorSingleSplitOnly() {
  if (isShapeTensor() || !maybeShapeTensor()) {
    return false;
  }
  std::vector<SplitSpec> split_specs;
  for (int64_t i = 0; i < getShape()[0]; i++) {
    // FIXME: passing `*this` doesn't match the semantics of the hsp expected by
    // `buildSingleSplit`.
    split_specs.push_back(SplitSpec::buildSingleSplit(*this, i));
  }
  return setShapeTensor(split_specs);
}

bool ShardingProperty::isShapeTensorSingleSplitOnly() {
  if (!isShapeTensor()) {
    return false;
  }
  return std::all_of(shape_split_specs_.begin(), shape_split_specs_.end(),
                     [](const SplitSpec& split_spec) {
                       return split_spec.getType() ==
                              SplitSpec::SplitType::SINGLE;
                     });
}

const std::vector<SplitSpec>& ShardingProperty::getShapeSplitSpecs() const {
  return shape_split_specs_;
}

const SplitSpec& ShardingProperty::getSizeSplitSpec() const {
  return size_split_spec_;
}

bool ShardingProperty::isElementFloatingPointType() const {
  return (element_type_ == DataType::FLOAT32 ||
          element_type_ == DataType::FLOAT16 ||
          element_type_ == DataType::BFLOAT16 ||
          element_type_ == DataType::FLOAT64);
}

bool ShardingProperty::isElementIntegerType() const {
  return element_type_ == DataType::INTEGER;
}

SplitSpec SplitSpec::buildFromSizes(const ShardingProperty& prop, int64_t dim,
                                    const std::vector<int64_t>& sizes,
                                    const std::vector<int64_t>& stage_offsets) {
  SplitSpec spec;
  spec.sizes_ = sizes;
  spec.type_ = SplitType::SIZE;
  spec.stage_offsets_ = stage_offsets;
  return spec;
}

SplitSpec SplitSpec::buildFromRatios(
    const ShardingProperty& prop, int64_t dim, const std::vector<float>& ratios,
    const std::vector<int64_t>& stage_offsets) {
  SplitSpec spec;
  spec.ratios_ = ratios;
  spec.type_ = SplitType::RATIO;
  spec.stage_offsets_ = stage_offsets;
  return spec;
}

SplitSpec SplitSpec::buildSingleSplit(const ShardingProperty& prop,
                                      int64_t dim) {
  SplitSpec spec;
  spec.type_ = SplitType::SINGLE;
  return spec;
}

SplitSpec::SplitSpec() : type_(SplitType::UNINIT) {}

bool SplitSpec::isInitialized() const { return type_ != SplitType::UNINIT; }

std::size_t SplitSpec::size() const {
  switch (type_) {
    case SplitType::SINGLE:
      return 1;
    case SplitType::RATIO:
      return ratios_.size();
    case SplitType::SIZE:
      return sizes_.size();
    default:
      return 0;
  }
}

}  // namespace as
