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

#include <stdint.h>

#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "xpuautoshard/common/device_info.h"
#include "xpuautoshard/common/op_desc.h"
#include "xpuautoshard/common/ref_base.h"

namespace as {

class ShardingProperty;

/**
 * @brief Describe how a specific dim is split. The data structure is designed
 * to be constructed read-only. When initialized, a SplitSpec is only meaningful
 * with respect to a specific sharding property and the dim on which it
 * specifies.
 *
 * Currently, four types of splitting are supported:
 * UNINIT: not initialized
 * SINGLE: no split with same size on all shards, the content is not necessarily
 * the same, e.g., apply a Shape op on tensor shards split by sizes, the size of
 * the shape tensor are the same for all shards but the exact shapes of these
 * shards are not necessarily the same, meaning the content might be different.
 *         work for both concrete or dynamic shape of the specific dim.
 * RATIO: split by percentage ratios, work for both concrete or dynamic shape of
 * the specific dim. SIZE: split by concrete sizes, only work for concrete shape
 * of the specific dim.
 *
 * TODO(itex): support halo split
 */
class SplitSpec {
 public:
  enum class SplitType {
    UNINIT,
    SINGLE,
    RATIO,
    SIZE,
  };

  /**
   * @brief Build a SplitSpec object with specific `sizes` for the given
   * sharding property `prop` on the `dim`. If `stage_offsets` is not empty, its
   * size is the same as the devices of `prop` and specifies the offsets into
   * the `sizes` for devices. For example, a split on two devices with `sizes`
   * [20,20,10,10,10,5] and `stage_offsets` [2,6] means 2 stages with sizes
   * [20,20] on the first device and 4 stages with sizes [10,10,10,5] on the
   * second.
   *
   * @param prop
   * @param dim
   * @param sizes
   * @param stage_offsets
   * @return SplitSpec
   */
  static SplitSpec buildFromSizes(
      const ShardingProperty& prop, int64_t dim,
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& stage_offsets = std::vector<int64_t>());

  /**
   * @brief Similar to `buildFromSizes` but the tensor is split with a ratio,
   * not a concrete size.
   *
   * @param prop
   * @param dim
   * @param ratios
   * @param stage_offsets
   * @return SplitSpec
   */
  static SplitSpec buildFromRatios(
      const ShardingProperty& prop, int64_t dim,
      const std::vector<float>& ratios,
      const std::vector<int64_t>& stage_offsets = std::vector<int64_t>());

  static SplitSpec buildSingleSplit(const ShardingProperty& prop, int64_t dim);

  /**
   * @brief Construct an uninitialized Split Spec object
   *
   */
  SplitSpec();

  bool operator==(const SplitSpec& rhs) const {
    bool equal = this->type_ == rhs.type_;
    if (this->type_ == SplitType::RATIO) {
      equal &= this->ratios_ == rhs.ratios_;
    } else if (this->type_ == SplitType::SIZE) {
      equal &= this->sizes_ == rhs.sizes_;
    }
    equal &= stage_offsets_ == rhs.stage_offsets_;
    return equal;
  }

  bool operator!=(const SplitSpec& rhs) const { return !(*this == rhs); }

  /**
   * @brief Determine whether two `SplitType::SIZE` split specs can be
   * converted equivalently. Equivalent conversion means that the
   * number and ratio of split are equal.
   * @param rhs
   * @return bool
   */
  bool isSizeSplitConvertEquivalently(const SplitSpec& rhs) const {
    if (this->getType() != rhs.getType()) {
      return false;
    }
    auto&& sizes_vec_1 = this->getSizes();
    auto&& sizes_vec_2 = rhs.getSizes();
    if (sizes_vec_1.size() != sizes_vec_2.size()) {
      return false;
    }
    auto sum_1 = this->getTotalSize();
    auto sum_2 = rhs.getTotalSize();
    for (auto id = 0; id < sizes_vec_1.size(); id++) {
      if (sizes_vec_1[id] / sum_1 != sizes_vec_2[id] / sum_2) {
        return false;
      }
    }
    return true;
  }

  bool isInitialized() const;

  /**
   * @brief The number of splits this SplitSpec specifies
   *
   * @return std::size_t
   */
  std::size_t size() const;

  const std::vector<int64_t>& getSizes() const { return sizes_; }

  /**
   * @brief Get the shape of the original dim under split
   *
   * @return const int64_t
   */
  const int64_t getTotalSize() const {
    return std::accumulate(sizes_.begin(), sizes_.end(), 0);
  }

  const std::vector<float>& getRatios() const { return ratios_; }

  const SplitType getType() const { return type_; }

  const std::vector<int64_t>& getStageOffsets() const { return stage_offsets_; }

  bool isMultiStages() const {
    auto stage_nums = getStageNums();
    for (auto stage_num : stage_nums) {
      if (stage_num > 1) {
        return true;
      }
    }
    return false;
  }

  std::vector<int64_t> getStageNums() const {
    std::vector<int64_t> stage_nums;
    int64_t pre_offset = 0;
    for (auto offset : stage_offsets_) {
      stage_nums.push_back(offset - pre_offset);
      pre_offset = offset;
    }
    return stage_nums;
  }

 private:
  std::vector<int64_t> sizes_;
  std::vector<float> ratios_;
  SplitType type_;
  std::vector<int64_t> stage_offsets_;
};

/**
 * @brief A descriptor corresponding to one shard of
 * a sharded tensor.
 *
 */
class ShardDesc {
 public:
  /**
   * @brief Construct a new ShardDesc object
   *
   * @param num The shard number among all shards (starting from 0)
   * @param rank The rank of this shard
   * @param shape The shape of this shard
   * @param device_id The device the shard resides
   * @param ordered_split_dims According to which dims the shard is split from
   * the original tensor
   * @param split_specs Corresponding SplitSpec for the `ordered_split_dims`
   */
  ShardDesc(
      size_t num, int64_t rank, const std::vector<int64_t>& shape,
      DeviceId device_id,
      const std::vector<int64_t>& ordered_split_dims = std::vector<int64_t>(),
      const std::vector<SplitSpec>& split_specs = std::vector<SplitSpec>(),
      float ratio = 1.0f)
      : num_(num),
        rank_(rank),
        shape_(shape),
        device_id_(device_id),
        ordered_split_dims_(ordered_split_dims),
        split_specs_(split_specs),
        ratio_(ratio) {}

  /**
   * @brief Get the rank of this shard
   *
   */
  int64_t getRank() const { return rank_; }
  /**
   * @brief Get the shape of this shard
   *
   */
  const std::vector<int64_t>& getShape() const { return shape_; }

  /**
   * @brief Get the times of split operation applied to
   * the sharded tensor so that the shard is generated.
   * Returns 0 if the tensor is single split only. Note that
   * a sharded tensor could be split at multiple dims.
   */
  size_t getTimesOfSplit() const { return ordered_split_dims_.size(); }
  /**
   * @brief Get the split dim of the `split_num`-th split. If
   * the ShardDesc is single split only, the returned value is undefined.
   *
   * @param split_num The number of a split, -1 means the last split.
   */
  int64_t getSplitDim(int split_num) const {
    if (split_num < 0) {
      split_num += ordered_split_dims_.size();
    }
    return ordered_split_dims_[split_num];
  }
  /**
   * @brief Get the SplitSpec object of the `split_num`-th split. If
   * the ShardDesc is single split only, the returned value is undefined.
   *
   * @param split_num The number of a split, -1 means the last split
   */
  const SplitSpec& getSplitSpec(int split_num) const {
    if (split_num < 0) {
      split_num += split_specs_.size();
    }
    return split_specs_[split_num];
  }
  /**
   * @brief Get the device id this shard resides.
   *
   * @return DeviceId
   */
  DeviceId getDeviceId() const { return device_id_; }
  /**
   * @brief Get the shard number among all shards (starting from 0)
   *
   * @return size_t
   */
  size_t getNum() const { return num_; }

  /**
   * @brief Get the ratio of this shard w.r.t. the original tensor
   *
   * @return float
   */
  float getRatio() const { return ratio_; }

 private:
  size_t num_;
  int64_t rank_;
  std::vector<int64_t> shape_;
  DeviceId device_id_;
  std::vector<int64_t> ordered_split_dims_;
  std::vector<SplitSpec> split_specs_;
  float ratio_;
};

/**
 * @brief A post-op representing a compute on an output tensor from
 * an op.
 *
 */
class PostOp {
 public:
  virtual ~PostOp() = default;
  virtual std::string getName() const = 0;
  virtual PostOp* clone() = 0;
  virtual bool operator==(const PostOp& rhs) = 0;
  virtual bool operator!=(const PostOp& rhs) { return !(*this == rhs); }
};

template <typename PostOpClass>
class PostOpBase : public PostOp {
 public:
  virtual ~PostOpBase() = default;
  std::string getName() const override { return typeid(PostOpClass).name(); }
  PostOp* clone() override {
    return new PostOpClass(*dynamic_cast<const PostOpClass*>(this));
  }
  bool operator==(const PostOp& rhs) override {
    return dynamic_cast<const PostOpClass*>(&rhs) != nullptr;
  }
};

/**
 * @brief All-reduce sum post-op
 *
 */
class AllReduceSumPostOp : public PostOpBase<AllReduceSumPostOp> {
 public:
  virtual ~AllReduceSumPostOp() = default;
};

class AllReduceMaxPostOp : public PostOpBase<AllReduceMaxPostOp> {
 public:
  virtual ~AllReduceMaxPostOp() = default;
};

class AllReduceMinPostOp : public PostOpBase<AllReduceMinPostOp> {
 public:
  virtual ~AllReduceMinPostOp() = default;
};

class AllReduceProdPostOp : public PostOpBase<AllReduceProdPostOp> {
 public:
  virtual ~AllReduceProdPostOp() = default;
};

class AllReduceL2PostOp : public PostOpBase<AllReduceL2PostOp> {
 public:
  virtual ~AllReduceL2PostOp() = default;
};

class ArgMaxPostOp : public PostOpBase<ArgMaxPostOp> {
 public:
  virtual ~ArgMaxPostOp() = default;
};

class WeightedScalePostOp : public PostOpBase<WeightedScalePostOp> {
 public:
  /**
   * @brief Construct a new Weighted Scale Post Op object
   *
   * @param weights Flattened vector of the weights tensor with shape: <shape of
   * unsharded tensor> * num_of_shards
   * @param shape The shape of the unsharded tensor
   */
  WeightedScalePostOp(
      const std::vector<float>& weights,
      const std::vector<int64_t>& shape = std::vector<int64_t>())
      : weights_(weights), shape_(shape) {}
  virtual ~WeightedScalePostOp() = default;

  bool operator==(const PostOp& rhs) override {
    auto&& p_rhs = dynamic_cast<const WeightedScalePostOp*>(&rhs);
    return p_rhs != nullptr && p_rhs->getWeights() == this->getWeights();
  }

  const std::vector<float>& getWeights() const { return weights_; }

  const std::vector<int64_t>& getTensorShape() const { return shape_; }

  size_t getTensorSize() const {
    int64_t tensor_size = 1;
    for (auto dim : shape_) {
      tensor_size *= dim;
    }
    return tensor_size;
  }

  size_t getNumShards() const { return weights_.size() / getTensorSize(); }

 private:
  std::vector<float> weights_;
  std::vector<int64_t> shape_;
};

/**
 * @brief Add a slice operation on a replicate tensor according
 * to the output HSP having multiple splits.
 *
 */
class SlicePostOp : public PostOpBase<SlicePostOp> {
 public:
  virtual ~SlicePostOp() = default;
};

/**
 * @brief Indicate a shape tensor needs to be sliced
 *
 */
class ShapeSlicePostOp : public PostOpBase<ShapeSlicePostOp> {
 public:
  ShapeSlicePostOp(const std::vector<int64_t>& split_dims,
                   const std::vector<SplitSpec>& split_specs)
      : split_dims_(split_dims), split_specs_(split_specs) {}
  virtual ~ShapeSlicePostOp() = default;
  /**
   * @brief Get dims under multiple splits
   *
   * @return std::vector<int64_t>
   */
  const std::vector<int64_t>& getSplitDims() const { return split_dims_; }

  const std::vector<SplitSpec>& getSplitSpecs() const { return split_specs_; }

 private:
  std::vector<int64_t> split_dims_;
  std::vector<SplitSpec> split_specs_;
};

// TODO(itex): Describe how a specific dim after split is mapped
// to devices. Devices are organized in a list sorted by their
// ids.
// For now, we just use std::vector for simplicity.
using DeviceMapping = std::vector<DeviceId>;

/**
 * @brief A sharding property describes how a tensor is sharded on a set of
 * devices the tensor resides. Here, the tensor could be either ranked with
 * known or unknown dim info, or unranked. Non-tensor type is treated as an
 * unranked tensor. The device set the tensor resides in could be of different
 * types, e.g., 2 CPU sockets plus 2 GPU cards. Therefore, the sharding property
 * is also referred to as "heterogeneous" sharding property, or HSP in short in
 * the code.
 *
 * Partial order is (implicitly) defined for the sharding property to guarantee
 * monotonicity in HSP inference.
 *
 * The definition of partial order is as follows:
 *
 * Top is "initialized" meaning 1) devices are assigned; 2) number of compute
 * stages set to each device; 3) SplitSpec is initialized for each dim of the
 * tensor if it is ranked. The SplitSpec defines how a particular dim is sharded
 * among the devices. Sharding properties other than "Top" are called
 * "uninitialized" in the code for convenience.
 *
 * Bottom is a sharding property without devices assigned and unranked.
 *
 * Partial order diagram as follows:
 * Bottom (unranked, devices unassigned)  -->  Ranked, device unassigned
 * ----------- |                                        | | |     Ranked, device
 * assigned, number of compute stages set      | (single compute stage assumed)
 *                |                                        | | |      Ranked,
 * device assigned, number of compute stages set, not all SplitSpec initialized
 *                 \                                       |
 * Top (Unranked, device assigned) or (Ranked, device assigned, number of
 * compute stages set, all SplitSpec initialized)
 *
 * Unranked tensor or non-tensor type is by nature single split only after
 * devices are assigned.
 *
 * Note that we assume the type inference is already applied before the auto
 * sharding pass is invoked. Therefore, if a tensor is marked as "unranked", we
 * don't have to infer its rank and turn an unranked HSP into a ranked one for
 * simplicity.
 *
 */
class ShardingProperty {
 public:
  // NOTE: both unranked tensor and non-tensor types are
  // treated as UNRANKED.
  static constexpr int64_t UNRANKED = -1;
  static constexpr int64_t DYNAMIC_DIM_SIZE = -1;

  static bool isDynamicSize(int64_t size) { return size < 0; }

  ShardingProperty(const DeviceInfo& device_info,
                   DataType element_type = DataType::UNKNOWN,
                   int64_t rank = UNRANKED,
                   const std::vector<int64_t>& shapes = std::vector<int64_t>());

  ShardingProperty(const ShardingProperty& rhs);

  bool operator==(const ShardingProperty& rhs) const;

  bool operator!=(const ShardingProperty& rhs) const { return !(*this == rhs); }

  /**
   * @brief Set split=1 on the given `dim`.
   *
   * @param dim
   * @return true
   * @return false The `dim` is already split, or it is
   * an unranked tensor.
   */
  bool splitSingleAt(int64_t dim);

  /**
   * @brief Initialize HSP by setting split to 1 to all dims
   * if the corresponding dims not initialized yet. Do nothing
   * for unranked tensor.
   *
   * @return true Initialization succeeds.
   * @return false HSP already initialized.
   */
  bool splitSingleOnly(
      const std::vector<int64_t>& stage_offsets = std::vector<int64_t>());

  /**
   * @brief Check if all dims having split 1, or an initialized
   * unranked HSP.
   *
   * @return true
   * @return false
   */
  bool isSplitSingleOnly() const;

  /**
   * @brief Split at the specific `dim` per `split_spec`.
   *
   * @param dim
   * @param split_spec
   * @return true
   * @return false The `dim` is already split.
   */
  bool splitAt(int64_t dim, const SplitSpec& split_spec);

  /**
   * @brief Check if the given `dim` has been split. This could be
   * the dim is either single split or multiple split.
   *
   * @param dim
   * @return true
   * @return false
   */
  bool isSplitAt(int64_t dim) const;

  /**
   * @brief Check if the given `dim` has single split.
   *
   * @param dim
   * @return true
   * @return false
   */
  bool isSingleSplitAt(int64_t dim) const;

  /**
   * @brief Whether each device has unique splits at given `dim`.
   *
   * @return true
   * @return false
   */
  bool isCompleteSplitAt(int64_t dim) const;

  /**
   * @brief Get the split spec at the given `dim`.
   *
   * @param dim
   * @return SplitSpec
   */
  SplitSpec getSplitSpec(int64_t dim) const;

  /**
   * @brief Get the stage offsets summarized from all dims or
   * empty vector if it is for single stage.
   *
   * @return const std::vector<int64_t>
   */
  std::vector<int64_t> getStageOffsets() const;

  /**
   * @brief All dims have been split (or no split which means split=1)
   *
   * @return true
   * @return false
   */
  bool isInitialized() const;

  DataType getElementType() const { return element_type_; }

  int64_t getRank() const { return rank_; }

  const std::vector<int64_t>& getShape() const { return shape_; }

  const std::vector<DeviceId>& getDeviceIds() const { return device_ids_; }

  const size_t getNumDevices() const { return device_ids_.size(); }

  /**
   * @brief Get the device id that the logical shard number `shard_num` resides.
   *
   * @param shard_num Logical shard number (starting from 0)
   * @return DeviceId
   */
  DeviceId getDeviceIdPerShardNum(size_t shard_num) const;

  /**
   * @brief Get split spec for each dim for a ranked tensor. If the tensor
   * is unranked, an empty vector is returned. Otherwise, the size of the
   * vector equals to the rank of the tensor.
   *
   * @return const std::vector<SplitSpec>&
   */
  const std::vector<SplitSpec>& getSplitSpecs() const { return split_specs_; }

  /**
   * @brief Get the order of the split dims. The order would impact the
   * device placement of the split shards. For example, a batch-dim split
   * followed by a spatial-dim split would have different device placement
   * from a spatial-dim split followed by a batch-dim split.
   *
   * @return const std::vector<int64_t>&
   */
  const std::vector<int64_t>& getOrderedSplitDims() const {
    return ordered_split_dims_;
  }

  /**
   * @brief Return the total number of logical shards of the sharding property.
   * Multiple logical shards can refer to a same physical tensor shard, which
   * could happen when multiple devices or multiple stages of a device share a
   * single physical copy of the tensor shard. The number of logical shards of a
   * sharded tensor for single-stage compute equals to the number of devices the
   * sharded tensor resides. For multi-stage compute, it is the total number of
   * stages on all the devices.
   *
   * @return size_t
   */
  size_t getNumLogicalShards() const;

  /**
   * @brief Get per-device number of logical shards.
   *
   * @return std::vector<size_t>
   */
  std::vector<size_t> getNumLogicalShardsPerdevice() const;

  /**
   * @brief Return the number of dims having > 1 split type sharding.
   *
   * @return size_t
   */
  size_t numMultiSplitDims() const { return getOrderedSplitDims().size(); }

  /**
   * @brief Get the number of splits.
   *
   * @return size_t
   */
  size_t getNumSplits() const {
    size_t num_splits = 1;
    for (auto&& split_spec : split_specs_) {
      num_splits *= split_spec.size();
    }
    return num_splits;
  }

  void appendPostOp(std::unique_ptr<PostOp>&& post_op) {
    post_ops_.push_back(std::move(post_op));
  }

  size_t getNumPostOps() const { return post_ops_.size(); }

  const std::vector<std::unique_ptr<PostOp>>& getPostOps() const {
    return post_ops_;
  }

  /**
   * @brief Get the ShardDescs of the shards from the sharded tensors.
   * Each shard would reside on a particular device and is sorted according
   * to the order of the devices and the order of the splits specified by
   * the SplitSpec.
   *
   * @return std::vector<ShardDesc>
   */
  std::vector<ShardDesc> getShardDescriptors() const;

  /**
   * @brief Check if the sharding property annotates a shape tensor. A shape
   * tensor is always 1D of integer type with single split only.
   *
   * @return true
   * @return false
   */
  bool isShapeTensor() const;

  /**
   * @brief Check if the sharding property annotates a size tensor. A size
   * tensor is 0D of integer type.
   *
   * @return true
   * @return false
   */
  bool isSizeTensor() const;

  /**
   * @brief Check if the sharding property marks either a shape tensor or
   * a size tensor.
   *
   * @return true
   * @return false
   */
  bool isShapeOrSizeTensor() const;

  /**
   * @brief Mark the sharding property to annotate a shape tensor. The function
   * returns true with the preconditions below:
   * 1. The HSP is for 1D ranked tensor.
   * 2. The size of the tensor matches the size of given `split_specs`.
   * 3. HSP is single split only.
   * 4. Element type is integer.
   * 4. isShapeTensor() returns false.
   *
   * @param split_specs The split spec for each dim of 1D ranked tensor. It
   * marks how the dims of the original tensor that this shape tensor describes
   * are sharded.
   *
   * @return true
   * @return false
   */
  bool setShapeTensor(const std::vector<SplitSpec>& split_specs);

  /**
   * @brief Mark the sharding property with either shape or size tensor
   * following the `rhs`.
   *
   * @param rhs
   * @return true
   * @return false
   */
  bool setShapeOrSizeTensorWith(const ShardingProperty& rhs);

  /**
   * @brief Mark the sharding property to annotate a size tensor. The function
   * returns true with the preconditions below:
   * 1. The HSP is for 0D ranked tensor, hence single split only.
   * 2. Element type is integer.
   * 3. isSizeTensor() returns false.
   *
   * @param rhs The sharding property of the tensor to take the size with.
   * @return true
   * @return false
   */
  bool setSizeTensor(const ShardingProperty& rhs);

  /**
   * @brief Mark the sharding property to annotate a size tensor, according to
   * the size split spec `split_spec`.
   *
   * @param split_spec
   * @return true
   * @return false
   */
  bool setSizeTensor(const SplitSpec& split_spec);

  /**
   * @brief Similar to `setShapeTensor` but split spec for each dim is single
   * split.
   *
   * @return true
   * @return false
   */
  bool setShapeTensorSingleSplitOnly();

  /**
   * @brief Check if the shape tensor describes a single split tensor
   *
   * @return true
   * @return false
   */
  bool isShapeTensorSingleSplitOnly();

  /**
   * @brief Get the split spec for each dim of 1D ranked tensor. It marks
   * how the dims of the original tensor that this shape tensor describes are
   * sharded.
   *
   * @return const std::vector<SplitSpec>&
   */
  const std::vector<SplitSpec>& getShapeSplitSpecs() const;

  /**
   * @brief Get the split spec for the original tensor as if it is flattened.
   *
   * @return const SplitSpec&
   */
  const SplitSpec& getSizeSplitSpec() const;

  /**
   * @brief Check if the sharding property annotates a tensor with elements
   * of a floating point type.
   *
   * @return true
   * @return false
   */
  bool isElementFloatingPointType() const;

  /**
   * @brief Check if the sharding property annotates a tensor with elements
   * of an integer type.
   *
   * @return true
   * @return false
   */
  bool isElementIntegerType() const;

 private:
  DataType element_type_;
  int64_t rank_;
  std::vector<int64_t> shape_;
  std::vector<DeviceId> device_ids_;
  std::vector<SplitSpec> split_specs_;
  std::vector<int64_t> ordered_split_dims_;
  std::vector<std::unique_ptr<PostOp>> post_ops_;
  /// multi-stage offsets for single split only tensor (including 0-rank or
  /// unranked tensor) Empty if it is single stage
  std::vector<int64_t> single_split_only_stage_offsets_;
  bool is_shape_or_size_tensor_;
  std::vector<SplitSpec> shape_split_specs_;
  SplitSpec size_split_spec_;

  bool isRanked() const { return rank_ >= 0; }

  bool isScalar() const { return isRanked() && rank_ == 0; }

  bool isDevicesAssigned() const { return device_ids_.size() > 0; }

  bool isAnyDimSplit() const {
    for (auto& split_spec : getSplitSpecs()) {
      if (split_spec.getType() == SplitSpec::SplitType::RATIO ||
          split_spec.getType() == SplitSpec::SplitType::SIZE) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Get device placements of the `split_num`-th split at `dim` specified
   * by `split_spec`
   *
   * @param dim
   * @param split_spec
   * @param split_num
   * @return std::set<DeviceId>
   */
  std::set<DeviceId> getDeviceSetBySpec(int64_t dim,
                                        const SplitSpec& split_spec,
                                        size_t split_num) const;

  /**
   * @brief Check if the sharding property maybe annotates a shape tensor that
   * meets the following conditions:
   * 1. 1D rank
   * 2. Shape is not dynamic (shape[0] >= 0)
   * 2. Integer type
   * 3. Single split only
   *
   * @return true
   * @return false
   */
  bool maybeShapeTensor() const;

  /**
   * @brief Check if the sharding property maybe annotates a size tensor that
   * meets the following conditions:
   * 1. 0D rank (implying single split only)
   * 2. Integer type
   *
   * @return true
   * @return false
   */
  bool maybeSizeTensor() const;
};

using ShardingPropertyRef = Ref<ShardingProperty>;
using ShardingPropertyRefVec = std::vector<ShardingPropertyRef>;

}  // namespace as
