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

#include <assert.h>

#include <vector>

#include "xpuautoshard/common/ref_base.h"

namespace as {

class DimItem {
 public:
  DimItem(bool is_input, int64_t num, int64_t dim)
      : is_input_(is_input), num_(num), dim_(dim) {}

  DimItem() : DimItem(true, -1, -1) {}

  bool isInput() const { return is_input_; }

  int64_t getNum() const { return num_; }

  int64_t getDim() const { return dim_; }

 private:
  bool is_input_;
  int64_t num_;
  int64_t dim_;
};

enum class DimGroupType {
  IDENTICAL,
  CONTRACTING,
  BROADCASTING,
  WINDOWED,
};

enum ContractionType { SUM, MEAN, ARGMAX, L2, Max, Min, Prod };

class DimGroup {
 public:
  static DimGroup create(const std::vector<DimItem>& items) {
    assert(items.size() > 0);
    auto group = DimGroup(items);
    return group;
  }

  static DimGroup createContracting(const std::vector<DimItem>& items,
                                    int64_t output_num,
                                    ContractionType contraction_type) {
    DimGroup group = create(items);
    group.contraction_output_num_ = output_num;
    group.contraction_type_ = contraction_type;
    assert(group.getType() == DimGroupType::CONTRACTING);
    return group;
  }

  DimGroupType getType() const {
    // TODO(itex): handle DimGroupType::WINDOWED
    bool has_input = false;
    bool has_output = false;
    for (auto item : dim_items_) {
      if (item.isInput()) {
        has_input = true;
      } else {
        has_output = true;
      }
    }
    if (has_input && has_output) {
      return DimGroupType::IDENTICAL;
    } else if (has_input) {
      return DimGroupType::CONTRACTING;
    } else {
      return DimGroupType::BROADCASTING;
    }
  }
  std::vector<DimItem>::const_iterator begin() const {
    return dim_items_.begin();
  }
  std::vector<DimItem>::const_iterator end() const { return dim_items_.end(); }
  std::vector<DimItem>::iterator begin() { return dim_items_.begin(); }
  std::vector<DimItem>::iterator end() { return dim_items_.end(); }
  size_t size() const { return dim_items_.size(); }
  int64_t getContractionOutputNum() const { return contraction_output_num_; }
  ContractionType getContractionType() const { return contraction_type_; }

 private:
  explicit DimGroup(const std::vector<DimItem>& items)
      : dim_items_(items),
        contraction_output_num_(0),
        contraction_type_(ContractionType::SUM) {}

  std::vector<DimItem> dim_items_;
  int64_t contraction_output_num_;
  ContractionType contraction_type_;
};

}  // namespace as
