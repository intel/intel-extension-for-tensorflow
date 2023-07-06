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

#include "xpuautoshard/common/hsp_inference/dim_group.h"
#include "xpuautoshard/common/sharding_property.h"

namespace as {

class HspException : public std::exception {
 public:
  explicit HspException(const std::string& msg) : msg_(msg) {}

 protected:
  const std::string msg_;
};

class HspUnimplementedException : public HspException {
 public:
  HspUnimplementedException(const std::string& msg, const ShardingProperty& hsp)
      : HspException(msg), hsp_(hsp) {}

  explicit HspUnimplementedException(const ShardingProperty& hsp)
      : HspUnimplementedException("", hsp) {}

 private:
  ShardingProperty hsp_;
};

class HspMismatchException : public HspException {
 public:
  HspMismatchException(const std::string& msg, const ShardingProperty& lhs,
                       const ShardingProperty& rhs)
      : HspException(msg), lhs_(lhs), rhs_(rhs) {}

  HspMismatchException(const ShardingProperty& lhs, const ShardingProperty& rhs)
      : HspMismatchException("", lhs, rhs) {}

 protected:
  ShardingProperty lhs_;
  ShardingProperty rhs_;
};

class SplitSpecMismatchException : public HspException {
 public:
  SplitSpecMismatchException(const std::string& msg,
                             const ShardingProperty& lhs,
                             const DimItem& lhs_dim_item,
                             const ShardingProperty& rhs,
                             const DimItem& rhs_dim_item)
      : HspException(msg),
        lhs_(lhs),
        lhs_dim_item_(lhs_dim_item),
        rhs_(rhs),
        rhs_dim_item_(rhs_dim_item) {}

  SplitSpecMismatchException(const ShardingProperty& lhs,
                             const DimItem& lhs_dim_item,
                             const ShardingProperty& rhs,
                             const DimItem& rhs_dim_item)
      : SplitSpecMismatchException("", lhs, lhs_dim_item, rhs, rhs_dim_item) {}

 protected:
  ShardingProperty lhs_;
  DimItem lhs_dim_item_;
  ShardingProperty rhs_;
  DimItem rhs_dim_item_;
};

class SplitAtException : public HspException {
 public:
  SplitAtException(const std::string& msg, const ShardingProperty& prop,
                   const DimItem& dim_item, const SplitSpec& split_spec)
      : HspException(msg),
        prop_(prop),
        dim_item_(dim_item),
        split_spec_(split_spec) {}

  SplitAtException(const ShardingProperty& prop, const DimItem& dim_item,
                   const SplitSpec& split_spec)
      : SplitAtException("", prop, dim_item, split_spec) {}

 protected:
  ShardingProperty prop_;
  DimItem dim_item_;
  SplitSpec split_spec_;
};

class SplitSingleAtException : public SplitAtException {
 public:
  SplitSingleAtException(const std::string& msg, const ShardingProperty& prop,
                         const DimItem& dim_item)
      : SplitAtException(msg, prop, dim_item,
                         SplitSpec::buildSingleSplit(prop, dim_item.getDim())) {
  }

  SplitSingleAtException(const ShardingProperty& prop, const DimItem& dim_item)
      : SplitAtException("SplitSingleAt", prop, dim_item,
                         SplitSpec::buildSingleSplit(prop, dim_item.getDim())) {
  }
};

class HspGroupingConflictException : public HspException {
 public:
  HspGroupingConflictException(const std::string& msg,
                               const ShardingPropertyRefVec& props)
      : HspException(msg), props_(props) {}

  explicit HspGroupingConflictException(const ShardingPropertyRefVec& props)
      : HspGroupingConflictException("HspGroupingConflict", props) {}

 private:
  ShardingPropertyRefVec props_;
};

}  // namespace as
