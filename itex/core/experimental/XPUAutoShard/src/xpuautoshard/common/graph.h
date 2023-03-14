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

#include <memory>

#include "xpuautoshard/common/op_desc.h"
#include "xpuautoshard/common/ref_base.h"

namespace as {

/**
 * @brief Breadth first iterator over graph ops
 *
 */
class BreadthFirstGraphIterator {
 public:
  class Impl {
   public:
    virtual ~Impl() = default;
    virtual OpDescRef operator*() = 0;
    virtual const Impl& operator++() = 0;
    virtual bool operator!=(const Impl& rhs) const = 0;
  };
  using ImplRef = Ref<Impl>;

  explicit BreadthFirstGraphIterator(ImplRef impl) : impl_(impl) {}
  OpDescRef operator*() { return *(*impl_); }

  const BreadthFirstGraphIterator& operator++() {
    impl_->operator++();
    return *this;
  }

  bool operator!=(const BreadthFirstGraphIterator& rhs) const {
    return *impl_ != *rhs.impl_;
  }

 private:
  ImplRef impl_;
};

/**
 * @brief A breadth-first iterator range over a graph
 *
 */
class BreadthFirstGraphIterRange {
 public:
  virtual ~BreadthFirstGraphIterRange() = default;
  virtual BreadthFirstGraphIterator begin() = 0;
  virtual BreadthFirstGraphIterator end() = 0;
};

using BreadthFirstGraphIterRangeRef = Ref<BreadthFirstGraphIterRange>;

/**
 * @brief An opaque representation of a framework graph
 *
 */
class Graph {
 public:
  virtual ~Graph() = default;
  /**
   * @brief Get the Breadth First Iter Range object on the graph
   *
   * @return BreadthFirstGraphIterRangeRef
   */
  virtual BreadthFirstGraphIterRangeRef getBreadthFirstIterRange() = 0;
};

using GraphRef = Ref<Graph>;

}  // namespace as
