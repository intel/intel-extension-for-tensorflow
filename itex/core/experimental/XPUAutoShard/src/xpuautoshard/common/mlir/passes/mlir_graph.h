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
#include <utility>
#include <vector>

#include "xpuautoshard/common/graph.h"
#include "xpuautoshard/common/mlir/dialect.h"
#include "xpuautoshard/common/op_desc.h"

namespace mlir {
namespace hs {

using as::OpDesc;
using as::ValueDesc;

class MLIRValueDesc : public ValueDesc {
 public:
  explicit MLIRValueDesc(Value v) : v_(v) {}

  int64_t getRank() const override {
    if (auto ranked_type = v_.getType().dyn_cast<RankedTensorType>()) {
      return ranked_type.getRank();
    } else {
      return as::UNRANKED;
    }
  }

  as::DataType getElementType() const override {
    if (auto ranked_type = v_.getType().dyn_cast<RankedTensorType>()) {
      if (ranked_type.getElementType().isa<Float32Type>()) {
        return as::DataType::FLOAT32;
      } else if (ranked_type.getElementType().isa<BFloat16Type>()) {
        return as::DataType::BFLOAT16;
      } else if (ranked_type.getElementType().isa<Float16Type>()) {
        return as::DataType::FLOAT16;
      } else if (ranked_type.getElementType().isa<Float64Type>()) {
        return as::DataType::FLOAT64;
      } else if (ranked_type.getElementType().isa<IntegerType>()) {
        return as::DataType::INTEGER;
      }
    }
    return as::DataType::UNKNOWN;
  }

  bool isConcreteDims() const override {
    if (auto ranked_type = v_.getType().dyn_cast<RankedTensorType>()) {
      for (unsigned dim = 0; dim < ranked_type.getRank(); dim++) {
        if (ranked_type.isDynamicDim(dim)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  bool isDynamicDim(int64_t dim) const override {
    auto ranked_type = v_.getType().dyn_cast<RankedTensorType>();
    assert(ranked_type && "Expecting ranked tensor for dim info");
    return ranked_type.isDynamicDim(dim);
  }

  int64_t getDimSize(int64_t dim) const override {
    auto ranked_type = v_.getType().dyn_cast<RankedTensorType>();
    assert(ranked_type && "Expecting ranked tensor for dim info");
    return ranked_type.getShape()[dim];
  }

  std::vector<int64_t> getConstVecInt64() const override;

 private:
  Value v_;
};

class MLIROpDesc : public OpDesc {
 public:
  explicit MLIROpDesc(Operation* op)
      : op_(op), op_name_(op->getName().getStringRef().str()) {
    for (auto&& operand : op->getOperands()) {
      operands_.push_back(MLIRValueDesc(operand));
    }
    for (auto&& result : op->getResults()) {
      results_.push_back(MLIRValueDesc(result));
    }
  }

  const std::string& getName() const override { return op_name_; }

  ValueDesc& getOperand(unsigned idx) override { return operands_[idx]; }

  const ValueDesc& getOperand(unsigned idx) const override {
    return operands_[idx];
  }

  size_t getNumOperands() const override { return operands_.size(); }

  ValueDesc& getResult(unsigned idx) override { return results_[idx]; }

  const ValueDesc& getResult(unsigned idx) const override {
    return results_[idx];
  }

  size_t getNumResults() const override { return results_.size(); }

  bool hasAttr(const std::string& attr_name) const override {
    return op_->hasAttr(attr_name);
  }

  bool getAttrBool(const std::string& attr_name) const override {
    auto bool_attr = op_->getAttrOfType<BoolAttr>(attr_name);
    assert(bool_attr && "Cannot find op attribute of bool type");
    return bool_attr.getValue();
  }

  int64_t getAttrInt64(const std::string& attr_name) const override {
    auto int_attr = op_->getAttrOfType<IntegerAttr>(attr_name);
    assert(int_attr && "Cannot find op attribute of integer type");
    return int_attr.getInt();
  }

  std::string getAttrString(const std::string& attr_name) const override {
    auto str_attr = op_->getAttrOfType<StringAttr>(attr_name);
    assert(str_attr && "Cannot find op attribute of string type");
    return str_attr.getValue().str();
  }

  std::vector<int64_t> getAttrVecInt64(
      const std::string& attr_name) const override;

 private:
  Operation* op_;
  std::string op_name_;
  std::vector<MLIRValueDesc> operands_;
  std::vector<MLIRValueDesc> results_;
};

class MLIRGraph : public as::Graph {
 public:
  explicit MLIRGraph(Operation* root_op) : root_op_(root_op) {}
  Operation* getRoot() { return root_op_; }

  as::BreadthFirstGraphIterRangeRef getBreadthFirstIterRange() override;

 private:
  class BreadthFirstIterRangeImpl : public as::BreadthFirstGraphIterRange {
   public:
    explicit BreadthFirstIterRangeImpl(std::vector<Operation*>&& bf_ordered_vec)
        : bf_ordered_vec_(std::move(bf_ordered_vec)) {}
    as::BreadthFirstGraphIterator begin() override;
    as::BreadthFirstGraphIterator end() override;

   private:
    std::vector<Operation*> bf_ordered_vec_;
  };

  class BreadFirstIterImpl : public as::BreadthFirstGraphIterator::Impl {
   public:
    explicit BreadFirstIterImpl(const std::vector<Operation*>::iterator& iter)
        : iter_(iter) {}
    explicit BreadFirstIterImpl(std::vector<Operation*>::iterator&& iter)
        : iter_(std::move(iter)) {}
    as::OpDescRef operator*() override;
    const BreadFirstIterImpl& operator++() override {
      iter_++;
      return *this;
    }
    bool operator!=(const Impl& rhs) const override {
      return iter_ != dynamic_cast<const BreadFirstIterImpl&>(rhs).iter_;
    }

   private:
    std::vector<Operation*>::iterator iter_;
  };

  Operation* root_op_;
};

using MLIRGraphRef = as::Ref<MLIRGraph>;

}  // namespace hs
}  // namespace mlir
