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

#include <exception>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "xpuautoshard/common/ref_base.h"

namespace as {

constexpr int64_t UNRANKED = -1;
constexpr int64_t DYNAMIC_DIM_SIZE = -1;

enum DataType {
  UNKNOWN = 0,
  FLOAT64,
  FLOAT32,
  FLOAT16,
  BFLOAT16,
  INTEGER  // all integer types including bool, TODO: support int8
};

class ValueDesc {
 public:
  virtual ~ValueDesc() = default;
  /**
   * @brief Get the rank of the operand or -1 if the operand is unranked
   * or not of tensor type.
   *
   * @return int64_t
   */
  virtual int64_t getRank() const = 0;

  virtual bool isRanked() const { return getRank() >= 0; }

  /**
   * @brief Get the Element Type of a tensor type
   *
   * @return DataType
   */
  virtual DataType getElementType() const = 0;

  /**
   * @brief Check if the tensor element of the operand is of integer type
   *
   * @return true
   * @return false
   */
  virtual bool isElementIntegerType() const {
    return getElementType() == DataType::INTEGER;
  }

  /**
   * @brief Check if the given `dim` is of dynamic size
   *
   * @param dim
   * @return true
   * @return false
   */
  virtual bool isDynamicDim(int64_t dim) const = 0;

  /**
   * @brief Check if the value has concrete size on all dims.
   *
   * @return true ranked and all dims are concrete
   * @return false unranked or any of the dim is dynamic
   */
  virtual bool isConcreteDims() const = 0;

  /**
   * @brief Return the size of the given `dim`
   *
   * @param dim
   * @return int64_t
   */
  virtual int64_t getDimSize(int64_t dim) const = 0;

  /**
   * @brief Get the number of elements of the value if it is a tensor
   *
   * @return int64_t total number of elements if it is a tensor, otherwise 0.
   */
  virtual int64_t getNumElements() const {
    if (!isRanked()) {
      return 0;
    }
    int64_t num_elems = 1;
    for (size_t i = 0; i < getRank(); i++) {
      num_elems *= getDimSize(i);
    }
    return num_elems;
  }

  /**
   * @brief Get the constant operand. Empty vector returned if not a constant.
   *
   * @tparam T
   * @return const std::vector<T>&
   */
  virtual std::vector<int64_t> getConstVecInt64() const = 0;
};

using ValueDescRef = Ref<ValueDesc>;

/**
 * @brief Describes the statically known meta information of a high-level DL
 * operation that takes tensors (or some primitive or user-defined data types)
 * as input and output. It also carries named attributes with constant values.
 *
 */
class OpDesc {
 public:
  virtual ~OpDesc() = default;
  /**
   * @brief Get the op name
   *
   * @return std::string
   */
  virtual const std::string& getName() const = 0;

  /**
   * @brief Get the Operand object at given idx
   *
   * @param idx
   * @return const ValueDesc&
   */
  virtual ValueDesc& getOperand(unsigned idx) = 0;

  /**
   * @brief Get the Operand object at given idx
   *
   * @param idx
   * @return const ValueDesc&
   */
  virtual const ValueDesc& getOperand(unsigned idx) const = 0;

  /**
   * @brief Get the Num Operands
   *
   * @return size_t
   */
  virtual size_t getNumOperands() const = 0;

  /**
   * @brief Get the Result object at given idx
   *
   * @param idx
   * @return ValueDesc&
   */
  virtual ValueDesc& getResult(unsigned idx) = 0;

  /**
   * @brief Get the Result object at given idx
   *
   * @param idx
   * @return ValueDesc&
   */
  virtual const ValueDesc& getResult(unsigned idx) const = 0;

  /**
   * @brief Get the Num Results
   *
   * @return size_t
   */
  virtual size_t getNumResults() const = 0;

  /**
   * @brief Check whether a named attribute is defined by the op.
   *
   * @param attr_name
   * @return true
   * @return false
   */
  virtual bool hasAttr(const std::string& attr_name) const = 0;

  /**
   * @brief Get bool attribute value
   *
   * @param attr_name
   * @return true
   * @return false
   */
  virtual bool getAttrBool(const std::string& attr_name) const = 0;

  /**
   * @brief Get integer attribute value
   *
   * @param attr_name
   * @return int64_t
   */
  virtual int64_t getAttrInt64(const std::string& attr_name) const = 0;

  /**
   * @brief Get string attribute value
   *
   * @param attr_name
   * @return std::string
   */
  virtual std::string getAttrString(const std::string& attr_name) const = 0;

  /**
   * @brief Get integer vector attribute value
   *
   * @param attr_name
   * @return std::vector<int64_t>
   */
  virtual std::vector<int64_t> getAttrVecInt64(
      const std::string& attr_name) const = 0;
};

using OpDescRef = Ref<OpDesc>;

}  // namespace as
