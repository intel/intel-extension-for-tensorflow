/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef ITEX_CORE_UTILS_ONEDNN_ONEDNN_LAYOUT_UTIL_H_
#define ITEX_CORE_UTILS_ONEDNN_ONEDNN_LAYOUT_UTIL_H_

#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"
#ifndef INTEL_CPU_ONLY
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif  // INTEL_CPU_ONLY

#include "itex/core/utils/logging.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/strcat.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {

// Valid LLGA id should be a non-negative integer. When an OneDnnShape object
// has invalid llga id, it is a meta tensor for ITEX block tensor. Currently,
// ITEX block layout and LLGA share the same data structure "OneDnnShape"
constexpr int INVALID_LLGA_ID = -1;

// TODO(itex): Create another class for LLGA meta tensor.
class OneDnnShape {
 private:
  typedef struct OneDnnShapeData {
    // Flag to indicate if the tensor is an OneDnn tensor or not
    bool is_onednn_tensor_ = false;
    OneDnnTensorFormat tf_data_format_ = OneDnnTensorFormat::FORMAT_INVALID;
    // OneDnn layout
    dnnl::memory::desc md_;
    // TF dimension corresponding to this OneDnn dimension
    dnnl_dims_t map_;
    // TODO(itex): For Tensorflow, oneDNN Graph shape and stride are actually
    // the same thing, and we could merge them together. TF don't have
    // additional stride information for Tensor. shape for OneDnn Graph logical
    // tensor
    dnnl_graph_dims_t shape_;
    // stride for OneDnn Graph logical tensor
    dnnl_graph_dims_t stride_;
    // layout_id for OneDnn Graph logical tensor
    int64_t layout_id_ = INVALID_LLGA_ID;
  } OneDnnShapeData;
  OneDnnShapeData data_;

 public:
  OneDnnShape() {
    for (size_t i = 0; i < sizeof(data_.shape_) / sizeof(data_.shape_[0]);
         ++i) {
      data_.shape_[i] = -1;
      data_.stride_[i] = -1;
    }
  }
  ~OneDnnShape() = default;

  // Equality function for OneDnnShape objects
  // @return true if both are equal; false otherwise.
  bool operator==(const OneDnnShape& other) const;

  inline const bool IsOneDnnTensor() const {
    return (data_.is_onednn_tensor_ && data_.layout_id_ == INVALID_LLGA_ID);
  }
  inline const bool IsLLGATensor() const {
    return (data_.is_onednn_tensor_ &&
            (data_.layout_id_ != INVALID_LLGA_ID || data_.stride_[0] != -1));
  }
  inline void SetOneDnnTensor(bool is_onednn_tensor) {
    data_.is_onednn_tensor_ = is_onednn_tensor;
  }

  // Returns and dnnl::memory::dims object that contains the sizes of this
  // OneDnnShape object.
  inline dnnl::memory::dims GetSizesAsOneDnnDims() const {
    ITEX_CHECK_EQ(data_.is_onednn_tensor_, true);
    return data_.md_.dims();
  }

  // Get DataType
  inline dnnl::memory::data_type GetElemType() const {
    return data_.md_.data_type();
  }

  // Return TensorShape that describes the Tensorflow shape of the tensor
  // represented by this OneDnnShape.
  TensorShape GetTfShape() const;

  inline void SetOneDnnLayout(const dnnl::memory::desc& md) { data_.md_ = md; }

  // Get memory desc for OneDnn layout
  inline const dnnl::memory::desc GetOneDnnLayout() const { return data_.md_; }

  // Get memory desc for TF layout, only used in onednntotf op
  const dnnl::memory::desc GetTfLayout() const;

  // Set TfDataFormat and map_
  void SetTfDataFormat(OneDnnTensorFormat tf_data_format);

  inline OneDnnTensorFormat GetTfDataFormat() const {
    return data_.tf_data_format_;
  }

  // Get the OneDnn dim axis from Tf dim axis
  // e.g. OneDnn dim sequence {N, C, H, W}, Tf dim sequence {N, H, W, C}
  // In this case, TfDimIdx(3) = 1
  inline size_t TfDimIdx(int index) const { return data_.map_[index]; }

  // Check whether the sequence of OneDnn and Tf dims are the same, used for
  // debugging
  inline bool IsDimAligned() {
    for (size_t i = 0; i < sizeof(data_.map_) / sizeof(data_.map_[0]); ++i) {
      if (i != static_cast<size_t>(data_.map_[i])) return false;
    }
    return true;
  }

  // Save OneDnnShape data to meta tensor
  void SerializeOneDnnShape(unsigned char* buf, size_t buf_size) const;

  // Load OneDnnShape data to meta tensor
  void DeSerializeOneDnnShape(const unsigned char* buf, size_t buf_size);

  // Get Size of OneDnnShapeData, it is used to allocate buffer for meta tensor
  inline size_t GetSerializeBufferSize() const {
    return sizeof(OneDnnShapeData);
  }

  // Set shape of logical tensor.
  inline void SetShape(dnnl::graph::logical_tensor::dims_t shape) {
    for (size_t i = 0; i < shape.size(); i++) data_.shape_[i] = shape[i];
  }

  // Get shape of logical tensor.
  inline const dnnl::graph::logical_tensor::dims_t GetShape() {
    dnnl::graph::logical_tensor::dims_t retVal;
    for (int i = 0; i < DNNL_GRAPH_MAX_NDIMS; i++)
      if (data_.shape_[i] != -1) retVal.push_back(data_.shape_[i]);
    return retVal;
  }

  // Set stride of logical tensor.
  inline void SetStride(dnnl::graph::logical_tensor::dims_t stride) {
    for (int i = 0; i < stride.size(); i++) data_.stride_[i] = stride[i];
  }

  // Get stride of logical tensor.
  inline const dnnl::graph::logical_tensor::dims_t GetStride() {
    dnnl::graph::logical_tensor::dims_t retVal;
    for (int i = 0; i < DNNL_GRAPH_MAX_NDIMS; i++)
      if (data_.stride_[i] != -1) retVal.push_back(data_.stride_[i]);
    return retVal;
  }

  inline void SetLayoutId(int64_t layout_id) { data_.layout_id_ = layout_id; }

  inline const int64_t GetLayoutId() { return data_.layout_id_; }

  void* Raw() { return static_cast<void*>(&data_); }

 private:
  // Set the data_.map_ with data format information. We can use `map_`
  // to know the relationship between OneDnn dims (data_.size_) and actual TF
  // tensor dims
  void SetTfDimOrder(OneDnnTensorFormat format);
};

// Get input onednnshape by metatensor
// Don't change the OneDnnShape loads from the meta tensor
// TODO(itex): change the API to
// const OneDnnShape* GetOneDnnShape(OpKernelContext* ctext, int n);
void GetOneDnnShape(OpKernelContext* ctext, int n, OneDnnShape* onednn_shape);

// Allocate output meta tensor, and save onednnshape data
void AllocateMetaData(OpKernelContext* ctext, int dst_index,
                      const OneDnnShape& onednn_shape);

// Try to forward input to ouput meta tenosr.
void ForwardMetaData(OpKernelContext* ctext, int src_index, int dst_index,
                     const OneDnnShape& onednn_shape);

// Allocate output data tensor and meta tensor, save onednnshape data
void AllocateOutputSetOneDnnShape(OpKernelContext* ctext, int dst_index,
                                  Tensor** output, const TensorShape& tf_shape,
                                  const OneDnnShape& onednn_shape);

// Try to forward input to ouput meta tenosr. If failed, allocate output
// meta tensor, and save onednnshape data. is_forward_success indicates whether
// forward success or not.
void ForwardOrAllocateOutputSetOneDnnShape(OpKernelContext* ctext,
                                           int src_index, int dst_index,
                                           Tensor** output,
                                           const TensorShape& tf_shape,
                                           const OneDnnShape& onednn_shape,
                                           int* is_forward_success = nullptr);

// Similar to `ForwardOrAllocateOutputSetOneDnnShape`, the difference is try to
// forward data tensor and always allocate meta tensor. This function is only
// used in conv + bias + add + relu fusion, since there is u8/s8 cast and the
// meta tensor may not be forwarded
void ForwardOrAllocateOutputDataOnlySetOneDnnShape(
    OpKernelContext* ctext, int src_index, int dst_index, Tensor** output,
    const TensorShape& tf_shape, const OneDnnShape& onednn_shape,
    int* is_forward_success = nullptr);

// Set TF and OneDNN shape for dst according to `is_onednn`:
//   1) true:  TF shape is set to 1D, and OneDNN shape is set with params.
//   2) false: TF shape keeps unchanged, and OneDNN shape is unset.
inline void SetOutputTensorShape(const dnnl::memory::desc& dst_md,
                                 OneDnnTensorFormat format,
                                 TensorShape* tf_shape,
                                 OneDnnShape* onednn_shape, bool is_onednn) {
  onednn_shape->SetOneDnnTensor(is_onednn);

  if (is_onednn) {
    onednn_shape->SetOneDnnLayout(dst_md);
    onednn_shape->SetTfDataFormat(format);

    // Set TF shape to 1D with full size.
    TensorShape new_tf_shape;
    new_tf_shape.AddDim(dst_md.get_size() /
                        dnnl::memory::data_type_size(dst_md.data_type()));
    *tf_shape = new_tf_shape;
  }
}

// Check whether the layout is blocked directly from its attribute
inline bool IsBlockedMd(const dnnl::memory::desc& md) {
  return md.data.format_desc.blocking.inner_nblks != 0;
}

inline int GetTensorMetaDataIndex(int n, int num_inputs) {
  return n + num_inputs / 2;
}

inline bool IsInputSame(OpKernelContext* ctx, int index,
                        std::vector<int64> shape, OneDnnShape onednn_shape) {
  if (!ctx->is_input_same(index, shape)) return false;
  void* data =
      ctx->tensor_data(GetTensorMetaDataIndex(index, ctx->num_inputs()));
  OneDnnShape others;
  others.DeSerializeOneDnnShape(static_cast<uint8*>(data),
                                onednn_shape.GetSerializeBufferSize());
  return onednn_shape == others;
}

}  // namespace itex

#endif  // ITEX_CORE_UTILS_ONEDNN_ONEDNN_LAYOUT_UTIL_H_
