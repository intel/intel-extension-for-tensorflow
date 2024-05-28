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

#include "itex/core/utils/onednn/onednn_layout_util.h"

#include <numeric>
#include <vector>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {

bool OneDnnShape::operator==(const OneDnnShape& other) const {
  if (this->IsOneDnnTensor() != other.IsOneDnnTensor()) {
    return false;
  }
  // If input tensors are in OneDnn layout, then we check for dimensions and
  // data_.sizes_.
  if (this->IsOneDnnTensor()) {
    return this->GetTfShape() == other.GetTfShape() &&
           this->GetOneDnnLayout() == other.GetOneDnnLayout();
  }
  // Both inputs are not OneDnn tensors.
  return true;
}

TensorShape OneDnnShape::GetTfShape() const {
  ITEX_CHECK_EQ(data_.is_onednn_tensor_, true);

  dnnl::memory::dims onednn_dims = GetSizesAsOneDnnDims();
  size_t dimension = onednn_dims.size();

  std::vector<int32> shape(dimension);
  for (size_t idx = 0; idx < dimension; ++idx) {
    shape[idx] = onednn_dims[TfDimIdx(idx)];
  }

  TensorShape ts;
  bool ret = TensorShapeUtils::MakeShape(shape, &ts).ok();
  ITEX_CHECK_EQ(ret, true);
  return ts;
}

void OneDnnShape::SetTfDataFormat(OneDnnTensorFormat tf_data_format) {
  data_.tf_data_format_ = tf_data_format;
  SetTfDimOrder(tf_data_format);
}

const dnnl::memory::desc OneDnnShape::GetTfLayout() const {
  dnnl::memory::dims dims = GetSizesAsOneDnnDims();
  dnnl::memory::data_type dt = GetElemType();

  auto format_tag = OneDnnTensorFormatToTag(data_.tf_data_format_);
  ITEX_DCHECK_NE(static_cast<int>(format_tag),
                 static_cast<int>(dnnl::memory::format_tag::undef));
  return dnnl::memory::desc(dims, dt, format_tag);
}

const dnnl::memory::format_tag OneDnnShape::GetFormatTag() const {
  auto format_tag = OneDnnTensorFormatToTag(data_.tf_data_format_);
  return format_tag;
}

void OneDnnShape::SetTfDimOrder(OneDnnTensorFormat format) {
  // Aparts from nchw/nhwc/ndhwc/ncdhw, dims sequence for normal block layout
  // is the same as the sequence for TF plain layout
  switch (format) {
    case OneDnnTensorFormat::FORMAT_X:
      std::iota(data_.map_, data_.map_ + 1, 0);
      return;
    case OneDnnTensorFormat::FORMAT_NC:
      std::iota(data_.map_, data_.map_ + 2, 0);
      return;
    case OneDnnTensorFormat::FORMAT_TNC:
      std::iota(data_.map_, data_.map_ + 3, 0);
      return;
    default:
      // Fall through.
      break;
  }

  TensorFormat data_format = OneDnnDataFormatToTFDataFormat(format);
  if (format == OneDnnTensorFormat::FORMAT_NHWC ||
      format == OneDnnTensorFormat::FORMAT_NCHW) {
    data_.map_[GetTensorDimIndex<2>(data_format, 'W')] = DimensionIndex::Dim_W;
    data_.map_[GetTensorDimIndex<2>(data_format, 'H')] = DimensionIndex::Dim_H;
    data_.map_[GetTensorDimIndex<2>(data_format, 'C')] = DimensionIndex::Dim_C;
    data_.map_[GetTensorDimIndex<2>(data_format, 'N')] = DimensionIndex::Dim_N;
  } else if (format == OneDnnTensorFormat::FORMAT_NDHWC ||
             format == OneDnnTensorFormat::FORMAT_NCDHW) {
    data_.map_[GetTensorDimIndex<3>(data_format, '0')] =
        DimensionIndex3D::Dim3d_D;
    data_.map_[GetTensorDimIndex<3>(data_format, '1')] =
        DimensionIndex3D::Dim3d_H;
    data_.map_[GetTensorDimIndex<3>(data_format, '2')] =
        DimensionIndex3D::Dim3d_W;
    data_.map_[GetTensorDimIndex<3>(data_format, 'C')] =
        DimensionIndex3D::Dim3d_C;
    data_.map_[GetTensorDimIndex<3>(data_format, 'N')] =
        DimensionIndex3D::Dim3d_N;
  }
}

void OneDnnShape::SerializeOneDnnShape(unsigned char* buf,
                                       size_t buf_size) const {
  ITEX_CHECK(buf_size >= GetSerializeBufferSize())
      << "Buffer size is too small to SerializeOneDnnShape";
  *reinterpret_cast<OneDnnShapeData*>(buf) = data_;
  std::copy_n(md_.data(), data_.md_size_, buf + sizeof(OneDnnShapeData));
}

void OneDnnShape::DeSerializeOneDnnShape(const unsigned char* buf,
                                         size_t buf_size) {
  // Make sure buffer holds at least data_.is_onednn_tensor_.
  ITEX_CHECK(buf_size >= sizeof(data_.is_onednn_tensor_))
      << "Buffer size is too small in DeSerializeOneDnnShape";

  const bool is_onednn_tensor_ = *reinterpret_cast<const bool*>(buf);
  if (is_onednn_tensor_) {  // If it is an OneDnn Tensor then read the rest
    ITEX_CHECK(buf_size >= GetSerializeBufferSize())
        << "Buffer size is too small in DeSerializeOneDnnShape";
    data_ = *reinterpret_cast<const OneDnnShapeData*>(buf);
    md_.resize(data_.md_size_);
    std::copy_n(buf + sizeof(OneDnnShapeData), data_.md_size_, md_.data());
  } else {
    data_.is_onednn_tensor_ = false;
    data_.md_size_ = 0;
    md_.clear();
  }
}

inline int GetTensorDataIndex(int n, int num_inputs) { return n; }

void GetOneDnnShape(OpKernelContext* ctext, int n, OneDnnShape* onednn_shape) {
  const Tensor& meta_input =
      ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()));

  onednn_shape->DeSerializeOneDnnShape(
      meta_input.flat<uint8>().data(),
      meta_input.flat<uint8>().size() * sizeof(uint8));
}

// Allocate output meta tensor, and save onednnshape data
void AllocateMetaData(OpKernelContext* ctext, int dst_index,
                      const OneDnnShape& onednn_shape) {
  Tensor* second_tensor = nullptr;
  TensorShape second_shape;
  second_shape.AddDim(onednn_shape.GetSerializeBufferSize());
  OP_REQUIRES_OK(ctext,
                 ctext->allocate_output(
                     GetTensorMetaDataIndex(dst_index, ctext->num_outputs()),
                     second_shape, &second_tensor));
  onednn_shape.SerializeOneDnnShape(
      second_tensor->flat<uint8>().data(),
      second_tensor->flat<uint8>().size() * sizeof(uint8));
}

// Try to forward input to ouput meta tenosr.
// Unlike the data tensor buffer, the meta tensor buffer will never be modified
// since it is created. We could consider this buffer is read-only. It is safe
// to reuse the original buffer.
void ForwardMetaData(OpKernelContext* ctext, int src_index, int dst_index,
                     const OneDnnShape& onednn_shape) {
  int src_meta_index = GetTensorMetaDataIndex(src_index, ctext->num_inputs());
  int dst_meta_index = GetTensorMetaDataIndex(dst_index, ctext->num_outputs());
  ctext->set_output(dst_meta_index, ctext->input(src_meta_index));
}

void AllocateOutputSetOneDnnShape(OpKernelContext* ctext, int dst_index,
                                  Tensor** output, const TensorShape& tf_shape,
                                  const OneDnnShape& onednn_shape) {
  OP_REQUIRES_OK(ctext, ctext->allocate_output(
                            GetTensorDataIndex(dst_index, ctext->num_outputs()),
                            tf_shape, output));
  AllocateMetaData(ctext, dst_index, onednn_shape);
}

void ForwardOrAllocateOutputSetOneDnnShape(OpKernelContext* ctext,
                                           int src_index, int dst_index,
                                           Tensor** output,
                                           const TensorShape& tf_shape,
                                           const OneDnnShape& onednn_shape,
                                           int* is_forward_success) {
  OP_REQUIRES_OK(
      ctext, ctext->forward_input_or_allocate_output(
                 {GetTensorDataIndex(src_index, ctext->num_inputs())},
                 GetTensorDataIndex(dst_index, ctext->num_outputs()), tf_shape,
                 output, is_forward_success));
  ForwardMetaData(ctext, src_index, dst_index, onednn_shape);
}

void ForwardOrAllocateOutputDataOnlySetOneDnnShape(
    OpKernelContext* ctext, int src_index, int dst_index, Tensor** output,
    const TensorShape& tf_shape, const OneDnnShape& onednn_shape,
    int* is_forward_success) {
  OP_REQUIRES_OK(
      ctext, ctext->forward_input_or_allocate_output(
                 {GetTensorDataIndex(src_index, ctext->num_inputs())},
                 GetTensorDataIndex(dst_index, ctext->num_outputs()), tf_shape,
                 output, is_forward_success));
  AllocateMetaData(ctext, dst_index, onednn_shape);
}
}  // namespace itex
