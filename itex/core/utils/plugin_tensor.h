/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_UTILS_PLUGIN_TENSOR_H_
#define ITEX_CORE_UTILS_PLUGIN_TENSOR_H_

#include <string>
#include <utility>

#include "itex/core/utils/logging.h"
#include "itex/core/utils/refcount.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/tf_buffer.h"
#include "itex/core/utils/types.h"
#include "protos/tensor.pb.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
class TensorProto;
class TensorBuffer;

/// Interface to access the raw ref-counted data buffer.
class TensorBuffer : public core::RefCounted {
 public:
  explicit TensorBuffer(void* data_ptr) : data_(data_ptr) {}
  ~TensorBuffer() override {}

  /// \brief data() points to a memory region of size() bytes.
  ///
  /// NOTE(mrry): The `data()` method is not virtual for performance reasons.
  /// It can be called multiple times when the contents of a `Tensor` are
  /// accessed, and so making it non-virtual allows the body to be inlined.
  void* data() const { return data_; }

  //  /// \brief Size (in bytes) of the buffer.
  //  virtual size_t size() const = 0;

  //  /// \brief If this TensorBuffer is sub-buffer of another TensorBuffer,
  //  /// returns that TensorBuffer. Otherwise, returns this.
  //  virtual TensorBuffer* root_buffer() = 0;

  //  /// \brief Fills metadata about the allocation into the proto.
  //  virtual void FillAllocationDescription(
  //      AllocationDescription* proto) const = 0;

  //  virtual bool GetAllocatedBytes(size_t* out_bytes) const;

  /// \brief Helper method to reinterpret the buffer as an array of `T`.
  template <typename T>
  T* base() const {
    return reinterpret_cast<T*>(data());
  }

  //  /// \brief Whether this TensorBuffer owns the underlying memory.
  //  virtual bool OwnsMemory() const { return true; }

 private:
  void* const data_;
};

class Tensor {
 public:
  explicit Tensor(TF_Tensor* buf);

  Tensor() : buf_(nullptr) {}

  explicit Tensor(DataType type) : shape_(type), buf_(nullptr) {}

  // The Tensor buf_ will be allocated by `type` and `shape`.
  explicit Tensor(DataType type, const TensorShape& shape);

  explicit Tensor(DataType type, const TensorShape& shape, TF_Tensor* buf);

  // TODO(itex): Combine Tensor(Tensor&) and Tensor(Tensor&&)
  // into a single function
  Tensor(const Tensor& other) : shape_(other.shape_), buf_(nullptr) {
    TF_Status* tf_status = TF_NewStatus();
    const int64_t dims[1] = {1};
    buf_ = TF_AllocateTensor(static_cast<TF_DataType>(other.dtype()), dims, 1,
                             DataTypeSize(other.dtype()));
    set_dtype(other.dtype());
    TF_TensorBitcastFrom(other.buf_, static_cast<TF_DataType>(other.dtype()),
                         buf_, shape_.dim_sizes().data(), shape_.dims(),
                         tf_status);
    TF_DeleteStatus(tf_status);
  }

  Tensor(Tensor&& other) : shape_(std::move(other.shape_)), buf_(nullptr) {
    const int64_t dims[1] = {1};
    buf_ = TF_AllocateTensor(static_cast<TF_DataType>(other.dtype()), dims, 1,
                             DataTypeSize(other.dtype()));
    TF_Status* tf_status = TF_NewStatus();
    TF_TensorBitcastFrom(other.buf_, static_cast<TF_DataType>(other.dtype()),
                         buf_, shape_.dim_sizes().data(), shape_.dims(),
                         tf_status);
    TF_DeleteTensor(other.buf_);
    other.buf_ = nullptr;
    TF_DeleteStatus(tf_status);
  }

  Tensor& operator=(const Tensor& other) {
    CopyFromInternal(other, other.shape_);
    shape_ = other.shape_;
    return *this;
  }

  // TODO(itex): investigate whether buf_ can be not empty
  Tensor& operator=(Tensor&& t) {
    // Avoid self-assignment, since we might destroy our underlying buffer.
    if (this != &t) {
      CopyFromInternal(t, t.shape_);
      shape_ = std::move(t.shape_);
      TF_DeleteTensor(t.buf_);
      t.buf_ = nullptr;
    }
    return *this;
  }

  ~Tensor() {
    if (buf_ != nullptr) {
      TF_DeleteTensor(buf_);
      buf_ = nullptr;
    }
  }

  // Difference from Copyfrom, BitCastFrom can explicitly set the DataType
  // TODO(itex): Combine BitCastFrom and CopyFromInternal into single
  // function
  Status BitcastFrom(const Tensor& other, DataType dtype,
                     const TensorShape& shape) {
    ITEX_DCHECK_EQ(shape.num_elements(), other.NumElements());

    // This is a workaround because of `to` in TF_TensorBitcastFrom can't be
    // `nullptr`. Don't worry about the memory leak. It will be destructed out
    // of tensor life scope.
    if (!buf_) {
      const int64_t dims[1] = {1};
      buf_ = TF_AllocateTensor(static_cast<TF_DataType>(dtype), dims, 1,
                               DataTypeSize(dtype));
      ITEX_CHECK_NOTNULL(buf_);
    }

    // copy the shape and dtype.
    shape_ = shape;
    set_dtype(dtype);
    TF_Status* tf_status = TF_NewStatus();
    TF_TensorBitcastFrom(other.buf_, static_cast<TF_DataType>(dtype), buf_,
                         shape_.dim_sizes().data(), shape_.dims(), tf_status);

    Status status = StatusFromTF_Status(tf_status);
    TF_DeleteStatus(tf_status);
    return status;
  }

  bool FromProto(const TensorProto& proto);
  /// \brief Fills in `proto` with `*this` tensor's content.
  ///
  /// `AsProtoField()` fills in the repeated field for `proto.dtype()`, while
  /// `AsProtoTensorContent()` encodes the content in `proto.tensor_content()`
  /// in a compact form.
  //  void AsProtoField(TensorProto* proto);
  void AsProtoTensorContent(TensorProto* proto);

  DataType dtype() const { return shape_.data_type(); }

  const TensorShape& shape() const { return shape_; }
  const TF_Tensor* GetTFTensor() const { return buf_; }
  TF_Tensor* GetTFTensor() { return buf_; }

  int dims() const { return shape().dims(); }

  int64 dim_size(int d) const { return shape().dim_size(d); }

  int64 NumElements() const { return shape().num_elements(); }

  bool IsSameSize(const Tensor& b) const {
    return shape().IsSameSize(b.shape());
  }

  bool IsInitialized() const {
    return buf_ != nullptr && TF_TensorData(buf_) != nullptr;
  }

  size_t TotalBytes() const;

  size_t AllocatedBytes() const;

  bool IsAligned() const {
    if (buf_ != nullptr) {
      return TF_TensorIsAligned(buf_);
    }
    return true;
  }

  StringPiece tensor_data() const;
  void* data() const { return base<void>(); }

  void set_dtype(DataType t) { shape_.set_data_type(t); }

  void set_shape(const TensorShape& shape) {
    DataType dt = dtype();
    shape_ = shape;
    set_dtype(dt);
  }

  template <typename T>
  typename TTypes<T>::Vec vec() {
    return tensor<T, 1>();
  }

  template <typename T>
  typename TTypes<T>::Matrix matrix() {
    return tensor<T, 2>();
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor();

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// same size but a bitwise cast to the specified dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// NOTE: this is the same as `tensor()` except a bitcast is allowed.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor bit_casted_tensor();

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// last dimension elements converted into single elements of a larger type.
  ///
  /// For example, this is useful for kernels that can treat NCHW_VECT_C int8
  /// tensors as NCHW int32 tensors. The sizeof(T) should equal the size of
  /// the original element type * num elements in the original last dimension.
  /// NDIMS should be 1 less than the original number of dimensions.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor reinterpret_last_dimension();

  /// \brief Return the tensor data as an `Eigen::Tensor` of the data type and a
  /// specified shape.
  ///
  /// These methods allow you to access the data with the dimensions
  /// and sizes of your choice.  You do not need to know the number of
  /// dimensions of the Tensor to call them.  However, they `ITEX_CHECK` that
  /// the type matches and the dimensions requested creates an
  /// `Eigen::Tensor` with the same number of elements as the tensor.
  ///
  /// Example:
  ///
  /// ```c++
  ///
  ///     typedef float T;
  ///     Tensor my_ten(...built with Shape{planes: 4, rows: 3, cols: 5}...);
  ///     // 1D Eigen::Tensor, size 60:
  ///     auto flat = my_ten.flat<T>();
  ///     // 2D Eigen::Tensor 12 x 5:
  ///     auto inner = my_ten.flat_inner_dims<T>();
  ///     // 2D Eigen::Tensor 4 x 15:
  ///     auto outer = my_ten.shaped<T, 2>({4, 15});
  ///     // ITEX_CHECK fails, bad num elements:
  ///     auto outer = my_ten.shaped<T, 2>({4, 8});
  ///     // 3D Eigen::Tensor 6 x 5 x 2:
  ///     auto weird = my_ten.shaped<T, 3>({6, 5, 2});
  ///     // ITEX_CHECK fails, type mismatch:
  ///     auto bad   = my_ten.flat<int32>();
  ///
  /// ```
  template <typename T>
  typename TTypes<T>::Flat flat() {
    return shaped<T, 1>({NumElements()});
  }

  template <typename T>
  typename TTypes<T>::UnalignedFlat unaligned_flat() {
    return unaligned_shaped<T, 1>({NumElements()});
  }

  /// Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all
  /// Tensor dimensions but the last NDIMS-1 into the first dimension of the
  /// result. If NDIMS > dims() then leading dimensions of size 1 will be
  /// added to make the output rank NDIMS.
  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::Tensor flat_inner_dims();

  /// Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all
  /// Tensor dimensions but the first NDIMS-1 into the last dimension of the
  /// result. If NDIMS > dims() then trailing dimensions of size 1 will be
  /// added to make the output rank NDIMS.
  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::Tensor flat_outer_dims();

  /// Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing the
  /// first 'begin' Tensor dimensions into the first dimension of the result and
  /// the Tensor dimensions of the last dims() - 'begin' - NDIMS into the last
  /// dimension of the result. If 'begin' < 0 then the |'begin'| leading
  /// dimensions of size 1 will be added. If 'begin' + NDIMS > dims() then
  /// 'begin' + NDIMS - dims() trailing dimensions of size 1 will be added.
  template <typename T, size_t NDIMS = 3>
  typename TTypes<T, NDIMS>::Tensor flat_inner_outer_dims(int64 begin);

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(gtl::ArraySlice<int64> new_sizes);

  /// \brief Return the tensor data to an `Eigen::Tensor` with the new
  /// shape specified in `new_sizes` and cast to a new dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// The allowed bitcast is the only difference from `shaped()`.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor bit_casted_shaped(
      gtl::ArraySlice<int64> new_sizes);

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedTensor unaligned_shaped(
      gtl::ArraySlice<int64> new_sizes);

  /// \brief Return the Tensor data as a `TensorMap` of fixed size 1:
  /// `TensorMap<TensorFixedSize<T, 1>>`.

  /// Using `scalar()` allows the compiler to perform optimizations as
  /// the size of the tensor is known at compile time.
  template <typename T>
  typename TTypes<T>::Scalar scalar();

  /// Const versions of all the methods above.
  template <typename T>
  typename TTypes<T>::ConstVec vec() const {
    return tensor<T, 1>();
  }

  template <typename T>
  typename TTypes<T>::ConstMatrix matrix() const {
    return tensor<T, 2>();
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor tensor() const;

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// same size but a bitwise cast to the specified dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// NOTE: this is the same as `tensor()` except a bitcast is allowed.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor bit_casted_tensor() const;

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// last dimension elements converted into single elements of a larger type.
  ///
  /// For example, this is useful for kernels that can treat NCHW_VECT_C int8
  /// tensors as NCHW int32 tensors. The sizeof(T) should equal the size of
  /// the original element type * num elements in the original last dimension.
  /// NDIMS should be 1 less than the original number of dimensions.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor reinterpret_last_dimension() const;

  template <typename T>
  typename TTypes<T>::ConstFlat flat() const {
    return shaped<T, 1>({NumElements()});
  }

  template <typename T>
  typename TTypes<T>::UnalignedConstFlat unaligned_flat() const {
    return unaligned_shaped<T, 1>({NumElements()});
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor shaped(
      gtl::ArraySlice<int64> new_sizes) const;

  /// \brief Return the tensor data to an `Eigen::Tensor` with the new
  /// shape specified in `new_sizes` and cast to a new dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// The allowed bitcast is the only difference from `shaped()`.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor bit_casted_shaped(
      gtl::ArraySlice<int64> new_sizes) const;

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedConstTensor unaligned_shaped(
      gtl::ArraySlice<int64> new_sizes) const;

  template <typename T>
  typename TTypes<T>::ConstScalar scalar() const;

  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::ConstTensor flat_inner_dims() const;

  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::ConstTensor flat_outer_dims() const;

  template <typename T, size_t NDIMS = 3>
  typename TTypes<T, NDIMS>::ConstTensor flat_inner_outer_dims(
      int64 begin) const;

  /// A human-readable summary of the tensor suitable for debugging.
  // `num_values` is the number of actual data values in the tensor
  // included in the message. If the tensor might be resident in
  // GPU/TPU memory use DeviceSafeDebugString instead.
  std::string DebugString(int num_values) const;
  std::string DebugString() const { return DebugString(3); }

  /// Render the first `max_entries` values in `*this` into a string.
  std::string SummarizeValue(int64 max_entries, bool print_v2 = false) const;

  // Variant of DebugString() that should be used for possibly non-CPU tensors.
  // If the tensor is not resident on CPU, we can't read its values as
  // DebugString() does.
  std::string DeviceSafeDebugString() const;

  /// \brief Copy the tensor from other and reshape it.
  ///
  /// The current shape will be replaced with shape and
  /// type will be replaced with other.dtype().
  bool CopyFrom(const Tensor& other, const TensorShape& shape) {
    if (other.NumElements() != shape.num_elements()) return false;
    CopyFromInternal(other, shape);
    shape_ = shape;
    // Manually create shape doesn't have datatype information, we obtain the
    // type information from src tensor. TF-proper uses the same logic.
    shape_.set_data_type(other.dtype());
    return true;
  }

  bool SharesBufferWith(const Tensor& other);

  bool RefCountIsOne();

 private:
  void CheckType(DataType expected_dtype) const;
  void CheckTypeAndIsAligned(DataType expected_dtype) const;
  void CheckIsAlignedAndSingleElement() const;

  // TensorShape's InlineVector.
  static gtl::InlinedVector<int64, 4> ComputeFlatInnerDims(
      gtl::ArraySlice<int64> orig, int64 num_out_dims);
  static gtl::InlinedVector<int64, 4> ComputeFlatOuterDims(
      gtl::ArraySlice<int64> orig, int64 num_out_dims);

  template <size_t NDIMS>
  void FillDimsAndValidateCompatibleShape(
      gtl::ArraySlice<int64> new_sizes,
      Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const;

  template <typename T, size_t NDIMS>
  void FillDimsAndValidateCompatibleShape(
      gtl::ArraySlice<int64> new_sizes,
      Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const;

  template <typename T>
  T* base() const {
    return reinterpret_cast<T*>(TF_TensorData(buf_));
  }

  inline void CopyFromInternal(const Tensor& other, const TensorShape& shape) {
    ITEX_DCHECK_EQ(shape.num_elements(), other.NumElements());

    // This is a workaround because of `to` in TF_TensorBitcastFrom can't be
    // `nullptr`. Don't worry about the memory leak. It will be destructed out
    // of tensor life scope.
    if (!buf_) {
      const int64_t dims[1] = {1};
      buf_ = TF_AllocateTensor(static_cast<TF_DataType>(other.dtype()), dims, 1,
                               DataTypeSize(other.dtype()));
      ITEX_CHECK_NOTNULL(buf_);
    }

    // copy the dtype.
    DataType dtype = other.dtype();
    set_dtype(dtype);

    TF_Status* tf_status = TF_NewStatus();
    TF_TensorBitcastFrom(other.buf_, static_cast<TF_DataType>(dtype), buf_,
                         shape.dim_sizes().data(), shape.dims(), tf_status);
    Status s = StatusFromTF_Status(tf_status);
    ITEX_CHECK_EQ(Status::OK(), s);
    TF_DeleteStatus(tf_status);
  }

 private:
  TensorShape shape_;
  TF_Tensor* buf_;
};

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::tensor() {
  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  return typename TTypes<T, NDIMS>::Tensor(base<T>(),
                                           shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::tensor() const {
  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(),
                                                shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::bit_casted_tensor() {
  ITEX_CHECK(IsAligned());
  return typename TTypes<T, NDIMS>::Tensor(base<T>(),
                                           shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::bit_casted_tensor() const {
  ITEX_CHECK(IsAligned());
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(),
                                                shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::reinterpret_last_dimension() {
  if (NDIMS == dims()) {
    return tensor<T, NDIMS>();
  }
  ITEX_CHECK(IsAligned());
  ITEX_CHECK_EQ(NDIMS, dims() - 1);
  ITEX_CHECK_EQ(sizeof(T), shape_.dim_sizes()[NDIMS] * DataTypeSize(dtype()));
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; ++d) {
    dims[d] = shape_.dim_sizes()[d];
  }
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::reinterpret_last_dimension()
    const {
  if (NDIMS == dims()) {
    return tensor<T, NDIMS>();
  }
  ITEX_CHECK(IsAligned());
  ITEX_CHECK_EQ(NDIMS, dims() - 1);
  ITEX_CHECK_EQ(sizeof(T), shape_.dim_sizes()[NDIMS] * DataTypeSize(dtype()));
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; ++d) {
    dims[d] = shape_.dim_sizes()[d];
  }
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(), dims);
}

template <size_t NDIMS>
void Tensor::FillDimsAndValidateCompatibleShape(
    gtl::ArraySlice<int64> new_sizes,
    Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const {
  ITEX_CHECK_EQ(NDIMS, new_sizes.size());
  int64 new_num_elements = 1;
  for (size_t d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    (*dims)[d] = new_sizes[d];
  }
  ITEX_CHECK_EQ(new_num_elements, NumElements());
}

template <typename T, size_t NDIMS>
void Tensor::FillDimsAndValidateCompatibleShape(
    gtl::ArraySlice<int64> new_sizes,
    Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const {
  ITEX_CHECK_EQ(NDIMS, new_sizes.size());
  int64 new_num_elements = 1;
  for (size_t d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    (*dims)[d] = new_sizes[d];
  }
  const int element_size = DataTypeSize(BaseType(dtype()));
  if (element_size > 0) {
    ITEX_CHECK_EQ(new_num_elements * sizeof(T), NumElements() * element_size);
  } else {
    // DataTypeSize() returns 0 for some data types. In this case, assume that T
    // has the same size as the buffer type.
    // NOTE: If we can be sure that DataTypeSize() does not return 0 for all POD
    // types, then we should check DataTypeToEnum<T>::v() == dtype(). Or simply
    // check if `element_size > 0` to err when bit cast is attempted on Tensor
    // of unknown data type size.
    ITEX_CHECK_EQ(new_num_elements, NumElements());
  }
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::shaped(
    gtl::ArraySlice<int64> new_sizes) {
  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::bit_casted_shaped(
    gtl::ArraySlice<int64> new_sizes) {
  ITEX_CHECK(IsAligned());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape<T>(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64> new_sizes) {
  CheckType(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::UnalignedTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::shaped(
    gtl::ArraySlice<int64> new_sizes) const {
  CheckType(DataTypeToEnum<T>::v());
  ITEX_CHECK(IsAligned());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);

  return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::bit_casted_shaped(
    gtl::ArraySlice<int64> new_sizes) const {
  ITEX_CHECK(IsAligned());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape<T>(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedConstTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64> new_sizes) const {
  CheckType(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::UnalignedConstTensor(base<T>(), dims);
}

template <typename T>
typename TTypes<T>::Scalar Tensor::scalar() {
  static_assert(!std::is_same<T, std::string>::value,
                "std::string is no longer a scalar type, use itex::tstring");
  CheckIsAlignedAndSingleElement();
  return typename TTypes<T>::Scalar(base<T>());
}

template <typename T>
typename TTypes<T>::ConstScalar Tensor::scalar() const {
  static_assert(!std::is_same<T, std::string>::value,
                "std::string is no longer a scalar type, use itex::tstring");
  CheckIsAlignedAndSingleElement();
  return typename TTypes<T>::ConstScalar(base<T>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_inner_dims() {
  return shaped<T, NDIMS>(ComputeFlatInnerDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_outer_dims() {
  return shaped<T, NDIMS>(ComputeFlatOuterDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_inner_outer_dims(int64 begin) {
  gtl::InlinedVector<int64, 4> flat_outer =
      ComputeFlatOuterDims(shape_.dim_sizes(), begin + NDIMS);
  return shaped<T, NDIMS>(ComputeFlatInnerDims(flat_outer, NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_inner_dims() const {
  return shaped<T, NDIMS>(ComputeFlatInnerDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_outer_dims() const {
  return shaped<T, NDIMS>(ComputeFlatOuterDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_inner_outer_dims(
    int64 begin) const {
  gtl::InlinedVector<int64, 4> flat_outer =
      ComputeFlatOuterDims(shape_.dim_sizes(), begin + NDIMS);
  return shaped<T, NDIMS>(ComputeFlatInnerDims(flat_outer, NDIMS));
}
///////////////////////////////// old

// Make a TensorShape from the contents of shape_t. Shape_t must be a
// 1-dimensional tensor of type int32 or int64.
Status MakeShape(const Tensor& shape_t, TensorShape* out);

}  // namespace itex

#endif  // ITEX_CORE_UTILS_PLUGIN_TENSOR_H_
