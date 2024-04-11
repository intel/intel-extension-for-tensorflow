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

#ifndef ITEX_BUILD_JAX
#include "itex/core/utils/plugin_tensor.h"

#include <algorithm>
#include <vector>

#include "itex/core/utils/gtl/inlined_vector.h"
#include "itex/core/utils/numeric_types.h"
#include "itex/core/utils/stringpiece.h"
#include "itex/core/utils/tensor_coding.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/type_traits.h"
#ifdef USING_NEXTPLUGGABLE_DEVICE
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"
#endif

namespace itex {
namespace {

// A set of helper functions depending on T.
template <typename T>
struct Helper {
  // By default, we assume T is a simple type (float, int32, etc.)
  static_assert(is_simple_type<T>::value, "T is not a simple type.");
  typedef protobuf::RepeatedField<T> RepeatedFieldType;

  // Encoder of simple type T to a string.  We do a copy.
  template <typename Destination>
  static void Encode(TensorBuffer* in, int64_t n, Destination* out) {
    port::AssignRefCounted(StringPiece(in->base<const char>(), sizeof(T) * n),
                           in, out);
  }

  static int64_t TotalBytes(TensorBuffer* in, int64_t n) {
    return sizeof(T) * n;
  }
};

// Helper specialization for string (the only non-simple type we
// support).
template <>
struct Helper<tstring> {
  // Proto message uses RepeatedFieldType to hold repeated T.
  typedef protobuf::RepeatedPtrField<string> RepeatedFieldType;

  // Encodes "n" elements of type string stored in "in" into Cord
  // "out", which is usually the TensorProto::tensor_content.
  template <typename Destination>
  static void Encode(TensorBuffer* in, int64_t n, Destination* out) {
    port::EncodeStringList(in->base<const tstring>(), n, out);
  }

  // Returns the estimated memory usage of "n" elements of type T
  // stored in buffer "in".
  static int64_t TotalBytes(TensorBuffer* in, int n) {
    int64_t tot = sizeof(tstring) * n;
    const tstring* p = in->base<const tstring>();
    for (int i = 0; i < n; ++i, ++p) tot += p->size();
    return tot;
  }
};

// TODO(itex): when support ResourceHandle and Variant, please add two
// struct. template <> struct Helper<ResourceHandle> {
//   // Proto message uses RepeatedFieldType to hold repeated T.
//   typedef protobuf::RepeatedPtrField<string> RepeatedFieldType;
//
//   // Encodes "n" elements of type ResourceHandle stored in "in" into
//   destination
//   // "out", which is usually the TensorProto::tensor_content.
//   template <typename Destination>
//   static void Encode(TensorBuffer* in, int64_t n, Destination* out) {
//     EncodeResourceHandleList(in->base<const ResourceHandle>(), n,
//                              port::NewStringListEncoder(out));
//   }
//
//
//   // Returns the estimated memory usage of "n" elements of type T
//   // stored in buffer "in".
//   static int64_t TotalBytes(TensorBuffer* in, int n) {
//     return n * sizeof(ResourceHandle);
//   }
// };
//
// template <>
// struct Helper<Variant> {
//   // Encodes "n" elements of type Variant stored in "in" into destination
//   // "out", which is usually the TensorProto::tensor_content.
//   template <typename Destination>
//   static void Encode(TensorBuffer* in, int64_t n, Destination* out) {
//     EncodeVariantList(in->base<const Variant>(), n,
//                       port::NewStringListEncoder(out));
//   }
//
//   // Returns the estimated memory usage of "n" elements of type T
//   // stored in buffer "in".
//   static int64_t TotalBytes(TensorBuffer* in, int n) {
//     return n * sizeof(Variant);
//   }
// };

}  // namespace

#ifdef USING_NEXTPLUGGABLE_DEVICE
void* tensor_get_raw_data(TF_Tensor* tf_tensor) {
  void* data_ptr = TF_TensorData(tf_tensor);
  if (data_ptr == nullptr) return nullptr;
  uintptr_t value = reinterpret_cast<uintptr_t>(data_ptr);

  if (value & kTag) {
    TF_Status* tf_status = TF_NewStatus();
    PJRT_Buffer* pjrt_c_buffer = TF_GetPjRtCBuffer(tf_tensor, tf_status);
    TF_DeleteStatus(tf_status);
    return ITEXOpaqueDataPointerFromPjRtBuffer(pjrt_c_buffer);
  } else {
    return data_ptr;
  }
}

bool pointer_is_pjrt_tensor(TF_Tensor* tf_tensor) {
  uintptr_t value = reinterpret_cast<uintptr_t>(TF_TensorData(tf_tensor));
  if (value & kTag) {
    return true;
  } else {
    return false;
  }
}

void create_pjrt_buffer_to_tensor(TF_OpKernelContext* tf_ctx,
                                  TF_Tensor* tf_tensor,
                                  const TensorShape& shape, DataType dtype) {
  if (pointer_is_pjrt_tensor(tf_tensor)) {
    TF_Status* tf_status = TF_NewStatus();
    PJRT_Buffer* pjrt_c_buffer = TF_GetPjRtCBuffer(tf_tensor, tf_status);
    if (pjrt_c_buffer == nullptr) {
      int device_id = TF_GetDeviceId(tf_ctx);
      PJRT_Client* pjrt_c_client = TF_GetPjRtCClient(DEVICE_XPU, tf_status);

      int rank = shape.dims();
      std::vector<int64_t> dimensions(rank);
      for (int d = 0; d < rank; ++d) {
        dimensions[d] = shape.dim_size(d);
      }
      size_t size = shape.num_elements() * DataTypeSize(dtype);

      ITEXNpdConfig& npdConfig = ITEXNpdConfig::getNpdConfig();
      if (npdConfig.isXlaAutoJitEnabled()) {
        std::vector<int64_t> layout(rank);
        std::iota(layout.rbegin(), layout.rend(), 0);
        TF_CreatePjRtBuffer(
            tf_tensor,
            ITEXCreateSEPjRtBuffer(device_id, DataTypeString(dtype), dimensions,
                                   layout, pjrt_c_client),
            "XPU", tf_status);
      } else {
        TF_CreatePjRtBuffer(
            tf_tensor,
            ITEXCreatePjRtBuffer(device_id, DataTypeString(dtype), &dimensions,
                                 size, pjrt_c_client),
            "XPU", tf_status);
      }
      TF_DeleteStatus(tf_status);
    }
  }
}
#endif  // USING_NEXTPLUGGABLE_DEVICE

void Tensor::CheckTypeAndIsAligned(DataType expected_dtype) const {
  ITEX_CHECK_EQ(dtype(), expected_dtype)
      << " " << DataTypeString(expected_dtype) << " expected, got "
      << DataTypeString(dtype());
  ITEX_CHECK(IsAligned()) << "ptr = " << buf_;
}

void Tensor::CheckType(DataType expected_dtype) const {
  ITEX_CHECK_EQ(dtype(), expected_dtype)
      << " " << DataTypeString(expected_dtype) << " expected, got "
      << DataTypeString(dtype());
}

void Tensor::CheckIsAlignedAndSingleElement() const {
  ITEX_CHECK(IsAligned()) << "Aligned and single element";
  ITEX_CHECK_EQ(1, NumElements()) << "Must have a one element tensor";
}

size_t Tensor::TotalBytes() const {
  return NumElements() * DataTypeSize(dtype());
}

size_t Tensor::AllocatedBytes() const { return TotalBytes(); }

Tensor::Tensor(DataType type, const TensorShape& shape, TF_Tensor* buf)
#ifdef USING_NEXTPLUGGABLE_DEVICE
    : shape_(shape), buf_(buf), npdConfig_(ITEXNpdConfig::getNpdConfig()) {
#else
    : shape_(shape), buf_(buf) {
#endif
  shape_.set_data_type(type);
  if (!buf_) {
    ITEX_LOG(ERROR)
        << "When create a new tensor, buf must be a non-null pointer!";
  }
}

Tensor::Tensor(TF_Tensor* buf)
#ifdef USING_NEXTPLUGGABLE_DEVICE
    : npdConfig_(ITEXNpdConfig::getNpdConfig()) {
#else
{
#endif
  buf_ = buf;
  TensorShape shape;
  int num_dim = TF_NumDims(buf_);
  for (int i = 0; i < num_dim; i++) {
    shape.AddDim(TF_Dim(buf_, i));
  }
  shape_ = shape;
  shape_.set_data_type(static_cast<DataType>(TF_TensorType(buf)));
}

Tensor::Tensor(DataType type, const TensorShape& shape)
#ifdef USING_NEXTPLUGGABLE_DEVICE
    : shape_(shape), buf_(nullptr), npdConfig_(ITEXNpdConfig::getNpdConfig()) {
#else
    : shape_(shape), buf_(nullptr) {
#endif
  shape_.set_data_type(type);

  gtl::InlinedVector<int64, 4> dims(shape_.dims(), 0);

  for (int i = 0; i < shape_.dims(); ++i) {
    dims[i] = shape_.dim_size(i);
  }

  buf_ = TF_AllocateTensor(static_cast<TF_DataType>(type), dims.data(),
                           shape_.dims(), TotalBytes());
}

gtl::InlinedVector<int64, 4> Tensor::ComputeFlatInnerDims(
    gtl::ArraySlice<int64> orig, int64 num_out_dims) {
  gtl::InlinedVector<int64, 4> out_dims(num_out_dims, 0);
  int64 offset = static_cast<int64>(orig.size()) - num_out_dims;
  for (int64 out_dim = num_out_dims - 1; out_dim >= 0; --out_dim) {
    const int64 in_dim = out_dim + offset;
    out_dims[out_dim] = in_dim < 0 ? 1 : orig[in_dim];
  }
  for (int64 in_dim = 0; in_dim < offset; ++in_dim) {
    out_dims[0] *= orig[in_dim];
  }
  return out_dims;
}

gtl::InlinedVector<int64, 4> Tensor::ComputeFlatOuterDims(
    gtl::ArraySlice<int64> orig, int64 num_out_dims) {
  gtl::InlinedVector<int64, 4> out_dims(num_out_dims, 0);
  for (int64 out_dim = 0; out_dim <= num_out_dims - 1; ++out_dim) {
    out_dims[out_dim] =
        out_dim >= static_cast<int64>(orig.size()) ? 1 : orig[out_dim];
  }
  for (int64 in_dim = num_out_dims; in_dim < static_cast<int64>(orig.size());
       ++in_dim) {
    out_dims[num_out_dims - 1] *= orig[in_dim];
  }
  return out_dims;
}

bool Tensor::SharesBufferWith(const Tensor& other) {
  char* start = reinterpret_cast<char*>(data());
  char* end = start + NumElements() * DataTypeSize(dtype());

  char* other_start = reinterpret_cast<char*>(other.data());
  char* other_end =
      other_start + other.NumElements() + DataTypeSize(other.dtype());

  if (start < other_start && end > other_end) {
    return true;
  }
  return false;
}

string Tensor::DebugString(int num_values) const {
  return strings::StrCat("Tensor<type: ", DataTypeString(dtype()),
                         " shape: ", shape().DebugString(),
                         " values: ", SummarizeValue(num_values), ">");
}

bool Tensor::RefCountIsOne() {
  TF_Tensor* tmp = TF_TensorMaybeMove(buf_);
  if (tmp) {
    return true;
  }
  return false;
}

StringPiece Tensor::tensor_data() const {
  if (GetTFTensor() == nullptr)
    return StringPiece();  // Don't die for empty tensors
  return StringPiece(static_cast<char*>(data()), TotalBytes());
}

Status MakeShape(const Tensor& shape, TensorShape* out) {
  if (!TensorShapeUtils::IsVector(shape.shape())) {
    return errors::InvalidArgument(
        "shape must be a vector of {int32,int64}, got shape ",
        shape.shape().DebugString());
  }
  if (shape.dtype() == DataType::DT_INT32) {
    auto vec = shape.flat<int32>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else if (shape.dtype() == DataType::DT_INT64) {
    auto vec = shape.flat<int64>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else {
    return errors::InvalidArgument("shape must be a vector of {int32,int64}.");
  }
}

bool Tensor::FromProto(const TensorProto& proto) {
  TF_Buffer* buffer = TF_NewBuffer();
  Status status = MessageToBuffer(proto, buffer);
  if (!status.ok()) {
    TF_DeleteBuffer(buffer);
    return false;
  }

  TensorShape shape(proto.tensor_shape());
  shape_ = shape;
  set_dtype(proto.dtype());

  if (!buf_) {
    gtl::InlinedVector<int64, 4> dims(shape_.dims(), 0);

    for (int i = 0; i < shape_.dims(); ++i) {
      dims[i] = shape_.dim_size(i);
    }

    buf_ = TF_AllocateTensor(static_cast<TF_DataType>(proto.dtype()),
                             dims.data(), shape_.dims(), TotalBytes());
    ITEX_CHECK_NOTNULL(buf_);
  }

  TF_Status* tf_status = TF_NewStatus();
  TF_TensorFromProto(buffer, buf_, tf_status);

  status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  TF_DeleteBuffer(buffer);
  if (!status.ok()) return false;
  return true;
}

// The macro CASES() expands to a switch statement conditioned on
// TYPE_ENUM. Each case expands the STMTS after a typedef for T.
#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)               \
  case DataTypeToEnum<TYPE>::value: {   \
    typedef TF_ATTRIBUTE_UNUSED TYPE T; \
    STMTS;                              \
    break;                              \
  }

// After support ResourceHandle and Variant, please add two case.
// CASE(ResourceHandle, SINGLE_ARG(STMTS))
// CASE(Variant, SINGLE_ARG(STMTS)
#define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT) \
  switch (TYPE_ENUM) {                                         \
    CASE(float, SINGLE_ARG(STMTS))                             \
    CASE(double, SINGLE_ARG(STMTS))                            \
    CASE(int32, SINGLE_ARG(STMTS))                             \
    CASE(uint8, SINGLE_ARG(STMTS))                             \
    CASE(uint16, SINGLE_ARG(STMTS))                            \
    CASE(uint32, SINGLE_ARG(STMTS))                            \
    CASE(uint64, SINGLE_ARG(STMTS))                            \
    CASE(int16, SINGLE_ARG(STMTS))                             \
    CASE(int8, SINGLE_ARG(STMTS))                              \
    CASE(tstring, SINGLE_ARG(STMTS))                           \
    CASE(complex64, SINGLE_ARG(STMTS))                         \
    CASE(complex128, SINGLE_ARG(STMTS))                        \
    CASE(int64_t, SINGLE_ARG(STMTS))                           \
    CASE(bool, SINGLE_ARG(STMTS))                              \
    CASE(Eigen::bfloat16, SINGLE_ARG(STMTS))                   \
    CASE(Eigen::half, SINGLE_ARG(STMTS))                       \
    case DT_INVALID:                                           \
      INVALID;                                                 \
      break;                                                   \
    default:                                                   \
      DEFAULT;                                                 \
      break;                                                   \
  }

#define CASES(TYPE_ENUM, STMTS)                                           \
  CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, ITEX_LOG(FATAL) << "Type not set"; \
                     , ITEX_LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)

void Tensor::AsProtoTensorContent(TensorProto* proto) const {
  proto->Clear();
  proto->set_dtype(dtype());
  shape_.AsProto(proto->mutable_tensor_shape());
  if (buf_) {
    TensorBuffer* tmp_buf_ = nullptr;
#ifndef USING_NEXTPLUGGABLE_DEVICE
    tmp_buf_ =
        new itex::TensorBuffer(reinterpret_cast<void*>(TF_TensorData(buf_)));
#else
    ITEXNpdConfig& npdConfig = ITEXNpdConfig::getNpdConfig();
    if (npdConfig.IfEnableNextPluggableDevice()) {
      tmp_buf_ = new itex::TensorBuffer(
          reinterpret_cast<void*>(tensor_get_raw_data(buf_)));
    } else {
      tmp_buf_ =
          new itex::TensorBuffer(reinterpret_cast<void*>(TF_TensorData(buf_)));
    }
#endif
    CASES(dtype(), Helper<T>::Encode(tmp_buf_, shape_.num_elements(),
                                     proto->mutable_tensor_content()));
    tmp_buf_->Unref();
  }
}

namespace {

// StrCat and StrAppend don't support Eigen::half directly at the moment, and
// we would like to keep them compatible with their absl counterparts, for ease
// of migration. We could rely on errors::internal::PrepareForStrCat() but the
// logic is so simple we can just replicate it here, where it is close to its
// usage and easy to change later. And there's the extra benefit of not
// accessing an 'internal' namespace.
inline const strings::AlphaNum& PrintOneElement(const strings::AlphaNum& a,
                                                bool print_v2) {
  return a;
}

inline float PrintOneElement(const Eigen::half& h, bool print_v2) {
  return static_cast<float>(h);
}

inline float PrintOneElement(Eigen::bfloat16 f, bool print_v2) {
  return static_cast<float>(f);
}

// Print from left dim to right dim recursively.
template <typename T>
void PrintOneDim(int dim_index, const gtl::InlinedVector<int64, 4>& shape,
                 int64 limit, int shape_size, const T* data, int64* data_index,
                 string* result) {
  if (*data_index >= limit) return;
  int64 element_count = shape[dim_index];
  // We have reached the right-most dimension of the tensor.
  if (dim_index == shape_size - 1) {
    for (int64 i = 0; i < element_count; i++) {
      if (*data_index >= limit) {
        // If not enough elements has been printed, append "...".
        if (dim_index != 0) {
          strings::StrAppend(result, "...");
        }
        return;
      }
      if (i > 0) strings::StrAppend(result, " ");
      strings::StrAppend(result, PrintOneElement(data[(*data_index)++], false));
    }
    return;
  }
  // Loop every element of one dim.
  for (int64 i = 0; i < element_count; i++) {
    bool flag = false;
    if (*data_index < limit) {
      strings::StrAppend(result, "[");
      flag = true;
    }
    // As for each element, print the sub-dim.
    PrintOneDim(dim_index + 1, shape, limit, shape_size, data, data_index,
                result);
    if (*data_index < limit || flag) {
      strings::StrAppend(result, "]");
      flag = false;
    }
  }
}

// Appends the spacing between elements for a given dim onto a result string
void PrintDimSpacing(int dim_index, int num_dims, string* result) {
  if (dim_index == num_dims - 1) {
    strings::StrAppend(result, " ");
    return;
  }
  for (int j = 0; j < num_dims - dim_index - 1; j++) {
    strings::StrAppend(result, "\n");
  }
  for (int j = 0; j <= dim_index; j++) {
    strings::StrAppend(result, " ");
  }
}

// Print from left dim to right dim recursively.
template <typename T>
void PrintOneDimV2(int dim_index, const gtl::InlinedVector<int64, 4>& shape,
                   int64 num_elts_at_ends, int num_dims, const T* data,
                   int64 data_index, string* result) {
  // We have recursed beyond all the dimensions into a single element
  // of the tensor.
  if (dim_index == num_dims) {
    strings::StrAppend(result, PrintOneElement(data[data_index], true));
    return;
  }

  strings::StrAppend(result, "[");
  int64 element_count = shape[dim_index];
  int64 start_of_end =
      std::max(num_elts_at_ends, element_count - num_elts_at_ends);

  // Loop every element of one dim.
  int64 elements_per_iter = 1;
  for (int i = dim_index + 1; i < num_dims; i++) {
    elements_per_iter *= shape[i];
  }
  for (int64 i = 0; (i < num_elts_at_ends) && (i < element_count); i++) {
    if (i > 0) {
      PrintDimSpacing(dim_index, num_dims, result);
    }

    // As for each element, print the sub-dim.
    PrintOneDimV2(dim_index + 1, shape, num_elts_at_ends, num_dims, data,
                  data_index + elements_per_iter * i, result);
  }
  if (element_count > 2 * num_elts_at_ends) {
    PrintDimSpacing(dim_index, num_dims, result);
    strings::StrAppend(result, "...");
  }
  for (int64 i = start_of_end; i < element_count; i++) {
    // As for each element, print the sub-dim.
    PrintDimSpacing(dim_index, num_dims, result);
    PrintOneDimV2(dim_index + 1, shape, num_elts_at_ends, num_dims, data,
                  data_index + elements_per_iter * i, result);
  }

  strings::StrAppend(result, "]");
}

template <typename T>
string SummarizeArray(int64 limit, int64 num_elts,
                      const TensorShape& tensor_shape, const char* data,
                      const bool print_v2) {
  string ret;
  const T* array = reinterpret_cast<const T*>(data);

  const gtl::InlinedVector<int64, 4> shape = tensor_shape.dim_sizes();
  if (shape.empty()) {
    for (int64 i = 0; i < limit; ++i) {
      if (i > 0) strings::StrAppend(&ret, " ");
      strings::StrAppend(&ret, PrintOneElement(array[i], print_v2));
    }
    if (num_elts > limit) strings::StrAppend(&ret, "...");
    return ret;
  }
  if (print_v2) {
    const int num_dims = tensor_shape.dims();
    PrintOneDimV2(0, shape, limit, num_dims, array, 0, &ret);
  } else {
    int64 data_index = 0;
    const int shape_size = tensor_shape.dims();
    PrintOneDim(0, shape, limit, shape_size, array, &data_index, &ret);

    if (num_elts > limit) strings::StrAppend(&ret, "...");
  }

  return ret;
}
}  // namespace

string Tensor::DeviceSafeDebugString() const {
  return strings::StrCat("Tensor<type: ", DataTypeString(dtype()),
                         " shape: ", shape().DebugString(), ">");
}

string Tensor::SummarizeValue(int64 max_entries, bool print_v2) const {
  const int64 num_elts = NumElements();
  if (max_entries < 0) {
    max_entries = num_elts;
  }
  size_t limit = std::min(max_entries, num_elts);
  if ((limit > 0) && (buf_ == nullptr)) {
    return strings::StrCat("uninitialized Tensor of ", num_elts,
                           " elements of type ", dtype());
  }
  const char* data = limit > 0 ? tensor_data().data() : nullptr;
  switch (dtype()) {
    case DT_BFLOAT16:
      return SummarizeArray<Eigen::bfloat16>(limit, num_elts, shape_, data,
                                             print_v2);
      break;
    case DT_HALF:
      return SummarizeArray<Eigen::half>(limit, num_elts, shape_, data,
                                         print_v2);
      break;
    case DT_FLOAT:
      return SummarizeArray<float>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_DOUBLE:
      return SummarizeArray<double>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_UINT32:
      return SummarizeArray<uint32>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_INT32:
      return SummarizeArray<int32>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_UINT8:
    case DT_QUINT8:
      return SummarizeArray<uint8>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_UINT16:
    case DT_QUINT16:
      return SummarizeArray<uint16>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_INT16:
    case DT_QINT16:
      return SummarizeArray<int16>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_INT8:
    case DT_QINT8:
      return SummarizeArray<int8>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_UINT64:
      return SummarizeArray<uint64>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_INT64:
      return SummarizeArray<int64>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_BOOL:
      return SummarizeArray<bool>(limit, num_elts, shape_, data, print_v2);
      break;
    case DT_STRING:
      return SummarizeArray<tstring>(limit, num_elts, shape_, data, print_v2);
      break;
    default: {
      // All irregular cases
      string ret;
      if (print_v2) {
        strings::StrAppend(&ret, "[");
      }
      for (size_t i = 0; i < limit; ++i) {
        if (i > 0) strings::StrAppend(&ret, " ");
        switch (dtype()) {
          case DT_VARIANT: {
            ITEX_VLOG(0) << "Variant type not supported.";
          } break;
          default:
            strings::StrAppend(&ret, "?");
        }
      }
      if (max_entries < num_elts) strings::StrAppend(&ret, "...");
      if (print_v2) {
        strings::StrAppend(&ret, "]");
      }
      return ret;
    }
  }
}
}  // namespace itex
#endif  // ITEX_BUILD_JAX
