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

#ifndef ITEX_CORE_UTILS_ONEDNN_ONEDNN_UTIL_H_
#define ITEX_CORE_UTILS_ONEDNN_ONEDNN_UTIL_H_

#include <dlfcn.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "dnnl.hpp"  // NOLINT(build/include_subdir)

#ifndef INTEL_CPU_ONLY
#include "dnnl_sycl.hpp"  // NOLINT(build/include_subdir)
#endif                    // INTEL_CPU_ONLY

#include "itex/core/utils/logging.h"
#include "itex/core/utils/onednn/mkl_threadpool.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/strcat.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/wrapper/itex_cpu_wrapper.h"

namespace itex {
static std::once_flag read_env_once_flag;
static bool enable_omp = true;
typedef dnnl_status_t (*dnnl_stream_create_internal)(dnnl_stream_t*,
                                                     dnnl_engine_t, void*);
static dnnl_stream_create_internal make_stream;

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

#ifndef INTEL_CPU_ONLY
const int MAX_NDIMS = 6;
#else
const int MAX_NDIMS = DNNL_MAX_NDIMS;
#endif

namespace dnnl_port {
// Link to the definition in onednn/src/common/bfloat16.hpp
struct bfloat16_t;
}  // namespace dnnl_port

#ifndef INTEL_CPU_ONLY
static dnnl::engine& FindOrCreateEngine(ITEX_GPUStream* stream) {
  static std::map<ITEX_GPUStream*, dnnl::engine> stream_engine_map;
  auto iter = stream_engine_map.find(stream);
  if (iter != stream_engine_map.end()) return iter->second;

  dnnl::engine engine;
  engine = dnnl::sycl_interop::make_engine(stream->get_device(),
                                           stream->get_context());
  return stream_engine_map
      .insert(std::pair<ITEX_GPUStream*, dnnl::engine>(stream, engine))
      .first->second;
}
#endif

// The dimensions order that DNNL internally uses for 2D activations
// [Batch, Channel, Height, Width] and
// for 2D filters [Out_Channel, In_Channel, Height, Width].
typedef enum {
  Dim_N = 0,
  Dim_C = 1,
  Dim_H = 2,
  Dim_W = 3,
  Dim_O = 0,
  Dim_I = 1
} DimensionIndex;

// The dimensions order that DNNL internally uses for 3D activations
// [Batch, Channel, Depth, Height, Width] and
// for 3D filters [Out_Channel, In_Channel, Depth, Height, Width].
typedef enum {
  Dim3d_N = 0,
  Dim3d_C = 1,
  Dim3d_D = 2,
  Dim3d_H = 3,
  Dim3d_W = 4,
  Dim3d_O = 0,
  Dim3d_I = 1
} DimensionIndex3D;

// In oneDNN, the format (ex. NCHW) used to initialize a memory descriptor
// (md) structure will no longer be recorded in its `format` field. Instead, it
// will be set to a canonical `blocked` format for every fully described md.
//
// Currently, we query this `format` field while mapping oneDNN's data format
// to TF's data format. Due to the above restriction, we will now get this data
// format information from TF's `data_format` attribute (i.e. via
// `TensorFormat`) for oneDNN.

//  1) FORMAT_INVALID: for error-checking (ex. unsupported format)
//  2) FORMAT_X, FORMAT_NC, FORMAT_TNC: to distinguish between DNNL tensors
//     based on their dimensions in operators such as Softmax, i.e.:
//        FORMAT_X   - 1D tensor
//        FORMAT_NC  - 2D tensor
//        FORMAT_TNC - 3D tensor
enum class OneDnnTensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
  FORMAT_NDHWC = 2,
  FORMAT_NCDHW = 3,
  FORMAT_X = 4,
  FORMAT_NC = 5,
  FORMAT_TNC = 6,
  FORMAT_INVALID = 7,
};

// Enum for the order of dimensions of a TF 2D filter with shape [filter_height,
// filter_width, in_channels, out_channels]
typedef enum {
  TF_2DFILTER_DIM_H = 0,
  TF_2DFILTER_DIM_W = 1,
  TF_2DFILTER_DIM_I = 2,
  TF_2DFILTER_DIM_O = 3
} TFFilterDims2d;

// Enum for the order of dimensions of a TF 3D filter with shape [filter_depth,
// filter_height, filter_width, in_channels, out_channels]
typedef enum {
  TF_3DFILTER_DIM_P = 0,
  TF_3DFILTER_DIM_H = 1,
  TF_3DFILTER_DIM_W = 2,
  TF_3DFILTER_DIM_I = 3,
  TF_3DFILTER_DIM_O = 4
} TFFilterDims3d;

// The dimensions order that oneDNN requires for the filter in a grouped
// convolution (2D only)
typedef enum {
  GROUP_FILTER_DIM_G = 0,
  GROUP_FILTER_DIM_O = 1,
  GROUP_FILTER_DIM_I = 2,
  GROUP_FILTER_DIM_H = 3,
  GROUP_FILTER_DIM_W = 4
} FilterGroupDims;

/// Return oneDNN data type (memory::data_type) for input type T
///
/// @input None
/// @return dnnl::memory::data_type corresponding to type T
template <typename T>
inline dnnl::memory::data_type OneDnnType();

/// Instantiation for float type. Add similar instantiations for other
/// type if needed.
template <>
inline dnnl::memory::data_type OneDnnType<float>() {
  return dnnl::memory::data_type::f32;
}

template <>
inline dnnl::memory::data_type OneDnnType<double>() {
  return dnnl::memory::data_type::f64;
}

template <>
inline dnnl::memory::data_type OneDnnType<Eigen::half>() {
  return dnnl::memory::data_type::f16;
}

template <>
inline dnnl::memory::data_type OneDnnType<quint8>() {
  return dnnl::memory::data_type::u8;
}

template <>
inline dnnl::memory::data_type OneDnnType<uint8>() {
  return dnnl::memory::data_type::u8;
}

template <>
inline dnnl::memory::data_type OneDnnType<qint8>() {
  return dnnl::memory::data_type::s8;
}

template <>
inline dnnl::memory::data_type OneDnnType<int8>() {
  return dnnl::memory::data_type::s8;
}

template <>
inline dnnl::memory::data_type OneDnnType<qint32>() {
  return dnnl::memory::data_type::s32;
}

template <>
inline dnnl::memory::data_type OneDnnType<Eigen::bfloat16>() {
  return dnnl::memory::data_type::bf16;
}

#ifndef ITEX_BUILD_JAX
template <typename Device>
inline dnnl::engine& CreateDnnlEngine(const OpKernelContext& ctx);

#ifndef INTEL_CPU_ONLY
template <>
inline dnnl::engine& CreateDnnlEngine<GPUDevice>(const OpKernelContext& ctx) {
  auto* ITEX_GPU_stream = ctx.GetDeviceStream();
  return FindOrCreateEngine(ITEX_GPU_stream);
}
#endif  // INTEL_CPU_ONLY

template <>
inline dnnl::engine& CreateDnnlEngine<CPUDevice>(const OpKernelContext& ctx) {
  // Right now ITEX doesn't own proper TF CPU device and NUMA info is
  // unavailable, so simply consider ITEX only have 1 CPU device.
  // TODO(itex): Check NUMA after integrating new CPU device.
  ITEX_CHECK(&(ctx.eigen_cpu_device()) == &(ctx.eigen_cpu_device_singleton()))
      << "Global oneDNN CPU engine mismatched with current context";
  static dnnl::engine cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
  return cpu_engine;
}

inline dnnl::stream CreateDnnlStream(const OpKernelContext& ctx,
                                     const dnnl::engine& engine,
                                     int num_thread = -1) {
#ifndef INTEL_CPU_ONLY
  // GPU
  ITEX_CHECK(engine.get_kind() == dnnl::engine::kind::gpu)
      << "Create oneDNN stream for unsupported engine.";
  auto* ITEX_GPU_stream = ctx.GetDeviceStream();
  return dnnl::sycl_interop::make_stream(engine, *ITEX_GPU_stream);
#else
#ifndef CC_BUILD
  // CPU and python build
  std::call_once(read_env_once_flag, []() {
    ITEX_CHECK_OK(
        itex::ReadBoolFromEnvVar("ITEX_OMP_THREADPOOL", true, &enable_omp));
    if (!enable_omp) {
      make_stream = reinterpret_cast<dnnl_stream_create_internal>(
          dlsym(onednn_handle, "dnnl_threadpool_interop_stream_create"));
    }
  });
  if (enable_omp) {
    ITEX_CHECK(engine.get_kind() == dnnl::engine::kind::cpu)
        << "Create oneDNN stream for unsupported engine.";
    return dnnl::stream(engine);
  } else {
    if (num_thread == 1) return dnnl::stream(engine);

    MklDnnThreadPool* eigen_tp = new MklDnnThreadPool(&ctx, num_thread);
    dnnl_stream_t c_stream;
    make_stream(&c_stream, engine.get(), eigen_tp);
    dnnl::stream tp_stream = dnnl::stream(c_stream);
    return tp_stream;
  }
#else
// CPU and C++ BUILD
#ifdef CC_THREADPOOL_BUILD
  // CPU and C++ BUILD with eigen thread pool
  if (num_thread == 1) return dnnl::stream(engine);
  MklDnnThreadPool* eigen_tp = new MklDnnThreadPool(&ctx, num_thread);
  dnnl::stream tp_stream =
      dnnl::stream(dnnl::threadpool_interop::make_stream(engine, eigen_tp));
  return tp_stream;
#else
  // CPU and C++ BUILD with OMP thread pool
  // Default path, always assume it's CPU engine.
  ITEX_CHECK(engine.get_kind() == dnnl::engine::kind::cpu)
      << "Create oneDNN stream for unsupported engine.";
  return dnnl::stream(engine);
#endif  // CC_THREADPOOL_BUILD
#endif  // CC_BUILD
#endif  // INTEL_CPU_ONLY
}

#endif  // ITEX_BUILD_JAX
inline dnnl::memory CreateDnnlMemory(const dnnl::memory::desc& md,
                                     const dnnl::engine& engine,
                                     void* data_handle = nullptr) {
#ifndef INTEL_CPU_ONLY
  if (engine.get_kind() == dnnl::engine::kind::gpu) {
    auto kind = dnnl::sycl_interop::memory_kind::usm;
    if (data_handle == nullptr)
      return dnnl::sycl_interop::make_memory(md, engine, kind,
                                             DNNL_MEMORY_ALLOCATE);
    else
      return dnnl::sycl_interop::make_memory(md, engine, kind, data_handle);
  }
#endif  // INTEL_CPU_ONLY

  // Default path, always assume it's CPU engine.
  ITEX_CHECK(engine.get_kind() == dnnl::engine::kind::cpu)
      << "Create oneDNN memory for unsupported engine.";
  if (data_handle == nullptr)
    return dnnl::memory(md, engine);
  else
    return dnnl::memory(md, engine, data_handle);
}

// Map OneDnnTensorFormat to oneDNN format tag
//
// @input: OneDnnTensorFormat i.e. TensorFlow data format
// @return: OneDNN's memory format tag corresponding to OneDnnTensorFormat.
//          Fails with an error if invalid data format.
inline dnnl::memory::format_tag OneDnnTensorFormatToTag(
    OneDnnTensorFormat format) {
  if (format == OneDnnTensorFormat::FORMAT_NHWC)
    return dnnl::memory::format_tag::nhwc;
  if (format == OneDnnTensorFormat::FORMAT_NCHW)
    return dnnl::memory::format_tag::nchw;
  if (format == OneDnnTensorFormat::FORMAT_NDHWC)
    return dnnl::memory::format_tag::ndhwc;
  if (format == OneDnnTensorFormat::FORMAT_NCDHW)
    return dnnl::memory::format_tag::ncdhw;
  if (format == OneDnnTensorFormat::FORMAT_X)
    return dnnl::memory::format_tag::x;
  if (format == OneDnnTensorFormat::FORMAT_NC)
    return dnnl::memory::format_tag::nc;
  if (format == OneDnnTensorFormat::FORMAT_TNC)
    return dnnl::memory::format_tag::tnc;
  return dnnl::memory::format_tag::undef;
}

/// Map TensorFlow data format into oneDNN data format. This is used in TF
/// kernels which have `data_format` attributes, such as Conv/BatchNorm/...
/// `TensorFormat` is original TF tensor attr, it's always NCHW or NHWC no
/// matter the rank is 4D or 5D.
///
/// @input: TensorFlow data format, Boolean to indicate whether it's 2D format
/// @return: OneDNN data format corresponding to TensorFlow data format;
///          Fails with an error if invalid data format.
inline OneDnnTensorFormat TFDataFormatToOneDnnDataFormat(TensorFormat format,
                                                         bool is_2d = true) {
  if (is_2d) {
    if (format == FORMAT_NHWC) return OneDnnTensorFormat::FORMAT_NHWC;
    if (format == FORMAT_NCHW) return OneDnnTensorFormat::FORMAT_NCHW;
  } else {
    if (format == FORMAT_NHWC) return OneDnnTensorFormat::FORMAT_NDHWC;
    if (format == FORMAT_NCHW) return OneDnnTensorFormat::FORMAT_NCDHW;
  }

  ITEX_CHECK_OK(Status(TF_INVALID_ARGUMENT, "Unsupported data format"));
  return OneDnnTensorFormat::FORMAT_INVALID;
}

/// Map oneDNN data format into TensorFlow data format
///
/// @input: OneDNN data format
/// @return: Tensorflow data format corresponding to oneDNN data format;
///          Fails with an error if invalid data format.
inline TensorFormat OneDnnDataFormatToTFDataFormat(OneDnnTensorFormat format) {
  if (format == OneDnnTensorFormat::FORMAT_NHWC ||
      format == OneDnnTensorFormat::FORMAT_NDHWC)
    return FORMAT_NHWC;
  if (format == OneDnnTensorFormat::FORMAT_NCHW ||
      format == OneDnnTensorFormat::FORMAT_NCDHW)
    return FORMAT_NCHW;
  ITEX_CHECK_OK(Status(TF_INVALID_ARGUMENT, "Unsupported data format"));

  // Return to prevent compiler warnings, otherwise ITEX_CHECK_OK will ensure
  // that we don't come here.
  return FORMAT_NHWC;
}

/// Map TensorShape object into dnnl::memory::dims required by oneDNN
///
/// This function will simply map input TensorShape into oneDNN dims
/// naively. So it will preserve the order of dimensions. E.g., if
/// input tensor is in NHWC format, then dims will be in NHWC format also.
///
/// @input TensorShape object in shape
/// @return dnnl::memory::dims corresponding to TensorShape
inline dnnl::memory::dims TFShapeToOneDnnDims(const TensorShape& shape) {
  if (shape.dims() == 0) {
    dnnl::memory::dims dims{shape.num_elements()};
    return dims;
  }
  dnnl::memory::dims dims(shape.dims());
  for (int d = 0; d < shape.dims(); ++d) {
    dims[d] = shape.dim_size(d);
  }
  return dims;
}

/// Map TensorShape object into dnnl::memory::dims in NC...(NCHW/NCDHW) format
/// since oneDnn has channel first logical dimension sequence requirement.
///
/// This function is a specific one than above function. It will map input
/// TensorShape into oneDNN dims in NCHW/NCDHW format. So it may not preserve
/// the order of dimensions. E.g., if input tensor is in NHWC format, then dims
/// will be in NCHW format, and not in NHWC format.
///
/// Commonly used in below scenarios:
/// 1) Create oneDNN primitive from TF tensor in kernel which has `data_format`
///    attr, such as Conv/BatchNorm/Pooling;
/// 2) Reorder TF/oneDNN tensors to same oneDNN format in kernel which has
///    multiply inputs, such as AddN/Concat;
///
/// @input TensorShape object in shape, tensor format, Boolean to indicate
///        whether it's 2D format
/// @return dnnl::memory::dims in oneDNN required NC...(NCHW/NCDHW) format
inline dnnl::memory::dims TFShapeToOneDnnDimsInNC(const TensorShape& shape,
                                                  TensorFormat format,
                                                  bool is_2d = true) {
  // Check validity of format.
  ITEX_DCHECK_NE(
      static_cast<int>(TFDataFormatToOneDnnDataFormat(format, is_2d)),
      static_cast<int>(OneDnnTensorFormat::FORMAT_INVALID));

  if (is_2d) {
    int n = shape.dim_size(GetTensorDimIndex(format, 'N'));
    int c = shape.dim_size(GetTensorDimIndex(format, 'C'));
    int h = shape.dim_size(GetTensorDimIndex(format, 'H'));
    int w = shape.dim_size(GetTensorDimIndex(format, 'W'));

    // oneDNN requires dimensions in NCHW format.
    return dnnl::memory::dims({n, c, h, w});
  } else {
    int n = shape.dim_size(GetTensorDimIndex<3>(format, 'N'));
    int c = shape.dim_size(GetTensorDimIndex<3>(format, 'C'));
    int d = shape.dim_size(GetTensorDimIndex<3>(format, '0'));
    int h = shape.dim_size(GetTensorDimIndex<3>(format, '1'));
    int w = shape.dim_size(GetTensorDimIndex<3>(format, '2'));

    // oneDNN requires dimensions in NCDHW format.
    return dnnl::memory::dims({n, c, d, h, w});
  }
}

/// Overloaded version of function TFShapeToOneDnnDimsInNC above.
/// Input parameters are self-explanatory.
inline dnnl::memory::dims OneDnnDimsInNC(const dnnl::memory::dims& in_dims,
                                         TensorFormat format,
                                         bool is_2d = true) {
  // Validate format.
  ITEX_DCHECK_NE(
      static_cast<int>(TFDataFormatToOneDnnDataFormat(format, is_2d)),
      static_cast<int>(OneDnnTensorFormat::FORMAT_INVALID));

  if (is_2d) {
    int n = in_dims[GetTensorDimIndex(format, 'N')];
    int c = in_dims[GetTensorDimIndex(format, 'C')];
    int h = in_dims[GetTensorDimIndex(format, 'H')];
    int w = in_dims[GetTensorDimIndex(format, 'W')];

    // OneDNN requires dimensions in NCHW format.
    return dnnl::memory::dims({n, c, h, w});
  } else {
    int n = in_dims[GetTensorDimIndex<3>(format, 'N')];
    int c = in_dims[GetTensorDimIndex<3>(format, 'C')];
    int d = in_dims[GetTensorDimIndex<3>(format, '0')];
    int h = in_dims[GetTensorDimIndex<3>(format, '1')];
    int w = in_dims[GetTensorDimIndex<3>(format, '2')];

    // OneDNN requires dimensions in NCDHW format.
    return dnnl::memory::dims({n, c, d, h, w});
  }
}

/// Map oneDNN dnnl::memory::dims object into TensorShape object.
///
/// This function will simply map input shape in OneDNN dnnl::memory::dims
/// format in Tensorflow's TensorShape object by preserving dimension order.
///
/// @input OneDNN dnnl::memory::dims object
/// @output TensorShape corresponding to dnnl::memory::dims
inline TensorShape OneDnnDimsToTFShape(const dnnl::memory::dims& dims) {
  std::vector<int32> shape(dims.size(), -1);
  for (size_t d = 0; d < dims.size(); d++) {
    shape[d] = dims[d];
  }

  TensorShape ret;
  ITEX_CHECK_EQ(TensorShapeUtils::MakeShape(shape, &ret).ok(), true);
  return ret;
}

/// Function to calculate strides given tensor shape in Tensorflow order
/// E.g., if dims_tf_order is {1, 2, 3, 4}, then as per Tensorflow convention,
/// dimension with size 1 is outermost dimension; while dimension with size 4 is
/// innermost dimension. So strides for this tensor would be {4 * 3 * 2,
/// 4 * 3, 4, 1}, i.e., {24, 12, 4, 1}.
///
/// @input Tensorflow shape in memory::dims type
/// @return memory::dims containing strides for the tensor.
inline dnnl::memory::dims CalculateTFStrides(
    const dnnl::memory::dims& dims_tf_order) {
  ITEX_CHECK_GT(dims_tf_order.size(), 0);
  dnnl::memory::dims strides(dims_tf_order.size(), 1);
  for (int d = strides.size() - 2; d >= 0; d--) {
    strides[d] = strides[d + 1] * dims_tf_order[d + 1];
  }
  return strides;
}

#ifndef ITEX_BUILD_JAX
template <typename T>
inline void* GetTensorBuffer(const Tensor* tensor) {
  ITEX_CHECK_NOTNULL(tensor);
  return const_cast<void*>(static_cast<const void*>(tensor->flat<T>().data()));
}

// Create memory desc with format tag, it is the equivalent way to create memory
// desc with strides in CreateBlockedMemDesc
template <typename T>
inline dnnl::memory::desc CreatePlainMemDescWithFormatTag(
    const dnnl::memory::dims& onednn_dims) {
  if (onednn_dims.size() > MAX_NDIMS)
    ITEX_LOG(FATAL) << "Max dims for current device is " << MAX_NDIMS;

  if (onednn_dims.size() == 1)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::a);
  else if (onednn_dims.size() == 2)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::ab);
  else if (onednn_dims.size() == 3)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abc);
  else if (onednn_dims.size() == 4)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcd);
  else if (onednn_dims.size() == 5)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcde);
  else if (onednn_dims.size() == 6)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdef);
  else if (onednn_dims.size() == 7)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefg);
  else if (onednn_dims.size() == 8)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefgh);
  else if (onednn_dims.size() == 9)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefghi);
  else if (onednn_dims.size() == 10)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefghij);
  else if (onednn_dims.size() == 11)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefghijk);
  else
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefghijkl);
}

// Reorder src memory to expected memory
void ReorderMemory(const OpKernelContext& context,
                   const dnnl::memory* src_memory, dnnl::memory* reorder_memory,
                   const dnnl::engine& onednn_engine);

// Weight cache is used to avoid weight reorder repetitively when target weight
// block md is different frome original weight plain md.
template <typename T>
class WeightCacheManager {
 public:
  WeightCacheManager() = default;
  ~WeightCacheManager() = default;

  bool IsEmpty() TF_LOCKS_EXCLUDED(mu_);

  // Cache the reordered weight buffer & weight md as persistent tensors.
  // Only one thread can execute this method at any given time.
  void SetCache(OpKernelContext* context,
                const dnnl::memory::desc& weight_original_md,
                const dnnl::memory::desc& weight_expected_md, void* weight_data,
                const dnnl::engine& onednn_engine) TF_LOCKS_EXCLUDED(mu_);

  // Get the cached weight buffer
  T* GetCache(OpKernelContext* context, const dnnl::memory::desc& expected_md)
      TF_LOCKS_EXCLUDED(mu_);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WeightCacheManager);

  mutex mu_;
  PersistentTensor weight_cached_data_ TF_GUARDED_BY(mu_);
  PersistentTensor weight_cached_md_ TF_GUARDED_BY(mu_);
};

// Bias cache is used to avoid scale the bias tensor repetitively in INT8 kernel
template <typename T>
class BiasCacheManager {
 public:
  BiasCacheManager() = default;
  ~BiasCacheManager() = default;

  bool IsEmpty() TF_LOCKS_EXCLUDED(mu_);

  // Cache the scaled bias buffer as persistent tensors.
  // Only one thread can execute this method at any given time.
  void SetCache(OpKernelContext* context, const dnnl::memory::desc& bias_md,
                const dnnl::primitive_attr& bias_attr, void* bias_data,
                const dnnl::engine& onednn_engine,
                const dnnl::memory& scales_mem = dnnl::memory())
      TF_LOCKS_EXCLUDED(mu_);

  // Get the cached bias buffer
  T* GetCache(OpKernelContext* context) TF_LOCKS_EXCLUDED(mu_);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BiasCacheManager);

  mutex mu_;
  PersistentTensor bias_cached_data_ TF_GUARDED_BY(mu_);
};
#endif

template <typename Device>
inline dnnl::fpmath_mode GetFP32MathMode() {
  std::string fp32_math_mode = "fp32";
  ITEX_CHECK_OK(
      ReadStringFromEnvVar("ITEX_FP32_MATH_MODE", "fp32", &fp32_math_mode));
  fp32_math_mode = str_util::Lowercase(fp32_math_mode);
  if (fp32_math_mode == "fp32") {
    return dnnl::fpmath_mode::strict;
  }
  if (fp32_math_mode == "tf32") {
    if (std::is_same<Device, CPUDevice>::value) {
      ITEX_LOG(FATAL) << "Did not support TF32 math mode on CPU ";
    }
    return dnnl::fpmath_mode::tf32;
  }
  if (fp32_math_mode == "bf32") {
    if (std::is_same<Device, GPUDevice>::value) {
      ITEX_LOG(FATAL) << "Did not support BF32 math mode on GPU ";
    }
    return dnnl::fpmath_mode::bf16;
  }
  ITEX_LOG(FATAL)
      << "Invalid ITEX_FP32_MATH_MODE, should be FP32, TF32 or BF32, but got "
      << fp32_math_mode;
}

}  // namespace itex
#endif  // ITEX_CORE_UTILS_ONEDNN_ONEDNN_UTIL_H_
