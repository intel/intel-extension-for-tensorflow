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

#ifndef ITEX_BUILD_JAX
#include "itex/core/utils/onednn/onednn_util.h"

#include <unordered_map>

#include "itex/core/utils/register_types.h"

namespace itex {
// Helper function to reorder oneDNN memory without `OpKernelContext`.
void ReorderMemoryInternal(const dnnl::memory* src_memory,
                           dnnl::memory* reorder_memory,
                           dnnl::stream& onednn_stream) {
  dnnl::reorder reorder_primitive = dnnl::reorder(*src_memory, *reorder_memory);
  std::unordered_map<int, dnnl::memory> reorder_args = {
      {DNNL_ARG_SRC, *src_memory}, {DNNL_ARG_DST, *reorder_memory}};
  reorder_primitive.execute(onednn_stream, reorder_args);
}

void ReorderMemory(const OpKernelContext& context,
                   const dnnl::memory* src_memory, dnnl::memory* reorder_memory,
                   const dnnl::engine& onednn_engine) {
  dnnl::stream onednn_stream = CreateDnnlStream(context, onednn_engine);
  ReorderMemoryInternal(src_memory, reorder_memory, onednn_stream);
}

// TF datatype and shape is meaningless for some tensors, such as scratchpad
// tensor and memory desc tensor in weight cache. These tensors are only used
// in OneDnn primitive, not related to Tensorflow. We only need to choose a
// short length datatype, ensure the it is divisible by allocated buffer.
using ShortDT = uint8;

template <typename T>
bool WeightCacheManager<T>::IsEmpty() TF_LOCKS_EXCLUDED(mu_) {
  tf_shared_lock lock(&mu_);
  // TODO(itex): investigate why weight_cached_data_.NumElements() == 1
  // instead of 0,  while weight_cached_data_.IsInitialized() == True
  return (!weight_cached_data_.IsInitialized());
}

template <typename T>
void WeightCacheManager<T>::SetCache(
    OpKernelContext* context, const dnnl::memory::desc& weight_original_md,
    const dnnl::memory::desc& weight_expected_md, void* weight_data,
    const dnnl::engine& onednn_engine) TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock lock(&mu_);

  if (weight_cached_data_.IsInitialized()) {
    return;
  }

  // Create original memory
  dnnl::memory weight_mem =
      CreateDnnlMemory(weight_original_md, onednn_engine, weight_data);

  // Create cached weight buffer
  Tensor* weight_cached_tensor = nullptr;
  size_t weight_size = weight_expected_md.get_size();
  TensorShape weight_tf_shape;
  weight_tf_shape.AddDim(weight_size / sizeof(T));
  OP_REQUIRES_OK(context, context->allocate_persistent(
                              DataTypeToEnum<T>::value, weight_tf_shape,
                              &weight_cached_data_, &weight_cached_tensor));

  // Create cached weight memory
  void* weight_cached_data = const_cast<void*>(
      static_cast<const void*>(weight_cached_tensor->flat<T>().data()));
  dnnl::memory weight_reorder_mem =
      CreateDnnlMemory(weight_expected_md, onednn_engine, weight_cached_data);

  // Execute reorder
  ReorderMemory(*context, &weight_mem, &weight_reorder_mem, onednn_engine);

  // Cache the memory descriptor
  Tensor* weight_md_cached_tensor = nullptr;
  TensorShape weight_md_tf_shape;
  weight_md_tf_shape.AddDim(sizeof(weight_expected_md) / sizeof(ShortDT));

  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  OP_REQUIRES_OK(context,
                 context->allocate_persistent(
                     DataTypeToEnum<ShortDT>::value, weight_md_tf_shape,
                     &weight_cached_md_, &weight_md_cached_tensor, alloc_attr));
  dnnl_memory_desc_t c_weight_expected_md;
  dnnl_memory_desc_clone(&c_weight_expected_md, weight_expected_md.get());
  *reinterpret_cast<dnnl_memory_desc_t*>(
      weight_md_cached_tensor->flat<ShortDT>().data()) = c_weight_expected_md;
}

template <typename T>
T* WeightCacheManager<T>::GetCache(OpKernelContext* context,
                                   const dnnl::memory::desc& expected_md)
    TF_LOCKS_EXCLUDED(mu_) {
  tf_shared_lock lock(&mu_);
  const Tensor* weight_cached_data = weight_cached_data_.AccessTensor(context);
  const Tensor* weight_cached_md = weight_cached_md_.AccessTensor(context);

  // Check if the memory descriptor of the cached weight is same as
  // expected_md. if so use the cached memory, else return nullptr
  if (weight_cached_md->flat<ShortDT>().size()) {
    dnnl::memory::desc* cached_md = reinterpret_cast<dnnl::memory::desc*>(
        const_cast<ShortDT*>(weight_cached_md->flat<ShortDT>().data()));
    if (*cached_md == expected_md) {
      return reinterpret_cast<T*>(
          const_cast<T*>(weight_cached_data->flat<T>().data()));
    } else {
      return nullptr;
      // TODO(itex): Weight cache format can change in the case that matmul
      // src has dymanic shape. Is it possible to cache weights with different
      // format? OP_REQUIRES_PTR(
      //     context, false,
      //     errors::Aborted("Unexpected memory descriptor from cached
      //     tensor!"));
    }
  } else {
    OP_REQUIRES_PTR(
        context, false,
        errors::Aborted(
            "Size of cached filter memory descriptor must not be zero!"));
  }
}

#define DEFINE_WEIGHT_CACHE(T) template class WeightCacheManager<T>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_WEIGHT_CACHE);
TF_CALL_QUANTIZED_TYPES(DEFINE_WEIGHT_CACHE);
TF_CALL_double(DEFINE_WEIGHT_CACHE);
#undef DEFINE_WEIGHT_CACHE

template <typename T>
bool BiasCacheManager<T>::IsEmpty() TF_LOCKS_EXCLUDED(mu_) {
  tf_shared_lock lock(&mu_);
  // TODO(itex): investigate why bias_cached_data_.NumElements() == 1
  // instead of 0,  when bias_cached_data_.IsInitialized() == True
  return (!bias_cached_data_.IsInitialized());
}

template <typename T>
void BiasCacheManager<T>::SetCache(OpKernelContext* context,
                                   const dnnl::memory::desc& bias_md,
                                   const dnnl::primitive_attr& bias_attr,
                                   void* bias_data,
                                   const dnnl::engine& onednn_engine,
                                   const dnnl::memory& scales_mem)
    TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock lock(&mu_);

  if (bias_cached_data_.IsInitialized()) {
    return;
  }

  // Create original bias memory
  dnnl::memory bias_mem = CreateDnnlMemory(bias_md, onednn_engine, bias_data);

  // Create scaled bias buffer
  Tensor* bias_cached_tensor = nullptr;
  size_t bias_size = bias_md.get_size();
  TensorShape bias_tf_shape;
  bias_tf_shape.AddDim(bias_size / sizeof(T));
  OP_REQUIRES_OK(context, context->allocate_persistent(
                              DataTypeToEnum<T>::value, bias_tf_shape,
                              &bias_cached_data_, &bias_cached_tensor));

  // Create cached bias memory
  void* bias_cached_data = const_cast<void*>(
      static_cast<const void*>(bias_cached_tensor->flat<T>().data()));
  dnnl::memory bias_scaled_mem =
      CreateDnnlMemory(bias_md, onednn_engine, bias_cached_data);

  // Bias scaling attributes
  dnnl::reorder reorder_primitive =
      dnnl::reorder(bias_mem, bias_scaled_mem, bias_attr);
  std::unordered_map<int, dnnl::memory> reorder_args = {
      {DNNL_ARG_SRC, bias_mem}, {DNNL_ARG_DST, bias_scaled_mem}};
  if (scales_mem != dnnl::memory())
    reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scales_mem});

  // Execute reorder
  auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
  reorder_primitive.execute(onednn_stream, reorder_args);
}

template <typename T>
T* BiasCacheManager<T>::GetCache(OpKernelContext* context)
    TF_LOCKS_EXCLUDED(mu_) {
  tf_shared_lock lock(&mu_);
  const Tensor* bias_cached_data = bias_cached_data_.AccessTensor(context);
  return reinterpret_cast<T*>(
      const_cast<T*>(bias_cached_data->flat<T>().data()));
}

#define DEFINE_BIAS_CACHE(T) template class BiasCacheManager<T>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_BIAS_CACHE);
TF_CALL_QUANTIZED_TYPES(DEFINE_BIAS_CACHE);
#undef DEFINE_BIAS_CACHE

}  // namespace itex
#endif  // ITEX_BUILD_JAX
