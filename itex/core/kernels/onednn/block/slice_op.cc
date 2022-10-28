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

#include "itex/core/kernels/common/slice_functor.h"

using dnnl::memory;

namespace itex {

template <typename T>
static void SharedSliceCommonCases(OpKernelContext* context,
                                   const Tensor& input,
                                   const TensorShape& src_tf_shape,
                                   TensorShape* dst_tf_shape,
                                   gtl::InlinedVector<int64, 4>* begin,
                                   gtl::InlinedVector<int64, 4>* size,
                                   bool* done) {
  constexpr int kSrcIndex = 0;
  constexpr int kDstIndex = 0;

  bool is_identity = true;
  bool slice_dim0 = true;
  *done = false;

  SharedSliceValidation(context, src_tf_shape, dst_tf_shape, &is_identity,
                        &slice_dim0, begin, size);
  if (!context->status().ok()) return;
  if (is_identity) {
    ITEX_VLOG(2) << "Slice identity";
    // Data tensor
    context->set_output(kDstIndex, input);

    // Meta tensor
    OneDnnShape src_onednn_shape;
    GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
    ForwardMetaData(context, kSrcIndex, kDstIndex, src_onednn_shape);
    *done = true;
    return;
  }
}

template <typename Device, typename T>
class OneDnnSliceOp : public OpKernel {
 public:
  explicit OneDnnSliceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> size;

    const Tensor& src_tensor = context->input(kSrcIndex);
    OneDnnShape src_onednn_shape;
    GetOneDnnShape(context, kSrcIndex, &src_onednn_shape);
    bool is_src_onednn = src_onednn_shape.IsOneDnnTensor();
    TensorShape src_tf_shape =
        is_src_onednn ? src_onednn_shape.GetTfShape() : src_tensor.shape();

    bool done = false;
    TensorShape dst_tf_shape;
    // Quick Path for Slice op
    SharedSliceCommonCases<T>(context, src_tensor, src_tf_shape, &dst_tf_shape,
                              &begin, &size, &done);
    if (!context->status().ok() || done == true) return;

    // TODO(itex): Try to reorder directly "block input-> plain slice output",
    // instead of "block input-> plain input -> plain slice output"

    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);
      Tensor src_reorder_tensor;
      memory src_mem, src_reorder_mem;
      bool is_src_reordered = false;
      if (is_src_onednn) {
        // The slice operation can only be operated on plain tensor. If the
        // input is block layout, reorder it to plain layout.
        memory::desc src_onednn_md = src_onednn_shape.GetOneDnnLayout();
        memory::desc src_tf_md = src_onednn_shape.GetTfLayout();
        src_mem = CreateDnnlMemory(src_onednn_md, onednn_engine,
                                   GetTensorBuffer<T>(&src_tensor));
        is_src_reordered = (src_onednn_md != src_tf_md);
        if (is_src_reordered) {
          int64 src_reorder_size = src_tf_md.get_size() / sizeof(T);
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                TensorShape({src_reorder_size}),
                                                &src_reorder_tensor));
          src_reorder_mem =
              CreateDnnlMemory(src_tf_md, onednn_engine,
                               GetTensorBuffer<T>(&src_reorder_tensor));
          ReorderMemory(*context, &src_mem, &src_reorder_mem, onednn_engine);
        }
      }

      memory::desc src_md, dst_md;
      memory::dims begin_dims, size_dims;
      if (is_src_onednn) {
        // if input is blocked layout, it is already be reordered to plain
        // format
        src_md = src_onednn_shape.GetTfLayout();
        begin_dims = memory::dims(begin.begin(), begin.end());
        size_dims = memory::dims(size.begin(), size.end());
        auto src_tf_data_format =
            OneDnnDataFormatToTFDataFormat(src_onednn_shape.GetTfDataFormat());
        begin_dims = OneDnnDimsInNC(begin_dims, src_tf_data_format,
                                    src_tf_shape.dims() == 4);
        size_dims = OneDnnDimsInNC(size_dims, src_tf_data_format,
                                   src_tf_shape.dims() == 4);
        // Note: size_dims here is in logical sequence for output, not actual
        // tf_shape for output is dst_tf_shape

        dst_md = CreatePlainMemDescWithFormatTag<T>(size_dims);
      } else {
        memory::dims src_dims = TFShapeToOneDnnDims(src_tensor.shape());
        src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
        begin_dims = memory::dims(begin.begin(), begin.end());
        size_dims = memory::dims(size.begin(), size.end());
        dst_md = CreatePlainMemDescWithFormatTag<T>(size_dims);
      }

      memory::desc src_sub_md = src_md.submemory_desc(size_dims, begin_dims);
      dnnl::reorder::primitive_desc reorder_pd(onednn_engine, src_sub_md,
                                               onednn_engine, dst_md);
      dnnl::reorder reorder_prim(reorder_pd);

      OneDnnShape dst_onednn_shape;
      SetOutputTensorShape(dst_md, src_onednn_shape.GetTfDataFormat(),
                           &dst_tf_shape, &dst_onednn_shape,
                           false /* output is always plain layout*/);

      Tensor* dst_tensor = nullptr;
      AllocateOutputSetOneDnnShape(context, kDstIndex, &dst_tensor,
                                   dst_tf_shape, dst_onednn_shape);

      if (!is_src_onednn) {
        // For plain layout input, src memory is not created yet.
        src_mem = CreateDnnlMemory(src_md, onednn_engine,
                                   GetTensorBuffer<T>(&src_tensor));
      }

      // Create dst memory
      dnnl::memory dst_mem = CreateDnnlMemory(dst_md, onednn_engine,
                                              GetTensorBuffer<T>(dst_tensor));
      // Create scratch pad
      Tensor scratchpad_tensor;
      int64 scratchpad_size =
          reorder_pd.scratchpad_desc().get_size() / sizeof(T);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({scratchpad_size}),
                                            &scratchpad_tensor));
      auto scratchpad_mem =
          dnnl::memory(reorder_pd.scratchpad_desc(), onednn_engine,
                       GetTensorBuffer<T>(&scratchpad_tensor));
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);
      std::unordered_map<int, memory> reorder_primitive_args = {
          {DNNL_ARG_SRC, is_src_reordered ? src_reorder_mem : src_mem},
          {DNNL_ARG_DST, dst_mem},
          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}};
      reorder_prim.execute(onednn_stream, reorder_primitive_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status:" + std::to_string(e.status) +
                         ", message: " + string(e.message) + ". in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context, errors::Aborted("Compute received an exception:",
                                              error_msg));
    }
  }

 private:
  const int kSrcIndex = 0;
  const int kBeginIndex = 1;
  const int kSizeIndex = 2;
  const int kDstIndex = 0;
};

#ifndef INTEL_CPU_ONLY
#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("_OneDnnSlice")            \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T")  \
                              .HostMemory("begin")        \
                              .HostMemory("size")         \
                              .HostMemory("input_meta")   \
                              .HostMemory("begin_meta")   \
                              .HostMemory("size_meta")    \
                              .HostMemory("output_meta"), \
                          OneDnnSliceOp<GPUDevice, TYPE>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#else
#define REGISTER_KERNEL(TYPE)                                            \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_OneDnnSlice").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      OneDnnSliceOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);

#endif  // INTEL_CPU_ONLY
}  // namespace itex
