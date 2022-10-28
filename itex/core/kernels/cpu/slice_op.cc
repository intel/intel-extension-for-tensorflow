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

    *done = true;
    return;
  }
}

template <typename Device, typename T>
class SliceOp : public OpKernel {
 public:
  explicit SliceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> size;

    const Tensor& src_tensor = context->input(kSrcIndex);
    TensorShape src_tf_shape = src_tensor.shape();

    bool done = false;
    TensorShape dst_tf_shape;
    // Quick Path for Slice op
    SharedSliceCommonCases<T>(context, src_tensor, src_tf_shape, &dst_tf_shape,
                              &begin, &size, &done);
    if (!context->status().ok() || done == true) return;

    // plain input -> plain slice output

    try {
      auto onednn_engine = CreateDnnlEngine<Device>(*context);

      memory::dims src_dims = TFShapeToOneDnnDims(src_tensor.shape());
      memory::dims begin_dims = memory::dims(begin.begin(), begin.end());
      memory::dims size_dims = memory::dims(size.begin(), size.end());

      memory::desc src_md = CreatePlainMemDescWithFormatTag<T>(src_dims);
      memory::desc dst_md = CreatePlainMemDescWithFormatTag<T>(size_dims);

      memory::desc src_sub_md = src_md.submemory_desc(size_dims, begin_dims);
      dnnl::reorder::primitive_desc reorder_pd(onednn_engine, src_sub_md,
                                               onednn_engine, dst_md);
      dnnl::reorder reorder_prim(reorder_pd);

      Tensor* dst_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(kDstIndex, dst_tf_shape,
                                                       &dst_tensor));

      // Create src memory
      dnnl::memory src_mem = CreateDnnlMemory(src_md, onednn_engine,
                                              GetTensorBuffer<T>(&src_tensor));
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
          {DNNL_ARG_SRC, src_mem},
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

#define REGISTER_KERNEL(TYPE)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("_ITEXSlice").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      SliceOp<CPUDevice, TYPE>)
TF_CALL_CPU_NUMBER_TYPES(REGISTER_KERNEL);

}  // namespace itex
