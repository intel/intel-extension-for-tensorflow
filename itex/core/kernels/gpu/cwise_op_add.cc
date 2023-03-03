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

#include "itex/core/kernels/common/cwise_ops_common.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"

namespace itex {

template <typename Device, typename Functor, dnnl::algorithm alg_kind,
          typename T>
class DnnBinaryOp : public BinaryOp<Device, Functor> {
 public:
  explicit DnnBinaryOp(OpKernelConstruction* context)
      : BinaryOp<Device, Functor>(context) {}

  void Compute(OpKernelContext* context) override {
    const int kNumInputs = 2;
    std::vector<void*> inputs_data(kNumInputs);
    std::vector<TensorShape> tf_input_shapes(kNumInputs);
    std::vector<dnnl::memory::dims> src_dims(kNumInputs);

    for (int i = 0; i < kNumInputs; i++) {
      const Tensor& input = context->input(i);
      tf_input_shapes[i] = input.shape();
      inputs_data[i] = GetTensorBuffer<T>(&input);
    }

    // oneDNN only supports inputs with same rank size, so here we calculate and
    // expand dimension if they are not consistent. E.g. 8x4 * 4 --> 8x4 * 1x4.
    CalculateDims(tf_input_shapes[0], tf_input_shapes[1], &src_dims[0],
                  &src_dims[1]);
    ITEX_VLOG(3) << "Shapes (start DnnBinaryOp compute): "
                 << tf_input_shapes[0].DebugString() << " _and_ "
                 << tf_input_shapes[1].DebugString();

    auto onednn_engine = CreateDnnlEngine<Device>(*context);
    if (UnsupportShape(tf_input_shapes[0], tf_input_shapes[1])) {
      BinaryOp<Device, Functor>::BinaryOpCompute(context, context->input(0),
                                                 context->input(1));
    } else {
      try {
        auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

        // oneDNN only supports inputs[1] bcast to inputs[0]. So if inputs[1]
        // has more elements than inputs[0], swap the 2 inputs.
        // Use an index to indicate the swapped result.
        const int kFirst =
            needSwap(tf_input_shapes[0], tf_input_shapes[1]) ? 1 : 0;
        const int kSecond = 1 - kFirst;

        // create src memory descriptor
        dnnl::memory::desc dst_md_prefer;
        std::vector<dnnl::memory::desc> src_mds(kNumInputs);
        for (int i = 0; i < kNumInputs; i++) {
          src_mds[i] = CreatePlainMemDescWithFormatTag<T>(src_dims[i]);
        }

        // create dst memory descriptor.
        dst_md_prefer = CreatePlainMemDescWithFormatTag<T>(src_dims[kFirst]);

        Tensor* dst_tensor = nullptr;
        TensorShape tf_shape_dst = tf_input_shapes[kFirst];
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, tf_shape_dst, &dst_tensor));

        // create src & dst dnnl memory objects
        auto src0_mem =
            dnnl::memory(src_mds[kFirst], onednn_engine, inputs_data[kFirst]);
        auto src1_mem =
            dnnl::memory(src_mds[kSecond], onednn_engine, inputs_data[kSecond]);
        auto dst_mem = dnnl::memory(dst_md_prefer, onednn_engine,
                                    GetTensorBuffer<T>(dst_tensor));

        // create dnnl binary primitive
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifdef ITEX_ONEDNN_3_0
        auto binary_pd = dnnl::binary::primitive_desc(
            onednn_engine, alg_kind, src_mds[kFirst], src_mds[kSecond],
            dst_md_prefer, attr);
#else
        auto binary_d = dnnl::binary::desc(alg_kind, src_mds[kFirst],
                                           src_mds[kSecond], dst_md_prefer);
        auto binary_pd =
            dnnl::binary::primitive_desc(binary_d, attr, onednn_engine);
#endif
        auto binary_prim = dnnl::binary(binary_pd);

        Tensor scratchpad_tensor;
        int64 scratchpad_size =
            binary_pd.scratchpad_desc().get_size() / sizeof(T);
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::v(),
                                              TensorShape({scratchpad_size}),
                                              &scratchpad_tensor));
        auto scratchpad_mem =
            dnnl::memory(binary_pd.scratchpad_desc(), onednn_engine,
                         GetTensorBuffer<T>(&scratchpad_tensor));

        // primitive arguments
        std::unordered_map<int, dnnl::memory> binary_args;
        binary_args.insert({DNNL_ARG_SRC_0, src0_mem});
        binary_args.insert({DNNL_ARG_SRC_1, src1_mem});
        binary_args.insert({DNNL_ARG_DST, dst_mem});
        binary_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

        // primitive execution
        binary_prim.execute(onednn_stream, binary_args);
      } catch (dnnl::error& e) {
        string error_msg = "Status: " + std::to_string(e.status) +
                           ", message: " + string(e.message) + ", in file " +
                           string(__FILE__) + ":" + std::to_string(__LINE__);
        OP_REQUIRES_OK(
            context,
            errors::Aborted("Operation received an exception:", error_msg));
      }
    }
  }

 private:
  inline bool UnsupportShape(const TensorShape& shape0,
                             const TensorShape& shape1) {
// Bi-bcast like 8x1 * 1x4 isn't supported in oneDNN. Compare output
// shape(8x4) with input shapes, and fall back to Eigen if output has more
// elements than all inputs.
#define MAX_NDIMS 6
    if ((shape0.dims() > MAX_NDIMS) || (shape1.dims()) > MAX_NDIMS) return true;
#undef MAX_NDIMS
    int64 dst_elements = 1;

    TensorShape l = shape0.dims() > shape1.dims() ? shape0 : shape1;
    TensorShape s = shape0.dims() > shape1.dims() ? shape1 : shape0;

    int gap = l.dims() - s.dims();
    for (int i = 0; i < gap; ++i) dst_elements *= l.dim_size(i);
    for (int i = 0; i < s.dims(); ++i)
      dst_elements *= std::max(s.dim_size(i), l.dim_size(i + gap));

    if (dst_elements > shape0.num_elements() &&
        dst_elements > shape1.num_elements())
      return true;

    if (shape0.num_elements() == 0 || shape1.num_elements() == 0) return true;

    // When doing _OneDnnSub, if it requires swap due to bcast limitation in
    // OneDNN, then roll back to Eigen.
    // TODO(itex): Remove this limitation if bidirectional broadcast is
    // supported in OneDNN
    if ((alg_kind == dnnl::algorithm::binary_sub) && needSwap(shape0, shape1))
      return true;

    return false;
  }

  inline bool needSwap(const TensorShape& in0_shape,
                       const TensorShape& in1_shape) {
    return (in1_shape.num_elements() > in0_shape.num_elements()) ||
           ((in1_shape.num_elements() == in0_shape.num_elements()) &&
            (in1_shape.dims() > in0_shape.dims() ||
             (in0_shape.dims() == 0 && in1_shape.dims() > 0)));
  }

  inline void ExpandDim(int ndims, dnnl::memory::dims* dims) {
    std::vector<int> td(ndims, 1);
    for (int i = 0; i < dims->size(); ++i) {
      td[ndims - dims->size() + i] = (*dims)[i];
    }
    dims->resize(ndims);
    for (int i = 0; i < ndims; ++i) {
      (*dims)[i] = td[i];
    }
    return;
  }

  inline void CalculateDims(TensorShape in0_shape, TensorShape in1_shape,
                            dnnl::memory::dims* src0_dims,
                            dnnl::memory::dims* src1_dims) {
    *src0_dims = TFShapeToOneDnnDims(in0_shape);
    *src1_dims = TFShapeToOneDnnDims(in1_shape);

    if (src0_dims->size() == src1_dims->size()) {
      return;
    } else if (src0_dims->size() > src1_dims->size()) {
      ExpandDim(src0_dims->size(), src1_dims);
      return;
    } else {
      ExpandDim(src1_dims->size(), src0_dims);
      return;
    }
  }
};

REGISTER3(BinaryOp, GPU, "Add", functor::add, float, Eigen::half,
          Eigen::bfloat16);
REGISTER2(BinaryOp, GPU, "Add", functor::add, int64, complex64);
#ifdef ITEX_ENABLE_DOUBLE
REGISTER2(BinaryOp, GPU, "Add", functor::add, double, complex128);
#endif  // ITEX_ENABLE_DOUBLE

REGISTER6(BinaryOp, GPU, "AddV2", functor::add, Eigen::half, uint8, int64,
          float, complex64, Eigen::bfloat16);

#define REGISTER_GPU_KERNELS_E(N, F, A, T)                                   \
  REGISTER_KERNEL_BUILDER(Name(N).Device(DEVICE_GPU).TypeConstraint<T>("T"), \
                          DnnBinaryOp<GPUDevice, F<T>, A, T>);

// TODO(guizi) remove this when fix eigen bcast binary issue in 16 bit.
#define REGISTER_GPU_KERNELS(T)                                               \
  REGISTER_GPU_KERNELS_E("Mul", functor::mul, dnnl::algorithm::binary_mul, T) \
  REGISTER_GPU_KERNELS_E("Add", functor::add, dnnl::algorithm::binary_add, T) \
  REGISTER_GPU_KERNELS_E("AddV2", functor::add, dnnl::algorithm::binary_add, T)

// TF_CALL_bfloat16(REGISTER_GPU_KERNELS);

// A special GPU kernel for int32.
REGISTER_KERNEL_BUILDER(Name("Add")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);
REGISTER_KERNEL_BUILDER(Name("AddV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);

#ifdef ITEX_ENABLE_DOUBLE
REGISTER2(BinaryOp, GPU, "AddV2", functor::add, double, complex128);
#endif  // ITEX_ENABLE_DOUBLE
}  // namespace itex
