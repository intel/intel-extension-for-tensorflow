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

#include "itex/core/kernels/common/cwise_ops_common.h"

#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"

namespace itex {

template <typename Device, typename Functor, dnnl::algorithm alg_kind,
          typename T>
class OneDnnBinaryOp : public BinaryOp<Device, Functor> {
 public:
  explicit OneDnnBinaryOp(OpKernelConstruction* context)
      : BinaryOp<Device, Functor>(context) {}

  void Compute(OpKernelContext* context) override {
    const int kNumInputs = 2;
    std::vector<void*> inputs_data(kNumInputs);
    // inputs_tensor_reordered is needed when falling back to eigen.
    std::vector<Tensor> inputs_tensor_reordered(kNumInputs);
    std::vector<TensorShape> tf_input_shapes(kNumInputs);
    std::vector<OneDnnShape> tf_input_onednn_shapes(kNumInputs);
    std::vector<dnnl::memory::dims> src_dims(kNumInputs);

    for (int i = 0; i < kNumInputs; i++) {
      GetOneDnnShape(context, i, &tf_input_onednn_shapes[i]);
      const Tensor& input = context->input(i);
      inputs_data[i] = GetTensorBuffer<T>(&input);
      if (tf_input_onednn_shapes[i].IsOneDnnTensor())
        tf_input_shapes[i] = tf_input_onednn_shapes[i].GetTfShape();
      else
        tf_input_shapes[i] = input.shape();
    }

    // oneDNN only supports inputs with same rank size, so here we calculate and
    // expand dimension if they are not consistent. E.g. 8x4 * 4 --> 8x4 * 1x4.
    CalculateDims(tf_input_onednn_shapes[0], tf_input_onednn_shapes[1],
                  tf_input_shapes[0], tf_input_shapes[1], &src_dims[0],
                  &src_dims[1]);
    ITEX_VLOG(3) << "Shapes (start OneDnnBinaryOp compute): "
                 << tf_input_shapes[0].DebugString() << " _and_ "
                 << tf_input_shapes[1].DebugString();

    auto onednn_engine = CreateDnnlEngine<Device>(*context);
    if (UnsupportShape(tf_input_shapes[0], tf_input_shapes[1])) {
      for (int i = 0; i < kNumInputs; ++i) {
        if (tf_input_onednn_shapes[i].IsOneDnnTensor()) {
          auto src_onednn_md = tf_input_onednn_shapes[i].GetOneDnnLayout();
          auto src_tf_md = tf_input_onednn_shapes[i].GetTfLayout();
          TensorShape dst_shape = tf_input_onednn_shapes[i].GetTfShape();

          if (src_onednn_md != src_tf_md) {
            // Allocate buffer for reordered tensor
            OP_REQUIRES_OK(context, context->allocate_temp(
                                        DataTypeToEnum<T>::v(), dst_shape,
                                        &inputs_tensor_reordered[i]));

            auto src_mem =
                CreateDnnlMemory(src_onednn_md, onednn_engine, inputs_data[i]);
            auto reorder_mem = CreateDnnlMemory(
                src_tf_md, onednn_engine,
                GetTensorBuffer<T>(&inputs_tensor_reordered[i]));
            ReorderMemory(*context, &src_mem, &reorder_mem, onednn_engine);
          } else {
            inputs_tensor_reordered[i] = context->input(i);
          }
        } else {
          inputs_tensor_reordered[i] = context->input(i);
        }
      }

      BinaryOp<Device, Functor>::BinaryOpCompute(
          context, inputs_tensor_reordered[0], inputs_tensor_reordered[1]);

      OneDnnShape dst_onednn_shape;
      dst_onednn_shape.SetOneDnnTensor(false);
      AllocateMetaData(context, 0, dst_onednn_shape);
    } else {
      try {
        auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

        // oneDNN only supports inputs[1] bcast to inputs[0]. So if inputs[1]
        // has more elements than inputs[0], swap the 2 inputs.
        // Use an index to indicate the swapped result.
        const int kFirst =
            needSwap(tf_input_shapes[0], tf_input_shapes[1]) ? 1 : 0;
        const int kSecond = 1 - kFirst;

        bool is_onednn = false;
        OneDnnTensorFormat tf_format = OneDnnTensorFormat::FORMAT_INVALID;
        if (tf_input_onednn_shapes[kFirst].IsOneDnnTensor()) {
          tf_format = tf_input_onednn_shapes[kFirst].GetTfDataFormat();
          is_onednn = true;
        } else if (tf_input_onednn_shapes[kSecond].IsOneDnnTensor()) {
          tf_format = tf_input_onednn_shapes[kSecond].GetTfDataFormat();
          is_onednn = true;
        }
        // create src memory descriptor, plain input should have the same
        // tf_format with block input in logic
        std::vector<dnnl::memory::desc> src_mds(kNumInputs);
        for (int i = 0; i < kNumInputs; i++) {
          if (tf_input_onednn_shapes[i].IsOneDnnTensor()) {
            src_mds[i] = tf_input_onednn_shapes[i].GetOneDnnLayout();
          } else {
            if (is_onednn)
              src_mds[i] =
                  dnnl::memory::desc(src_dims[i], OneDnnType<T>(),
                                     OneDnnTensorFormatToTag(tf_format));
            else
              src_mds[i] = CreatePlainMemDescWithFormatTag<T>(src_dims[i]);
          }
        }

        // create dst memory descriptor.
        dnnl::memory::desc dst_md_prefer;
        if (is_onednn) {
          // If there is onednn format in inputs, the output will be onednn
          // format. And output shape should be the same with kFirst since
          // kFirst has more elements after swap.
          dst_md_prefer =
              dnnl::memory::desc(src_dims[kFirst], OneDnnType<T>(),
                                 OneDnnTensorFormatToTag(tf_format));
        } else {
          dst_md_prefer = CreatePlainMemDescWithFormatTag<T>(src_dims[kFirst]);
        }

        Tensor* dst_tensor = nullptr;
        OneDnnShape dst_onednn_shape;
        TensorShape tf_shape_dst = tf_input_shapes[kFirst];
        SetOutputTensorShape(dst_md_prefer, tf_format, &tf_shape_dst,
                             &dst_onednn_shape, is_onednn);
        AllocateOutputSetOneDnnShape(context, 0, &dst_tensor, tf_shape_dst,
                                     dst_onednn_shape);

        // create src & dst dnnl memory objects
        auto src0_mem =
            dnnl::memory(src_mds[kFirst], onednn_engine, inputs_data[kFirst]);
        auto src1_mem =
            dnnl::memory(src_mds[kSecond], onednn_engine, inputs_data[kSecond]);
        auto dst_mem = dnnl::memory(dst_md_prefer, onednn_engine,
                                    GetTensorBuffer<T>(dst_tensor));

        // create dnnl binary primitive
        // TODO(itex): OneDNN `ref` path has poor performance. Fall back to
        // Eigen if needed.
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        auto binary_d = dnnl::binary::desc(alg_kind, src_mds[kFirst],
                                           src_mds[kSecond], dst_md_prefer);
        auto binary_pd =
            dnnl::binary::primitive_desc(binary_d, attr, onednn_engine);
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
#ifndef INTEL_CPU_ONLY
#define MAX_NDIMS 6
#else
#define MAX_NDIMS 12
#endif
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

  inline void CalculateDims(const OneDnnShape& in0_onednn_shape,
                            const OneDnnShape& in1_onednn_shape,
                            const TensorShape& in0_shape,
                            const TensorShape& in1_shape,
                            dnnl::memory::dims* src0_dims,
                            dnnl::memory::dims* src1_dims) {
    if (in0_onednn_shape.IsOneDnnTensor() &&
        in1_onednn_shape.IsOneDnnTensor()) {
      *src0_dims = in0_onednn_shape.GetSizesAsOneDnnDims();
      *src1_dims = in1_onednn_shape.GetSizesAsOneDnnDims();
    } else if (in0_onednn_shape.IsOneDnnTensor() &&
               !in1_onednn_shape.IsOneDnnTensor()) {
      *src0_dims = in0_onednn_shape.GetSizesAsOneDnnDims();
      OneDnnTensorFormat src0_onednn_data_format =
          in0_onednn_shape.GetTfDataFormat();
      TensorShape in1_shape_tmp = in1_shape;
      for (int i = in0_shape.dims() - in1_shape.dims(); i > 0; i--) {
        in1_shape_tmp.InsertDim(0, 1);
      }
      if (in1_shape_tmp.dims() == 4 || in1_shape_tmp.dims() == 5) {
        auto src0_tf_data_format =
            OneDnnDataFormatToTFDataFormat(src0_onednn_data_format);
        *src1_dims = TFShapeToOneDnnDimsInNC(in1_shape_tmp, src0_tf_data_format,
                                             in1_shape_tmp.dims() == 4);
      } else {
        *src1_dims = TFShapeToOneDnnDims(in1_shape_tmp);
      }
    } else if (!in0_onednn_shape.IsOneDnnTensor() &&
               in1_onednn_shape.IsOneDnnTensor()) {
      *src1_dims = in1_onednn_shape.GetSizesAsOneDnnDims();
      OneDnnTensorFormat src1_onednn_data_format =
          in1_onednn_shape.GetTfDataFormat();
      TensorShape in0_shape_tmp = in0_shape;
      for (int i = in1_shape.dims() - in0_shape.dims(); i > 0; i--) {
        in0_shape_tmp.InsertDim(0, 1);
      }
      if (in0_shape_tmp.dims() == 4 || in0_shape_tmp.dims() == 5) {
        auto src1_tf_data_format =
            OneDnnDataFormatToTFDataFormat(src1_onednn_data_format);
        *src0_dims = TFShapeToOneDnnDimsInNC(in0_shape_tmp, src1_tf_data_format,
                                             in0_shape_tmp.dims() == 4);
      } else {
        *src0_dims = TFShapeToOneDnnDims(in0_shape_tmp);
      }
    } else {
      *src0_dims = TFShapeToOneDnnDims(in0_shape);
      *src1_dims = TFShapeToOneDnnDims(in1_shape);
    }

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

#define REGISTER_GPU_KERNELS_E(N, F, A, T)             \
  REGISTER_KERNEL_BUILDER(Name(N)                      \
                              .Device(DEVICE_GPU)      \
                              .HostMemory("x_meta")    \
                              .HostMemory("y_meta")    \
                              .HostMemory("z_meta")    \
                              .TypeConstraint<T>("T"), \
                          OneDnnBinaryOp<GPUDevice, F<T>, A, T>);

#define REGISTER_GPU_KERNELS(T)                          \
  REGISTER_GPU_KERNELS_E("_OneDnnAdd", functor::add,     \
                         dnnl::algorithm::binary_add, T) \
  REGISTER_GPU_KERNELS_E("_OneDnnAddV2", functor::add,   \
                         dnnl::algorithm::binary_add, T) \
  REGISTER_GPU_KERNELS_E("_OneDnnMul", functor::mul,     \
                         dnnl::algorithm::binary_mul, T) \
  REGISTER_GPU_KERNELS_E("_OneDnnSub", functor::sub,     \
                         dnnl::algorithm::binary_sub, T)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);

}  // namespace itex
