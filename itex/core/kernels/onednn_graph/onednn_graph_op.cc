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

#ifndef INTEL_CPU_ONLY
#include "itex/core/devices/bfc_allocator.h"
#include "itex/core/devices/gpu/gpu_pool_allocator.h"
#include "third_party/build_option/dpcpp/runtime/itex_gpu_runtime.h"
#endif  // INTEL_CPU_ONLY
#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_graph_util.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/types.h"

namespace itex {

#ifdef ITEX_ONEDNN_3_0
// oneDNN Graph prefer use make_engine_with_allocator to create oneDNN engine.
// Thus here engine creation is different from onednn_util.h. For oneDNN stream,
// oneDNN Graph & oneDNN uses the same function.
template <typename Device>
dnnl::engine CreateDnnlEngine(OpKernelContext* ctx);

// Spicialization for CPU
template <>
dnnl::engine CreateDnnlEngine<CPUDevice>(OpKernelContext* ctx) {
  static dnnl::graph::allocator alloc{};
  static dnnl::engine cpu_engine =
      make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
  return cpu_engine;
}

#ifndef INTEL_CPU_ONLY
void* sycl_malloc_wrapper(size_t n, size_t alignment, const void* device,
                          const void* ctx) {
  // TODO(itex): Currently, we ignore the alignment argument. The default
  // alignment in ITEX is 256.
  auto& device_ptr = *static_cast<const ITEX_GPUDevice*>(device);
  ITEX_GPUDevice* device_handle;
  DeviceOrdinal device_ordinal;
  ITEX_GPUGetDeviceOrdinal(device_ptr, &device_ordinal);
  ITEX_GPUGetDevice(&device_handle, device_ordinal);
  std::shared_ptr<BFCAllocator> alloc;
  auto status = ITEX_GPUGetAllocator(device_handle, &alloc);
  ITEX_CHECK(status == ITEX_GPU_SUCCESS)
      << "Failed to get device allocator, device handle: " << device_handle;
  return alloc->AllocateRaw(n);
}

void sycl_free_wrapper(void* ptr, const void* device, const void* context,
                       void* e) {
  auto& device_ptr = *static_cast<const ITEX_GPUDevice*>(device);
  ITEX_GPUDevice* device_handle;
  DeviceOrdinal device_ordinal;
  ITEX_GPUGetDeviceOrdinal(device_ptr, &device_ordinal);
  ITEX_GPUGetDevice(&device_handle, device_ordinal);
  std::shared_ptr<BFCAllocator> alloc;
  auto status = ITEX_GPUGetAllocator(device_handle, &alloc);
  ITEX_CHECK(status == ITEX_GPU_SUCCESS)
      << "Failed to get device allocator, device handle: " << device_handle;
  alloc->DeallocateRaw(ptr);
}

// Spicialization for GPU
template <>
dnnl::engine CreateDnnlEngine<GPUDevice>(OpKernelContext* ctx) {
  auto* queue = ctx->GetDeviceStream();
  static dnnl::graph::allocator allocator =
      dnnl::graph::sycl_interop::make_allocator(sycl_malloc_wrapper,
                                                sycl_free_wrapper);
  static dnnl::engine gpu_engine =
      dnnl::graph::sycl_interop::make_engine_with_allocator(
          queue->get_device(), queue->get_context(), allocator);
  return gpu_engine;
}

#endif

#else
template <typename Device>
dnnl::graph::engine CreateDnnlEngine(OpKernelContext* ctx);
template <typename Device>
dnnl::graph::stream CreateDnnlStream(
    OpKernelContext* ctx,
    dnnl::graph::engine& engine);  // NOLINT(runtime/references)

// Spicialization for CPU
template <>
dnnl::graph::engine CreateDnnlEngine<CPUDevice>(OpKernelContext* ctx) {
  static dnnl::graph::engine cpu_engine(dnnl::graph::engine::kind::cpu, 0);
  return cpu_engine;
}
template <>
dnnl::graph::stream CreateDnnlStream<CPUDevice>(
    OpKernelContext* ctx,
    dnnl::graph::engine& engine) {  // NOLINT(runtime/references)
  static dnnl::graph::stream cpu_stream{engine};
  return cpu_stream;
}

#ifndef INTEL_CPU_ONLY
void* sycl_malloc_wrapper(size_t n, size_t alignment, const void* device,
                          const void* ctx) {
  // TODO(itex): Currently, we ignore the alignment argument. The default
  // alignment in ITEX is 256.
  auto& device_ptr = *static_cast<const ITEX_GPUDevice*>(device);
  ITEX_GPUDevice* device_handle;
  DeviceOrdinal device_ordinal;
  ITEX_GPUGetDeviceOrdinal(device_ptr, &device_ordinal);
  ITEX_GPUGetDevice(&device_handle, device_ordinal);
  std::shared_ptr<BFCAllocator> alloc;
  auto status = ITEX_GPUGetAllocator(device_handle, &alloc);
  ITEX_CHECK(status == ITEX_GPU_SUCCESS)
      << "Failed to get device allocator, device handle: " << device_handle;
  return alloc->AllocateRaw(n);
}
void sycl_free_wrapper(void* ptr, const void* device, const void* context,
                       void* e) {
  auto& device_ptr = *static_cast<const ITEX_GPUDevice*>(device);
  ITEX_GPUDevice* device_handle;
  DeviceOrdinal device_ordinal;
  ITEX_GPUGetDeviceOrdinal(device_ptr, &device_ordinal);
  ITEX_GPUGetDevice(&device_handle, device_ordinal);
  std::shared_ptr<BFCAllocator> alloc;
  auto status = ITEX_GPUGetAllocator(device_handle, &alloc);
  ITEX_CHECK(status == ITEX_GPU_SUCCESS)
      << "Failed to get device allocator, device handle: " << device_handle;
  alloc->DeallocateRaw(ptr);
}

// Spicialization for GPU
template <>
dnnl::graph::engine CreateDnnlEngine<GPUDevice>(OpKernelContext* ctx) {
  auto* queue = ctx->GetDeviceStream();
  static dnnl::graph::allocator allocator =
      dnnl::graph::sycl_interop::make_allocator(sycl_malloc_wrapper,
                                                sycl_free_wrapper);
  dnnl::graph::engine gpu_engine = dnnl::graph::sycl_interop::make_engine(
      queue->get_device(), queue->get_context(), allocator);
  return gpu_engine;
}

template <>
dnnl::graph::stream CreateDnnlStream<GPUDevice>(
    OpKernelContext* ctx,
    dnnl::graph::engine& engine) {  // NOLINT(runtime/references)
  auto* queue = ctx->GetDeviceStream();
  dnnl::graph::stream gpu_stream =
      dnnl::graph::sycl_interop::make_stream(engine, *queue);
  return gpu_stream;
}
#endif
#endif  // ITEX_ONEDNN_3_0

// TODO(itex): Add UT to verify the LLGA inplace
// Collect the output/input pair of OneDnn Graph Inplace
void GetInplaceIdMap(
    const dnnl::graph::compiled_partition& c_partition,
    const std::vector<dnnl::graph::logical_tensor>& l_input_logical_tensor,
    const std::vector<dnnl::graph::logical_tensor>& l_output_logical_tensor,
    std::unordered_map<size_t, size_t>* inplace_id_map) {
  for (auto& p : c_partition.get_inplace_ports()) {
    size_t input_id = p.first;
    size_t output_id = p.second;
    auto input_lt_iter = std::find_if(
        l_input_logical_tensor.begin(), l_input_logical_tensor.end(),
        [input_id](const dnnl::graph::logical_tensor& lt) {
          return input_id == lt.get_id();
        });
    auto input_lt_idx = static_cast<size_t>(
        std::distance(l_input_logical_tensor.begin(), input_lt_iter));

    auto output_lt_iter = std::find_if(
        l_output_logical_tensor.begin(), l_output_logical_tensor.end(),
        [output_id](const dnnl::graph::logical_tensor& lt) {
          return output_id == lt.get_id();
        });
    auto output_lt_idx = static_cast<size_t>(
        std::distance(l_output_logical_tensor.begin(), output_lt_iter));

    inplace_id_map->insert({output_lt_idx, input_lt_idx});
  }
}

// Currently, LLGA kernels only works with Layout pass ON. Because meta tensor
// is required to pass the LLGA layout information
// TODO(itex): Enable LLGA with ITEX plain format.
template <typename Device>
class OneDnnGraphOp : public OpKernel {
 public:
  explicit OneDnnGraphOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("partition_id", &partition_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_dt_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_edge_ids", &input_edge_ids_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_edge_ids", &output_edge_ids_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("is_constant_input_edge", &is_constant_input_edge_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("candidate_inplace_input_edge",
                                     &candidate_inplace_input_edge_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("framework_ops", &framework_ops_));
  }

  void Compute(OpKernelContext* ctx) {
    ITEX_VLOG(3) << "IN COMPUTE ";

    ITEX_VLOG(3) << "partition_id_ " << partition_id_;
    std::vector<dnnl::graph::logical_tensor> l_input_logical_tensor;
    std::vector<dnnl::graph::logical_tensor> l_output_logical_tensor;
    std::vector<dnnl::graph::tensor> l_input_tensor;
    std::vector<dnnl::graph::tensor> l_output_tensor;

#ifdef ITEX_ONEDNN_3_0
    dnnl::engine onednn_engine = CreateDnnlEngine<Device>(ctx);
    dnnl::stream onednn_stream = CreateDnnlStream(*ctx, onednn_engine);
#else
    dnnl::graph::engine onednn_engine = CreateDnnlEngine<Device>(ctx);
    dnnl::graph::stream onednn_stream =
        CreateDnnlStream<Device>(ctx, onednn_engine);
#endif
    auto partition = std::make_shared<dnnl::graph::partition>(
        graph::GetOneDnnGraphPartition(partition_id_));

    ITEX_CHECK_EQ(input_edge_ids_.size(), is_constant_input_edge_.size());

    // Prepare input tensors and logical tensors
    for (int index = 0; index < ctx->num_inputs(); index++) {
      const Tensor& input = ctx->input(index);
      void* current_src_ptr = input.data();

      auto input_data_type =
          graph::GetOneDnnGraphDataType(ctx->input(index).dtype());
      std::vector<int64_t> onednn_graph_input_shape;

      auto input_constant_property =
          is_constant_input_edge_[index]
              ? dnnl::graph::logical_tensor::property_type::constant
              : dnnl::graph::logical_tensor::property_type::undef;

      auto tf_input_shape = ctx->input(index).shape();
      if (tf_input_shape.dims() == 0) {
#ifdef ITEX_ONEDNN_3_0
        onednn_graph_input_shape = {};
#else
        onednn_graph_input_shape = {1};
#endif
      } else {
        for (int i = 0; i < tf_input_shape.dims(); i++)
          onednn_graph_input_shape.push_back(tf_input_shape.dim_size(i));
      }

      l_input_logical_tensor.push_back(dnnl::graph::logical_tensor(
          input_edge_ids_[index], input_data_type, onednn_graph_input_shape,
          dnnl::graph::logical_tensor::layout_type::strided,
          input_constant_property));
      l_input_tensor.push_back(dnnl::graph::tensor(
          l_input_logical_tensor[index], onednn_engine, current_src_ptr));
    }

    // Prepare output logical tensors
    for (int index = 0; index < output_edge_ids_.size(); index++) {
      auto output_data_type =
          graph::GetOneDnnGraphDataType(output_dt_types_[index]);
      l_output_logical_tensor.push_back(dnnl::graph::logical_tensor(
          output_edge_ids_[index], output_data_type,
          -1 /* output shape unknown */,
          dnnl::graph::logical_tensor::layout_type::strided));
    }

    auto c_partition = partition->compile(
        l_input_logical_tensor, l_output_logical_tensor, onednn_engine);

    std::unordered_map<size_t, size_t> inplace_id_map;  // <output_id, input_id>
    GetInplaceIdMap(c_partition, l_input_logical_tensor,
                    l_output_logical_tensor, &inplace_id_map);

    // Prepare output tensors
    for (int index = 0; index < output_edge_ids_.size(); index++) {
      TensorShape tf_shape;
      auto output_logical_tensor =
          c_partition.query_logical_tensor(output_edge_ids_[index]);
      for (int dim : output_logical_tensor.get_dims()) {
        tf_shape.AddDim(dim);
      }

      if (inplace_id_map.find(index) != inplace_id_map.end() &&
          candidate_inplace_input_edge_[inplace_id_map[index]] == true) {
        // TODO(itex): Check whether LLGA and TensorFlow inplace mechanism
        // are exacly the same

        int input_index = inplace_id_map[index];
        const Tensor& input_tensor = ctx->input(input_index);

        if (input_tensor.dtype() != ctx->expected_output_dtype(index)) {
          // Special case for Conv (u8) + Bias + Add (s8) + Relu case
          Tensor& mutable_input_tensor = const_cast<Tensor&>(input_tensor);
          OP_REQUIRES_OK(
              ctx, mutable_input_tensor.BitcastFrom(
                       mutable_input_tensor, ctx->expected_output_dtype(index),
                       mutable_input_tensor.shape()));
        }

        ctx->set_output(index, ctx->input(input_index));

        l_output_tensor.emplace_back(
            dnnl::graph::tensor(output_logical_tensor, onednn_engine,
                                const_cast<void*>(input_tensor.data())));
        ITEX_VLOG(3) << "inplace llga node name: " << this->name();
        ITEX_VLOG(3) << "inplace input index: " << input_index;
        ITEX_VLOG(3) << "inplace logical input tensor id: "
                     << input_edge_ids_[input_index];
      } else {
        Tensor* dst_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(index, tf_shape, &dst_tensor));
        l_output_tensor.emplace_back(dnnl::graph::tensor(
            output_logical_tensor, onednn_engine, dst_tensor->data()));
      }
    }

    // Execute
    c_partition.execute(onednn_stream, l_input_tensor, l_output_tensor);
    ITEX_VLOG(3) << "PARTITION EXECUTED SUCCESSFULLY";
  }

 private:
  int partition_id_;
  std::vector<DataType> output_dt_types_;
  std::vector<int64_t> input_edge_ids_;
  std::vector<int64_t> output_edge_ids_;
  std::vector<bool> is_constant_input_edge_;
  std::vector<bool> candidate_inplace_input_edge_;
  std::vector<string> framework_ops_;
};

#define MATCH_TYPE_AND_SIZE(TYPE) \
  case TYPE:                      \
    return sizeof(EnumToDataType<TYPE>::Type);

int get_sizeof(DataType dt) {
  switch (dt) {
    MATCH_TYPE_AND_SIZE(DT_BFLOAT16);
    MATCH_TYPE_AND_SIZE(DT_HALF);
    MATCH_TYPE_AND_SIZE(DT_FLOAT);
    MATCH_TYPE_AND_SIZE(DT_DOUBLE);
    MATCH_TYPE_AND_SIZE(DT_UINT8);
    MATCH_TYPE_AND_SIZE(DT_INT8);
    MATCH_TYPE_AND_SIZE(DT_INT32);
    MATCH_TYPE_AND_SIZE(DT_QUINT8);
    MATCH_TYPE_AND_SIZE(DT_QINT8);
    MATCH_TYPE_AND_SIZE(DT_QINT32);
    default:
      ITEX_LOG(ERROR) << "Unsupported data type: " << DataTypeString(dt);
      return -1;
  }
}
#undef MATCH_TYPE_AND_SIZE

template <typename Device>
class OneDnnGraphWithLayoutOp : public OpKernel {
 public:
  explicit OneDnnGraphWithLayoutOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("partition_id", &partition_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_dt_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_edge_ids", &input_edge_ids_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_edge_ids", &output_edge_ids_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("is_constant_input_edge", &is_constant_input_edge_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("candidate_inplace_input_edge",
                                     &candidate_inplace_input_edge_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("framework_ops", &framework_ops_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_end_node", &is_end_node_));
  }

  void Compute(OpKernelContext* ctx) {
    ITEX_VLOG(3) << "IN COMPUTE ";

    ITEX_VLOG(3) << "partition_id_ " << partition_id_;
    std::vector<dnnl::graph::logical_tensor> l_input_logical_tensor;
    std::vector<dnnl::graph::logical_tensor> l_output_logical_tensor;
    std::vector<dnnl::graph::tensor> l_input_tensor;
    std::vector<dnnl::graph::tensor> l_output_tensor;

#ifdef ITEX_ONEDNN_3_0
    dnnl::engine onednn_engine = CreateDnnlEngine<Device>(ctx);
    dnnl::stream onednn_stream = CreateDnnlStream(*ctx, onednn_engine);
#else
    dnnl::graph::engine onednn_engine = CreateDnnlEngine<Device>(ctx);
    dnnl::graph::stream onednn_stream =
        CreateDnnlStream<Device>(ctx, onednn_engine);
#endif
    auto partition = std::make_shared<dnnl::graph::partition>(
        graph::GetOneDnnGraphPartition(partition_id_));

    ITEX_CHECK_EQ(input_edge_ids_.size(), is_constant_input_edge_.size());

    // Prepare input tensors and logical tensors
    for (int index = 0; index < ctx->num_inputs() / 2; index++) {
      const Tensor& input = ctx->input(index);
      OneDnnShape src_onednn_shape;
      GetOneDnnShape(ctx, index, &src_onednn_shape);
      void* current_src_ptr = input.data();

      auto input_data_type =
          graph::GetOneDnnGraphDataType(ctx->input(index).dtype());
      std::vector<int64_t> onednn_graph_input_shape;

      auto input_constant_property =
          is_constant_input_edge_[index]
              ? dnnl::graph::logical_tensor::property_type::constant
              : dnnl::graph::logical_tensor::property_type::undef;

      if (src_onednn_shape.IsLLGATensor()) {
        if (src_onednn_shape.GetLayoutId() > 0) {
          l_input_logical_tensor.push_back(dnnl::graph::logical_tensor(
              input_edge_ids_[index], input_data_type,
              src_onednn_shape.GetShape(), src_onednn_shape.GetLayoutId()));
        } else {
          l_input_logical_tensor.push_back(dnnl::graph::logical_tensor(
              input_edge_ids_[index], input_data_type,
              src_onednn_shape.GetShape(), src_onednn_shape.GetStride()));
        }
      } else {
        auto tf_input_shape = ctx->input(index).shape();
        if (tf_input_shape.dims() == 0) {
#ifdef ITEX_ONEDNN_3_0
          onednn_graph_input_shape = {};
#else
          onednn_graph_input_shape = {1};
#endif
          l_input_logical_tensor.push_back(dnnl::graph::logical_tensor(
              input_edge_ids_[index], input_data_type, onednn_graph_input_shape,
              dnnl::graph::logical_tensor::layout_type::strided,
              input_constant_property));
        } else {
          for (int i = 0; i < tf_input_shape.dims(); i++)
            onednn_graph_input_shape.push_back(tf_input_shape.dim_size(i));
          l_input_logical_tensor.push_back(dnnl::graph::logical_tensor(
              input_edge_ids_[index], input_data_type, onednn_graph_input_shape,
              dnnl::graph::logical_tensor::layout_type::strided,
              input_constant_property));
        }
      }

      l_input_tensor.push_back(dnnl::graph::tensor(
          l_input_logical_tensor[index], onednn_engine, current_src_ptr));
    }

    // Prepare output logical tensors
    for (int index = 0; index < output_edge_ids_.size(); index++) {
      auto output_data_type =
          graph::GetOneDnnGraphDataType(output_dt_types_[index]);

      if (is_end_node_[index])
        l_output_logical_tensor.push_back(dnnl::graph::logical_tensor(
            output_edge_ids_[index], output_data_type,
            -1 /* output shape unknown */,
            dnnl::graph::logical_tensor::layout_type::strided));
      else
        l_output_logical_tensor.push_back(dnnl::graph::logical_tensor(
            output_edge_ids_[index], output_data_type,
            -1 /* output shape unknown */,
            dnnl::graph::logical_tensor::layout_type::any));
    }

    auto c_partition = partition->compile(
        l_input_logical_tensor, l_output_logical_tensor, onednn_engine);

    std::unordered_map<size_t, size_t> inplace_id_map;  // <output_id, input_id>
    GetInplaceIdMap(c_partition, l_input_logical_tensor,
                    l_output_logical_tensor, &inplace_id_map);

    // Prepare output tensors
    for (int index = 0; index < output_edge_ids_.size(); index++) {
      auto output_logical_tensor =
          c_partition.query_logical_tensor(output_edge_ids_[index]);
      TensorShape tf_shape;
      if (is_end_node_[index]) {
        auto sizes = output_logical_tensor.get_dims();
        for (int size : sizes) {
          tf_shape.AddDim(size);
        }
      } else {
        auto size = output_logical_tensor.get_mem_size() /
                    get_sizeof(output_dt_types_[index]);
        tf_shape.AddDim(size);
      }

      // Set output OneDnnShape
      OneDnnShape dnn_shape_dst;
      dnn_shape_dst.SetShape(output_logical_tensor.get_dims());
      if (is_end_node_[index]) {
        dnn_shape_dst.SetOneDnnTensor(false);
        // get_layout_id is not allowed for public format, so set 0 here.
        dnn_shape_dst.SetLayoutId(0);
      } else {
        dnn_shape_dst.SetOneDnnTensor(true);
        if (output_logical_tensor.get_layout_type() ==
            dnnl::graph::logical_tensor::layout_type::opaque) {
          dnn_shape_dst.SetLayoutId(output_logical_tensor.get_layout_id());
        } else if (output_logical_tensor.get_layout_type() ==
                   dnnl::graph::logical_tensor::layout_type::strided) {
          dnn_shape_dst.SetStride(output_logical_tensor.get_strides());
        } else {
          ITEX_VLOG(3) << "unsupported logical tensor layout type, opaque or "
                          "strided layout type are expected";
        }
      }

      if (inplace_id_map.find(index) != inplace_id_map.end() &&
          candidate_inplace_input_edge_[inplace_id_map[index]] == true) {
        // TODO(itex): Check whether LLGA and TensorFlow inplace mechanism
        // are exacly the same
        int input_index = inplace_id_map[index];
        const Tensor& input_tensor = ctx->input(input_index);

        if (input_tensor.dtype() != ctx->expected_output_dtype(index)) {
          // Special case for Conv (u8) + Bias + Add (s8) + Relu case
          Tensor& mutable_input_tensor = const_cast<Tensor&>(input_tensor);
          OP_REQUIRES_OK(
              ctx, mutable_input_tensor.BitcastFrom(
                       mutable_input_tensor, ctx->expected_output_dtype(index),
                       mutable_input_tensor.shape()));
        }

        const Tensor& src_tensor = ctx->input(input_index);
        TensorShape src_shape = src_tensor.shape();

        if (tf_shape != src_shape) {
          Tensor dst_tensor;
          ITEX_CHECK(dst_tensor.CopyFrom(src_tensor, tf_shape));
          ctx->set_output(index, dst_tensor);
        } else {
          ctx->set_output(index, src_tensor);
        }

        AllocateMetaData(ctx, index, dnn_shape_dst);
        l_output_tensor.emplace_back(
            dnnl::graph::tensor(output_logical_tensor, onednn_engine,
                                const_cast<void*>(input_tensor.data())));
        ITEX_VLOG(3) << "inplace llga node name: " << this->name();
        ITEX_VLOG(3) << "inplace input index: " << input_index;
        ITEX_VLOG(3) << "inplace logical input tensor id: "
                     << input_edge_ids_[input_index];
      } else {
        Tensor* dst_tensor = nullptr;
        AllocateOutputSetOneDnnShape(ctx, index, &dst_tensor, tf_shape,
                                     dnn_shape_dst);

        l_output_tensor.emplace_back(dnnl::graph::tensor(
            output_logical_tensor, onednn_engine, dst_tensor->data()));
      }
    }

    // Execute
    c_partition.execute(onednn_stream, l_input_tensor, l_output_tensor);
    ITEX_VLOG(3) << "PARTITION EXECUTED SUCCESSFULLY";
  }

 private:
  int partition_id_;
  std::vector<DataType> output_dt_types_;
  std::vector<int64_t> input_edge_ids_;
  std::vector<int64_t> output_edge_ids_;
  std::vector<bool> is_constant_input_edge_;
  std::vector<bool> candidate_inplace_input_edge_;
  std::vector<string> framework_ops_;
  std::vector<bool> is_end_node_;
};

#ifdef INTEL_CPU_ONLY
REGISTER_KERNEL_BUILDER(Name("_OneDnnGraph").Device(DEVICE_CPU),
                        OneDnnGraphWithLayoutOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("OneDnnGraph").Device(DEVICE_CPU),
                        OneDnnGraphOp<CPUDevice>);
#else
REGISTER_KERNEL_BUILDER(Name("_OneDnnGraph")
                            .Device(DEVICE_GPU)
                            .HostMemory("args_meta")
                            .HostMemory("results_meta"),
                        OneDnnGraphWithLayoutOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("OneDnnGraph").Device(DEVICE_GPU),
                        OneDnnGraphOp<GPUDevice>);
#endif
}  // namespace itex
