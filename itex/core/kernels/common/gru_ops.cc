/* Copyright (c) 2022 Intel Corporation

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

#include <string>
#include <unordered_map>
#include <vector>

#include "itex/core/utils/errors.h"
#include "itex/core/utils/onednn/onednn_layout_util.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/types.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"
using dnnl::augru_forward;
using dnnl::engine;
using dnnl::gru_forward;
using dnnl::memory;
using dnnl::prop_kind;
using dnnl::rnn_direction;

namespace itex {

/*=================================================================
  GRU Forward op
==================================================================*/
template <typename Device, typename T, typename GruType>
class OneDnnGRUForwardOp : public OpKernel {
 protected:
  bool is_filter_const_ = false;
  WeightCacheManager<T> weight_layer_cache_manager_;
  WeightCacheManager<T> weight_iter_cache_manager_;
  BiasCacheManager<T> bias_cache_manager_;
  typename GruType::primitive_desc augru_pd;
  GruType augru_prim;

 public:
  explicit OneDnnGRUForwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    if (ctx->HasAttr("is_filter_const")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("is_filter_const", &is_filter_const_));
    }
  }

  ~OneDnnGRUForwardOp() {}

  void Compute(OpKernelContext* ctx) override {
    // Tensors for input/output memory.
    const Tensor* x_tensor = nullptr;
    const Tensor* h_prev_tensor = nullptr;
    const Tensor* w_ru_tensor = nullptr;
    const Tensor* w_c_tensor = nullptr;
    const Tensor* b_ru_tensor = nullptr;
    const Tensor* b_c_tensor = nullptr;
    const Tensor* au_x_tensor = nullptr;
    Tensor* h_tensor = nullptr;

    // Tensors for internal memory space.
    Tensor user_weights_layer_tensor;
    Tensor user_weights_iter_tensor;
    Tensor bias_tensor;
    Tensor augru_weights_layer_tensor;
    Tensor augru_weights_iter_tensor;
    Tensor dst_iter_tensor;
    Tensor workspace_tensor;
    Tensor scratchpad_tensor;

    memory::dim N = 1,  // batch size
        TimeS = 1,      // time steps
        Channels = 1,   // channels
        Gates = 3,      // gates
        Layers = 1,     // layers
        Dir = 1,        // direction
        biasG = Gates, Inputs = 1;

    InitInputsAndOutputs(ctx, &x_tensor, &h_prev_tensor, &au_x_tensor,
                         &h_tensor, &TimeS, &N, &Channels, &Inputs);

    //
    // Create memory dims.
    //
    memory::dims src_dims = {TimeS, N, Channels};
    memory::dims src_iter_dims = {Layers, Dir, N, Channels};
    memory::dims dst_dims = {TimeS, N, Channels};
    memory::dims weights_dims = {Layers, Dir, Inputs, Gates, Channels};
    memory::dims weights_iter_dims = {Layers, Dir, Channels, Gates, Channels};
    memory::dims bias_dims = {Layers, Dir, biasG, Channels};
    memory::dims attention_dims = {TimeS, N, 1};

    auto dnnl_engine = CreateDnnlEngine<Device>(*ctx);
    auto dnnl_stream = CreateDnnlStream(*ctx, dnnl_engine);

    if (augru_pd.get(true) == nullptr) {
      //
      // Create memory descriptors.
      //
      auto src_layer_md =
          memory::desc(src_dims, OneDnnType<T>(), memory::format_tag::tnc);
      auto src_iter_md = memory::desc(src_iter_dims, OneDnnType<T>(),
                                      memory::format_tag::ldnc);
      auto attention_md = memory::desc(attention_dims, OneDnnType<T>(),
                                       memory::format_tag::tnc);
      auto bias_md =
          memory::desc(bias_dims, OneDnnType<T>(), memory::format_tag::ldgo);
      auto dst_layer_md =
          memory::desc(dst_dims, OneDnnType<T>(), memory::format_tag::tnc);

      // Create memory descriptors for weights with format_tag::any. This
      // enables the AUGRU primitive to choose the optimized memory layout.
      auto augru_weights_layer_md =
          memory::desc(weights_dims, OneDnnType<T>(), memory::format_tag::any);
      auto augru_weights_iter_md = memory::desc(
          weights_iter_dims, OneDnnType<T>(), memory::format_tag::any);

      // Optional memory descriptors for recurrent data.
      auto dst_iter_md = memory::desc();

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifndef ITEX_ONEDNN_3_0
      // Create operation descriptor.
      typename GruType::desc* desc;
      auto augru_desc = augru_forward::desc(
          prop_kind::forward_inference,
          rnn_direction::unidirectional_left2right, src_layer_md, src_iter_md,
          attention_md, augru_weights_layer_md, augru_weights_iter_md, bias_md,
          dst_layer_md, dst_iter_md);
      auto gru_desc = gru_forward::desc(
          prop_kind::forward_inference,
          rnn_direction::unidirectional_left2right, src_layer_md, src_iter_md,
          augru_weights_layer_md, augru_weights_iter_md, bias_md, dst_layer_md,
          dst_iter_md);
      if (std::is_same<GruType, augru_forward>()) {
        desc = reinterpret_cast<typename GruType::desc*>(&augru_desc);
      } else {
        desc = reinterpret_cast<typename GruType::desc*>(&gru_desc);
      }
      augru_pd = typename GruType::primitive_desc(*desc, attr, dnnl_engine);
#else
      dnnl::augru_forward::primitive_desc augru_pd_tmp(
          dnnl_engine, prop_kind::forward_inference,
          rnn_direction::unidirectional_left2right, src_layer_md, src_iter_md,
          attention_md, augru_weights_layer_md, augru_weights_iter_md, bias_md,
          dst_layer_md, dst_iter_md, attr);
      dnnl::gru_forward::primitive_desc gru_pd_tmp(
          dnnl_engine, prop_kind::forward_inference,
          rnn_direction::unidirectional_left2right, src_layer_md, src_iter_md,
          augru_weights_layer_md, augru_weights_iter_md, bias_md, dst_layer_md,
          dst_iter_md, attr);

      if (std::is_same<GruType, augru_forward>()) {
        augru_pd =
            *reinterpret_cast<typename GruType::primitive_desc*>(&augru_pd_tmp);
      } else {
        augru_pd =
            *reinterpret_cast<typename GruType::primitive_desc*>(&gru_pd_tmp);
      }
#endif
    }

    //
    // Prepare memory contents.
    //
    ProcessInputs(&x_tensor, &h_prev_tensor, &au_x_tensor, ctx, dnnl_engine);
    auto src_layer_mem = CreateDnnlMemory(augru_pd.src_layer_desc(),
                                          dnnl_engine, x_tensor->data());
    auto src_iter_mem = CreateDnnlMemory(augru_pd.src_iter_desc(), dnnl_engine,
                                         h_prev_tensor->data());
    auto dst_layer_mem = CreateDnnlMemory(augru_pd.dst_layer_desc(),
                                          dnnl_engine, h_tensor->data());
    memory attention_mem;
    if (std::is_same<GruType, augru_forward>()) {
      attention_mem = CreateDnnlMemory(
          (reinterpret_cast<augru_forward::primitive_desc*>(&augru_pd))
              ->attention_desc(),
          dnnl_engine, au_x_tensor->data());
    }

    memory augru_weights_layer_mem;
    memory augru_weights_iter_mem;
    memory bias_mem;

    bool NeedPrepareMemory = true;
    if (is_filter_const_ && augru_pd.get(true) != nullptr) {
      void* weight_layer_cached_data = nullptr;
      void* weight_iter_cached_data = nullptr;
      void* bias_cached_data = nullptr;

      if (!weight_layer_cache_manager_.IsEmpty())
        weight_layer_cached_data =
            weight_layer_cache_manager_.GetCache(ctx, augru_pd.weights_desc());
      if (!weight_iter_cache_manager_.IsEmpty())
        weight_iter_cached_data = weight_iter_cache_manager_.GetCache(
            ctx, augru_pd.weights_iter_desc());
      if (!bias_cache_manager_.IsEmpty())
        bias_cached_data = bias_cache_manager_.GetCache(ctx);

      //
      // Only weight_layer, weight_iter and bias are all get cache success, we
      // use the cached contents. If any of them get cache fail, we will
      // re-generate these memory.
      //
      if (weight_layer_cached_data != nullptr &&
          weight_iter_cached_data != nullptr && bias_cached_data != nullptr) {
        NeedPrepareMemory = false;
        augru_weights_layer_mem = CreateDnnlMemory(
            augru_pd.weights_desc(), dnnl_engine, weight_layer_cached_data);
        augru_weights_iter_mem = CreateDnnlMemory(
            augru_pd.weights_iter_desc(), dnnl_engine, weight_iter_cached_data);
        bias_mem = CreateDnnlMemory(augru_pd.bias_desc(), dnnl_engine,
                                    bias_cached_data);
      }
    }

    //
    // The Gru/Augru weight/bias always need concat the data in w_ru_tensor and
    // w_c_tensor, b_ru_tensor and b_c_tensor. So, here always first generate
    // the weight, bias's memory. Then if is_filter_const_, use cache manager to
    // cache them.
    //
    if (NeedPrepareMemory) {
      InitWeights(ctx, &w_ru_tensor, &w_c_tensor, &b_ru_tensor, &b_c_tensor,
                  Channels, Inputs);

      auto weights_layer_md = memory::desc(weights_dims, OneDnnType<T>(),
                                           memory::format_tag::ldgoi);
      auto weights_iter_md = memory::desc(weights_iter_dims, OneDnnType<T>(),
                                          memory::format_tag::ldgoi);

      // Create memory objects for weights/bias.
      auto user_weights_layer_mem = AllocateMemory(
          ctx, weights_layer_md, dnnl_engine, &user_weights_layer_tensor);
      auto user_weights_iter_mem = AllocateMemory(
          ctx, weights_iter_md, dnnl_engine, &user_weights_iter_tensor);
      bias_mem =
          AllocateMemory(ctx, augru_pd.bias_desc(), dnnl_engine, &bias_tensor);

      int input_size = Inputs;
      int cell_size = Channels;

      auto w_ru_tensor_mem =
          memory(memory::desc(
                     {1, 1, w_ru_tensor->dim_size(0), w_ru_tensor->dim_size(1)},
                     OneDnnType<T>(), memory::format_tag::abcd),
                 dnnl_engine, w_ru_tensor->data());
      auto w_c_tensor_mem = memory(
          memory::desc({1, 1, w_c_tensor->dim_size(0), w_c_tensor->dim_size(1)},
                       OneDnnType<T>(), memory::format_tag::abcd),
          dnnl_engine, w_c_tensor->data());

      auto temp_weights_layer_mem =
          memory(memory::desc({1, 1, input_size, cell_size * 3},
                              OneDnnType<T>(), memory::format_tag::abdc),
                 dnnl_engine, user_weights_layer_mem.get_data_handle());
      auto temp_weights_iter_mem =
          memory(memory::desc({1, 1, cell_size, cell_size * 3}, OneDnnType<T>(),
                              memory::format_tag::abdc),
                 dnnl_engine, user_weights_iter_mem.get_data_handle());

      SliceCopy(&temp_weights_layer_mem, {0, 0}, {input_size, cell_size},
                &w_ru_tensor_mem, {0, cell_size}, {input_size, cell_size}, ctx,
                dnnl_engine);
      SliceCopy(&temp_weights_layer_mem, {0, cell_size},
                {input_size, cell_size}, &w_ru_tensor_mem, {0, 0},
                {input_size, cell_size}, ctx, dnnl_engine);
      SliceCopy(&temp_weights_layer_mem, {0, cell_size * 2},
                {input_size, cell_size}, &w_c_tensor_mem, {0, 0},
                {input_size, cell_size}, ctx, dnnl_engine);
      SliceCopy(&temp_weights_iter_mem, {0, 0}, {cell_size, cell_size},
                &w_ru_tensor_mem, {input_size, cell_size},
                {cell_size, cell_size}, ctx, dnnl_engine);
      SliceCopy(&temp_weights_iter_mem, {0, cell_size}, {cell_size, cell_size},
                &w_ru_tensor_mem, {input_size, 0}, {cell_size, cell_size}, ctx,
                dnnl_engine);
      SliceCopy(&temp_weights_iter_mem, {0, cell_size * 2},
                {cell_size, cell_size}, &w_c_tensor_mem, {input_size, 0},
                {cell_size, cell_size}, ctx, dnnl_engine);

      auto b_ru_tensor_mem =
          memory(memory::desc({1, 1, 2, cell_size}, OneDnnType<T>(),
                              memory::format_tag::ldgo),
                 dnnl_engine, b_ru_tensor->data());
      auto b_c_tensor_mem =
          memory(memory::desc({1, 1, 1, cell_size}, OneDnnType<T>(),
                              memory::format_tag::ldgo),
                 dnnl_engine, b_c_tensor->data());

      SliceCopy(&bias_mem, {0, 0}, {2, cell_size}, &b_ru_tensor_mem, {0, 0},
                {2, cell_size}, ctx, dnnl_engine);
      SliceCopy(&bias_mem, {2, 0}, {1, cell_size}, &b_c_tensor_mem, {0, 0},
                {1, cell_size}, ctx, dnnl_engine);

      // For now, assume that the weights memory layout generated by the
      // primitive and the ones provided by the user are identical.

      augru_weights_layer_mem = user_weights_layer_mem;
      augru_weights_iter_mem = user_weights_iter_mem;

      // Reorder the data in case the weights memory layout generated by the
      // primitive and the one provided by the user are different. In this case,
      // we create additional memory objects with internal buffers that will
      // contain the reordered data.
      if (augru_pd.weights_desc() != user_weights_layer_mem.get_desc()) {
        augru_weights_layer_mem =
            AllocateMemory(ctx, augru_pd.weights_desc(), dnnl_engine,
                           &augru_weights_layer_tensor);
        ReorderMemory(*ctx, &user_weights_layer_mem, &augru_weights_layer_mem,
                      dnnl_engine);
      }

      if (augru_pd.weights_iter_desc() != user_weights_iter_mem.get_desc()) {
        augru_weights_iter_mem =
            AllocateMemory(ctx, augru_pd.weights_iter_desc(), dnnl_engine,
                           &augru_weights_iter_tensor);
        ReorderMemory(*ctx, &user_weights_iter_mem, &augru_weights_iter_mem,
                      dnnl_engine);
      }

      if (is_filter_const_) {
        weight_layer_cache_manager_.SetCache(
            ctx, user_weights_layer_mem.get_desc(), augru_pd.weights_desc(),
            user_weights_layer_mem.get_data_handle(), dnnl_engine);
        weight_iter_cache_manager_.SetCache(
            ctx, user_weights_iter_mem.get_desc(), augru_pd.weights_iter_desc(),
            user_weights_iter_mem.get_data_handle(), dnnl_engine);
        bias_cache_manager_.SetCache(ctx, augru_pd.bias_desc(),
                                     dnnl::primitive_attr(),
                                     bias_mem.get_data_handle(), dnnl_engine);
      }
    }

    //
    // Start primitive execution.
    //
    // Create the primitive.
    if (augru_prim.get(true) == nullptr) augru_prim = GruType(augru_pd);

    // Primitive arguments
    std::unordered_map<int, memory> augru_args;
    augru_args.insert({DNNL_ARG_SRC_LAYER, src_layer_mem});
    augru_args.insert({DNNL_ARG_WEIGHTS_LAYER, augru_weights_layer_mem});
    augru_args.insert({DNNL_ARG_WEIGHTS_ITER, augru_weights_iter_mem});
    augru_args.insert({DNNL_ARG_BIAS, bias_mem});
    augru_args.insert({DNNL_ARG_DST_LAYER, dst_layer_mem});
    augru_args.insert({DNNL_ARG_SRC_ITER, src_iter_mem});

    if (augru_pd.dst_iter_desc().get_size() != 0) {
      auto dst_iter_mem = AllocateMemory(ctx, augru_pd.dst_iter_desc(),
                                         dnnl_engine, &dst_iter_tensor);
      augru_args.insert({DNNL_ARG_DST_ITER, dst_iter_mem});
    }
    if (augru_pd.workspace_desc().get_size() != 0) {
      auto workspace_mem = AllocateMemory(ctx, augru_pd.workspace_desc(),
                                          dnnl_engine, &workspace_tensor);
      augru_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});
    }
    if (augru_pd.scratchpad_desc().get_size() != 0) {
      auto scratchpad_mem = AllocateMemory(ctx, augru_pd.scratchpad_desc(),
                                           dnnl_engine, &scratchpad_tensor);
      augru_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});
    }

    if (std::is_same<GruType, augru_forward>())
      augru_args.insert({DNNL_ARG_AUGRU_ATTENTION, attention_mem});

    // Primitive execution: AUGRU.
    augru_prim.execute(dnnl_stream, augru_args);

    // Wait for the computation to finalize.
    dnnl_stream.wait();

    ProcessOutputs(dst_layer_mem, ctx, dnnl_engine);
  }

  inline void InitInputsAndOutputs(
      OpKernelContext* ctx, const Tensor** x_tensor,
      const Tensor** h_prev_tensor, const Tensor** au_x_tensor,
      Tensor** h_tensor, memory::dim* TimeDim, memory::dim* batch_size,
      memory::dim* cell_size, memory::dim* input_size) {
    OP_REQUIRES_OK(ctx, ctx->input("x", x_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", h_prev_tensor));

    if (std::is_same<GruType, augru_forward>()) {
      OP_REQUIRES_OK(ctx, ctx->input("au_x", au_x_tensor));
    }

    GetDimsInfoFromInputs(ctx, *x_tensor, *h_prev_tensor, TimeDim, batch_size,
                          cell_size, input_size);
    CheckInputShapes(ctx, *h_prev_tensor, *batch_size, *cell_size, *input_size);
    CheckPrimitiveCache(*TimeDim, *batch_size, *cell_size, *input_size);
    CreateOutputs(ctx, h_tensor, *TimeDim, *batch_size, *cell_size);
  }

  inline void CheckPrimitiveCache(memory::dim TimeDim, memory::dim batch_size,
                                  memory::dim cell_size,
                                  memory::dim input_size) {
    if (augru_pd.get(true) == nullptr) return;
#ifdef ITEX_ONEDNN_3_0
    auto src_dims = augru_pd.src_layer_desc().get_dims();
#else
    auto src_dims = augru_pd.src_layer_desc().dims();
#endif
    if (src_dims[0] == TimeDim && src_dims[1] == batch_size &&
        src_dims[2] == cell_size)
      return;

    augru_pd.reset(nullptr);
    augru_prim.reset(nullptr);
  }

  inline void InitWeights(OpKernelContext* ctx, const Tensor** w_ru_tensor,
                          const Tensor** w_c_tensor, const Tensor** b_ru_tensor,
                          const Tensor** b_c_tensor, memory::dim cell_size,
                          memory::dim input_size) {
    OP_REQUIRES_OK(ctx, ctx->input("w_ru", w_ru_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("w_c", w_c_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("b_ru", b_ru_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("b_c", b_c_tensor));

    CheckWeightsShapes(ctx, *w_ru_tensor, *w_c_tensor, *b_ru_tensor,
                       *b_c_tensor, cell_size, input_size);
  }

  inline void CheckInputShapes(OpKernelContext* ctx,
                               const Tensor* h_prev_tensor,
                               memory::dim batch_size, memory::dim cell_size,
                               memory::dim input_size) {
    OP_REQUIRES(ctx, input_size == cell_size,
                errors::InvalidArgument("input_size != cell_size: ", input_size,
                                        " vs. ", cell_size));

    // Shape of 'h' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));
  }

  inline void CheckWeightsShapes(
      OpKernelContext* ctx, const Tensor* w_ru_tensor, const Tensor* w_c_tensor,
      const Tensor* b_ru_tensor, const Tensor* b_c_tensor,
      memory::dim cell_size, memory::dim input_size) {
    // Shape of 'w_ru' must be [input_size+cell_size, 2*cell_size]
    OP_REQUIRES(ctx, w_ru_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_ru.dim_size(0) != input_size + cell_size: ",
                    w_ru_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_ru_tensor->dim_size(1) == cell_size * 2,
                errors::InvalidArgument("w_ru.dim_size(1) != cell_size * 2: ",
                                        w_ru_tensor->dim_size(1), " vs. ",
                                        cell_size * 2));

    // Shape of 'w_c' must be [input_size+cell_size, cell_size]
    OP_REQUIRES(ctx, w_c_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(0) != input_size + cell_size: ",
                    w_c_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_c_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(1) != cell_size: ", w_c_tensor->dim_size(1),
                    " vs. ", cell_size));

    // Shape of 'b_ru' must be [2*cell_size]
    OP_REQUIRES(ctx, b_ru_tensor->dim_size(0) == cell_size * 2,
                errors::InvalidArgument("b_ru.dim_size(0) != cell_size * 2: ",
                                        b_ru_tensor->dim_size(0), " vs. ",
                                        cell_size * 2));

    OP_REQUIRES(ctx, b_ru_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_ru must be 1",
                                        b_ru_tensor->dims(), " vs. 1", 1));
    // Shape of 'b_c' must be [cell_size]
    OP_REQUIRES(ctx, b_c_tensor->dim_size(0) == cell_size,
                errors::InvalidArgument(
                    "b_c.dim_size(0) != cell_size: ", b_c_tensor->dim_size(0),
                    " vs. ", cell_size));
    OP_REQUIRES(ctx, b_c_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_c must be 1",
                                        b_c_tensor->dims(), " vs. 1"));
  }

  void SliceCopy(memory* dst, std::vector<int64> dst_offset,
                 std::vector<int64> dst_size, memory* src,
                 std::vector<int64> src_offset, std::vector<int64> src_size,
                 OpKernelContext* ctx, const engine& dnnl_engine) {
    memory::desc dst_desc = dst->get_desc();
    memory::desc src_desc = src->get_desc();
    memory dst_sub_mem =
        memory(dst_desc.submemory_desc({1, 1, dst_size[0], dst_size[1]},
                                       {0, 0, dst_offset[0], dst_offset[1]}),
               dnnl_engine, dst->get_data_handle());
    memory src_sub_mem =
        memory(src_desc.submemory_desc({1, 1, src_size[0], src_size[1]},
                                       {0, 0, src_offset[0], src_offset[1]}),
               dnnl_engine, src->get_data_handle());

    ReorderMemory(*ctx, &src_sub_mem, &dst_sub_mem, dnnl_engine);
  }

  void* AllocateTensorMemory(OpKernelContext* ctx, const memory::desc& desc,
                             Tensor* tensor) {
    int64_t reorder_size = desc.get_size() / sizeof(T);
    OP_REQUIRES_OK_PTR(ctx,
                       ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                          TensorShape({reorder_size}), tensor));
    return GetTensorBuffer<T>(tensor);
  }

  inline memory AllocateMemory(OpKernelContext* ctx, const memory::desc& desc,
                               const engine& dnnl_engine, Tensor* tensor) {
    return CreateDnnlMemory(desc, dnnl_engine,
                            AllocateTensorMemory(ctx, desc, tensor));
  }

  virtual void GetDimsInfoFromInputs(
      OpKernelContext* ctx, const Tensor* x_tensor, const Tensor* h_prev_tensor,
      memory::dim* TimeDim, memory::dim* batch_size, memory::dim* cell_size,
      memory::dim* input_size) {
    *TimeDim = 1;
    *batch_size = x_tensor->dim_size(0);
    *input_size = x_tensor->dim_size(1);
    *cell_size = h_prev_tensor->dim_size(1);
  }

  virtual void CreateOutputs(OpKernelContext* ctx, Tensor** h_tensor,
                             memory::dim TimeDim, memory::dim batch_size,
                             memory::dim cell_size) {
    const int kInputIdx_h_prev = 1, kOutputIdx_h = 3;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {kInputIdx_h_prev}, kOutputIdx_h,
                            TensorShape({batch_size, cell_size}), h_tensor));
  }

  virtual void ProcessOutputs(const memory& output_mem, OpKernelContext* ctx,
                              const engine& dnnl_engine) {}

  virtual void ProcessInputs(const Tensor** x_tensor,
                             const Tensor** h_prev_tensor,
                             const Tensor** au_x_tensor, OpKernelContext* ctx,
                             const engine& dnnl_engine) {}
};

template <typename Device, typename T, typename GruType>
class MklGRUForwardOp : public OneDnnGRUForwardOp<Device, T, GruType> {
 protected:
  bool X_format_tnc = true;
  bool AUX_format_tnc = true;
  Tensor** h_n_tensor = nullptr;
  Tensor* x_reorder_tensor = nullptr;
  Tensor* au_x_reorder_tensor = nullptr;

 public:
  explicit MklGRUForwardOp(OpKernelConstruction* ctx)
      : OneDnnGRUForwardOp<Device, T, GruType>(ctx) {
    std::string format = "";
    if (ctx->HasAttr("x_format")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("x_format", &format));
      X_format_tnc = (format == "TNC");
    }
    format = "";
    if (ctx->HasAttr("au_format")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("au_format", &format));
      AUX_format_tnc = (format == "TNC");
    }
  }

  ~MklGRUForwardOp() {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* h_n_tensor_local;
    Tensor x_reorder_tensor_local;
    Tensor au_x_reorder_tensor_local;
    h_n_tensor = &h_n_tensor_local;
    x_reorder_tensor = &x_reorder_tensor_local;
    au_x_reorder_tensor = &au_x_reorder_tensor_local;
    this->OneDnnGRUForwardOp<Device, T, GruType>::Compute(ctx);
    h_n_tensor = nullptr;
    x_reorder_tensor = nullptr;
    au_x_reorder_tensor = nullptr;
  }

  void GetDimsInfoFromInputs(OpKernelContext* ctx, const Tensor* x_tensor,
                             const Tensor* h_prev_tensor, memory::dim* TimeDim,
                             memory::dim* batch_size, memory::dim* cell_size,
                             memory::dim* input_size) {
    if (X_format_tnc) {
      *TimeDim = x_tensor->dim_size(0);
      *batch_size = x_tensor->dim_size(1);
    } else {
      *TimeDim = x_tensor->dim_size(1);
      *batch_size = x_tensor->dim_size(0);
    }
    *input_size = x_tensor->dim_size(2);
    *cell_size = h_prev_tensor->dim_size(1);
  }
  void CreateOutputs(OpKernelContext* ctx, Tensor** h_tensor,
                     memory::dim TimeDim, memory::dim batch_size,
                     memory::dim cell_size) {
    const int kOutputIdx_h_out = 0, kOutputIdx_h_n = 1;

    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(kOutputIdx_h_out,
                                  TensorShape({TimeDim, batch_size, cell_size}),
                                  h_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            kOutputIdx_h_n,
                            TensorShape({batch_size, cell_size}), h_n_tensor));
  }

  void ProcessOutputs(const memory& output_mem, OpKernelContext* ctx,
                      const engine& dnnl_engine) {
    memory dst_sub_mem = CreateDnnlMemory(
        memory::desc(
            {1, 1, (*h_n_tensor)->dim_size(0), (*h_n_tensor)->dim_size(1)},
            OneDnnType<T>(), memory::format_tag::abcd),
        dnnl_engine, GetTensorBuffer<T>(*h_n_tensor));
    auto h_out_desc = output_mem.get_desc();
#ifdef ITEX_ONEDNN_3_0
    auto src_dims = h_out_desc.get_dims();
#else
    auto src_dims = h_out_desc.dims();
#endif
    auto src_desc = memory::desc({1, src_dims[0], src_dims[1], src_dims[2]},
                                 OneDnnType<T>(), memory::format_tag::abcd);
    memory src_sub_mem =
        memory(src_desc.submemory_desc({1, 1, src_dims[1], src_dims[2]},
                                       {0, src_dims[0] - 1, 0, 0}),
               dnnl_engine, output_mem.get_data_handle());

    ReorderMemory(*ctx, &src_sub_mem, &dst_sub_mem, dnnl_engine);
  }

  void ProcessInputs(const Tensor** x_tensor, const Tensor** h_prev_tensor,
                     const Tensor** au_x_tensor, OpKernelContext* ctx,
                     const engine& dnnl_engine) {
    if (!X_format_tnc)
      *x_tensor = ReorderInput(*x_tensor, x_reorder_tensor, ctx, dnnl_engine);
    if (std::is_same<GruType, augru_forward>()) {
      if (!AUX_format_tnc)
        *au_x_tensor =
            ReorderInput(*au_x_tensor, au_x_reorder_tensor, ctx, dnnl_engine);
    }
  }

  Tensor* ReorderInput(const Tensor* reorder_tensor, Tensor* reordered_tensor,
                       OpKernelContext* ctx, const engine& dnnl_engine) {
    int first_dim = reorder_tensor->dim_size(0);
    int secon_dim = reorder_tensor->dim_size(1);
    int third_dim = reorder_tensor->dim_size(2);
    dnnl::memory::dims src_dims{first_dim, secon_dim, third_dim};
    dnnl::memory::dims dst_dims{first_dim, secon_dim, third_dim};
    auto src_mem = CreateDnnlMemory(
        memory::desc(src_dims, OneDnnType<T>(), memory::format_tag::ntc),
        dnnl_engine, GetTensorBuffer<T>(reorder_tensor));
    auto dst_mem = this->AllocateMemory(
        ctx, memory::desc(dst_dims, OneDnnType<T>(), memory::format_tag::tnc),
        dnnl_engine, reordered_tensor);
    ReorderMemory(*ctx, &src_mem, &dst_mem, dnnl_engine);
    return reordered_tensor;
  }
};

// Register DNN kernels for supported operations and supported types - right now
// Register the Block GRU cell kernel for CPU.
#ifdef INTEL_CPU_ONLY
#define REGISTER_GRU_KERNELS(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXGRUCell").Device(DEVICE_CPU).TypeConstraint<T>("T"),      \
      OneDnnGRUForwardOp<CPUDevice, T, dnnl::gru_forward>);                \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXAUGRUCell").Device(DEVICE_CPU).TypeConstraint<T>("T"),    \
      OneDnnGRUForwardOp<CPUDevice, T, dnnl::augru_forward>);              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXForwardGRU").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      MklGRUForwardOp<CPUDevice, T, dnnl::gru_forward>);                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ITEXForwardAUGRU").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MklGRUForwardOp<CPUDevice, T, dnnl::augru_forward>);                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MklGRU").Device(DEVICE_CPU).TypeConstraint<T>("T"),            \
      MklGRUForwardOp<CPUDevice, T, dnnl::gru_forward>);                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MklAUGRU").Device(DEVICE_CPU).TypeConstraint<T>("T"),          \
      MklGRUForwardOp<CPUDevice, T, dnnl::augru_forward>);

TF_CALL_CPU_NUMBER_TYPES(REGISTER_GRU_KERNELS);
#undef REGISTER_GRU_KERNELS
#else
// TODO(itex): Implement GRU/AUGRU for GPU.
#endif  // INTEL_CPU_ONLY

}  // namespace itex
