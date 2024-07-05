/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/stateful_random_ops.h"

#include "itex/core/kernels/common/fill_functor.h"
#include "itex/core/kernels/gpu/random_op_gpu.h"
#include "itex/core/kernels/gpu/training_op_helpers.h"
#include "itex/core/utils/bounds_check.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/lib/random/philox_random.h"
#include "itex/core/utils/lib/random/random_distributions.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

template <typename Distribution>
struct FillKernelTask {
  FillKernelTask(sycl::local_accessor<char, 1> local_philox_acc,
                 StateElementType* state_data,
                 typename Distribution::ResultElementType* output_data,
                 int64_t output_size, sycl::accessor<int, 1> item_count_acc,
                 Distribution dist)
      :

        local_philox_acc(local_philox_acc),
        state_data(state_data),
        output_data(output_data),
        output_size(output_size),
        item_count_acc(item_count_acc),
        dist(dist) {}
  void operator()(sycl::nd_item<1> myItem) const {
    // Items in a group share `philox`. Item 0 is responsible for
    // initializing it.
    char* philox_raw = local_philox_acc.get_pointer();
    PhiloxRandom* philox = reinterpret_cast<PhiloxRandom*>(philox_raw);
    auto id = myItem.get_local_id()[0];
    if (id == 0) {
      *philox = GetPhiloxRandomFromMem(state_data);
    }
    sycl::group_barrier(myItem.get_group(), sycl::memory_scope_work_group);

    functor::FillPhiloxRandomKernel<Distribution,
                                    Distribution::kVariableSamplesPerOutput>
        f(output_data, output_size, *philox, const_cast<Distribution&>(dist),
          nullptr, nullptr);
    f(myItem);
    // The last item updates the state.
    auto total_item_count = myItem.get_global_range()[0];
    auto item_count_ptr =
        item_count_acc.template get_multi_ptr<sycl::access::decorated::no>();
    auto atomic_val =
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(
            *item_count_ptr);
    auto old_counter_value = atomic_val.fetch_add(1);
    if (old_counter_value == total_item_count - 1) {
      UpdateMemWithPhiloxRandom(*philox, output_size, state_data);
    }
  }

 private:
  sycl::local_accessor<char, 1> local_philox_acc;
  StateElementType* state_data;
  typename Distribution::ResultElementType* output_data;
  int64_t output_size;
  sycl::accessor<int, 1> item_count_acc;
  Distribution dist;
};

using random::PhiloxRandom;

template <typename Distribution>
void FillKernel(const GPUDevice& d, const int total_count, Distribution dist,
                int64 state_size, int64 output_size,
                StateElementType* state_data,
                typename Distribution::ResultElementType* output_data) {
  auto stream = d.stream();
  auto work_group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto work_group = (total_count + work_group_size - 1) / work_group_size;

  int item_count = 0;
  sycl::buffer<int, 1> item_count_buf{&item_count, 1};

  stream->submit([&](sycl::handler& cgh) {
    auto item_count_acc =
        item_count_buf.get_access<sycl::access::mode::read_write,
                                  sycl::access::target::device>(cgh);
    sycl::local_accessor<char, 1> local_philox_acc(
        sycl::range<1>(sizeof(PhiloxRandom)), cgh);
    FillKernelTask<Distribution> task(local_philox_acc, state_data, output_data,
                                      output_size, item_count_acc, dist);
    cgh.parallel_for<FillKernelTask<Distribution>>(
        sycl::nd_range<1>(sycl::range<1>(work_group * work_group_size),
                          sycl::range<1>(work_group_size)),
        task);
  });
}

template <typename Distribution>
void UpdateVariableAndFill_Philox<GPUDevice, Distribution>::operator()(
    OpKernelContext* ctx, const GPUDevice& d, Distribution dist,
    int64 output_size, int64 alg_tag_skip, Tensor* state_tensor,
    typename Distribution::ResultElementType* output_data) {
  OP_REQUIRES(
      ctx, alg_tag_skip == 0,
      errors::InvalidArgument(
          "GPU kernel doesn't support reading algorithm from state variable, "
          "so alg_tag_skip must be 0; got",
          alg_tag_skip));
  auto state_tensor_flat = state_tensor->flat<StateElementType>();
  auto state_size = state_tensor_flat.size();
  auto state_data = state_tensor_flat.data();

  // maximize occupancy
  const int kGroupSize = Distribution::kResultElementCount;
  int work_element_count = (output_size + kGroupSize - 1) / kGroupSize;

  FillKernel<Distribution>(d, work_element_count, dist, state_size, output_size,
                           state_data, output_data);
}

class SkipKernelTask;

// Precondition: there is only 1 work_group and 1 item.
void SkipKernel(const GPUDevice& d, const StateElementType* in_data,
                int64 delta, StateElementType* out_data) {
  auto stream = d.stream();
  stream->submit([&](sycl::handler& cgh) {
    cgh.single_task<SkipKernelTask>([=]() {
      auto philox = GetPhiloxRandomFromMem(in_data);
      UpdateMemWithPhiloxRandom(philox, delta, out_data);
    });
  });
}

void RngSkip_Philox<GPUDevice>::operator()(const GPUDevice& d,
                                           const StateElementType* in_data,
                                           uint64 delta,
                                           StateElementType* out_data) {
  SkipKernel(d, in_data, delta, out_data);
}

// Explicit instantiation of the GPU distributions functors.

template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, Eigen::half>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, float>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice,
    random::NormalDistribution<random::PhiloxRandom, Eigen::bfloat16>>;

template struct UpdateVariableAndFill_Philox<
    GPUDevice,
    random::TruncatedNormalDistribution<
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::half>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, float>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice,
    random::TruncatedNormalDistribution<
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::bfloat16>>;

template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, Eigen::half>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, float>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice,
    random::UniformDistribution<random::PhiloxRandom, Eigen::bfloat16>>;

template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int32>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int64>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformFullIntDistribution<random::PhiloxRandom, int32>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformFullIntDistribution<random::PhiloxRandom, int64>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice,
    random::UniformFullIntDistribution<random::PhiloxRandom, uint32>>;
template struct UpdateVariableAndFill_Philox<
    GPUDevice,
    random::UniformFullIntDistribution<random::PhiloxRandom, uint64>>;

Status CheckState(const Tensor& state) {
  if (state.dtype() != STATE_ELEMENT_DTYPE) {
    return errors::InvalidArgument("dtype of RNG state variable must be ",
                                   DataTypeString(STATE_ELEMENT_DTYPE),
                                   ", not ", DataTypeString(state.dtype()));
  }
  if (state.dims() != 1) {
    return errors::InvalidArgument(
        "RNG state must have one and only one dimension, not ", state.dims());
  }
  return Status::OK();
}

Status CheckPhiloxState(const Tensor& state, int64 alg_tag_skip = 0) {
  static_assert(std::is_same<StateElementType, int64>::value,
                "StateElementType must be int64");
  static_assert(std::is_same<PhiloxRandom::ResultElementType, uint32>::value,
                "PhiloxRandom::ResultElementType must be uint32");
  if (state.NumElements() < alg_tag_skip + PHILOX_MIN_STATE_SIZE) {
    return errors::InvalidArgument(
        "For the Philox algorithm, the size of state"
        " must be at least ",
        alg_tag_skip + PHILOX_MIN_STATE_SIZE, "; got ", state.NumElements());
  }
  return Status::OK();
}

Status MakeShapeFromTensor(const Tensor& shape, TensorShape* out) {
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

template <typename Device, typename Distribution>
Status UpdateVariableAndFill(
    OpKernelContext* ctx, Distribution dist, int state_input_idx,
    bool read_alg_from_state, Algorithm alg, int64 output_size,
    typename Distribution::ResultElementType* output_data) {
  auto locks = MaybeLockVariableInputMutexesInOrder(ctx, /* do_lock */ true,
                                                    /* sparse */ false, {0});
  Tensor var_tensor;
  Status status = GetInputTensorFromVariable(ctx, 0, true, false, &var_tensor);
  if (!status.ok()) return status;
  TF_RETURN_IF_ERROR(CheckState(var_tensor));
  auto var_tensor_flat = var_tensor.flat<StateElementType>();
  int64 alg_tag_skip = 0;
  if (read_alg_from_state) {
    alg_tag_skip = 1;
    if (var_tensor_flat.size() < 1) {
      return errors::InvalidArgument("Size of tensor must be at least 1");
    }
    alg = Algorithm(var_tensor_flat(0));
  }
  if (alg == RNG_ALG_PHILOX || alg == RNG_ALG_AUTO_SELECT) {
    TF_RETURN_IF_ERROR(CheckPhiloxState(var_tensor, alg_tag_skip));
    UpdateVariableAndFill_Philox<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(), dist, output_size, alg_tag_skip,
        &var_tensor, output_data);
    return Status::OK();
  } else {
    return errors::InvalidArgument("Unsupported algorithm id: ", alg);
  }
}

// Preconditon: input(0) is an existing resource.
template <typename Device, class Distribution>
void StatefulRandomCompute(OpKernelContext* ctx, Distribution dist,
                           int state_input_idx, int shape_input_idx,
                           bool read_alg_from_state, Algorithm alg) {
  using T = typename Distribution::ResultElementType;
  const Tensor& shape_t = ctx->input(shape_input_idx);
  TensorShape shape;
  OP_REQUIRES_OK(ctx, MakeShapeFromTensor(shape_t, &shape));
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  auto output_flat = output->flat<T>();
  OP_REQUIRES_OK(ctx, UpdateVariableAndFill<Device>(
                          ctx, dist, state_input_idx, read_alg_from_state, alg,
                          output_flat.size(), output_flat.data()));
}

template <typename Device, class Distribution>
class StatefulRandomOp : public OpKernel {
 public:
  explicit StatefulRandomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    StatefulRandomCompute<Device>(ctx, Distribution(), 0, 1, true, 0);
  }
};

template <typename T>
Status GetScalar(const Tensor& tensor, int input_idx, T* result) {
  auto dtype = DataTypeToEnum<T>::v();
  if (tensor.dims() != 0) {
    return errors::InvalidArgument("input ", std::to_string(input_idx),
                                   " (0-based) must have shape [], not ",
                                   tensor.shape().DebugString());
  }
  if (tensor.dtype() != dtype) {
    return errors::InvalidArgument("dtype of input ", std::to_string(input_idx),
                                   " (0-based) must be ", DataTypeString(dtype),
                                   ", not ", DataTypeString(tensor.dtype()));
  }
  *result = tensor.flat<T>()(0);
  return Status::OK();
}

template <typename AlgEnumType>
Status GetAlg(OpKernelContext* ctx, int input_idx, Algorithm* alg) {
  AlgEnumType alg_id;
  TF_RETURN_IF_ERROR(GetScalar(ctx->input(input_idx), input_idx, &alg_id));
  *alg = Algorithm(alg_id);
  return Status::OK();
}

template <typename Device, class Distribution>
class StatefulRandomOpV2 : public OpKernel {
 public:
  explicit StatefulRandomOpV2(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetAlg<int64>(ctx, 1, &alg));
    StatefulRandomCompute<Device>(ctx, Distribution(), /*state_input_idx=*/0,
                                  /*shape_input_idx=*/2,
                                  /*read_alg_from_state=*/false, alg);
  }
};

template <typename Device, class IntType>
class StatefulUniformIntOp : public OpKernel {
 public:
  explicit StatefulUniformIntOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetAlg<int64>(ctx, 1, &alg));
    const Tensor& minval = ctx->input(3);
    const Tensor& maxval = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));

    // Verify that minval < maxval.  This check intentionally happens after the
    // early exit for empty output.  Zero impossible things are fine.
    IntType lo = minval.scalar<IntType>()();
    IntType hi = maxval.scalar<IntType>()();
    OP_REQUIRES(
        ctx, lo < hi,
        errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    // Build distribution
    typedef random::UniformDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist(lo, hi);

    StatefulRandomCompute<Device>(ctx, dist, /*state_input_idx=*/0,
                                  /*shape_input_idx=*/2,
                                  /*read_alg_from_state=*/false, alg);
  }
};

template <typename Device, class IntType>
class StatefulUniformFullIntOp : public OpKernel {
 public:
  explicit StatefulUniformFullIntOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetAlg<int64>(ctx, 1, &alg));
    StatefulRandomCompute<Device>(
        ctx,
        random::UniformFullIntDistribution<random::PhiloxRandom, IntType>(),
        /*state_input_idx=*/0, /*shape_input_idx=*/2,
        /*read_alg_from_state=*/false, alg);
  }
};

template <typename Device, typename AlgEnumType = int64,
          typename DeltaType = int64, bool read_old_value = false>
class RngSkipOp : public OpKernel {
 public:
  explicit RngSkipOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto alg_input_idx = 1;
    auto delta_input_idx = 2;
    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetAlg<AlgEnumType>(ctx, alg_input_idx, &alg));
    DeltaType delta_;
    OP_REQUIRES_OK(
        ctx, GetScalar(ctx->input(delta_input_idx), delta_input_idx, &delta_));
    uint64 delta = static_cast<uint64>(delta_);
    using T = StateElementType;
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, /* do_lock */ true,
                                                      /* sparse */ false, {0});
    Tensor var_tensor;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable(ctx, 0, true, false, &var_tensor));

    OP_REQUIRES_OK(ctx, CheckState(var_tensor));
    if (read_old_value) {
      Tensor* output;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, {RNG_MAX_COUNTER_SIZE + RNG_KEY_SIZE},
                                    &output));
      auto output_flat = output->flat<T>();
      if (RNG_MAX_COUNTER_SIZE > GetCounterSize(alg)) {
        functor::SetZeroFunctor<Device, T>()(ctx->eigen_device<Device>(),
                                             output_flat);
      }
      functor::DenseUpdate<Device, T, ASSIGN>()(
          ctx->eigen_device<Device>(), output_flat,
          const_cast<const Tensor*>(&var_tensor)->flat<T>());
    }
    if (alg == RNG_ALG_PHILOX) {
      OP_REQUIRES_OK(ctx, CheckPhiloxState(var_tensor));
      // var_tensor layout is counter+key, so var_tensor data is also counter
      // data.
      auto counter_data = var_tensor.flat<T>().data();
      RngSkip_Philox<Device>()(ctx->eigen_device<Device>(), counter_data, delta,
                               counter_data);
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }
  }
};

// So far the 'Distribution' type parameter is only used when the algorithm is
// philox, so 'NormalDistribution<PhiloxRandom, ...>' is fine for now.
#define REGISTER_FloatOps(DEVICE, TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatefulStandardNormalV2")                                      \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("resource")                                           \
          .HostMemory("algorithm")                                          \
          .HostMemory("shape")                                              \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatefulRandomOpV2<DEVICE##Device,                                    \
                         random::NormalDistribution<PhiloxRandom, TYPE>>);  \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatefulUniform")                                               \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("resource")                                           \
          .HostMemory("algorithm")                                          \
          .HostMemory("shape")                                              \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatefulRandomOpV2<DEVICE##Device,                                    \
                         random::UniformDistribution<PhiloxRandom, TYPE>>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatefulTruncatedNormal")                                       \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("resource")                                           \
          .HostMemory("algorithm")                                          \
          .HostMemory("shape")                                              \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatefulRandomOpV2<                                                   \
          DEVICE##Device,                                                   \
          random::TruncatedNormalDistribution<                              \
              random::SingleSampleAdapter<PhiloxRandom>, TYPE>>);

#define REGISTER_FloatOps_GPU(TYPE) REGISTER_FloatOps(GPU, TYPE)

#define REGISTER_StatefulUniformInt(DEVICE, TYPE)             \
  REGISTER_KERNEL_BUILDER(Name("StatefulUniformInt")          \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("resource")         \
                              .HostMemory("algorithm")        \
                              .HostMemory("shape")            \
                              .HostMemory("minval")           \
                              .HostMemory("maxval")           \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatefulUniformIntOp<DEVICE##Device, TYPE>);

#define REGISTER_StatefulUniformInt_GPU(TYPE) \
  REGISTER_StatefulUniformInt(GPU, TYPE)

#define REGISTER_StatefulUniformFullInt(DEVICE, TYPE)         \
  REGISTER_KERNEL_BUILDER(Name("StatefulUniformFullInt")      \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("resource")         \
                              .HostMemory("algorithm")        \
                              .HostMemory("shape")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatefulUniformFullIntOp<DEVICE##Device, TYPE>);

#define REGISTER_StatefulUniformFullInt_GPU(TYPE) \
  REGISTER_StatefulUniformFullInt(GPU, TYPE)

#define REGISTER_RngSkip(DEVICE)                       \
  REGISTER_KERNEL_BUILDER(Name("RngSkip")              \
                              .Device(DEVICE_##DEVICE) \
                              .HostMemory("resource")  \
                              .HostMemory("algorithm") \
                              .HostMemory("delta"),    \
                          RngSkipOp<DEVICE##Device>);  \
  REGISTER_KERNEL_BUILDER(Name("RngReadAndSkip")       \
                              .Device(DEVICE_##DEVICE) \
                              .HostMemory("resource")  \
                              .HostMemory("alg")       \
                              .HostMemory("delta"),    \
                          RngSkipOp<DEVICE##Device, int32, uint64, true>);

TF_CALL_half(REGISTER_FloatOps_GPU);
TF_CALL_float(REGISTER_FloatOps_GPU);
TF_CALL_bfloat16(REGISTER_FloatOps_GPU);
TF_CALL_int32(REGISTER_StatefulUniformInt_GPU);
TF_CALL_int64(REGISTER_StatefulUniformInt_GPU);
TF_CALL_int32(REGISTER_StatefulUniformFullInt_GPU);
TF_CALL_int64(REGISTER_StatefulUniformFullInt_GPU);
TF_CALL_uint32(REGISTER_StatefulUniformFullInt_GPU);
TF_CALL_uint64(REGISTER_StatefulUniformFullInt_GPU);
#ifdef ITEX_ENABLE_DOUBLE
TF_CALL_double(REGISTER_FloatOps_GPU);
#endif  // ITEX_ENABLE_DOUBLE
REGISTER_RngSkip(GPU);

}  // end namespace itex
