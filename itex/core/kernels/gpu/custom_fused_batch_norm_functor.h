#ifndef ITEX_CORE_KERNELS_GPU_CUSTOM_FUSED_BATCH_NORM_FUNCTOR_H_
#define ITEX_CORE_KERNELS_GPU_CUSTOM_FUSED_BATCH_NORM_FUNCTOR_H_

#include <algorithm>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"

namespace itex {
namespace functor {

enum {
  MaxGroupSize = 256,
  SubGroupSize = 16,
  MaxSubGroup = MaxGroupSize / SubGroupSize,
  ElemsPerItem = 16,
  VecSize = 4,
  MaxLocalItemOnZ = 8
};

inline void PacketStore(float* ptr, int offset,
                        const sycl::vec<float, VecSize>& array) {
  *(reinterpret_cast<sycl::vec<float, VecSize>*>(ptr + offset)) = array;
}

inline void PacketStore(Eigen::half* ptr, int offset,
                        const sycl::vec<float, VecSize>& array) {
  sycl::vec<sycl::half, VecSize> tmp;
#pragma unroll
  for (int i = 0; i < VecSize; ++i) tmp[i] = static_cast<sycl::half>(array[i]);
  *(reinterpret_cast<sycl::vec<sycl::half, VecSize>*>(ptr + offset)) = tmp;
}

inline void PacketStore(Eigen::bfloat16* ptr, int offset,
                        const sycl::vec<float, VecSize>& array) {
  sycl::vec<uint16_t, VecSize> tmp;
#pragma unroll
  for (int i = 0; i < VecSize; ++i)
    tmp[i] = Eigen::bfloat16_impl::float_to_bfloat16_rtne<true>(array[i]).value;
  *(reinterpret_cast<sycl::vec<uint16_t, VecSize>*>(ptr + offset)) = tmp;
}

inline void PacketLoad(const float* ptr, int offset,
                       sycl::vec<float, VecSize>* array) {
  *array = *(reinterpret_cast<const sycl::vec<float, VecSize>*>(ptr + offset));
}

inline void PacketLoad(const Eigen::half* ptr, int offset,
                       sycl::vec<float, VecSize>* array) {
  sycl::vec<sycl::half, VecSize> in_array =
      *(reinterpret_cast<const sycl::vec<sycl::half, VecSize>*>(ptr + offset));

#pragma unroll
  for (int i = 0; i < VecSize; ++i)
    (*array)[i] = static_cast<float>(in_array[i]);
}

inline void PacketLoad(const Eigen::bfloat16* ptr, int offset,
                       sycl::vec<float, VecSize>* array) {
  sycl::vec<uint16_t, VecSize> in_array =
      *(reinterpret_cast<const sycl::vec<uint16_t, VecSize>*>(ptr + offset));

#pragma unroll
  for (int i = 0; i < VecSize; ++i)
    (*array)[i] = Eigen::bfloat16_impl::bfloat16_to_float(
        Eigen::bfloat16_impl::raw_uint16_to_bfloat16(in_array[i]));
}

template <typename T, typename LocalAccessor>
struct VecSecondStepKernel {
  VecSecondStepKernel(const float* inter_array1_, const float* inter_array2_,
                      T* array1_, T* array2_, LocalAccessor scratch1_,
                      LocalAccessor scratch2_, const int extend_x_,
                      const int extend_y_, const int extend_z_,
                      const int elems_per_item_, const int num_sub_group_,
                      const int num_segments_y_, const int k_,
                      const int local_item_on_z_)
      : inter_array1(inter_array1_),
        inter_array2(inter_array2_),
        array1(array1_),
        array2(array2_),
        scratch1(scratch1_),
        scratch2(scratch2_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_),
        elems_per_item(elems_per_item_),
        num_sub_group(num_sub_group_),
        num_segments_y(num_segments_y_),
        k(k_),
        local_item_on_z(local_item_on_z_) {}
  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<3> item) const {
    using vecT = sycl::vec<float, VecSize>;

    // get start index
    int x_group_id = item.get_group(0);
    int y_group_id = item.get_group(1);
    int z_group_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int subgroup_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_linear_id();
    int group_z_id = lane_id % local_item_on_z;
    int group_k_id = lane_id / local_item_on_z;

    int x_offset = x_group_id * extend_y * extend_z;

    // each subgroup load data and reduce elems_per_item

    vecT aggregate1(0);
    vecT aggregate2(0);

    int z_offset =
        z_group_id * local_item_on_z * VecSize + group_z_id * VecSize;

    for (int i = 0; i < elems_per_item; ++i) {
      int y_idx = y_group_id * num_sub_group * elems_per_item * k +
                  subgroup_id * elems_per_item * k + group_k_id + i * k;
      if (y_idx >= extend_y) break;
      int offset = x_offset + y_idx * extend_z + z_offset;
      vecT tmp1 = *(reinterpret_cast<const vecT*>(inter_array1 + offset));
      vecT tmp2 = *(reinterpret_cast<const vecT*>(inter_array2 + offset));

      aggregate1 += tmp1;
      aggregate2 += tmp2;
    }
    // each subgroup write result to slm
    scratch1[subgroup_id * k + group_k_id + group_z_id * num_sub_group * k] =
        aggregate1;
    scratch2[subgroup_id * k + group_k_id + group_z_id * num_sub_group * k] =
        aggregate2;
    item.barrier(sycl::access::fence_space::local_space);

    // ------------------------------------------------------------------
    // -------------slm reduce-------------------
    // slm: (SubGroupSize * k) * local_item_on_z
    // ------------------------------------------------------------------
    int slm_z_id = subgroup_id / k;
    int slm_k_id = subgroup_id % k;
    vecT value1 = scratch1[slm_z_id * num_sub_group * k +
                           slm_k_id * SubGroupSize + lane_id];
    vecT value2 = scratch2[slm_z_id * num_sub_group * k +
                           slm_k_id * SubGroupSize + lane_id];

    // reduce within each subgroup
    for (int i = 0; i < VecSize; ++i) {
      value1[i] = sycl::reduce_over_group(sg, value1[i], sycl::plus<T>());
      value2[i] = sycl::reduce_over_group(sg, value2[i], sycl::plus<T>());
    }

    // lane0 write result of each subgroup
    if (lane_id == 0) {
      scratch1[slm_z_id * num_sub_group * k + slm_k_id * SubGroupSize] = value1;
      scratch2[slm_z_id * num_sub_group * k + slm_k_id * SubGroupSize] = value2;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // collect result of k subgroup and store output
    if (lane_id == 0 && slm_k_id == 0) {
      vecT tmp1 = scratch1[slm_z_id * k * num_sub_group];
      vecT tmp2 = scratch2[slm_z_id * k * num_sub_group];
      for (int i = 1; i < k; ++i) {
        tmp1 += scratch1[slm_z_id * k * num_sub_group + i * SubGroupSize];
        tmp2 += scratch2[slm_z_id * k * num_sub_group + i * SubGroupSize];
      }

      int offset = x_group_id * extend_z * num_segments_y +
                   y_group_id * extend_z +
                   z_group_id * local_item_on_z * VecSize + slm_z_id * VecSize;

      PacketStore(array1, offset, tmp1);
      PacketStore(array2, offset, tmp2);
    }
  }
  const float* inter_array1;
  const float* inter_array2;
  T* array1;
  T* array2;
  LocalAccessor scratch1;
  LocalAccessor scratch2;
  const int extend_x;
  const int extend_y;
  const int extend_z;
  const int elems_per_item;
  const int num_sub_group;
  const int num_segments_y;
  const int k;
  const int local_item_on_z;
};

template <typename T, typename LocalAccessor>
struct SecondStepKernel {
  SecondStepKernel(const float* inter_array1_, const float* inter_array2_,
                   T* array1_, T* array2_, LocalAccessor scratch1_,
                   LocalAccessor scratch2_, const int extend_x_,
                   const int extend_y_, const int extend_z_,
                   const int elems_per_item_, const int num_sub_group_,
                   const int num_segments_y_)
      : inter_array1(inter_array1_),
        inter_array2(inter_array2_),
        array1(array1_),
        array2(array2_),
        scratch1(scratch1_),
        scratch2(scratch2_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_),
        elems_per_item(elems_per_item_),
        num_sub_group(num_sub_group_),
        num_segments_y(num_segments_y_) {}
  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<3> item) const {
    // get start index
    int x_group_id = item.get_group(0);
    int y_group_id = item.get_group(1);
    int z_group_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int subgroup_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_linear_id();

    int x_offset = x_group_id * extend_y * extend_z;

    // each subgroup load data and reduce elems_per_item
    float value1 = 0.0f;
    float value2 = 0.0f;
    int z_offset = z_group_id * SubGroupSize + lane_id;
    if (z_offset < extend_z) {
      for (int i = 0; i < elems_per_item; ++i) {
        int y_idx = y_group_id * num_sub_group * elems_per_item + i +
                    subgroup_id * elems_per_item;
        if (y_idx >= extend_y) break;
        int offset = x_offset + y_idx * extend_z + z_offset;
        value1 += inter_array1[offset];
        value2 += inter_array2[offset];
      }
    }
    // each subgroup write result to slm
    scratch1[subgroup_id + lane_id * num_sub_group] = value1;
    scratch2[subgroup_id + lane_id * num_sub_group] = value2;
    item.barrier(sycl::access::fence_space::local_space);

    // slm reduce and write output
    value1 = scratch1[subgroup_id * num_sub_group + lane_id];
    value2 = scratch2[subgroup_id * num_sub_group + lane_id];
    float update_value1 =
        sycl::reduce_over_group(sg, value1, sycl::plus<float>());
    float update_value2 =
        sycl::reduce_over_group(sg, value2, sycl::plus<float>());

    z_offset = z_group_id * SubGroupSize + subgroup_id;
    if (z_offset < extend_z && lane_id == 0) {
      int offset = x_group_id * extend_z * num_segments_y +
                   y_group_id * extend_z + z_group_id * SubGroupSize +
                   subgroup_id;
      array1[offset] = static_cast<T>(update_value1);
      array2[offset] = static_cast<T>(update_value2);
    }
  }
  const float* inter_array1;
  const float* inter_array2;
  T* array1;
  T* array2;
  LocalAccessor scratch1;
  LocalAccessor scratch2;
  const int extend_x;
  const int extend_y;
  const int extend_z;
  const int elems_per_item;
  const int num_sub_group;
  const int num_segments_y;
};

template <typename T>
using LocalAcc = sycl::accessor<T, 1, sycl::access::mode::read_write,
                                sycl::access::target::local>;
template <typename InT, typename OutT>
struct SimpleMeanVarReductionKernel {
  SimpleMeanVarReductionKernel(const InT* in_data_, OutT* mean_data_,
                               OutT* var_data_, const int extend_x_,
                               const int extend_y_, const int extend_z_)
      : in_data(in_data_),
        mean_data(mean_data_),
        var_data(var_data_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_) {}
  void operator()(sycl::nd_item<1> item) const {
    int id = item.get_global_linear_id();
    const int out_size = extend_x * extend_z;
    if (id < out_size) {
      int outer = id / extend_z;
      int inner = id - outer * extend_z;

      int in_offset = outer * extend_y * extend_z + inner;

      OutT mean = OutT(0);
      OutT mean_of_square = OutT(0);
#pragma unroll
      for (int i = 0; i < extend_y; ++i) {
        OutT value = static_cast<OutT>(in_data[in_offset + i * extend_z]);
        mean += value;
        mean_of_square += value * value;
      }
      mean_data[id] = mean;
      var_data[id] = mean_of_square;
    }
  }
  const InT* in_data;
  OutT* mean_data;
  OutT* var_data;
  const int extend_x;
  const int extend_y;
  const int extend_z;
};

template <typename InT, typename OutT>
void SimpleMeanVarReduction(OpKernelContext* context, const InT* in_data,
                            OutT* mean_data, OutT* var_data, const int extend_x,
                            const int extend_y, const int extend_z) {
  auto* stream = context->GetDeviceStream();
  const int out_size = extend_x * extend_z;
  int work_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  int num_wg = (out_size + work_group_size - 1) / work_group_size;
  sycl::nd_range<1> range(num_wg * work_group_size, work_group_size);
  stream->submit([&](sycl::handler& cgh) {
    SimpleMeanVarReductionKernel<InT, OutT> task(in_data, mean_data, var_data,
                                                 extend_x, extend_y, extend_z);
    cgh.parallel_for<SimpleMeanVarReductionKernel<InT, OutT>>(range, task);
  });
}

// always use float as computation type
template <typename InT, typename OutT, typename LocalAccessor>
struct MeanVarFirstStepKernel {
  MeanVarFirstStepKernel(const InT* in_data_, OutT* sum_data_,
                         OutT* sum_of_square_data_, LocalAccessor scratch1_,
                         LocalAccessor scratch2_, const int extend_x_,
                         const int extend_y_, const int extend_z_,
                         const int elems_per_item_, const int num_sub_group_,
                         const int num_segments_y_)
      : in_data(in_data_),
        sum_data(sum_data_),
        sum_of_square_data(sum_of_square_data_),
        scratch1(scratch1_),
        scratch2(scratch2_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_),
        elems_per_item(elems_per_item_),
        num_sub_group(num_sub_group_),
        num_segments_y(num_segments_y_) {}
  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<3> item) const {
    // get start index
    int x_group_id = item.get_group(0);
    int y_group_id = item.get_group(1);
    int z_group_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int subgroup_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_linear_id();

    int x_offset = x_group_id * extend_y * extend_z;

    // each subgroup load data and reduce elems_per_item
    float sum = 0.0f;
    float sum_of_square = 0.0f;
    int z_offset = z_group_id * SubGroupSize + lane_id;
    if (z_offset < extend_z) {
      for (int i = 0; i < elems_per_item; ++i) {
        int y_idx = y_group_id * num_sub_group * elems_per_item + i +
                    subgroup_id * elems_per_item;
        if (y_idx >= extend_y) break;
        int offset = x_offset + y_idx * extend_z + z_offset;
        float tmp = static_cast<float>(in_data[offset]);
        sum += tmp;
        sum_of_square += tmp * tmp;
      }
    }
    // each subgroup write result to slm
    scratch1[subgroup_id + lane_id * num_sub_group] = sum;
    scratch2[subgroup_id + lane_id * num_sub_group] = sum_of_square;
    item.barrier(sycl::access::fence_space::local_space);

    // slm reduce and write output
    sum = scratch1[subgroup_id * num_sub_group + lane_id];
    sum_of_square = scratch2[subgroup_id * num_sub_group + lane_id];
    float update_sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    float update_sum_of_square =
        sycl::reduce_over_group(sg, sum_of_square, sycl::plus<float>());

    z_offset = z_group_id * SubGroupSize + subgroup_id;
    if (z_offset < extend_z && lane_id == 0) {
      int offset = x_group_id * extend_z * num_segments_y +
                   y_group_id * extend_z + z_group_id * SubGroupSize +
                   subgroup_id;
      sum_data[offset] = static_cast<OutT>(update_sum);
      sum_of_square_data[offset] = static_cast<OutT>(update_sum_of_square);
    }
  }
  const InT* in_data;
  OutT* sum_data;
  OutT* sum_of_square_data;
  LocalAccessor scratch1;
  LocalAccessor scratch2;
  const int extend_x;
  const int extend_y;
  const int extend_z;
  const int elems_per_item;
  const int num_sub_group;
  const int num_segments_y;
};

template <typename InT, typename OutT>
void SGMeanVarReduction(OpKernelContext* ctx, const InT* in_data,
                        OutT* mean_data, OutT* var_data, const int extend_x,
                        const int extend_y, const int extend_z) {
  auto* stream = ctx->GetDeviceStream();
  int elems_per_item = ElemsPerItem;
  int num_sub_group = MaxSubGroup;

  if (extend_y * 2 <= num_sub_group * elems_per_item) {
    while (num_sub_group * elems_per_item >= extend_y * 2 &&
           elems_per_item > 1) {
      elems_per_item >>= 1;
    }
  }

  int num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item);
  int num_segments_z = DivUp(extend_z, static_cast<int>(SubGroupSize));

  if (num_segments_y > 1) {
    while (num_segments_y > num_sub_group * ElemsPerItem) {
      elems_per_item <<= 1;
      num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item);
    }
    sycl::range<3> local(1, num_sub_group, SubGroupSize);
    sycl::range<3> global(extend_x, num_segments_y * local[1],
                          num_segments_z * local[2]);

    Tensor inter_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(
                       DataTypeToEnum<float>::value,
                       TensorShape({2 * extend_x * num_segments_y * extend_z}),
                       &inter_tensor));
    float* inter_sum = inter_tensor.flat<float>().data();
    float* inter_sum_of_square =
        inter_sum + extend_x * num_segments_y * extend_z;

    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<float> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<float> scratch2(num_sub_group * SubGroupSize, cgh);
      MeanVarFirstStepKernel<InT, float, LocalAcc<float>> task(
          in_data, inter_sum, inter_sum_of_square, scratch1, scratch2, extend_x,
          extend_y, extend_z, elems_per_item, num_sub_group, num_segments_y);
      cgh.parallel_for<MeanVarFirstStepKernel<InT, float, LocalAcc<float>>>(
          sycl::nd_range<3>(global, local), task);
    });

    global = sycl::range<3>{static_cast<size_t>(extend_x), local[1],
                            num_segments_z * local[2]};
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<float> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<float> scratch2(num_sub_group * SubGroupSize, cgh);
      SecondStepKernel<OutT, LocalAcc<float>> task(
          inter_sum, inter_sum_of_square, mean_data, var_data, scratch1,
          scratch2, extend_x, num_segments_y, extend_z, ElemsPerItem,
          num_sub_group, 1);
      cgh.parallel_for<SecondStepKernel<OutT, LocalAcc<float>>>(
          sycl::nd_range<3>(global, local), task);
    });

  } else {
    sycl::range<3> local(1, num_sub_group, SubGroupSize);
    sycl::range<3> global(extend_x, local[1], num_segments_z * local[2]);
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<float> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<float> scratch2(num_sub_group * SubGroupSize, cgh);
      MeanVarFirstStepKernel<InT, OutT, LocalAcc<float>> task(
          in_data, mean_data, var_data, scratch1, scratch2, extend_x, extend_y,
          extend_z, elems_per_item, num_sub_group, 1);
      cgh.parallel_for<MeanVarFirstStepKernel<InT, OutT, LocalAcc<float>>>(
          sycl::nd_range<3>(global, local), task);
    });
  }
}

template <typename InT, typename OutT, typename LocalAccessor>
struct FwdVecFirstStepKernel {
  FwdVecFirstStepKernel(const InT* in_data_, OutT* out1_, OutT* out2_,
                        LocalAccessor scratch1_, LocalAccessor scratch2_,
                        const int extend_x_, const int extend_y_,
                        const int extend_z_, const int elems_per_item_,
                        const int num_sub_group_, const int num_segments_y_,
                        const int k_, const int local_item_on_z_)
      : in_data(in_data_),
        out1(out1_),
        out2(out2_),
        scratch1(scratch1_),
        scratch2(scratch2_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_),
        elems_per_item(elems_per_item_),
        num_sub_group(num_sub_group_),
        num_segments_y(num_segments_y_),
        k(k_),
        local_item_on_z(local_item_on_z_) {}
  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<3> item) const {
    typedef sycl::vec<float, VecSize> vecT;

    // get start index
    int x_group_id = item.get_group(0);
    int y_group_id = item.get_group(1);
    int z_group_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int subgroup_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_linear_id();
    int group_z_id = lane_id % local_item_on_z;
    int group_k_id = lane_id / local_item_on_z;

    int x_offset = x_group_id * extend_y * extend_z;

    // each subgroup load data and reduce elems_per_item
    vecT aggregate1(0.0f);
    vecT aggregate2(0.0f);

    int z_offset =
        z_group_id * local_item_on_z * VecSize + group_z_id * VecSize;

    for (int i = 0; i < elems_per_item; ++i) {
      int y_idx = y_group_id * num_sub_group * elems_per_item * k +
                  subgroup_id * elems_per_item * k + group_k_id + i * k;
      if (y_idx >= extend_y) break;
      int offset = x_offset + y_idx * extend_z + z_offset;

      vecT tmp;
      PacketLoad(in_data, offset, &tmp);

      for (int j = 0; j < VecSize; ++j) {
        aggregate1[j] += tmp[j];
        aggregate2[j] += tmp[j] * tmp[j];
      }
    }
    // each subgroup write result to slm
    scratch1[subgroup_id * k + group_k_id + group_z_id * num_sub_group * k] =
        aggregate1;
    scratch2[subgroup_id * k + group_k_id + group_z_id * num_sub_group * k] =
        aggregate2;
    item.barrier(sycl::access::fence_space::local_space);

    // ------------------------------------------------------------------
    // -------------slm reduce-------------------
    // slm: (SubGroupSize * k) * local_item_on_z
    // ------------------------------------------------------------------
    int slm_z_id = subgroup_id / k;
    int slm_k_id = subgroup_id % k;
    vecT value1 = scratch1[slm_z_id * num_sub_group * k +
                           slm_k_id * SubGroupSize + lane_id];
    vecT value2 = scratch2[slm_z_id * num_sub_group * k +
                           slm_k_id * SubGroupSize + lane_id];

    // reduce within each subgroup
    for (int i = 0; i < VecSize; ++i) {
      value1[i] = sycl::reduce_over_group(sg, value1[i], sycl::plus<float>());
      value2[i] = sycl::reduce_over_group(sg, value2[i], sycl::plus<float>());
    }

    // lane0 write result of each subgrop
    if (lane_id == 0) {
      scratch1[slm_z_id * num_sub_group * k + slm_k_id * SubGroupSize] = value1;
      scratch2[slm_z_id * num_sub_group * k + slm_k_id * SubGroupSize] = value2;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // collect result of k subgroup and store output
    if (lane_id == 0 && slm_k_id == 0) {
      vecT tmp1 = scratch1[slm_z_id * k * num_sub_group];
      vecT tmp2 = scratch2[slm_z_id * k * num_sub_group];
      for (int i = 1; i < k; ++i) {
        tmp1 += scratch1[slm_z_id * k * num_sub_group + i * SubGroupSize];
        tmp2 += scratch2[slm_z_id * k * num_sub_group + i * SubGroupSize];
      }
      int offset = x_group_id * extend_z * num_segments_y +
                   y_group_id * extend_z +
                   z_group_id * local_item_on_z * VecSize + slm_z_id * VecSize;

      PacketStore(out1, offset, tmp1);
      PacketStore(out2, offset, tmp2);
    }
  }
  const InT* in_data;

  OutT* out1;
  OutT* out2;
  LocalAccessor scratch1;
  LocalAccessor scratch2;
  const int extend_x;
  const int extend_y;
  const int extend_z;
  const int elems_per_item;
  const int num_sub_group;
  const int num_segments_y;
  const int k;
  const int local_item_on_z;
};

template <typename InT, typename OutT>
void SGFwdVecColReduction(OpKernelContext* context, const InT* in_data,
                          OutT* mean_data, OutT* var_data, const int extend_x,
                          const int extend_y, const int extend_z) {
  using vecT = sycl::vec<float, VecSize>;

  int elems_per_item = ElemsPerItem;
  int num_sub_group = MaxSubGroup;
  int work_item_on_z = extend_z / VecSize;
  int local_item_on_z =
      work_item_on_z <= MaxLocalItemOnZ ? work_item_on_z : MaxLocalItemOnZ;
  int k = SubGroupSize / local_item_on_z;

  if (extend_y * 2 <= num_sub_group * elems_per_item * k) {
    while (num_sub_group * elems_per_item * k >= extend_y * 2 &&
           elems_per_item > 1) {
      elems_per_item >>= 1;
    }
  }

  int num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item * k);
  int num_segments_z = work_item_on_z / local_item_on_z;

  auto* stream = context->GetDeviceStream();

  if (num_segments_y > 1) {
    while (num_segments_y > num_sub_group * ElemsPerItem * k) {
      elems_per_item <<= 1;
      num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item * k);
    }

    sycl::range<3> local(1, num_sub_group * k, local_item_on_z);
    sycl::range<3> global(extend_x, num_segments_y * local[1],
                          num_segments_z * local[2]);

    Tensor inter_out_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<float>::value,
                       TensorShape({2 * extend_x * num_segments_y * extend_z}),
                       &inter_out_tensor));
    float* inter_sum = inter_out_tensor.flat<float>().data();
    float* inter_sum_of_square =
        inter_sum + extend_x * num_segments_y * extend_z;

    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<vecT> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<vecT> scratch2(num_sub_group * SubGroupSize, cgh);
      FwdVecFirstStepKernel<InT, float, LocalAcc<vecT>> task(
          in_data, inter_sum, inter_sum_of_square, scratch1, scratch2, extend_x,
          extend_y, extend_z, elems_per_item, num_sub_group, num_segments_y, k,
          local_item_on_z);
      cgh.parallel_for<FwdVecFirstStepKernel<InT, float, LocalAcc<vecT>>>(
          sycl::nd_range<3>(global, local), task);
    });

    global = sycl::range<3>{static_cast<size_t>(extend_x), local[1],
                            num_segments_z * local[2]};
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<vecT> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<vecT> scratch2(num_sub_group * SubGroupSize, cgh);
      VecSecondStepKernel<float, LocalAcc<vecT>> task(
          inter_sum, inter_sum_of_square, mean_data, var_data, scratch1,
          scratch2, extend_x, num_segments_y, extend_z, ElemsPerItem,
          num_sub_group, 1, k, local_item_on_z);
      cgh.parallel_for<VecSecondStepKernel<float, LocalAcc<vecT>>>(
          sycl::nd_range<3>(global, local), task);
    });

  } else {
    sycl::range<3> local(1, num_sub_group * k, local_item_on_z);
    sycl::range<3> global(extend_x, local[1], num_segments_z * local[2]);
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<vecT> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<vecT> scratch2(num_sub_group * SubGroupSize, cgh);
      FwdVecFirstStepKernel<InT, OutT, LocalAcc<vecT>> task(
          in_data, mean_data, var_data, scratch1, scratch2, extend_x, extend_y,
          extend_z, elems_per_item, num_sub_group, 1, k, local_item_on_z);
      cgh.parallel_for<FwdVecFirstStepKernel<InT, OutT, LocalAcc<vecT>>>(
          sycl::nd_range<3>(global, local), task);
    });
  }
}

// Note, here we don't get real mean/variance, but sum(in_data), sum(in_data *
// in_data), we leave next computation to BN forward eltwise kernel
template <typename InT, typename OutT>
void MeanVarReduction(OpKernelContext* context, const InT* in_data,
                      OutT* mean_data, OutT* var_data, const int extend_x,
                      const int extend_y, const int extend_z) {
  int elems_per_item = extend_y / (extend_x * extend_z);
  bool use_vectorization_pass = (extend_z % VecSize == 0 &&
                                 (SubGroupSize % (extend_z / VecSize) == 0 ||
                                  (extend_z / VecSize) % MaxLocalItemOnZ == 0));
  if (extend_y < 32 && elems_per_item < 4)
    SimpleMeanVarReduction(context, in_data, mean_data, var_data, extend_x,
                           extend_y, extend_z);
  else if (use_vectorization_pass)
    SGFwdVecColReduction(context, in_data, mean_data, var_data, extend_x,
                         extend_y, extend_z);
  else
    SGMeanVarReduction(context, in_data, mean_data, var_data, extend_x,
                       extend_y, extend_z);
}

template <typename InT, typename ScaleT, int VecSizeSp, int VecSizeIc,
          bool IsTrain, bool FuseNormRelu, bool FuseNormAddRelu>
struct BnForwardOptimizedKernel {
  BnForwardOptimizedKernel(const InT* in_, const ScaleT* mean_,
                           const ScaleT* var_, const ScaleT* scale_,
                           const ScaleT* offset_, const InT* side_input_,
                           InT* out_, const ScaleT* old_mean_,
                           const ScaleT* old_var_, ScaleT* new_mean_,
                           ScaleT* new_var_, ScaleT* saved_mean_,
                           ScaleT* saved_var_, const int sp_, const int ic_,
                           const float epsilon_,
                           const float exponential_avg_factor_)
      : in(in_),
        mean(mean_),
        var(var_),
        scale(scale_),
        offset(offset_),
        side_input(side_input_),
        out(out_),
        old_mean(old_mean_),
        old_var(old_var_),
        new_mean(new_mean_),
        new_var(new_var_),
        saved_mean(saved_mean_),
        saved_var(saved_var_),
        sp(sp_),
        ic(ic_),
        epsilon(epsilon_),
        exponential_avg_factor(exponential_avg_factor_) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    const int nelems = sp * ic;
    int n = ic / VecSizeIc;
    int x_id = id / n;
    int y_id = id - (x_id * n);

    int y_offset = y_id * VecSizeIc;
    int x_offset = x_id * VecSizeSp;
    int base_offset = x_offset * ic + y_offset;

    if (base_offset >= nelems) return;
    InT values[VecSizeSp * VecSizeIc], out_values[VecSizeSp * VecSizeIc];
    InT side_input_values[VecSizeSp * VecSizeIc];
    using VecT = sycl::vec<InT, VecSizeIc>;
    VecT* vec_values = reinterpret_cast<VecT*>(values);
    VecT* vec_out_values = reinterpret_cast<VecT*>(out_values);
    VecT* vec_side_input_values = reinterpret_cast<VecT*>(side_input_values);

    int ic_idx[VecSizeIc];
    ScaleT mean_values[VecSizeIc], var_values[VecSizeIc];

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      vec_values[i] = *(reinterpret_cast<const VecT*>(in + offset));
      if (FuseNormAddRelu) {
        vec_side_input_values[i] =
            *(reinterpret_cast<const VecT*>(side_input + offset));
      }
    }

    for (int j = 0; j < VecSizeIc; ++j) {
      ic_idx[j] = y_offset + j;
      ScaleT mean_value = mean[ic_idx[j]];
      ScaleT var_value = var[ic_idx[j]];
      if (IsTrain) {
        mean_value /= sp;
        var_value /= sp;
        var_value -= mean_value * mean_value;
      }
      mean_values[j] = mean_value;
      var_values[j] = var_value;
    }

    for (int i = 0; i < VecSizeSp; ++i) {
      for (int j = 0; j < VecSizeIc; ++j) {
        ScaleT inv = sycl::rsqrt(var_values[j] + epsilon) * scale[ic_idx[j]];
        InT temp = static_cast<InT>(
            (static_cast<ScaleT>(values[i * VecSizeIc + j]) - mean_values[j]) *
                inv +
            offset[ic_idx[j]]);
        if (FuseNormRelu) {
          temp = temp > 0 ? temp : InT(0);
        }
        if (FuseNormAddRelu) {
          temp += side_input_values[i * VecSizeIc + j];
          temp = temp > 0 ? temp : InT(0);
        }
        out_values[i * VecSizeIc + j] = temp;
      }
    }

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      *(reinterpret_cast<VecT*>(out + offset)) = vec_out_values[i];
    }

    if (x_id == 0) {
      for (int j = 0; j < VecSizeIc; ++j) {
        ScaleT old_mean_value = ScaleT(0);
        ScaleT old_var_value = ScaleT(0);
        if (!IsTrain || exponential_avg_factor < 1) {
          old_mean_value = old_mean[ic_idx[j]];
          old_var_value = old_var[ic_idx[j]];
        }
        if (IsTrain) {
          saved_mean[ic_idx[j]] = mean_values[j];
          saved_var[ic_idx[j]] = var_values[j];
          new_mean[ic_idx[j]] = (1 - exponential_avg_factor) * old_mean_value +
                                exponential_avg_factor * mean_values[j];
          new_var[ic_idx[j]] =
              (1 - exponential_avg_factor) * old_var_value +
              exponential_avg_factor * var_values[j] *
                  (static_cast<float>(sp) / static_cast<float>(sp - 1));
        } else {
          new_mean[ic_idx[j]] = mean_values[j];
          new_var[ic_idx[j]] = var_values[j];
        }
      }
    }
  }
  const InT* in;
  const ScaleT* mean;
  const ScaleT* var;
  const ScaleT* scale;
  const ScaleT* offset;
  const InT* side_input;
  InT* out;
  const ScaleT* old_mean;
  const ScaleT* old_var;
  ScaleT* new_mean;
  ScaleT* new_var;
  ScaleT* saved_mean;
  ScaleT* saved_var;
  const int sp;
  const int ic;
  const float epsilon;
  const float exponential_avg_factor;
};

template <typename ScaleT, int VecSizeSp, int VecSizeIc, bool IsTrain,
          bool FuseNormRelu, bool FuseNormAddRelu>
struct BnForwardOptimizedKernel<Eigen::half, ScaleT, VecSizeSp, VecSizeIc,
                                IsTrain, FuseNormRelu, FuseNormAddRelu> {
  typedef Eigen::half InT;
  typedef sycl::half T;
  BnForwardOptimizedKernel(const InT* in_, const ScaleT* mean_,
                           const ScaleT* var_, const ScaleT* scale_,
                           const ScaleT* offset_, const InT* side_input_,
                           InT* out_, const ScaleT* old_mean_,
                           const ScaleT* old_var_, ScaleT* new_mean_,
                           ScaleT* new_var_, ScaleT* saved_mean_,
                           ScaleT* saved_var_, const int sp_, const int ic_,
                           const float epsilon_,
                           const float exponential_avg_factor_)
      : in(in_),
        mean(mean_),
        var(var_),
        scale(scale_),
        offset(offset_),
        side_input(side_input_),
        out(out_),
        old_mean(old_mean_),
        old_var(old_var_),
        new_mean(new_mean_),
        new_var(new_var_),
        saved_mean(saved_mean_),
        saved_var(saved_var_),
        sp(sp_),
        ic(ic_),
        epsilon(epsilon_),
        exponential_avg_factor(exponential_avg_factor_) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    const int nelems = sp * ic;
    int n = ic / VecSizeIc;
    int x_id = id / n;
    int y_id = id - (x_id * n);

    int y_offset = y_id * VecSizeIc;
    int x_offset = x_id * VecSizeSp;
    int base_offset = x_offset * ic + y_offset;

    if (base_offset >= nelems) return;
    T values[VecSizeSp * VecSizeIc], out_values[VecSizeSp * VecSizeIc];
    T side_input_values[VecSizeSp * VecSizeIc];

    using VecT = sycl::vec<T, VecSizeIc>;
    VecT* vec_values = reinterpret_cast<VecT*>(values);
    VecT* vec_out_values = reinterpret_cast<VecT*>(out_values);
    VecT* vec_side_input_values = reinterpret_cast<VecT*>(side_input_values);

    int ic_idx[VecSizeIc];
    ScaleT mean_values[VecSizeIc], var_values[VecSizeIc];

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      vec_values[i] = *(reinterpret_cast<const VecT*>(in + offset));
      if (FuseNormAddRelu) {
        vec_side_input_values[i] =
            *(reinterpret_cast<const VecT*>(side_input + offset));
      }
    }

    for (int j = 0; j < VecSizeIc; ++j) {
      ic_idx[j] = y_offset + j;
      ScaleT mean_value = mean[ic_idx[j]];
      ScaleT var_value = var[ic_idx[j]];
      if (IsTrain) {
        mean_value /= sp;
        var_value /= sp;
        var_value -= mean_value * mean_value;
      }
      mean_values[j] = mean_value;
      var_values[j] = var_value;
    }

    for (int i = 0; i < VecSizeSp; ++i) {
      for (int j = 0; j < VecSizeIc; ++j) {
        ScaleT inv = sycl::rsqrt(var_values[j] + epsilon) * scale[ic_idx[j]];
        T temp = static_cast<T>(
            (static_cast<ScaleT>(values[i * VecSizeIc + j]) - mean_values[j]) *
                inv +
            offset[ic_idx[j]]);
        if (FuseNormRelu) {
          temp = temp > T(0) ? temp : T(0);
        }
        if (FuseNormAddRelu) {
          temp += side_input_values[i * VecSizeIc + j];
          temp = temp > T(0) ? temp : T(0);
        }
        out_values[i * VecSizeIc + j] = temp;
      }
    }

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      *(reinterpret_cast<VecT*>(out + offset)) = vec_out_values[i];
    }

    if (x_id == 0) {
      for (int j = 0; j < VecSizeIc; ++j) {
        ScaleT old_mean_value = ScaleT(0);
        ScaleT old_var_value = ScaleT(0);
        if (!IsTrain || exponential_avg_factor < 1) {
          old_mean_value = old_mean[ic_idx[j]];
          old_var_value = old_var[ic_idx[j]];
        }
        if (IsTrain) {
          saved_mean[ic_idx[j]] = mean_values[j];
          saved_var[ic_idx[j]] = var_values[j];
          new_mean[ic_idx[j]] = (1 - exponential_avg_factor) * old_mean_value +
                                exponential_avg_factor * mean_values[j];
          new_var[ic_idx[j]] =
              (1 - exponential_avg_factor) * old_var_value +
              exponential_avg_factor * var_values[j] *
                  (static_cast<float>(sp) / static_cast<float>(sp - 1));
        } else {
          new_mean[ic_idx[j]] = mean_values[j];
          new_var[ic_idx[j]] = var_values[j];
        }
      }
    }
  }
  const InT* in;
  const ScaleT* mean;
  const ScaleT* var;
  const ScaleT* scale;
  const ScaleT* offset;
  const InT* side_input;
  InT* out;
  const ScaleT* old_mean;
  const ScaleT* old_var;
  ScaleT* new_mean;
  ScaleT* new_var;
  ScaleT* saved_mean;
  ScaleT* saved_var;
  const int sp;
  const int ic;
  const float epsilon;
  const float exponential_avg_factor;
};

template <typename ScaleT, int VecSizeSp, int VecSizeIc, bool IsTrain,
          bool FuseNormRelu, bool FuseNormAddRelu>
struct BnForwardOptimizedKernel<Eigen::bfloat16, ScaleT, VecSizeSp, VecSizeIc,
                                IsTrain, FuseNormRelu, FuseNormAddRelu> {
  typedef Eigen::bfloat16 InT;
  typedef uint16_t T;
  typedef sycl::vec<T, VecSizeIc> VecT;
  BnForwardOptimizedKernel(const InT* in_, const ScaleT* mean_,
                           const ScaleT* var_, const ScaleT* scale_,
                           const ScaleT* offset_, const InT* side_input_,
                           InT* out_, const ScaleT* old_mean_,
                           const ScaleT* old_var_, ScaleT* new_mean_,
                           ScaleT* new_var_, ScaleT* saved_mean_,
                           ScaleT* saved_var_, const int sp_, const int ic_,
                           const float epsilon_,
                           const float exponential_avg_factor_)
      : in(in_),
        mean(mean_),
        var(var_),
        scale(scale_),
        offset(offset_),
        side_input(side_input_),
        out(out_),
        old_mean(old_mean_),
        old_var(old_var_),
        new_mean(new_mean_),
        new_var(new_var_),
        saved_mean(saved_mean_),
        saved_var(saved_var_),
        sp(sp_),
        ic(ic_),
        epsilon(epsilon_),
        exponential_avg_factor(exponential_avg_factor_) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    const int nelems = sp * ic;
    int n = ic / VecSizeIc;
    int x_id = id / n;
    int y_id = id - (x_id * n);

    int y_offset = y_id * VecSizeIc;
    int x_offset = x_id * VecSizeSp;
    int base_offset = x_offset * ic + y_offset;

    if (base_offset >= nelems) return;
    T values[VecSizeSp * VecSizeIc];
    T out_values[VecSizeSp * VecSizeIc];
    T side_input_values[VecSizeSp * VecSizeIc];

    VecT* vec_values = reinterpret_cast<VecT*>(values);
    VecT* vec_out_values = reinterpret_cast<VecT*>(out_values);
    VecT* vec_side_input_values = reinterpret_cast<VecT*>(side_input_values);

    int ic_idx[VecSizeIc];
    ScaleT mean_values[VecSizeIc], var_values[VecSizeIc];

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      vec_values[i] = *(reinterpret_cast<const VecT*>(in + offset));
      if (FuseNormAddRelu) {
        vec_side_input_values[i] =
            *(reinterpret_cast<const VecT*>(side_input + offset));
      }
    }

    for (int j = 0; j < VecSizeIc; ++j) {
      ic_idx[j] = y_offset + j;
      ScaleT mean_value = mean[ic_idx[j]];
      ScaleT var_value = var[ic_idx[j]];
      if (IsTrain) {
        mean_value /= sp;
        var_value /= sp;
        var_value -= mean_value * mean_value;
      }
      mean_values[j] = mean_value;
      var_values[j] = var_value;
    }

    for (int i = 0; i < VecSizeSp; ++i) {
      for (int j = 0; j < VecSizeIc; ++j) {
        ScaleT inv = sycl::rsqrt(var_values[j] + epsilon) * scale[ic_idx[j]];
        ScaleT temp = (Eigen::bfloat16_impl::bfloat16_to_float(
                           Eigen::bfloat16_impl::raw_uint16_to_bfloat16(
                               values[i * VecSizeIc + j])) -
                       mean_values[j]) *
                          inv +
                      offset[ic_idx[j]];
        if (FuseNormRelu) {
          temp = temp > ScaleT(0) ? temp : ScaleT(0);
        }
        if (FuseNormAddRelu) {
          temp += Eigen::bfloat16_impl::bfloat16_to_float(
              Eigen::bfloat16_impl::raw_uint16_to_bfloat16(
                  side_input_values[i * VecSizeIc + j]));
          temp = temp > ScaleT(0) ? temp : ScaleT(0);
        }
        out_values[i * VecSizeIc + j] =
            Eigen::bfloat16_impl::float_to_bfloat16_rtne<true>(temp).value;
      }
    }

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      *(reinterpret_cast<VecT*>(out + offset)) = vec_out_values[i];
    }

    if (x_id == 0) {
      for (int j = 0; j < VecSizeIc; ++j) {
        ScaleT old_mean_value = ScaleT(0);
        ScaleT old_var_value = ScaleT(0);
        if (!IsTrain || exponential_avg_factor < 1) {
          old_mean_value = old_mean[ic_idx[j]];
          old_var_value = old_var[ic_idx[j]];
        }
        if (IsTrain) {
          saved_mean[ic_idx[j]] = mean_values[j];
          saved_var[ic_idx[j]] = var_values[j];
          new_mean[ic_idx[j]] = (1 - exponential_avg_factor) * old_mean_value +
                                exponential_avg_factor * mean_values[j];
          new_var[ic_idx[j]] =
              (1 - exponential_avg_factor) * old_var_value +
              exponential_avg_factor * var_values[j] *
                  (static_cast<float>(sp) / static_cast<float>(sp - 1));
        } else {
          new_mean[ic_idx[j]] = mean_values[j];
          new_var[ic_idx[j]] = var_values[j];
        }
      }
    }
  }
  const InT* in;
  const ScaleT* mean;
  const ScaleT* var;
  const ScaleT* scale;
  const ScaleT* offset;
  const InT* side_input;
  InT* out;
  const ScaleT* old_mean;
  const ScaleT* old_var;
  ScaleT* new_mean;
  ScaleT* new_var;
  ScaleT* saved_mean;
  ScaleT* saved_var;
  const int sp;
  const int ic;
  const float epsilon;
  const float exponential_avg_factor;
};

template <typename InT, typename ScaleT, int VecSizeSp, int VecSizeIc,
          bool IsTrain, bool FuseNormRelu, bool FuseNormAddRelu>
void BNForwardOptimizedEltwise(ITEX_GPUStream* stream, const InT* in,
                               const ScaleT* mean, const ScaleT* var,
                               const ScaleT* scale, const ScaleT* offset,
                               const InT* side_input, InT* out,
                               const ScaleT* old_mean, const ScaleT* old_var,
                               ScaleT* new_mean, ScaleT* new_var,
                               ScaleT* saved_mean, ScaleT* saved_var,
                               const int sp, const int ic, const float epsilon,
                               const float exponential_avg_factor) {
  const int nelems = sp * ic;
  const int max_wg_size =
      (*stream)
          .get_device()
          .get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_wg_size);
  int num_wg = DivUp(nelems / (VecSizeIc * VecSizeSp), group_size);
  sycl::nd_range<1> range(num_wg * group_size, group_size);

  stream->submit([&](sycl::handler& cgh) {
    BnForwardOptimizedKernel<InT, ScaleT, VecSizeSp, VecSizeIc, IsTrain,
                             FuseNormRelu, FuseNormAddRelu>
        task(in, mean, var, scale, offset, side_input, out, old_mean, old_var,
             new_mean, new_var, saved_mean, saved_var, sp, ic, epsilon,
             exponential_avg_factor);
    cgh.parallel_for<
        BnForwardOptimizedKernel<InT, ScaleT, VecSizeSp, VecSizeIc, IsTrain,
                                 FuseNormRelu, FuseNormAddRelu>>(range, task);
  });
}

template <typename InT, typename ScaleT, int VecSizeSp, bool IsTrain,
          bool FuseNormRelu, bool FuseNormAddRelu>
struct BnForwardKernel {
  BnForwardKernel(const InT* in_, const ScaleT* mean_, const ScaleT* var_,
                  const ScaleT* scale_, const ScaleT* offset_,
                  const InT* side_input_, InT* out_, const ScaleT* old_mean_,
                  const ScaleT* old_var_, ScaleT* new_mean_, ScaleT* new_var_,
                  ScaleT* saved_mean_, ScaleT* saved_var_, const int sp_,
                  const int ic_, const float epsilon_,
                  const float exponential_avg_factor_)
      : in(in_),
        mean(mean_),
        var(var_),
        scale(scale_),
        offset(offset_),
        side_input(side_input_),
        out(out_),
        old_mean(old_mean_),
        old_var(old_var_),
        new_mean(new_mean_),
        new_var(new_var_),
        saved_mean(saved_mean_),
        saved_var(saved_var_),
        sp(sp_),
        ic(ic_),
        epsilon(epsilon_),
        exponential_avg_factor(exponential_avg_factor_) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    const int nelems = sp * ic;
    int sp_idx = id / ic;
    int ic_idx = id - sp_idx * ic;
    int base_offset = sp_idx * VecSizeSp * ic + ic_idx;
    float correct_factor =
        static_cast<float>(sp) / sycl::max(static_cast<float>(sp - 1), 1.0f);

    if (base_offset >= nelems) return;
    InT values[VecSizeSp], out_values[VecSizeSp];
    InT side_input_values[VecSizeSp];

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      if (offset < nelems)
        values[i] = in[offset];
      else
        values[i] = InT(0);
    }

    if (FuseNormAddRelu) {
#pragma unroll
      for (int i = 0; i < VecSizeSp; ++i) {
        int offset = base_offset + i * ic;
        if (offset < nelems)
          side_input_values[i] = side_input[offset];
        else
          side_input_values[i] = InT(0);
      }
    }

    ScaleT mean_value = mean[ic_idx];
    ScaleT var_value = var[ic_idx];
    // Note: as training we only calculated sum(input)/sum(input_of_square), not
    // real mean/var but inference, we used estimated mean/var
    if (IsTrain) {
      mean_value /= sp;
      var_value /= sp;
      var_value -= mean_value * mean_value;
    }
    ScaleT inv = sycl::rsqrt(var_value + epsilon) * scale[ic_idx];

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      out_values[i] = static_cast<InT>(
          (static_cast<ScaleT>(values[i]) - mean_value) * inv + offset[ic_idx]);
      if (FuseNormRelu) {
        out_values[i] = out_values[i] > InT(0) ? out_values[i] : InT(0);
      }
      if (FuseNormAddRelu) {
        out_values[i] += side_input_values[i];
        out_values[i] = out_values[i] > InT(0) ? out_values[i] : InT(0);
      }
    }

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      if (offset < nelems) out[offset] = out_values[i];
    }
    if (sp_idx == 0) {
      ScaleT old_mean_value = ScaleT(0);
      ScaleT old_var_value = ScaleT(0);
      if (!IsTrain || exponential_avg_factor != 1) {
        old_mean_value = old_mean[ic_idx];
        old_var_value = old_var[ic_idx];
      }
      if (IsTrain) {
        saved_mean[ic_idx] = mean_value;
        saved_var[ic_idx] = var_value;
        new_mean[ic_idx] = (1 - exponential_avg_factor) * old_mean_value +
                           exponential_avg_factor * mean_value;
        new_var[ic_idx] = (1 - exponential_avg_factor) * old_var_value +
                          (exponential_avg_factor * correct_factor) * var_value;
      } else {
        new_mean[ic_idx] = mean_value;
        new_var[ic_idx] = var_value;
      }
    }
  }
  const InT* in;
  const ScaleT* mean;
  const ScaleT* var;
  const ScaleT* scale;
  const ScaleT* offset;
  const InT* side_input;
  InT* out;
  const ScaleT* old_mean;
  const ScaleT* old_var;
  ScaleT* new_mean;
  ScaleT* new_var;
  ScaleT* saved_mean;
  ScaleT* saved_var;
  const int sp;
  const int ic;
  const float epsilon;
  const float exponential_avg_factor;
};

template <typename InT, typename ScaleT, int VecSizeSp, bool IsTrain,
          bool FuseNormRelu, bool FuseNormAddRelu>
void BNForwardEltwise(ITEX_GPUStream* stream, const InT* in, const ScaleT* mean,
                      const ScaleT* var, const ScaleT* scale,
                      const ScaleT* offset, const InT* side_input, InT* out,
                      const ScaleT* old_mean, const ScaleT* old_var,
                      ScaleT* new_mean, ScaleT* new_var, ScaleT* saved_mean,
                      ScaleT* saved_var, const int sp, const int ic,
                      const float epsilon, const float exponential_avg_factor) {
  const int nelems = sp * ic;
  const int max_wg_size =
      (*stream)
          .get_device()
          .get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_wg_size);
  int num_wg =
      ((nelems + VecSizeSp - 1) / VecSizeSp + group_size - 1) / group_size;
  sycl::nd_range<1> range(num_wg * group_size, group_size);

  stream->submit([&](sycl::handler& cgh) {
    BnForwardKernel<InT, ScaleT, VecSizeSp, IsTrain, FuseNormRelu,
                    FuseNormAddRelu>
        task(in, mean, var, scale, offset, side_input, out, old_mean, old_var,
             new_mean, new_var, saved_mean, saved_var, sp, ic, epsilon,
             exponential_avg_factor);
    cgh.parallel_for<BnForwardKernel<InT, ScaleT, VecSizeSp, IsTrain,
                                     FuseNormRelu, FuseNormAddRelu>>(range,
                                                                     task);
  });
}

template <typename InT, typename ScaleT, bool IsTrain, bool FuseNormRelu,
          bool FuseNormAddRelu>
void BNForward(OpKernelContext* context, const InT* in, const ScaleT* mean,
               const ScaleT* var, const ScaleT* scale, const ScaleT* offset,
               const InT* side_input, InT* out, const ScaleT* old_mean,
               const ScaleT* old_var, ScaleT* new_mean, ScaleT* new_var,
               ScaleT* saved_mean, ScaleT* saved_var, const int sp,
               const int ic, const float epsilon,
               const float exponential_avg_factor) {
  auto* stream = context->GetDeviceStream();
  constexpr int VecSizeSp = 4;
  constexpr int VecSizeIc = sizeof(float) / sizeof(InT);
  bool use_optimized_impl = (sp % VecSizeSp == 0) && (ic % VecSizeIc == 0);
  if (use_optimized_impl) {
    BNForwardOptimizedEltwise<InT, ScaleT, VecSizeSp, VecSizeIc, IsTrain,
                              FuseNormRelu, FuseNormAddRelu>(
        stream, in, mean, var, scale, offset, side_input, out, old_mean,
        old_var, new_mean, new_var, saved_mean, saved_var, sp, ic, epsilon,
        exponential_avg_factor);
  } else {
    BNForwardEltwise<InT, ScaleT, VecSizeSp, IsTrain, FuseNormRelu,
                     FuseNormAddRelu>(stream, in, mean, var, scale, offset,
                                      side_input, out, old_mean, old_var,
                                      new_mean, new_var, saved_mean, saved_var,
                                      sp, ic, epsilon, exponential_avg_factor);
  }
}

template <typename InT, typename OutT, bool FuseNormRelu, bool FuseNormAddRelu>
struct SimpleBwkReductionKernel {
  SimpleBwkReductionKernel(const InT* in_data_, const InT* grad_data_,
                           const OutT* mean_data_, const InT* y_data_,
                           OutT* out1_data_, OutT* out2_data_,
                           const int extend_x_, const int extend_y_,
                           const int extend_z_)
      : in_data(in_data_),
        grad_data(grad_data_),
        mean_data(mean_data_),
        y_data(y_data_),
        out1_data(out1_data_),
        out2_data(out2_data_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_) {}
  void operator()(sycl::nd_item<1> item) const {
    int id = item.get_global_linear_id();
    const int out_size = extend_x * extend_z;

    if (id < out_size) {
      int outer = id / extend_z;
      int inner = id - outer * extend_z;

      int in_offset = outer * extend_y * extend_z + inner;
      int ic_idx = inner % extend_z;

      OutT sum1 = OutT(0);
      OutT sum2 = OutT(0);
#pragma unroll
      for (int i = 0; i < extend_y; ++i) {
        OutT grad_value =
            static_cast<OutT>(grad_data[in_offset + i * extend_z]);
        if (FuseNormRelu || FuseNormAddRelu) {
          grad_value *=
              (y_data[in_offset + i * extend_z] > InT(0) ? OutT(1) : OutT(0));
        }
        OutT x_value = static_cast<OutT>(in_data[in_offset + i * extend_z]);
        sum1 += grad_value;
        sum2 += grad_value * (x_value - mean_data[ic_idx]);
      }
      out1_data[id] = sum1;
      out2_data[id] = sum2;
    }
  }
  const InT* in_data;
  const InT* grad_data;
  const OutT* mean_data;
  const InT* y_data;
  OutT* out1_data;
  OutT* out2_data;
  const int extend_x;
  const int extend_y;
  const int extend_z;
};

template <typename InT, typename OutT, bool FuseNormRelu, bool FuseNormAddRelu>
void SimpleBwdReduction(OpKernelContext* context, const InT* in_data,
                        const InT* grad_data, const OutT* mean_data,
                        const InT* y_data, OutT* out1_data, OutT* out2_data,
                        const int extend_x, const int extend_y,
                        const int extend_z) {
  auto* stream = context->GetDeviceStream();
  const int out_size = extend_x * extend_z;
  const int max_wg_size =
      (*stream)
          .get_device()
          .get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_wg_size);
  int num_wg = (out_size + group_size - 1) / group_size;
  sycl::nd_range<1> range(num_wg * group_size, group_size);
  stream->submit([&](sycl::handler& cgh) {
    SimpleBwkReductionKernel<InT, OutT, FuseNormRelu, FuseNormAddRelu> task(
        in_data, grad_data, mean_data, y_data, out1_data, out2_data, extend_x,
        extend_y, extend_z);
    cgh.parallel_for<
        SimpleBwkReductionKernel<InT, OutT, FuseNormRelu, FuseNormAddRelu>>(
        range, task);
  });
}

// always use float as computation type
template <typename InT, typename OutT, typename LocalAccessor,
          bool FuseNormRelu, bool FuseNormAddRelu>
struct BWKFirstStepKernel {
  BWKFirstStepKernel(const InT* in_data_, const InT* grad_data_,
                     const OutT* mean_data_, const InT* y_data_, OutT* out1_,
                     OutT* out2_, LocalAccessor scratch1_,
                     LocalAccessor scratch2_, const int extend_x_,
                     const int extend_y_, const int extend_z_,
                     const int elems_per_item_, const int num_sub_group_,
                     const int num_segments_y_)
      : in_data(in_data_),
        grad_data(grad_data_),
        mean_data(mean_data_),
        y_data(y_data_),
        out1(out1_),
        out2(out2_),
        scratch1(scratch1_),
        scratch2(scratch2_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_),
        elems_per_item(elems_per_item_),
        num_sub_group(num_sub_group_),
        num_segments_y(num_segments_y_) {}
  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<3> item) const {
    // get start index
    int x_group_id = item.get_group(0);
    int y_group_id = item.get_group(1);
    int z_group_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int subgroup_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_linear_id();

    int x_offset = x_group_id * extend_y * extend_z;

    // each subgroup load data and reduce elems_per_item
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    int z_offset = z_group_id * SubGroupSize + lane_id;
    if (z_offset < extend_z) {
      for (int i = 0; i < elems_per_item; ++i) {
        int y_idx = y_group_id * num_sub_group * elems_per_item + i +
                    subgroup_id * elems_per_item;
        if (y_idx >= extend_y) break;
        int offset = x_offset + y_idx * extend_z + z_offset;
        float tmp1 = static_cast<float>(grad_data[offset]);
        if (FuseNormRelu || FuseNormAddRelu) {
          tmp1 *= (y_data[offset] > InT(0) ? 1.0f : 0.0f);
        }
        float tmp2 = static_cast<float>(in_data[offset]);
        sum1 += tmp1;
        sum2 += tmp1 * (tmp2 - mean_data[z_offset]);
      }
    }
    // each subgroup write result to slm
    scratch1[subgroup_id + lane_id * num_sub_group] = sum1;
    scratch2[subgroup_id + lane_id * num_sub_group] = sum2;
    item.barrier(sycl::access::fence_space::local_space);

    // slm reduce and write output
    sum1 = scratch1[subgroup_id * num_sub_group + lane_id];
    sum2 = scratch2[subgroup_id * num_sub_group + lane_id];
    float update_sum1 = sycl::reduce_over_group(sg, sum1, sycl::plus<float>());
    float update_sum2 = sycl::reduce_over_group(sg, sum2, sycl::plus<float>());

    z_offset = z_group_id * SubGroupSize + subgroup_id;
    if (z_offset < extend_z) {
      int offset = x_group_id * extend_z * num_segments_y +
                   y_group_id * extend_z + z_group_id * SubGroupSize +
                   subgroup_id;
      out1[offset] = static_cast<OutT>(update_sum1);
      out2[offset] = static_cast<OutT>(update_sum2);
    }
  }
  const InT* in_data;
  const InT* grad_data;
  const OutT* mean_data;
  const InT* y_data;
  OutT* out1;
  OutT* out2;
  LocalAccessor scratch1;
  LocalAccessor scratch2;
  const int extend_x;
  const int extend_y;
  const int extend_z;
  const int elems_per_item;
  const int num_sub_group;
  const int num_segments_y;
};

template <typename InT, typename OutT, bool FuseNormRelu, bool FuseNormAddRelu>
void BwdSGColReduction(OpKernelContext* context, const InT* in_data,
                       const InT* grad_data, const OutT* mean_data,
                       const InT* y_data, OutT* out1_data, OutT* out2_data,
                       const int extend_x, const int extend_y,
                       const int extend_z) {
  auto* stream = context->GetDeviceStream();

  int elems_per_item = ElemsPerItem;
  int num_sub_group = MaxSubGroup;

  if (extend_y * 2 <= num_sub_group * elems_per_item) {
    while (num_sub_group * elems_per_item >= extend_y * 2 &&
           elems_per_item > 1) {
      elems_per_item >>= 1;
    }
  }

  int num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item);
  int num_segments_z = DivUp(extend_z, static_cast<int>(SubGroupSize));

  if (num_segments_y > 1) {
    while (num_segments_y > num_sub_group * ElemsPerItem) {
      elems_per_item <<= 1;
      num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item);
    }

    sycl::range<3> local(1, num_sub_group, SubGroupSize);
    sycl::range<3> global(extend_x, num_segments_y * local[1],
                          num_segments_z * local[2]);

    Tensor inter_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<float>::value,
                       TensorShape({2 * extend_x * num_segments_y * extend_z}),
                       &inter_tensor));
    float* inter_out1 = inter_tensor.flat<float>().data();
    float* inter_out2 = inter_out1 + extend_x * num_segments_y * extend_z;

    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<float> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<float> scratch2(num_sub_group * SubGroupSize, cgh);
      BWKFirstStepKernel<InT, float, LocalAcc<float>, FuseNormRelu,
                         FuseNormAddRelu>
          task(in_data, grad_data, mean_data, y_data, inter_out1, inter_out2,
               scratch1, scratch2, extend_x, extend_y, extend_z, elems_per_item,
               num_sub_group, num_segments_y);
      cgh.parallel_for<BWKFirstStepKernel<InT, float, LocalAcc<float>,
                                          FuseNormRelu, FuseNormAddRelu>>(
          sycl::nd_range<3>(global, local), task);
    });

    global = sycl::range<3>{static_cast<size_t>(extend_x), local[1],
                            num_segments_z * local[2]};
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<float> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<float> scratch2(num_sub_group * SubGroupSize, cgh);
      SecondStepKernel<OutT, LocalAcc<float>> task(
          inter_out1, inter_out2, out1_data, out2_data, scratch1, scratch2,
          extend_x, num_segments_y, extend_z, ElemsPerItem, num_sub_group, 1);
      cgh.parallel_for<SecondStepKernel<OutT, LocalAcc<float>>>(
          sycl::nd_range<3>(global, local), task);
    });
  } else {
    sycl::range<3> local(1, num_sub_group, SubGroupSize);
    sycl::range<3> global(extend_x, local[1], num_segments_z * local[2]);
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<float> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<float> scratch2(num_sub_group * SubGroupSize, cgh);
      BWKFirstStepKernel<InT, OutT, LocalAcc<float>, FuseNormRelu,
                         FuseNormAddRelu>
          task(in_data, grad_data, mean_data, y_data, out1_data, out2_data,
               scratch1, scratch2, extend_x, extend_y, extend_z, elems_per_item,
               num_sub_group, 1);
      cgh.parallel_for<BWKFirstStepKernel<InT, OutT, LocalAcc<float>,
                                          FuseNormRelu, FuseNormAddRelu>>(
          sycl::nd_range<3>(global, local), task);
    });
  }
}

template <typename InT, typename OutT, typename LocalAccessor,
          bool FuseNormRelu, bool FuseNormAddRelu>
struct BWKVecFirstStepKernel {
  BWKVecFirstStepKernel(const InT* in_data_, const InT* grad_data_,
                        const OutT* mean_data_, const InT* y_data_, OutT* out1_,
                        OutT* out2_, LocalAccessor scratch1_,
                        LocalAccessor scratch2_, const int extend_x_,
                        const int extend_y_, const int extend_z_,
                        const int elems_per_item_, const int num_sub_group_,
                        const int num_segments_y_, const int k_,
                        const int local_item_on_z_)
      : in_data(in_data_),
        grad_data(grad_data_),
        mean_data(mean_data_),
        y_data(y_data_),
        out1(out1_),
        out2(out2_),
        scratch1(scratch1_),
        scratch2(scratch2_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_),
        elems_per_item(elems_per_item_),
        num_sub_group(num_sub_group_),
        num_segments_y(num_segments_y_),
        k(k_),
        local_item_on_z(local_item_on_z_) {}
  [[intel::reqd_sub_group_size(SubGroupSize)]] void operator()(
      sycl::nd_item<3> item) const {
    using vecT = sycl::vec<float, VecSize>;

    // get start index
    int x_group_id = item.get_group(0);
    int y_group_id = item.get_group(1);
    int z_group_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int subgroup_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_linear_id();
    int group_z_id = lane_id % local_item_on_z;
    int group_k_id = lane_id / local_item_on_z;

    int x_offset = x_group_id * extend_y * extend_z;

    // each subgroup load data and reduce elems_per_item
    vecT aggregate1(0.0f);
    vecT aggregate2(0.0f);

    int z_offset =
        z_group_id * local_item_on_z * VecSize + group_z_id * VecSize;

    vecT mean_value = *(reinterpret_cast<const vecT*>(mean_data + z_offset));
    for (int i = 0; i < elems_per_item; ++i) {
      int y_idx = y_group_id * num_sub_group * elems_per_item * k +
                  subgroup_id * elems_per_item * k + group_k_id + i * k;
      if (y_idx >= extend_y) break;
      int offset = x_offset + y_idx * extend_z + z_offset;

      vecT tmp1, tmp2, tmp3;
      PacketLoad(grad_data, offset, &tmp1);
      PacketLoad(in_data, offset, &tmp2);

      if (FuseNormRelu || FuseNormAddRelu) {
        PacketLoad(y_data, offset, &tmp3);
        for (int j = 0; j < VecSize; ++j) {
          tmp1[j] *= (tmp3[j] > 0.0f ? 1.0f : 0.0f);
        }
      }

      for (int j = 0; j < VecSize; ++j) {
        aggregate1[j] += tmp1[j];
        aggregate2[j] += tmp1[j] * (tmp2[j] - mean_value[j]);
      }
    }
    // each subgroup write result to slm
    scratch1[subgroup_id * k + group_k_id + group_z_id * num_sub_group * k] =
        aggregate1;
    scratch2[subgroup_id * k + group_k_id + group_z_id * num_sub_group * k] =
        aggregate2;
    item.barrier(sycl::access::fence_space::local_space);

    // ------------------------------------------------------------------
    // -------------slm reduce-------------------
    // slm: (SubGroupSize * k) * local_item_on_z
    // ------------------------------------------------------------------
    int slm_z_id = subgroup_id / k;
    int slm_k_id = subgroup_id % k;
    vecT value1 = scratch1[slm_z_id * num_sub_group * k +
                           slm_k_id * SubGroupSize + lane_id];
    vecT value2 = scratch2[slm_z_id * num_sub_group * k +
                           slm_k_id * SubGroupSize + lane_id];

    // reduce within each subgroup
    for (int i = 0; i < VecSize; ++i) {
      value1[i] = sycl::reduce_over_group(sg, value1[i], sycl::plus<float>());
      value2[i] = sycl::reduce_over_group(sg, value2[i], sycl::plus<float>());
    }

    // lane0 write result of each subgrop
    if (lane_id == 0) {
      scratch1[slm_z_id * num_sub_group * k + slm_k_id * SubGroupSize] = value1;
      scratch2[slm_z_id * num_sub_group * k + slm_k_id * SubGroupSize] = value2;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // collect result of k subgroup and store output
    if (lane_id == 0 && slm_k_id == 0) {
      vecT tmp1 = scratch1[slm_z_id * k * num_sub_group];
      vecT tmp2 = scratch2[slm_z_id * k * num_sub_group];
      for (int i = 1; i < k; ++i) {
        tmp1 += scratch1[slm_z_id * k * num_sub_group + i * SubGroupSize];
        tmp2 += scratch2[slm_z_id * k * num_sub_group + i * SubGroupSize];
      }
      int offset = x_group_id * extend_z * num_segments_y +
                   y_group_id * extend_z +
                   z_group_id * local_item_on_z * VecSize + slm_z_id * VecSize;

      PacketStore(out1, offset, tmp1);
      PacketStore(out2, offset, tmp2);
    }
  }
  const InT* in_data;
  const InT* grad_data;
  const OutT* mean_data;
  const InT* y_data;
  OutT* out1;
  OutT* out2;
  LocalAccessor scratch1;
  LocalAccessor scratch2;
  const int extend_x;
  const int extend_y;
  const int extend_z;
  const int elems_per_item;
  const int num_sub_group;
  const int num_segments_y;
  const int k;
  const int local_item_on_z;
};

template <typename InT, typename OutT, bool FuseNormRelu, bool FuseNormAddRelu>
void BwdSGVecColReduction(OpKernelContext* context, const InT* in_data,
                          const InT* grad_data, const OutT* mean_data,
                          const InT* y_data, OutT* out1_data, OutT* out2_data,
                          int extend_x, int extend_y, int extend_z) {
  using vecT = sycl::vec<float, VecSize>;

  int elems_per_item = ElemsPerItem;
  int num_sub_group = MaxSubGroup;
  int work_item_on_z = extend_z / VecSize;
  int local_item_on_z =
      work_item_on_z <= MaxLocalItemOnZ ? work_item_on_z : MaxLocalItemOnZ;
  int k = SubGroupSize / local_item_on_z;

  if (extend_y * 2 <= num_sub_group * elems_per_item * k) {
    while (num_sub_group * elems_per_item * k >= extend_y * 2 &&
           elems_per_item > 1) {
      elems_per_item >>= 1;
    }
  }

  int num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item * k);
  int num_segments_z = work_item_on_z / local_item_on_z;

  auto* stream = context->GetDeviceStream();

  if (num_segments_y > 1) {
    while (num_segments_y > num_sub_group * ElemsPerItem * k) {
      elems_per_item <<= 1;
      num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item * k);
    }

    sycl::range<3> local(1, num_sub_group * k, local_item_on_z);
    sycl::range<3> global(extend_x, num_segments_y * local[1],
                          num_segments_z * local[2]);

    Tensor inter_out_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<float>::value,
                       TensorShape({2 * extend_x * num_segments_y * extend_z}),
                       &inter_out_tensor));
    float* inter_out1 = inter_out_tensor.flat<float>().data();
    float* inter_out2 = inter_out1 + extend_x * num_segments_y * extend_z;

    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<vecT> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<vecT> scratch2(num_sub_group * SubGroupSize, cgh);
      BWKVecFirstStepKernel<InT, float, LocalAcc<vecT>, FuseNormRelu,
                            FuseNormAddRelu>
          task(in_data, grad_data, mean_data, y_data, inter_out1, inter_out2,
               scratch1, scratch2, extend_x, extend_y, extend_z, elems_per_item,
               num_sub_group, num_segments_y, k, local_item_on_z);
      cgh.parallel_for<BWKVecFirstStepKernel<InT, float, LocalAcc<vecT>,
                                             FuseNormRelu, FuseNormAddRelu>>(
          sycl::nd_range<3>(global, local), task);
    });

    global = sycl::range<3>{static_cast<size_t>(extend_x), local[1],
                            num_segments_z * local[2]};
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<vecT> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<vecT> scratch2(num_sub_group * SubGroupSize, cgh);
      VecSecondStepKernel<float, LocalAcc<vecT>> task(
          inter_out1, inter_out2, out1_data, out2_data, scratch1, scratch2,
          extend_x, num_segments_y, extend_z, ElemsPerItem, num_sub_group, 1, k,
          local_item_on_z);
      cgh.parallel_for<VecSecondStepKernel<float, LocalAcc<vecT>>>(
          sycl::nd_range<3>(global, local), task);
    });
  } else {
    sycl::range<3> local(1, num_sub_group * k, local_item_on_z);
    sycl::range<3> global(extend_x, local[1], num_segments_z * local[2]);
    stream->submit([&](sycl::handler& cgh) {
      LocalAcc<vecT> scratch1(num_sub_group * SubGroupSize, cgh);
      LocalAcc<vecT> scratch2(num_sub_group * SubGroupSize, cgh);
      BWKVecFirstStepKernel<InT, OutT, LocalAcc<vecT>, FuseNormRelu,
                            FuseNormAddRelu>
          task(in_data, grad_data, mean_data, y_data, out1_data, out2_data,
               scratch1, scratch2, extend_x, extend_y, extend_z, elems_per_item,
               num_sub_group, 1, k, local_item_on_z);
      cgh.parallel_for<BWKVecFirstStepKernel<InT, OutT, LocalAcc<vecT>,
                                             FuseNormRelu, FuseNormAddRelu>>(
          sycl::nd_range<3>(global, local), task);
    });
  }
}

template <typename InT, typename OutT, bool FuseNormRelu, bool FuseNormAddRelu>
void BnormBwkReduction(OpKernelContext* context, const InT* in_data,
                       const InT* grad_data, const OutT* mean_data,
                       const InT* y_data, OutT* out1_data, OutT* out2_data,
                       int extend_x, int extend_y, int extend_z) {
  int elems_per_item = extend_y / (extend_x * extend_z);
  bool use_vectorization_pass = (extend_z % VecSize == 0 &&
                                 (SubGroupSize % (extend_z / VecSize) == 0 ||
                                  (extend_z / VecSize) % MaxLocalItemOnZ == 0));
  if (extend_y < 32 && elems_per_item < 4)
    SimpleBwdReduction<InT, OutT, FuseNormRelu, FuseNormAddRelu>(
        context, in_data, grad_data, mean_data, y_data, out1_data, out2_data,
        extend_x, extend_y, extend_z);
  else if (use_vectorization_pass)
    BwdSGVecColReduction<InT, OutT, FuseNormRelu, FuseNormAddRelu>(
        context, in_data, grad_data, mean_data, y_data, out1_data, out2_data,
        extend_x, extend_y, extend_z);
  else
    BwdSGColReduction<InT, OutT, FuseNormRelu, FuseNormAddRelu>(
        context, in_data, grad_data, mean_data, y_data, out1_data, out2_data,
        extend_x, extend_y, extend_z);
}

template <typename InT, typename ScaleT, int VecSizeSp, int VecSizeIc,
          bool FuseNormRelu, bool FuseNormAddRelu, bool Training>
struct BnBackwardOptimizedKernel {
  BnBackwardOptimizedKernel(const InT* x_, const InT* dy_, const ScaleT* mean_,
                            const ScaleT* var_, const InT* y_,
                            const ScaleT* scale_, const ScaleT* sum_dy_,
                            const ScaleT* sum_dy_x_center_, InT* dx_,
                            ScaleT* dscale_, ScaleT* doffset_, InT* dside_x_,
                            const int sp_, const int ic_, float epsilon_)
      : x(x_),
        dy(dy_),
        mean(mean_),
        var(var_),
        y(y_),
        scale(scale_),
        sum_dy(sum_dy_),
        sum_dy_x_center(sum_dy_x_center_),
        dx(dx_),
        dscale(dscale_),
        doffset(doffset_),
        dside_x(dside_x_),
        sp(sp_),
        ic(ic_),
        epsilon(epsilon_) {}
  void operator()(sycl::nd_item<1> item) const {
    using VecT = sycl::vec<InT, VecSizeIc>;
    const int nelems = sp * ic;
    auto id = item.get_global_linear_id();
    const int n = ic / VecSizeIc;
    const int x_id = id / n;
    const int y_id = id - (x_id * n);

    const int y_offset = y_id * VecSizeIc;
    const int x_offset = x_id * VecSizeSp;
    const int base_offset = x_offset * ic + y_offset;

    if (base_offset >= nelems) return;
    InT x_values[VecSizeSp * VecSizeIc];
    InT dx_values[VecSizeSp * VecSizeIc];
    InT dy_values[VecSizeSp * VecSizeIc];
    VecT* vec_x_values = reinterpret_cast<VecT*>(x_values);
    VecT* vec_dx_values = reinterpret_cast<VecT*>(dx_values);
    VecT* vec_dy_values = reinterpret_cast<VecT*>(dy_values);

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      vec_x_values[i] = *(reinterpret_cast<const VecT*>(x + offset));
      vec_dy_values[i] = *(reinterpret_cast<const VecT*>(dy + offset));
      if (FuseNormRelu || FuseNormAddRelu) {
        VecT tmp = *(reinterpret_cast<const VecT*>(y + offset));
        for (int j = 0; j < VecSizeIc; ++j) {
          vec_dy_values[i][j] *= (tmp[j] > 0 ? 1 : 0);
        }
      }
    }

    int ic_idx[VecSizeIc];
#pragma unroll
    for (int j = 0; j < VecSizeIc; ++j) {
      ic_idx[j] = y_offset + j;
    }

    ScaleT dscale_values[VecSizeIc];
    ScaleT doffset_values[VecSizeIc];

    for (int j = 0; j < VecSizeIc; ++j) {
      ScaleT inv = sycl::rsqrt(var[ic_idx[j]] + epsilon);
      ScaleT sum_dy_value = sum_dy[ic_idx[j]];
      ScaleT sum_dy_x_center_value = sum_dy_x_center[ic_idx[j]];
      ScaleT coef = sum_dy_x_center_value / (sp * (var[ic_idx[j]] + epsilon));
      doffset_values[j] = sum_dy_value;
      dscale_values[j] = sum_dy_x_center_value * inv;

      for (int i = 0; i < VecSizeSp; ++i) {
        if (Training) {
          dx_values[i * VecSizeIc + j] =
              scale[ic_idx[j]] * inv *
              (dy_values[i * VecSizeIc + j] - (sum_dy_value / sp) -
               coef * (x_values[i * VecSizeIc + j] - mean[ic_idx[j]]));
        } else {
          dx_values[i * VecSizeIc + j] =
              dy_values[i * VecSizeIc + j] * scale[ic_idx[j]] * inv;
        }
      }
    }

    if (x_id == 0) {
      for (int j = 0; j < VecSizeIc; ++j) {
        doffset[y_offset + j] = doffset_values[j];
        dscale[y_offset + j] = dscale_values[j];
      }
    }

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      *(reinterpret_cast<VecT*>(dx + offset)) = vec_dx_values[i];
      if (FuseNormAddRelu)
        *(reinterpret_cast<VecT*>(dside_x + offset)) = vec_dy_values[i];
    }
  }
  const InT* x;
  const InT* dy;
  const ScaleT* mean;
  const ScaleT* var;
  const InT* y;
  const ScaleT* scale;
  const ScaleT* sum_dy;
  const ScaleT* sum_dy_x_center;
  InT* dx;
  ScaleT* dscale;
  ScaleT* doffset;
  InT* dside_x;
  const int sp;
  const int ic;
  const float epsilon;
};

template <typename ScaleT, int VecSizeSp, int VecSizeIc, bool FuseNormRelu,
          bool FuseNormAddRelu, bool Training>
struct BnBackwardOptimizedKernel<Eigen::bfloat16, ScaleT, VecSizeSp, VecSizeIc,
                                 FuseNormRelu, FuseNormAddRelu, Training> {
  typedef Eigen::bfloat16 InT;
  typedef uint16_t T;
  BnBackwardOptimizedKernel(const InT* x_, const InT* dy_, const ScaleT* mean_,
                            const ScaleT* var_, const InT* y_,
                            const ScaleT* scale_, const ScaleT* sum_dy_,
                            const ScaleT* sum_dy_x_center_, InT* dx_,
                            ScaleT* dscale_, ScaleT* doffset_, InT* dside_x_,
                            const int sp_, const int ic_, float epsilon_)
      : x(x_),
        dy(dy_),
        mean(mean_),
        var(var_),
        y(y_),
        scale(scale_),
        sum_dy(sum_dy_),
        sum_dy_x_center(sum_dy_x_center_),
        dx(dx_),
        dscale(dscale_),
        doffset(doffset_),
        dside_x(dside_x_),
        sp(sp_),
        ic(ic_),
        epsilon(epsilon_) {}
  void operator()(sycl::nd_item<1> item) const {
    using VecT = sycl::vec<T, VecSizeIc>;
    const int nelems = sp * ic;
    auto id = item.get_global_linear_id();
    int n = ic / VecSizeIc;
    int x_id = id / n;
    int y_id = id - (x_id * n);

    int y_offset = y_id * VecSizeIc;
    int x_offset = x_id * VecSizeSp;
    int base_offset = x_offset * ic + y_offset;

    if (base_offset >= nelems) return;
    T x_values[VecSizeSp * VecSizeIc];
    T dx_values[VecSizeSp * VecSizeIc];
    T dy_values[VecSizeSp * VecSizeIc];
    VecT* vec_x_values = reinterpret_cast<VecT*>(x_values);
    VecT* vec_dx_values = reinterpret_cast<VecT*>(dx_values);
    VecT* vec_dy_values = reinterpret_cast<VecT*>(dy_values);

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      vec_x_values[i] = *(reinterpret_cast<const VecT*>(x + offset));
      vec_dy_values[i] = *(reinterpret_cast<const VecT*>(dy + offset));
      if (FuseNormRelu || FuseNormAddRelu) {
        VecT tmp = *(reinterpret_cast<const VecT*>(y + offset));
        for (int j = 0; j < VecSizeIc; ++j) {
          vec_dy_values[i][j] *= (Eigen::bfloat16_impl::raw_uint16_to_bfloat16(
                                      tmp[j]) > Eigen::bfloat16(0)
                                      ? T(1)
                                      : T(0));
        }
      }
    }

    int ic_idx[VecSizeIc];
#pragma unroll
    for (int j = 0; j < VecSizeIc; ++j) {
      ic_idx[j] = y_offset + j;
    }

    ScaleT dscale_values[VecSizeIc];
    ScaleT doffset_values[VecSizeIc];

    for (int j = 0; j < VecSizeIc; ++j) {
      ScaleT inv = sycl::rsqrt(var[ic_idx[j]] + epsilon);
      ScaleT sum_dy_value = sum_dy[ic_idx[j]];
      ScaleT sum_dy_x_center_value = sum_dy_x_center[ic_idx[j]];
      ScaleT coef = sum_dy_x_center_value / (sp * (var[ic_idx[j]] + epsilon));
      doffset_values[j] = sum_dy_value;
      dscale_values[j] = sum_dy_x_center_value * inv;

      for (int i = 0; i < VecSizeSp; ++i) {
        if (Training) {
          ScaleT temp = scale[ic_idx[j]] * inv *
                        ((Eigen::bfloat16_impl::bfloat16_to_float(
                             Eigen::bfloat16_impl::raw_uint16_to_bfloat16(
                                 dy_values[i * VecSizeIc + j]))) -
                         (sum_dy_value / sp) -
                         (Eigen::bfloat16_impl::bfloat16_to_float(
                              Eigen::bfloat16_impl::raw_uint16_to_bfloat16(
                                  x_values[i * VecSizeIc + j])) -
                          mean[ic_idx[j]]) *
                             coef);
          dx_values[i * VecSizeIc + j] =
              Eigen::bfloat16_impl::float_to_bfloat16_rtne<true>(temp).value;
        } else {
          ScaleT temp = Eigen::bfloat16_impl::bfloat16_to_float(
                            Eigen::bfloat16_impl::raw_uint16_to_bfloat16(
                                dy_values[i * VecSizeIc + j])) *
                        scale[ic_idx[j]] * inv;
          dx_values[i * VecSizeIc + j] =
              Eigen::bfloat16_impl::float_to_bfloat16_rtne<true>(temp).value;
        }
      }
    }

    if (x_id == 0) {
      for (int j = 0; j < VecSizeIc; ++j) {
        doffset[y_offset + j] = doffset_values[j];
        dscale[y_offset + j] = dscale_values[j];
      }
    }

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      *(reinterpret_cast<VecT*>(dx + offset)) = vec_dx_values[i];
      if (FuseNormAddRelu)
        *(reinterpret_cast<VecT*>(dside_x + offset)) = vec_dy_values[i];
    }
  }
  const InT* x;
  const InT* dy;
  const ScaleT* mean;
  const ScaleT* var;
  const InT* y;
  const ScaleT* scale;
  const ScaleT* sum_dy;
  const ScaleT* sum_dy_x_center;
  InT* dx;
  ScaleT* dscale;
  ScaleT* doffset;
  InT* dside_x;
  const int sp;
  const int ic;
  const float epsilon;
};

template <typename InT, typename ScaleT, int VecSizeSp, int VecSizeIc,
          bool FuseNormRelu, bool FuseNormAddRelu, bool Training>
void bn_backward_optimized_kernel(
    ITEX_GPUStream* stream, const InT* x, const InT* dy, const ScaleT* mean,
    const ScaleT* var, const InT* y, const ScaleT* scale, const ScaleT* sum_dy,
    const ScaleT* sum_dy_x_center, InT* dx, ScaleT* dscale, ScaleT* doffset,
    InT* dside_x, const int sp, const int ic, const float epsilon) {
  const int nelems_vec = (sp / VecSizeSp) * (ic / VecSizeIc);
  const int max_wg_size =
      (*stream)
          .get_device()
          .get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_wg_size);
  int num_wg = DivUp(nelems_vec, group_size);
  sycl::nd_range<1> range(num_wg * group_size, group_size);

  stream->submit([&](sycl::handler& cgh) {
    BnBackwardOptimizedKernel<InT, ScaleT, VecSizeSp, VecSizeIc, FuseNormRelu,
                              FuseNormAddRelu, Training>
        task(x, dy, mean, var, y, scale, sum_dy, sum_dy_x_center, dx, dscale,
             doffset, dside_x, sp, ic, epsilon);
    cgh.parallel_for<
        BnBackwardOptimizedKernel<InT, ScaleT, VecSizeSp, VecSizeIc,
                                  FuseNormRelu, FuseNormAddRelu, Training>>(
        range, task);
  });
}

template <typename InT, typename ScaleT, int VecSizeSp, bool FuseNormRelu,
          bool FuseNormAddRelu, bool Training>
struct BnBackwardKernel {
  BnBackwardKernel(const InT* x_, const InT* dy_, const ScaleT* mean_,
                   const ScaleT* var_, const InT* y_, const ScaleT* scale_,
                   const ScaleT* sum_dy_, const ScaleT* sum_dy_x_center_,
                   InT* dx_, ScaleT* dscale_, ScaleT* doffset_, InT* dside_x_,
                   const int sp_, const int ic_, const float epsilon_)
      : x(x_),
        dy(dy_),
        mean(mean_),
        var(var_),
        y(y_),
        scale(scale_),
        sum_dy(sum_dy_),
        sum_dy_x_center(sum_dy_x_center_),
        dx(dx_),
        dscale(dscale_),
        doffset(doffset_),
        dside_x(dside_x_),
        sp(sp_),
        ic(ic_),
        epsilon(epsilon_) {}
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    const int nelems = sp * ic;

    int sp_idx = id / ic;
    int ic_idx = id - sp_idx * ic;
    const int base_offset = sp_idx * VecSizeSp * ic + ic_idx;

    if (base_offset >= nelems) return;
    InT x_values[VecSizeSp], dy_values[VecSizeSp], dx_values[VecSizeSp];

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      if (offset < nelems) {
        x_values[i] = x[base_offset + i * ic];
        dy_values[i] = dy[base_offset + i * ic];
        if (FuseNormRelu || FuseNormAddRelu) {
          InT temp_y = y[base_offset + i * ic];
          dy_values[i] *= (temp_y > InT(0) ? InT(1) : InT(0));
        }
      } else {
        x_values[i] = InT(0);
        dy_values[i] = InT(0);
      }
    }

    float var_value = var[ic_idx];
    float inv = sycl::rsqrt(var_value + epsilon);
    float sum_dy_value = sum_dy[ic_idx];
    float sum_dy_x_center_value = sum_dy_x_center[ic_idx];
    float coef = sum_dy_x_center_value / (sp * (var_value + epsilon));

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      float mean_dy = sum_dy_value / sp;
      if (Training) {
        dx_values[i] = static_cast<InT>(
            scale[ic_idx] * inv *
            (dy_values[i] - mean_dy - (x_values[i] - mean[ic_idx]) * coef));
      } else {
        dx_values[i] = static_cast<InT>(dy_values[i] * scale[ic_idx] * inv);
      }
    }
    if (sp_idx == 0) {
      doffset[ic_idx] = sum_dy_value;
      dscale[ic_idx] = sum_dy_x_center_value * inv;
    }

#pragma unroll
    for (int i = 0; i < VecSizeSp; ++i) {
      int offset = base_offset + i * ic;
      if (offset < nelems) {
        dx[offset] = dx_values[i];
        if (FuseNormAddRelu) dside_x[offset] = dy_values[i];
      }
    }
  }
  const InT* x;
  const InT* dy;
  const ScaleT* mean;
  const ScaleT* var;
  const InT* y;
  const ScaleT* scale;
  const ScaleT* sum_dy;
  const ScaleT* sum_dy_x_center;
  InT* dx;
  ScaleT* dscale;
  ScaleT* doffset;
  InT* dside_x;
  const int sp;
  const int ic;
  const float epsilon;
};

template <typename InT, typename ScaleT, int VecSizeSp, bool FuseNormRelu,
          bool FuseNormAddRelu, bool Training>
void bn_backward_kernel(ITEX_GPUStream* stream, const InT* x, const InT* dy,
                        const ScaleT* mean, const ScaleT* var, const InT* y,
                        const ScaleT* scale, const ScaleT* sum_dy,
                        const ScaleT* sum_dy_x_center, InT* dx, ScaleT* dscale,
                        ScaleT* doffset, InT* dside_x, const int sp,
                        const int ic, const float epsilon) {
  const int nelems = sp * ic;
  const int max_wg_size =
      (*stream)
          .get_device()
          .get_info<sycl::info::device::max_work_group_size>();
  int group_size = std::min(512, max_wg_size);
  int num_wg =
      ((nelems + VecSizeSp - 1) / VecSizeSp + group_size - 1) / group_size;
  sycl::nd_range<1> range(num_wg * group_size, group_size);

  stream->submit([&](sycl::handler& cgh) {
    BnBackwardKernel<InT, ScaleT, VecSizeSp, FuseNormRelu, FuseNormAddRelu,
                     Training>
        task(x, dy, mean, var, y, scale, sum_dy, sum_dy_x_center, dx, dscale,
             doffset, dside_x, sp, ic, epsilon);
    cgh.parallel_for<BnBackwardKernel<InT, ScaleT, VecSizeSp, FuseNormRelu,
                                      FuseNormAddRelu, Training>>(range, task);
  });
}

template <typename InT, typename ScaleT, bool FuseNormRelu,
          bool FuseNormAddRelu, bool Training>
void BnBackward(OpKernelContext* context, const InT* x, const InT* dy,
                const ScaleT* mean, const ScaleT* var, const InT* y,
                const ScaleT* scale, const ScaleT* sum_dy,
                const ScaleT* sum_dy_x_center, InT* dx, ScaleT* dscale,
                ScaleT* doffset, InT* dside_x, const int sp, const int ic,
                const float epsilon) {
  constexpr int VecSizeSp = 4;
  constexpr int VecSizeIc = sizeof(float) / sizeof(InT);
  bool use_optimized_impl = (sp % VecSizeSp == 0) && (ic % VecSizeIc == 0);
  auto* stream = context->GetDeviceStream();

  if (use_optimized_impl) {
    bn_backward_optimized_kernel<InT, ScaleT, VecSizeSp, VecSizeIc,
                                 FuseNormRelu, FuseNormAddRelu, Training>(
        stream, x, dy, mean, var, y, scale, sum_dy, sum_dy_x_center, dx, dscale,
        doffset, dside_x, sp, ic, epsilon);
  } else {
    bn_backward_kernel<InT, ScaleT, VecSizeSp, FuseNormRelu, FuseNormAddRelu,
                       Training>(stream, x, dy, mean, var, y, scale, sum_dy,
                                 sum_dy_x_center, dx, dscale, doffset, dside_x,
                                 sp, ic, epsilon);
  }
}
}  // namespace functor
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_CUSTOM_FUSED_BATCH_NORM_FUNCTOR_H_
