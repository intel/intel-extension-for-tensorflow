#ifndef ITEX_CORE_KERNELS_GPU_COL_REDUCTION_KERNELS_H_
#define ITEX_CORE_KERNELS_GPU_COL_REDUCTION_KERNELS_H_

#include <algorithm>

#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {

struct ColReductionPolicy {
  static constexpr int GROUP_SIZE = 256;
  static constexpr int SUB_GROUP_SIZE = 16;
  static constexpr int NUM_SUB_GROUP = GROUP_SIZE / SUB_GROUP_SIZE;
  static constexpr int ITEM_PER_THREAD = 16;
  static constexpr int MAX_LOCAL_ITEM_ON_Z = 8;
};

#define PACKET_DEF(T, N)                                                      \
  inline void PacketLoad(const T* ptr, int offset, sycl::vec<T, N>* array) {  \
    *array = *(reinterpret_cast<const sycl::vec<T, N>*>(ptr + offset));       \
  }                                                                           \
  inline void PacketStore(T* ptr, int offset, const sycl::vec<T, N>& array) { \
    *(reinterpret_cast<sycl::vec<T, N>*>(ptr + offset)) = array;              \
  }
PACKET_DEF(float, 4)
PACKET_DEF(float, 8)
PACKET_DEF(double, 2)
PACKET_DEF(itex::int32, 4)
PACKET_DEF(itex::int64, 2)
PACKET_DEF(uint8_t, 16)
#undef PACKET_DEF

inline void PacketLoad(const sycl::half* ptr, int offset,
                       sycl::vec<float, 8>* array) {
  sycl::vec<sycl::half, 8> in_array =
      *(reinterpret_cast<const sycl::vec<sycl::half, 8>*>(ptr + offset));

#pragma unroll
  for (int i = 0; i < 8; ++i) (*array)[i] = static_cast<float>(in_array[i]);
}

inline void PacketLoad(const Eigen::half* ptr, int offset,
                       sycl::vec<float, 8>* array) {
  sycl::vec<sycl::half, 8> in_array =
      *(reinterpret_cast<const sycl::vec<sycl::half, 8>*>(ptr + offset));

#pragma unroll
  for (int i = 0; i < 8; ++i) (*array)[i] = static_cast<float>(in_array[i]);
}

#define PACKET_STORE_HALF(T, N)                                             \
  inline void PacketStore(T* ptr, int offset,                               \
                          const sycl::vec<float, N>& array) {               \
    sycl::vec<sycl::half, N> tmp;                                           \
    for (int i = 0; i < N; ++i) tmp[i] = static_cast<sycl::half>(array[i]); \
    *(reinterpret_cast<sycl::vec<sycl::half, N>*>(ptr + offset)) = tmp;     \
  }
PACKET_STORE_HALF(sycl::half, 4)
PACKET_STORE_HALF(sycl::half, 8)
PACKET_STORE_HALF(Eigen::half, 4)
PACKET_STORE_HALF(Eigen::half, 8)
#undef PACKET_STORE_HALF

inline void PacketLoad(const Eigen::bfloat16* ptr, int offset,
                       sycl::vec<float, 8>* array) {
  sycl::vec<uint16_t, 8> in_array =
      *(reinterpret_cast<const sycl::vec<uint16_t, 8>*>(ptr + offset));

#pragma unroll
  for (int i = 0; i < 8; ++i)
    (*array)[i] = Eigen::bfloat16_impl::bfloat16_to_float(
        Eigen::bfloat16_impl::raw_uint16_to_bfloat16(in_array[i]));
}

#define PACKET_STORE_BF(N)                                                    \
  inline void PacketStore(Eigen::bfloat16* ptr, int offset,                   \
                          const sycl::vec<float, N>& array) {                 \
    sycl::vec<uint16_t, N> tmp;                                               \
    for (int i = 0; i < N; ++i)                                               \
      tmp[i] =                                                                \
          Eigen::bfloat16_impl::float_to_bfloat16_rtne<true>(array[i]).value; \
    *(reinterpret_cast<sycl::vec<uint16_t, N>*>(ptr + offset)) = tmp;         \
  }
PACKET_STORE_BF(4)
PACKET_STORE_BF(8)
#undef PACKET_STORE_BF

template <typename InputT, typename OutputT, typename InitValueT,
          typename InputFunctor, typename OutputFunctor, typename BinaryOp>
struct SimpleColReduction {
  SimpleColReduction(InputT* in_data, OutputT* out_data, InitValueT init,
                     int extend_x, int extend_y, int extend_z,
                     InputFunctor in_func, OutputFunctor out_func, BinaryOp op)
      : in_data_(in_data),
        out_data_(out_data),
        init_(init),
        extend_x_(extend_x),
        extend_y_(extend_y),
        extend_z_(extend_z),
        in_func_(in_func),
        out_func_(out_func),
        op_(op) {}
  void operator()(sycl::nd_item<1> item) const {
    int id = item.get_global_linear_id();
    const int out_size = extend_x_ * extend_z_;
    if (id < out_size) {
      int outer = id / extend_z_;
      int inner = id - outer * extend_z_;

      int in_offset = outer * extend_y_ * extend_z_ + inner;
      InitValueT aggregate = InitValueT(in_func_(in_data_[in_offset]));
#pragma unroll
      for (int i = 1; i < extend_y_; ++i) {
        InitValueT tmp =
            InitValueT(in_func_(in_data_[in_offset + i * extend_z_]));
        aggregate = op_(aggregate, tmp);
      }
      out_data_[id] = OutputT(out_func_(aggregate));
    }
  }

  InputT* in_data_;
  OutputT* out_data_;
  InitValueT init_;
  int extend_x_;
  int extend_y_;
  int extend_z_;

  InputFunctor in_func_;
  OutputFunctor out_func_;
  BinaryOp op_;
};

inline void compute_tile(int* num_sg, int* num_row, const int extend_y,
                         const int tile_z, const int max_sub_group) {
  constexpr int minimum = 4;
  *num_sg = DivUp(*num_row * tile_z, ColReductionPolicy::SUB_GROUP_SIZE);
  int gap = *num_sg * ColReductionPolicy::SUB_GROUP_SIZE - *num_row * tile_z;
  while (*num_sg < max_sub_group && *num_row < extend_y && gap > minimum) {
    ++*num_row;
    *num_sg = DivUp(*num_row * tile_z, ColReductionPolicy::SUB_GROUP_SIZE);
    gap = *num_sg * ColReductionPolicy::SUB_GROUP_SIZE - *num_row * tile_z;
  }
}

inline void compute_tile(const int group_size, int* num_row, const int extend_y,
                         const int tile_z) {
  constexpr int minimum = 4;
  int gap = group_size - *num_row * tile_z;
  while (gap > minimum && *num_row < extend_y) {
    ++*num_row;
    gap = group_size - *num_row * tile_z;
  }
  if (gap < 0) --*num_row;
}

template <typename InT, typename OutT, typename InitValueT,
          typename LocalAccessor, typename InputFunctor, typename OutputFunctor,
          typename BinaryOp>
struct ColReductionKernel {
  ColReductionKernel(InT* in_data, OutT* out_data, const int extend_x,
                     const int extend_y, const int extend_z, InitValueT init,
                     const int tile_y, const int num_segments_y,
                     const int elems_per_thread, const int tile_z,
                     const int steps, BinaryOp op, LocalAccessor scratch,
                     InputFunctor in_func, OutputFunctor out_func)
      : in_data(in_data),
        out_data(out_data),
        extend_x(extend_x),
        extend_y(extend_y),
        extend_z(extend_z),
        init(init),
        tile_y(tile_y),
        num_segments_y(num_segments_y),
        elems_per_thread(elems_per_thread),
        tile_z(tile_z),
        steps(steps),
        op(op),
        scratch(scratch),
        in_func(in_func),
        out_func(out_func) {}
  void operator()(sycl::nd_item<3> item) const {
    int x_group_id = item.get_group(0);
    int y_group_id = item.get_group(1) * tile_y * elems_per_thread;
    int z_group_id = item.get_group(2) * tile_z;

    int local_id = item.get_local_linear_id();
    int local_y_id = local_id / tile_z;
    int local_z_id = local_id - tile_z * local_y_id;

    bool is_valid =
        (local_id < tile_y * tile_z) && (z_group_id + local_z_id < extend_z);
    int base_offset = x_group_id * extend_y * extend_z +
                      (y_group_id + local_y_id) * extend_z + z_group_id +
                      local_z_id;

    InitValueT aggregate = init;
    if (is_valid) {
      for (int i = 0; i < elems_per_thread; ++i) {
        int y_id = y_group_id + local_y_id + i * tile_y;
        if (y_id < extend_y)
          aggregate =
              op(aggregate, InitValueT(in_func(
                                in_data[base_offset + i * tile_y * extend_z])));
      }
      scratch[local_id] = aggregate;
    }
    item.barrier(cl::sycl::access::fence_space::local_space);

    int end = tile_y;
    int stride = (end + 2 - 1) / 2;
    for (int i = 0; i < steps; ++i) {
      if (is_valid && local_y_id < stride && local_y_id + stride < end) {
        scratch[local_id] =
            op(scratch[local_id], scratch[local_id + stride * tile_z]);
      }
      end = stride;
      stride = (end + 2 - 1) / 2;
      item.barrier(cl::sycl::access::fence_space::local_space);
    }

    if (is_valid && local_y_id == 0) {
      int offset = x_group_id * extend_z * num_segments_y +
                   item.get_group(1) * extend_z + z_group_id + local_z_id;
      out_data[offset] = OutT(out_func(scratch[local_z_id]));
    }
  }
  InT* in_data;
  OutT* out_data;
  const int extend_x;
  const int extend_y;
  const int extend_z;
  InitValueT init;
  const int tile_y;
  const int num_segments_y;
  const int elems_per_thread;
  const int tile_z;
  const int steps;
  BinaryOp op;
  LocalAccessor scratch;
  InputFunctor in_func;
  OutputFunctor out_func;
};

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, typename InputFunctor, typename OutputFunctor>
void TreeColReduction(OpKernelContext* ctx, InputT* in_data, OutputT* out_data,
                      int extend_x, int extend_y, int extend_z, InitValueT init,
                      BinaryOp op, InputFunctor in_func,
                      OutputFunctor out_func) {
  static constexpr int SubGroupSize = ColReductionPolicy::SUB_GROUP_SIZE;
  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();
  int max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  int tile_z = extend_z > 64 ? 16 : extend_z;
  int num_segments_z = DivUp(extend_z, tile_z);

  int num_sg = 1;
  int tile_y = 8;
  compute_tile(&num_sg, &tile_y, extend_y, tile_z,
               max_group_size / SubGroupSize);
  int steps = ceil_log2(tile_y);

  bool is_full_occu = (extend_x * num_segments_z) >
                      (64 * (max_group_size / (num_sg * SubGroupSize)));

  if (!is_full_occu) {
    int max_elems_per_thread =
        DivUp(num_segments_z * extend_x * (extend_y / tile_y),
              (4 * 64 * (max_group_size / (num_sg * SubGroupSize))));
    int min_elems_per_treahd =
        DivUp(DivUp(extend_y, tile_y) * tile_z, max_group_size * 16);
    int elems_per_thread = std::max(max_elems_per_thread, min_elems_per_treahd);

    int num_segments_y = DivUp(extend_y, tile_y * elems_per_thread);
    Tensor scratch_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<InitValueT>::value,
                            TensorShape({extend_x * num_segments_y * extend_z}),
                            &scratch_tensor));
    InitValueT* inter_out = scratch_tensor.flat<InitValueT>().data();

    sycl::range<3> local(1, 1, num_sg * SubGroupSize);
    sycl::range<3> global(extend_x, num_segments_y, num_segments_z * local[2]);
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<InitValueT> scratch(tile_y * tile_z, cgh);
      ColReductionKernel<InputT, InitValueT, InitValueT,
                         reduciton_helper::LocalAcc<InitValueT>, InputFunctor,
                         reduciton_helper::Identity<InitValueT>, BinaryOp>
          task(in_data, inter_out, extend_x, extend_y, extend_z, init, tile_y,
               num_segments_y, elems_per_thread, tile_z, steps, op, scratch,
               in_func, reduciton_helper::Identity<InitValueT>());
      cgh.parallel_for<ColReductionKernel<
          InputT, InitValueT, InitValueT,
          reduciton_helper::LocalAcc<InitValueT>, InputFunctor,
          reduciton_helper::Identity<InitValueT>, BinaryOp>>(
          sycl::nd_range<3>{global, local}, task);
    });
    compute_tile(max_group_size, &tile_y, num_segments_y, tile_z);

    steps = ceil_log2(tile_y);
    elems_per_thread = DivUp(num_segments_y, tile_y);
    local = sycl::range<3>{1, 1, static_cast<size_t>(max_group_size)};
    global = sycl::range<3>{static_cast<size_t>(extend_x), 1,
                            num_segments_z * local[2]};
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<InitValueT> scratch(tile_y * tile_z, cgh);
      ColReductionKernel<InitValueT, OutputT, InitValueT,
                         reduciton_helper::LocalAcc<InitValueT>,
                         reduciton_helper::Identity<InitValueT>, OutputFunctor,
                         BinaryOp>
          task(inter_out, out_data, extend_x, num_segments_y, extend_z, init,
               tile_y, 1, elems_per_thread, tile_z, steps, op, scratch,
               reduciton_helper::Identity<InitValueT>(), out_func);
      cgh.parallel_for<ColReductionKernel<
          InitValueT, OutputT, InitValueT,
          reduciton_helper::LocalAcc<InitValueT>,
          reduciton_helper::Identity<InitValueT>, OutputFunctor, BinaryOp>>(
          sycl::nd_range<3>{global, local}, task);
    });

    return;
  } else {
    sycl::range<3> local(1, 1, num_sg * SubGroupSize);
    sycl::range<3> global(extend_x, 1, num_segments_z * local[2]);
    int elems_per_thread = DivUp(extend_y, tile_y);

    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<InitValueT> scratch(tile_y * tile_z, cgh);
      ColReductionKernel<InputT, OutputT, InitValueT,
                         reduciton_helper::LocalAcc<InitValueT>, InputFunctor,
                         OutputFunctor, BinaryOp>
          task(in_data, out_data, extend_x, extend_y, extend_z, init, tile_y, 1,
               elems_per_thread, tile_z, steps, op, scratch, in_func, out_func);
      cgh.parallel_for<ColReductionKernel<
          InputT, OutputT, InitValueT, reduciton_helper::LocalAcc<InitValueT>,
          InputFunctor, OutputFunctor, BinaryOp>>(
          sycl::nd_range<3>{global, local}, task);
    });
    return;
  }
}

template <typename InT, typename OutT, typename InitValueT,
          typename LocalAccessor, typename InputFunctor, typename OutputFunctor,
          typename BinaryOp>
struct SGVecColReductionKernel {
  SGVecColReductionKernel(const InT* in_data_, OutT* out_,
                          LocalAccessor scratch_, const int extend_x_,
                          const int extend_y_, const int extend_z_,
                          const int elems_per_item_, const int num_sub_group_,
                          const int num_segments_y_, const int k_,
                          const int local_item_on_z_, InputFunctor in_func_,
                          OutputFunctor out_func_, BinaryOp op_)
      : in_data(in_data_),
        out(out_),
        scratch(scratch_),
        extend_x(extend_x_),
        extend_y(extend_y_),
        extend_z(extend_z_),
        elems_per_item(elems_per_item_),
        num_sub_group(num_sub_group_),
        num_segments_y(num_segments_y_),
        k(k_),
        local_item_on_z(local_item_on_z_),
        in_func(in_func_),
        out_func(out_func_),
        op(op_) {}
  [[intel::reqd_sub_group_size(ColReductionPolicy::SUB_GROUP_SIZE)]] void
  operator()(sycl::nd_item<3> item) const {
    constexpr int VEC_SIZE = 4 * sizeof(float) / sizeof(InT);
    typedef sycl::vec<InitValueT, VEC_SIZE> vecT;

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
    vecT aggregate(0.0f);

    int z_offset =
        z_group_id * local_item_on_z * VEC_SIZE + group_z_id * VEC_SIZE;

    for (int i = 0; i < elems_per_item; ++i) {
      int y_idx = y_group_id * num_sub_group * elems_per_item * k +
                  subgroup_id * elems_per_item * k + group_k_id + i * k;
      if (y_idx >= extend_y) break;
      int offset = x_offset + y_idx * extend_z + z_offset;

      vecT tmp;
      PacketLoad(in_data, offset, &tmp);

      for (int j = 0; j < VEC_SIZE; ++j) {
        aggregate[j] = op(aggregate[j], tmp[j]);
      }
    }
    // each subgroup write result to slm
    scratch[subgroup_id * k + group_k_id + group_z_id * num_sub_group * k] =
        aggregate;
    item.barrier(sycl::access::fence_space::local_space);

    // ------------------------------------------------------------------
    // -------------slm reduce-------------------
    // slm: (ColReductionPolicy::SUB_GROUP_SIZE * k) * local_item_on_z
    // ------------------------------------------------------------------
    int slm_z_id = subgroup_id / k;
    int slm_k_id = subgroup_id % k;
    vecT value =
        scratch[slm_z_id * num_sub_group * k +
                slm_k_id * ColReductionPolicy::SUB_GROUP_SIZE + lane_id];

    // reduce within each subgroup
    for (int i = 0; i < VEC_SIZE; ++i) {
      value[i] = sycl::reduce_over_group(sg, value[i], op);
    }

    // lane0 write result of each subgrop
    if (lane_id == 0) {
      scratch[slm_z_id * num_sub_group * k +
              slm_k_id * ColReductionPolicy::SUB_GROUP_SIZE] = value;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // collect result of k subgroup and store output
    if (lane_id == 0 && slm_k_id == 0) {
      vecT tmp = scratch[slm_z_id * k * num_sub_group];
      for (int i = 1; i < k; ++i) {
        for (int j = 0; j < VEC_SIZE; ++j) {
          tmp[j] =
              op(tmp[j], scratch[slm_z_id * k * num_sub_group +
                                 i * ColReductionPolicy::SUB_GROUP_SIZE][j]);
        }
      }
      int offset =
          x_group_id * extend_z * num_segments_y + y_group_id * extend_z +
          z_group_id * local_item_on_z * VEC_SIZE + slm_z_id * VEC_SIZE;

      PacketStore(out, offset, tmp);
    }
  }
  const InT* in_data;
  OutT* out;
  LocalAccessor scratch;
  const int extend_x;
  const int extend_y;
  const int extend_z;
  const int elems_per_item;
  const int num_sub_group;
  const int num_segments_y;
  const int k;
  const int local_item_on_z;
  InputFunctor in_func;
  OutputFunctor out_func;
  BinaryOp op;
};

template <typename InputT, typename OutputT, typename InitValueT,
          typename BinaryOp, typename InputFunctor, typename OutputFunctor>
void SGVecColReduction(OpKernelContext* ctx, InputT* in_data, OutputT* out_data,
                       int extend_x, int extend_y, int extend_z,
                       InitValueT init, BinaryOp op, InputFunctor in_func,
                       OutputFunctor out_func) {
  constexpr int VEC_SIZE = 4 * sizeof(float) / sizeof(InputT);
  typedef sycl::vec<InitValueT, VEC_SIZE> vecT;
  typedef typename std::remove_cv<
      typename std::remove_reference<InputT>::type>::type BaseInputT;

  BaseInputT* in_unqualified = const_cast<BaseInputT*>(in_data);

  int elems_per_item = ColReductionPolicy::ITEM_PER_THREAD;
  int num_sub_group = ColReductionPolicy::NUM_SUB_GROUP;
  int work_item_on_z = extend_z / VEC_SIZE;
  int local_item_on_z =
      work_item_on_z <= ColReductionPolicy::MAX_LOCAL_ITEM_ON_Z
          ? work_item_on_z
          : ColReductionPolicy::MAX_LOCAL_ITEM_ON_Z;
  int k = ColReductionPolicy::SUB_GROUP_SIZE / local_item_on_z;

  if (extend_y * 2 <= num_sub_group * elems_per_item * k) {
    while (num_sub_group * elems_per_item * k >= extend_y * 2 &&
           elems_per_item > 1) {
      elems_per_item >>= 1;
    }
  }

  int num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item * k);
  int num_segments_z = work_item_on_z / local_item_on_z;

  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();

  if (num_segments_y > 1) {
    while (num_segments_y >
           num_sub_group * ColReductionPolicy::ITEM_PER_THREAD * k) {
      elems_per_item <<= 1;
      num_segments_y = DivUp(extend_y, num_sub_group * elems_per_item * k);
    }

    sycl::range<3> local(1, num_sub_group * k, local_item_on_z);
    sycl::range<3> global(extend_x, num_segments_y * local[1],
                          num_segments_z * local[2]);

    Tensor scratch_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<InitValueT>::value,
                            TensorShape({extend_x * num_segments_y * extend_z}),
                            &scratch_tensor));
    InitValueT* inter_out = scratch_tensor.flat<InitValueT>().data();

    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<vecT> scratch(
          num_sub_group * ColReductionPolicy::SUB_GROUP_SIZE, cgh);
      SGVecColReductionKernel<BaseInputT, InitValueT, InitValueT,
                              reduciton_helper::LocalAcc<vecT>, InputFunctor,
                              reduciton_helper::Identity<InitValueT>, BinaryOp>
          task(in_unqualified, inter_out, scratch, extend_x, extend_y, extend_z,
               elems_per_item, num_sub_group, num_segments_y, k,
               local_item_on_z, in_func,
               reduciton_helper::Identity<InitValueT>(), op);
      cgh.parallel_for<SGVecColReductionKernel<
          BaseInputT, InitValueT, InitValueT, reduciton_helper::LocalAcc<vecT>,
          InputFunctor, reduciton_helper::Identity<InitValueT>, BinaryOp>>(
          sycl::nd_range<3>(global, local), task);
    });

    global = sycl::range<3>{static_cast<size_t>(extend_x), local[1],
                            num_segments_z * local[2]};

    typedef sycl::vec<InitValueT, 4 * sizeof(float) / sizeof(InitValueT)> vecT2;
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<vecT2> scratch(
          num_sub_group * ColReductionPolicy::SUB_GROUP_SIZE, cgh);
      SGVecColReductionKernel<
          InitValueT, OutputT, InitValueT, reduciton_helper::LocalAcc<vecT2>,
          reduciton_helper::Identity<InitValueT>, OutputFunctor, BinaryOp>
          task(inter_out, out_data, scratch, extend_x, num_segments_y, extend_z,
               ColReductionPolicy::ITEM_PER_THREAD, num_sub_group, 1, k,
               local_item_on_z, reduciton_helper::Identity<InitValueT>(),
               out_func, op);
      cgh.parallel_for<SGVecColReductionKernel<
          InitValueT, OutputT, InitValueT, reduciton_helper::LocalAcc<vecT2>,
          reduciton_helper::Identity<InitValueT>, OutputFunctor, BinaryOp>>(
          sycl::nd_range<3>(global, local), task);
    });
    return;
  } else {
    sycl::range<3> local(1, num_sub_group * k, local_item_on_z);
    sycl::range<3> global(extend_x, local[1], num_segments_z * local[2]);
    stream->submit([&](sycl::handler& cgh) {
      reduciton_helper::LocalAcc<vecT> scratch(
          num_sub_group * ColReductionPolicy::SUB_GROUP_SIZE, cgh);
      SGVecColReductionKernel<BaseInputT, OutputT, InitValueT,
                              reduciton_helper::LocalAcc<vecT>, InputFunctor,
                              OutputFunctor, BinaryOp>
          task(in_unqualified, out_data, scratch, extend_x, extend_y, extend_z,
               elems_per_item, num_sub_group, 1, k, local_item_on_z, in_func,
               out_func, op);
      cgh.parallel_for<SGVecColReductionKernel<
          BaseInputT, OutputT, InitValueT, reduciton_helper::LocalAcc<vecT>,
          InputFunctor, OutputFunctor, BinaryOp>>(
          sycl::nd_range<3>(global, local), task);
    });
    return;
  }
}

template <typename InputT, typename OutputT, typename InitValueT, typename Op,
          typename InputFunctor = reduciton_helper::Identity<InputT>,
          typename OutputFunctor = reduciton_helper::Identity<InitValueT>>
void LaunchColReduction(
    OpKernelContext* ctx, InputT* in_data, OutputT* out_data,
    InitValueT init_val, int extend_x, int extend_y, int extend_z, Op op,
    InputFunctor in_func = reduciton_helper::Identity<InputT>(),
    OutputFunctor out_func = reduciton_helper::Identity<InitValueT>()) {
  const auto& d = ctx->eigen_gpu_device();
  auto stream = d.stream();
  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();

  int elems_per_item = extend_y / (extend_x * extend_z);

  // maximum_work_item = EU * Thread_Per_EU * native_SIMD_width
  // choose native_SIMD_width rather than max is for safety consideration
  auto dev = stream->get_device();
  const int hardware_reside_work_item =
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>() *
      dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>() *
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
  constexpr int VEC_SIZE = 4 * sizeof(float) / sizeof(InputT);

  bool reach_minimum_occpu =
      (extend_x * extend_y * extend_z / VEC_SIZE) >= hardware_reside_work_item;
  bool reach_vec_alignment =
      (extend_z % VEC_SIZE == 0) &&
      ((ColReductionPolicy::SUB_GROUP_SIZE % (extend_z / VEC_SIZE) == 0) ||
       ((extend_z / VEC_SIZE) % ColReductionPolicy::MAX_LOCAL_ITEM_ON_Z == 0));
  bool use_vectorization_pass = reach_minimum_occpu && reach_vec_alignment;

  if (use_vectorization_pass) {
    SGVecColReduction<InputT, OutputT, InitValueT, Op, InputFunctor,
                      OutputFunctor>(ctx, in_data, out_data, extend_x, extend_y,
                                     extend_z, init_val, op, in_func, out_func);
  } else if (elems_per_item < 4 && extend_y < 64) {
    const int out_size = extend_x * extend_z;
    int GroupSize = std::min(512, max_group_size);
    int num_wg = (out_size + GroupSize - 1) / GroupSize;
    sycl::nd_range<1> range(num_wg * GroupSize, GroupSize);

    stream->submit([&](sycl::handler& cgh) {
      SimpleColReduction<InputT, OutputT, InitValueT, InputFunctor,
                         OutputFunctor, Op>
          task(in_data, out_data, init_val, extend_x, extend_y, extend_z,
               in_func, out_func, op);
      cgh.parallel_for<SimpleColReduction<InputT, OutputT, InitValueT,
                                          InputFunctor, OutputFunctor, Op>>(
          range, task);
    });
  } else {
    TreeColReduction<InputT, OutputT, InitValueT, Op, InputFunctor,
                     OutputFunctor>(ctx, in_data, out_data, extend_x, extend_y,
                                    extend_z, init_val, op, in_func, out_func);
  }
}
}  // namespace itex
#endif  // ITEX_CORE_KERNELS_GPU_COL_REDUCTION_KERNELS_H_
