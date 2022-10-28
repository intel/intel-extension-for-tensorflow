/* Copyright (c) 2021-2022 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/kernels/gpu/depthtospace_op.h"
#include "itex/core/utils/register_types.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace {

using GPUDevice = Eigen::GpuDevice;

// Depth2Space kernel for FORMAT_NHWC.
// See 'depthtospace_op.h' for a more detailed description.
template <typename dtype>
struct D2S_NHWC {
  D2S_NHWC(const int32 nthreads, const dtype* input_ptr, const int block_size,
           const int batch_size, const int input_height, const int input_width,
           const int input_depth, const int output_height,
           const int output_width, const int output_depth, dtype* output_ptr)
      : nthreads(nthreads),
        input_ptr(input_ptr),
        block_size(block_size),
        batch_size(batch_size),
        input_height(input_height),
        input_width(input_width),
        input_depth(input_depth),
        output_height(output_height),
        output_width(output_width),
        output_depth(output_depth),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    const auto out_idx = item.get_global_linear_id();
    if (out_idx >= nthreads) return;
    // out_idx = d + output_depth * (w + output_width * (h + output_height * b))
    const int d = out_idx % output_depth;
    const int out_idx2 = out_idx / output_depth;
    const int w = out_idx2 % output_width;
    const int out_idx3 = out_idx2 / output_width;
    const int h = out_idx3 % output_height;
    const int b = out_idx3 / output_height;

    const int in_h = h / block_size;
    const int offset_h = h % block_size;
    const int in_w = w / block_size;
    const int offset_w = w % block_size;
    const int offset_d = (offset_h * block_size + offset_w) * output_depth;
    const int in_d = d + offset_d;
    const int inp_idx =
        in_d + input_depth * (in_w + input_width * (in_h + input_height * b));
    *(output_ptr + out_idx) = input_ptr[inp_idx];
  }

 private:
  const int32 nthreads;
  const dtype* input_ptr;
  const int block_size;
  const int batch_size;
  const int input_height;
  const int input_width;
  const int input_depth;
  const int output_height;
  const int output_width;
  const int output_depth;
  dtype* output_ptr;
};

// Depth2Space kernel for FORMAT_NCHW.
// See 'spacetodepth_op.h' for a more detailed description.
template <typename dtype>
struct D2S_NCHW {
  D2S_NCHW(const int32 nthreads, const dtype* input_ptr, const int block_size,
           const int input_width, const int output_depth_by_input_height,
           dtype* output_ptr)
      : nthreads(nthreads),
        input_ptr(input_ptr),
        block_size(block_size),
        input_width(input_width),
        output_depth_by_input_height(output_depth_by_input_height),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto input_idx = item.get_global_linear_id();
    if (input_idx >= nthreads) return;
    // We will be converting the image from ordering:
    // n, bY, bX, oC, iY, iX    (== input_idx)   to
    // n, oC, iY, bY, iX, bX

    // Start reading the input data straight away since we know the address.
    // We calculate the output address in parallel while this is being fetched.

    const int n_bY_bX_oC_iY = input_idx / input_width;
    const int iX = input_idx - n_bY_bX_oC_iY * input_width;

    const int n_bY_bX = n_bY_bX_oC_iY / output_depth_by_input_height;
    const int oC_iY = n_bY_bX_oC_iY - n_bY_bX * output_depth_by_input_height;

    const int n_bY = n_bY_bX / block_size;
    const int bX = n_bY_bX - n_bY * block_size;

    const int n = n_bY / block_size;
    const int bY = n_bY - n * block_size;

    const int output_idx =
        bX +
        block_size *
            (iX + input_width *
                      (bY + block_size *
                                (oC_iY + n * output_depth_by_input_height)));

    *(output_ptr + output_idx) = input_ptr[input_idx];
  }

 private:
  const int32 nthreads;
  const dtype* input_ptr;
  const int block_size;
  const int input_width;
  const int output_depth_by_input_height;
  dtype* output_ptr;
};

template <typename dtype, int block_size>
struct D2S_NCHW_LOOP {
  D2S_NCHW_LOOP(const int32 nthreads, const dtype* input, const int input_width,
                const int output_width, const int output_depth_by_input_area,
                const int input_depth_by_input_area, dtype* output)
      : nthreads(nthreads),
        input(input),
        input_width(input_width),
        output_width(output_width),
        output_depth_by_input_area(output_depth_by_input_area),
        input_depth_by_input_area(input_depth_by_input_area),
        output(output) {}
  void operator()(sycl::nd_item<1> item) const {
    const auto thread_idx = item.get_global_linear_id();
    if (thread_idx >= nthreads) return;
    // We will be converting the image from ordering:
    // n, bY, bX, oC, iY, iX   to
    // n, oC, iY, bY, iX, bX

    // We assume thread_idx encodes n_oC_iY_iX, and use an unrolled loop over
    // bY and bX coordinates within the block. This kernel is significantly
    // more performant than the D2S_NCHW kernel.
    //   A likely explanation of the improvement is that although both kernels
    // get input coalescing, this one would write the output data more densely
    // per warp, so would benefit assuming delayed cache writeback is used.

    const int n_oC_iY = thread_idx / input_width;
    const int iX = thread_idx - n_oC_iY * input_width;

    const int n = thread_idx / output_depth_by_input_area;
    const int oC_iY_iX = thread_idx - n * output_depth_by_input_area;

    // Recombine the components and apply to the input and output pointers.
    auto input_ptr = input + n * input_depth_by_input_area + oC_iY_iX;
    auto output_ptr = output + (n_oC_iY * output_width + iX) * block_size;

#pragma unroll
    // Copy a patch of data to the output batch image.
    for (int bY = 0; bY < block_size; ++bY) {
#pragma unroll
      for (int bX = 0; bX < block_size; ++bX) {
        output_ptr[bY * output_width + bX] =
            input_ptr[(bY * block_size + bX) * output_depth_by_input_area];
      }
    }
  }

 private:
  const int32 nthreads;
  const dtype* input;
  const int input_width;
  const int output_width;
  const int output_depth_by_input_area;
  const int input_depth_by_input_area;
  dtype* output;
};

}  // namespace

// Specialization of DepthToSpaceOpFunctor for a GPUDevice.
namespace functor {

template <typename T>
struct DepthToSpaceOpFunctor<GPUDevice, T, FORMAT_NHWC> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int input_height = input.dimension(1);
    const int input_width = input.dimension(2);
    const int input_depth = input.dimension(3);
    const int output_height = output.dimension(1);
    const int output_width = output.dimension(2);
    const int output_depth = output.dimension(3);

    const int total_count =
        batch_size * output_height * output_width * output_depth;
    if (total_count == 0) {
      return;
    }

    const auto stream = d.stream();

    const auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    const auto num_workgroup = (total_count + group_size - 1) / group_size;
    auto event = stream->submit([&](sycl::handler& cgh) {
      D2S_NHWC<T> task(total_count, input.data(), block_size, batch_size,
                       input_height, input_width, input_depth, output_height,
                       output_width, output_depth, output.data());
      cgh.parallel_for<D2S_NHWC<T>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          task);
    });
  }
};

template <typename T>
struct DepthToSpaceOpFunctor<GPUDevice, T, FORMAT_NCHW> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int input_depth = input.dimension(1);
    const int input_height = input.dimension(2);
    const int input_width = input.dimension(3);
    const int output_depth = output.dimension(1);
    const int input_area = input_width * input_height;
    const int input_depth_by_input_area = input_depth * input_area;

    const auto stream = d.stream();

    const auto group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    // We improve performance by generating instantiations of the loop kernel
    // for the most common block sizes.
    if (block_size <= 4) {
      const int output_width = output.dimension(3);
      const int output_depth_by_input_area = output_depth * input_area;
      const int total_count = batch_size * output_depth_by_input_area;
      if (total_count == 0) {
        return;
      }
      const auto num_workgroup = (total_count + group_size - 1) / group_size;
      switch (block_size) {
        case 2:
          stream->submit([&](sycl::handler& cgh) {
            D2S_NCHW_LOOP<T, 2> task(total_count, input.data(), input_width,
                                     output_width, output_depth_by_input_area,
                                     input_depth_by_input_area, output.data());
            cgh.parallel_for<D2S_NCHW_LOOP<T, 2>>(
                sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                                  sycl::range<1>(group_size)),
                task);
          });
          return;
        case 3:
          stream->submit([&](sycl::handler& cgh) {
            D2S_NCHW_LOOP<T, 3> task(total_count, input.data(), input_width,
                                     output_width, output_depth_by_input_area,
                                     input_depth_by_input_area, output.data());
            cgh.parallel_for<D2S_NCHW_LOOP<T, 3>>(
                sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                                  sycl::range<1>(group_size)),
                task);
          });
          return;
        case 4:
          stream->submit([&](sycl::handler& cgh) {
            D2S_NCHW_LOOP<T, 4> task(total_count, input.data(), input_width,
                                     output_width, output_depth_by_input_area,
                                     input_depth_by_input_area, output.data());
            cgh.parallel_for<D2S_NCHW_LOOP<T, 4>>(
                sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                                  sycl::range<1>(group_size)),
                task);
          });
          return;
      }
    }

    // Other block sizes are processed by the generic kernel.
    const int total_count = batch_size * input_depth_by_input_area;
    if (total_count == 0) {
      return;
    }
    const auto num_workgroup = (total_count + group_size - 1) / group_size;

    stream->submit([&](sycl::handler& cgh) {
      D2S_NCHW<T> task(total_count, input.data(), block_size, input_width,
                       output_depth * input_height, output.data());
      cgh.parallel_for<D2S_NCHW<T>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          task);
    });
  }
};
}  // end namespace functor

// Instantiate the GPU implementations for float.
#define INSTANTIATE_D2S_FUNCTOR(type)                             \
  template struct functor::DepthToSpaceOpFunctor<GPUDevice, type, \
                                                 FORMAT_NCHW>;    \
  template struct functor::DepthToSpaceOpFunctor<GPUDevice, type, FORMAT_NHWC>;

TF_CALL_GPU_NUMBER_TYPES(INSTANTIATE_D2S_FUNCTOR);

#undef INSTANTIATE_D2S_FUNCTOR

}  // end namespace itex
