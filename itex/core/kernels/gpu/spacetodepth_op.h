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

#ifndef ITEX_CORE_KERNELS_GPU_SPACETODEPTH_OP_H_
#define ITEX_CORE_KERNELS_GPU_SPACETODEPTH_OP_H_

#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;

// Space2Depth kernel for FORMAT_NHWC.
// See 'spacetodepth_op.h' for a more detailed description.
template <typename dtype>
struct S2D_NHWC_KERNEL {
  S2D_NHWC_KERNEL(const int32_t nthreads, const dtype* input_ptr,
                  const int block_size, const int batch_size,
                  const int input_height, const int input_width,
                  const int input_depth, const int output_height,
                  const int output_width, const int output_depth,
                  dtype* output_ptr)
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
    auto inp_idx = item.get_global_linear_id();
    if (inp_idx >= nthreads) return;
    // inp_idx = d + input_depth * (w + input_width * (h + input_height * b))
    const int d = inp_idx % input_depth;
    const int inp_idx2 = inp_idx / input_depth;
    const int w = inp_idx2 % input_width;
    const int inp_idx3 = inp_idx2 / input_width;
    const int h = inp_idx3 % input_height;
    const int b = inp_idx3 / input_height;

    const int out_h = h / block_size;
    const int offset_h = h % block_size;
    const int out_w = w / block_size;
    const int offset_w = w % block_size;
    const int offset_d = (offset_h * block_size + offset_w) * input_depth;
    const int out_d = d + offset_d;
    const int out_idx =
        out_d +
        output_depth * (out_w + output_width * (out_h + output_height * b));
    *(output_ptr + out_idx) = *(input_ptr + inp_idx);
  }

 private:
  int32_t nthreads;
  const dtype* input_ptr;
  int block_size;
  int batch_size;
  int input_height;
  int input_width;
  int input_depth;
  int output_height;
  int output_width;
  int output_depth;
  dtype* output_ptr;
};

// Space2Depth kernel for FORMAT_NCHW.
// See 'spacetodepth_op.h' for a more detailed description.
template <typename dtype>
struct S2D_NCHW_KERNEL {
  S2D_NCHW_KERNEL(const int32 nthreads, const dtype* input_ptr,
                  const int block_size, const int output_width,
                  const int input_depth_by_output_height, dtype* output_ptr)
      : nthreads(nthreads),
        input_ptr(input_ptr),
        block_size(block_size),
        output_width(output_width),
        input_depth_by_output_height(input_depth_by_output_height),
        output_ptr(output_ptr) {}
  void operator()(sycl::nd_item<1> item) const {
    auto input_idx = item.get_global_linear_id();
    if (input_idx >= nthreads) return;
    // We assume both the input and output are packed NCHW tensors.
    // input_idx represents an index within the flattened input tensor.
    // We can consider the block width and height as extra tensor dimensions,
    // then isolate the relevant components of input_idx and recombine them to
    // form output_idx. The layout transform performed is:
    // n, iC, oY, bY, oX, bX    (== input_idx)   to
    // n, bY, bX, iC, oY, oX    (== output_idx).

    const int n_iC_oY_bY_oX = input_idx / block_size;
    const int bX = input_idx - n_iC_oY_bY_oX * block_size;

    const int n_iC_oY_bY = n_iC_oY_bY_oX / output_width;
    const int oX = n_iC_oY_bY_oX - n_iC_oY_bY * output_width;

    const int n_iC_oY = n_iC_oY_bY / block_size;
    const int bY = n_iC_oY_bY - n_iC_oY * block_size;

    const int n = n_iC_oY / input_depth_by_output_height;
    const int iC_oY = n_iC_oY - n * input_depth_by_output_height;

    const int output_idx = oX + (((n * block_size + bY) * block_size + bX) *
                                     input_depth_by_output_height +
                                 iC_oY) *
                                    output_width;

    *(output_ptr + output_idx) = *(input_ptr + input_idx);
  }

 private:
  int32 nthreads;
  const dtype* input_ptr;
  int block_size;
  int output_width;
  int input_depth_by_output_height;
  dtype* output_ptr;
};

// Space2Depth kernel for FORMAT_NCHW using a loop over block area.
// See 'spacetodepth_op.h' for functional specification.
template <typename dtype, int block_size>
struct S2D_NCHW_LOOP_KERNEL {
  S2D_NCHW_LOOP_KERNEL(const int32 nthreads, const dtype* input,
                       const int output_width, const int input_width,
                       const int input_depth_by_output_area,
                       const int output_depth_by_output_area, dtype* output)
      : nthreads(nthreads),
        input(input),
        output_width(output_width),
        input_width(input_width),
        input_depth_by_output_area(input_depth_by_output_area),
        output_depth_by_output_area(output_depth_by_output_area),
        output(output) {}
  void operator()(sycl::nd_item<1> item) const {
    auto thread_idx = item.get_global_linear_id();
    if (thread_idx >= nthreads) return;
    // We will be converting the image from ordering:
    // n, iC, oY, bY, oX, bX   (== input index) to
    // n, bY, bX, iC, oY, oX   (== output index)

    // We assume thread_idx encodes n_iC_oY_oX, and use an unrolled loop over
    // bY and bX coordinates within the block. This kernel gets a small
    // performance improvement compared with S2D_NCHW due to a denser access
    // pattern on the input side. (Note: the equivalent D2S kernel gets a larger
    // improvement as a denser pattern on the output side makes more
    // difference).

    const int n_iC_oY = thread_idx / output_width;
    const int oX = thread_idx - n_iC_oY * output_width;
    const int n = thread_idx / input_depth_by_output_area;
    const int iC_oY_oX = thread_idx - n * input_depth_by_output_area;

    // Recombine the components and apply to the input and output pointers.
    auto input_ptr = input + (n_iC_oY * input_width + oX) * block_size;
    auto output_ptr = output + n * output_depth_by_output_area + iC_oY_oX;

#pragma unroll
    // Copy a patch of data to the output batch image.
    for (int bY = 0; bY < block_size; ++bY) {
#pragma unroll
      for (int bX = 0; bX < block_size; ++bX) {
        output_ptr[(bY * block_size + bX) * input_depth_by_output_area] =
            *(input_ptr + bY * input_width + bX);
      }
    }
  }

 private:
  int32 nthreads;
  const dtype* input;
  int output_width;
  int input_width;
  int input_depth_by_output_area;
  int output_depth_by_output_area;
  dtype* output;
};

namespace functor {
// Functor used by SpaceToDepthOp to do the computations.
// Implements a family of Space to Depth transforms for a 4D 'input' tensor
// to a 4D 'output' tensor, both tensors use type 'T' and layout 'data_format'.
// These transforms divide the vertical and horizontal image sizes by
// 'block_size', and multiply the depth dimension size by
// (block_size * block_size). The offset within each block_size * block_size
// patch within the image is combined with the input channel index to form
// the output channel index, with the Y, X coordinates within each block of
// the input image used as the high order component of the output channel.
// e.g. for data_format = NHWC:
//      Each element in the input tensor can be specified via 6 coordinates,
//      ordered by decreasing memory layout significance as:
//      n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
//                         within the output image, bX, bY means coordinates
//                         within the input block, iC means input channels).
//      The output would be a transpose to the following layout:
//      n,oY,oX,bY,bX,iC
template <typename Device, typename T, TensorFormat data_format>
struct SpaceToDepthOpFunctor {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output);

  // This 5-D version is to support NCHW_VECT_C.
  void operator()(const Device& d, typename TTypes<T, 5>::ConstTensor input,
                  int block_size, typename TTypes<T, 5>::Tensor output);
};

template <typename T>
struct SpaceToDepthOpFunctor<GPUDevice, T, FORMAT_NHWC> {
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
        batch_size * input_height * input_width * input_depth;
    if (total_count == 0) {
      return;
    }

    // GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);
    auto stream = d.stream();
    auto max_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    const int workgroup_count =
        (total_count + max_group_size - 1) / max_group_size;
    sycl::range<1> global(workgroup_count * max_group_size);
    sycl::range<1> local(max_group_size);

    stream->submit([&](sycl::handler& cgh) {
      S2D_NHWC_KERNEL<T> task(total_count, input.data(), block_size, batch_size,
                              input_height, input_width, input_depth,
                              output_height, output_width, output_depth,
                              output.data());
      cgh.parallel_for<S2D_NHWC_KERNEL<T>>(sycl::nd_range<1>(global, local),
                                           task);
    });
  }

  void operator()(const GPUDevice& d, typename TTypes<T, 5>::ConstTensor input,
                  int block_size, typename TTypes<T, 5>::Tensor output) {
    ITEX_LOG(FATAL) << "5-D tensors should not be used with NHWC format";
  }
};

template <typename T>
struct SpaceToDepthOpFunctor<GPUDevice, T, FORMAT_NCHW> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int input_depth = input.dimension(1);
    const int output_depth = output.dimension(1);
    const int output_height = output.dimension(2);
    const int output_width = output.dimension(3);
    const int output_area = output_width * output_height;
    const int output_depth_by_output_area = output_depth * output_area;

    auto stream = d.stream();
    auto max_group_size =
        (*stream)
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();

    // We improve performance by generating instantiations of the loop kernel
    // for the most common block sizes.
    if (block_size <= 4) {
      const int input_width = input.dimension(3);
      const int input_depth_by_output_area = input_depth * output_area;
      const int total_count = batch_size * input_depth_by_output_area;
      if (total_count == 0) {
        return;
      }
      const int workgroup_count =
          (total_count + max_group_size - 1) / max_group_size;
      sycl::range<1> global(workgroup_count * max_group_size);
      sycl::range<1> local(max_group_size);

      switch (block_size) {
        case 2:
          stream->submit([&](sycl::handler& cgh) {
            S2D_NCHW_LOOP_KERNEL<T, 2> task(
                total_count, input.data(), output_width, input_width,
                input_depth_by_output_area, output_depth_by_output_area,
                output.data());
            cgh.parallel_for<S2D_NCHW_LOOP_KERNEL<T, 2>>(
                sycl::nd_range<1>(global, local), task);
          });
          return;
        case 3:
          stream->submit([&](sycl::handler& cgh) {
            S2D_NCHW_LOOP_KERNEL<T, 3> task(
                total_count, input.data(), output_width, input_width,
                input_depth_by_output_area, output_depth_by_output_area,
                output.data());
            cgh.parallel_for<S2D_NCHW_LOOP_KERNEL<T, 3>>(
                sycl::nd_range<1>(global, local), task);
          });
          return;
        case 4:
          stream->submit([&](sycl::handler& cgh) {
            S2D_NCHW_LOOP_KERNEL<T, 4> task(
                total_count, input.data(), output_width, input_width,
                input_depth_by_output_area, output_depth_by_output_area,
                output.data());

            cgh.parallel_for<S2D_NCHW_LOOP_KERNEL<T, 4>>(
                sycl::nd_range<1>(global, local), task);
          });
          return;
      }
    }

    // Other block sizes are processed by the generic kernel.
    const int total_count = batch_size * output_depth_by_output_area;
    if (total_count == 0) {
      return;
    }
    const int workgroup_count =
        (total_count + max_group_size - 1) / max_group_size;
    sycl::range<1> global(workgroup_count * max_group_size);
    sycl::range<1> local(max_group_size);

    stream->submit([&](sycl::handler& cgh) {
      S2D_NCHW_KERNEL<T> task(total_count, input.data(), block_size,
                              output_width, input_depth * output_height,
                              output.data());
      cgh.parallel_for<S2D_NCHW_KERNEL<T>>(sycl::nd_range<1>(global, local),
                                           task);
    });
    return;
  }
  void operator()(const GPUDevice& d, typename TTypes<T, 5>::ConstTensor input,
                  int block_size, typename TTypes<T, 5>::Tensor output) {
    ITEX_LOG(FATAL) << "5-D tensors should not be used with NCHW format";
  }
};

}  // namespace functor
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_SPACETODEPTH_OP_H_
