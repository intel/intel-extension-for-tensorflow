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

#include <algorithm>
#include <vector>

#include "itex/core/kernels/gpu/image/non_max_suppression_op.h"
#include "itex/core/kernels/gpu/topk_op.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/numeric_types.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_types.h"
#include "itex/core/utils/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {
typedef Eigen::GpuDevice GPUDevice;
typedef sycl::vec<float, 2> float2;
typedef sycl::vec<float, 3> float3;
typedef sycl::vec<float, 4> float4;
template <typename T>
using LocalMem = sycl::local_accessor<T, 1>;
constexpr int kNmsBoxesPerWorkItem = 8 * sizeof(int);
constexpr int kNmsGroupDim = 16;

namespace {

struct GeneratePreNMSUprightBoxesKernelTask {
  GeneratePreNMSUprightBoxesKernelTask(
      int nboxes_to_generate, int num_images, const int* d_sorted_scores_keys,
      const float4* d_bbox_deltas, const float4* d_anchors, int height,
      int width, int num_anchors, float min_size, const float* d_img_info_vec,
      float bbox_xform_clip, float4* d_out_boxes, int prenms_nboxes,
      char* d_boxes_keep_flags)
      : nboxes_to_generate(nboxes_to_generate),
        num_images(num_images),
        d_sorted_scores_keys(d_sorted_scores_keys),
        d_bbox_deltas(d_bbox_deltas),
        d_anchors(d_anchors),
        height(height),
        width(width),
        num_anchors(num_anchors),
        min_size(min_size),
        d_img_info_vec(d_img_info_vec),
        bbox_xform_clip(bbox_xform_clip),
        d_out_boxes(d_out_boxes),
        prenms_nboxes(prenms_nboxes),
        d_boxes_keep_flags(d_boxes_keep_flags) {}
  void operator()(sycl::nd_item<1> item) const {
    const int idx = item.get_global_linear_id();
    if (idx >= nboxes_to_generate * num_images) {
      return;
    }
    const int anchor_stride = height * width;              // Stride of Anchor
    const int height_stride = width * num_anchors;         // Stride of height
    const int image_stride = anchor_stride * num_anchors;  // Stride of image

    const int image_index = idx / nboxes_to_generate;
    const int ibox = idx % nboxes_to_generate;
    // box_conv_index : # of the same box, but indexed in the
    // scores from the conv layer, of shape (height,width,num_anchors) the
    // num_images dimension was already removed box_conv_index =
    // a*image_stride + h*width + w
    const int box_conv_index =
        d_sorted_scores_keys[image_index * image_stride + ibox];

    // We want to decompose box_conv_index in (h,w,a)
    // such as box_conv_index = h*width*num_anchors + width*num_anchors +
    // a (avoiding modulos in the process)
    int remaining = box_conv_index;
    const int delta_height = height_stride;  // stride of height
    const int h = remaining / delta_height;
    remaining -= h * delta_height;
    const int delta_width = num_anchors;  // stride of width
    const int w = remaining / delta_width;
    remaining -= w * delta_width;
    // Loading the anchor a
    // float4 is a struct with float x,y,z,w
    const float4 anchor = d_anchors[box_conv_index];
    // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
    float x1 = anchor.y();
    float x2 = anchor.w();
    float y1 = anchor.x();
    float y2 = anchor.z();

    // TODO(itex): use fast math when possible

    // Deltas of shape (N,height,width,num_anchors x 4)
    int deltas_idx = box_conv_index + image_index * image_stride;
    float4 deltas = d_bbox_deltas[deltas_idx];
    float dx = deltas.y();
    float dy = deltas.x();
    float dw = deltas.w();
    float dh = deltas.z();
    // Upper bound on dw,dh
    dw = fmin(dw, bbox_xform_clip);
    dh = fmin(dh, bbox_xform_clip);

    // Applying the deltas
    float width = x2 - x1;
    const float ctr_x = x1 + 0.5f * width;
    const float pred_ctr_x = ctr_x + width * dx;  // TODO(itex): fuse madd
    const float pred_w = width * expf(dw);
    x1 = pred_ctr_x - 0.5f * pred_w;
    x2 = pred_ctr_x + 0.5f * pred_w;

    float height = y2 - y1;
    const float ctr_y = y1 + 0.5f * height;
    const float pred_ctr_y = ctr_y + height * dy;
    const float pred_h = height * expf(dh);
    y1 = pred_ctr_y - 0.5f * pred_h;
    y2 = pred_ctr_y + 0.5f * pred_h;

    // Clipping box to image
    const float img_height = d_img_info_vec[5 * image_index + 0];
    const float img_width = d_img_info_vec[5 * image_index + 1];
    const float min_size_scaled =
        min_size * d_img_info_vec[5 * image_index + 2];
    x1 = fmax(fmin(x1, img_width), 0.0f);
    y1 = fmax(fmin(y1, img_height), 0.0f);
    x2 = fmax(fmin(x2, img_width), 0.0f);
    y2 = fmax(fmin(y2, img_height), 0.0f);

    // Filter boxes
    // Removing boxes with one dim < min_size
    // (center of box is in image, because of previous step)
    width = x2 - x1;  // may have changed
    height = y2 - y1;
    bool keep_box = fmin(width, height) >= min_size_scaled;

    // We are not deleting the box right now even if !keep_box
    // we want to keep the relative order of the elements stable
    // we'll do it in such a way later
    // d_boxes_keep_flags size: (num_images,prenms_nboxes)
    // d_out_boxes size: (num_images,prenms_nboxes)
    const int out_index = image_index * prenms_nboxes + ibox;

    d_boxes_keep_flags[out_index] = keep_box;
    d_out_boxes[out_index] = {x1, y1, x2, y2};
  }

 private:
  int nboxes_to_generate;
  int num_images;
  const int* d_sorted_scores_keys;
  const float4* d_bbox_deltas;
  const float4* d_anchors;
  int height;
  int width;
  int num_anchors;
  float min_size;
  const float* d_img_info_vec;
  float bbox_xform_clip;
  float4* d_out_boxes;
  int prenms_nboxes;
  char* d_boxes_keep_flags;
};

struct WriteUprightBoxesOutputTask {
  WriteUprightBoxesOutputTask(int work_element_count,
                              const float4* d_image_boxes,
                              const float* d_image_scores,
                              const int* d_image_boxes_keep_list, int n_rois,
                              float* d_image_out_rois,
                              float* d_image_out_rois_probs)
      : work_element_count(work_element_count),
        d_image_boxes(d_image_boxes),
        d_image_scores(d_image_scores),
        d_image_boxes_keep_list(d_image_boxes_keep_list),
        n_rois(n_rois),
        d_image_out_rois(d_image_out_rois),
        d_image_out_rois_probs(d_image_out_rois_probs) {}
  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_linear_id();
    if (i >= work_element_count) {
      return;
    }
    if (i < n_rois) {  // copy rois to output
      const int ibox = d_image_boxes_keep_list[i];
      const float4 box = d_image_boxes[ibox];
      const float score = d_image_scores[ibox];
      // Scattered memory accesses
      // postnms_nboxes is small anyway
      d_image_out_rois_probs[i] = score;
      const int base_idx = 4 * i;
      d_image_out_rois[base_idx + 0] = box.y();
      d_image_out_rois[base_idx + 1] = box.x();
      d_image_out_rois[base_idx + 2] = box.w();
      d_image_out_rois[base_idx + 3] = box.z();
    } else {  // set trailing entries to 0
      d_image_out_rois_probs[i] = 0.;
      const int base_idx = 4 * i;
      d_image_out_rois[base_idx + 0] = 0.;
      d_image_out_rois[base_idx + 1] = 0.;
      d_image_out_rois[base_idx + 2] = 0.;
      d_image_out_rois[base_idx + 3] = 0.;
    }
  }

 private:
  int work_element_count;
  const float4* d_image_boxes;
  const float* d_image_scores;
  const int* d_image_boxes_keep_list;
  int n_rois;
  float* d_image_out_rois;
  float* d_image_out_rois_probs;
};
struct InitializeDataKernel {
  InitializeDataKernel(int image_size, int num_images, int* d_image_offsets,
                       int* d_boxes_keys_iota)
      : image_size(image_size),
        num_images(num_images),
        d_image_offsets(d_image_offsets),
        d_boxes_keys_iota(d_boxes_keys_iota) {}
  void operator()(sycl::nd_item<1> item) const {
    const int idx = item.get_global_linear_id();
    if (idx >= image_size * num_images) {
      return;
    }
    const int img_idx = idx / image_size;
    const int box_idx = idx % image_size;
    d_boxes_keys_iota[img_idx * image_size + box_idx] = box_idx;

    // One 1D line sets the 1D data
    if (box_idx == 0) {
      d_image_offsets[img_idx] = image_size * img_idx;
      // One thread sets the last+1 offset
      if (img_idx == 0) d_image_offsets[num_images] = image_size * num_images;
    }
  }

 private:
  int image_size;
  int num_images;
  int* d_image_offsets;
  int* d_boxes_keys_iota;
};

class InitInputIndicesKernel;
class GroupSortKernel;

// Decode d_bbox_deltas with respect to anchors into absolute coordinates,
// clipping if necessary.
// prenms_nboxes maximum number of boxes per image to decode.
// d_boxes_keep_flags mask for boxes to consider in NMS.
// min_size is the lower bound of the shortest edge for the boxes to consider.
// bbox_xform_clip is the upper bound of encoded width and height.
Status GeneratePreNMSUprightBoxesKernel(
    const GPUDevice& device, const int nboxes_to_generate, const int num_images,
    const int* d_sorted_scores_keys, const float4* d_bbox_deltas,
    const float4* d_anchors, const int height, const int width,
    const int num_anchors, const float min_size,
    const float* d_img_info_vec,  // Input "image_info" to the op [N,5]
    const float bbox_xform_clip, float4* d_out_boxes,
    const int prenms_nboxes,  // leading dimension of out_boxes
    char* d_boxes_keep_flags) {
  // constants to calculate offsets in to the input and output arrays.
  auto stream = device.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup =
      (nboxes_to_generate * num_images + group_size - 1) / group_size;
  stream->submit([&](sycl::handler& cgh) {
    GeneratePreNMSUprightBoxesKernelTask task(
        nboxes_to_generate, num_images, d_sorted_scores_keys, d_bbox_deltas,
        d_anchors, height, width, num_anchors, min_size, d_img_info_vec,
        bbox_xform_clip, d_out_boxes, prenms_nboxes, d_boxes_keep_flags);
    cgh.parallel_for<GeneratePreNMSUprightBoxesKernelTask>(
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                          sycl::range<1>(group_size)),
        task);
  });
  return Status::OK();
}

// Copy the selected boxes and scores to output tensors.
//
Status WriteUprightBoxesOutput(const GPUDevice& device,
                               const int work_element_count,
                               const float4* d_image_boxes,
                               const float* d_image_scores,
                               const int* d_image_boxes_keep_list,
                               const int n_rois, float* d_image_out_rois,
                               float* d_image_out_rois_probs) {
  auto stream = device.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (work_element_count + group_size - 1) / group_size;
  stream->submit([&](sycl::handler& cgh) {
    WriteUprightBoxesOutputTask task(work_element_count, d_image_boxes,
                                     d_image_scores, d_image_boxes_keep_list,
                                     n_rois, d_image_out_rois,
                                     d_image_out_rois_probs);

    cgh.parallel_for<WriteUprightBoxesOutputTask>(
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                          sycl::range<1>(group_size)),
        task);
  });
  return Status::OK();
}

template <typename T>
Status ResetTensor(Tensor* t, const Eigen::GpuDevice& d) {
  auto count = t->NumElements();
  auto stream = d.stream();
  stream->fill<T>(t, T(0), count);
  return Status::OK();
}
// Allocate scratch spaces that are needed for operation
//

Status AllocateGenerationTempTensors(
    OpKernelContext* context, Tensor* d_conv_layer_indexes,
    Tensor* d_image_offset, /*Tensor* d_cub_temp_buffer,*/
    Tensor* d_sorted_conv_layer_indexes, Tensor* d_sorted_scores,
    Tensor* dev_boxes, Tensor* dev_boxes_keep_flags, int num_images,
    int conv_layer_nboxes, /*size_t cub_temp_storage_bytes,*/
    int num_boxes_to_generate, int box_dim) {
  auto d = context->eigen_gpu_device();
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images, conv_layer_nboxes}),
      d_conv_layer_indexes));
  TF_RETURN_IF_ERROR(ResetTensor<int>(d_conv_layer_indexes, d));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images + 1}), d_image_offset));
  TF_RETURN_IF_ERROR(ResetTensor<int>(d_image_offset, d));
  /*TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_temp_storage_bytes}),
      d_cub_temp_buffer));*/
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images, conv_layer_nboxes}),
      d_sorted_conv_layer_indexes));
  TF_RETURN_IF_ERROR(ResetTensor<int32>(d_sorted_conv_layer_indexes, d));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_images, conv_layer_nboxes}),
      d_sorted_scores));
  TF_RETURN_IF_ERROR(ResetTensor<float>(d_sorted_scores, d));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT,
      TensorShape({num_images, box_dim * num_boxes_to_generate}), dev_boxes));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_boxes, d));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({num_images, num_boxes_to_generate}),
      dev_boxes_keep_flags));
  TF_RETURN_IF_ERROR(ResetTensor<int8>(dev_boxes_keep_flags, d));
  return Status::OK();
}

// Allocate workspace for NMS operation
Status AllocatePreNMSTempTensors(
    OpKernelContext* context, Tensor* dev_image_prenms_boxes,
    Tensor* dev_image_prenms_scores, Tensor* dev_image_boxes_keep_list,
    Tensor* dev_postnms_rois, Tensor* dev_postnms_rois_probs,
    Tensor* dev_prenms_nboxes, int num_images, int num_boxes_to_generate,
    int box_dim, int post_nms_topn, int pre_nms_topn) {
  auto d = context->eigen_gpu_device();
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({box_dim * num_boxes_to_generate}),
      dev_image_prenms_boxes));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_image_prenms_boxes, d));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_boxes_to_generate}),
      dev_image_prenms_scores));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_image_prenms_scores, d));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes_to_generate}),
      dev_image_boxes_keep_list));
  TF_RETURN_IF_ERROR(ResetTensor<int32>(dev_image_boxes_keep_list, d));

  const int max_postnms_nboxes = std::min(num_boxes_to_generate, post_nms_topn);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT,
      TensorShape({box_dim * num_images * max_postnms_nboxes}),
      dev_postnms_rois));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_postnms_rois, d));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_images * max_postnms_nboxes}),
      dev_postnms_rois_probs));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_postnms_rois_probs, d));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images}), dev_prenms_nboxes));
  TF_RETURN_IF_ERROR(ResetTensor<int32>(dev_prenms_nboxes, d));

  return Status::OK();
}

// Initialize index and offset arrays.
// num_images is the batch size.
Status DoInitializeDataKernel(const GPUDevice& device, const int image_size,
                              const int num_images, int* d_image_offsets,
                              int* d_boxes_keys_iota) {
  auto stream = device.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (image_size * num_images + group_size - 1) / group_size;
  stream->submit([&](sycl::handler& cgh) {
    InitializeDataKernel task(image_size, num_images, d_image_offsets,
                              d_boxes_keys_iota);
    cgh.parallel_for<InitializeDataKernel>(
        sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                          sycl::range<1>(group_size)),
        task);
  });
  return Status::OK();
}

template <bool flip_box>
inline void Flipped(float4& box);  // NOLINT(runtime/references)

template <>
inline void Flipped<false>(float4& box) {}  // NOLINT(runtime/references)

template <>
inline void Flipped<true>(float4& box) {  // NOLINT(runtime/references)
  // float4: x(),y(),z(),w()  box: x1,y1,x2,y2
  if (box.x() > box.z()) Swap(box.x(), box.z());
  if (box.y() > box.w()) Swap(box.y(), box.w());
}

// Check whether two boxes have an iou greater than threshold
template <typename T>
inline bool OverThreshold(const float4& box_a, const float4& box_b,
                          const float a_area, const T iou_threshold) {
  // box a
  const float xmin_a = box_a.x();
  const float ymin_a = box_a.y();
  const float xmax_a = box_a.z();
  const float ymax_a = box_a.w();
  // box b
  const float xmin_b = box_b.x();
  const float ymin_b = box_b.y();
  const float xmax_b = box_b.z();
  const float ymax_b = box_b.w();

  const float b_area = (xmax_b - xmin_b) * (ymax_b - ymin_b);
  if (a_area <= 0.0f || b_area <= 0.0f) return false;

  // coord for intersection box
  const float xmin = (xmin_a > xmin_b) ? xmin_a : xmin_b;
  const float ymin = (ymin_a > ymin_b) ? ymin_a : ymin_b;
  const float xmax = (xmax_a < xmax_b) ? xmax_a : xmax_b;
  const float ymax = (ymax_a < ymax_b) ? ymax_a : ymax_b;
  const float width = (xmax > xmin) ? (xmax - xmin) : 0.0f;
  const float height = (ymax > ymin) ? (ymax - ymin) : 0.0f;
  const float intersection = width * height;
  return intersection > (a_area + b_area - intersection) * iou_threshold;
}

template <typename IN_T, typename OUT_T, typename LocalAcc>
void InclusiveScan(const sycl::nd_item<1>& item, LocalAcc local_acc,
                   const IN_T* input, OUT_T* data, OUT_T* carry,
                   const int group_size, const int local_range,
                   size_t idx_base) {
  auto idx = item.get_local_linear_id();
  // set data to local memory
  if (idx < local_range) {
    local_acc[idx] = input[idx_base + idx];
  }
  sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);
  if (idx == 0) local_acc[0] += *carry;
  sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);
#pragma unroll
  for (int k = 1; k < local_range; k *= 2) {
    if (idx - k >= 0) {
      local_acc[idx] += local_acc[idx - k];
    }
    sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);
  }
  sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);
  // copy back to global memory
  if (idx < local_range) {
    data[idx_base + idx] = local_acc[idx];
  }

  *carry = local_acc[local_range - 1];

  sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);
}

template <typename IN_T, typename INDEX_T, typename OUTPUT_T, typename LocalAcc>
void selectIf(const sycl::nd_item<1>& item, const int num, const int group_size,
              LocalAcc scratch, const IN_T* flag, OUTPUT_T* prefix_sum,
              const INDEX_T* flag_indices, INDEX_T* output) {
  auto idx_base = 0;
  auto idx = item.get_local_linear_id();
  const int block = num / group_size;
  const int left = num - block * group_size;
  OUTPUT_T carry = 0;
  size_t i = 0;
  for (i = 0; i < block; ++i) {
    idx_base = i * group_size;
    InclusiveScan(item, scratch, flag, prefix_sum, &carry, group_size,
                  group_size, idx_base);
  }
  if (left != 0) {
    idx_base = i * group_size;
    InclusiveScan(item, scratch, flag, prefix_sum, &carry, group_size, left,
                  idx_base);
  }
  sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);

  int range = item.get_global_range(0);
#pragma unroll
  for (int id = idx; id < num; id += range) {
    if (flag[id]) {
      output[prefix_sum[id] - 1] = flag_indices[id];
    }
  }
}

template <typename IN_T, typename OUT_T, typename LocalAcc>
struct SelectNonzeroKernel {
  SelectNonzeroKernel(int num, size_t max_group_size, LocalAcc scratch,
                      const IN_T* flag, int* flag_prefix_sum,
                      const OUT_T* flag_indices, OUT_T* selected_indices)
      : num(num),
        max_group_size(max_group_size),
        scratch(scratch),
        flag(flag),
        flag_prefix_sum(flag_prefix_sum),
        flag_indices(flag_indices),
        selected_indices(selected_indices) {}
  void operator()(sycl::nd_item<1> item) const {
    selectIf(item, num, max_group_size, scratch, flag, flag_prefix_sum,
             flag_indices, selected_indices);
  }

 private:
  int num;
  size_t max_group_size;
  LocalAcc scratch;
  const IN_T* flag;
  int* flag_prefix_sum;
  const OUT_T* flag_indices;
  OUT_T* selected_indices;
};

template <typename T, typename Index>
void SelectNonzero(OpKernelContext* context, const int num, const T* flag,
                   const Index* flag_indices, Index* selected_indices,
                   int* num_selected) {
  auto* stream = context->GetDeviceStream();
  Tensor d_flag_prefix_sum;
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataType::DT_INT32, TensorShape({num}),
                                        &d_flag_prefix_sum));
  int* flag_prefix_sum = d_flag_prefix_sum.flat<int32>().data();

  auto max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  stream->submit([&](sycl::handler& cgh) {
    LocalMem<int> scratch(max_group_size, cgh);
    SelectNonzeroKernel<T, Index, LocalMem<int>> task(
        num, max_group_size, scratch, flag, flag_prefix_sum, flag_indices,
        selected_indices);

    cgh.parallel_for<SelectNonzeroKernel<T, Index, LocalMem<int>>>(
        sycl::nd_range<1>(sycl::range<1>(max_group_size),
                          sycl::range<1>(max_group_size)),
        task);
  });

  stream->memcpy(num_selected, flag_prefix_sum + num - 1, sizeof(int));
}

// bit_mask must be a pointer
template <typename T>
inline bool CheckBit(T* bit_mask, int bit) {
  constexpr int kShiftLen = NumBits(8 * sizeof(T)) - 1;
  constexpr int kRemainderMask = 8 * sizeof(T) - 1;
  int bin = bit >> kShiftLen;
  return (bit_mask[bin] >> (bit & kRemainderMask)) & 1;
}

template <typename T>
void Iota(sycl::nd_item<1> item, const int total_num, T* to_fill,
          const T offset) {
  auto idx = item.get_global_linear_id();
  if (idx >= total_num) return;
  to_fill[idx] = static_cast<T>(idx) + offset;
}

template <typename Index>
inline void SelectHelper(const Index i_selected, const Index i_original) {}

template <typename Index, typename T, typename... Args>
inline void SelectHelper(const Index i_selected, const Index i_original,
                         const T* original, T* selected, Args... args) {
  selected[i_selected] = original[i_original];
  SelectHelper(i_selected, i_original, args...);
}

template <typename Index, typename T, typename... Args>
inline void IndexMultiSelect(sycl::nd_item<1> item, const int total_num,
                             Index* indices, const T* original, T* selected,
                             Args... args) {
  auto idx = item.get_global_linear_id();
  if (idx >= total_num) return;
  SelectHelper(static_cast<Index>(idx), indices[idx], original, selected,
               args...);
}

template <typename T>
void FilterByThreshold(sycl::nd_item<1> item, const int total_num, T* data,
                       T threshold, int* num) {
  auto idx = item.get_global_linear_id();
  if (idx >= total_num || data[idx] <= threshold) return;
  auto atom_num =
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(*num);
  atom_num.fetch_add(1);
}

template <typename T, bool flip_box>
void NMSMaskComputaion(const sycl::nd_item<2>& item, T* sorted_boxes,
                       int* delete_mask, const int num_boxes,
                       const float iou_threshold, const int bit_mask_len) {
  const int row_id = item.get_global_id(0);
  const int col_id = item.get_global_id(1);
  if (row_id >= num_boxes) return;
  float4& cur_box = sorted_boxes[row_id];
  Flipped<flip_box>(cur_box);

  float area_cur_box =
      (cur_box.w() - cur_box.y()) * (cur_box.z() - cur_box.x());

  const int compute_start_id = col_id * kNmsBoxesPerWorkItem;
  if (compute_start_id >= num_boxes) return;

  int mask = 0;
  for (int i = 0; i < kNmsBoxesPerWorkItem; ++i) {
    int compute_id = compute_start_id + i;
    if (compute_id <= row_id) continue;
    if (compute_id >= num_boxes) break;
    float4& compute_box = sorted_boxes[compute_id];
    Flipped<flip_box>(compute_box);
    if (OverThreshold(cur_box, compute_box, area_cur_box, iou_threshold)) {
      mask |= (1U << i);
    }
  }
  delete_mask[row_id * bit_mask_len + col_id] = mask;
}

void NMSReduce(const sycl::nd_item<1>& item, LocalMem<int> scratch,
               const int* delete_mask, const int wg_size,
               const int bit_mask_len, const int num_boxes,
               const int max_output_size, char* result_mask) {
  auto id = item.get_global_linear_id();
  int* local_ptr = ITEXGetLocalAccPointer<int>(scratch);
  auto range = wg_size;
  for (int i = id; i < bit_mask_len; i += range) {
    local_ptr[i] = ~(static_cast<int>(0));
  }
  sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);

  int accepted_boxes = 0;
  for (int box = 0; box < num_boxes; ++box) {
    // if current box is masked by an earlier box, skip it.
    if (!CheckBit<int>(local_ptr, box)) {
      continue;
    }
    ++accepted_boxes;
    int offset = box * bit_mask_len;
    for (int b = id; b < bit_mask_len; b += range) {
      int mask = delete_mask[offset + b];
      scratch[b] &= ~mask;
    }
    sycl::group_barrier(item.get_group(), sycl::memory_scope_work_group);
    if (accepted_boxes > max_output_size) break;
  }
  for (int box = id; box < num_boxes; box += range) {
    result_mask[box] = CheckBit<int>(local_ptr, box);
  }
}

template <typename T>
struct IotaKernel2 {
  IotaKernel2(int num_boxes, T* indices, T offset)
      : num_boxes(num_boxes), indices(indices) {}
  void operator()(sycl::nd_item<1> item) const {
    Iota<int>(item, num_boxes, indices, 0);
  }

 private:
  int num_boxes;
  T* indices;
};

template <typename T1, typename T2>
struct IndexMultiSelectKernel {
  IndexMultiSelectKernel(int num_boxes_compute, T1* sorted_indices,
                         const T2* original_boxes, T2* sorted_boxes)
      : num_boxes_compute(num_boxes_compute),
        sorted_indices(sorted_indices),
        original_boxes(original_boxes),
        sorted_boxes(sorted_boxes) {}
  void operator()(sycl::nd_item<1> item) const {
    IndexMultiSelect<T1, T2>(item, num_boxes_compute, sorted_indices,
                             original_boxes, sorted_boxes);
  }

 private:
  int num_boxes_compute;
  T1* sorted_indices;
  const T2* original_boxes;
  T2* sorted_boxes;
};

template <typename T1, bool T2>
struct NMSMaskComputaionKernel {
  NMSMaskComputaionKernel(float4* sorted_boxes, int* delete_mask, int num_boxes,
                          float iou_threshold, int bit_mask_len)
      : sorted_boxes(sorted_boxes),
        delete_mask(delete_mask),
        num_boxes(num_boxes),
        iou_threshold(iou_threshold),
        bit_mask_len(bit_mask_len) {}
  void operator()(sycl::nd_item<2> item) const;

 private:
  float4* sorted_boxes;
  int* delete_mask;
  int num_boxes;
  float iou_threshold;
  int bit_mask_len;
};

template <>
void NMSMaskComputaionKernel<float4, true>::operator()(
    sycl::nd_item<2> item) const {
  NMSMaskComputaion<float4, true>(item, sorted_boxes, delete_mask, num_boxes,
                                  iou_threshold, bit_mask_len);
}

template <>
void NMSMaskComputaionKernel<float4, false>::operator()(
    sycl::nd_item<2> item) const {
  NMSMaskComputaion<float4, false>(item, sorted_boxes, delete_mask, num_boxes,
                                   iou_threshold, bit_mask_len);
}

struct NMSReduceKernel {
  NMSReduceKernel(LocalMem<int> scratch, int* delete_mask,
                  size_t max_group_size, int bit_mask_len, int num_boxes,
                  size_t max_output_size, char* selected)
      : scratch(scratch),
        delete_mask(delete_mask),
        max_group_size(max_group_size),
        bit_mask_len(bit_mask_len),
        num_boxes(num_boxes),
        max_output_size(max_output_size),
        selected(selected) {}
  void operator()(sycl::nd_item<1> item) const {
    NMSReduce(item, scratch, delete_mask, max_group_size, bit_mask_len,
              num_boxes, max_output_size, selected);
  }

 private:
  LocalMem<int> scratch;
  int* delete_mask;
  size_t max_group_size;
  int bit_mask_len;
  int num_boxes;
  size_t max_output_size;
  char* selected;
};

struct FilterByThresholdKernel {
  FilterByThresholdKernel(int num_boxes, float* sorted_scores,
                          float score_threshold, int* valid_num_boxes)
      : num_boxes(num_boxes),
        sorted_scores(sorted_scores),
        score_threshold(score_threshold),
        valid_num_boxes(valid_num_boxes) {}
  void operator()(sycl::nd_item<1> item) const {
    FilterByThreshold(item, num_boxes, sorted_scores, score_threshold,
                      valid_num_boxes);
  }

 private:
  int num_boxes;
  float* sorted_scores;
  float score_threshold;
  int* valid_num_boxes;
};

Status NmsGpu(float4* sorted_boxes, const int num_boxes,
              const float iou_threshold, int* selected_indices,
              int* h_nkeep, /* host pointer*/
              OpKernelContext* context, const int max_output_size,
              bool flip_boxes = false) {
  auto* stream = context->GetDeviceStream();

  // step1: compute IOU mask matrix
  const int bit_mask_len =
      (num_boxes + kNmsBoxesPerWorkItem - 1) / kNmsBoxesPerWorkItem;
  const int64 mask_size = num_boxes * bit_mask_len;

  Tensor d_delete_mask;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({mask_size}), &d_delete_mask));
  int* delete_mask = d_delete_mask.flat<int32>().data();
  stream->fill<int>(delete_mask, 0, mask_size);

  int max_group_size = kNmsGroupDim;
  int num_group = DivUp(num_boxes, max_group_size);
  sycl::range<2> global(num_group * max_group_size, num_group * max_group_size);
  sycl::range<2> local(max_group_size, max_group_size);

  if (flip_boxes) {
    stream->submit([&](sycl::handler& cgh) {
      NMSMaskComputaionKernel<float4, true> task(
          sorted_boxes, delete_mask, num_boxes, iou_threshold, bit_mask_len);
      cgh.parallel_for<NMSMaskComputaionKernel<float4, true>>(
          sycl::nd_range<2>(global, local), task);
    });
  } else {
    stream->submit([&](sycl::handler& cgh) {
      NMSMaskComputaionKernel<float4, false> task(
          sorted_boxes, delete_mask, num_boxes, iou_threshold, bit_mask_len);
      cgh.parallel_for<NMSMaskComputaionKernel<float4, false>>(
          sycl::nd_range<2>(global, local), task);
    });
  }

  // step2: nms reduce within the same class
  Tensor selected_boxes;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({num_boxes}), &selected_boxes));
  char* selected = reinterpret_cast<char*>(selected_boxes.flat<int8>().data());
  max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  stream->submit([&](sycl::handler& cgh) {
    LocalMem<int> scratch(bit_mask_len, cgh);
    NMSReduceKernel task(scratch, delete_mask, max_group_size, bit_mask_len,
                         num_boxes, max_output_size, selected);
    cgh.parallel_for<NMSReduceKernel>(
        sycl::nd_range<1>(sycl::range<1>(max_group_size),
                          sycl::range<1>(max_group_size)),
        task);
  });

  // step3: select non zero index from NMSReduce result
  Tensor d_indices;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes}), &d_indices));
  max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  num_group = DivUp(num_boxes, max_group_size);
  int* indices = d_indices.flat<int>().data();
  stream->submit([&](sycl::handler& cgh) {
    IotaKernel2<int> task(num_boxes, indices, 0);
    cgh.parallel_for<IotaKernel2<int>>(
        sycl::nd_range<1>(sycl::range<1>(num_group * max_group_size),
                          sycl::range<1>(max_group_size)),
        task);
  });

  Tensor d_num_selected;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({1}), &d_num_selected));
  int* num_selected = d_num_selected.flat<int>().data();
  SelectNonzero(context, num_boxes, selected, indices, selected_indices,
                num_selected);
  stream->memcpy(h_nkeep, num_selected, sizeof(int)).wait();
  return Status::OK();
}
}  // namespace

class GenerateBoundingBoxProposals : public itex::OpKernel {
 public:
  explicit GenerateBoundingBoxProposals(itex::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("post_nms_topn", &post_nms_topn_));
    OP_REQUIRES(context, post_nms_topn_ > 0,
                errors::InvalidArgument("post_nms_topn can't be 0 or less"));
    bbox_xform_clip_default_ = log(1000.0 / 16.);
  }

  template <typename T>
  Status GetScalarValue(OpKernelContext* context, int input, T* value) {
    const Tensor& scalar_tensor = context->input(input);
    if (!TensorShapeUtils::IsScalar(scalar_tensor.shape())) {
      return errors::InvalidArgument("Expected a scalar in input ", input,
                                     "but got shape ",
                                     scalar_tensor.shape().DebugString());
    }
    *value = scalar_tensor.scalar<T>()();
    return Status::OK();
  }

  void Compute(itex::OpKernelContext* context) override {
    const GPUDevice& d = context->eigen_gpu_device();
    const auto scores = context->input(0);
    const auto bbox_deltas = context->input(1);
    const auto image_info = context->input(2);
    const auto anchors = context->input(3);
    const auto num_images = scores.dim_size(0);
    const auto num_anchors = scores.dim_size(3);
    const auto height = scores.dim_size(1);
    const auto width = scores.dim_size(2);
    const auto box_dim = anchors.dim_size(2) / num_anchors;
    OP_REQUIRES(context, box_dim == 4,
                errors::OutOfRange("Box dimensions need to be 4"));
    // TODO(skama): make sure that inputs are ok.
    const int image_stride = height * width;
    const int conv_layer_nboxes =
        image_stride *
        num_anchors;  // total number of boxes when decoded on anchors.

    float nms_threshold;
    int pre_nms_topn;
    float min_size;
    OP_REQUIRES_OK(context, GetScalarValue(context, 4, &nms_threshold));
    if (nms_threshold < 0 || nms_threshold > 1.0) {
      context->SetStatus(errors::InvalidArgument(
          "nms_threshold should be between 0 and 1. Got ", nms_threshold));
      return;
    }
    OP_REQUIRES_OK(context, GetScalarValue(context, 5, &pre_nms_topn));
    if (pre_nms_topn <= 0) {
      context->SetStatus(errors::InvalidArgument(
          "pre_nms_topn should be greater than 0", pre_nms_topn));
      return;
    }
    OP_REQUIRES_OK(context, GetScalarValue(context, 6, &min_size));
    const int input_num = num_images * conv_layer_nboxes;

    Tensor d_conv_layer_indexes;  // box indices on device
    Tensor d_image_offset;        // starting offsets boxes for each image
    Tensor d_cub_temp_buffer;     // buffer for cub sorting
    Tensor d_sorted_conv_layer_indexes;  // output of cub sorting, indices of
                                         // the sorted boxes
    Tensor dev_sorted_scores;            // sorted scores, cub output
    Tensor dev_boxes;                    // boxes on device
    Tensor dev_boxes_keep_flags;  // bitmask for keeping the boxes or rejecting
                                  // from output
    const int nboxes_to_generate = std::min(conv_layer_nboxes, pre_nms_topn);

    OP_REQUIRES_OK(context,
                   AllocateGenerationTempTensors(
                       context, &d_conv_layer_indexes, &d_image_offset,
                       &d_sorted_conv_layer_indexes, &dev_sorted_scores,
                       &dev_boxes, &dev_boxes_keep_flags, num_images,
                       conv_layer_nboxes, nboxes_to_generate, box_dim));
    // create box indices and offsets for each image on device
    OP_REQUIRES_OK(context, DoInitializeDataKernel(
                                d, conv_layer_nboxes, num_images,
                                d_image_offset.flat<int>().data(),
                                d_conv_layer_indexes.flat<int>().data()));

    // sort boxes with their scores.
    // d_sorted_conv_layer_indexes will hold the pointers to old indices.

    auto sorted_scores_t = dev_sorted_scores.flat_inner_dims<float>();
    auto sorted_indices_t = d_sorted_conv_layer_indexes.flat_inner_dims<int>();
    const auto& original_scores_t = scores.flat_inner_dims<float>();
    functor::TopKFunctor<GPUDevice, float, int>()(
        context, original_scores_t, sorted_scores_t, sorted_indices_t,
        true /*sorted*/, input_num);

    // Keeping only the topN pre_nms
    // create box y1,x1,y2,x2 from box_deltas and anchors (decode the boxes) and
    // mark the boxes which are smaller that min_size ignored.
    OP_REQUIRES_OK(
        context,
        GeneratePreNMSUprightBoxesKernel(
            d, nboxes_to_generate, num_images,
            d_sorted_conv_layer_indexes.flat<int>().data(),
            reinterpret_cast<const float4*>(bbox_deltas.flat<float>().data()),
            reinterpret_cast<const float4*>(anchors.flat<float>().data()),
            height, width, num_anchors, min_size,
            image_info.flat<float>().data(), bbox_xform_clip_default_,
            reinterpret_cast<float4*>(dev_boxes.flat<float>().data()),
            nboxes_to_generate,
            reinterpret_cast<char*>(dev_boxes_keep_flags.flat<int8>().data())));
    const int nboxes_generated = nboxes_to_generate;
    const int roi_cols = box_dim;
    Tensor dev_image_prenms_boxes;
    Tensor dev_image_prenms_scores;
    Tensor dev_image_boxes_keep_list;
    Tensor dev_postnms_rois;
    Tensor dev_postnms_rois_probs;
    Tensor dev_prenms_nboxes;
    // Allocate workspaces needed for NMS
    OP_REQUIRES_OK(
        context, AllocatePreNMSTempTensors(
                     context, &dev_image_prenms_boxes, &dev_image_prenms_scores,
                     &dev_image_boxes_keep_list, &dev_postnms_rois,
                     &dev_postnms_rois_probs, &dev_prenms_nboxes, num_images,
                     nboxes_generated, box_dim, post_nms_topn_, pre_nms_topn));
    // get the pointers for temp storages
    int* d_prenms_nboxes = dev_prenms_nboxes.flat<int>().data();
    int h_prenms_nboxes = 0;
    float* d_image_prenms_boxes = dev_image_prenms_boxes.flat<float>().data();
    float* d_image_prenms_scores = dev_image_prenms_scores.flat<float>().data();
    int* d_image_boxes_keep_list = dev_image_boxes_keep_list.flat<int>().data();

    // get the pointers to boxes and scores
    char* d_boxes_keep_flags =
        reinterpret_cast<char*>(dev_boxes_keep_flags.flat<int8>().data());
    float* d_boxes = dev_boxes.flat<float>().data();
    float* d_sorted_scores = dev_sorted_scores.flat<float>().data();

    // Create output tensors
    Tensor* output_rois = nullptr;
    Tensor* output_roi_probs = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({num_images, post_nms_topn_, roi_cols}),
                       &output_rois));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({num_images, post_nms_topn_}),
                                &output_roi_probs));
    float* d_postnms_rois = (*output_rois).flat<float>().data();
    float* d_postnms_rois_probs = (*output_roi_probs).flat<float>().data();
    // gpuEvent_t copy_done;
    // gpuEventCreate(&copy_done);

    // Do  per-image nms
    for (int image_index = 0; image_index < num_images; ++image_index) {
      // reset output workspaces
      OP_REQUIRES_OK(context,
                     ResetTensor<int32>(&dev_image_boxes_keep_list, d));
      // Sub matrices for current image
      // boxes
      const float* d_image_boxes =
          &d_boxes[image_index * nboxes_generated * box_dim];
      // scores
      const float* d_image_sorted_scores =
          &d_sorted_scores[image_index * image_stride * num_anchors];
      // keep flags
      char* d_image_boxes_keep_flags =
          &d_boxes_keep_flags[image_index * nboxes_generated];

      // Output buffer for image
      float* d_image_postnms_rois =
          &d_postnms_rois[image_index * roi_cols * post_nms_topn_];
      float* d_image_postnms_rois_probs =
          &d_postnms_rois_probs[image_index * post_nms_topn_];

      // Moving valid boxes (ie the ones with d_boxes_keep_flags[ibox] == true)
      // to the output tensors
      SelectNonzero(context, nboxes_generated, d_image_boxes_keep_flags,
                    d_image_boxes, d_image_prenms_boxes, d_prenms_nboxes);
      SelectNonzero(context, nboxes_generated, d_image_boxes_keep_flags,
                    d_image_sorted_scores, d_image_prenms_scores,
                    d_prenms_nboxes);
      d.memcpyDeviceToHost(&h_prenms_nboxes, d_prenms_nboxes, sizeof(int));
      // We know prenms_boxes <= topN_prenms,
      // because nboxes_generated <= topN_prenms.
      // Calling NMS on the generated boxes
      const int prenms_nboxes = h_prenms_nboxes;
      int nkeep;

      OP_REQUIRES_OK(
          context,
          itex::NmsGpu(reinterpret_cast<float4*>(d_image_prenms_boxes),
                       prenms_nboxes, nms_threshold, d_image_boxes_keep_list,
                       &nkeep, context, post_nms_topn_));
      // All operations done after previous sort
      // were keeping the relative order of the
      // elements the elements are still sorted keep
      // topN <=> truncate the array
      const int postnms_nboxes = std::min(nkeep, post_nms_topn_);
      // Moving the out boxes to the output tensors,
      // adding the image_index dimension on the fly
      // make this single kernel
      OP_REQUIRES_OK(
          context,
          WriteUprightBoxesOutput(
              d, post_nms_topn_,
              reinterpret_cast<const float4*>(d_image_prenms_boxes),
              d_image_prenms_scores, d_image_boxes_keep_list, postnms_nboxes,
              d_image_postnms_rois, d_image_postnms_rois_probs));
      return;
    }
  }

 private:
  int post_nms_topn_;
  float bbox_xform_clip_default_;
};

REGISTER_KERNEL_BUILDER(Name("GenerateBoundingBoxProposals")
                            .Device(DEVICE_GPU)
                            .HostMemory("nms_threshold")
                            .HostMemory("min_size")
                            .HostMemory("pre_nms_topn"),
                        itex::GenerateBoundingBoxProposals);
}  // namespace itex
