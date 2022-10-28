/* Copyright (c) 2021 Intel Corporation

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

#ifndef ITEX_CORE_UTILS_COMMON_SHAPE_FNS_H_
#define ITEX_CORE_UTILS_COMMON_SHAPE_FNS_H_

#include "itex/core/utils/padding.h"
#include "itex/core/utils/status.h"

namespace itex {

// The V2 version computes the same outputs with arbitrary dilation_rate.
Status Get3dOutputSizeV2(const std::array<int64, 3>& input,
                         const std::array<int64, 3>& window,
                         const std::array<int64, 3>& dilations,
                         const std::array<int64, 3>& strides,
                         Padding padding_type, std::array<int64, 3>* output_ptr,
                         std::array<int64, 3>* padding_before,
                         std::array<int64, 3>* padding_after);

// Returns the same output dimensions as in GetWindowedOutputSize, but returns
// verbose padding dimensions (before/after), and EXPLICIT padding is supported.
// When padding_type is EXPLICIT, *padding_before and *padding_after must
// already point to initialized integers with the padding amounts. Otherwise,
// *padding_before and *padding_after are set by this function, and any
// excess padding (caused by an odd padding size value) is added to the
// 'padding_after' dimension.
Status GetWindowedOutputSizeVerbose(int64 input_size, int64 filter_size,
                                    int64 stride, Padding padding_type,
                                    int64* output_size, int64* padding_before,
                                    int64* padding_after);

// The V2 version computes the same outputs with arbitrary dilation_rate. For
// detailed equations, refer to the comments for GetWindowedOutputSizeV2().
Status GetWindowedOutputSizeVerboseV2(int64 input_size, int64 filter_size,
                                      int64 dilation_rate, int64 stride,
                                      Padding padding_type, int64* output_size,
                                      int64* padding_before,
                                      int64* padding_after);

// GetWindowedOutputSize(): Given an input tensor, kernel, stride and padding
// type, the function computes the output and padding dimensions.
//
// For example, ignoring batches or multiple features, a 1D convolution
// takes as input a 1D tensor of shape (H), and convolves it with a filter of
// shape (K).
//
// It also takes in a few additional parameters:
//
// Stride (S): the stride with which we apply the filters. This is the offset
// between locations where we apply the filters. A larger stride
// means that the output will be spatially smaller.
//
// Padding (P): the padding we apply to the input tensor along each
// dimension. This is usually used to make sure that the spatial dimensions
// do not shrink when we progress with convolutions. This function supports two
// types of padding.
//   SAME: the pad value is computed so that the output will have size H/S.
//   VALID: no padding is carried out.
// If you want to use EXPLICIT padding, GetWindowedOutputSizeVerbose must be
// called instead. Note the padded area is zero-filled.
//
// The output dimensions for convolution and many other operations, when given
// all the parameters above, are as follows:
// - When Padding = SAME: the output size is (H'), where
//     H' = ceil(float(H) / float(S))
//   where ceil is the ceiling function. The number of padded cells
//   is computed as:
//     Pc = ((H' - 1) * S + K - H) / 2
//   When the stride is 1, the expression simplifies to
//     H' = H, Pc = (K-1)/2.
//   This is where SAME comes from - the output has the same size as the input
//   has.
//
// - When Padding = VALID: the output size is computed as
//     H' = ceil(float(H - K + 1) / float(S))
//   and the number of padded cells is always zero.
//   When the stride is 1, the expression simplifies to
//     H' = H-K+1.
//
// For convolution, mathematically, the output value at location (r')
// is the inner product of two vectors: the chunk of input at
//    ((r'*S-Pr) : (r'*S-Pr+K)),
// and the filter.
//
// For 2D and 3D convolutions, the spatial dimensions are orthogonal, so the
// size and padding of each spatial dimension can be computed by calling
// GetWindowedOutputSize separately for each dimension.
//
Status GetWindowedOutputSize(int64 input_size, int64 filter_size, int64 stride,
                             Padding padding_type, int64* output_size,
                             int64* padding_size);

}  // namespace itex

#endif  // ITEX_CORE_UTILS_COMMON_SHAPE_FNS_H_
