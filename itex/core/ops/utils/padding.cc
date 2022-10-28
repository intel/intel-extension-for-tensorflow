/* Copyright (c) 2021 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "itex/core/ops/utils/padding.h"

const char* GetConvnetDataFormatAttrString() {
  static const char kDataFormat[] = "data_format: { 'NHWC', 'NCHW' } = 'NHWC' ";
  return kDataFormat;
}

const char* GetConvnet3dDataFormatAttrString() {
  static const char k3dDataFormat[] =
      "data_format: { 'NDHWC', 'NCDHW' } = 'NDHWC' ";
  return k3dDataFormat;
}

const char* GetPaddingAttrString() {
  static const char kPadding[] = "padding: {'SAME', 'VALID'}";
  return kPadding;
}

const char* GetPaddingAttrStringWithExplicit() {
  static const char kPaddingWithEplicit[] =
      "padding: {'SAME', 'VALID', 'EXPLICIT'}";
  return kPaddingWithEplicit;
}

const char* GetExplicitPaddingsAttrString() {
  static const char kExplicitPadding[] = "explicit_paddings: list(int) = []";
  return kExplicitPadding;
}
