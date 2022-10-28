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

#ifndef ITEX_CORE_OPS_UTILS_PADDING_H_
#define ITEX_CORE_OPS_UTILS_PADDING_H_

// This file contains helper routines to deal with padding in various ops and
// kernels.

#include <string>
#ifdef __cplusplus
extern "C" {
#endif

const char* GetConvnetDataFormatAttrString();

const char* GetConvnet3dDataFormatAttrString();

// Return the string containing the list of valid padding types, that can be
// used as an Attr() in REGISTER_OP.
const char* GetPaddingAttrString();

// Like GetPaddingAttrString(), but also includes EXPLICIT.
const char* GetPaddingAttrStringWithExplicit();

const char* GetExplicitPaddingsAttrString();

#ifdef __cplusplus
}
#endif

#endif  // ITEX_CORE_OPS_UTILS_PADDING_H_
