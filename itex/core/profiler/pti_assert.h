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

//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef ITEX_CORE_PROFILER_PTI_ASSERT_H_
#define ITEX_CORE_PROFILER_PTI_ASSERT_H_

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#define PTI_ASSERT(X) assert(X)

#endif  // ITEX_CORE_PROFILER_PTI_ASSERT_H_
