/* Copyright (c) 2021 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_UTILS_OP_REQUIRES_H_
#define ITEX_CORE_UTILS_OP_REQUIRES_H_

#include "itex/core/utils/macros.h"

namespace itex {

// Convenience macros for asserting and handling exceptional conditions.
// Analogous to the ITEX_CHECK* macros provided by logging.h.
//
// Example use:
// void Compute(OperationContext* context) {
//   OP_REQUIRES(context, context->num_inputs() == 2,
//               errors::InvalidArgument("FooOp requires 2 arguments"));
//   ...
//   Status status = SomeUncertainMethod();
//   OP_REQUIRES_OK(context, status);
//   ...
// }
//
// These macros depend on CheckNotInComputeAsync, which must be defined before
// invoking the macro. We specifically don't include op_kernel.h from this
// header to reduce this header's dependencies. These macros may be used with
// alternative implementations of OpKernelContext with fewer dependencies.

#define OP_REQUIRES(CTX, EXP, STATUS)                     \
  do {                                                    \
    if (!ITEX_PREDICT_TRUE(EXP)) {                        \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      return;                                             \
    }                                                     \
  } while (0)

#define OP_REQUIRES_RETURN_STATUS(CTX, EXP, STATUS)       \
  do {                                                    \
    if (!ITEX_PREDICT_TRUE(EXP)) {                        \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      return STATUS;                                      \
    }                                                     \
  } while (0)

#define OP_REQUIRES_OK_RETURN_STATUS(CTX, ...)               \
  do {                                                       \
    ::itex::Status _s(__VA_ARGS__);                          \
    if (!ITEX_PREDICT_TRUE(_s.ok())) {                       \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC"); \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);  \
      return _s;                                             \
    }                                                        \
  } while (0)

#define OP_REQUIRES_OK(CTX, ...)                             \
  do {                                                       \
    ::itex::Status _s(__VA_ARGS__);                          \
    if (!ITEX_PREDICT_TRUE(_s.ok())) {                       \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC"); \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);  \
      return;                                                \
    }                                                        \
  } while (0)

#define OP_REQUIRES_ASYNC(CTX, EXP, STATUS, CALLBACK)  \
  do {                                                 \
    if (!ITEX_PREDICT_TRUE(EXP)) {                     \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS)); \
      (CALLBACK)();                                    \
      return;                                          \
    }                                                  \
  } while (0)

#define OP_REQUIRES_OK_ASYNC(CTX, STATUS, CALLBACK)         \
  do {                                                      \
    ::itex::Status _s(STATUS);                              \
    if (!ITEX_PREDICT_TRUE(_s.ok())) {                      \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      (CALLBACK)();                                         \
      return;                                               \
    }                                                       \
  } while (0)

#define OP_REQUIRES_PTR(CTX, EXP, STATUS)                 \
  do {                                                    \
    if (!ITEX_PREDICT_TRUE(EXP)) {                        \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      return nullptr;                                     \
    }                                                     \
  } while (0)

#define OP_REQUIRES_OK_PTR(CTX, ...)                         \
  do {                                                       \
    ::itex::Status _s(__VA_ARGS__);                          \
    if (!ITEX_PREDICT_TRUE(_s.ok())) {                       \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC"); \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);  \
      return nullptr;                                        \
    }                                                        \
  } while (0)

#define OP_REQUIRES_ASYNC_PTR(CTX, EXP, STATUS, CALLBACK) \
  do {                                                    \
    if (!ITEX_PREDICT_TRUE(EXP)) {                        \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      (CALLBACK)();                                       \
      return nullptr;                                     \
    }                                                     \
  } while (0)

#define OP_REQUIRES_OK_ASYNC_PTR(CTX, STATUS, CALLBACK)     \
  do {                                                      \
    ::itex::Status _s(STATUS);                              \
    if (!ITEX_PREDICT_TRUE(_s.ok())) {                      \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      (CALLBACK)();                                         \
      return nullptr;                                       \
    }                                                       \
  } while (0)

}  // namespace itex

#endif  // ITEX_CORE_UTILS_OP_REQUIRES_H_
