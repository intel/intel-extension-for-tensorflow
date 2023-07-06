/* Copyright (c) 2022 Intel Corporation

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

// Class and associated machinery for specifying an Op's OpDef and shape
// inference function for Op registration.

#ifndef ITEX_CORE_UTILS_OP_DEF_BUILDER_H_
#define ITEX_CORE_UTILS_OP_DEF_BUILDER_H_

#include <map>
#include <string>
#include <vector>

#include "itex/core/utils/macros.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/statusor.h"
#include "itex/core/utils/stringpiece.h"
#include "itex/core/utils/types.h"
#include "protos/full_type.pb.h"
#include "protos/op_def.pb.h"

namespace itex {

// TODO(b/62899350): Refactor without proto dependencies.
typedef std::function<Status(OpDef* c)> OpTypeConstructor;

typedef std::vector<std::reference_wrapper<const FullTypeDef>> TypeRefVector;
typedef std::map<std::string, std::reference_wrapper<const FullTypeDef>>
    TypeRefMap;

// A type inference function, called for each node during type inference
// (possibly multiple times).
// The first argument (input_types) will hold the type of each of the node's
// inputs. The second argument (type_vars) will hold the return type of
// each function referred from any type variable (e.g. `FuncVar`) present
// in the node's corresponding op definition.
//
// TODO(mdan): Consider a vector-in, vector-out contract.
// TODO(mdan): Rename to just TypeInferenceFn (since it's not always "forward").
typedef std::function<StatusOr<FullTypeDef>(const TypeRefVector&,
                                            const TypeRefMap&)>
    ForwardTypeInferenceFn;

}  // namespace itex

#endif  // ITEX_CORE_UTILS_OP_DEF_BUILDER_H_
