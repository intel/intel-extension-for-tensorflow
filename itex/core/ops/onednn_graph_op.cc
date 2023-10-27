/* Copyright (c) 2021 Intel Corporation

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

#include "itex/core/ops/shape_inference_fns.h"
#include "itex/core/ops/utils/logging.h"
#include "itex/core/ops/utils/status.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

void Register_OneDnnGraphOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
#ifdef INTEL_CPU_ONLY
        TF_NewOpDefinitionBuilder("OneDnnGraphCPU");
#else
        TF_NewOpDefinitionBuilder("OneDnnGraph");
#endif
    TF_OpDefinitionBuilderAddInput(op_builder, "args: Tin");
    TF_OpDefinitionBuilderAddOutput(op_builder, "results: Tout");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tin: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "partition_id: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "input_edge_ids: list(int) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "output_edge_ids: list(int) >= 0");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "is_constant_input_edge: list(bool) >= 0");  // Used for constant cache
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "candidate_inplace_input_edge: list(bool) >= 0");  // Used for inplace
    TF_OpDefinitionBuilderAddAttr(op_builder, "framework_ops: list(string)");

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "OneDnnGraph op registration failed: ";
  }
  {
    TF_OpDefinitionBuilder* op_builder =
#ifdef INTEL_CPU_ONLY
        TF_NewOpDefinitionBuilder("_OneDnnGraphCPU");
#else
        TF_NewOpDefinitionBuilder("_OneDnnGraph");
#endif
    TF_OpDefinitionBuilderAddInput(op_builder, "args: Tin");
    TF_OpDefinitionBuilderAddInput(op_builder, "args_meta: Tin_meta");
    TF_OpDefinitionBuilderAddOutput(op_builder, "results: Tout");
    TF_OpDefinitionBuilderAddOutput(op_builder, "results_meta: Tout_meta");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tin: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tin_meta: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout_meta: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "partition_id: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "input_edge_ids: list(int) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder,
                                  "output_edge_ids: list(int) >= 0");
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "is_constant_input_edge: list(bool) >= 0");  // Used for constant cache
    TF_OpDefinitionBuilderAddAttr(
        op_builder,
        "candidate_inplace_input_edge: list(bool) >= 0");  // Used for inplace
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_end_node: list(bool) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "framework_ops: list(string)");

    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_OneDnnGraph op registration failed: ";
  }
}
