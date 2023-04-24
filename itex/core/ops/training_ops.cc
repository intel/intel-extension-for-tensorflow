/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/ops/shape_inference_fns.h"
#include "itex/core/ops/utils/logging.h"
#include "itex/core/ops/utils/status.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

void Register_ITEXFusedApplyMomentumOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedApplyMomentum");

    TF_OpDefinitionBuilderAddInput(op_builder, "var: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "accum: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "lr: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "momentum: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_input: num_mul_inputs * T");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "addn_input: num_addn_inputs * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Ref(T)");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_locking: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_nesterov: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_addn_inputs: int >= 0 = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_mul_inputs: int >= 1 = 2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedApplyMomentum op registration failed: ";
  }
}

void Register_ITEXFusedResourceApplyMomentumOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedResourceApplyMomentum");

    TF_OpDefinitionBuilderAddInput(op_builder, "var: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "accum: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "lr: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "momentum: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_input: num_mul_inputs * T");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "addn_input: num_addn_inputs * T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_locking: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_nesterov: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_addn_inputs: int >= 0 = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_mul_inputs: int >= 1 = 2");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedResourceApplyMomentum op registration failed: ";
  }
}

void Register_ITEXFusedApplyAdamOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedApplyAdam");

    TF_OpDefinitionBuilderAddInput(op_builder, "var: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "m: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "v: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "lr: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "epsilon: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_left: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_right: T");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "addn_input: num_addn_inputs * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Ref(T)");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_locking: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_nesterov: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_addn_inputs: int >= 0 = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedApplyAdam op registration failed: ";
  }
}

void Register_ITEXFusedResourceApplyAdamOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedResourceApplyAdam");

    TF_OpDefinitionBuilderAddInput(op_builder, "var: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "m: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "v: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "lr: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "epsilon: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_left: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_right: T");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "addn_input: num_addn_inputs * T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_locking: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_nesterov: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_addn_inputs: int >= 0 = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedResourceApplyAdam op registration failed: ";
  }
}

void Register_ITEXApplyAdamWithWeightDecayOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ITEXApplyAdamWithWeightDecay");

    TF_OpDefinitionBuilderAddInput(op_builder, "var: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "m: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "v: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "lr: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "epsilon: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "weight_decay: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Ref(T)");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_locking: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_nesterov: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(
        op_builder, &apply_adam_with_weight_decay_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ITEXApplyAdamWithWeightDecay op registration failed: ";
  }
}

void Register_ITEXResourceApplyAdamWithWeightDecayOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ITEXResourceApplyAdamWithWeightDecay");

    TF_OpDefinitionBuilderAddInput(op_builder, "var: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "m: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "v: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "lr: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "epsilon: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "weight_decay: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_locking: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_nesterov: bool = false");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &empty_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "ITEXResourceApplyAdamWithWeightDecay op registration failed: ";
  }
}

void Register_ITEXFusedApplyAdamWithWeightDecayOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedApplyAdamWithWeightDecay");

    TF_OpDefinitionBuilderAddInput(op_builder, "var: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "m: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "v: Ref(T)");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "lr: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "epsilon: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "weight_decay: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_left: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_right: T");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "addn_input: num_addn_inputs * T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "out: Ref(T)");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_locking: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_nesterov: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_addn_inputs: int >= 0 = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedApplyAdamWithWeightDecay op registration failed: ";
  }
}

void Register_ITEXFusedResourceApplyAdamWithWeightDecayOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("_ITEXFusedResourceApplyAdamWithWeightDecay");

    TF_OpDefinitionBuilderAddInput(op_builder, "var: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "m: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "v: resource");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2_power: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "lr: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta1: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "beta2: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "epsilon: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "weight_decay: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_left: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "mul_right: T");
    TF_OpDefinitionBuilderAddInput(op_builder,
                                   "addn_input: num_addn_inputs * T");

    TF_OpDefinitionBuilderAddAttr(op_builder, "T: numbertype");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_locking: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "use_nesterov: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "num_addn_inputs: int >= 0 = 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unknown_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "_ITEXFusedResourceApplyAdamWithWeightDecay op registration "
           "failed: ";
  }
}

void RegisterRMSPropComputeRMSOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ApplyRMSPropComputeRMS");
    TF_OpDefinitionBuilderAddInput(op_builder, "ms: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "rho: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "RMSPropComputeRMS op registration failed: ";
  }
}

void RegisterRMSPropVarUpdateOp() {
  itex::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder =
        TF_NewOpDefinitionBuilder("ApplyRMSPropVarUpdate");
    TF_OpDefinitionBuilderAddInput(op_builder, "var: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "ms: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "lr: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "epsilon: T");
    TF_OpDefinitionBuilderAddInput(op_builder, "grad: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {half, bfloat16, float}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);
    TF_RegisterOpDefinition(op_builder, status.get());
    ITEX_CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "RMSPropVarUpdate op registration failed: ";
  }
}
