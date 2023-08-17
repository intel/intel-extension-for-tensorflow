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

#ifndef ITEX_CORE_GRAPH_UTILS_LAYOUT_UTILS_H_
#define ITEX_CORE_GRAPH_UTILS_LAYOUT_UTILS_H_

#include <string>
#include <unordered_set>

#include "itex/core/graph/utils/graph_view.h"
#include "itex/core/utils/cpu_info.h"
#include "itex/core/utils/function.h"
#include "itex/core/utils/op_def_util.h"

namespace itex {
namespace graph {

// Structure to specify a forward op, a backward op, and the slot numbers
// in the forward and backward ops where we will add a workspace edge.
typedef struct {
  string bwd_op;  // Name of a backward op in the graph

  int bwd_slot;     // Input slot in the backward op node where actual
                    // Input tensor resides
  int ws_fwd_slot;  // Output slot in the forward op node where workspace
                    // edge is added
} WorkSpaceInfo;

string GetInputName(const NodeDef* input, int out_slot);

// Check whether output_node in wsinfo, find input_node, add workspace edge
// between input and output, return input_node.
NodeDef* AddWorkspace(
    const itex::graph::utils::MutableNodeView* ori_output_node_view,
    NodeDef* new_output_node_def);

//////////////////////////////////////////////////////////////////////////
// DataType Check
//////////////////////////////////////////////////////////////////////////
bool IsQuantizedOp(const string& op_name);
// Some int8 kernels are only registered by intel tensorflow, not available in
// stock tensorflow. We need to always rewrite these ops.
bool IsDataTypeExemptOp(const string& op_name);

bool IsLayoutRewriteSupportedDataType(const NodeDef& node_def);

bool IsOneDnnLayoutPartialDependentOp(const string& op_name);

bool IsOneDnnLayoutDependentOp(const string& op_name);

bool IsPlainLayoutOp(const string& op_name);

//////////////////////////////////////////////////////////////////////////
// Rewrite functions
//////////////////////////////////////////////////////////////////////////

// Default rewrite rule to be used in scenario 1 for rewrite.
// @return - true (since we want to always rewrite)
bool AlwaysRewrite(const utils::MutableNodeView& node_view);

// Backward only supports FP32, BF16
bool RewriteBackwardDataType(const utils::MutableNodeView& node_view);

// Conv op is rewritten only if there are OneDnn ops in its input or output.
bool RewriteOneDnnConv(const utils::MutableNodeView& node_view);

bool RewriteLayerNorm(const utils::MutableNodeView& node_view);

bool RewriteLayerNormGrad(const utils::MutableNodeView& node_view);

// FusedBatchNormEx is rewritten when input is 4D tensor and only one ReLU
// side_input
bool RewriteFusedBatchNormEx(const utils::MutableNodeView& node_view);

// FusedBatchNormExGrad is rewritten when input is 4D tensor and only one
// ReluGrad side_input
bool RewriteFusedBatchNormExGrad(const utils::MutableNodeView& node_view);

bool RewriteFusedConv(const utils::MutableNodeView& node_view);

bool RewriteOneDnnFusedConv(const utils::MutableNodeView& node_view);

// MatMul is not rewritten when trans_a/trans_b = True.
bool RewriteMatMul(const utils::MutableNodeView& node_view);

// Rewrite rule for Conv2DBackprop.
// @return - true if `padding` and data type are supported.
bool RewriteConv2DBackprop(const utils::MutableNodeView& node_view);

// Rewrite rule for PoolOp(MaxPool/AvgPool).
// @return - true if following conditions are all true:
//   1) Padding type is not `EXPLICIT`.
//   2) Not perform pooling on depth(C) or batch(N).
bool RewritePool(const utils::MutableNodeView& node_view);
bool RewriteOneDnnPool(const utils::MutableNodeView& node_view);

// Rewrite only if there is _OneDnnMaxpool
// Only MaxPoolGrad requires the input from MaxPool. AvgPool doesn't have such
// input tensor.
bool RewriteMaxPoolGrad(const utils::MutableNodeView& node_view);

bool RewriteRandomUniform(const utils::MutableNodeView& node_view);

bool RewriteQuantize(const utils::MutableNodeView& node_view);

bool RewriteResize(const utils::MutableNodeView& node_view);

bool RewriteNativeCast(const utils::MutableNodeView& node_view);

bool RewriteWithBlockInput(const utils::MutableNodeView& node_view);

bool RewriteBinary(const utils::MutableNodeView& node_view);

bool RewriteCast(const utils::MutableNodeView& node_view);

// Only rewrite for s8 datatype which TF proper doesn't support
bool RewriteQuantizeReshape(const utils::MutableNodeView& node_view);

//////////////////////////////////////////////////////////////////////////
// Op-specific functions to copy attributes from old node to new node
//////////////////////////////////////////////////////////////////////////

void CopyAttrsCast(const utils::MutableNodeView* orig_node_view,
                   NodeDef* new_node);

// Generic function to copy all attributes from original node to target.
// graph_view is needed to get information from input node of orig_node
void CopyAttrsAll(const utils::MutableNodeView* orig_node_view,
                  NodeDef* new_node);

// Generic function to copy all attributes and check if filter is const.
void CopyAttrsAllCheckConstFilter(const utils::MutableNodeView* orig_node_view,
                                  NodeDef* new_node);

void CopyAttrsForTensorArray(const utils::MutableNodeView* orig_node_view,
                             NodeDef* new_node);

// Function to copy attributes of OneDnnGraph
void CopyAttrsOneDnnGraph(const utils::MutableNodeView* orig_node_view,
                          NodeDef* new_node);

void CopyAttrsQuantizedConv2D(const utils::MutableNodeView* orig_node_view,
                              NodeDef* new_node);

void CopyAttrsQuantizedMatMul(const utils::MutableNodeView* orig_node_view,
                              NodeDef* new_node);

void CopyAttrsQuantize(const utils::MutableNodeView* orig_node_view,
                       NodeDef* new_node);

//////////////////////////////////////////////////////////////////////////
// Helper function to handle layout process
//////////////////////////////////////////////////////////////////////////

// Sub function to copy attrs from original node to new node.
void CopyAllAttrs(const NodeDef& orig_node, NodeDef* new_node);

OpDef GetOpDef(const NodeDef& node_def);

// Check and set filter attribute
void CheckConstFilter(const utils::MutableNodeView* node_view,
                      const std::unordered_set<string>& nodes_to_preserve);

void SetConstFilterAttr(const utils::MutableNodeView* orig_node_view,
                        NodeDef* new_node,
                        const std::unordered_set<string>& nodes_to_preserve);

}  // namespace graph
}  // namespace itex

#endif  // ITEX_CORE_GRAPH_UTILS_LAYOUT_UTILS_H_
