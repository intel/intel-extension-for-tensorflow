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

#include "itex/core/graph/utils/graph_properties.h"

#include "itex/core/utils/errors.h"
#include "itex/core/utils/tf_buffer.h"
#include "protos/op_performance_data.pb.h"

namespace itex {
namespace graph {

GraphProperties::GraphProperties(const GrapplerItem& item) {
  graph_prop_ = TF_NewGraphProperties(item.GetTfGrapplerItem());
}
GraphProperties::~GraphProperties() { TF_DeleteGraphProperties(graph_prop_); }

Status GraphProperties::InferStatically(bool assume_valid_feeds,
                                        bool aggressive_shape_inference,
                                        bool include_input_tensor_values,
                                        bool include_output_tensor_values) {
  TF_Status* tf_status = TF_NewStatus();
  TF_InferStatically(graph_prop_, static_cast<TF_Bool>(assume_valid_feeds),
                     static_cast<TF_Bool>(aggressive_shape_inference),
                     static_cast<TF_Bool>(include_input_tensor_values),
                     static_cast<TF_Bool>(include_output_tensor_values),
                     tf_status);
  Status status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  return status;
}

typedef void (*GetPropertiesListSizePtr)(TF_GraphProperties* graph_properties,
                                         const char* name, int* num_values,
                                         TF_Status* status);

typedef void (*GetPropertiesListPtr)(TF_GraphProperties* graph_properties,
                                     const char* name, TF_Buffer** properties,
                                     int num_values, TF_Status* status);

static Status GetProperties(TF_GraphProperties* graph_prop,
                            const string& node_name,
                            std::vector<OpInfo_TensorProperties>* props,
                            GetPropertiesListSizePtr get_properties_list_size,
                            GetPropertiesListPtr get_properties_list) {
  TF_Status* tf_status = TF_NewStatus();
  int num_props = 0;

  get_properties_list_size(graph_prop, node_name.c_str(), &num_props,
                           tf_status);
  props->resize(num_props);

  TF_Buffer* props_buf[num_props];
  for (int i = 0; i < num_props; ++i) props_buf[i] = TF_NewBuffer();

  get_properties_list(graph_prop, node_name.c_str(), props_buf, num_props,
                      tf_status);
  for (int i = 0; i < num_props; ++i) {
    TF_ABORT_IF_ERROR(BufferToMessage(props_buf[i], props->at(i)));
    TF_DeleteBuffer(props_buf[i]);
  }
  const Status status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  return status;
}

Status GraphProperties::GetInputProperties(
    const string& node_name,
    std::vector<OpInfo_TensorProperties>* input_props) const {
  return GetProperties(graph_prop_, node_name, input_props,
                       TF_GetInputPropertiesListSize,
                       TF_GetInputPropertiesList);
}

Status GraphProperties::GetOutputProperties(
    const string& node_name,
    std::vector<OpInfo_TensorProperties>* output_props) const {
  return GetProperties(graph_prop_, node_name, output_props,
                       TF_GetOutputPropertiesListSize,
                       TF_GetOutputPropertiesList);
}

}  // namespace graph
}  // namespace itex
