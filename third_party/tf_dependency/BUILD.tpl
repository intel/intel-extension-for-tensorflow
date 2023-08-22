package(default_visibility = ["//visibility:public"])

load(
    "@intel_extension_for_tensorflow//third_party:common.bzl",
    "template_rule",
)
load("@intel_extension_for_tensorflow//itex/core/utils:build_config.bzl", "cc_proto")
load("@intel_extension_for_tensorflow//itex:itex.bzl", "cc_library")

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

template_rule(
    name = "types_plugin",
    src = "include/tensorflow/core/framework/types.proto",
    out = "include/protos/types.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
    },
)

template_rule(
    name = "tensor_shape_plugin",
    src = "include/tensorflow/core/framework/tensor_shape.proto",
    out = "include/protos/tensor_shape.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
    },
)

template_rule(
    name = "versions_plugin",
    src = "include/tensorflow/core/framework/versions.proto",
    out = "include/protos/versions.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
    },
)

template_rule(
    name = "cost_graph_plugin",
    src = "include/tensorflow/core/framework/cost_graph.proto",
    out = "include/protos/cost_graph.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/tensor_shape.proto": "tensor_shape.proto",
        "tensorflow/core/framework/types.proto": "types.proto",
    },
)

template_rule(
    name = "resource_handle_plugin",
    src = "include/tensorflow/core/framework/resource_handle.proto",
    out = "include/protos/resource_handle.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/tensor_shape.proto": "tensor_shape.proto",
        "tensorflow/core/framework/types.proto": "types.proto",
    },
)

template_rule(
    name = "tensor_plugin",
    src = "include/tensorflow/core/framework/tensor.proto",
    out = "include/protos/tensor.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/tensor_shape.proto": "tensor_shape.proto",
        "tensorflow/core/framework/types.proto": "types.proto",
        "tensorflow/core/framework/resource_handle.proto": "resource_handle.proto",
    },
)

template_rule(
    name = "attr_value_plugin",
    src = "include/tensorflow/core/framework/attr_value.proto",
    out = "include/protos/attr_value.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/tensor.proto": "tensor.proto",
        "tensorflow/core/framework/tensor_shape.proto": "tensor_shape.proto",
        "tensorflow/core/framework/types.proto": "types.proto",
    },
)

template_rule(
    name = "full_type_plugin",
    src = "include/tensorflow/core/framework/full_type.proto",
    out = "include/protos/full_type.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
    },
)

template_rule(
    name = "node_def_plugin",
    src = "include/tensorflow/core/framework/node_def.proto",
    out = "include/protos/node_def.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/attr_value.proto": "attr_value.proto",
        "tensorflow/core/framework/full_type.proto": "full_type.proto",
    },
)

template_rule(
    name = "graph_debug_info_plugin",
    src = select({
        "@intel_extension_for_tensorflow//itex:tf_version_2_12": "include/tensorflow/core/protobuf/graph_debug_info.proto",
        "//conditions:default": "include/tensorflow/core/framework/graph_debug_info.proto",
    }),
    out = "include/protos/graph_debug_info.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
    },
)

template_rule(
    name = "op_def_plugin",
    src = "include/tensorflow/core/framework/op_def.proto",
    out = "include/protos/op_def.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/attr_value.proto": "attr_value.proto",
        "tensorflow/core/framework/full_type.proto": "full_type.proto",
        "tensorflow/core/framework/resource_handle.proto": "resource_handle.proto",
        "tensorflow/core/framework/types.proto": "types.proto",
    },
)

template_rule(
    name = "kernel_def_plugin",
    src = "include/tensorflow/core/framework/kernel_def.proto",
    out = "include/protos/kernel_def.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/attr_value.proto": "attr_value.proto",
    },
)

template_rule(
    name = "function_plugin",
    src = "include/tensorflow/core/framework/function.proto",
    out = "include/protos/function.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/attr_value.proto": "attr_value.proto",
        "tensorflow/core/framework/node_def.proto": "node_def.proto",
        "tensorflow/core/framework/op_def.proto": "op_def.proto",
    },
)

template_rule(
    name = "graph_plugin",
    src = "include/tensorflow/core/framework/graph.proto",
    out = "include/protos/graph.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/function.proto": "function.proto",
        "tensorflow/core/framework/graph_debug_info.proto": "graph_debug_info.proto",
        "tensorflow/core/framework/node_def.proto": "node_def.proto",
        "tensorflow/core/framework/versions.proto": "versions.proto",
    },
)

template_rule(
    name = "device_properties_plugin",
    src = "include/tensorflow/core/protobuf/device_properties.proto",
    out = "include/protos/device_properties.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
    },
)

template_rule(
    name = "op_performance_data_plugin",
    src = "include/tensorflow/core/grappler/costs/op_performance_data.proto",
    out = "include/protos/op_performance_data.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/tensor.proto": "tensor.proto",
        "tensorflow/core/framework/tensor_shape.proto": "tensor_shape.proto",
        "tensorflow/core/framework/types.proto": "types.proto",
        "tensorflow/core/framework/attr_value.proto": "attr_value.proto",
        "tensorflow/core/protobuf/device_properties.proto": "device_properties.proto",
    },
)

template_rule(
    name = "api_def_plugin",
    src = "include/tensorflow/core/framework/api_def.proto",
    out = "include/protos/api_def.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/attr_value.proto": "attr_value.proto",
    },
)

template_rule(
    name = "xplane_plugin",
    src = "include/tensorflow/tsl/profiler/protobuf/xplane.proto",
    out = "include/protos/xplane.proto",
    substitutions = {
        "package tensorflow.profiler;": "package itex;",
    },
)

template_rule(
    name = "summary_plugin",
    src = "include/tensorflow/core/framework/summary.proto",
    out = "include/protos/summary.proto",
    substitutions = {
        "package tensorflow;": "package itex;",
        "tensorflow/core/framework/tensor.proto": "tensor.proto",
        "import public \"tensorflow/tsl/protobuf/histogram.proto\";": "",
        "HistogramProto histo = 5;": "",
    },
)

cc_proto(
    name = "types",
    src = "types.proto",
)

cc_proto(
    name = "tensor_shape",
    src = "tensor_shape.proto",
)

cc_proto(
    name = "versions",
    src = "versions.proto",
)

cc_proto(
    name = "cost_graph",
    src = "cost_graph.proto",
    deps = [
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

cc_proto(
    name = "resource_handle",
    src = "resource_handle.proto",
    deps = [
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

cc_proto(
    name = "tensor",
    src = "tensor.proto",
    deps = [
        ":resource_handle_proto",
        ":types_proto",
        ":tensor_shape_proto",
    ],
)

cc_proto(
    name = "attr_value",
    src = "attr_value.proto",
    deps = [
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

cc_proto(
    name = "full_type",
    src = "full_type.proto",
)

cc_proto(
    name = "node_def",
    src = "node_def.proto",
    deps = [
        ":attr_value_proto",
        ":full_type_proto",
    ],
)

cc_proto(
    name = "graph_debug_info",
    src = "graph_debug_info.proto",
)

cc_proto(
    name = "op_def",
    src = "op_def.proto",
    deps = [
        ":attr_value_proto",
        ":full_type_proto",
        ":resource_handle_proto",
        ":types_proto",
    ],
)

cc_proto(
    name = "kernel_def",
    src = "kernel_def.proto",
    deps = [
        ":attr_value_proto",
    ],
)

cc_proto(
    name = "function",
    src = "function.proto",
    deps = [
        ":attr_value_proto",
        ":node_def_proto",
        ":op_def_proto",
    ],
)

cc_proto(
    name = "graph",
    src = "graph.proto",
    deps = [
        ":function_proto",
        ":graph_debug_info_proto",
        ":node_def_proto",
        ":versions_proto",
    ],
)

cc_proto(
    name = "device_properties",
    src = "device_properties.proto",
)

cc_proto(
    name = "op_performance_data",
    src = "op_performance_data.proto",
    deps = [
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
        ":attr_value_proto",
        ":device_properties_proto",
    ],
)

cc_proto(
    name = "api_def",
    src = "api_def.proto",
    deps = [
        ":attr_value_proto",
    ],
)

cc_proto(
    name = "xplane",
    src = "xplane.proto",
)

cc_proto(
    name = "summary",
    src = "summary.proto",
    deps = [
        ":tensor_proto",
    ],
)

cc_library(
    name = "protos_all",
    visibility = ["//visibility:public"],
    deps = [
        ":api_def_proto",
        ":graph_debug_info_proto",
        ":graph_proto",
        ":kernel_def_proto",
        ":op_performance_data_proto",
        ":summary_proto",
        ":tensor_proto",
        ":xplane_proto",
    ],
)

cc_library(
    name = "_pywrap_tensorflow_internal",
    srcs = ["%{TF_SHARED_LIBRARY_NAME}"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "jax_internal",
    srcs = ["%{JAX_SHARED_LIBRARY_NAME}"],
    visibility = ["//visibility:public"],
)

%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}
%{JAX_SHARED_LIBRARY_GENRULE}
