load("//itex/core/utils:build_config.bzl", "tf_proto_library")

tf_proto_library(
    name = "error_codes",
    srcs = ["error_codes.proto"],
)

tf_proto_library(
    name = "versions",
    srcs = ["versions.proto"],
)

tf_proto_library(
    name = "types",
    srcs = ["types.proto"],
)

tf_proto_library(
    name = "tensor_shape",
    srcs = ["tensor_shape.proto"],
)

tf_proto_library(
    name = "cost_graph",
    srcs = ["cost_graph.proto"],
    protodeps = [
        ":types",
        ":tensor_shape",
    ],
)

tf_proto_library(
    name = "resource_handle",
    srcs = ["resource_handle.proto"],
    protodeps = [
        ":types",
        ":tensor_shape",
    ],
)

tf_proto_library(
    name = "tensor",
    srcs = ["tensor.proto"],
    protodeps = [
        ":resource_handle",
    ],
)

tf_proto_library(
    name = "attr_value",
    srcs = ["attr_value.proto"],
    protodeps = [
        ":tensor",
    ],
)

tf_proto_library(
    name = "full_type",
    srcs = ["full_type.proto"],
)

tf_proto_library(
    name = "node_def",
    srcs = ["node_def.proto"],
    protodeps = [
        ":attr_value",
        ":full_type",
    ],
)

tf_proto_library(
    name = "graph_debug_info",
    srcs = ["graph_debug_info.proto"],
)

tf_proto_library(
    name = "op_def",
    srcs = ["op_def.proto"],
    protodeps = [
        ":attr_value",
        ":full_type",
    ],
)

tf_proto_library(
    name = "kernel_def",
    srcs = ["kernel_def.proto"],
    protodeps = [
        ":attr_value",
    ],
)

tf_proto_library(
    name = "function",
    srcs = ["function.proto"],
    protodeps = [
        ":node_def",
        ":op_def",
    ],
)

tf_proto_library(
    name = "graph",
    srcs = ["graph.proto"],
    protodeps = [
        ":function",
        ":node_def",
        ":versions",
    ],
)

tf_proto_library(
    name = "device_properties",
    srcs = ["device_properties.proto"],
)

tf_proto_library(
    name = "op_performance_data",
    srcs = ["op_performance_data.proto"],
    protodeps = [
        ":attr_value",
        ":device_properties",
    ],
)

tf_proto_library(
    name = "api_def",
    srcs = ["api_def.proto"],
    protodeps = [
        ":attr_value",
    ],
)

tf_proto_library(
    name = "summary",
    srcs = ["summary.proto"],
    protodeps = [
        ":tensor",
    ],
)

tf_proto_library(
    name = "tf_protos_all",
    protodeps = [
        ":error_codes",
        ":api_def",
        ":graph",
        # ":xplane_proto",
        ":op_performance_data",
        ":kernel_def",
        ":graph_debug_info",
        ":tensor",
        ":summary",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "xla_data_proto",
    srcs = ["xla_data.proto"],
)

tf_proto_library(
    name = "xla_proto",
    srcs = ["xla.proto"],
    protodeps = [
        ":xla_data_proto",
        ":hlo_proto",
    ],
)

tf_proto_library(
    name = "hlo_proto",
    srcs = ["hlo.proto"],
    protodeps = [":xla_data_proto"],
)

tf_proto_library(
    name = "hlo_profile_printer_data",
    srcs = ["hlo_profile_printer_data.proto"],
)

tf_proto_library(
    name = "hlo_execution_profile_data",
    srcs = ["hlo_execution_profile_data.proto"],
    protodeps = [":hlo_profile_printer_data"],
)

tf_proto_library(
    name = "backend_configs",
    srcs = ["backend_configs.proto"],
    protodeps = [
        ":xla_data_proto",
    ],
)

tf_proto_library(
    name = "compile_options_proto",
    srcs = ["compile_options.proto"],
    protodeps = [
        ":xla_data_proto",
        ":xla_proto",
    ],
)

tf_proto_library(
    name = "xla_protos_all",
    protodeps = [
        ":compile_options_proto",
        ":hlo_execution_profile_data",
        ":backend_configs",
    ],
    visibility = ["//visibility:public"],
)
