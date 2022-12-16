load("//itex:itex.bzl", "tf_copts")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "grappler_item",
    srcs = ["grappler_item.cc"],
    hdrs = [
        "grappler_item.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "graph_properties",
    srcs = ["graph_properties.cc"],
    hdrs = [
        "graph_properties.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":grappler_item",
        "//itex/core/utils:common_utils",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "symbolic_shapes",
    srcs = ["symbolic_shapes.cc"],
    hdrs = [
        "symbolic_shapes.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "graph_view_internal",
    hdrs = ["graph_view_internal.h"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "graph_view",
    srcs = ["graph_view.cc"],
    hdrs = ["graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view_internal",
        ":op_types",
    ],
)

cc_library(
    name = "graph_common_utils",
    srcs = ["graph_common_utils.cc"],
    hdrs = ["graph_common_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        ":op_types",
    ],
)

cc_library(
    name = "op_types",
    srcs = ["op_types.cc"],
    hdrs = ["op_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "node_type_attr_map",
    srcs = ["node_type_attr_map.cc"],
    hdrs = [
        "node_type_attr_map.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//itex/core/utils:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "pattern_utils",
    srcs = ["pattern_utils.cc"],
    hdrs = ["pattern_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "layout_utils",
    srcs = ["layout_utils.cc"],
    hdrs = ["layout_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/graph:optimizer_config",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/utils/onednn:onednn_util",
    ],
)
