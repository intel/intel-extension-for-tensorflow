load("//itex:itex.bzl", "cc_library")
load("//itex/core/utils:build_config.bzl", "tf_protobuf_deps")

cc_library(
    name = "check_const_filter",
    srcs = ["check_const_filter.cc"],
    hdrs = ["check_const_filter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/graph:optimizer_config",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/graph/utils:layout_utils",
    ],
)

cc_library(
    name = "weight_prepack",
    srcs = ["weight_prepack.cc"],
    hdrs = ["weight_prepack.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/graph/utils:grappler_item",
        "//itex/core/graph/utils:layout_utils",
    ] + tf_protobuf_deps(),
    alwayslink = True,
)

cc_library(
    name = "memory_opt_pass",
    srcs = ["memory_opt_pass.cc"],
    hdrs = ["memory_opt_pass.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":check_const_filter",
        ":weight_prepack",
        "//itex/core/devices:xpu_device_util",
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/graph/utils:grappler_item",
        "//itex/core/graph/utils:layout_utils",
        "//itex/core/graph/utils:node_type_attr_map",
    ] + tf_protobuf_deps(),
    alwayslink = True,
)
