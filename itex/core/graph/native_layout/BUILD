load(
    "//itex/core/utils:build_config.bzl",
    "tf_protobuf_deps",
)
load("//itex:itex.bzl", "cc_library")

cc_library(
    name = "native_layout",
    srcs = ["native_layout.cc"],
    hdrs = ["native_layout.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/graph/utils:grappler_item",
        "//itex/core/graph/utils:layout_utils",
        "//itex/core/graph/utils:node_type_attr_map",
        "//third_party/build_option/sycl:itex_gpu_header",
    ] + tf_protobuf_deps(),
    alwayslink = True,
)
