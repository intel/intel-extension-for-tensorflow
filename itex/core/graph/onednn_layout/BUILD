load(
    "//itex/core/utils:build_config.bzl",
    "tf_protobuf_deps",
)

cc_library(
    name = "onednn_layout",
    srcs = ["onednn_layout.cc"],
    hdrs = ["onednn_layout.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/graph/utils:grappler_item",
        "//itex/core/graph/utils:layout_utils",
        "//itex/core/graph/utils:node_type_attr_map",
        "//itex/core/utils/onednn:onednn_util",
    ] + tf_protobuf_deps(),
    alwayslink = True,
)
