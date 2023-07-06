load("//itex:itex.bzl", "cc_library")

cc_library(
    name = "generic_layout_optimizer",
    srcs = ["generic_layout_optimizer.cc"],
    hdrs = [
        "generic_layout_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/graph/remapper",
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/graph/utils:grappler_item",
        "//itex/core/graph/utils:layout_utils",
        "//itex/core/graph/utils:node_type_attr_map",
        "//itex/core/utils/onednn:onednn_util",
    ],
    alwayslink = True,
)
