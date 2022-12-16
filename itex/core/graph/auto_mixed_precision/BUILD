cc_library(
    name = "auto_mixed_precision",
    srcs = ["auto_mixed_precision.cc"],
    hdrs = [
        "auto_mixed_precision.h",
        "auto_mixed_precision_lists.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/graph:optimizer_config",
        "//itex/core/graph/graph_view:mutable_graph_view",
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/graph/utils:grappler_item",
        "//itex/core/graph/utils:node_type_attr_map",
        "//itex/core/graph/utils:symbolic_shapes",
        "//itex/core/utils:common_utils",
    ],
    alwayslink = True,
)
