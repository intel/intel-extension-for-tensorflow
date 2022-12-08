cc_library(
    name = "onednn_graph",
    srcs = ["onednn_graph.cc"],
    hdrs = ["onednn_graph.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/graph:optimizer_config",
        "//itex/core/graph/utils:function",
        "//itex/core/graph/utils:graph_common_utils",
        "//itex/core/graph/utils:graph_properties",
        "//itex/core/graph/utils:graph_view",
        "//itex/core/graph/utils:grappler_item",
        "//itex/core/graph/utils:node_type_attr_map",
        "//itex/core/utils:common_utils",
        "//itex/core/utils/onednn:onednn_graph_util",
    ],
    alwayslink = True,
)
