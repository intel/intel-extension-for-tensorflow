package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "graph_view",
    srcs = ["graph_view.cc"],
    hdrs = ["graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core:protos_all_cc",
        "//itex/core/graph/utils",
        "//itex/core/graph/utils:op_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "mutable_graph_view",
    srcs = [
        "mutable_graph_view.cc",
    ],
    hdrs = ["mutable_graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        "//itex/core:protos_all_cc",
        "//itex/core/graph/utils",
        "//itex/core/graph/utils:op_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
