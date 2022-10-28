load("//itex:itex.bzl", "tf_copts")

cc_library(
    name = "onednn_graph_op",
    srcs = ["onednn_graph_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/utils/onednn:onednn_graph_util",
    ],
    alwayslink = True,
)
