load("//itex:itex.bzl", "cc_library", "if_using_nextpluggable_device", "tf_copts")

cc_library(
    name = "onednn_graph_op",
    srcs = ["onednn_graph_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/utils/onednn:onednn_graph_util",
    ] + if_using_nextpluggable_device([
        "//itex/core/utils:libintel_xla",
    ]),
    alwayslink = True,
)
