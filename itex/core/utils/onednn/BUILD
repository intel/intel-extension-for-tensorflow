load("//itex:itex.bzl", "cc_library")
load("//third_party/onednn:build_defs.bzl", "onednn_deps", "onednn_graph_deps")

cc_library(
    name = "onednn_util",
    srcs = [
        "onednn_post_op_util.cc",
        "onednn_util.cc",
    ],
    hdrs = [
        "onednn_post_op_util.h",
        "onednn_util.h",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
    ] + onednn_deps(),
)

cc_library(
    name = "onednn_layout_util",
    srcs = glob([
        "onednn_layout_util.cc",
    ]),
    hdrs = glob([
        "onednn_layout_util.h",
    ]),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = onednn_graph_deps() + [
        ":onednn_util",
        "//itex/core/utils:common_utils",
    ] + onednn_deps(),  #TODO3.0: LLGA and onednn use different branch.
    # onednn now contains part of llga (same file name, different code...).
    # use the same commit when avaliable.
)

cc_library(
    name = "onednn_graph_util",
    srcs = ["onednn_graph_util.cc"],
    hdrs = [
        "onednn_graph_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
        "//itex/core/utils:types",
        "//third_party/eigen3",
        "@eigen_archive//:eigen",
        "@local_config_tf//:protos_all",
    ] + onednn_graph_deps(),
    alwayslink = True,
)
