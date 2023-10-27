load("//itex:itex.bzl", "cc_library", "if_cc_threadpool_build")
load("//third_party/onednn:build_defs.bzl", "onednn_deps", "onednn_graph_deps")

cc_library(
    name = "onednn_util",
    srcs = [
        "onednn_post_op_util.cc",
        "onednn_util.cc",
    ],
    hdrs = [
        "mkl_threadpool.h",
        "onednn_post_op_util.h",
        "onednn_util.h",
        "//itex/core/wrapper:itex_cpu_wrapper_hdr",
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
    defines = select({
        "//third_party/onednn:build_with_onednn_graph": ["ITEX_ONEDNN_GRAPH"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = select({
        "//third_party/onednn:build_with_onednn_graph": [":onednn_graph_util"],
        "//conditions:default": [],
    }) + [
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
