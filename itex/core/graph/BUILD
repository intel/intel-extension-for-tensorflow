load("//itex:itex.bzl", "cc_library")
load("@local_config_dpcpp//dpcpp:build_defs.bzl", "if_dpcpp")

cc_library(
    name = "xpu_optimizer",
    srcs = ["xpu_optimizer.cc"],
    hdrs = ["xpu_optimizer.h"],
    defines = select({
        "//third_party/onednn:build_with_onednn_graph": ["ITEX_ONEDNN_GRAPH"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":optimizer_config_hdr",
        "//itex/core/devices:xpu_device_util",
        "//itex/core/graph/auto_mixed_precision",
        "//itex/core/graph/generic_layout_optimizer",
        "//itex/core/graph/memory_opt_pass",
        "//itex/core/graph/native_layout",
        "//itex/core/graph/onednn_layout",
        "//itex/core/graph/remapper",
    ] + select({
        "//third_party/onednn:build_with_onednn_graph": ["//itex/core/graph/onednn_graph"],
        "//conditions:default": [],
    }) + if_dpcpp(
        # TFG should be disabled when building with CPU, otherwise it will introduce llvm symbol conflict.
        # TFG depends on llvm-15, while CPU graph compiler needs llvm-13.
        [
            # "//itex/core/graph/tfg_optimizer_hook",
        ],
    ),
    alwayslink = True,
)

cc_library(
    name = "xpu_graph",
    srcs = ["xpu_graph.cc"],
    hdrs = ["xpu_graph.h"],
    textual_hdrs = ["//itex/core:itex_version_generator"],
    visibility = ["//visibility:public"],
    deps = [
        ":config_util_hdr",
        ":optimizer_config",
        "//itex/core/devices:xpu_device_util",
        "//itex/core/graph:xpu_optimizer",
        "//itex/core/utils:hw_info_hdr",
    ],
    alwayslink = True,
)

cc_library(
    name = "optimizer_config",
    srcs = ["optimizer_config.cc"],
    hdrs = ["optimizer_config.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":config_util_hdr",
        "//itex/core/utils:common_utils",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "optimizer_config_hdr",
    hdrs = ["optimizer_config.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/eigen3",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "config_util",
    srcs = ["config_util.cc"],
    hdrs = ["config_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core:protos_all_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "config_util_hdr",
    hdrs = ["config_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core:protos_all_cc",
    ],
)
