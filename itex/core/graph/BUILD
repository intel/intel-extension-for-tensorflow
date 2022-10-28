cc_library(
    name = "xpu_optimizer",
    srcs = ["xpu_optimizer.cc"],
    hdrs = ["xpu_optimizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":optimizer_config_hdr",
        "//itex/core/devices:xpu_device_util",
        "//itex/core/graph/auto_mixed_precision",
        "//itex/core/graph/memory_opt_pass",
        "//itex/core/graph/native_layout",
        "//itex/core/graph/onednn_graph",
        "//itex/core/graph/onednn_layout",
        "//itex/core/graph/remapper",
    ],
    alwayslink = True,
)

cc_library(
    name = "xpu_graph",
    srcs = ["xpu_graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":optimizer_config",
        "//itex/core/devices:xpu_device_util",
        "//itex/core/graph:xpu_optimizer",
    ],
    alwayslink = True,
)

cc_library(
    name = "optimizer_config",
    srcs = ["optimizer_config.cc"],
    hdrs = ["optimizer_config.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:device_backend_util_hdr",
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
