load("//itex:itex.bzl", "tf_copts")

cc_library(
    name = "onednn_array_ops",
    srcs = ["onednn_array_ops.cc"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/ops:shape_inference_fns",
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:status",
    ],
)

cc_library(
    name = "onednn_nn_ops",
    srcs = ["onednn_nn_ops.cc"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/ops:shape_inference_fns",
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:padding",
        "//itex/core/ops/utils:status",
    ],
)

cc_library(
    name = "onednn_math_ops",
    srcs = ["onednn_math_ops.cc"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/ops:shape_inference_fns",
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:status",
    ],
)

cc_library(
    name = "onednn_ops",
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":onednn_array_ops",
        ":onednn_math_ops",
        ":onednn_nn_ops",
    ],
)
