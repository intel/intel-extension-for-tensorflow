load("//itex:itex.bzl", "cc_library", "tf_copts")

filegroup(
    name = "core_ops_hdrs",
    srcs = [
        "op_init.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "array_ops",
    srcs = ["array_ops.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:status",
    ],
    alwayslink = True,
)

cc_library(
    name = "rnn_ops",
    srcs = ["rnn_ops.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:status",
    ],
    alwayslink = True,
)

cc_library(
    name = "math_ops",
    srcs = ["math_ops.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:status",
    ],
    alwayslink = True,
)

cc_library(
    name = "nn_ops",
    srcs = ["nn_ops.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:padding",
        "//itex/core/ops/utils:status",
        "//itex/core/utils:tf_version",
        "@local_config_tf//:protos_all",
    ],
    alwayslink = True,
)

cc_library(
    name = "onednn_graph_op",
    srcs = ["onednn_graph_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:status",
    ],
    alwayslink = True,
)

cc_library(
    name = "training_ops",
    srcs = ["training_ops.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:status",
    ],
    alwayslink = True,
)

OPS = [
    ":array_ops",
    ":math_ops",
    ":rnn_ops",
    ":nn_ops",
    ":onednn_graph_op",
    ":training_ops",
]

cc_library(
    name = "shape_inference_fns",
    srcs = ["shape_inference_fns.cc"],
    hdrs = [
        "shape_inference_fns.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/ops/utils:logging",
        "//itex/core/ops/utils:status",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings:cord",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "op_impl",
    srcs = ["op_init.cc"],
    hdrs = [
        "op_init.h",
    ],
    copts = ["-fvisibility=hidden"],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/ops/onednn:onednn_ops",
    ] + OPS,
    alwayslink = True,
)
