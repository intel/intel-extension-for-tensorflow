load("//itex:itex.bzl", "cc_library", "tf_copts")

cc_library(
    name = "logging",
    srcs = [
        "env_time.cc",
        "logging.cc",
    ],
    hdrs = [
        "env_time.h",
        "integral_types.h",
        "logging.h",
        "macros.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//itex/core/ops:__subpackages__"],
    deps = [
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//itex/core/ops:__subpackages__"],
    deps = [
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)
