#load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
#load("//tensorflow/core/platform:build_config.bzl", "tf_stream_executor_deps")
#load("//tensorflow/compiler/xla/stream_executor:build_defs.bzl", "stream_executor_friends")
package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "platform",
    textual_hdrs = [
        "initialize.h",
        "logging.h",
        "platform.h",
        "port.h",
    ],
    deps = [
        "//itex/core/utils:common_utils",
    ],
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    deps = [
        ":platform",
    ],
)
