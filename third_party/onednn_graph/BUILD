load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)

config_setting(
    name = "build_with_llga_debug",
    define_values = {
        "build_with_llga_debug": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_graph_compiler",
    define_values = {
        "build_with_graph_compiler": "true",
    },
    visibility = ["//visibility:public"],
)
