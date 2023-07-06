load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")

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
    name = "build_with_graph_compiler_flag",
    define_values = {
        "build_with_graph_compiler": "true",
    },
)

selects.config_setting_group(
    name = "build_with_graph_compiler",
    match_all = [
        "@intel_extension_for_tensorflow//itex:cpu_build",
        ":build_with_graph_compiler_flag",
    ],
    visibility = ["//visibility:public"],
)
