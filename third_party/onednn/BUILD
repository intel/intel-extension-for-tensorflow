load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")

exports_files(["LICENSE"])

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

config_setting(
    name = "build_with_onednn",
    define_values = {
        "build_with_onednn": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_tbb",
    define_values = {
        "build_with_tbb": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "onednn_v3_and_gpu",
    define_values = {
        "device": "gpu",
        "onednn_version": "3",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "onednn_v2_and_gpu",
    define_values = {
        "device": "gpu",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "onednn_v3_and_cpu",
    define_values = {
        "device": "cpu",
        "onednn_version": "3",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "onednn_v2_and_cpu",
    define_values = {
        "device": "cpu",
    },
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)

cc_library(
    name = "intel_binary_blob",
    visibility = ["//visibility:public"],
    deps = select({
        "@intel_extension_for_tensorflow//itex:linux_x86_64": [
            "@onednn_cpu//:onednn_libs_linux",
        ],
        "//conditions:default": [],
    }) + select({
        ":build_with_tbb": ["@oneTBB//:tbb"],
        "//conditions:default": [],
    }),
)

py_binary(
    name = "gen_gpu_kernel_list",
    srcs = ["gen_gpu_kernel_list.py"],
    visibility = [
        "@onednn_gpu//:__subpackages__",
    ],
)

py_binary(
    name = "gen_onednn_version",
    srcs = ["gen_onednn_version.py"],
    visibility = [
        "@onednn_cpu//:__subpackages__",
        "@onednn_gpu//:__subpackages__",
    ],
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
