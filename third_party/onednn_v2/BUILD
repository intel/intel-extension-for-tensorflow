load("@bazel_skylib//lib:selects.bzl", "selects")
load("@intel_extension_for_tensorflow//itex:itex.bzl", "cc_library")

exports_files(["LICENSE"])

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

config_setting(
    name = "build_with_tbb",
    define_values = {
        "build_with_tbb": "true",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "intel_binary_blob",
    visibility = ["//visibility:public"],
    deps = select({
        "@intel_extension_for_tensorflow//itex:linux_x86_64": [
            "@onednn_cpu_v2//:onednn_libs_linux",
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
        "@onednn_gpu_v2//:__subpackages__",
    ],
)

py_binary(
    name = "gen_onednn_version",
    srcs = ["gen_onednn_version.py"],
    visibility = [
        "@onednn_cpu_v2//:__subpackages__",
        "@onednn_gpu_v2//:__subpackages__",
    ],
)
