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
    name = "build_with_onednn_graph",
    define_values = {
        "build_with_onednn_graph": "true",
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
    name = "build_with_threadpool",
    define_values = {
        "build_with_threadpool": "true",
    },
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "cc_build_with_threadpool",
    match_all = [
        ":build_with_threadpool",
        "//itex:cpu_cc_build",
    ],
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

cc_binary(
    name = "onednn_cpu_so",
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../_solib_$(TARGET_CPU)/_U@onednn_Ucpu_S_S_Conednn_Ulibs_Ulinux___Uexternal_Sllvm_Uopenmp",
        "-Wl,-rpath,$$ORIGIN/../_solib_local/_U@onednn_Ucpu_S_S_Conednn_Ulibs_Ulinux___Uexternal_Sllvm_Uopenmp",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":intel_binary_blob",
    ] + select({
        "//third_party/onednn:build_with_onednn_graph": ["@onednn_cpu//:onednn_graph_cpu_lib"],
        "//conditions:default": ["@onednn_cpu//:onednn_cpu_lib"],
    }),
)

cc_binary(
    name = "onednn_cpu_eigen_so",
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../_solib_$(TARGET_CPU)/_U@onednn_Ucpu_S_S_Conednn_Ulibs_Ulinux___Uexternal_Sllvm_Uopenmp",
        "-Wl,-rpath,$$ORIGIN/../_solib_local/_U@onednn_Ucpu_S_S_Conednn_Ulibs_Ulinux___Uexternal_Sllvm_Uopenmp",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":intel_binary_blob",
    ] + select({
        "//third_party/onednn:build_with_onednn_graph": ["@onednn_cpu_eigen//:onednn_graph_cpu_lib"],
        "//conditions:default": ["@onednn_cpu_eigen//:onednn_cpu_lib"],
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
        "@onednn_cpu_eigen//:__subpackages__",
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
)
