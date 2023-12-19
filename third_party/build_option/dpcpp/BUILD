licenses(["notice"])

load("@intel_extension_for_tensorflow//itex:itex.bzl", "cc_library", "itex_xpu_library")

config_setting(
    name = "build_with_dpcpp",
    define_values = {
        "build_with_dpcpp": "true",
    },
    visibility = ["//visibility:public"],
)

itex_xpu_library(
    name = "itex_gpu_header",
    srcs = ["runtime/itex_gpu_runtime.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "eigen_itex_gpu_header",
    srcs = ["runtime/eigen_itex_gpu_runtime.h"],
    visibility = ["//visibility:public"],
    deps = [":itex_gpu_header"],
)
