load("//itex:itex.bzl", "cc_binary", "cc_library", "if_using_nextpluggable_device", "itex_xpu_library", "tf_copts")
load("@itex_local_config_sycl//sycl:build_defs.bzl", "if_sycl", "if_xetla")

XPU_KERNEL_DEPS = [
    "//itex/core/devices:device_backend_util_hdr",
    "//itex/core/kernels/common:no_ops",
    "@local_config_python//:python_headers",
    "@local_config_tf//:tf_header_lib",
] + if_sycl(
    [
        "//itex/core/kernels/gpu:gpu_kernel_impl",
    ],
    [
        "//itex/core/kernels/cpu:cpu_kernel_impl",
    ],
)

itex_xpu_library(
    name = "xpu_kernel",
    srcs = ["xpu_kernel.cc"],
    hdrs = ["xpu_kernel.h"],
    copts = if_using_nextpluggable_device(
        ["-DUSING_NEXTPLUGGABLE_DEVICE"],
    ),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":libitex_common",
    ] + XPU_KERNEL_DEPS,
    alwayslink = True,
)

itex_xpu_library(
    name = "xpu_kernel_cc",
    srcs = ["xpu_kernel.cc"],
    copts = if_using_nextpluggable_device(
        ["-DUSING_NEXTPLUGGABLE_DEVICE"],
    ),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":itex_common_cc",
    ] + XPU_KERNEL_DEPS,
    alwayslink = True,
)

cc_import(
    name = "libitex_common",
    hdrs = [
        "common.h",
        "//itex/core/ops:core_ops_hdrs",
    ],
    shared_library = ":itex_common",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libitex_common.so",
    srcs = [
        "common.cc",
        "common.h",
    ],
    linkshared = True,
    set_target = "dummy",
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:device_backend_util",
        "//itex/core/graph:config_util",
        "//itex/core/ops:op_impl",
        "@local_config_tf//:_pywrap_tensorflow_internal",
    ],
)

cc_library(
    name = "itex_common_cc",
    srcs = [
        "common.cc",
        "common.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:device_backend_util",
        "//itex/core/graph:config_util",
        "//itex/core/ops:op_impl",
    ],
)
