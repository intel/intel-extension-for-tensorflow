load("//itex:itex.bzl", "itex_xpu_library", "tf_copts")

itex_xpu_library(
    name = "xpu_kernel",
    srcs = ["xpu_kernel.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":libitex_common",
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/common:no_ops",
        "@local_config_tf//:tf_header_lib",
    ] + select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//itex/core/kernels/gpu:gpu_kernel_impl",
        ],
        "//conditions:default": [
            "//itex/core/kernels/cpu:cpu_kernel_impl",
        ],
    }),
    alwayslink = True,
)

cc_import(
    name = "libitex_common",
    hdrs = [
        "common.h",
        "//itex/core/ops:core_ops_hdrs",
    ],
    shared_library = ":libitex_common.so",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libitex_common.so",
    srcs = [
        "common.cc",
        "common.h",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/ops:op_impl",
    ],
)
