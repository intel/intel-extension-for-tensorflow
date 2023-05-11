package(
    licenses = ["notice"],  # Apache 2.0
)

load("//itex:itex.bzl", "cc_library", "itex_xpu_library", "tf_copts")

cc_library(
    name = "qmatmul_op",
    srcs = ["qmatmul_op.cc"],
    hdrs = [
        "matmul_common.h",
        "//itex/core/kernels/common:cast_hdrs",
        "//itex/core/kernels/common:matmul_hdrs",
        "//itex/core/kernels/common:noop_hdrs",
        "//itex/core/kernels/onednn/block:quantized_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "quantized_fused_matmul",
    srcs = ["quantized_fused_matmul.cc"],
    hdrs = [
        "matmul_common.h",
        "//itex/core/kernels/common:cast_hdrs",
        "//itex/core/kernels/common:matmul_hdrs",
        "//itex/core/kernels/common:noop_hdrs",
        "//itex/core/kernels/onednn/block:quantized_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/devices:xpu_device_util",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "legacy_kernels",
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":qmatmul_op",
        ":quantized_fused_matmul",
    ],
    alwayslink = True,
)
