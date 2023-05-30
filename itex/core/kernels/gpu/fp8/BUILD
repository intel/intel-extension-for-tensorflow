load("//itex:itex.bzl", "itex_xpu_library", "tf_copts")

itex_xpu_library(
    name = "fp8_quantize_op",
    srcs = ["fp8_quantize_op.cc"],
    hdrs = [
        "fp8_quantize_gpu.h",
        "utils.h",
        "vectorized_pointwise.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fp8_op",
    visibility = ["//visibility:public"],
    deps = [
        ":fp8_quantize_op",
    ],
)
