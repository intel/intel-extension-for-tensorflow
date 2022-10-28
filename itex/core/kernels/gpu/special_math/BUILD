load("//itex:itex.bzl", "itex_xpu_library", "tf_copts")

# List all kernels here.
itex_xpu_library(
    name = "special_math_op",
    srcs = [
        "special_math_op_bessel.cc",
        "special_math_op_dawsn.cc",
        "special_math_op_expint.cc",
        "special_math_op_fresnel_cos.cc",
        "special_math_op_fresnel_sin.cc",
        "special_math_op_spence.cc",
    ],
    hdrs = ["special_math_op_misc_impl.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:cwise_ops_lib",
    ],
    alwayslink = True,
)
