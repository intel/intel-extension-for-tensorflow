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
    name = "fp8_layernorm_op",
    srcs = ["fp8_layernorm_op.cc"],
    hdrs = [
        "fp8_layernorm_gpu.h",
        "utils.h",
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
    name = "fp8_gelu_op",
    srcs = ["fp8_gelu_op.cc"],
    hdrs = [
        "fp8_gelu_gpu.h",
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
    name = "fp8_quantize_fusion_op",
    srcs = ["fp8_quantize_fusion_op.cc"],
    hdrs = [
        "fp8_quantize_fusion_gpu.h",
        "utils.h",
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
        ":fp8_gelu_op",
        ":fp8_layernorm_op",
        ":fp8_quantize_fusion_op",
        ":fp8_quantize_op",
    ],
)
