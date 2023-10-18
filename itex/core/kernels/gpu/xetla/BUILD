load("//itex:itex.bzl", "itex_xetla_library", "tf_copts")

itex_xetla_library(
    name = "mlp_op",
    srcs = [
        "mlp_op.cc",
        "mlp_op_gpu.cc",
    ],
    hdrs = [
        "mlp_op.h",
        "mlp_op_gpu.h",
    ],
    copts = tf_copts() + [],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "@xetla//:xetla_header",
    ],
    alwayslink = True,
)

itex_xetla_library(
    name = "mha_op",
    srcs = ["mha_op.cc"],
    hdrs = [
        "fmha_forward.h",
        "fmha_policy.h",
        "fmha_utils.h",
        "mha_op.h",
        "non_flash_sdp/mha_backward.h",
        "non_flash_sdp/mha_forward.h",
        "non_flash_sdp/mha_policy.h",
        "non_flash_sdp/mha_util.h",
    ],
    copts = tf_copts() + [],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
        "@xetla//:xetla_header",
    ],
    alwayslink = True,
)
