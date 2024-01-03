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
        "@itex_xetla//:xetla_header",
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
        "@itex_xetla//:xetla_header",
    ],
    alwayslink = True,
)

itex_xetla_library(
    name = "fmha_op",
    srcs = ["fmha_op.cc"],
    hdrs = [
        "fmha_backward.h",
        "fmha_forward.h",
        "fmha_op.h",
        "fmha_policy.h",
        "fmha_utils.h",
    ],
    copts = tf_copts() + [],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/common:fill_functor",
        "@itex_xetla//:xetla_header",
    ],
    alwayslink = True,
)

itex_xetla_library(
    name = "fused_einsum_impl",
    srcs = ["fused_einsum_impl.cc"],
    hdrs = [
        "fused_einsum_impl.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
        "//itex/core/kernels/gpu/linalg:fused_einsum_hdrs",
        "@itex_xetla//:xetla_header",
    ],
    alwayslink = True,
)
