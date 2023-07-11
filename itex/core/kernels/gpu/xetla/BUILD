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
