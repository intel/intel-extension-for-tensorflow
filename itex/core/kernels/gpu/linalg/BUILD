load("//itex:itex.bzl", "itex_xpu_library", "tf_copts")

itex_xpu_library(
    name = "linalg",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":cholesky_op",
        ":determinant_op",
        ":einsum_op_impl",
        ":eye_functor",
        ":lu_op",
        ":matrix_diag_op",
        ":matrix_inverse_op",
        ":matrix_set_diag_op",
        ":matrix_solve_op",
        ":matrix_triangular_solve_op",
        # ":qr_op",
        ":self_adjoint_eig_v2_op",
        ":svd_op",
        ":tridiagonal_solve_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "cholesky_op",
    srcs = ["cholesky_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/gpu:matrix_band_part_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "fused_einsum_hdrs",
    hdrs = ["fused_einsum_helper.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "einsum_op_impl",
    srcs = ["einsum_op_impl.cc"],
    hdrs = [
        "//itex/core/kernels/common:einsum_hdrs",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/kernels/common:fill_functor",
        "//itex/core/kernels/gpu:libitex_gpu_xetla",
        "//itex/core/kernels/gpu:matmul_op",
        "//itex/core/kernels/gpu:reduction_ops",
        "//itex/core/kernels/gpu/linalg:fused_einsum_hdrs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "determinant_op",
    srcs = ["determinant_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/common:fill_functor",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "eye_functor",
    srcs = ["eye_functor.cc"],
    hdrs = ["eye_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "matrix_diag_op",
    srcs = [
        "matrix_diag_op.cc",
    ],
    hdrs = [
        "matrix_diag_op.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//itex:core",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "matrix_inverse_op",
    srcs = ["matrix_inverse_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/gpu:transpose_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "matrix_set_diag_op",
    srcs = [
        "matrix_set_diag_op.cc",
    ],
    hdrs = [
        "matrix_set_diag_op.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":matrix_diag_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "matrix_solve_op",
    srcs = ["matrix_solve_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/gpu:transpose_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "matrix_triangular_solve_op",
    srcs = ["matrix_triangular_solve_op.cc"],
    hdrs = ["matrix_triangular_solve_op_impl.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/gpu:matmul_op",
        "//itex/core/kernels/gpu:transpose_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "lu_op",
    srcs = ["lu_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/gpu:transpose_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "qr_op",
    srcs = ["qr_op.cc"],
    hdrs = ["qr_op_impl.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/kernels/gpu:matrix_band_part_op",
        "//itex/core/kernels/gpu:transpose_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "tridiagonal_solve_op",
    srcs = ["tridiagonal_solve_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/gpu:transpose_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "self_adjoint_eig_v2_op",
    srcs = ["self_adjoint_eig_v2_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/gpu:transpose_op",
    ],
    alwayslink = True,
)

itex_xpu_library(
    name = "svd_op",
    srcs = ["svd_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":eye_functor",
        "//itex/core/devices:xpu_device_util",
        "//itex/core/kernels/gpu:transpose_op",
    ],
    alwayslink = True,
)
